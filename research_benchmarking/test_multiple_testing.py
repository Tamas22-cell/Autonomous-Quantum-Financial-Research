from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    MultipleTestingMethod,
    StatisticalTestResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.multiple_testing import (
    VERSION,
    CorrectedHypothesis,
    MultipleTestingEngine,
    MultipleTestingError,
    MultipleTestingReport,
    _validate_p_values,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    significance_threshold: float = 0.05,
    method: MultipleTestingMethod = (
        MultipleTestingMethod.BENJAMINI_HOCHBERG
    ),
) -> ValidationConfig:
    return ValidationConfig(
        significance_threshold=significance_threshold,
        multiple_testing_method=method,
    )


def make_hypothesis(
    *,
    index: int = 0,
    original_p_value: float = 0.01,
    adjusted_p_value: float = 0.02,
    rejected: bool = True,
    method: MultipleTestingMethod = (
        MultipleTestingMethod.BENJAMINI_HOCHBERG
    ),
    significance_threshold: float = 0.05,
) -> CorrectedHypothesis:
    return CorrectedHypothesis(
        index=index,
        original_p_value=original_p_value,
        adjusted_p_value=adjusted_p_value,
        rejected=rejected,
        method=method,
        significance_threshold=significance_threshold,
    )


def make_report(
    *,
    method: MultipleTestingMethod = (
        MultipleTestingMethod.BENJAMINI_HOCHBERG
    ),
    rejected_count: int = 2,
    retained_count: int = 2,
    passed: bool = True,
) -> MultipleTestingReport:
    hypotheses = [
        make_hypothesis(
            index=0,
            original_p_value=0.001,
            adjusted_p_value=0.004,
            rejected=True,
            method=method,
        ),
        make_hypothesis(
            index=1,
            original_p_value=0.01,
            adjusted_p_value=0.02,
            rejected=True,
            method=method,
        ),
        make_hypothesis(
            index=2,
            original_p_value=0.20,
            adjusted_p_value=0.2666666667,
            rejected=False,
            method=method,
        ),
        make_hypothesis(
            index=3,
            original_p_value=0.50,
            adjusted_p_value=0.50,
            rejected=False,
            method=method,
        ),
    ]

    return MultipleTestingReport(
        method=method,
        significance_threshold=0.05,
        hypotheses=hypotheses,
        rejected_count=rejected_count,
        retained_count=retained_count,
        discovery_rate=rejected_count / 4,
        passed=passed,
    )


def make_statistical_test(
    *,
    name: str = "alpha_test",
    p_value: float = 0.01,
    statistic: float = 2.5,
    effect_size: float = 0.50,
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_name=name,
        statistic=statistic,
        p_value=p_value,
        significance_threshold=0.05,
        effect_size=effect_size,
        null_hypothesis="no alpha",
        alternative_hypothesis="alpha exists",
        metadata={
            "source": "unit_test",
        },
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert MultipleTestingEngine.VERSION == "v6.0.2"


# ============================================================
# P-value validation
# ============================================================


def test_validate_p_values():
    result = _validate_p_values(
        [0.01, 0.05, 0.50, 1.0]
    )

    assert result == [
        0.01,
        0.05,
        0.50,
        1.0,
    ]


def test_validate_p_values_accepts_zero():
    assert _validate_p_values(
        [0.0]
    ) == [0.0]


def test_validate_p_values_accepts_one():
    assert _validate_p_values(
        [1.0]
    ) == [1.0]


def test_validate_p_values_converts_int():
    assert _validate_p_values(
        [0, 1]
    ) == [0.0, 1.0]


def test_validate_p_values_empty():
    with pytest.raises(
        MultipleTestingError
    ):
        _validate_p_values([])


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        -0.01,
        1.01,
    ],
)
def test_validate_p_values_invalid(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_p_values(
            [0.01, value]
        )


# ============================================================
# CorrectedHypothesis model
# ============================================================


def test_corrected_hypothesis_creation():
    hypothesis = make_hypothesis()

    assert hypothesis.index == 0
    assert hypothesis.original_p_value == 0.01
    assert hypothesis.adjusted_p_value == 0.02
    assert hypothesis.rejected is True

    assert (
        hypothesis.method
        == MultipleTestingMethod.BENJAMINI_HOCHBERG
    )


def test_corrected_hypothesis_frozen():
    hypothesis = make_hypothesis()

    with pytest.raises(FrozenInstanceError):
        hypothesis.rejected = False


def test_corrected_hypothesis_fingerprint():
    first = make_hypothesis()
    second = make_hypothesis()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_corrected_hypothesis_fingerprint_changes():
    first = make_hypothesis(
        adjusted_p_value=0.02
    )

    second = make_hypothesis(
        adjusted_p_value=0.03
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "index",
    [
        -1,
        True,
        1.5,
    ],
)
def test_invalid_hypothesis_index(
    index,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_hypothesis(
            index=index
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_original_p_value(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_hypothesis(
            original_p_value=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_adjusted_p_value(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_hypothesis(
            adjusted_p_value=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        -0.01,
        1.01,
        True,
        float("nan"),
    ],
)
def test_invalid_hypothesis_threshold(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_hypothesis(
            significance_threshold=value
        )


def test_invalid_hypothesis_method():
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_hypothesis(
            method="holm"
        )


# ============================================================
# MultipleTestingReport model
# ============================================================


def test_report_creation():
    report = make_report()

    assert report.hypothesis_count == 4
    assert report.rejected_count == 2
    assert report.retained_count == 2
    assert report.discovery_rate == 0.5
    assert report.passed is True


def test_report_rejected_indices():
    report = make_report()

    assert report.rejected_indices == [
        0,
        1,
    ]


def test_report_fingerprint():
    first = make_report()
    second = make_report()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_frozen():
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.passed = False


def test_report_invalid_method():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method="bad",
            significance_threshold=0.05,
            hypotheses=[
                make_hypothesis()
            ],
            rejected_count=1,
            retained_count=0,
            discovery_rate=1.0,
            passed=True,
        )


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_report_invalid_threshold(
    threshold,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method=(
                MultipleTestingMethod.BONFERRONI
            ),
            significance_threshold=threshold,
            hypotheses=[
                make_hypothesis(
                    method=(
                        MultipleTestingMethod.BONFERRONI
                    )
                )
            ],
            rejected_count=1,
            retained_count=0,
            discovery_rate=1.0,
            passed=True,
        )


def test_report_empty_hypotheses():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method=MultipleTestingMethod.HOLM,
            significance_threshold=0.05,
            hypotheses=[],
            rejected_count=0,
            retained_count=0,
            discovery_rate=0.0,
            passed=False,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        1.5,
    ],
)
def test_report_invalid_rejected_count(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method=MultipleTestingMethod.HOLM,
            significance_threshold=0.05,
            hypotheses=[
                make_hypothesis(
                    method=MultipleTestingMethod.HOLM
                )
            ],
            rejected_count=value,
            retained_count=0,
            discovery_rate=0.0,
            passed=False,
        )


def test_report_inconsistent_counts():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method=MultipleTestingMethod.HOLM,
            significance_threshold=0.05,
            hypotheses=[
                make_hypothesis(
                    method=MultipleTestingMethod.HOLM
                )
            ],
            rejected_count=1,
            retained_count=1,
            discovery_rate=1.0,
            passed=True,
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_report_invalid_discovery_rate(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingReport(
            method=MultipleTestingMethod.HOLM,
            significance_threshold=0.05,
            hypotheses=[
                make_hypothesis(
                    method=MultipleTestingMethod.HOLM
                )
            ],
            rejected_count=1,
            retained_count=0,
            discovery_rate=value,
            passed=True,
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = MultipleTestingEngine()

    assert (
        engine.method
        == MultipleTestingMethod.BENJAMINI_HOCHBERG
    )

    assert engine.minimum_discoveries == 1


@pytest.mark.parametrize(
    "method",
    list(MultipleTestingMethod),
)
def test_engine_all_methods(method):
    engine = MultipleTestingEngine(
        method=method
    )

    assert engine.method == method


def test_engine_uses_config_method():
    config = make_config(
        method=MultipleTestingMethod.HOLM
    )

    engine = MultipleTestingEngine(
        config
    )

    assert (
        engine.method
        == MultipleTestingMethod.HOLM
    )


def test_engine_explicit_method_overrides_config():
    config = make_config(
        method=MultipleTestingMethod.HOLM
    )

    engine = MultipleTestingEngine(
        config,
        method=(
            MultipleTestingMethod.BONFERRONI
        ),
    )

    assert (
        engine.method
        == MultipleTestingMethod.BONFERRONI
    )


def test_engine_invalid_method():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingEngine(
            method="bad"
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        1.5,
    ],
)
def test_invalid_minimum_discoveries(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MultipleTestingEngine(
            minimum_discoveries=value
        )


def test_zero_minimum_discoveries_allowed():
    engine = MultipleTestingEngine(
        minimum_discoveries=0
    )

    assert engine.minimum_discoveries == 0


# ============================================================
# Bonferroni
# ============================================================


def test_bonferroni():
    result = MultipleTestingEngine.bonferroni(
        [
            0.001,
            0.01,
            0.03,
            0.20,
        ]
    )

    assert result == pytest.approx(
        [
            0.004,
            0.04,
            0.12,
            0.80,
        ]
    )


def test_bonferroni_caps_at_one():
    result = MultipleTestingEngine.bonferroni(
        [
            0.50,
            0.90,
        ]
    )

    assert result == [
        1.0,
        1.0,
    ]


def test_bonferroni_single():
    assert MultipleTestingEngine.bonferroni(
        [0.03]
    ) == [0.03]


# ============================================================
# Holm
# ============================================================


def test_holm():
    result = MultipleTestingEngine.holm(
        [
            0.01,
            0.04,
            0.03,
            0.002,
        ]
    )

    assert result == pytest.approx(
        [
            0.03,
            0.06,
            0.06,
            0.008,
        ]
    )


def test_holm_preserves_original_order():
    values = [
        0.50,
        0.001,
        0.20,
    ]

    result = MultipleTestingEngine.holm(
        values
    )

    assert result[1] <= result[2]
    assert result[2] <= result[0]


def test_holm_single():
    assert MultipleTestingEngine.holm(
        [0.03]
    ) == [0.03]


def test_holm_caps_at_one():
    result = MultipleTestingEngine.holm(
        [
            0.80,
            0.90,
        ]
    )

    assert all(
        value <= 1.0
        for value in result
    )


# ============================================================
# Benjamini-Hochberg
# ============================================================


def test_benjamini_hochberg():
    result = (
        MultipleTestingEngine
        .benjamini_hochberg(
            [
                0.001,
                0.01,
                0.03,
                0.20,
            ]
        )
    )

    assert result == pytest.approx(
        [
            0.004,
            0.02,
            0.04,
            0.20,
        ]
    )


def test_bh_preserves_original_order():
    result = (
        MultipleTestingEngine
        .benjamini_hochberg(
            [
                0.50,
                0.001,
                0.10,
            ]
        )
    )

    assert result[1] < result[2]
    assert result[2] < result[0]


def test_bh_single():
    result = (
        MultipleTestingEngine
        .benjamini_hochberg(
            [0.03]
        )
    )

    assert result == [0.03]


def test_bh_monotonic_sorted_values():
    original = [
        0.001,
        0.01,
        0.02,
        0.50,
    ]

    adjusted = (
        MultipleTestingEngine
        .benjamini_hochberg(
            original
        )
    )

    assert adjusted == sorted(
        adjusted
    )


def test_bh_caps_at_one():
    result = (
        MultipleTestingEngine
        .benjamini_hochberg(
            [
                0.90,
                1.0,
            ]
        )
    )

    assert all(
        0.0 <= value <= 1.0
        for value in result
    )


# ============================================================
# Adjust dispatch
# ============================================================


def test_adjust_none():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.NONE
    )

    values = [
        0.01,
        0.20,
    ]

    assert engine.adjust(values) == values


def test_adjust_bonferroni():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    assert engine.adjust(
        [0.01, 0.20]
    ) == pytest.approx(
        [0.02, 0.40]
    )


def test_adjust_holm():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.HOLM
    )

    result = engine.adjust(
        [0.01, 0.20]
    )

    assert result == pytest.approx(
        [0.02, 0.20]
    )


def test_adjust_bh():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BENJAMINI_HOCHBERG
        )
    )

    result = engine.adjust(
        [0.01, 0.20]
    )

    assert result == pytest.approx(
        [0.02, 0.20]
    )


def test_adjust_empty():
    engine = MultipleTestingEngine()

    with pytest.raises(
        MultipleTestingError
    ):
        engine.adjust([])


# ============================================================
# Evaluate
# ============================================================


@pytest.mark.parametrize(
    "method",
    list(MultipleTestingMethod),
)
def test_evaluate_all_methods(
    method,
):
    engine = MultipleTestingEngine(
        method=method,
        minimum_discoveries=1,
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.method == method
    assert report.hypothesis_count == 4

    assert (
        report.rejected_count
        + report.retained_count
        == 4
    )

    assert (
        0.0
        <= report.discovery_rate
        <= 1.0
    )


def test_evaluate_bonferroni():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.rejected_count == 2
    assert report.rejected_indices == [
        0,
        1,
    ]

    assert report.passed is True


def test_evaluate_holm():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.HOLM
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.rejected_count == 2
    assert report.passed is True


def test_evaluate_bh():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BENJAMINI_HOCHBERG
        )
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.rejected_count == 2
    assert report.passed is True


def test_evaluate_none():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.NONE
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.rejected_count == 2
    assert report.passed is True


def test_evaluate_no_discoveries():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    report = engine.evaluate(
        [
            0.20,
            0.30,
            0.40,
        ]
    )

    assert report.rejected_count == 0
    assert report.retained_count == 3
    assert report.discovery_rate == 0.0
    assert report.passed is False


def test_evaluate_minimum_discoveries():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BENJAMINI_HOCHBERG
        ),
        minimum_discoveries=3,
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    assert report.rejected_count == 2
    assert report.passed is False


def test_evaluate_zero_minimum():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        ),
        minimum_discoveries=0,
    )

    report = engine.evaluate(
        [
            0.90,
            0.95,
        ]
    )

    assert report.rejected_count == 0
    assert report.passed is True


# ============================================================
# StatisticalTestResult integration
# ============================================================


def test_correct_statistical_tests():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    tests = [
        make_statistical_test(
            name="test_1",
            p_value=0.01,
        ),
        make_statistical_test(
            name="test_2",
            p_value=0.20,
        ),
    ]

    corrected = (
        engine.correct_statistical_tests(
            tests
        )
    )

    assert len(corrected) == 2

    assert corrected[
        0
    ].adjusted_p_value == pytest.approx(
        0.02
    )

    assert corrected[
        1
    ].adjusted_p_value == pytest.approx(
        0.40
    )


def test_correct_statistical_tests_preserves_data():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.HOLM
    )

    original = make_statistical_test(
        name="important_test",
        p_value=0.01,
        statistic=3.0,
        effect_size=0.75,
    )

    corrected = (
        engine.correct_statistical_tests(
            [original]
        )[0]
    )

    assert (
        corrected.test_name
        == original.test_name
    )

    assert (
        corrected.statistic
        == original.statistic
    )

    assert (
        corrected.p_value
        == original.p_value
    )

    assert (
        corrected.effect_size
        == original.effect_size
    )

    assert (
        corrected.null_hypothesis
        == original.null_hypothesis
    )

    assert (
        corrected.alternative_hypothesis
        == original.alternative_hypothesis
    )

    assert (
        corrected.metadata["source"]
        == "unit_test"
    )

    assert (
        corrected.metadata[
            "multiple_testing_method"
        ]
        == MultipleTestingMethod.HOLM.value
    )


def test_correct_statistical_tests_empty():
    engine = MultipleTestingEngine()

    with pytest.raises(
        MultipleTestingError
    ):
        engine.correct_statistical_tests(
            []
        )


def test_correct_statistical_tests_invalid_type():
    engine = MultipleTestingEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.correct_statistical_tests(
            [
                make_statistical_test(),
                "bad",
            ]
        )


# ============================================================
# Score
# ============================================================


def test_score_bounded():
    engine = MultipleTestingEngine()

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_score_no_discoveries():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    report = engine.evaluate(
        [
            0.80,
            0.90,
        ]
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score < 0.50


def test_score_zero_minimum():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        ),
        minimum_discoveries=0,
    )

    report = engine.evaluate(
        [
            0.90,
            0.95,
        ]
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BENJAMINI_HOCHBERG
        )
    )

    report = engine.evaluate(
        [
            0.001,
            0.01,
            0.20,
            0.50,
        ]
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.MULTIPLE_TESTING
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    assert (
        result.diagnostics["rejected_count"]
        == 2
    )

    assert (
        result.diagnostics[
            "rejected_indices"
        ]
        == [0, 1]
    )

    assert (
        result.metadata["research_only"]
        is True
    )

    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_to_validation_result_fail():
    engine = MultipleTestingEngine(
        method=(
            MultipleTestingMethod.BONFERRONI
        )
    )

    report = engine.evaluate(
        [
            0.20,
            0.30,
            0.40,
        ]
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


# ============================================================
# Validate convenience API
# ============================================================


def test_validate():
    engine = MultipleTestingEngine(
        method=MultipleTestingMethod.HOLM
    )

    report = engine.validate(
        [
            0.001,
            0.01,
            0.20,
        ]
    )

    assert isinstance(
        report,
        MultipleTestingReport,
    )

    assert report.passed is True


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = MultipleTestingEngine(
        make_config(
            significance_threshold=0.01,
            method=MultipleTestingMethod.HOLM,
        ),
        minimum_discoveries=2,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert (
        snapshot["method"]
        == MultipleTestingMethod.HOLM.value
    )

    assert (
        snapshot["significance_threshold"]
        == 0.01
    )

    assert (
        snapshot["minimum_discoveries"]
        == 2
    )

    assert snapshot["research_only"] is True

    assert (
        snapshot["real_money_execution"]
        is False
    )

    assert len(
        snapshot["config_fingerprint"]
    ) == 64

    assert (
        MultipleTestingMethod.BONFERRONI.value
        in snapshot["supported_methods"]
    )

    assert (
        MultipleTestingMethod.HOLM.value
        in snapshot["supported_methods"]
    )

    assert (
        MultipleTestingMethod
        .BENJAMINI_HOCHBERG
        .value
        in snapshot["supported_methods"]
    )


# ============================================================
# Algorithm relationships
# ============================================================


def test_bonferroni_not_less_conservative_than_bh():
    values = [
        0.001,
        0.01,
        0.03,
        0.20,
        0.50,
    ]

    bonferroni = (
        MultipleTestingEngine.bonferroni(
            values
        )
    )

    bh = (
        MultipleTestingEngine
        .benjamini_hochberg(
            values
        )
    )

    for bonferroni_value, bh_value in zip(
        bonferroni,
        bh,
    ):
        assert (
            bonferroni_value
            >= bh_value
        )


def test_adjusted_p_values_never_negative():
    values = [
        0.0,
        0.001,
        0.50,
        1.0,
    ]

    for method in [
        MultipleTestingMethod.BONFERRONI,
        MultipleTestingMethod.HOLM,
        MultipleTestingMethod.BENJAMINI_HOCHBERG,
    ]:
        engine = MultipleTestingEngine(
            method=method
        )

        adjusted = engine.adjust(
            values
        )

        assert all(
            0.0 <= value <= 1.0
            for value in adjusted
        )


# ============================================================
# Determinism
# ============================================================


@pytest.mark.parametrize(
    "method",
    list(MultipleTestingMethod),
)
def test_adjustment_deterministic(
    method,
):
    engine = MultipleTestingEngine(
        method=method
    )

    values = [
        0.001,
        0.01,
        0.03,
        0.20,
        0.50,
    ]

    first = engine.adjust(values)
    second = engine.adjust(values)

    assert first == second


@pytest.mark.parametrize(
    "method",
    list(MultipleTestingMethod),
)
def test_report_deterministic(
    method,
):
    engine = MultipleTestingEngine(
        method=method
    )

    values = [
        0.001,
        0.01,
        0.03,
        0.20,
        0.50,
    ]

    first = engine.evaluate(values)
    second = engine.evaluate(values)

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# Complete multiple-testing flow
# ============================================================


def test_complete_multiple_testing_flow():
    config = ValidationConfig(
        significance_threshold=0.05,
        multiple_testing_method=(
            MultipleTestingMethod
            .BENJAMINI_HOCHBERG
        ),
        research_only=True,
        real_money_execution=False,
    )

    engine = MultipleTestingEngine(
        config,
        minimum_discoveries=2,
    )

    p_values = [
        0.001,
        0.004,
        0.015,
        0.20,
        0.50,
        0.80,
    ]

    adjusted = engine.adjust(
        p_values
    )

    assert len(adjusted) == len(
        p_values
    )

    assert all(
        0.0 <= value <= 1.0
        for value in adjusted
    )

    report = engine.evaluate(
        p_values
    )

    assert report.hypothesis_count == 6
    assert report.rejected_count >= 2
    assert report.passed is True

    assert (
        report.method
        == MultipleTestingMethod
        .BENJAMINI_HOCHBERG
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.MULTIPLE_TESTING
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    assert (
        result.metadata["research_only"]
        is True
    )

    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )

    assert len(
        report.fingerprint
    ) == 64