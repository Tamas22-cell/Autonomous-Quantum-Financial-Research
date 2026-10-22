from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    ConfidenceInterval,
    SignificanceLevel,
    StatisticalTestResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.significance_engine import (
    VERSION,
    SignificanceDiagnostics,
    SignificanceEngine,
    SignificanceReport,
    SignificanceValidationError,
    _normal_cdf,
    _normal_critical_value,
    _normal_two_sided_p_value,
    _sample_standard_deviation,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    minimum_observations: int = 30,
    confidence_level: float = 0.95,
    significance_threshold: float = 0.05,
) -> ValidationConfig:
    return ValidationConfig(
        minimum_observations=minimum_observations,
        confidence_level=confidence_level,
        significance_threshold=significance_threshold,
    )


def positive_values(
    count: int = 100,
) -> list[float]:
    return [
        0.015 + (index % 10) * 0.001
        for index in range(count)
    ]


def zero_mean_values(
    count: int = 100,
) -> list[float]:
    return [
        0.02 if index % 2 == 0 else -0.02
        for index in range(count)
    ]


def make_interval(
    *,
    lower: float = 0.01,
    upper: float = 0.03,
    estimate: float = 0.02,
) -> ConfidenceInterval:
    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        confidence_level=0.95,
        estimate=estimate,
        method="normal_mean_interval",
    )


def make_diagnostics(
    *,
    sample_size: int = 100,
    sample_mean: float = 0.02,
    null_mean: float = 0.0,
    standard_deviation: float = 0.01,
    standard_error: float = 0.001,
    test_statistic: float = 20.0,
    p_value: float = 0.001,
    effect_size: float = 2.0,
    lower: float = 0.01,
    upper: float = 0.03,
    significance_level: SignificanceLevel = (
        SignificanceLevel.STRONG
    ),
    statistically_significant: bool = True,
    direction_positive: bool = True,
) -> SignificanceDiagnostics:
    return SignificanceDiagnostics(
        sample_size=sample_size,
        sample_mean=sample_mean,
        null_mean=null_mean,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        test_statistic=test_statistic,
        p_value=p_value,
        effect_size=effect_size,
        confidence_interval=make_interval(
            lower=lower,
            upper=upper,
            estimate=sample_mean,
        ),
        significance_level=significance_level,
        statistically_significant=statistically_significant,
        direction_positive=direction_positive,
    )


def make_test_result(
    *,
    p_value: float = 0.001,
    effect_size: float = 2.0,
) -> StatisticalTestResult:
    return StatisticalTestResult(
        test_name="Mean vs Null Significance Test",
        statistic=20.0,
        p_value=p_value,
        significance_threshold=0.05,
        effect_size=effect_size,
        null_hypothesis="population mean = 0.0",
        alternative_hypothesis="population mean != 0.0",
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


# ============================================================
# Normal distribution helpers
# ============================================================


def test_normal_cdf_zero():
    assert _normal_cdf(0.0) == pytest.approx(0.5)


def test_normal_cdf_positive():
    assert _normal_cdf(1.96) == pytest.approx(
        0.975,
        abs=0.001,
    )


def test_normal_cdf_negative():
    assert _normal_cdf(-1.96) == pytest.approx(
        0.025,
        abs=0.001,
    )


def test_two_sided_p_zero():
    assert _normal_two_sided_p_value(
        0.0
    ) == pytest.approx(1.0)


def test_two_sided_p_196():
    assert _normal_two_sided_p_value(
        1.96
    ) == pytest.approx(
        0.05,
        abs=0.001,
    )


def test_two_sided_p_symmetric():
    assert _normal_two_sided_p_value(
        2.5
    ) == pytest.approx(
        _normal_two_sided_p_value(-2.5)
    )


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.80, 1.2815515655446004),
        (0.90, 1.6448536269514722),
        (0.95, 1.959963984540054),
        (0.98, 2.3263478740408408),
        (0.99, 2.5758293035489004),
    ],
)
def test_common_critical_values(
    confidence,
    expected,
):
    assert _normal_critical_value(
        confidence
    ) == pytest.approx(expected)


def test_custom_critical_value():
    value = _normal_critical_value(0.92)

    assert value > 1.64
    assert value < 1.96


@pytest.mark.parametrize(
    "confidence",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_invalid_confidence_level(
    confidence,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _normal_critical_value(
            confidence
        )


# ============================================================
# Standard deviation
# ============================================================


def test_sample_standard_deviation():
    result = _sample_standard_deviation(
        [1.0, 2.0, 3.0]
    )

    assert result == pytest.approx(1.0)


def test_sample_standard_deviation_constant():
    assert _sample_standard_deviation(
        [5.0, 5.0, 5.0]
    ) == 0.0


def test_sample_standard_deviation_single():
    assert _sample_standard_deviation(
        [5.0]
    ) == 0.0


def test_sample_standard_deviation_empty():
    with pytest.raises(
        SignificanceValidationError
    ):
        _sample_standard_deviation([])


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_standard_deviation_input(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _sample_standard_deviation(
            [1.0, value]
        )


# ============================================================
# Significance classification
# ============================================================


@pytest.mark.parametrize(
    "p_value,expected",
    [
        (0.0, SignificanceLevel.STRONG),
        (0.005, SignificanceLevel.STRONG),
        (0.01, SignificanceLevel.STRONG),
        (0.02, SignificanceLevel.MODERATE),
        (0.05, SignificanceLevel.MODERATE),
        (0.075, SignificanceLevel.WEAK),
        (0.10, SignificanceLevel.WEAK),
        (
            0.11,
            SignificanceLevel.NOT_SIGNIFICANT,
        ),
        (
            1.0,
            SignificanceLevel.NOT_SIGNIFICANT,
        ),
    ],
)
def test_classify_significance(
    p_value,
    expected,
):
    assert (
        SignificanceEngine.classify_significance(
            p_value
        )
        == expected
    )


@pytest.mark.parametrize(
    "p_value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
    ],
)
def test_invalid_significance_p_value(
    p_value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SignificanceEngine.classify_significance(
            p_value
        )


# ============================================================
# Diagnostics model
# ============================================================


def test_diagnostics_creation():
    diagnostics = make_diagnostics()

    assert diagnostics.sample_size == 100
    assert diagnostics.sample_mean == 0.02
    assert diagnostics.null_mean == 0.0
    assert diagnostics.mean_difference == 0.02
    assert diagnostics.statistically_significant is True
    assert diagnostics.direction_positive is True


def test_diagnostics_is_frozen():
    diagnostics = make_diagnostics()

    with pytest.raises(FrozenInstanceError):
        diagnostics.p_value = 0.5


def test_diagnostics_fingerprint_stable():
    first = make_diagnostics()
    second = make_diagnostics()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "sample_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_diagnostics_sample_size(
    sample_size,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_diagnostics(
            sample_size=sample_size
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_mean", True),
        ("sample_mean", float("nan")),
        ("null_mean", float("inf")),
        (
            "standard_deviation",
            float("nan"),
        ),
        ("standard_error", float("inf")),
        ("test_statistic", float("nan")),
        ("p_value", float("nan")),
        ("effect_size", float("inf")),
    ],
)
def test_invalid_diagnostics_numeric_fields(
    field,
    value,
):
    kwargs = {
        "sample_size": 100,
        "sample_mean": 0.02,
        "null_mean": 0.0,
        "standard_deviation": 0.01,
        "standard_error": 0.001,
        "test_statistic": 20.0,
        "p_value": 0.001,
        "effect_size": 2.0,
        "lower": 0.01,
        "upper": 0.03,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        make_diagnostics(**kwargs)


def test_negative_standard_deviation_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_diagnostics(
            standard_deviation=-0.01
        )


def test_negative_standard_error_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_diagnostics(
            standard_error=-0.01
        )


@pytest.mark.parametrize(
    "p_value",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_diagnostics_probability(
    p_value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_diagnostics(
            p_value=p_value
        )


def test_invalid_significance_level_type():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SignificanceDiagnostics(
            sample_size=100,
            sample_mean=0.02,
            null_mean=0.0,
            standard_deviation=0.01,
            standard_error=0.001,
            test_statistic=20.0,
            p_value=0.001,
            effect_size=2.0,
            confidence_interval=make_interval(),
            significance_level="STRONG",
            statistically_significant=True,
            direction_positive=True,
        )


# ============================================================
# Significance report
# ============================================================


def test_report_creation():
    report = SignificanceReport(
        diagnostics=make_diagnostics(),
        statistical_test=make_test_result(),
        minimum_effect_size=0.20,
        effect_size_sufficient=True,
        passed=True,
    )

    assert report.passed is True
    assert report.effect_size_sufficient is True


def test_report_fingerprint_stable():
    first = SignificanceReport(
        diagnostics=make_diagnostics(),
        statistical_test=make_test_result(),
        minimum_effect_size=0.20,
        effect_size_sufficient=True,
        passed=True,
    )

    second = SignificanceReport(
        diagnostics=make_diagnostics(),
        statistical_test=make_test_result(),
        minimum_effect_size=0.20,
        effect_size_sufficient=True,
        passed=True,
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_report_effect_size(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SignificanceReport(
            diagnostics=make_diagnostics(),
            statistical_test=make_test_result(),
            minimum_effect_size=value,
            effect_size_sufficient=False,
            passed=False,
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = SignificanceEngine()

    assert engine.null_mean == 0.0
    assert engine.minimum_effect_size == 0.20


def test_custom_engine():
    engine = SignificanceEngine(
        make_config(
            significance_threshold=0.01
        ),
        null_mean=0.005,
        minimum_effect_size=0.50,
    )

    assert engine.null_mean == 0.005
    assert engine.minimum_effect_size == 0.50

    assert (
        engine.config.significance_threshold
        == 0.01
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        float("nan"),
        float("inf"),
        "bad",
    ],
)
def test_invalid_null_mean(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SignificanceEngine(
            null_mean=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_minimum_effect_size(
    value,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SignificanceEngine(
            minimum_effect_size=value
        )


# ============================================================
# Analyze
# ============================================================


def test_analyze_positive_values():
    engine = SignificanceEngine(
        make_config(
            minimum_observations=30
        )
    )

    diagnostics = engine.analyze(
        positive_values()
    )

    assert diagnostics.sample_size == 100
    assert diagnostics.sample_mean > 0.0
    assert diagnostics.standard_deviation > 0.0
    assert diagnostics.standard_error > 0.0
    assert diagnostics.test_statistic > 0.0
    assert diagnostics.p_value <= 0.05
    assert diagnostics.effect_size > 0.0
    assert diagnostics.direction_positive is True
    assert diagnostics.statistically_significant is True


def test_analyze_zero_mean_values():
    engine = SignificanceEngine()

    diagnostics = engine.analyze(
        zero_mean_values()
    )

    assert diagnostics.sample_mean == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert diagnostics.test_statistic == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert diagnostics.p_value == pytest.approx(
        1.0,
        abs=1e-12,
    )

    assert diagnostics.statistically_significant is False


def test_analyze_constant_positive_values():
    engine = SignificanceEngine()

    diagnostics = engine.analyze(
        [0.01] * 100
    )

    assert diagnostics.standard_deviation == 0.0
    assert diagnostics.standard_error == 0.0
    assert diagnostics.test_statistic == 1e12
    assert diagnostics.p_value == 0.0
    assert diagnostics.effect_size == 1e12
    assert diagnostics.direction_positive is True


def test_analyze_constant_negative_values():
    engine = SignificanceEngine()

    diagnostics = engine.analyze(
        [-0.01] * 100
    )

    assert diagnostics.standard_deviation == 0.0
    assert diagnostics.test_statistic == -1e12
    assert diagnostics.p_value == 0.0
    assert diagnostics.effect_size == -1e12
    assert diagnostics.direction_positive is False


def test_analyze_constant_null_values():
    engine = SignificanceEngine()

    diagnostics = engine.analyze(
        [0.0] * 100
    )

    assert diagnostics.standard_deviation == 0.0
    assert diagnostics.standard_error == 0.0
    assert diagnostics.test_statistic == 0.0
    assert diagnostics.p_value == 1.0
    assert diagnostics.effect_size == 0.0
    assert diagnostics.direction_positive is False


def test_analyze_custom_null_mean():
    engine = SignificanceEngine(
        null_mean=0.01
    )

    diagnostics = engine.analyze(
        [0.02] * 100
    )

    assert diagnostics.null_mean == 0.01
    assert diagnostics.mean_difference == pytest.approx(
        0.01
    )

    assert diagnostics.direction_positive is True


def test_analyze_confidence_interval():
    engine = SignificanceEngine(
        make_config(
            confidence_level=0.90
        )
    )

    diagnostics = engine.analyze(
        positive_values()
    )

    interval = diagnostics.confidence_interval

    assert interval.confidence_level == 0.90
    assert interval.lower < diagnostics.sample_mean
    assert interval.upper > diagnostics.sample_mean

    assert (
        interval.method
        == "normal_mean_interval"
    )


def test_analyze_insufficient_observations():
    engine = SignificanceEngine(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        SignificanceValidationError
    ):
        engine.analyze(
            positive_values(99)
        )


def test_analyze_empty_values():
    engine = SignificanceEngine()

    with pytest.raises(
        SignificanceValidationError
    ):
        engine.analyze([])


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        float("nan"),
        float("inf"),
    ],
)
def test_analyze_invalid_observation(
    value,
):
    engine = SignificanceEngine()

    values = positive_values(99)
    values.append(value)

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.analyze(values)


# ============================================================
# StatisticalTestResult adapter
# ============================================================


def test_to_statistical_test():
    engine = SignificanceEngine()

    diagnostics = make_diagnostics()

    result = engine.to_statistical_test(
        diagnostics
    )

    assert isinstance(
        result,
        StatisticalTestResult,
    )

    assert (
        result.test_name
        == "Mean vs Null Significance Test"
    )

    assert (
        result.statistic
        == diagnostics.test_statistic
    )

    assert (
        result.p_value
        == diagnostics.p_value
    )

    assert (
        result.effect_size
        == diagnostics.effect_size
    )

    assert (
        result.significance_threshold
        == engine.config.significance_threshold
    )


def test_statistical_test_hypotheses():
    engine = SignificanceEngine(
        null_mean=0.005
    )

    result = engine.to_statistical_test(
        make_diagnostics(
            null_mean=0.005
        )
    )

    assert "0.005" in result.null_hypothesis
    assert "0.005" in result.alternative_hypothesis


# ============================================================
# Evaluation
# ============================================================


def test_evaluate_pass():
    engine = SignificanceEngine(
        minimum_effect_size=0.20
    )

    report = engine.evaluate(
        make_diagnostics(
            effect_size=1.0,
            p_value=0.001,
            statistically_significant=True,
            direction_positive=True,
        )
    )

    assert report.effect_size_sufficient is True
    assert report.passed is True


def test_evaluate_not_significant():
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics(
            p_value=0.50,
            significance_level=(
                SignificanceLevel.NOT_SIGNIFICANT
            ),
            statistically_significant=False,
        )
    )

    assert report.passed is False


def test_evaluate_negative_direction():
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics(
            sample_mean=-0.02,
            test_statistic=-20.0,
            effect_size=-2.0,
            direction_positive=False,
        )
    )

    assert report.effect_size_sufficient is True
    assert report.passed is False


def test_evaluate_small_effect():
    engine = SignificanceEngine(
        minimum_effect_size=0.50
    )

    report = engine.evaluate(
        make_diagnostics(
            effect_size=0.10
        )
    )

    assert report.effect_size_sufficient is False
    assert report.passed is False


# ============================================================
# Score
# ============================================================


def test_score_bounded():
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics()
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_strong_result_high_score():
    engine = SignificanceEngine(
        minimum_effect_size=0.20
    )

    report = engine.evaluate(
        make_diagnostics(
            p_value=0.0001,
            effect_size=2.0,
            lower=0.01,
            upper=0.03,
        )
    )

    assert (
        engine._calculate_score(report)
        > 0.90
    )


def test_negative_result_low_score():
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics(
            sample_mean=-0.02,
            p_value=0.50,
            effect_size=-1.0,
            lower=-0.03,
            upper=-0.01,
            significance_level=(
                SignificanceLevel.NOT_SIGNIFICANT
            ),
            statistically_significant=False,
            direction_positive=False,
        )
    )

    assert (
        engine._calculate_score(report)
        < 0.20
    )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics()
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.STATISTICAL_SIGNIFICANCE
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    assert (
        result.confidence_interval
        is not None
    )

    assert len(
        result.statistical_tests
    ) == 1

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
    engine = SignificanceEngine()

    report = engine.evaluate(
        make_diagnostics(
            p_value=0.50,
            effect_size=0.01,
            significance_level=(
                SignificanceLevel.NOT_SIGNIFICANT
            ),
            statistically_significant=False,
        )
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


# ============================================================
# Convenience validation
# ============================================================


def test_validate_positive_values():
    engine = SignificanceEngine(
        make_config(
            minimum_observations=50
        ),
        minimum_effect_size=0.20,
    )

    report = engine.validate(
        positive_values(200)
    )

    assert report.passed is True
    assert report.diagnostics.p_value <= 0.05
    assert report.diagnostics.effect_size > 0.20


def test_validate_zero_mean_values():
    engine = SignificanceEngine()

    report = engine.validate(
        zero_mean_values(200)
    )

    assert report.passed is False

    assert (
        report.diagnostics
        .statistically_significant
        is False
    )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = SignificanceEngine(
        make_config(
            minimum_observations=50,
            confidence_level=0.90,
            significance_threshold=0.01,
        ),
        null_mean=0.005,
        minimum_effect_size=0.50,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"
    assert snapshot["null_mean"] == 0.005

    assert (
        snapshot["minimum_effect_size"]
        == 0.50
    )

    assert (
        snapshot["confidence_level"]
        == 0.90
    )

    assert (
        snapshot["significance_threshold"]
        == 0.01
    )

    assert (
        snapshot["minimum_observations"]
        == 50
    )

    assert (
        snapshot["test_method"]
        == "normal_approximation_mean_test"
    )

    assert snapshot["research_only"] is True

    assert (
        snapshot["real_money_execution"]
        is False
    )

    assert len(
        snapshot["config_fingerprint"]
    ) == 64


# ============================================================
# Complete significance flow
# ============================================================


def test_complete_significance_flow():
    config = ValidationConfig(
        confidence_level=0.95,
        significance_threshold=0.05,
        minimum_observations=50,
        random_seed=42,
    )

    engine = SignificanceEngine(
        config,
        null_mean=0.0,
        minimum_effect_size=0.20,
    )

    values = [
        0.012
        + (index % 10) * 0.0005
        for index in range(250)
    ]

    diagnostics = engine.analyze(
        values
    )

    assert diagnostics.sample_size == 250
    assert diagnostics.sample_mean > 0.0
    assert diagnostics.standard_deviation > 0.0
    assert diagnostics.standard_error > 0.0
    assert diagnostics.test_statistic > 0.0
    assert diagnostics.p_value <= 0.05
    assert diagnostics.effect_size > 0.20

    assert (
        diagnostics.significance_level
        == SignificanceLevel.STRONG
    )

    assert (
        diagnostics.statistically_significant
        is True
    )

    assert diagnostics.direction_positive is True

    assert (
        diagnostics.confidence_interval.lower
        > 0.0
    )

    report = engine.evaluate(
        diagnostics
    )

    assert report.effect_size_sufficient is True
    assert report.passed is True

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.STATISTICAL_SIGNIFICANCE
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert result.score > 0.90

    assert len(
        result.statistical_tests
    ) == 1

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
        diagnostics.fingerprint
    ) == 64

    assert len(report.fingerprint) == 64