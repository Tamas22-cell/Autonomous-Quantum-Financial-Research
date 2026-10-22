from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    OverfittingRisk,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.overfitting_guard import (
    VERSION,
    GeneralizationGap,
    OverfittingGuard,
    OverfittingGuardError,
    OverfittingGuardReport,
    ParameterInstabilityDiagnostics,
    SelectionBiasDiagnostics,
    _validate_non_negative,
    _validate_numeric,
    _validate_positive_integer,
    _validate_ratio,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    anti_overfitting_enabled: bool = True,
) -> ValidationConfig:
    return ValidationConfig(
        minimum_observations=30,
        anti_overfitting_enabled=anti_overfitting_enabled,
        research_only=True,
        real_money_execution=False,
    )


def make_gap(
    *,
    acceptable: bool = True,
) -> GeneralizationGap:
    return GeneralizationGap(
        in_sample_score=1.0,
        out_of_sample_score=0.8,
        absolute_gap=0.2,
        degradation_ratio=0.2,
        retained_performance_ratio=0.8,
        positive_out_of_sample=True,
        acceptable=acceptable,
    )


def make_selection_bias(
    *,
    risk_score: float = 0.2,
) -> SelectionBiasDiagnostics:
    return SelectionBiasDiagnostics(
        tested_strategies=10,
        selected_rank=1,
        best_score=1.0,
        median_score=0.8,
        score_advantage=0.2,
        normalized_advantage=0.2,
        selection_pressure=0.3,
        risk_score=risk_score,
    )


def make_parameter_instability(
    *,
    instability_score: float = 0.1,
) -> ParameterInstabilityDiagnostics:
    return ParameterInstabilityDiagnostics(
        parameter_count=3,
        stable_parameter_count=3,
        stable_parameter_ratio=1.0,
        average_stability_score=0.9,
        minimum_stability_score=0.8,
        instability_score=instability_score,
    )


def make_report(
    *,
    overall_risk_score: float = 0.2,
    risk_level: OverfittingRisk = OverfittingRisk.LOW,
    passed: bool = True,
) -> OverfittingGuardReport:
    return OverfittingGuardReport(
        generalization_gap=make_gap(),
        selection_bias=make_selection_bias(),
        parameter_instability=make_parameter_instability(),
        multiple_testing_risk=0.1,
        regime_instability_risk=0.1,
        robustness_failure_risk=0.1,
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,
        passed=passed,
        reasons=[],
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert OverfittingGuard.VERSION == "v6.0.2"


# ============================================================
# Validation helpers
# ============================================================


@pytest.mark.parametrize(
    "value",
    [0, 1, -1, 0.5, -0.5],
)
def test_validate_numeric_valid(value):
    assert _validate_numeric(
        value,
        "value",
    ) == float(value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_numeric_invalid(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_numeric(
            value,
            "value",
        )


@pytest.mark.parametrize(
    "value",
    [0, 0.5, 1.0, 10],
)
def test_validate_non_negative_valid(value):
    assert _validate_non_negative(
        value,
        "value",
    ) == float(value)


@pytest.mark.parametrize(
    "value",
    [-0.01, -1, True, float("nan")],
)
def test_validate_non_negative_invalid(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_non_negative(
            value,
            "value",
        )


@pytest.mark.parametrize(
    "value",
    [0.0, 0.25, 0.5, 0.75, 1.0],
)
def test_validate_ratio_valid(value):
    assert _validate_ratio(
        value,
        "value",
    ) == float(value)


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
def test_validate_ratio_invalid(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_ratio(
            value,
            "value",
        )


@pytest.mark.parametrize(
    "value",
    [1, 2, 10, 100],
)
def test_validate_positive_integer_valid(value):
    assert _validate_positive_integer(
        value,
        "value",
    ) == value


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "1", None],
)
def test_validate_positive_integer_invalid(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_positive_integer(
            value,
            "value",
        )


# ============================================================
# GeneralizationGap
# ============================================================


def test_generalization_gap_creation():
    gap = make_gap()

    assert gap.in_sample_score == 1.0
    assert gap.out_of_sample_score == 0.8
    assert gap.degradation_ratio == 0.2
    assert gap.acceptable is True


def test_generalization_gap_frozen():
    gap = make_gap()

    with pytest.raises(FrozenInstanceError):
        gap.acceptable = False


def test_generalization_gap_fingerprint():
    first = make_gap()
    second = make_gap()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_generalization_gap_fingerprint_changes():
    first = make_gap(
        acceptable=True
    )
    second = make_gap(
        acceptable=False
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "field",
    [
        "positive_out_of_sample",
        "acceptable",
    ],
)
def test_generalization_gap_invalid_bool(field):
    kwargs = {
        "in_sample_score": 1.0,
        "out_of_sample_score": 0.8,
        "absolute_gap": 0.2,
        "degradation_ratio": 0.2,
        "retained_performance_ratio": 0.8,
        "positive_out_of_sample": True,
        "acceptable": True,
    }

    kwargs[field] = 1

    with pytest.raises(
        ValidationConfigurationError
    ):
        GeneralizationGap(**kwargs)


# ============================================================
# SelectionBiasDiagnostics
# ============================================================


def test_selection_bias_creation():
    result = make_selection_bias()

    assert result.tested_strategies == 10
    assert result.selected_rank == 1
    assert result.risk_score == 0.2


def test_selection_bias_frozen():
    result = make_selection_bias()

    with pytest.raises(FrozenInstanceError):
        result.risk_score = 0.5


def test_selection_bias_fingerprint():
    first = make_selection_bias()
    second = make_selection_bias()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "tested",
    [0, -1, True, 1.5],
)
def test_selection_bias_invalid_tested_count(
    tested,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SelectionBiasDiagnostics(
            tested_strategies=tested,
            selected_rank=1,
            best_score=1.0,
            median_score=0.8,
            score_advantage=0.2,
            normalized_advantage=0.2,
            selection_pressure=0.3,
            risk_score=0.2,
        )


def test_selection_bias_rank_exceeds_count():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SelectionBiasDiagnostics(
            tested_strategies=5,
            selected_rank=6,
            best_score=1.0,
            median_score=0.8,
            score_advantage=0.2,
            normalized_advantage=0.2,
            selection_pressure=0.3,
            risk_score=0.2,
        )


# ============================================================
# ParameterInstabilityDiagnostics
# ============================================================


def test_parameter_instability_creation():
    result = make_parameter_instability()

    assert result.parameter_count == 3
    assert result.stable_parameter_count == 3
    assert result.instability_score == 0.1


def test_parameter_instability_frozen():
    result = make_parameter_instability()

    with pytest.raises(FrozenInstanceError):
        result.instability_score = 0.5


def test_parameter_instability_fingerprint():
    first = make_parameter_instability()
    second = make_parameter_instability()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_parameter_instability_stable_count_exceeds_total():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterInstabilityDiagnostics(
            parameter_count=2,
            stable_parameter_count=3,
            stable_parameter_ratio=1.0,
            average_stability_score=0.9,
            minimum_stability_score=0.8,
            instability_score=0.1,
        )


@pytest.mark.parametrize(
    "stable_count",
    [-1, True, 1.5],
)
def test_parameter_instability_invalid_stable_count(
    stable_count,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterInstabilityDiagnostics(
            parameter_count=3,
            stable_parameter_count=stable_count,
            stable_parameter_ratio=0.5,
            average_stability_score=0.5,
            minimum_stability_score=0.5,
            instability_score=0.5,
        )


# ============================================================
# OverfittingGuardReport
# ============================================================


def test_report_creation():
    report = make_report()

    assert report.risk_level == OverfittingRisk.LOW
    assert report.passed is True
    assert report.reasons == []


def test_report_frozen():
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.passed = False


def test_report_fingerprint():
    first = make_report()
    second = make_report()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_invalid_gap():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingGuardReport(
            generalization_gap="bad",
            selection_bias=make_selection_bias(),
            parameter_instability=(
                make_parameter_instability()
            ),
            multiple_testing_risk=0.1,
            regime_instability_risk=0.1,
            robustness_failure_risk=0.1,
            overall_risk_score=0.2,
            risk_level=OverfittingRisk.LOW,
            passed=True,
        )


def test_report_invalid_risk_level():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingGuardReport(
            generalization_gap=make_gap(),
            selection_bias=make_selection_bias(),
            parameter_instability=(
                make_parameter_instability()
            ),
            multiple_testing_risk=0.1,
            regime_instability_risk=0.1,
            robustness_failure_risk=0.1,
            overall_risk_score=0.2,
            risk_level="LOW",
            passed=True,
        )


def test_report_invalid_reasons():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingGuardReport(
            generalization_gap=make_gap(),
            selection_bias=make_selection_bias(),
            parameter_instability=(
                make_parameter_instability()
            ),
            multiple_testing_risk=0.1,
            regime_instability_risk=0.1,
            robustness_failure_risk=0.1,
            overall_risk_score=0.2,
            risk_level=OverfittingRisk.LOW,
            passed=True,
            reasons="bad",
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    guard = OverfittingGuard()

    assert guard.maximum_generalization_gap == 0.50
    assert guard.minimum_retained_performance == 0.50
    assert guard.maximum_overall_risk == 0.50
    assert guard.high_risk_threshold == 0.70
    assert guard.critical_risk_threshold == 0.85


def test_custom_engine():
    guard = OverfittingGuard(
        make_config(),
        maximum_generalization_gap=0.40,
        minimum_retained_performance=0.60,
        maximum_overall_risk=0.40,
        high_risk_threshold=0.65,
        critical_risk_threshold=0.80,
    )

    assert guard.maximum_generalization_gap == 0.40
    assert guard.minimum_retained_performance == 0.60
    assert guard.maximum_overall_risk == 0.40
    assert guard.high_risk_threshold == 0.65
    assert guard.critical_risk_threshold == 0.80


def test_invalid_threshold_order_high():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingGuard(
            maximum_overall_risk=0.60,
            high_risk_threshold=0.50,
        )


def test_invalid_threshold_order_critical():
    with pytest.raises(
        ValidationConfigurationError
    ):
        OverfittingGuard(
            high_risk_threshold=0.80,
            critical_risk_threshold=0.70,
        )


# ============================================================
# Generalization analysis
# ============================================================


def test_generalization_good():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=1.0,
        out_of_sample_score=0.8,
    )

    assert result.absolute_gap == pytest.approx(0.2)
    assert result.degradation_ratio == pytest.approx(0.2)
    assert result.retained_performance_ratio == pytest.approx(0.8)
    assert result.positive_out_of_sample is True
    assert result.acceptable is True


def test_generalization_bad():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=1.0,
        out_of_sample_score=0.2,
    )

    assert result.degradation_ratio == pytest.approx(0.8)
    assert result.acceptable is False


def test_generalization_negative_oos():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=1.0,
        out_of_sample_score=-0.1,
    )

    assert result.positive_out_of_sample is False
    assert result.acceptable is False


def test_generalization_zero_zero():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=0.0,
        out_of_sample_score=0.0,
    )

    assert result.degradation_ratio == 0.0
    assert result.retained_performance_ratio == 1.0
    assert result.acceptable is False


def test_generalization_zero_positive():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=0.0,
        out_of_sample_score=0.5,
    )

    assert result.degradation_ratio == -1.0
    assert result.retained_performance_ratio == 1.0
    assert result.acceptable is True


def test_generalization_zero_negative():
    guard = OverfittingGuard()

    result = guard.analyze_generalization_gap(
        in_sample_score=0.0,
        out_of_sample_score=-0.5,
    )

    assert result.degradation_ratio == 1.0
    assert result.retained_performance_ratio == 0.0
    assert result.acceptable is False


# ============================================================
# Selection-bias analysis
# ============================================================


def test_selection_bias_single_strategy():
    guard = OverfittingGuard()

    result = guard.analyze_selection_bias(
        [1.0]
    )

    assert result.tested_strategies == 1
    assert result.selection_pressure == 0.0
    assert 0.0 <= result.risk_score <= 1.0


def test_selection_bias_many_strategies():
    guard = OverfittingGuard()

    scores = [
        float(value) / 100.0
        for value in range(1, 101)
    ]

    result = guard.analyze_selection_bias(
        scores
    )

    assert result.tested_strategies == 100
    assert result.selection_pressure > 0.0
    assert result.risk_score > 0.0


def test_selection_bias_even_median():
    guard = OverfittingGuard()

    result = guard.analyze_selection_bias(
        [0.2, 0.4, 0.6, 0.8]
    )

    assert result.median_score == pytest.approx(0.5)


def test_selection_bias_odd_median():
    guard = OverfittingGuard()

    result = guard.analyze_selection_bias(
        [0.2, 0.4, 0.6]
    )

    assert result.median_score == pytest.approx(0.4)


def test_selection_bias_selected_rank():
    guard = OverfittingGuard()

    result = guard.analyze_selection_bias(
        [1.0, 0.9, 0.8],
        selected_rank=2,
    )

    assert result.selected_rank == 2
    assert result.best_score == pytest.approx(0.9)


def test_selection_bias_empty():
    guard = OverfittingGuard()

    with pytest.raises(OverfittingGuardError):
        guard.analyze_selection_bias([])


def test_selection_bias_invalid_rank():
    guard = OverfittingGuard()

    with pytest.raises(
        ValidationConfigurationError
    ):
        guard.analyze_selection_bias(
            [1.0, 0.9],
            selected_rank=3,
        )


# ============================================================
# Parameter-instability analysis
# ============================================================


def test_parameter_instability_stable():
    guard = OverfittingGuard()

    result = guard.analyze_parameter_instability(
        [0.90, 0.85, 0.80]
    )

    assert result.parameter_count == 3
    assert result.stable_parameter_count == 3
    assert result.stable_parameter_ratio == 1.0
    assert result.instability_score < 0.20


def test_parameter_instability_unstable():
    guard = OverfittingGuard()

    result = guard.analyze_parameter_instability(
        [0.20, 0.30, 0.40]
    )

    assert result.stable_parameter_count == 0
    assert result.stable_parameter_ratio == 0.0
    assert result.instability_score > 0.50


def test_parameter_instability_custom_threshold():
    guard = OverfittingGuard()

    result = guard.analyze_parameter_instability(
        [0.70, 0.80],
        stable_threshold=0.75,
    )

    assert result.stable_parameter_count == 1
    assert result.stable_parameter_ratio == 0.5


def test_parameter_instability_empty():
    guard = OverfittingGuard()

    with pytest.raises(OverfittingGuardError):
        guard.analyze_parameter_instability([])


# ============================================================
# Risk classification
# ============================================================


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.00, OverfittingRisk.LOW),
        (0.25, OverfittingRisk.LOW),
        (0.50, OverfittingRisk.LOW),
        (0.51, OverfittingRisk.MODERATE),
        (0.69, OverfittingRisk.MODERATE),
        (0.70, OverfittingRisk.HIGH),
        (0.84, OverfittingRisk.HIGH),
        (0.85, OverfittingRisk.CRITICAL),
        (1.00, OverfittingRisk.CRITICAL),
    ],
)
def test_classify_risk(
    score,
    expected,
):
    guard = OverfittingGuard()

    assert guard.classify_risk(
        score
    ) == expected


# ============================================================
# Full evaluation
# ============================================================


def test_evaluate_low_risk_pass():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=0.9,
        strategy_scores=[
            0.90,
            0.88,
            0.86,
        ],
        parameter_stability_scores=[
            0.90,
            0.88,
            0.85,
        ],
        multiple_testing_risk=0.10,
        regime_instability_risk=0.10,
        robustness_failure_risk=0.10,
    )

    assert report.risk_level == OverfittingRisk.LOW
    assert report.passed is True
    assert report.overall_risk_score <= 0.50


def test_evaluate_high_risk_fail():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=-0.5,
        strategy_scores=[
            1.0,
            0.1,
            0.0,
            -0.1,
        ],
        parameter_stability_scores=[
            0.1,
            0.2,
            0.3,
        ],
        multiple_testing_risk=1.0,
        regime_instability_risk=1.0,
        robustness_failure_risk=1.0,
    )

    assert report.passed is False
    assert report.overall_risk_score > 0.50
    assert len(report.reasons) > 0


def test_evaluate_reasons():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=-0.2,
        strategy_scores=[
            1.0,
            0.1,
            0.0,
        ],
        parameter_stability_scores=[
            0.1,
            0.2,
        ],
        multiple_testing_risk=0.9,
        regime_instability_risk=0.9,
        robustness_failure_risk=0.9,
    )

    assert any(
        "generalization" in reason.lower()
        for reason in report.reasons
    )

    assert any(
        "multiple-testing" in reason.lower()
        for reason in report.reasons
    )

    assert any(
        "regime" in reason.lower()
        for reason in report.reasons
    )

    assert any(
        "robustness" in reason.lower()
        for reason in report.reasons
    )


def test_evaluate_anti_overfitting_disabled():
    guard = OverfittingGuard(
        make_config(
            anti_overfitting_enabled=False
        )
    )

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=0.9,
        strategy_scores=[
            0.9,
            0.88,
            0.86,
        ],
        parameter_stability_scores=[
            0.9,
            0.88,
            0.85,
        ],
    )

    assert report.passed is False


@pytest.mark.parametrize(
    "field",
    [
        "multiple_testing_risk",
        "regime_instability_risk",
        "robustness_failure_risk",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, True],
)
def test_evaluate_invalid_risk_component(
    field,
    value,
):
    guard = OverfittingGuard()

    kwargs = {
        "in_sample_score": 1.0,
        "out_of_sample_score": 0.9,
        "strategy_scores": [0.9, 0.8],
        "parameter_stability_scores": [0.9, 0.8],
        "multiple_testing_risk": 0.0,
        "regime_instability_risk": 0.0,
        "robustness_failure_risk": 0.0,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        guard.evaluate(**kwargs)


# ============================================================
# Mapping API
# ============================================================


def test_validate_mapping():
    guard = OverfittingGuard()

    report = guard.validate_mapping(
        {
            "in_sample_score": 1.0,
            "out_of_sample_score": 0.9,
            "strategy_scores": [
                0.9,
                0.88,
                0.86,
            ],
            "parameter_stability_scores": [
                0.9,
                0.88,
                0.85,
            ],
            "multiple_testing_risk": 0.1,
            "regime_instability_risk": 0.1,
            "robustness_failure_risk": 0.1,
        }
    )

    assert isinstance(
        report,
        OverfittingGuardReport,
    )


def test_validate_mapping_invalid_type():
    guard = OverfittingGuard()

    with pytest.raises(
        ValidationConfigurationError
    ):
        guard.validate_mapping("bad")


@pytest.mark.parametrize(
    "missing",
    [
        "in_sample_score",
        "out_of_sample_score",
        "strategy_scores",
        "parameter_stability_scores",
    ],
)
def test_validate_mapping_missing(missing):
    guard = OverfittingGuard()

    data = {
        "in_sample_score": 1.0,
        "out_of_sample_score": 0.9,
        "strategy_scores": [0.9, 0.8],
        "parameter_stability_scores": [0.9, 0.8],
    }

    del data[missing]

    with pytest.raises(
        OverfittingGuardError
    ):
        guard.validate_mapping(data)


# ============================================================
# Shared model adapter
# ============================================================


def test_shared_diagnostics_adapter_safe():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=0.9,
        strategy_scores=[
            0.9,
            0.88,
            0.86,
        ],
        parameter_stability_scores=[
            0.9,
            0.88,
            0.85,
        ],
    )

    diagnostics = (
        guard.to_overfitting_diagnostics(
            report
        )
    )

    assert (
        diagnostics is None
        or isinstance(
            diagnostics,
            object,
        )
    )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=0.9,
        strategy_scores=[
            0.9,
            0.88,
            0.86,
        ],
        parameter_stability_scores=[
            0.9,
            0.88,
            0.85,
        ],
        multiple_testing_risk=0.1,
        regime_instability_risk=0.1,
        robustness_failure_risk=0.1,
    )

    result = guard.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.OVERFITTING_CONTROL
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert 0.0 <= result.score <= 1.0

    assert result.metadata[
        "research_only"
    ] is True

    assert result.metadata[
        "real_money_execution"
    ] is False


def test_to_validation_result_fail():
    guard = OverfittingGuard()

    report = guard.evaluate(
        in_sample_score=1.0,
        out_of_sample_score=-0.5,
        strategy_scores=[
            1.0,
            0.1,
            0.0,
        ],
        parameter_stability_scores=[
            0.1,
            0.2,
        ],
        multiple_testing_risk=1.0,
        regime_instability_risk=1.0,
        robustness_failure_risk=1.0,
    )

    result = guard.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert result.score < 0.5


# ============================================================
# Determinism
# ============================================================


def test_generalization_deterministic():
    guard = OverfittingGuard()

    first = guard.analyze_generalization_gap(
        in_sample_score=1.0,
        out_of_sample_score=0.8,
    )

    second = guard.analyze_generalization_gap(
        in_sample_score=1.0,
        out_of_sample_score=0.8,
    )

    assert first.fingerprint == second.fingerprint


def test_evaluation_deterministic():
    guard = OverfittingGuard()

    kwargs = {
        "in_sample_score": 1.0,
        "out_of_sample_score": 0.9,
        "strategy_scores": [
            0.9,
            0.88,
            0.86,
        ],
        "parameter_stability_scores": [
            0.9,
            0.88,
            0.85,
        ],
        "multiple_testing_risk": 0.1,
        "regime_instability_risk": 0.1,
        "robustness_failure_risk": 0.1,
    }

    first = guard.evaluate(**kwargs)
    second = guard.evaluate(**kwargs)

    assert first.fingerprint == second.fingerprint


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    guard = OverfittingGuard(
        make_config(),
        maximum_generalization_gap=0.40,
        minimum_retained_performance=0.60,
        maximum_overall_risk=0.40,
        high_risk_threshold=0.65,
        critical_risk_threshold=0.80,
    )

    snapshot = guard.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert snapshot[
        "maximum_generalization_gap"
    ] == 0.40

    assert snapshot[
        "minimum_retained_performance"
    ] == 0.60

    assert snapshot[
        "maximum_overall_risk"
    ] == 0.40

    assert snapshot[
        "high_risk_threshold"
    ] == 0.65

    assert snapshot[
        "critical_risk_threshold"
    ] == 0.80

    assert snapshot[
        "anti_overfitting_enabled"
    ] is True

    assert snapshot[
        "research_only"
    ] is True

    assert snapshot[
        "real_money_execution"
    ] is False

    assert len(
        snapshot["config_fingerprint"]
    ) == 64


# ============================================================
# Complete anti-overfitting flow
# ============================================================


def test_complete_anti_overfitting_flow():
    guard = OverfittingGuard(
        make_config(),
        maximum_generalization_gap=0.50,
        minimum_retained_performance=0.50,
        maximum_overall_risk=0.50,
        high_risk_threshold=0.70,
        critical_risk_threshold=0.85,
    )

    report = guard.evaluate(
        in_sample_score=1.00,
        out_of_sample_score=0.85,
        strategy_scores=[
            0.85,
            0.83,
            0.81,
            0.79,
            0.78,
        ],
        parameter_stability_scores=[
            0.90,
            0.88,
            0.86,
            0.84,
        ],
        selected_rank=1,
        multiple_testing_risk=0.10,
        regime_instability_risk=0.10,
        robustness_failure_risk=0.10,
    )

    assert report.generalization_gap.acceptable is True

    assert (
        report.parameter_instability
        .stable_parameter_ratio
        == 1.0
    )

    assert (
        report.overall_risk_score
        <= 0.50
    )

    assert (
        report.risk_level
        == OverfittingRisk.LOW
    )

    assert report.passed is True

    validation = guard.to_validation_result(
        report
    )

    assert (
        validation.stage
        == ValidationStage.OVERFITTING_CONTROL
    )

    assert (
        validation.status
        == ValidationStatus.PASSED
    )

    assert validation.score > 0.50

    assert validation.metadata[
        "research_only"
    ] is True

    assert validation.metadata[
        "real_money_execution"
    ] is False

    assert len(report.fingerprint) == 64