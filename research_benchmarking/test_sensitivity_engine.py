from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.sensitivity_engine import (
    VERSION,
    ParameterPerturbation,
    ParameterSensitivity,
    SensitivityEngine,
    SensitivityReport,
    SensitivityValidationError,
    _validate_non_negative,
    _validate_numeric,
    _validate_ratio,
)


# ============================================================
# Helpers
# ============================================================


def make_config() -> ValidationConfig:
    return ValidationConfig(
        minimum_observations=30,
        research_only=True,
        real_money_execution=False,
    )


def make_perturbation(
    *,
    parameter_name: str = "lookback",
    baseline_value: float = 20.0,
    perturbed_value: float = 18.0,
    baseline_score: float = 1.0,
    perturbed_score: float = 0.90,
) -> ParameterPerturbation:
    return ParameterPerturbation(
        parameter_name=parameter_name,
        baseline_value=baseline_value,
        perturbed_value=perturbed_value,
        baseline_score=baseline_score,
        perturbed_score=perturbed_score,
    )


def make_parameter_sensitivity(
    *,
    parameter_name: str = "lookback",
    stability_score: float = 0.90,
    stable: bool = True,
) -> ParameterSensitivity:
    perturbations = [
        make_perturbation(
            parameter_name=parameter_name,
            perturbed_value=18.0,
            perturbed_score=0.90,
        ),
        make_perturbation(
            parameter_name=parameter_name,
            perturbed_value=22.0,
            perturbed_score=0.85,
        ),
    ]

    return ParameterSensitivity(
        parameter_name=parameter_name,
        perturbations=perturbations,
        average_absolute_degradation_ratio=0.125,
        worst_degradation_ratio=0.15,
        positive_ratio=1.0,
        stable_ratio=1.0,
        local_stability_score=stability_score,
        stable=stable,
    )


def make_report(
    *,
    passed: bool = True,
) -> SensitivityReport:
    results = [
        make_parameter_sensitivity(
            parameter_name="lookback",
            stability_score=0.90,
        ),
        make_parameter_sensitivity(
            parameter_name="threshold",
            stability_score=0.80,
        ),
    ]

    return SensitivityReport(
        parameter_results=results,
        average_stability_score=0.85,
        stable_parameter_ratio=1.0,
        minimum_stability_score=0.80,
        most_sensitive_parameter="threshold",
        least_sensitive_parameter="lookback",
        passed=passed,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert SensitivityEngine.VERSION == "v6.0.2"


# ============================================================
# Numeric validation
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
    [0, 0.5, 1, 10],
)
def test_validate_non_negative_valid(value):
    assert _validate_non_negative(
        value,
        "value",
    ) == float(value)


@pytest.mark.parametrize(
    "value",
    [-0.01, -1],
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


# ============================================================
# ParameterPerturbation
# ============================================================


def test_perturbation_creation():
    item = make_perturbation()

    assert item.parameter_name == "lookback"
    assert item.parameter_change == -2.0
    assert item.absolute_parameter_change == 2.0
    assert item.relative_parameter_change == pytest.approx(
        -0.10
    )
    assert item.score_change == pytest.approx(-0.10)
    assert item.degradation == pytest.approx(0.10)
    assert item.degradation_ratio == pytest.approx(0.10)
    assert item.positive_after_perturbation is True


def test_perturbation_improvement():
    item = make_perturbation(
        perturbed_score=1.20,
    )

    assert item.score_change == pytest.approx(0.20)
    assert item.degradation == pytest.approx(-0.20)
    assert item.degradation_ratio == pytest.approx(-0.20)


def test_zero_baseline_parameter_unchanged():
    item = make_perturbation(
        baseline_value=0.0,
        perturbed_value=0.0,
    )

    assert item.relative_parameter_change == 0.0


def test_zero_baseline_parameter_changed():
    item = make_perturbation(
        baseline_value=0.0,
        perturbed_value=0.1,
    )

    assert item.relative_parameter_change == 1.0


def test_zero_baseline_score_same():
    item = make_perturbation(
        baseline_score=0.0,
        perturbed_score=0.0,
    )

    assert item.degradation_ratio == 0.0


def test_zero_baseline_score_negative():
    item = make_perturbation(
        baseline_score=0.0,
        perturbed_score=-0.1,
    )

    assert item.degradation_ratio == 1.0


def test_zero_baseline_score_positive():
    item = make_perturbation(
        baseline_score=0.0,
        perturbed_score=0.1,
    )

    assert item.degradation_ratio == -1.0


def test_perturbation_frozen():
    item = make_perturbation()

    with pytest.raises(FrozenInstanceError):
        item.perturbed_score = 0.5


def test_perturbation_fingerprint():
    first = make_perturbation()
    second = make_perturbation()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_perturbation_fingerprint_changes():
    first = make_perturbation(
        perturbed_score=0.9
    )
    second = make_perturbation(
        perturbed_score=0.8
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "name",
    ["", "   ", None, 123],
)
def test_invalid_parameter_name(name):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_perturbation(
            parameter_name=name
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_value", True),
        ("baseline_value", float("nan")),
        ("perturbed_value", float("inf")),
        ("baseline_score", "bad"),
        ("perturbed_score", None),
    ],
)
def test_invalid_perturbation_numeric(
    field,
    value,
):
    kwargs = {
        "baseline_value": 20.0,
        "perturbed_value": 18.0,
        "baseline_score": 1.0,
        "perturbed_score": 0.9,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterPerturbation(
            parameter_name="lookback",
            **kwargs,
        )


# ============================================================
# ParameterSensitivity
# ============================================================


def test_parameter_sensitivity_creation():
    result = make_parameter_sensitivity()

    assert result.parameter_name == "lookback"
    assert result.perturbation_count == 2
    assert result.stable is True


def test_parameter_sensitivity_frozen():
    result = make_parameter_sensitivity()

    with pytest.raises(FrozenInstanceError):
        result.stable = False


def test_parameter_sensitivity_fingerprint():
    first = make_parameter_sensitivity()
    second = make_parameter_sensitivity()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_parameter_sensitivity_empty():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterSensitivity(
            parameter_name="lookback",
            perturbations=[],
            average_absolute_degradation_ratio=0.0,
            worst_degradation_ratio=0.0,
            positive_ratio=1.0,
            stable_ratio=1.0,
            local_stability_score=1.0,
            stable=True,
        )


def test_parameter_sensitivity_invalid_type():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterSensitivity(
            parameter_name="lookback",
            perturbations=["bad"],
            average_absolute_degradation_ratio=0.0,
            worst_degradation_ratio=0.0,
            positive_ratio=1.0,
            stable_ratio=1.0,
            local_stability_score=1.0,
            stable=True,
        )


def test_parameter_sensitivity_mixed_names():
    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterSensitivity(
            parameter_name="lookback",
            perturbations=[
                make_perturbation(
                    parameter_name="threshold"
                )
            ],
            average_absolute_degradation_ratio=0.1,
            worst_degradation_ratio=0.1,
            positive_ratio=1.0,
            stable_ratio=1.0,
            local_stability_score=0.9,
            stable=True,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("positive_ratio", -0.01),
        ("positive_ratio", 1.01),
        ("stable_ratio", -0.01),
        ("stable_ratio", 1.01),
        ("local_stability_score", -0.01),
        ("local_stability_score", 1.01),
    ],
)
def test_parameter_sensitivity_invalid_ratio(
    field,
    value,
):
    kwargs = {
        "positive_ratio": 1.0,
        "stable_ratio": 1.0,
        "local_stability_score": 0.9,
    }
    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        ParameterSensitivity(
            parameter_name="lookback",
            perturbations=[
                make_perturbation()
            ],
            average_absolute_degradation_ratio=0.1,
            worst_degradation_ratio=0.1,
            stable=True,
            **kwargs,
        )


# ============================================================
# SensitivityReport
# ============================================================


def test_report_creation():
    report = make_report()

    assert report.parameter_count == 2
    assert report.stable_parameter_count == 2
    assert report.unstable_parameters == []
    assert report.passed is True


def test_report_frozen():
    report = make_report()

    with pytest.raises(FrozenInstanceError):
        report.passed = False


def test_report_fingerprint():
    first = make_report()
    second = make_report()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_report_empty():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityReport(
            parameter_results=[],
            average_stability_score=0.0,
            stable_parameter_ratio=0.0,
            minimum_stability_score=0.0,
            most_sensitive_parameter="a",
            least_sensitive_parameter="b",
            passed=False,
        )


def test_report_invalid_result_type():
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityReport(
            parameter_results=["bad"],
            average_stability_score=0.5,
            stable_parameter_ratio=0.5,
            minimum_stability_score=0.5,
            most_sensitive_parameter="a",
            least_sensitive_parameter="b",
            passed=False,
        )


def test_report_unstable_parameters():
    results = [
        make_parameter_sensitivity(
            parameter_name="stable",
            stability_score=0.9,
            stable=True,
        ),
        make_parameter_sensitivity(
            parameter_name="unstable",
            stability_score=0.3,
            stable=False,
        ),
    ]

    report = SensitivityReport(
        parameter_results=results,
        average_stability_score=0.6,
        stable_parameter_ratio=0.5,
        minimum_stability_score=0.3,
        most_sensitive_parameter="unstable",
        least_sensitive_parameter="stable",
        passed=False,
    )

    assert report.stable_parameter_count == 1
    assert report.unstable_parameters == ["unstable"]


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = SensitivityEngine()

    assert engine.maximum_degradation_ratio == 0.50
    assert engine.minimum_positive_ratio == 0.70
    assert engine.minimum_local_stability_score == 0.65
    assert engine.minimum_stable_parameter_ratio == 0.70


def test_custom_engine():
    engine = SensitivityEngine(
        make_config(),
        maximum_degradation_ratio=0.30,
        minimum_positive_ratio=0.80,
        minimum_local_stability_score=0.75,
        minimum_stable_parameter_ratio=0.90,
    )

    assert engine.maximum_degradation_ratio == 0.30
    assert engine.minimum_positive_ratio == 0.80
    assert engine.minimum_local_stability_score == 0.75
    assert engine.minimum_stable_parameter_ratio == 0.90


@pytest.mark.parametrize(
    "value",
    [-0.01, True, float("nan")],
)
def test_invalid_maximum_degradation(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityEngine(
            maximum_degradation_ratio=value
        )


@pytest.mark.parametrize(
    "field",
    [
        "minimum_positive_ratio",
        "minimum_local_stability_score",
        "minimum_stable_parameter_ratio",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, True, float("nan")],
)
def test_invalid_engine_ratio(field, value):
    kwargs = {
        field: value
    }

    with pytest.raises(
        ValidationConfigurationError
    ):
        SensitivityEngine(**kwargs)


# ============================================================
# Perturbation grid
# ============================================================


def test_build_default_grid():
    engine = SensitivityEngine()

    grid = engine.build_perturbation_grid(
        parameter_name="lookback",
        baseline_value=100.0,
    )

    assert grid == pytest.approx(
        [
            80.0,
            90.0,
            95.0,
            105.0,
            110.0,
            120.0,
        ]
    )


def test_build_custom_grid():
    engine = SensitivityEngine()

    grid = engine.build_perturbation_grid(
        parameter_name="threshold",
        baseline_value=10.0,
        shifts=[-0.5, 0.5],
    )

    assert grid == pytest.approx(
        [5.0, 15.0]
    )


def test_build_grid_zero_baseline():
    engine = SensitivityEngine()

    grid = engine.build_perturbation_grid(
        parameter_name="threshold",
        baseline_value=0.0,
        shifts=[-0.2, 0.2],
    )

    assert grid == pytest.approx(
        [-0.2, 0.2]
    )


def test_build_grid_empty_shifts():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.build_perturbation_grid(
            parameter_name="lookback",
            baseline_value=20.0,
            shifts=[],
        )


def test_build_grid_invalid_name():
    engine = SensitivityEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.build_perturbation_grid(
            parameter_name="",
            baseline_value=20.0,
        )


# ============================================================
# Create perturbation
# ============================================================


def test_create_perturbation():
    engine = SensitivityEngine()

    item = engine.create_perturbation(
        parameter_name="lookback",
        baseline_value=20.0,
        perturbed_value=22.0,
        baseline_score=1.0,
        perturbed_score=0.9,
    )

    assert isinstance(
        item,
        ParameterPerturbation,
    )
    assert item.parameter_name == "lookback"
    assert item.metadata["research_only"] is True
    assert item.metadata["real_money_execution"] is False


# ============================================================
# Analyze parameter
# ============================================================


def test_analyze_parameter_stable():
    engine = SensitivityEngine()

    result = engine.analyze_parameter(
        parameter_name="lookback",
        baseline_value=20.0,
        baseline_score=1.0,
        perturbed_values=[
            16.0,
            18.0,
            19.0,
            21.0,
            22.0,
            24.0,
        ],
        perturbed_scores=[
            0.80,
            0.90,
            0.95,
            0.96,
            0.91,
            0.82,
        ],
    )

    assert result.perturbation_count == 6
    assert result.positive_ratio == 1.0
    assert result.stable_ratio == 1.0
    assert result.local_stability_score > 0.80
    assert result.stable is True


def test_analyze_parameter_unstable():
    engine = SensitivityEngine()

    result = engine.analyze_parameter(
        parameter_name="threshold",
        baseline_value=1.0,
        baseline_score=1.0,
        perturbed_values=[
            0.8,
            0.9,
            1.1,
            1.2,
        ],
        perturbed_scores=[
            -0.5,
            0.1,
            -0.2,
            0.2,
        ],
    )

    assert result.positive_ratio == 0.5
    assert result.stable is False


def test_analyze_parameter_improvements():
    engine = SensitivityEngine()

    result = engine.analyze_parameter(
        parameter_name="threshold",
        baseline_value=1.0,
        baseline_score=1.0,
        perturbed_values=[
            0.9,
            1.1,
        ],
        perturbed_scores=[
            1.1,
            1.2,
        ],
    )

    assert (
        result.average_absolute_degradation_ratio
        == 0.0
    )
    assert result.stable is True


def test_analyze_empty_values():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.analyze_parameter(
            parameter_name="lookback",
            baseline_value=20.0,
            baseline_score=1.0,
            perturbed_values=[],
            perturbed_scores=[0.9],
        )


def test_analyze_empty_scores():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.analyze_parameter(
            parameter_name="lookback",
            baseline_value=20.0,
            baseline_score=1.0,
            perturbed_values=[18.0],
            perturbed_scores=[],
        )


def test_analyze_length_mismatch():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.analyze_parameter(
            parameter_name="lookback",
            baseline_value=20.0,
            baseline_score=1.0,
            perturbed_values=[
                18.0,
                22.0,
            ],
            perturbed_scores=[
                0.9
            ],
        )


# ============================================================
# Aggregate
# ============================================================


def test_aggregate_pass():
    engine = SensitivityEngine()

    results = [
        engine.analyze_parameter(
            parameter_name="lookback",
            baseline_value=20.0,
            baseline_score=1.0,
            perturbed_values=[
                18.0,
                22.0,
            ],
            perturbed_scores=[
                0.9,
                0.85,
            ],
        ),
        engine.analyze_parameter(
            parameter_name="threshold",
            baseline_value=1.0,
            baseline_score=1.0,
            perturbed_values=[
                0.9,
                1.1,
            ],
            perturbed_scores=[
                0.95,
                0.90,
            ],
        ),
    ]

    report = engine.aggregate(results)

    assert report.parameter_count == 2
    assert report.stable_parameter_count == 2
    assert report.stable_parameter_ratio == 1.0
    assert report.passed is True


def test_aggregate_identifies_sensitive_parameters():
    engine = SensitivityEngine()

    strong = engine.analyze_parameter(
        parameter_name="strong",
        baseline_value=1.0,
        baseline_score=1.0,
        perturbed_values=[0.9, 1.1],
        perturbed_scores=[0.98, 0.97],
    )

    weak = engine.analyze_parameter(
        parameter_name="weak",
        baseline_value=1.0,
        baseline_score=1.0,
        perturbed_values=[0.9, 1.1],
        perturbed_scores=[0.55, 0.60],
    )

    report = engine.aggregate(
        [strong, weak]
    )

    assert (
        report.most_sensitive_parameter
        == "weak"
    )
    assert (
        report.least_sensitive_parameter
        == "strong"
    )


def test_aggregate_empty():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.aggregate([])


def test_aggregate_invalid_type():
    engine = SensitivityEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.aggregate(
            [
                make_parameter_sensitivity(),
                "bad",
            ]
        )


# ============================================================
# Validate parameter mappings
# ============================================================


def test_validate_parameters():
    engine = SensitivityEngine()

    report = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data={
            "lookback": {
                "baseline_value": 20.0,
                "perturbed_values": [
                    18.0,
                    22.0,
                ],
                "perturbed_scores": [
                    0.9,
                    0.85,
                ],
            },
            "threshold": {
                "baseline_value": 1.0,
                "perturbed_values": [
                    0.9,
                    1.1,
                ],
                "perturbed_scores": [
                    0.95,
                    0.90,
                ],
            },
        },
    )

    assert report.parameter_count == 2
    assert report.passed is True


def test_validate_parameters_empty():
    engine = SensitivityEngine()

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.validate_parameters(
            baseline_score=1.0,
            parameter_data={},
        )


def test_validate_parameters_invalid_entry():
    engine = SensitivityEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.validate_parameters(
            baseline_score=1.0,
            parameter_data={
                "lookback": "bad"
            },
        )


@pytest.mark.parametrize(
    "missing",
    [
        "baseline_value",
        "perturbed_values",
        "perturbed_scores",
    ],
)
def test_validate_parameters_missing_field(
    missing,
):
    engine = SensitivityEngine()

    data = {
        "baseline_value": 20.0,
        "perturbed_values": [
            18.0,
            22.0,
        ],
        "perturbed_scores": [
            0.9,
            0.85,
        ],
    }

    del data[missing]

    with pytest.raises(
        SensitivityValidationError
    ):
        engine.validate_parameters(
            baseline_score=1.0,
            parameter_data={
                "lookback": data
            },
        )


# ============================================================
# Score
# ============================================================


def test_score_bounded():
    engine = SensitivityEngine()

    report = make_report()

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_strong_report_high_score():
    engine = SensitivityEngine()

    strong = engine.analyze_parameter(
        parameter_name="lookback",
        baseline_value=20.0,
        baseline_score=1.0,
        perturbed_values=[
            18.0,
            19.0,
            21.0,
            22.0,
        ],
        perturbed_scores=[
            0.98,
            0.99,
            0.98,
            0.97,
        ],
    )

    report = engine.aggregate(
        [strong]
    )

    assert (
        engine._calculate_score(report)
        > 0.90
    )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = SensitivityEngine()

    report = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data={
            "lookback": {
                "baseline_value": 20.0,
                "perturbed_values": [
                    18.0,
                    22.0,
                ],
                "perturbed_scores": [
                    0.9,
                    0.85,
                ],
            },
        },
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.SENSITIVITY
    )
    assert (
        result.status
        == ValidationStatus.PASSED
    )
    assert 0.0 <= result.score <= 1.0
    assert result.diagnostics[
        "parameter_count"
    ] == 1
    assert result.metadata[
        "research_only"
    ] is True
    assert result.metadata[
        "real_money_execution"
    ] is False


def test_to_validation_result_fail():
    engine = SensitivityEngine()

    report = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data={
            "threshold": {
                "baseline_value": 1.0,
                "perturbed_values": [
                    0.8,
                    0.9,
                    1.1,
                    1.2,
                ],
                "perturbed_scores": [
                    -0.4,
                    0.1,
                    -0.2,
                    0.2,
                ],
            },
        },
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


# ============================================================
# Shared model adapter
# ============================================================


def test_shared_model_adapter_safe():
    engine = SensitivityEngine()

    report = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data={
            "lookback": {
                "baseline_value": 20.0,
                "perturbed_values": [
                    18.0,
                    22.0,
                ],
                "perturbed_scores": [
                    0.9,
                    0.85,
                ],
            },
        },
    )

    converted = engine.to_sensitivity_results(
        report
    )

    assert isinstance(converted, list)


# ============================================================
# Determinism
# ============================================================


def test_parameter_analysis_deterministic():
    engine = SensitivityEngine()

    kwargs = {
        "parameter_name": "lookback",
        "baseline_value": 20.0,
        "baseline_score": 1.0,
        "perturbed_values": [
            18.0,
            22.0,
        ],
        "perturbed_scores": [
            0.9,
            0.85,
        ],
    }

    first = engine.analyze_parameter(
        **kwargs
    )
    second = engine.analyze_parameter(
        **kwargs
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_report_deterministic():
    engine = SensitivityEngine()

    data = {
        "lookback": {
            "baseline_value": 20.0,
            "perturbed_values": [
                18.0,
                22.0,
            ],
            "perturbed_scores": [
                0.9,
                0.85,
            ],
        },
    }

    first = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data=data,
    )

    second = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data=data,
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = SensitivityEngine(
        make_config(),
        maximum_degradation_ratio=0.40,
        minimum_positive_ratio=0.80,
        minimum_local_stability_score=0.75,
        minimum_stable_parameter_ratio=0.90,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"
    assert snapshot[
        "maximum_degradation_ratio"
    ] == 0.40
    assert snapshot[
        "minimum_positive_ratio"
    ] == 0.80
    assert snapshot[
        "minimum_local_stability_score"
    ] == 0.75
    assert snapshot[
        "minimum_stable_parameter_ratio"
    ] == 0.90

    assert snapshot[
        "default_parameter_shifts"
    ] == [
        -0.20,
        -0.10,
        -0.05,
        0.05,
        0.10,
        0.20,
    ]

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
# Complete flow
# ============================================================


def test_complete_sensitivity_flow():
    engine = SensitivityEngine(
        make_config(),
        maximum_degradation_ratio=0.50,
        minimum_positive_ratio=0.70,
        minimum_local_stability_score=0.65,
        minimum_stable_parameter_ratio=0.70,
    )

    lookback_values = (
        engine.build_perturbation_grid(
            parameter_name="lookback",
            baseline_value=20.0,
        )
    )

    threshold_values = (
        engine.build_perturbation_grid(
            parameter_name="threshold",
            baseline_value=1.0,
        )
    )

    report = engine.validate_parameters(
        baseline_score=1.0,
        parameter_data={
            "lookback": {
                "baseline_value": 20.0,
                "perturbed_values": (
                    lookback_values
                ),
                "perturbed_scores": [
                    0.82,
                    0.90,
                    0.96,
                    0.95,
                    0.89,
                    0.81,
                ],
            },
            "threshold": {
                "baseline_value": 1.0,
                "perturbed_values": (
                    threshold_values
                ),
                "perturbed_scores": [
                    0.80,
                    0.88,
                    0.94,
                    0.96,
                    0.90,
                    0.83,
                ],
            },
        },
    )

    assert report.parameter_count == 2
    assert report.stable_parameter_count == 2
    assert report.stable_parameter_ratio == 1.0
    assert report.average_stability_score > 0.80
    assert report.passed is True

    validation = (
        engine.to_validation_result(
            report
        )
    )

    assert (
        validation.stage
        == ValidationStage.SENSITIVITY
    )
    assert (
        validation.status
        == ValidationStatus.PASSED
    )
    assert validation.score > 0.80

    assert validation.metadata[
        "research_only"
    ] is True
    assert validation.metadata[
        "real_money_execution"
    ] is False

    assert len(report.fingerprint) == 64