from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from research_benchmarking.models import (
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.robustness_engine import (
    VERSION,
    RobustnessEngine,
    RobustnessReport,
    RobustnessScenario,
    RobustnessScenarioResult,
    RobustnessValidationError,
    _validate_non_negative,
    _validate_numeric,
    _validate_ratio,
    _validate_scores,
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


def make_scenario(
    *,
    scenario_id: str = "stress_1",
    category: str = "transaction_cost",
    severity: float = 0.50,
    cost_bps: float = 10.0,
    slippage_bps: float = 0.0,
    execution_delay: int = 0,
    parameter_shift: float = 0.0,
    data_perturbation: float = 0.0,
) -> RobustnessScenario:
    return RobustnessScenario(
        scenario_id=scenario_id,
        category=category,
        severity=severity,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        execution_delay=execution_delay,
        parameter_shift=parameter_shift,
        data_perturbation=data_perturbation,
    )


def make_result(
    *,
    scenario_id: str = "stress_1",
    category: str = "transaction_cost",
    baseline_score: float = 1.0,
    stressed_score: float = 0.80,
    degradation: float = 0.20,
    degradation_ratio: float = 0.20,
    positive_after_stress: bool = True,
    stable: bool = True,
) -> RobustnessScenarioResult:
    return RobustnessScenarioResult(
        scenario=make_scenario(
            scenario_id=scenario_id,
            category=category,
        ),
        baseline_score=baseline_score,
        stressed_score=stressed_score,
        degradation=degradation,
        degradation_ratio=degradation_ratio,
        positive_after_stress=positive_after_stress,
        stable=stable,
    )


def make_report(
    *,
    passed: bool = True,
) -> RobustnessReport:
    results = [
        make_result(
            scenario_id="a",
            stressed_score=0.90,
            degradation=0.10,
            degradation_ratio=0.10,
        ),
        make_result(
            scenario_id="b",
            stressed_score=0.80,
            degradation=0.20,
            degradation_ratio=0.20,
        ),
    ]

    return RobustnessReport(
        baseline_score=1.0,
        scenario_results=results,
        average_stressed_score=0.85,
        average_degradation=0.15,
        average_degradation_ratio=0.15,
        positive_survival_ratio=1.0,
        stable_scenario_ratio=1.0,
        worst_stressed_score=0.80,
        worst_degradation_ratio=0.20,
        passed=passed,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


def test_engine_version():
    assert RobustnessEngine.VERSION == "v6.0.2"


# ============================================================
# Numeric helpers
# ============================================================


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -1,
        0.5,
        -0.5,
    ],
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
    [
        0,
        1,
        0.5,
        100,
    ],
)
def test_validate_non_negative_valid(value):
    assert _validate_non_negative(
        value,
        "value",
    ) == float(value)


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        -1,
    ],
)
def test_validate_non_negative_negative(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_non_negative(
            value,
            "value",
        )


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        0.25,
        0.50,
        0.75,
        1.0,
    ],
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


def test_validate_scores():
    assert _validate_scores(
        [1, 0.8, 0.5]
    ) == [
        1.0,
        0.8,
        0.5,
    ]


def test_validate_scores_empty():
    with pytest.raises(
        RobustnessValidationError
    ):
        _validate_scores([])


@pytest.mark.parametrize(
    "value",
    [
        True,
        "bad",
        float("nan"),
        float("inf"),
    ],
)
def test_validate_scores_invalid(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _validate_scores(
            [1.0, value]
        )


# ============================================================
# RobustnessScenario
# ============================================================


def test_scenario_creation():
    scenario = make_scenario()

    assert scenario.scenario_id == "stress_1"
    assert scenario.category == "transaction_cost"
    assert scenario.severity == 0.50
    assert scenario.cost_bps == 10.0


def test_scenario_frozen():
    scenario = make_scenario()

    with pytest.raises(FrozenInstanceError):
        scenario.severity = 0.9


def test_scenario_fingerprint():
    first = make_scenario()
    second = make_scenario()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_scenario_fingerprint_changes():
    first = make_scenario(
        severity=0.25
    )
    second = make_scenario(
        severity=0.75
    )

    assert first.fingerprint != second.fingerprint


@pytest.mark.parametrize(
    "scenario_id",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_scenario_id(scenario_id):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            scenario_id=scenario_id
        )


@pytest.mark.parametrize(
    "category",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_invalid_scenario_category(category):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            category=category
        )


@pytest.mark.parametrize(
    "severity",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
    ],
)
def test_invalid_scenario_severity(severity):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            severity=severity
        )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_cost_bps(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            cost_bps=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        True,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_slippage_bps(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            slippage_bps=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        True,
        1.5,
    ],
)
def test_invalid_execution_delay(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(
            execution_delay=value
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("parameter_shift", -0.01),
        ("parameter_shift", True),
        ("data_perturbation", -0.01),
        ("data_perturbation", float("nan")),
    ],
)
def test_invalid_perturbation(
    field,
    value,
):
    kwargs = {
        "parameter_shift": 0.0,
        "data_perturbation": 0.0,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        make_scenario(**kwargs)


# ============================================================
# RobustnessScenarioResult
# ============================================================


def test_scenario_result_creation():
    result = make_result()

    assert result.baseline_score == 1.0
    assert result.stressed_score == 0.80
    assert result.degradation == 0.20
    assert result.degradation_ratio == 0.20
    assert result.positive_after_stress is True
    assert result.stable is True


def test_scenario_result_frozen():
    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.stable = False


def test_scenario_result_fingerprint():
    first = make_result()
    second = make_result()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_invalid_result_scenario():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessScenarioResult(
            scenario="bad",
            baseline_score=1.0,
            stressed_score=0.8,
            degradation=0.2,
            degradation_ratio=0.2,
            positive_after_stress=True,
            stable=True,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("baseline_score", True),
        ("baseline_score", float("nan")),
        ("stressed_score", float("inf")),
        ("degradation", float("nan")),
        ("degradation_ratio", float("inf")),
    ],
)
def test_invalid_result_numeric(
    field,
    value,
):
    kwargs = {
        "baseline_score": 1.0,
        "stressed_score": 0.8,
        "degradation": 0.2,
        "degradation_ratio": 0.2,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessScenarioResult(
            scenario=make_scenario(),
            positive_after_stress=True,
            stable=True,
            **kwargs,
        )


def test_invalid_positive_after_stress():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessScenarioResult(
            scenario=make_scenario(),
            baseline_score=1.0,
            stressed_score=0.8,
            degradation=0.2,
            degradation_ratio=0.2,
            positive_after_stress=1,
            stable=True,
        )


def test_invalid_stable():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessScenarioResult(
            scenario=make_scenario(),
            baseline_score=1.0,
            stressed_score=0.8,
            degradation=0.2,
            degradation_ratio=0.2,
            positive_after_stress=True,
            stable=1,
        )


# ============================================================
# RobustnessReport
# ============================================================


def test_report_creation():
    report = make_report()

    assert report.scenario_count == 2
    assert report.stable_count == 2
    assert report.positive_count == 2
    assert report.failed_scenarios == []
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


def test_report_failed_scenarios():
    results = [
        make_result(
            scenario_id="good",
            stable=True,
        ),
        make_result(
            scenario_id="bad",
            stressed_score=0.2,
            degradation=0.8,
            degradation_ratio=0.8,
            stable=False,
        ),
    ]

    report = RobustnessReport(
        baseline_score=1.0,
        scenario_results=results,
        average_stressed_score=0.5,
        average_degradation=0.5,
        average_degradation_ratio=0.5,
        positive_survival_ratio=1.0,
        stable_scenario_ratio=0.5,
        worst_stressed_score=0.2,
        worst_degradation_ratio=0.8,
        passed=False,
    )

    assert report.failed_scenarios == [
        "bad"
    ]


def test_report_empty_results():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessReport(
            baseline_score=1.0,
            scenario_results=[],
            average_stressed_score=0.0,
            average_degradation=0.0,
            average_degradation_ratio=0.0,
            positive_survival_ratio=0.0,
            stable_scenario_ratio=0.0,
            worst_stressed_score=0.0,
            worst_degradation_ratio=0.0,
            passed=False,
        )


def test_report_invalid_result_type():
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessReport(
            baseline_score=1.0,
            scenario_results=["bad"],
            average_stressed_score=0.0,
            average_degradation=0.0,
            average_degradation_ratio=0.0,
            positive_survival_ratio=0.0,
            stable_scenario_ratio=0.0,
            worst_stressed_score=0.0,
            worst_degradation_ratio=0.0,
            passed=False,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("positive_survival_ratio", -0.01),
        ("positive_survival_ratio", 1.01),
        ("stable_scenario_ratio", -0.01),
        ("stable_scenario_ratio", 1.01),
    ],
)
def test_report_invalid_ratio(
    field,
    value,
):
    kwargs = {
        "positive_survival_ratio": 1.0,
        "stable_scenario_ratio": 1.0,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessReport(
            baseline_score=1.0,
            scenario_results=[
                make_result()
            ],
            average_stressed_score=0.8,
            average_degradation=0.2,
            average_degradation_ratio=0.2,
            worst_stressed_score=0.8,
            worst_degradation_ratio=0.2,
            passed=True,
            **kwargs,
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = RobustnessEngine()

    assert (
        engine.maximum_degradation_ratio
        == 0.50
    )

    assert (
        engine.minimum_positive_survival_ratio
        == 0.70
    )

    assert (
        engine.minimum_stable_scenario_ratio
        == 0.70
    )


def test_custom_engine():
    engine = RobustnessEngine(
        make_config(),
        maximum_degradation_ratio=0.30,
        minimum_positive_survival_ratio=0.80,
        minimum_stable_scenario_ratio=0.90,
    )

    assert (
        engine.maximum_degradation_ratio
        == 0.30
    )

    assert (
        engine.minimum_positive_survival_ratio
        == 0.80
    )

    assert (
        engine.minimum_stable_scenario_ratio
        == 0.90
    )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        True,
        float("nan"),
    ],
)
def test_invalid_maximum_degradation(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessEngine(
            maximum_degradation_ratio=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
    ],
)
def test_invalid_minimum_survival(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessEngine(
            minimum_positive_survival_ratio=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.01,
        1.01,
        True,
        float("nan"),
    ],
)
def test_invalid_minimum_stability(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        RobustnessEngine(
            minimum_stable_scenario_ratio=value
        )


# ============================================================
# Default scenarios
# ============================================================


def test_default_scenarios():
    engine = RobustnessEngine()

    scenarios = (
        engine.build_default_scenarios()
    )

    assert len(scenarios) == 10

    assert all(
        isinstance(
            scenario,
            RobustnessScenario,
        )
        for scenario in scenarios
    )


def test_default_scenario_ids_unique():
    engine = RobustnessEngine()

    scenarios = (
        engine.build_default_scenarios()
    )

    ids = [
        scenario.scenario_id
        for scenario in scenarios
    ]

    assert len(ids) == len(set(ids))


def test_default_scenario_categories():
    engine = RobustnessEngine()

    categories = {
        scenario.category
        for scenario
        in engine.build_default_scenarios()
    }

    assert categories == {
        "transaction_cost",
        "slippage",
        "execution_delay",
        "parameter_shift",
        "data_perturbation",
    }


# ============================================================
# Scenario result creation
# ============================================================


def test_create_scenario_result():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=1.0,
        stressed_score=0.75,
    )

    assert result.degradation == pytest.approx(
        0.25
    )

    assert result.degradation_ratio == pytest.approx(
        0.25
    )

    assert result.positive_after_stress is True
    assert result.stable is True


def test_create_scenario_result_unstable():
    engine = RobustnessEngine(
        maximum_degradation_ratio=0.30
    )

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=1.0,
        stressed_score=0.40,
    )

    assert result.degradation_ratio == pytest.approx(
        0.60
    )

    assert result.stable is False


def test_create_scenario_result_negative():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=1.0,
        stressed_score=-0.10,
    )

    assert result.positive_after_stress is False
    assert result.stable is False


def test_create_scenario_result_improvement():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=1.0,
        stressed_score=1.20,
    )

    assert result.degradation == pytest.approx(
        -0.20
    )

    assert result.degradation_ratio == pytest.approx(
        -0.20
    )

    assert result.stable is True


def test_create_scenario_result_zero_baseline_same():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=0.0,
        stressed_score=0.0,
    )

    assert result.degradation_ratio == 0.0
    assert result.positive_after_stress is False
    assert result.stable is False


def test_create_scenario_result_zero_baseline_negative():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=0.0,
        stressed_score=-0.1,
    )

    assert result.degradation_ratio == 1.0
    assert result.stable is False


def test_create_scenario_result_zero_baseline_positive():
    engine = RobustnessEngine()

    result = engine.create_scenario_result(
        make_scenario(),
        baseline_score=0.0,
        stressed_score=0.1,
    )

    assert result.degradation_ratio == -1.0
    assert result.stable is True


def test_create_result_invalid_scenario():
    engine = RobustnessEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.create_scenario_result(
            "bad",
            baseline_score=1.0,
            stressed_score=0.8,
        )


# ============================================================
# Aggregate
# ============================================================


def test_aggregate_pass():
    engine = RobustnessEngine()

    results = [
        engine.create_scenario_result(
            make_scenario(
                scenario_id="a"
            ),
            baseline_score=1.0,
            stressed_score=0.90,
        ),
        engine.create_scenario_result(
            make_scenario(
                scenario_id="b"
            ),
            baseline_score=1.0,
            stressed_score=0.80,
        ),
        engine.create_scenario_result(
            make_scenario(
                scenario_id="c"
            ),
            baseline_score=1.0,
            stressed_score=0.70,
        ),
    ]

    report = engine.aggregate(
        results
    )

    assert report.scenario_count == 3
    assert report.positive_count == 3
    assert report.stable_count == 3
    assert report.passed is True

    assert report.average_stressed_score == pytest.approx(
        0.80
    )

    assert report.average_degradation_ratio == pytest.approx(
        0.20
    )


def test_aggregate_fail():
    engine = RobustnessEngine()

    results = [
        engine.create_scenario_result(
            make_scenario(
                scenario_id="a"
            ),
            baseline_score=1.0,
            stressed_score=0.90,
        ),
        engine.create_scenario_result(
            make_scenario(
                scenario_id="b"
            ),
            baseline_score=1.0,
            stressed_score=-0.20,
        ),
        engine.create_scenario_result(
            make_scenario(
                scenario_id="c"
            ),
            baseline_score=1.0,
            stressed_score=-0.30,
        ),
    ]

    report = engine.aggregate(
        results
    )

    assert report.passed is False
    assert report.positive_survival_ratio < 0.70
    assert report.stable_scenario_ratio < 0.70


def test_aggregate_empty():
    engine = RobustnessEngine()

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.aggregate([])


def test_aggregate_invalid_type():
    engine = RobustnessEngine()

    with pytest.raises(
        ValidationConfigurationError
    ):
        engine.aggregate(
            [
                make_result(),
                "bad",
            ]
        )


def test_aggregate_different_baselines():
    engine = RobustnessEngine()

    results = [
        engine.create_scenario_result(
            make_scenario(
                scenario_id="a"
            ),
            baseline_score=1.0,
            stressed_score=0.8,
        ),
        engine.create_scenario_result(
            make_scenario(
                scenario_id="b"
            ),
            baseline_score=2.0,
            stressed_score=1.6,
        ),
    ]

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.aggregate(
            results
        )


# ============================================================
# Validate scores
# ============================================================


def test_validate_scores_pass():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.95,
            0.90,
            0.85,
            0.80,
            0.75,
        ],
    )

    assert report.passed is True
    assert report.scenario_count == 5


def test_validate_scores_fail():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            -0.1,
            -0.2,
            -0.3,
        ],
    )

    assert report.passed is False


def test_validate_scores_custom_scenarios():
    engine = RobustnessEngine()

    scenarios = [
        make_scenario(
            scenario_id="custom_1"
        ),
        make_scenario(
            scenario_id="custom_2"
        ),
    ]

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.9,
            0.8,
        ],
        scenarios=scenarios,
    )

    assert report.scenario_count == 2

    assert (
        report.scenario_results[
            0
        ].scenario.scenario_id
        == "custom_1"
    )


def test_validate_scores_length_mismatch():
    engine = RobustnessEngine()

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.validate_scores(
            baseline_score=1.0,
            stressed_scores=[
                0.9,
                0.8,
            ],
            scenarios=[
                make_scenario()
            ],
        )


def test_validate_scores_empty_scores():
    engine = RobustnessEngine()

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.validate_scores(
            baseline_score=1.0,
            stressed_scores=[],
        )


def test_validate_scores_empty_scenarios():
    engine = RobustnessEngine()

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.validate_scores(
            baseline_score=1.0,
            stressed_scores=[
                0.9
            ],
            scenarios=[],
        )


def test_validate_scores_more_than_default_scenarios():
    engine = RobustnessEngine()

    scores = [
        0.9
        for _ in range(15)
    ]

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=scores,
    )

    assert report.scenario_count == 15

    assert all(
        result.scenario.category
        == "generic"
        for result
        in report.scenario_results
    )


# ============================================================
# Mapping API
# ============================================================


def test_validate_mapping():
    engine = RobustnessEngine()

    report = engine.validate_mapping(
        baseline_score=1.0,
        stressed_scores={
            "cost": 0.90,
            "slippage": 0.85,
            "delay": 0.80,
        },
    )

    assert report.scenario_count == 3

    assert [
        result.scenario.scenario_id
        for result
        in report.scenario_results
    ] == [
        "cost",
        "slippage",
        "delay",
    ]


def test_validate_mapping_empty():
    engine = RobustnessEngine()

    with pytest.raises(
        RobustnessValidationError
    ):
        engine.validate_mapping(
            baseline_score=1.0,
            stressed_scores={},
        )


# ============================================================
# Category summary
# ============================================================


def test_category_summary():
    engine = RobustnessEngine()

    scenarios = [
        make_scenario(
            scenario_id="cost_1",
            category="cost",
        ),
        make_scenario(
            scenario_id="cost_2",
            category="cost",
        ),
        make_scenario(
            scenario_id="delay_1",
            category="delay",
        ),
    ]

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.9,
            0.8,
            0.7,
        ],
        scenarios=scenarios,
    )

    summary = engine.category_summary(
        report
    )

    assert set(summary) == {
        "cost",
        "delay",
    }

    assert (
        summary["cost"]["scenario_count"]
        == 2.0
    )

    assert (
        summary["delay"]["scenario_count"]
        == 1.0
    )

    assert (
        summary["cost"][
            "average_stressed_score"
        ]
        == pytest.approx(0.85)
    )


# ============================================================
# Score
# ============================================================


def test_score_bounded():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.9,
            0.8,
            0.7,
        ],
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_strong_robustness_high_score():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.95,
            0.95,
            0.95,
            0.95,
        ],
    )

    assert (
        engine._calculate_score(
            report
        )
        > 0.90
    )


def test_weak_robustness_low_score():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            -0.2,
            -0.3,
            -0.4,
        ],
    )

    assert (
        engine._calculate_score(
            report
        )
        < 0.30
    )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.95,
            0.90,
            0.85,
        ],
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.ROBUSTNESS
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

    assert (
        result.diagnostics[
            "scenario_count"
        ]
        == 3
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_to_validation_result_fail():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            -0.1,
            -0.2,
            -0.3,
        ],
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )


# ============================================================
# Determinism
# ============================================================


def test_validate_scores_deterministic():
    engine = RobustnessEngine()

    values = [
        0.9,
        0.8,
        0.7,
    ]

    first = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=values,
    )

    second = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=values,
    )

    assert (
        first.fingerprint
        == second.fingerprint
    )


def test_category_summary_deterministic():
    engine = RobustnessEngine()

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=[
            0.9,
            0.8,
        ],
    )

    assert (
        engine.category_summary(report)
        == engine.category_summary(report)
    )


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = RobustnessEngine(
        make_config(),
        maximum_degradation_ratio=0.40,
        minimum_positive_survival_ratio=0.80,
        minimum_stable_scenario_ratio=0.75,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert (
        snapshot[
            "maximum_degradation_ratio"
        ]
        == 0.40
    )

    assert (
        snapshot[
            "minimum_positive_survival_ratio"
        ]
        == 0.80
    )

    assert (
        snapshot[
            "minimum_stable_scenario_ratio"
        ]
        == 0.75
    )

    assert (
        snapshot[
            "default_scenario_count"
        ]
        == 10
    )

    assert (
        "transaction_cost"
        in snapshot["stress_dimensions"]
    )

    assert (
        "slippage"
        in snapshot["stress_dimensions"]
    )

    assert (
        "execution_delay"
        in snapshot["stress_dimensions"]
    )

    assert (
        "parameter_shift"
        in snapshot["stress_dimensions"]
    )

    assert (
        "data_perturbation"
        in snapshot["stress_dimensions"]
    )

    assert snapshot["research_only"] is True

    assert (
        snapshot[
            "real_money_execution"
        ]
        is False
    )

    assert len(
        snapshot[
            "config_fingerprint"
        ]
    ) == 64


# ============================================================
# Complete robustness flow
# ============================================================


def test_complete_robustness_flow():
    config = ValidationConfig(
        minimum_observations=30,
        research_only=True,
        real_money_execution=False,
    )

    engine = RobustnessEngine(
        config,
        maximum_degradation_ratio=0.50,
        minimum_positive_survival_ratio=0.70,
        minimum_stable_scenario_ratio=0.70,
    )

    scenarios = (
        engine.build_default_scenarios()
    )

    stressed_scores = [
        0.98,
        0.90,
        0.97,
        0.88,
        0.94,
        0.82,
        0.96,
        0.80,
        0.95,
        0.78,
    ]

    report = engine.validate_scores(
        baseline_score=1.0,
        stressed_scores=stressed_scores,
        scenarios=scenarios,
    )

    assert report.scenario_count == 10
    assert report.positive_count == 10
    assert report.stable_count == 10
    assert report.positive_survival_ratio == 1.0
    assert report.stable_scenario_ratio == 1.0
    assert report.average_stressed_score > 0.0
    assert report.passed is True

    category_summary = (
        engine.category_summary(
            report
        )
    )

    assert len(category_summary) == 5

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.ROBUSTNESS
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert result.score > 0.80

    assert (
        result.metadata[
            "research_only"
        ]
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