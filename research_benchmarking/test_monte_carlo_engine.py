from __future__ import annotations

import random
from dataclasses import FrozenInstanceError
from statistics import mean

import pytest

from research_benchmarking.models import (
    ConfidenceInterval,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)
from research_benchmarking.monte_carlo_engine import (
    VERSION,
    MonteCarloDistribution,
    MonteCarloEngine,
    MonteCarloReport,
    MonteCarloValidationError,
    _percentile,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    monte_carlo_simulations: int = 200,
    minimum_observations: int = 30,
    confidence_level: float = 0.95,
    significance_threshold: float = 0.05,
    random_seed: int = 42,
) -> ValidationConfig:
    return ValidationConfig(
        monte_carlo_simulations=monte_carlo_simulations,
        minimum_observations=minimum_observations,
        confidence_level=confidence_level,
        significance_threshold=significance_threshold,
        random_seed=random_seed,
    )


def positive_values(
    count: int = 100,
) -> list[float]:
    return [
        0.02 + (index % 10) * 0.001
        for index in range(count)
    ]


def mixed_values(
    count: int = 100,
) -> list[float]:
    return [
        0.02 if index % 2 == 0 else -0.02
        for index in range(count)
    ]


def make_interval(
    *,
    lower: float = -0.01,
    upper: float = 0.01,
    estimate: float = 0.0,
    method: str = "monte_carlo_sign_flip",
) -> ConfidenceInterval:
    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        confidence_level=0.95,
        estimate=estimate,
        method=method,
    )


def make_distribution(
    *,
    observed_estimate: float = 0.03,
    simulated_estimates: list[float] | None = None,
    null_mean: float = 0.0,
    null_standard_deviation: float = 0.01,
    empirical_p_value: float = 0.01,
    probability_exceeding_observed: float = 0.01,
    method: str = "sign_flip",
) -> MonteCarloDistribution:
    estimates = (
        simulated_estimates
        if simulated_estimates is not None
        else [-0.01, 0.0, 0.01]
    )

    return MonteCarloDistribution(
        observed_estimate=observed_estimate,
        simulated_estimates=estimates,
        null_mean=null_mean,
        null_standard_deviation=null_standard_deviation,
        empirical_p_value=empirical_p_value,
        probability_exceeding_observed=(
            probability_exceeding_observed
        ),
        confidence_interval=make_interval(
            method=f"monte_carlo_{method}",
        ),
        method=method,
        sample_size=100,
        simulations=len(estimates),
        random_seed=42,
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


# ============================================================
# Percentile
# ============================================================


def test_percentile_minimum():
    assert _percentile(
        [1.0, 2.0, 3.0],
        0.0,
    ) == 1.0


def test_percentile_maximum():
    assert _percentile(
        [1.0, 2.0, 3.0],
        1.0,
    ) == 3.0


def test_percentile_median():
    assert _percentile(
        [1.0, 2.0, 3.0],
        0.5,
    ) == 2.0


def test_percentile_interpolation():
    assert _percentile(
        [0.0, 10.0],
        0.25,
    ) == pytest.approx(2.5)


def test_percentile_single_value():
    assert _percentile(
        [7.0],
        0.95,
    ) == 7.0


def test_percentile_empty_rejected():
    with pytest.raises(
        MonteCarloValidationError
    ):
        _percentile([], 0.5)


@pytest.mark.parametrize(
    "probability",
    [
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_invalid_percentile_probability(
    probability,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        _percentile(
            [1.0, 2.0],
            probability,
        )


# ============================================================
# MonteCarloDistribution
# ============================================================


def test_distribution_creation():
    distribution = make_distribution()

    assert distribution.observed_estimate == 0.03
    assert distribution.null_mean == 0.0
    assert distribution.empirical_p_value == 0.01
    assert distribution.effect_above_null == 0.03
    assert distribution.z_score == pytest.approx(3.0)
    assert distribution.significant is True


def test_distribution_is_frozen():
    distribution = make_distribution()

    with pytest.raises(FrozenInstanceError):
        distribution.null_mean = 1.0


def test_distribution_effect_above_null():
    distribution = make_distribution(
        observed_estimate=0.05,
        null_mean=0.02,
    )

    assert (
        distribution.effect_above_null
        == pytest.approx(0.03)
    )


def test_distribution_z_score():
    distribution = make_distribution(
        observed_estimate=0.04,
        null_mean=0.01,
        null_standard_deviation=0.01,
    )

    assert distribution.z_score == pytest.approx(3.0)


def test_distribution_zero_std_positive_effect():
    distribution = make_distribution(
        observed_estimate=0.02,
        null_mean=0.0,
        null_standard_deviation=0.0,
    )

    assert distribution.z_score == 1e12


def test_distribution_zero_std_negative_effect():
    distribution = make_distribution(
        observed_estimate=-0.02,
        null_mean=0.0,
        null_standard_deviation=0.0,
    )

    assert distribution.z_score == -1e12


def test_distribution_zero_std_zero_effect():
    distribution = make_distribution(
        observed_estimate=0.0,
        null_mean=0.0,
        null_standard_deviation=0.0,
    )

    assert distribution.z_score == 0.0


def test_distribution_fingerprint_stable():
    first = make_distribution()
    second = make_distribution()

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_estimate", True),
        ("observed_estimate", float("nan")),
        ("null_mean", float("inf")),
        (
            "null_standard_deviation",
            float("nan"),
        ),
        ("empirical_p_value", float("nan")),
        (
            "probability_exceeding_observed",
            float("inf"),
        ),
    ],
)
def test_invalid_distribution_numeric_fields(
    field,
    value,
):
    kwargs = {
        "observed_estimate": 0.03,
        "simulated_estimates": [
            -0.01,
            0.0,
            0.01,
        ],
        "null_mean": 0.0,
        "null_standard_deviation": 0.01,
        "empirical_p_value": 0.01,
        "probability_exceeding_observed": 0.01,
        "confidence_interval": make_interval(),
        "method": "sign_flip",
        "sample_size": 100,
        "simulations": 3,
        "random_seed": 42,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(**kwargs)


def test_negative_null_std_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        make_distribution(
            null_standard_deviation=-0.01
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("empirical_p_value", -0.01),
        ("empirical_p_value", 1.01),
        (
            "probability_exceeding_observed",
            -0.01,
        ),
        (
            "probability_exceeding_observed",
            1.01,
        ),
    ],
)
def test_invalid_probability_fields(
    field,
    value,
):
    kwargs = {
        "observed_estimate": 0.03,
        "simulated_estimates": [
            -0.01,
            0.0,
            0.01,
        ],
        "null_mean": 0.0,
        "null_standard_deviation": 0.01,
        "empirical_p_value": 0.01,
        "probability_exceeding_observed": 0.01,
        "confidence_interval": make_interval(),
        "method": "sign_flip",
        "sample_size": 100,
        "simulations": 3,
        "random_seed": 42,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(**kwargs)


def test_empty_simulated_estimates_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(
            observed_estimate=0.03,
            simulated_estimates=[],
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
            probability_exceeding_observed=0.01,
            confidence_interval=make_interval(),
            method="sign_flip",
            sample_size=100,
            simulations=1,
            random_seed=42,
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
def test_invalid_simulated_estimate(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(
            observed_estimate=0.03,
            simulated_estimates=[
                0.0,
                value,
            ],
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
            probability_exceeding_observed=0.01,
            confidence_interval=make_interval(),
            method="sign_flip",
            sample_size=100,
            simulations=2,
            random_seed=42,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_size", 0),
        ("sample_size", -1),
        ("sample_size", True),
        ("simulations", 0),
        ("simulations", 1.5),
    ],
)
def test_invalid_distribution_counts(
    field,
    value,
):
    kwargs = {
        "observed_estimate": 0.03,
        "simulated_estimates": [0.0],
        "null_mean": 0.0,
        "null_standard_deviation": 0.01,
        "empirical_p_value": 0.01,
        "probability_exceeding_observed": 0.01,
        "confidence_interval": make_interval(),
        "method": "sign_flip",
        "sample_size": 100,
        "simulations": 1,
        "random_seed": 42,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(**kwargs)


@pytest.mark.parametrize(
    "random_seed",
    [
        True,
        1.5,
        "42",
    ],
)
def test_invalid_random_seed(random_seed):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(
            observed_estimate=0.03,
            simulated_estimates=[0.0],
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
            probability_exceeding_observed=0.01,
            confidence_interval=make_interval(),
            method="sign_flip",
            sample_size=100,
            simulations=1,
            random_seed=random_seed,
        )


def test_empty_method_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloDistribution(
            observed_estimate=0.03,
            simulated_estimates=[0.0],
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
            probability_exceeding_observed=0.01,
            confidence_interval=make_interval(),
            method=" ",
            sample_size=100,
            simulations=1,
            random_seed=42,
        )


# ============================================================
# MonteCarloReport
# ============================================================


def test_report_creation():
    report = MonteCarloReport(
        distribution=make_distribution(),
        significance_threshold=0.05,
        minimum_z_score=1.0,
        statistically_significant=True,
        effect_direction_positive=True,
        passed=True,
    )

    assert report.passed is True


def test_report_fingerprint_stable():
    first = MonteCarloReport(
        distribution=make_distribution(),
        significance_threshold=0.05,
        minimum_z_score=1.0,
        statistically_significant=True,
        effect_direction_positive=True,
        passed=True,
    )

    second = MonteCarloReport(
        distribution=make_distribution(),
        significance_threshold=0.05,
        minimum_z_score=1.0,
        statistically_significant=True,
        effect_direction_positive=True,
        passed=True,
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        1.0,
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_invalid_report_threshold(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloReport(
            distribution=make_distribution(),
            significance_threshold=value,
            minimum_z_score=1.0,
            statistically_significant=False,
            effect_direction_positive=False,
            passed=False,
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
def test_invalid_report_minimum_z(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloReport(
            distribution=make_distribution(),
            significance_threshold=0.05,
            minimum_z_score=value,
            statistically_significant=False,
            effect_direction_positive=False,
            passed=False,
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = MonteCarloEngine()

    assert engine.method == "sign_flip"
    assert engine.minimum_z_score == 1.0


@pytest.mark.parametrize(
    "method",
    [
        "sign_flip",
        "permutation",
        "centered_bootstrap",
    ],
)
def test_supported_methods(method):
    engine = MonteCarloEngine(
        method=method
    )

    assert engine.method == method


def test_invalid_method():
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloEngine(
            method="invalid"
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
def test_invalid_engine_minimum_z(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        MonteCarloEngine(
            minimum_z_score=value
        )


# ============================================================
# Sign-flip sample
# ============================================================


def test_sign_flip_sample_size():
    values = list(range(1, 51))

    result = MonteCarloEngine.sign_flip_sample(
        values,
        random.Random(42),
    )

    assert len(result) == len(values)


def test_sign_flip_preserves_absolute_values():
    values = [
        1.0,
        2.0,
        3.0,
        4.0,
    ]

    result = MonteCarloEngine.sign_flip_sample(
        values,
        random.Random(42),
    )

    assert [
        abs(value)
        for value in result
    ] == values


def test_sign_flip_deterministic():
    values = list(range(1, 50))

    first = MonteCarloEngine.sign_flip_sample(
        values,
        random.Random(42),
    )

    second = MonteCarloEngine.sign_flip_sample(
        values,
        random.Random(42),
    )

    assert first == second


def test_sign_flip_empty_rejected():
    with pytest.raises(
        MonteCarloValidationError
    ):
        MonteCarloEngine.sign_flip_sample(
            [],
            random.Random(42),
        )


# ============================================================
# Permutation sample
# ============================================================


def test_permutation_size():
    values = list(range(50))

    result = MonteCarloEngine.permutation_sample(
        values,
        random.Random(42),
    )

    assert len(result) == len(values)


def test_permutation_preserves_values():
    values = list(range(50))

    result = MonteCarloEngine.permutation_sample(
        values,
        random.Random(42),
    )

    assert sorted(result) == sorted(values)


def test_permutation_deterministic():
    values = list(range(50))

    first = MonteCarloEngine.permutation_sample(
        values,
        random.Random(42),
    )

    second = MonteCarloEngine.permutation_sample(
        values,
        random.Random(42),
    )

    assert first == second


# ============================================================
# Centered bootstrap
# ============================================================


def test_centered_bootstrap_size():
    values = positive_values(50)

    result = (
        MonteCarloEngine.centered_bootstrap_sample(
            values,
            random.Random(42),
        )
    )

    assert len(result) == len(values)


def test_centered_bootstrap_deterministic():
    values = positive_values(50)

    first = (
        MonteCarloEngine.centered_bootstrap_sample(
            values,
            random.Random(42),
        )
    )

    second = (
        MonteCarloEngine.centered_bootstrap_sample(
            values,
            random.Random(42),
        )
    )

    assert first == second


def test_centered_values_are_mean_adjusted():
    values = [
        1.0,
        2.0,
        3.0,
    ]

    sample = (
        MonteCarloEngine.centered_bootstrap_sample(
            values,
            random.Random(42),
        )
    )

    allowed = {
        -1.0,
        0.0,
        1.0,
    }

    assert all(
        value in allowed
        for value in sample
    )


# ============================================================
# Run
# ============================================================


def test_run_sign_flip():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=100
        ),
        method="sign_flip",
    )

    distribution = engine.run(
        positive_values()
    )

    assert distribution.method == "sign_flip"
    assert distribution.simulations == 100
    assert len(
        distribution.simulated_estimates
    ) == 100
    assert distribution.sample_size == 100


def test_run_centered_bootstrap():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=100
        ),
        method="centered_bootstrap",
    )

    distribution = engine.run(
        positive_values()
    )

    assert (
        distribution.method
        == "centered_bootstrap"
    )

    assert len(
        distribution.simulated_estimates
    ) == 100


def test_run_observed_mean():
    values = positive_values()

    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=100
        )
    )

    distribution = engine.run(values)

    assert (
        distribution.observed_estimate
        == pytest.approx(mean(values))
    )


def test_run_deterministic():
    config = make_config(
        monte_carlo_simulations=100,
        random_seed=123,
    )

    first = MonteCarloEngine(
        config
    ).run(
        positive_values()
    )

    second = MonteCarloEngine(
        config
    ).run(
        positive_values()
    )

    assert (
        first.simulated_estimates
        == second.simulated_estimates
    )

    assert (
        first.empirical_p_value
        == second.empirical_p_value
    )

    assert first.fingerprint == second.fingerprint


def test_run_custom_statistic():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=50
        )
    )

    distribution = engine.run(
        positive_values(),
        statistic=max,
    )

    assert (
        distribution.observed_estimate
        == max(positive_values())
    )


def test_run_insufficient_observations():
    engine = MonteCarloEngine(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        MonteCarloValidationError
    ):
        engine.run(
            positive_values(99)
        )


def test_run_non_finite_statistic():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=20
        )
    )

    def bad_statistic(values):
        return float("nan")

    with pytest.raises(
        MonteCarloValidationError
    ):
        engine.run(
            positive_values(),
            statistic=bad_statistic,
        )


def test_empirical_p_value_bounded():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=100
        )
    )

    distribution = engine.run(
        positive_values()
    )

    assert (
        0.0
        < distribution.empirical_p_value
        <= 1.0
    )


def test_null_confidence_interval():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=100,
            confidence_level=0.90,
        )
    )

    distribution = engine.run(
        positive_values()
    )

    assert (
        distribution.confidence_interval
        .confidence_level
        == 0.90
    )

    assert (
        distribution.confidence_interval.method
        == "monte_carlo_sign_flip"
    )


# ============================================================
# Important permutation property
# ============================================================


def test_permutation_mean_is_invariant():
    """
    Permuting a sequence cannot change its arithmetic mean.

    This test documents the statistical limitation of using
    raw permutation with mean as the test statistic.
    """

    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=50
        ),
        method="permutation",
    )

    values = positive_values()

    distribution = engine.run(values)

    assert all(
        estimate
        == pytest.approx(
            distribution.observed_estimate
        )
        for estimate
        in distribution.simulated_estimates
    )


# ============================================================
# Evaluation
# ============================================================


def test_evaluate_pass():
    engine = MonteCarloEngine(
        make_config(
            significance_threshold=0.05
        ),
        minimum_z_score=1.0,
    )

    report = engine.evaluate(
        make_distribution(
            observed_estimate=0.03,
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
        )
    )

    assert (
        report.statistically_significant
        is True
    )

    assert (
        report.effect_direction_positive
        is True
    )

    assert report.passed is True


def test_evaluate_not_significant():
    engine = MonteCarloEngine()

    report = engine.evaluate(
        make_distribution(
            empirical_p_value=0.20
        )
    )

    assert (
        report.statistically_significant
        is False
    )

    assert report.passed is False


def test_evaluate_negative_effect():
    engine = MonteCarloEngine()

    report = engine.evaluate(
        make_distribution(
            observed_estimate=-0.03,
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
        )
    )

    assert (
        report.effect_direction_positive
        is False
    )

    assert report.passed is False


def test_evaluate_insufficient_z_score():
    engine = MonteCarloEngine(
        minimum_z_score=3.0
    )

    report = engine.evaluate(
        make_distribution(
            observed_estimate=0.015,
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
        )
    )

    assert report.passed is False


# ============================================================
# Score
# ============================================================


def test_score_is_bounded():
    engine = MonteCarloEngine()

    report = engine.evaluate(
        make_distribution()
    )

    score = engine._calculate_score(
        report
    )

    assert 0.0 <= score <= 1.0


def test_strong_result_has_high_score():
    engine = MonteCarloEngine(
        minimum_z_score=1.0
    )

    report = engine.evaluate(
        make_distribution(
            observed_estimate=0.05,
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.001,
        )
    )

    assert (
        engine._calculate_score(report)
        > 0.80
    )


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = MonteCarloEngine()

    report = engine.evaluate(
        make_distribution(
            observed_estimate=0.03,
            null_mean=0.0,
            null_standard_deviation=0.01,
            empirical_p_value=0.01,
        )
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.MONTE_CARLO
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
    engine = MonteCarloEngine()

    report = engine.evaluate(
        make_distribution(
            empirical_p_value=0.50
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
# Convenience API
# ============================================================


def test_validate_positive_data():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=500,
            random_seed=42,
        ),
        method="sign_flip",
        minimum_z_score=1.0,
    )

    report = engine.validate(
        positive_values(150)
    )

    assert (
        report.distribution
        .observed_estimate
        > 0.0
    )

    assert (
        report.distribution
        .empirical_p_value
        <= 0.05
    )

    assert report.passed is True


def test_validate_centered_bootstrap():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=500,
            random_seed=42,
        ),
        method="centered_bootstrap",
        minimum_z_score=1.0,
    )

    report = engine.validate(
        positive_values(150)
    )

    assert (
        report.distribution.method
        == "centered_bootstrap"
    )

    assert report.passed is True


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = MonteCarloEngine(
        make_config(
            monte_carlo_simulations=350,
            confidence_level=0.90,
            significance_threshold=0.01,
            random_seed=123,
        ),
        method="centered_bootstrap",
        minimum_z_score=2.0,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert (
        snapshot["method"]
        == "centered_bootstrap"
    )

    assert (
        snapshot[
            "monte_carlo_simulations"
        ]
        == 350
    )

    assert (
        snapshot["confidence_level"]
        == 0.90
    )

    assert (
        snapshot[
            "significance_threshold"
        ]
        == 0.01
    )

    assert (
        snapshot["minimum_z_score"]
        == 2.0
    )

    assert snapshot["random_seed"] == 123

    assert snapshot["research_only"] is True

    assert (
        snapshot["real_money_execution"]
        is False
    )

    assert len(
        snapshot["config_fingerprint"]
    ) == 64


# ============================================================
# Complete Monte Carlo flow
# ============================================================


def test_complete_monte_carlo_flow():
    config = ValidationConfig(
        monte_carlo_simulations=600,
        confidence_level=0.95,
        significance_threshold=0.05,
        minimum_observations=50,
        random_seed=42,
    )

    engine = MonteCarloEngine(
        config,
        method="sign_flip",
        minimum_z_score=1.0,
    )

    values = [
        0.015
        + (index % 10) * 0.0005
        for index in range(250)
    ]

    distribution = engine.run(
        values
    )

    assert (
        distribution.simulations
        == 600
    )

    assert distribution.sample_size == 250

    assert (
        distribution.observed_estimate
        > 0.0
    )

    assert (
        distribution.empirical_p_value
        <= 0.05
    )

    assert (
        distribution.effect_above_null
        > 0.0
    )

    assert distribution.z_score > 1.0

    report = engine.evaluate(
        distribution
    )

    assert (
        report.statistically_significant
        is True
    )

    assert (
        report.effect_direction_positive
        is True
    )

    assert report.passed is True

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.MONTE_CARLO
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
        distribution.fingerprint
    ) == 64

    assert len(report.fingerprint) == 64