from __future__ import annotations

from dataclasses import FrozenInstanceError
from statistics import mean

import pytest

from research_benchmarking.bootstrap_engine import (
    VERSION,
    BootstrapDistribution,
    BootstrapEngine,
    BootstrapReport,
    BootstrapValidationError,
    _percentile,
)
from research_benchmarking.models import (
    ConfidenceInterval,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationStage,
    ValidationStatus,
)


# ============================================================
# Helpers
# ============================================================


def make_config(
    *,
    bootstrap_samples: int = 200,
    minimum_observations: int = 30,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> ValidationConfig:
    return ValidationConfig(
        bootstrap_samples=bootstrap_samples,
        minimum_observations=minimum_observations,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )


def positive_values(
    count: int = 100,
) -> list[float]:
    return [
        0.01 + (index % 10) * 0.001
        for index in range(count)
    ]


def mixed_values(
    count: int = 100,
) -> list[float]:
    return [
        0.02 if index % 2 == 0 else -0.01
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
        method="iid_percentile",
    )


def make_distribution(
    *,
    observed_estimate: float = 0.02,
    estimates: list[float] | None = None,
    standard_error: float = 0.002,
    bias: float = 0.0,
    lower: float = 0.01,
    upper: float = 0.03,
) -> BootstrapDistribution:
    bootstrap_estimates = (
        estimates
        if estimates is not None
        else [0.015, 0.020, 0.025]
    )

    return BootstrapDistribution(
        observed_estimate=observed_estimate,
        bootstrap_estimates=bootstrap_estimates,
        standard_error=standard_error,
        bias=bias,
        confidence_interval=make_interval(
            lower=lower,
            upper=upper,
            estimate=observed_estimate,
        ),
        method="iid_percentile",
        sample_size=100,
        bootstrap_samples=len(
            bootstrap_estimates
        ),
    )


# ============================================================
# Version
# ============================================================


def test_version():
    assert VERSION == "v6.0.2"


# ============================================================
# Percentile helper
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
    result = _percentile(
        [0.0, 10.0],
        0.25,
    )

    assert result == pytest.approx(2.5)


def test_percentile_single_value():
    assert _percentile(
        [5.0],
        0.95,
    ) == 5.0


def test_percentile_empty_rejected():
    with pytest.raises(
        BootstrapValidationError
    ):
        _percentile([], 0.5)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1],
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
# BootstrapDistribution
# ============================================================


def test_distribution_creation():
    distribution = make_distribution()

    assert (
        distribution.observed_estimate
        == 0.02
    )

    assert distribution.sample_size == 100
    assert distribution.contains_zero is False
    assert distribution.positive_interval is True


def test_distribution_is_frozen():
    distribution = make_distribution()

    with pytest.raises(FrozenInstanceError):
        distribution.bias = 1.0


def test_distribution_contains_zero():
    distribution = make_distribution(
        lower=-0.01,
        upper=0.03,
    )

    assert distribution.contains_zero is True
    assert distribution.positive_interval is False


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
        ("standard_error", float("inf")),
        ("bias", float("-inf")),
    ],
)
def test_invalid_distribution_numeric_fields(
    field,
    value,
):
    kwargs = {
        "observed_estimate": 0.02,
        "bootstrap_estimates": [
            0.01,
            0.02,
        ],
        "standard_error": 0.001,
        "bias": 0.0,
        "confidence_interval": (
            make_interval()
        ),
        "method": "iid_percentile",
        "sample_size": 100,
        "bootstrap_samples": 2,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(**kwargs)


def test_empty_bootstrap_estimates_rejected():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(
            observed_estimate=0.02,
            bootstrap_estimates=[],
            standard_error=0.001,
            bias=0.0,
            confidence_interval=(
                make_interval()
            ),
            method="iid_percentile",
            sample_size=100,
            bootstrap_samples=1,
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
def test_invalid_bootstrap_estimate(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(
            observed_estimate=0.02,
            bootstrap_estimates=[
                0.01,
                value,
            ],
            standard_error=0.001,
            bias=0.0,
            confidence_interval=(
                make_interval()
            ),
            method="iid_percentile",
            sample_size=100,
            bootstrap_samples=2,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("sample_size", 0),
        ("sample_size", -1),
        ("sample_size", True),
        ("bootstrap_samples", 0),
        ("bootstrap_samples", 1.5),
    ],
)
def test_invalid_distribution_counts(
    field,
    value,
):
    kwargs = {
        "observed_estimate": 0.02,
        "bootstrap_estimates": [
            0.01,
            0.02,
        ],
        "standard_error": 0.001,
        "bias": 0.0,
        "confidence_interval": (
            make_interval()
        ),
        "method": "iid_percentile",
        "sample_size": 100,
        "bootstrap_samples": 2,
    }

    kwargs[field] = value

    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(**kwargs)


@pytest.mark.parametrize(
    "block_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_distribution_block_size(
    block_size,
):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(
            observed_estimate=0.02,
            bootstrap_estimates=[
                0.01,
                0.02,
            ],
            standard_error=0.001,
            bias=0.0,
            confidence_interval=(
                make_interval()
            ),
            method="iid_percentile",
            sample_size=100,
            bootstrap_samples=2,
            block_size=block_size,
        )


def test_empty_distribution_method():
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapDistribution(
            observed_estimate=0.02,
            bootstrap_estimates=[
                0.01,
                0.02,
            ],
            standard_error=0.001,
            bias=0.0,
            confidence_interval=(
                make_interval()
            ),
            method=" ",
            sample_size=100,
            bootstrap_samples=2,
        )


# ============================================================
# BootstrapReport
# ============================================================


def test_report_creation():
    report = BootstrapReport(
        distribution=make_distribution(),
        probability_positive=1.0,
        relative_standard_error=0.10,
        passed=True,
    )

    assert report.passed is True
    assert report.probability_positive == 1.0


def test_report_fingerprint_stable():
    first = BootstrapReport(
        distribution=make_distribution(),
        probability_positive=1.0,
        relative_standard_error=0.10,
        passed=True,
    )

    second = BootstrapReport(
        distribution=make_distribution(),
        probability_positive=1.0,
        relative_standard_error=0.10,
        passed=True,
    )

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_invalid_probability_positive(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapReport(
            distribution=make_distribution(),
            probability_positive=value,
            relative_standard_error=0.10,
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
def test_invalid_relative_standard_error(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapReport(
            distribution=make_distribution(),
            probability_positive=0.95,
            relative_standard_error=value,
            passed=False,
        )


# ============================================================
# Engine construction
# ============================================================


def test_default_engine():
    engine = BootstrapEngine()

    assert engine.block_size is None

    assert (
        engine.minimum_probability_positive
        == 0.95
    )

    assert (
        engine.maximum_relative_standard_error
        == 1.0
    )


def test_custom_engine():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        ),
        block_size=5,
        minimum_probability_positive=0.90,
        maximum_relative_standard_error=0.50,
    )

    assert engine.block_size == 5

    assert (
        engine.minimum_probability_positive
        == 0.90
    )

    assert (
        engine.maximum_relative_standard_error
        == 0.50
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_engine_block_size(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapEngine(
            block_size=value
        )


@pytest.mark.parametrize(
    "value",
    [
        -0.1,
        1.1,
        True,
        float("nan"),
    ],
)
def test_invalid_minimum_probability(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapEngine(
            minimum_probability_positive=value
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
def test_invalid_maximum_relative_error(value):
    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapEngine(
            maximum_relative_standard_error=value
        )


# ============================================================
# IID resampling
# ============================================================


def test_iid_resample_size():
    import random

    values = list(range(100))
    rng = random.Random(42)

    result = BootstrapEngine.iid_resample(
        values,
        rng,
    )

    assert len(result) == len(values)


def test_iid_resample_only_original_values():
    import random

    values = [
        1.0,
        2.0,
        3.0,
    ]

    rng = random.Random(42)

    result = BootstrapEngine.iid_resample(
        values,
        rng,
    )

    assert all(
        value in values
        for value in result
    )


def test_iid_resample_deterministic():
    import random

    values = list(range(50))

    first = BootstrapEngine.iid_resample(
        values,
        random.Random(42),
    )

    second = BootstrapEngine.iid_resample(
        values,
        random.Random(42),
    )

    assert first == second


def test_iid_empty_rejected():
    import random

    with pytest.raises(
        BootstrapValidationError
    ):
        BootstrapEngine.iid_resample(
            [],
            random.Random(42),
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
def test_iid_invalid_observation(value):
    import random

    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapEngine.iid_resample(
            [1.0, value],
            random.Random(42),
        )


# ============================================================
# Moving-block resampling
# ============================================================


def test_block_resample_size():
    import random

    values = list(range(100))

    result = (
        BootstrapEngine.moving_block_resample(
            values,
            5,
            random.Random(42),
        )
    )

    assert len(result) == 100


def test_block_resample_only_original_values():
    import random

    values = list(range(30))

    result = (
        BootstrapEngine.moving_block_resample(
            values,
            4,
            random.Random(42),
        )
    )

    assert all(
        value in values
        for value in result
    )


def test_block_resample_deterministic():
    import random

    values = list(range(50))

    first = (
        BootstrapEngine.moving_block_resample(
            values,
            5,
            random.Random(42),
        )
    )

    second = (
        BootstrapEngine.moving_block_resample(
            values,
            5,
            random.Random(42),
        )
    )

    assert first == second


@pytest.mark.parametrize(
    "block_size",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_block_resample_invalid_size(
    block_size,
):
    import random

    with pytest.raises(
        ValidationConfigurationError
    ):
        BootstrapEngine.moving_block_resample(
            list(range(30)),
            block_size,
            random.Random(42),
        )


def test_block_size_exceeds_sample():
    import random

    with pytest.raises(
        BootstrapValidationError
    ):
        BootstrapEngine.moving_block_resample(
            list(range(10)),
            11,
            random.Random(42),
        )


# ============================================================
# Bootstrap run - IID
# ============================================================


def test_run_iid():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        )
    )

    distribution = engine.run(
        positive_values()
    )

    assert (
        distribution.bootstrap_samples
        == 100
    )

    assert len(
        distribution.bootstrap_estimates
    ) == 100

    assert distribution.sample_size == 100

    assert (
        distribution.method
        == "iid_percentile"
    )

    assert distribution.block_size is None


def test_run_observed_mean():
    values = positive_values()

    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        )
    )

    distribution = engine.run(values)

    assert (
        distribution.observed_estimate
        == pytest.approx(mean(values))
    )


def test_run_confidence_level():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100,
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


def test_run_deterministic():
    config = make_config(
        bootstrap_samples=100,
        random_seed=123,
    )

    first = BootstrapEngine(
        config
    ).run(
        positive_values()
    )

    second = BootstrapEngine(
        config
    ).run(
        positive_values()
    )

    assert (
        first.bootstrap_estimates
        == second.bootstrap_estimates
    )

    assert (
        first.confidence_interval
        .lower
        == second.confidence_interval.lower
    )

    assert first.fingerprint == second.fingerprint


def test_run_custom_statistic():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
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
    engine = BootstrapEngine(
        make_config(
            minimum_observations=100
        )
    )

    with pytest.raises(
        BootstrapValidationError
    ):
        engine.run(
            positive_values(99)
        )


def test_run_non_finite_custom_statistic():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=20
        )
    )

    def bad_statistic(values):
        return float("nan")

    with pytest.raises(
        BootstrapValidationError
    ):
        engine.run(
            positive_values(),
            statistic=bad_statistic,
        )


# ============================================================
# Bootstrap run - moving block
# ============================================================


def test_run_moving_block():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        ),
        block_size=5,
    )

    distribution = engine.run(
        positive_values()
    )

    assert (
        distribution.method
        == "moving_block_percentile"
    )

    assert distribution.block_size == 5

    assert len(
        distribution.bootstrap_estimates
    ) == 100


def test_run_override_block_size():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        ),
        block_size=5,
    )

    distribution = engine.run(
        positive_values(),
        block_size=10,
    )

    assert distribution.block_size == 10


def test_run_block_size_exceeds_data():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=20
        ),
        block_size=101,
    )

    with pytest.raises(
        BootstrapValidationError
    ):
        engine.run(
            positive_values(100)
        )


# ============================================================
# Evaluation
# ============================================================


def test_evaluate_positive_distribution():
    engine = BootstrapEngine(
        minimum_probability_positive=0.90
    )

    distribution = make_distribution(
        estimates=[
            0.01,
            0.02,
            0.03,
            0.025,
        ],
        lower=0.005,
        upper=0.04,
    )

    report = engine.evaluate(
        distribution
    )

    assert report.probability_positive == 1.0
    assert report.passed is True


def test_evaluate_interval_crosses_zero():
    engine = BootstrapEngine()

    distribution = make_distribution(
        estimates=[
            -0.01,
            0.01,
            0.02,
        ],
        lower=-0.02,
        upper=0.03,
    )

    report = engine.evaluate(
        distribution
    )

    assert report.passed is False


def test_evaluate_low_positive_probability():
    engine = BootstrapEngine(
        minimum_probability_positive=0.90
    )

    distribution = make_distribution(
        estimates=[
            -0.01,
            -0.01,
            0.02,
            0.03,
        ],
        lower=0.001,
        upper=0.04,
    )

    report = engine.evaluate(
        distribution
    )

    assert (
        report.probability_positive
        == pytest.approx(0.50)
    )

    assert report.passed is False


def test_evaluate_high_relative_error():
    engine = BootstrapEngine(
        maximum_relative_standard_error=0.20
    )

    distribution = make_distribution(
        observed_estimate=0.02,
        estimates=[
            0.01,
            0.02,
            0.03,
        ],
        standard_error=0.01,
        lower=0.001,
        upper=0.04,
    )

    report = engine.evaluate(
        distribution
    )

    assert (
        report.relative_standard_error
        == pytest.approx(0.50)
    )

    assert report.passed is False


def test_evaluate_zero_estimate_zero_error():
    engine = BootstrapEngine()

    distribution = make_distribution(
        observed_estimate=0.0,
        estimates=[
            0.0,
            0.0,
            0.0,
        ],
        standard_error=0.0,
        lower=0.0,
        upper=0.0,
    )

    report = engine.evaluate(
        distribution
    )

    assert (
        report.relative_standard_error
        == 0.0
    )

    assert report.passed is False


def test_evaluate_zero_estimate_nonzero_error():
    engine = BootstrapEngine()

    distribution = make_distribution(
        observed_estimate=0.0,
        estimates=[
            -0.01,
            0.01,
        ],
        standard_error=0.01,
        lower=-0.02,
        upper=0.02,
    )

    report = engine.evaluate(
        distribution
    )

    assert (
        report.relative_standard_error
        == 1e12
    )

    assert report.passed is False


# ============================================================
# ValidationResult adapter
# ============================================================


def test_to_validation_result_pass():
    engine = BootstrapEngine(
        minimum_probability_positive=0.90
    )

    report = engine.evaluate(
        make_distribution(
            estimates=[
                0.01,
                0.02,
                0.03,
            ],
            lower=0.005,
            upper=0.04,
        )
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.BOOTSTRAP
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert (
        result.confidence_interval
        is not None
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0

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
    engine = BootstrapEngine()

    report = engine.evaluate(
        make_distribution(
            estimates=[
                -0.01,
                0.01,
            ],
            lower=-0.02,
            upper=0.02,
        )
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.status
        == ValidationStatus.FAILED
    )

    assert result.score is not None
    assert 0.0 <= result.score <= 1.0


# ============================================================
# Convenience validation
# ============================================================


def test_validate_positive_data():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=200
        ),
        minimum_probability_positive=0.90,
    )

    report = engine.validate(
        positive_values()
    )

    assert report.passed is True

    assert (
        report.probability_positive
        >= 0.90
    )


def test_validate_moving_block():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=100
        ),
        block_size=5,
        minimum_probability_positive=0.90,
    )

    report = engine.validate(
        positive_values()
    )

    assert (
        report.distribution.method
        == "moving_block_percentile"
    )

    assert report.passed is True


# ============================================================
# Snapshot
# ============================================================


def test_snapshot():
    engine = BootstrapEngine(
        make_config(
            bootstrap_samples=250,
            confidence_level=0.90,
            random_seed=123,
        ),
        block_size=7,
        minimum_probability_positive=0.92,
        maximum_relative_standard_error=0.60,
    )

    snapshot = engine.snapshot()

    assert snapshot["version"] == "v6.0.2"

    assert (
        snapshot["bootstrap_samples"]
        == 250
    )

    assert (
        snapshot["confidence_level"]
        == 0.90
    )

    assert snapshot["random_seed"] == 123
    assert snapshot["block_size"] == 7

    assert (
        snapshot[
            "minimum_probability_positive"
        ]
        == 0.92
    )

    assert (
        snapshot[
            "maximum_relative_standard_error"
        ]
        == 0.60
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
# Complete bootstrap flow
# ============================================================


def test_complete_bootstrap_flow():
    config = ValidationConfig(
        bootstrap_samples=300,
        confidence_level=0.95,
        minimum_observations=50,
        random_seed=42,
    )

    engine = BootstrapEngine(
        config,
        block_size=5,
        minimum_probability_positive=0.95,
        maximum_relative_standard_error=0.75,
    )

    values = [
        0.008
        + (index % 10) * 0.0005
        for index in range(250)
    ]

    distribution = engine.run(
        values
    )

    assert (
        distribution.bootstrap_samples
        == 300
    )

    assert distribution.sample_size == 250

    assert (
        distribution.method
        == "moving_block_percentile"
    )

    assert distribution.block_size == 5

    assert (
        distribution.observed_estimate
        > 0.0
    )

    assert (
        distribution.confidence_interval
        .lower
        > 0.0
    )

    report = engine.evaluate(
        distribution
    )

    assert report.passed is True

    assert (
        report.probability_positive
        == 1.0
    )

    result = engine.to_validation_result(
        report
    )

    assert (
        result.stage
        == ValidationStage.BOOTSTRAP
    )

    assert (
        result.status
        == ValidationStatus.PASSED
    )

    assert result.score is not None
    assert result.score > 0.80

    assert (
        result.confidence_interval
        is not None
    )

    assert (
        result.confidence_interval.lower
        > 0.0
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

    assert len(
        distribution.fingerprint
    ) == 64

    assert len(report.fingerprint) == 64