import pytest

from autonomous_experiment_designer.quantum_monte_carlo_engine import (
    QuantumMonteCarloEngine,
    QuantumMonteCarloMethod,
    QuantumMonteCarloResult,
    QuantumMonteCarloStatus,
)


def constant_sampler():
    return 1.0


def constant_observable(value):
    return float(value)


def identity_observable(value):
    return float(value)


def test_default_configuration():
    engine = QuantumMonteCarloEngine()

    assert engine.sample_count == 1000
    assert (
        engine.method
        == QuantumMonteCarloMethod.PSEUDO_RANDOM
    )
    assert engine.confidence_level == 0.95
    assert engine.seed == 42


def test_custom_configuration():
    engine = QuantumMonteCarloEngine(
        sample_count=250,
        method=QuantumMonteCarloMethod.STRATIFIED,
        confidence_level=0.99,
        seed=123,
    )

    assert engine.sample_count == 250
    assert (
        engine.method
        == QuantumMonteCarloMethod.STRATIFIED
    )
    assert engine.confidence_level == 0.99
    assert engine.seed == 123


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "100"],
)
def test_invalid_sample_count_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumMonteCarloEngine(
            sample_count=value
        )


@pytest.mark.parametrize(
    "value",
    [None, "PSEUDO_RANDOM", 123],
)
def test_invalid_method_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumMonteCarloEngine(
            method=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        1,
        -0.1,
        1.1,
        True,
        "0.95",
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_confidence_level_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumMonteCarloEngine(
            confidence_level=value
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        1.5,
        "42",
        None,
    ],
)
def test_invalid_seed_is_rejected(value):
    with pytest.raises(ValueError):
        QuantumMonteCarloEngine(
            seed=value
        )


def test_pseudo_random_run_completes():
    engine = QuantumMonteCarloEngine(
        sample_count=10,
    )

    result = engine.run(
        constant_sampler,
        identity_observable,
    )

    assert isinstance(
        result,
        QuantumMonteCarloResult,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.COMPLETED
    )

    assert (
        result.method
        == QuantumMonteCarloMethod.PSEUDO_RANDOM
    )

    assert result.estimate == 1.0
    assert result.variance == 0.0
    assert result.standard_error == 0.0

    assert result.confidence_interval == (
        1.0,
        1.0,
    )

    assert result.sample_count == 10
    assert result.samples == (
        1.0,
    ) * 10


def test_stratified_method_completes():
    engine = QuantumMonteCarloEngine(
        sample_count=20,
        method=QuantumMonteCarloMethod.STRATIFIED,
    )

    result = engine.run(
        constant_sampler,
        identity_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.COMPLETED
    )

    assert (
        result.method
        == QuantumMonteCarloMethod.STRATIFIED
    )

    assert result.sample_count == 20


def test_quasi_random_method_completes():
    engine = QuantumMonteCarloEngine(
        sample_count=20,
        method=QuantumMonteCarloMethod.QUASI_RANDOM,
    )

    result = engine.run(
        constant_sampler,
        identity_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.COMPLETED
    )

    assert (
        result.method
        == QuantumMonteCarloMethod.QUASI_RANDOM
    )


def test_mean_is_calculated_correctly():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    def sampler():
        return next(values)

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        sampler,
        identity_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.COMPLETED
    )

    assert result.estimate == 2.5


def test_variance_is_calculated_correctly():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    assert result.variance == pytest.approx(
        5.0 / 3.0
    )


def test_standard_error_is_calculated_correctly():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    expected = (
        (5.0 / 3.0) / 4.0
    ) ** 0.5

    assert result.standard_error == pytest.approx(
        expected
    )


def test_confidence_interval_is_calculated():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
        confidence_level=0.95,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    margin = (
        1.960
        * result.standard_error
    )

    assert result.confidence_interval == pytest.approx(
        (
            result.estimate - margin,
            result.estimate + margin,
        )
    )


def test_confidence_interval_contains_estimate():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    lower, upper = (
        result.confidence_interval
    )

    assert lower <= result.estimate <= upper


def test_single_sample_has_zero_variance():
    result = QuantumMonteCarloEngine(
        sample_count=1,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.COMPLETED
    )

    assert result.estimate == 1.0
    assert result.variance == 0.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == (
        1.0,
        1.0,
    )


def test_sample_count_is_preserved():
    result = QuantumMonteCarloEngine(
        sample_count=37,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert result.sample_count == 37
    assert len(result.samples) == 37


def test_observable_is_applied_to_samples():
    values = iter(
        [1.0, 2.0, 3.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=3,
    ).run(
        lambda: next(values),
        lambda value: value * 2.0,
    )

    assert result.samples == (
        2.0,
        4.0,
        6.0,
    )

    assert result.estimate == 4.0


def test_invalid_sampler_is_rejected():
    engine = QuantumMonteCarloEngine(
        sample_count=5,
    )

    with pytest.raises(ValueError):
        engine.run(
            None,
            identity_observable,
        )


@pytest.mark.parametrize(
    "sampler",
    [
        "invalid",
        123,
        None,
    ],
)
def test_invalid_sampler_types_are_rejected(
    sampler,
):
    engine = QuantumMonteCarloEngine(
        sample_count=5,
    )

    with pytest.raises(ValueError):
        engine.run(
            sampler,
            identity_observable,
        )


@pytest.mark.parametrize(
    "observable",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_observable_is_rejected(
    observable,
):
    engine = QuantumMonteCarloEngine(
        sample_count=5,
    )

    with pytest.raises(ValueError):
        engine.run(
            constant_sampler,
            observable,
        )


def test_non_numeric_observable_result_fails():
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        lambda _: "invalid",
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.estimate is None
    assert result.variance is None
    assert result.standard_error is None
    assert result.confidence_interval is None
    assert result.samples == ()
    assert result.error is not None


def test_boolean_observable_result_fails():
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        lambda _: True,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.error is not None


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_nonfinite_observable_result_fails(value):
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        lambda _: value,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.error is not None


def test_sampler_exception_is_captured():
    def failing_sampler():
        raise RuntimeError(
            "controlled sampler failure"
        )

    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        failing_sampler,
        identity_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.estimate is None
    assert result.samples == ()

    assert (
        "controlled sampler failure"
        in result.error
    )


def test_observable_exception_is_captured():
    def failing_observable(_):
        raise RuntimeError(
            "controlled observable failure"
        )

    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        failing_observable,
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.estimate is None

    assert (
        "controlled observable failure"
        in result.error
    )


def test_metadata_contains_configuration():
    engine = QuantumMonteCarloEngine(
        sample_count=25,
        method=QuantumMonteCarloMethod.QUASI_RANDOM,
        confidence_level=0.99,
        seed=123,
    )

    result = engine.run(
        constant_sampler,
        identity_observable,
    )

    assert (
        result.metadata[
            "configured_sample_count"
        ]
        == 25
    )

    assert (
        result.metadata[
            "confidence_level"
        ]
        == 0.99
    )

    assert result.metadata["seed"] == 123

    assert (
        result.metadata["method"]
        == "QUASI_RANDOM"
    )


def test_result_is_immutable():
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        identity_observable,
    )

    with pytest.raises(AttributeError):
        result.status = (
            QuantumMonteCarloStatus.FAILED
        )


def test_samples_are_returned_as_tuple():
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert isinstance(
        result.samples,
        tuple,
    )


def test_result_is_repeatable():
    first = QuantumMonteCarloEngine(
        sample_count=20,
        seed=42,
    ).run(
        constant_sampler,
        identity_observable,
    )

    second = QuantumMonteCarloEngine(
        sample_count=20,
        seed=42,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert first == second


def test_different_seed_is_recorded():
    first = QuantumMonteCarloEngine(
        sample_count=5,
        seed=42,
    ).run(
        constant_sampler,
        identity_observable,
    )

    second = QuantumMonteCarloEngine(
        sample_count=5,
        seed=43,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert first.metadata["seed"] == 42
    assert second.metadata["seed"] == 43


def test_high_confidence_uses_larger_z_score():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
        confidence_level=0.99,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    margin = (
        2.576
        * result.standard_error
    )

    lower, upper = (
        result.confidence_interval
    )

    assert lower == pytest.approx(
        result.estimate - margin
    )

    assert upper == pytest.approx(
        result.estimate + margin
    )


def test_ninety_percent_confidence_level():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
        confidence_level=0.90,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    margin = (
        1.645
        * result.standard_error
    )

    assert result.confidence_interval == pytest.approx(
        (
            result.estimate - margin,
            result.estimate + margin,
        )
    )


def test_samples_preserve_measurement_order():
    values = iter(
        [0.1, 0.2, 0.3, 0.4, 0.5]
    )

    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    assert result.samples == (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    )


def test_zero_mean_is_supported():
    values = iter(
        [-1.0, 1.0, -2.0, 2.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    assert result.estimate == 0.0
    assert result.variance == pytest.approx(
        10.0 / 3.0
    )


def test_negative_observable_values_are_supported():
    values = iter(
        [-3.0, -2.0, -1.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=3,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    assert result.estimate == -2.0
    assert result.sample_count == 3


def test_large_observable_values_are_supported():
    values = iter(
        [
            1_000_000.0,
            1_000_001.0,
            999_999.0,
        ]
    )

    result = QuantumMonteCarloEngine(
        sample_count=3,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    assert result.estimate == 1_000_000.0


def test_result_contains_complete_statistical_output():
    result = QuantumMonteCarloEngine(
        sample_count=10,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert result.estimate is not None
    assert result.variance is not None
    assert result.standard_error is not None
    assert result.confidence_interval is not None
    assert result.samples
    assert result.sample_count == 10


def test_method_is_preserved_in_result():
    for method in QuantumMonteCarloMethod:
        result = QuantumMonteCarloEngine(
            sample_count=5,
            method=method,
        ).run(
            constant_sampler,
            identity_observable,
        )

        assert result.method == method


def test_engine_does_not_modify_external_state():
    values = [
        1.0,
        2.0,
        3.0,
    ]

    original = list(values)

    iterator = iter(values)

    result = QuantumMonteCarloEngine(
        sample_count=3,
    ).run(
        lambda: next(iterator),
        identity_observable,
    )

    assert values == original
    assert result.samples == (
        1.0,
        2.0,
        3.0,
    )


def test_constant_distribution_has_zero_standard_error():
    result = QuantumMonteCarloEngine(
        sample_count=100,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert result.standard_error == 0.0
    assert result.confidence_interval == (
        1.0,
        1.0,
    )


def test_confidence_interval_width_is_nonnegative():
    values = iter(
        [0.0, 1.0, 2.0, 3.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        identity_observable,
    )

    lower, upper = (
        result.confidence_interval
    )

    assert upper >= lower


def test_failed_result_has_zero_sample_count():
    result = QuantumMonteCarloEngine(
        sample_count=5,
    ).run(
        constant_sampler,
        lambda _: "invalid",
    )

    assert (
        result.status
        == QuantumMonteCarloStatus.FAILED
    )

    assert result.sample_count == 0
    assert result.samples == ()


def test_confidence_level_metadata_is_float():
    result = QuantumMonteCarloEngine(
        sample_count=5,
        confidence_level=0.95,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert isinstance(
        result.metadata["confidence_level"],
        float,
    )


def test_sample_count_metadata_matches_configuration():
    result = QuantumMonteCarloEngine(
        sample_count=17,
    ).run(
        constant_sampler,
        identity_observable,
    )

    assert (
        result.metadata[
            "configured_sample_count"
        ]
        == 17
    )

    assert result.sample_count == 17


def test_qmc_engine_supports_observable_transformation():
    values = iter(
        [1.0, 2.0, 3.0, 4.0]
    )

    result = QuantumMonteCarloEngine(
        sample_count=4,
    ).run(
        lambda: next(values),
        lambda value: value ** 2,
    )

    assert result.samples == (
        1.0,
        4.0,
        9.0,
        16.0,
    )

    assert result.estimate == 7.5


def test_result_equality_is_deterministic():
    def run():
        values = iter(
            [1.0, 2.0, 3.0]
        )

        return QuantumMonteCarloEngine(
            sample_count=3,
            seed=99,
        ).run(
            lambda: next(values),
            identity_observable,
        )

    first = run()
    second = run()

    assert first == second