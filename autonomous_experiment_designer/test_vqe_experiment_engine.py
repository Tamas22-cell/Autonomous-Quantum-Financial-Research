import pytest

from autonomous_experiment_designer.vqe_experiment_engine import (
    VQEExecutionStatus,
    VQEExperimentEngine,
    VQEExperimentResult,
    VQEOptimizationMethod,
)


def quadratic_energy(parameters):
    return sum(
        (value - 1.0) ** 2
        for value in parameters
    )


def shifted_energy(parameters):
    return sum(
        (value - 0.5) ** 2
        for value in parameters
    )


def test_default_configuration():
    engine = VQEExperimentEngine()

    assert engine.parameter_count == 2
    assert engine.method == VQEOptimizationMethod.GRID
    assert engine.parameter_bounds == (
        (0.0, 2.0),
        (0.0, 2.0),
    )
    assert engine.grid_points == 5
    assert engine.max_evaluations == 100


def test_custom_configuration():
    engine = VQEExperimentEngine(
        parameter_count=3,
        method=VQEOptimizationMethod.RANDOM,
        parameter_bounds=(
            (0.0, 1.0),
            (-1.0, 1.0),
            (0.0, 3.0),
        ),
        grid_points=7,
        max_evaluations=20,
    )

    assert engine.parameter_count == 3
    assert engine.method == VQEOptimizationMethod.RANDOM
    assert engine.parameter_bounds == (
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.0, 3.0),
    )
    assert engine.grid_points == 7
    assert engine.max_evaluations == 20


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "2"],
)
def test_invalid_parameter_count_is_rejected(value):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=value
        )


@pytest.mark.parametrize(
    "value",
    [None, "GRID", 123],
)
def test_invalid_method_is_rejected(value):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            method=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "5"],
)
def test_invalid_grid_points_are_rejected(value):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            grid_points=value
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5, "10"],
)
def test_invalid_max_evaluations_are_rejected(value):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            max_evaluations=value
        )


def test_default_bounds_match_parameter_count():
    engine = VQEExperimentEngine(
        parameter_count=4
    )

    assert engine.parameter_bounds == (
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
    )


def test_bounds_length_must_match_parameter_count():
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=3,
            parameter_bounds=(
                (0.0, 1.0),
                (0.0, 1.0),
            ),
        )


@pytest.mark.parametrize(
    "bounds",
    [
        "invalid",
        123,
    ],
)
def test_invalid_bounds_container_is_rejected(bounds):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=2,
            parameter_bounds=bounds,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (
            (0.0, 1.0),
            (0.0,),
        ),
        (
            (0.0, 1.0, 2.0),
            (0.0, 1.0),
        ),
        (
            "invalid",
            (0.0, 1.0),
        ),
    ],
)
def test_invalid_bound_shape_is_rejected(bounds):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=2,
            parameter_bounds=bounds,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        (
            (0.0, 0.0),
            (0.0, 1.0),
        ),
    ],
)
def test_invalid_bound_order_is_rejected(bounds):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=2,
            parameter_bounds=bounds,
        )


@pytest.mark.parametrize(
    "bounds",
    [
        (
            (float("nan"), 1.0),
            (0.0, 1.0),
        ),
        (
            (0.0, float("inf")),
            (0.0, 1.0),
        ),
        (
            (0.0, 1.0),
            (float("-inf"), 1.0),
        ),
    ],
)
def test_nonfinite_bounds_are_rejected(bounds):
    with pytest.raises(ValueError):
        VQEExperimentEngine(
            parameter_count=2,
            parameter_bounds=bounds,
        )


def test_grid_run_completes():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.GRID,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
        grid_points=5,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert isinstance(
        result,
        VQEExperimentResult,
    )
    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )
    assert result.method == VQEOptimizationMethod.GRID
    assert result.best_parameters == (
        1.0,
        1.0,
    )
    assert result.best_energy == 0.0
    assert result.evaluations == 25


def test_grid_run_finds_shifted_ground_state():
    engine = VQEExperimentEngine(
        parameter_count=2,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        grid_points=5,
    )

    result = engine.run(
        shifted_energy,
        initial_parameters=(0.5, 0.5),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )
    assert result.best_parameters == (
        0.5,
        0.5,
    )
    assert result.best_energy == 0.0


def test_grid_respects_evaluation_limit():
    engine = VQEExperimentEngine(
        parameter_count=3,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        grid_points=10,
        max_evaluations=17,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(0.5, 0.5, 0.5),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )
    assert result.evaluations == 17
    assert len(
        result.energy_history
    ) == 17


def test_random_run_is_deterministic_with_seed():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.RANDOM,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
        max_evaluations=20,
    )

    first = engine.run(
        quadratic_energy,
        seed=42,
        initial_parameters=(1.0, 1.0),
    )

    second = engine.run(
        quadratic_energy,
        seed=42,
        initial_parameters=(1.0, 1.0),
    )

    assert first.best_parameters == second.best_parameters
    assert first.best_energy == second.best_energy
    assert first.energy_history == second.energy_history


def test_random_run_changes_with_different_seed():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.RANDOM,
        max_evaluations=20,
    )

    first = engine.run(
        quadratic_energy,
        seed=42,
        initial_parameters=(1.0, 1.0),
    )

    second = engine.run(
        quadratic_energy,
        seed=43,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        first.energy_history
        != second.energy_history
    )


def test_local_run_uses_initial_parameters():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.LOCAL,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )
    assert result.best_parameters == (
        1.0,
        1.0,
    )
    assert result.best_energy == 0.0


def test_single_grid_point_uses_midpoint():
    engine = VQEExperimentEngine(
        parameter_count=2,
        grid_points=1,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert result.best_parameters == (
        1.0,
        1.0,
    )


@pytest.mark.parametrize(
    "parameters",
    [
        (),
        (1.0,),
        (1.0, 1.0, 1.0),
        "invalid",
        None,
    ],
)
def test_invalid_initial_parameters_are_captured(
    parameters,
):
    engine = VQEExperimentEngine(
        parameter_count=2,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=parameters,
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.error is not None


@pytest.mark.parametrize(
    "seed",
    [
        True,
        1.5,
        "42",
        None,
    ],
)
def test_invalid_seed_is_captured(seed):
    engine = VQEExperimentEngine()

    result = engine.run(
        quadratic_energy,
        seed=seed,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.error is not None


@pytest.mark.parametrize(
    "energy_function",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_energy_function_is_rejected(
    energy_function,
):
    engine = VQEExperimentEngine()

    with pytest.raises(ValueError):
        engine.run(
            energy_function,
            initial_parameters=(1.0, 1.0),
        )


def test_non_numeric_energy_result_fails():
    engine = VQEExperimentEngine()

    result = engine.run(
        lambda _: "invalid",
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.error is not None


def test_boolean_energy_result_fails():
    engine = VQEExperimentEngine()

    result = engine.run(
        lambda _: True,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.error is not None


def test_nonfinite_energy_result_fails():
    engine = VQEExperimentEngine()

    result = engine.run(
        lambda _: float("nan"),
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.error is not None


def test_energy_function_exception_is_captured():
    engine = VQEExperimentEngine()

    def failing_energy(_):
        raise RuntimeError(
            "controlled energy failure"
        )

    result = engine.run(
        failing_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.best_parameters == ()
    assert result.best_energy is None
    assert result.evaluations == 0
    assert (
        "controlled energy failure"
        in result.error
    )


def test_convergence_delta_is_recorded():
    engine = VQEExperimentEngine(
        parameter_count=1,
        parameter_bounds=(
            (0.0, 2.0),
        ),
        grid_points=5,
    )

    result = engine.run(
        lambda parameters: (
            parameters[0] - 1.0
        ) ** 2,
        initial_parameters=(1.0,),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )
    assert result.convergence_delta is not None
    assert result.convergence_delta >= 0.0


def test_metadata_contains_configuration():
    engine = VQEExperimentEngine(
        parameter_count=3,
        max_evaluations=12,
        grid_points=4,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0, 1.0),
    )

    assert (
        result.metadata["parameter_count"]
        == 3
    )
    assert (
        result.metadata["max_evaluations"]
        == 12
    )
    assert (
        result.metadata["grid_points"]
        == 4
    )
    assert result.metadata["seed"] == 42


def test_result_contains_energy_history():
    engine = VQEExperimentEngine(
        parameter_count=2,
        grid_points=3,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert len(
        result.energy_history
    ) == result.evaluations


def test_result_is_immutable():
    engine = VQEExperimentEngine()

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    with pytest.raises(AttributeError):
        result.status = VQEExecutionStatus.FAILED


def test_parameter_bounds_are_respected():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.RANDOM,
        parameter_bounds=(
            (-1.0, 1.0),
            (2.0, 3.0),
        ),
        max_evaluations=50,
    )

    result = engine.run(
        quadratic_energy,
        seed=42,
        initial_parameters=(0.0, 2.5),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )

    for parameter, bound in zip(
        result.best_parameters,
        engine.parameter_bounds,
    ):
        lower, upper = bound
        assert lower <= parameter <= upper


def test_random_run_respects_evaluation_limit():
    engine = VQEExperimentEngine(
        parameter_count=4,
        method=VQEOptimizationMethod.RANDOM,
        max_evaluations=7,
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(
            1.0,
            1.0,
            1.0,
            1.0,
        ),
    )

    assert result.evaluations == 7


def test_local_run_respects_parameter_bounds():
    engine = VQEExperimentEngine(
        parameter_count=2,
        method=VQEOptimizationMethod.LOCAL,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
        ),
    )

    result = engine.run(
        quadratic_energy,
        initial_parameters=(0.0, 0.0),
    )

    for parameter, bound in zip(
        result.best_parameters,
        engine.parameter_bounds,
    ):
        lower, upper = bound
        assert lower <= parameter <= upper


def test_energy_function_receives_tuple_parameters():
    engine = VQEExperimentEngine()

    received_types = []

    def energy(parameters):
        received_types.append(
            type(parameters)
        )
        return quadratic_energy(
            parameters
        )

    result = engine.run(
        energy,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )

    assert all(
        value is tuple
        for value in received_types
    )


def test_engine_does_not_modify_input_bounds():
    bounds = [
        [0.0, 2.0],
        [0.0, 2.0],
    ]

    engine = VQEExperimentEngine(
        parameter_count=2,
        parameter_bounds=bounds,
    )

    original = [
        [0.0, 2.0],
        [0.0, 2.0],
    ]

    engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert bounds == original


def test_vqe_result_status_is_completed_for_valid_energy():
    engine = VQEExperimentEngine()

    result = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.COMPLETED
    )


def test_failed_result_contains_no_best_solution():
    engine = VQEExperimentEngine()

    result = engine.run(
        lambda _: float("inf"),
        initial_parameters=(1.0, 1.0),
    )

    assert (
        result.status
        == VQEExecutionStatus.FAILED
    )
    assert result.best_parameters == ()
    assert result.best_energy is None


def test_engine_is_repeatable():
    engine = VQEExperimentEngine(
        parameter_count=2,
        grid_points=5,
    )

    first = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    second = engine.run(
        quadratic_energy,
        initial_parameters=(1.0, 1.0),
    )

    assert first == second