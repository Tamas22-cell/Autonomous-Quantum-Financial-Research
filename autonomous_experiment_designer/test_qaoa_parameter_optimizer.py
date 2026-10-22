import pytest

from autonomous_experiment_designer.qaoa_parameter_optimizer import (
    QAOAOptimizationResult,
    QAOAOptimizationStatus,
    QAOAOptimizerMethod,
    QAOAParameterOptimizer,
)


def quadratic_objective(parameters):
    return sum(
        (value - 1.0) ** 2
        for value in parameters
    )


def shifted_objective(parameters):
    return sum(
        (value - 0.5) ** 2
        for value in parameters
    )


def test_default_configuration():
    optimizer = QAOAParameterOptimizer()

    assert optimizer.parameter_count == 2
    assert optimizer.method == QAOAOptimizerMethod.GRID
    assert optimizer.parameter_bounds == (
        (0.0, 2.0),
        (0.0, 2.0),
    )
    assert optimizer.grid_points == 5
    assert optimizer.max_evaluations == 100


def test_custom_configuration():
    optimizer = QAOAParameterOptimizer(
        parameter_count=3,
        method=QAOAOptimizerMethod.RANDOM,
        parameter_bounds=(
            (0.0, 1.0),
            (-1.0, 1.0),
            (0.0, 3.0),
        ),
        grid_points=7,
        max_evaluations=20,
    )

    assert optimizer.parameter_count == 3
    assert optimizer.method == QAOAOptimizerMethod.RANDOM
    assert optimizer.parameter_bounds == (
        (0.0, 1.0),
        (-1.0, 1.0),
        (0.0, 3.0),
    )
    assert optimizer.grid_points == 7
    assert optimizer.max_evaluations == 20


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
        "2",
    ],
)
def test_invalid_parameter_count_is_rejected(value):
    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
            parameter_count=value
        )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "GRID",
        123,
    ],
)
def test_invalid_method_is_rejected(value):
    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
            method=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
        "5",
    ],
)
def test_invalid_grid_points_are_rejected(value):
    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
            grid_points=value
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
        "10",
    ],
)
def test_invalid_max_evaluations_are_rejected(value):
    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
            max_evaluations=value
        )


def test_default_bounds_match_parameter_count():
    optimizer = QAOAParameterOptimizer(
        parameter_count=4
    )

    assert optimizer.parameter_bounds == (
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (0.0, 2.0),
    )


def test_bounds_length_must_match_parameter_count():
    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
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
        None,
    ],
)
def test_invalid_bounds_container_is_rejected_when_not_default(
    bounds,
):
    if bounds is None:
        return

    with pytest.raises(ValueError):
        QAOAParameterOptimizer(
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
        QAOAParameterOptimizer(
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
        QAOAParameterOptimizer(
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
        QAOAParameterOptimizer(
            parameter_count=2,
            parameter_bounds=bounds,
        )


def test_grid_optimizer_completes():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.GRID,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
        grid_points=5,
    )

    result = optimizer.optimize(
        quadratic_objective
    )

    assert isinstance(
        result,
        QAOAOptimizationResult,
    )
    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )
    assert result.method == QAOAOptimizerMethod.GRID
    assert result.best_parameters == (
        1.0,
        1.0,
    )
    assert result.best_objective == 0.0
    assert result.evaluations == 25


def test_grid_optimizer_finds_shifted_optimum():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        grid_points=5,
    )

    result = optimizer.optimize(
        shifted_objective
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )
    assert result.best_parameters == (
        0.5,
        0.5,
    )
    assert result.best_objective == 0.0


def test_grid_evaluations_respect_maximum():
    optimizer = QAOAParameterOptimizer(
        parameter_count=3,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        grid_points=10,
        max_evaluations=17,
    )

    result = optimizer.optimize(
        quadratic_objective
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )
    assert result.evaluations == 17
    assert len(
        result.convergence_history
    ) == 17


def test_random_optimizer_is_deterministic_with_seed():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.RANDOM,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
        max_evaluations=20,
    )

    first = optimizer.optimize(
        quadratic_objective,
        seed=42,
    )

    second = optimizer.optimize(
        quadratic_objective,
        seed=42,
    )

    assert first.best_parameters == second.best_parameters
    assert first.best_objective == second.best_objective
    assert first.convergence_history == (
        second.convergence_history
    )


def test_random_optimizer_changes_with_different_seed():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.RANDOM,
        max_evaluations=20,
    )

    first = optimizer.optimize(
        quadratic_objective,
        seed=42,
    )

    second = optimizer.optimize(
        quadratic_objective,
        seed=43,
    )

    assert (
        first.convergence_history
        != second.convergence_history
    )


def test_local_optimizer_uses_initial_parameters():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.COBYLA,
        parameter_bounds=(
            (0.0, 2.0),
            (0.0, 2.0),
        ),
    )

    result = optimizer.optimize(
        quadratic_objective,
        initial_parameters=(
            1.0,
            1.0,
        ),
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )
    assert result.best_parameters == (
        1.0,
        1.0,
    )
    assert result.best_objective == 0.0


def test_spsa_method_is_supported():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.SPSA,
    )

    result = optimizer.optimize(
        quadratic_objective,
        initial_parameters=(
            1.0,
            1.0,
        ),
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )


def test_single_grid_point_uses_midpoint():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        grid_points=1,
    )

    result = optimizer.optimize(
        quadratic_objective
    )

    assert (
        result.best_parameters
        == (1.0, 1.0)
    )


@pytest.mark.parametrize(
    "parameters",
    [
        (1.0,),
        (1.0, 1.0, 1.0),
        "invalid",
        None,
    ],
)
def test_invalid_initial_parameters_are_captured(
    parameters,
):
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
    )

    result = optimizer.optimize(
        quadratic_objective,
        initial_parameters=parameters,
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
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
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        quadratic_objective,
        seed=seed,
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.error is not None


@pytest.mark.parametrize(
    "objective",
    [
        None,
        "invalid",
        123,
    ],
)
def test_invalid_objective_is_rejected(objective):
    optimizer = QAOAParameterOptimizer()

    with pytest.raises(ValueError):
        optimizer.optimize(
            objective
        )


def test_non_numeric_objective_result_fails_execution():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        lambda _: "invalid"
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.best_parameters == ()
    assert result.best_objective is None
    assert result.evaluations == 0
    assert result.error is not None


def test_boolean_objective_result_fails_execution():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        lambda _: True
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.error is not None


def test_nonfinite_objective_result_fails_execution():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        lambda _: float("nan")
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.error is not None


def test_objective_exception_is_captured():
    optimizer = QAOAParameterOptimizer()

    def failing_objective(_):
        raise RuntimeError(
            "controlled objective failure"
        )

    result = optimizer.optimize(
        failing_objective
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.best_parameters == ()
    assert result.best_objective is None
    assert result.evaluations == 0
    assert (
        "controlled objective failure"
        in result.error
    )


def test_convergence_history_is_monotonic_best_so_far():
    optimizer = QAOAParameterOptimizer(
        parameter_count=1,
        parameter_bounds=(
            (0.0, 2.0),
        ),
        grid_points=5,
    )

    result = optimizer.optimize(
        lambda parameters: (
            parameters[0] - 1.0
        ) ** 2
    )

    running_best = float("inf")

    for value in result.convergence_history:
        running_best = min(
            running_best,
            value,
        )


def test_metadata_contains_configuration():
    optimizer = QAOAParameterOptimizer(
        parameter_count=3,
        max_evaluations=12,
        grid_points=4,
    )

    result = optimizer.optimize(
        quadratic_objective
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


def test_result_contains_convergence_history():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        grid_points=3,
    )

    result = optimizer.optimize(
        quadratic_objective
    )

    assert len(
        result.convergence_history
    ) == result.evaluations


def test_result_is_immutable():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        quadratic_objective
    )

    with pytest.raises(AttributeError):
        result.status = (
            QAOAOptimizationStatus.FAILED
        )


def test_parameter_bounds_are_respected():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.RANDOM,
        parameter_bounds=(
            (-1.0, 1.0),
            (2.0, 3.0),
        ),
        max_evaluations=50,
    )

    result = optimizer.optimize(
        quadratic_objective,
        seed=42,
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )

    for parameter, bound in zip(
        result.best_parameters,
        optimizer.parameter_bounds,
    ):
        lower, upper = bound

        assert lower <= parameter <= upper


def test_random_optimizer_respects_evaluation_limit():
    optimizer = QAOAParameterOptimizer(
        parameter_count=4,
        method=QAOAOptimizerMethod.RANDOM,
        max_evaluations=7,
    )

    result = optimizer.optimize(
        quadratic_objective
    )

    assert result.evaluations == 7


def test_local_optimizer_respects_parameter_bounds():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        method=QAOAOptimizerMethod.COBYLA,
        parameter_bounds=(
            (0.0, 1.0),
            (0.0, 1.0),
        ),
    )

    result = optimizer.optimize(
        quadratic_objective,
        initial_parameters=(
            0.0,
            0.0,
        ),
    )

    for parameter, bound in zip(
        result.best_parameters,
        optimizer.parameter_bounds,
    ):
        lower, upper = bound

        assert lower <= parameter <= upper


def test_objective_receives_tuple_parameters():
    optimizer = QAOAParameterOptimizer()

    received_types = []

    def objective(parameters):
        received_types.append(
            type(parameters)
        )

        return quadratic_objective(
            parameters
        )

    result = optimizer.optimize(
        objective
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )

    assert all(
        value is tuple
        for value in received_types
    )


def test_optimizer_does_not_modify_input_bounds():
    bounds = [
        [0.0, 2.0],
        [0.0, 2.0],
    ]

    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        parameter_bounds=bounds,
    )

    original = [
        [0.0, 2.0],
        [0.0, 2.0],
    ]

    optimizer.optimize(
        quadratic_objective
    )

    assert bounds == original


def test_qaoa_result_status_is_completed_for_valid_objective():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        quadratic_objective
    )

    assert (
        result.status
        == QAOAOptimizationStatus.COMPLETED
    )


def test_failed_result_contains_no_best_solution():
    optimizer = QAOAParameterOptimizer()

    result = optimizer.optimize(
        lambda _: float("inf")
    )

    assert (
        result.status
        == QAOAOptimizationStatus.FAILED
    )
    assert result.best_parameters == ()
    assert result.best_objective is None


def test_optimizer_is_repeatable():
    optimizer = QAOAParameterOptimizer(
        parameter_count=2,
        grid_points=5,
    )

    first = optimizer.optimize(
        quadratic_objective
    )

    second = optimizer.optimize(
        quadratic_objective
    )

    assert first == second