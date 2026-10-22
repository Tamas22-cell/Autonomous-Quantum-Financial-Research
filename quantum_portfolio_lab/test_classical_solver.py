"""
Tests for the v5.0.8 Classical Exact QUBO Solver.

Coverage:

- exact exhaustive optimization
- global optimum verification
- full 2^n search-space enumeration
- selected asset decoding
- equal-weight portfolio derivation
- cardinality diagnostics
- infeasible optimum detection
- multiple global optima
- deterministic tie handling
- maximum problem-size protection
- absolute objective gap
- relative objective gap
- invalid inputs
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from quantum_portfolio_lab.classical_solver import (
    ClassicalExactQUBOSolver,
    ClassicalExactSolverConfig,
    ClassicalSolverError,
)
from quantum_portfolio_lab.qubo_builder import (
    PortfolioQUBOBuilder,
    PortfolioSelectionQUBORequest,
)
from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QUBOVariable,
    QuantumExecutionBackend,
    QuantumSolutionStatus,
    QuantumSolverType,
)


def build_request(
    *,
    target_cardinality: int = 2,
    budget_penalty: float = 10.0,
) -> PortfolioSelectionQUBORequest:
    return PortfolioSelectionQUBORequest(
        asset_symbols=(
            "SPY",
            "GLD",
            "TLT",
        ),
        expected_returns=np.array(
            [
                0.08,
                0.04,
                0.03,
            ],
            dtype=float,
        ),
        covariance_matrix=np.array(
            [
                [0.040, 0.010, -0.005],
                [0.010, 0.020, 0.002],
                [-0.005, 0.002, 0.015],
            ],
            dtype=float,
        ),
        target_cardinality=target_cardinality,
        risk_aversion=1.5,
        return_weight=2.0,
        budget_penalty=budget_penalty,
    )


def build_problem(
    *,
    target_cardinality: int = 2,
    budget_penalty: float = 10.0,
) -> PortfolioQUBOProblem:
    return PortfolioQUBOBuilder().build(
        build_request(
            target_cardinality=target_cardinality,
            budget_penalty=budget_penalty,
        )
    )


def build_simple_problem() -> PortfolioQUBOProblem:
    """
    Simple two-variable QUBO with unique optimum (1, 0).

    Energies:

        (0, 0) -> 0
        (1, 0) -> -2
        (0, 1) -> -1
        (1, 1) -> -3

    With a cardinality penalty targeting one selected asset,
    (1, 0) becomes the unique optimum.
    """

    return PortfolioQUBOProblem(
        variables=(
            QUBOVariable(
                name="x_A",
                index=0,
                asset_symbol="A",
            ),
            QUBOVariable(
                name="x_B",
                index=1,
                asset_symbol="B",
            ),
        ),
        qubo_matrix=np.array(
            [
                [-7.0, 5.0],
                [5.0, -6.0],
            ],
            dtype=float,
        ),
        offset=5.0,
        asset_symbols=(
            "A",
            "B",
        ),
        budget_penalty=5.0,
        target_cardinality=1,
    )


def build_tied_problem() -> PortfolioQUBOProblem:
    """
    Two-variable QUBO where (1, 0) and (0, 1)
    are identical global optima.
    """

    return PortfolioQUBOProblem(
        variables=(
            QUBOVariable(
                name="x_A",
                index=0,
                asset_symbol="A",
            ),
            QUBOVariable(
                name="x_B",
                index=1,
                asset_symbol="B",
            ),
        ),
        qubo_matrix=np.array(
            [
                [-1.0, 1.0],
                [1.0, -1.0],
            ],
            dtype=float,
        ),
        offset=0.0,
        asset_symbols=(
            "A",
            "B",
        ),
        target_cardinality=1,
        budget_penalty=1.0,
    )


def brute_force_reference(
    problem: PortfolioQUBOProblem,
) -> tuple[
    tuple[int, ...],
    float,
]:
    best_state = None
    best_energy = float("inf")

    for state in itertools.product(
        (0, 1),
        repeat=problem.variable_count,
    ):
        energy = problem.energy(
            state
        )

        if energy < best_energy:
            best_energy = energy
            best_state = state

    assert best_state is not None

    return (
        tuple(
            int(value)
            for value in best_state
        ),
        float(
            best_energy
        ),
    )


def test_default_config_is_created() -> None:
    config = ClassicalExactSolverConfig()

    assert config.max_variables == 24
    assert config.tie_tolerance == pytest.approx(
        1e-12
    )


def test_config_rejects_zero_max_variables() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="max_variables must be positive",
    ):
        ClassicalExactSolverConfig(
            max_variables=0
        )


def test_config_rejects_negative_max_variables() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="max_variables must be positive",
    ):
        ClassicalExactSolverConfig(
            max_variables=-1
        )


def test_config_rejects_negative_tolerance() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="tie_tolerance cannot be negative",
    ):
        ClassicalExactSolverConfig(
            tie_tolerance=-1.0
        )


def test_config_rejects_nonfinite_tolerance() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="tie_tolerance must be finite",
    ):
        ClassicalExactSolverConfig(
            tie_tolerance=np.nan
        )


def test_solver_uses_default_config() -> None:
    solver = ClassicalExactQUBOSolver()

    assert isinstance(
        solver.config,
        ClassicalExactSolverConfig,
    )


def test_solver_uses_custom_config() -> None:
    config = ClassicalExactSolverConfig(
        max_variables=10,
        tie_tolerance=1e-9,
    )

    solver = ClassicalExactQUBOSolver(
        config=config
    )

    assert solver.config is config


def test_solver_returns_completed_solution() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert (
        solution.status
        == QuantumSolutionStatus.COMPLETED
    )


def test_solver_identifies_itself_as_classical_exact() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert (
        solution.solver_type
        == QuantumSolverType.CLASSICAL_EXACT
    )

    assert (
        solution.backend
        == QuantumExecutionBackend.CLASSICAL
    )


def test_solver_finds_global_optimum() -> None:
    problem = build_problem()

    reference_state, reference_energy = (
        brute_force_reference(
            problem
        )
    )

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert (
        solution.binary_solution
        == reference_state
    )

    assert (
        solution.objective_value
        == pytest.approx(
            reference_energy
        )
    )


def test_simple_problem_has_expected_optimum() -> None:
    problem = build_simple_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert solution.binary_solution == (
        1,
        0,
    )


def test_solution_objective_equals_problem_energy() -> None:
    problem = build_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert (
        solution.objective_value
        == pytest.approx(
            problem.energy(
                solution.binary_solution
            )
        )
    )


def test_solver_enumerates_complete_search_space() -> None:
    problem = build_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert (
        solution.metadata[
            "evaluated_states"
        ]
        == 2 ** problem.variable_count
    )

    assert (
        solution.metadata[
            "search_space_size"
        ]
        == 2 ** problem.variable_count
    )


def test_global_optimum_is_guaranteed_metadata() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert (
        solution.metadata[
            "global_optimum_guaranteed"
        ]
        is True
    )


def test_algorithm_metadata_is_correct() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert (
        solution.metadata[
            "algorithm"
        ]
        == "exhaustive_enumeration"
    )


def test_runtime_is_nonnegative() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert (
        solution.runtime_seconds
        >= 0.0
    )


def test_selected_assets_match_binary_solution() -> None:
    problem = build_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    expected = tuple(
        variable.asset_symbol
        for variable, value
        in zip(
            problem.variables,
            solution.binary_solution,
        )
        if (
            value == 1
            and variable.asset_symbol
            is not None
        )
    )

    assert (
        solution.selected_assets
        == expected
    )


def test_equal_weights_are_assigned_to_selected_assets() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    selected_count = len(
        solution.selected_assets
    )

    assert selected_count > 0

    expected_weight = (
        1.0
        / selected_count
    )

    for symbol in solution.selected_assets:
        assert (
            solution.weights[
                symbol
            ]
            == pytest.approx(
                expected_weight
            )
        )


def test_selected_weights_sum_to_one() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert sum(
        solution.weights.values()
    ) == pytest.approx(
        1.0
    )


def test_unselected_assets_receive_zero_weight() -> None:
    problem = build_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    for symbol, value in zip(
        problem.asset_symbols,
        solution.binary_solution,
    ):
        if value == 0:
            assert (
                solution.weights[
                    symbol
                ]
                == pytest.approx(
                    0.0
                )
            )


def test_cardinality_diagnostic_is_present() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem()
        )
    )

    assert len(
        solution.constraint_diagnostics
    ) == 1

    diagnostic = (
        solution.constraint_diagnostics[
            0
        ]
    )

    assert (
        diagnostic.name
        == "cardinality"
    )


def test_feasible_optimum_satisfies_cardinality() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_problem(
                target_cardinality=2,
                budget_penalty=10.0,
            )
        )
    )

    diagnostic = (
        solution.constraint_diagnostics[
            0
        ]
    )

    assert diagnostic.satisfied is True
    assert diagnostic.violation == pytest.approx(
        0.0
    )
    assert diagnostic.penalty == pytest.approx(
        0.0
    )

    assert solution.feasible is True


def test_cardinality_metadata_contains_selected_count() -> None:
    problem = build_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    diagnostic = (
        solution.constraint_diagnostics[
            0
        ]
    )

    assert (
        diagnostic.metadata[
            "selected_count"
        ]
        == sum(
            solution.binary_solution
        )
    )

    assert (
        diagnostic.metadata[
            "target_cardinality"
        ]
        == problem.target_cardinality
    )


def test_zero_penalty_can_allow_infeasible_optimum() -> None:
    request = (
        PortfolioSelectionQUBORequest(
            asset_symbols=(
                "A",
                "B",
            ),
            expected_returns=np.array(
                [
                    1.0,
                    1.0,
                ]
            ),
            covariance_matrix=np.zeros(
                (
                    2,
                    2,
                )
            ),
            target_cardinality=1,
            risk_aversion=0.0,
            return_weight=1.0,
            budget_penalty=0.0,
        )
    )

    problem = (
        PortfolioQUBOBuilder()
        .build(
            request
        )
    )

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert solution.binary_solution == (
        1,
        1,
    )

    assert solution.feasible is False

    assert (
        solution.total_constraint_violation
        == pytest.approx(
            1.0
        )
    )


def test_solver_rejects_invalid_problem_type() -> None:
    solver = ClassicalExactQUBOSolver()

    with pytest.raises(
        ClassicalSolverError,
        match="PortfolioQUBOProblem",
    ):
        solver.solve(
            "invalid"
        )  # type: ignore[arg-type]


def test_solver_refuses_problem_above_variable_limit() -> None:
    problem = build_problem()

    solver = ClassicalExactQUBOSolver(
        ClassicalExactSolverConfig(
            max_variables=2
        )
    )

    with pytest.raises(
        ClassicalSolverError,
        match="maximum is 2",
    ):
        solver.solve(
            problem
        )


def test_solver_limit_allows_equal_variable_count() -> None:
    problem = build_problem()

    solver = ClassicalExactQUBOSolver(
        ClassicalExactSolverConfig(
            max_variables=3
        )
    )

    solution = solver.solve(
        problem
    )

    assert (
        solution.status
        == QuantumSolutionStatus.COMPLETED
    )


def test_solve_all_optima_returns_tuple() -> None:
    solutions = (
        ClassicalExactQUBOSolver()
        .solve_all_optima(
            build_tied_problem()
        )
    )

    assert isinstance(
        solutions,
        tuple,
    )


def test_solve_all_optima_finds_multiple_optima() -> None:
    solutions = (
        ClassicalExactQUBOSolver()
        .solve_all_optima(
            build_tied_problem()
        )
    )

    states = {
        solution.binary_solution
        for solution
        in solutions
    }

    assert states == {
        (
            1,
            0,
        ),
        (
            0,
            1,
        ),
    }


def test_all_optima_have_identical_objective() -> None:
    solutions = (
        ClassicalExactQUBOSolver()
        .solve_all_optima(
            build_tied_problem()
        )
    )

    values = {
        round(
            solution.objective_value,
            12,
        )
        for solution
        in solutions
    }

    assert len(
        values
    ) == 1


def test_all_optima_report_tie_count() -> None:
    solutions = (
        ClassicalExactQUBOSolver()
        .solve_all_optima(
            build_tied_problem()
        )
    )

    assert len(
        solutions
    ) == 2

    for solution in solutions:
        assert (
            solution.metadata[
                "optimal_tie_count"
            ]
            == 2
        )


def test_solve_returns_deterministic_first_tie() -> None:
    problem = build_tied_problem()

    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            problem
        )
    )

    assert solution.binary_solution == (
        0,
        1,
    )


def test_solve_reports_tie_count() -> None:
    solution = (
        ClassicalExactQUBOSolver()
        .solve(
            build_tied_problem()
        )
    )

    assert (
        solution.metadata[
            "optimal_tie_count"
        ]
        == 2
    )


def test_solve_all_optima_rejects_invalid_problem_type() -> None:
    solver = ClassicalExactQUBOSolver()

    with pytest.raises(
        ClassicalSolverError,
        match="PortfolioQUBOProblem",
    ):
        solver.solve_all_optima(
            "invalid"
        )  # type: ignore[arg-type]


def test_solve_all_optima_respects_variable_limit() -> None:
    solver = ClassicalExactQUBOSolver(
        ClassicalExactSolverConfig(
            max_variables=2
        )
    )

    with pytest.raises(
        ClassicalSolverError,
        match="maximum is 2",
    ):
        solver.solve_all_optima(
            build_problem()
        )


def test_objective_gap_zero_for_identical_values() -> None:
    gap = (
        ClassicalExactQUBOSolver
        .objective_gap(
            candidate_value=-1.5,
            reference_value=-1.5,
        )
    )

    assert gap == pytest.approx(
        0.0
    )


def test_objective_gap_is_absolute_difference() -> None:
    gap = (
        ClassicalExactQUBOSolver
        .objective_gap(
            candidate_value=-1.0,
            reference_value=-1.5,
        )
    )

    assert gap == pytest.approx(
        0.5
    )


def test_objective_gap_rejects_nonfinite_candidate() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="candidate_value must be finite",
    ):
        (
            ClassicalExactQUBOSolver
            .objective_gap(
                candidate_value=np.nan,
                reference_value=1.0,
            )
        )


def test_objective_gap_rejects_nonfinite_reference() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="reference_value must be finite",
    ):
        (
            ClassicalExactQUBOSolver
            .objective_gap(
                candidate_value=1.0,
                reference_value=np.inf,
            )
        )


def test_relative_gap_zero_for_optimum() -> None:
    gap = (
        ClassicalExactQUBOSolver
        .relative_objective_gap(
            candidate_value=-2.0,
            reference_value=-2.0,
        )
    )

    assert gap == pytest.approx(
        0.0
    )


def test_relative_gap_is_correct() -> None:
    gap = (
        ClassicalExactQUBOSolver
        .relative_objective_gap(
            candidate_value=-1.5,
            reference_value=-2.0,
        )
    )

    assert gap == pytest.approx(
        0.25
    )


def test_relative_gap_handles_zero_reference() -> None:
    gap = (
        ClassicalExactQUBOSolver
        .relative_objective_gap(
            candidate_value=0.5,
            reference_value=0.0,
            epsilon=0.1,
        )
    )

    assert gap == pytest.approx(
        5.0
    )


def test_relative_gap_rejects_zero_epsilon() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="epsilon must be positive",
    ):
        (
            ClassicalExactQUBOSolver
            .relative_objective_gap(
                candidate_value=1.0,
                reference_value=0.0,
                epsilon=0.0,
            )
        )


def test_relative_gap_rejects_negative_epsilon() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="epsilon must be positive",
    ):
        (
            ClassicalExactQUBOSolver
            .relative_objective_gap(
                candidate_value=1.0,
                reference_value=0.0,
                epsilon=-1.0,
            )
        )


def test_relative_gap_rejects_nonfinite_epsilon() -> None:
    with pytest.raises(
        ClassicalSolverError,
        match="epsilon must be finite",
    ):
        (
            ClassicalExactQUBOSolver
            .relative_objective_gap(
                candidate_value=1.0,
                reference_value=0.0,
                epsilon=np.nan,
            )
        )


def test_selected_assets_helper() -> None:
    problem = build_problem()

    selected = (
        ClassicalExactQUBOSolver
        ._selected_assets(
            problem=problem,
            binary_solution=(
                1,
                0,
                1,
            ),
        )
    )

    assert selected == (
        "SPY",
        "TLT",
    )


def test_equal_weights_helper() -> None:
    problem = build_problem()

    weights = (
        ClassicalExactQUBOSolver
        ._derive_equal_weights(
            problem=problem,
            binary_solution=(
                1,
                0,
                1,
            ),
        )
    )

    assert weights["SPY"] == pytest.approx(
        0.5
    )

    assert weights["GLD"] == pytest.approx(
        0.0
    )

    assert weights["TLT"] == pytest.approx(
        0.5
    )


def test_equal_weights_helper_handles_empty_selection() -> None:
    problem = build_problem()

    weights = (
        ClassicalExactQUBOSolver
        ._derive_equal_weights(
            problem=problem,
            binary_solution=(
                0,
                0,
                0,
            ),
        )
    )

    assert sum(
        weights.values()
    ) == pytest.approx(
        0.0
    )


def test_constraint_diagnostics_helper() -> None:
    problem = build_problem(
        target_cardinality=2,
        budget_penalty=10.0,
    )

    diagnostics = (
        ClassicalExactQUBOSolver
        ._constraint_diagnostics(
            problem=problem,
            binary_solution=(
                1,
                0,
                0,
            ),
        )
    )

    assert len(
        diagnostics
    ) == 1

    diagnostic = diagnostics[0]

    assert diagnostic.satisfied is False

    assert diagnostic.violation == pytest.approx(
        1.0
    )

    assert diagnostic.penalty == pytest.approx(
        10.0
    )