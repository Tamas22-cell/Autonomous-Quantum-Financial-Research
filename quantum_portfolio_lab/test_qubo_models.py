"""
Tests for the v5.0.8 Quantum Portfolio Lab core models.

Coverage:

- QUBO variables
- QUBO problem validation
- QUBO energy calculation
- constraint diagnostics
- standardized solver solutions
- benchmark metrics
- benchmark result validation
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QUBOConstraintDiagnostic,
    QUBOVariable,
    QuantumBenchmarkMetrics,
    QuantumBenchmarkResult,
    QuantumExecutionBackend,
    QuantumPortfolioModelError,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


def build_variables() -> tuple[QUBOVariable, ...]:
    return (
        QUBOVariable(
            name="x_SPY",
            index=0,
            asset_symbol="SPY",
        ),
        QUBOVariable(
            name="x_GLD",
            index=1,
            asset_symbol="GLD",
        ),
    )


def build_problem() -> PortfolioQUBOProblem:
    return PortfolioQUBOProblem(
        variables=build_variables(),
        qubo_matrix=np.array(
            [
                [-1.0, 0.25],
                [0.25, -0.50],
            ],
            dtype=float,
        ),
        offset=0.10,
        asset_symbols=("SPY", "GLD"),
        expected_returns=np.array(
            [0.08, 0.04],
            dtype=float,
        ),
        covariance_matrix=np.array(
            [
                [0.04, 0.01],
                [0.01, 0.02],
            ],
            dtype=float,
        ),
        risk_aversion=1.0,
        return_weight=1.0,
        budget_penalty=2.0,
        constraint_penalty=3.0,
        target_cardinality=1,
    )


def build_solution() -> QuantumPortfolioSolution:
    return QuantumPortfolioSolution(
        solver_type=QuantumSolverType.CLASSICAL_EXACT,
        backend=QuantumExecutionBackend.CLASSICAL,
        status=QuantumSolutionStatus.COMPLETED,
        binary_solution=(1, 0),
        objective_value=-0.90,
        selected_assets=("SPY",),
        weights={"SPY": 1.0, "GLD": 0.0},
        runtime_seconds=0.01,
        constraint_diagnostics=(
            QUBOConstraintDiagnostic(
                name="cardinality",
                satisfied=True,
                violation=0.0,
                penalty=0.0,
            ),
        ),
    )


def build_metrics() -> QuantumBenchmarkMetrics:
    return QuantumBenchmarkMetrics(
        solver_type=QuantumSolverType.CLASSICAL_EXACT,
        backend=QuantumExecutionBackend.CLASSICAL,
        objective_value=-0.90,
        runtime_seconds=0.01,
        solution_quality=1.0,
        optimality_gap=0.0,
        constraint_violation=0.0,
        stability_score=1.0,
        variable_count=2,
        repetition_count=1,
    )


def test_qubo_variable_is_created() -> None:
    variable = QUBOVariable(
        name="x_NVDA",
        index=0,
        asset_symbol="NVDA",
    )

    assert variable.name == "x_NVDA"
    assert variable.index == 0
    assert variable.asset_symbol == "NVDA"


def test_qubo_variable_supports_bit_position() -> None:
    variable = QUBOVariable(
        name="w_NVDA_0",
        index=0,
        asset_symbol="NVDA",
        bit_position=0,
    )

    assert variable.bit_position == 0


def test_empty_variable_name_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="name cannot be empty",
    ):
        QUBOVariable(
            name="",
            index=0,
        )


def test_negative_variable_index_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="index cannot be negative",
    ):
        QUBOVariable(
            name="x_SPY",
            index=-1,
        )


def test_negative_bit_position_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="bit_position cannot be negative",
    ):
        QUBOVariable(
            name="x_SPY",
            index=0,
            bit_position=-1,
        )


def test_constraint_diagnostic_is_created() -> None:
    diagnostic = QUBOConstraintDiagnostic(
        name="budget",
        satisfied=True,
    )

    assert diagnostic.name == "budget"
    assert diagnostic.satisfied is True
    assert diagnostic.violation == pytest.approx(0.0)


def test_negative_constraint_violation_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="violation cannot be negative",
    ):
        QUBOConstraintDiagnostic(
            name="budget",
            satisfied=False,
            violation=-0.1,
        )


def test_negative_constraint_penalty_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="penalty cannot be negative",
    ):
        QUBOConstraintDiagnostic(
            name="budget",
            satisfied=False,
            penalty=-1.0,
        )


def test_problem_is_created() -> None:
    problem = build_problem()

    assert problem.variable_count == 2
    assert problem.asset_symbols == (
        "SPY",
        "GLD",
    )


def test_qubo_matrix_is_copied() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    problem = PortfolioQUBOProblem(
        variables=build_variables(),
        qubo_matrix=matrix,
    )

    matrix[0, 0] = 999.0

    assert problem.qubo_matrix[0, 0] == pytest.approx(
        1.0
    )


def test_qubo_matrix_must_be_two_dimensional() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="two-dimensional",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.array(
                [1.0, 2.0]
            ),
        )


def test_qubo_matrix_must_be_square() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="square",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.array(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            ),
        )


def test_qubo_matrix_dimension_matches_variables() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="dimension must match",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(3),
        )


def test_qubo_matrix_must_be_symmetric() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="symmetric",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.array(
                [
                    [1.0, 2.0],
                    [0.0, 1.0],
                ]
            ),
        )


def test_qubo_matrix_must_be_finite() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="finite",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.array(
                [
                    [1.0, np.nan],
                    [np.nan, 1.0],
                ]
            ),
        )


def test_variable_indices_must_be_unique() -> None:
    variables = (
        QUBOVariable(
            name="x_SPY",
            index=0,
        ),
        QUBOVariable(
            name="x_GLD",
            index=0,
        ),
    )

    with pytest.raises(
        QuantumPortfolioModelError,
        match="indices must be unique",
    ):
        PortfolioQUBOProblem(
            variables=variables,
            qubo_matrix=np.eye(2),
        )


def test_variable_indices_must_be_contiguous() -> None:
    variables = (
        QUBOVariable(
            name="x_SPY",
            index=0,
        ),
        QUBOVariable(
            name="x_GLD",
            index=2,
        ),
    )

    with pytest.raises(
        QuantumPortfolioModelError,
        match="contiguous",
    ):
        PortfolioQUBOProblem(
            variables=variables,
            qubo_matrix=np.eye(2),
        )


def test_expected_returns_length_matches_assets() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="expected_returns length",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            asset_symbols=("SPY", "GLD"),
            expected_returns=np.array(
                [0.10]
            ),
        )


def test_expected_returns_must_be_finite() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="expected_returns",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            asset_symbols=("SPY", "GLD"),
            expected_returns=np.array(
                [0.10, np.nan]
            ),
        )


def test_covariance_dimension_matches_assets() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="dimension must match",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            asset_symbols=("SPY", "GLD"),
            covariance_matrix=np.eye(3),
        )


def test_covariance_must_be_symmetric() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="symmetric",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            asset_symbols=("SPY", "GLD"),
            covariance_matrix=np.array(
                [
                    [1.0, 2.0],
                    [0.0, 1.0],
                ]
            ),
        )


def test_negative_risk_aversion_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="risk_aversion",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            risk_aversion=-1.0,
        )


def test_negative_return_weight_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="return_weight",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            return_weight=-1.0,
        )


def test_invalid_target_cardinality_raises_error() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="target_cardinality must be positive",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            target_cardinality=0,
        )


def test_target_cardinality_cannot_exceed_assets() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="cannot exceed",
    ):
        PortfolioQUBOProblem(
            variables=build_variables(),
            qubo_matrix=np.eye(2),
            asset_symbols=("SPY", "GLD"),
            target_cardinality=3,
        )


def test_energy_zero_vector_equals_offset() -> None:
    problem = build_problem()

    assert problem.energy(
        [0, 0]
    ) == pytest.approx(
        problem.offset
    )


def test_energy_single_selected_asset() -> None:
    problem = build_problem()

    expected = (
        problem.qubo_matrix[0, 0]
        + problem.offset
    )

    assert problem.energy(
        [1, 0]
    ) == pytest.approx(
        expected
    )


def test_energy_multiple_selected_assets() -> None:
    problem = build_problem()

    vector = np.array(
        [1.0, 1.0]
    )

    expected = float(
        vector
        @ problem.qubo_matrix
        @ vector
        + problem.offset
    )

    assert problem.energy(
        [1, 1]
    ) == pytest.approx(
        expected
    )


def test_energy_rejects_wrong_vector_length() -> None:
    problem = build_problem()

    with pytest.raises(
        QuantumPortfolioModelError,
        match="length must match",
    ):
        problem.energy(
            [1]
        )


def test_energy_rejects_nonbinary_vector() -> None:
    problem = build_problem()

    with pytest.raises(
        QuantumPortfolioModelError,
        match="only 0 or 1",
    ):
        problem.energy(
            [1, 2]
        )


def test_solution_is_created() -> None:
    solution = build_solution()

    assert (
        solution.solver_type
        == QuantumSolverType.CLASSICAL_EXACT
    )

    assert (
        solution.status
        == QuantumSolutionStatus.COMPLETED
    )


def test_solution_is_feasible() -> None:
    solution = build_solution()

    assert solution.feasible is True


def test_solution_detects_constraint_violation() -> None:
    solution = QuantumPortfolioSolution(
        solver_type=QuantumSolverType.QAOA,
        backend=QuantumExecutionBackend.SIMULATOR,
        status=QuantumSolutionStatus.COMPLETED,
        binary_solution=(1, 1),
        objective_value=-0.50,
        runtime_seconds=0.10,
        constraint_diagnostics=(
            QUBOConstraintDiagnostic(
                name="cardinality",
                satisfied=False,
                violation=1.0,
                penalty=5.0,
            ),
        ),
    )

    assert solution.feasible is False

    assert (
        solution.total_constraint_violation
        == pytest.approx(1.0)
    )


def test_solution_rejects_nonbinary_solution() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="only 0 or 1",
    ):
        QuantumPortfolioSolution(
            solver_type=QuantumSolverType.QAOA,
            backend=QuantumExecutionBackend.SIMULATOR,
            status=QuantumSolutionStatus.COMPLETED,
            binary_solution=(1, 2),
            objective_value=0.0,
        )


def test_solution_rejects_negative_runtime() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="runtime_seconds",
    ):
        QuantumPortfolioSolution(
            solver_type=QuantumSolverType.QAOA,
            backend=QuantumExecutionBackend.SIMULATOR,
            status=QuantumSolutionStatus.COMPLETED,
            binary_solution=(1, 0),
            objective_value=0.0,
            runtime_seconds=-1.0,
        )


def test_benchmark_metrics_are_created() -> None:
    metrics = build_metrics()

    assert metrics.variable_count == 2
    assert metrics.repetition_count == 1


def test_benchmark_metrics_reject_negative_gap() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="optimality_gap",
    ):
        QuantumBenchmarkMetrics(
            solver_type=QuantumSolverType.QAOA,
            backend=QuantumExecutionBackend.SIMULATOR,
            objective_value=0.0,
            runtime_seconds=1.0,
            optimality_gap=-0.1,
        )


def test_benchmark_metrics_reject_invalid_stability() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="stability_score",
    ):
        QuantumBenchmarkMetrics(
            solver_type=QuantumSolverType.QAOA,
            backend=QuantumExecutionBackend.SIMULATOR,
            objective_value=0.0,
            runtime_seconds=1.0,
            stability_score=1.5,
        )


def test_benchmark_metrics_reject_zero_repetitions() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="repetition_count",
    ):
        QuantumBenchmarkMetrics(
            solver_type=QuantumSolverType.QAOA,
            backend=QuantumExecutionBackend.SIMULATOR,
            objective_value=0.0,
            runtime_seconds=1.0,
            repetition_count=0,
        )


def test_benchmark_result_is_created() -> None:
    result = QuantumBenchmarkResult(
        problem=build_problem(),
        solutions=(
            build_solution(),
        ),
        metrics=(
            build_metrics(),
        ),
        reference_solver=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
    )

    assert len(
        result.solutions
    ) == 1

    assert len(
        result.metrics
    ) == 1


def test_benchmark_requires_solution() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="at least one solution",
    ):
        QuantumBenchmarkResult(
            problem=build_problem(),
            solutions=(),
            metrics=(),
        )


def test_benchmark_solution_metric_counts_must_match() -> None:
    with pytest.raises(
        QuantumPortfolioModelError,
        match="same length",
    ):
        QuantumBenchmarkResult(
            problem=build_problem(),
            solutions=(
                build_solution(),
            ),
            metrics=(),
        )