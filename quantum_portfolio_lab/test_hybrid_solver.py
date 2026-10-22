"""
Tests for the v5.0.8 Hybrid Classical-Quantum Portfolio Solver.

Coverage:

- Hybrid solver configuration
- Configuration validation
- Classical/QAOA comparison
- Objective-gap metrics
- Feasibility-aware selection
- Tie-breaking policy
- Hybrid solution construction
- Metadata and audit information
- End-to-end classical + QAOA hybrid execution
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from quantum_portfolio_lab.classical_solver import (
    ClassicalExactSolverConfig,
)
from quantum_portfolio_lab.hybrid_solver import (
    HybridComparison,
    HybridPortfolioSolver,
    HybridSolveResult,
    HybridSolverConfig,
    HybridSolverError,
)
from quantum_portfolio_lab.qaoa_solver import (
    QAOASolverConfig,
)
from quantum_portfolio_lab.qubo_builder import (
    PortfolioQUBOBuilder,
    PortfolioSelectionQUBORequest,
)
from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QuantumExecutionBackend,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


def build_problem() -> PortfolioQUBOProblem:
    request = PortfolioSelectionQUBORequest(
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
        target_cardinality=2,
        risk_aversion=1.5,
        return_weight=2.0,
        budget_penalty=10.0,
        constraint_penalty=5.0,
        metadata={
            "dataset": "hybrid_unit_test",
        },
    )

    return PortfolioQUBOBuilder().build(
        request
    )


def build_solution(
    *,
    solver_type: QuantumSolverType,
    backend: QuantumExecutionBackend,
    binary_solution: tuple[int, ...],
    objective_value: float,
    feasible: bool = True,
) -> QuantumPortfolioSolution:
    selected_assets = tuple(
        symbol
        for symbol, selected
        in zip(
            (
                "SPY",
                "GLD",
                "TLT",
            ),
            binary_solution,
        )
        if selected == 1
    )

    if selected_assets:
        weight = 1.0 / len(
            selected_assets
        )

        weights = {
            symbol: weight
            for symbol
            in selected_assets
        }
    else:
        weights = {}

    from quantum_portfolio_lab.qubo_models import (
        QUBOConstraintDiagnostic,
    )

    diagnostic = QUBOConstraintDiagnostic(
        name="exact_cardinality",
        satisfied=feasible,
        violation=(
            0.0
            if feasible
            else 1.0
        ),
        penalty=(
            0.0
            if feasible
            else 10.0
        ),
        metadata={
            "target_cardinality": 2,
        },
    )

    return QuantumPortfolioSolution(
        solver_type=solver_type,
        backend=backend,
        status=(
            QuantumSolutionStatus.COMPLETED
        ),
        binary_solution=(
            binary_solution
        ),
        objective_value=float(
            objective_value
        ),
        selected_assets=(
            selected_assets
        ),
        weights=weights,
        runtime_seconds=0.01,
        constraint_diagnostics=(
            diagnostic,
        ),
        metadata={
            "test_solution": True,
        },
    )


# ==============================================================
# Configuration
# ==============================================================


def test_default_config() -> None:
    config = HybridSolverConfig()

    assert isinstance(
        config.classical_config,
        ClassicalExactSolverConfig,
    )

    assert isinstance(
        config.qaoa_config,
        QAOASolverConfig,
    )

    assert config.objective_tolerance == pytest.approx(
        1e-10
    )

    assert config.prefer_qaoa_on_tie is False

    assert config.require_feasible_solution is True


def test_custom_config() -> None:
    config = HybridSolverConfig(
        classical_config=(
            ClassicalExactSolverConfig(
                max_variables=10
            )
        ),
        qaoa_config=(
            QAOASolverConfig(
                reps=2,
                shots=512,
                maxiter=20,
                seed=7,
            )
        ),
        objective_tolerance=1e-6,
        prefer_qaoa_on_tie=True,
        require_feasible_solution=False,
        metadata={
            "experiment": "custom",
        },
    )

    assert (
        config.classical_config.max_variables
        == 10
    )

    assert config.qaoa_config.reps == 2
    assert config.qaoa_config.shots == 512

    assert config.objective_tolerance == pytest.approx(
        1e-6
    )

    assert config.prefer_qaoa_on_tie is True

    assert config.require_feasible_solution is False

    assert (
        config.metadata["experiment"]
        == "custom"
    )


@pytest.mark.parametrize(
    "value",
    [
        -1.0,
        np.nan,
        np.inf,
    ],
)
def test_invalid_objective_tolerance(
    value: float,
) -> None:
    with pytest.raises(
        HybridSolverError,
        match="objective_tolerance",
    ):
        HybridSolverConfig(
            objective_tolerance=value
        )


def test_invalid_classical_config() -> None:
    with pytest.raises(
        HybridSolverError,
        match="classical_config",
    ):
        HybridSolverConfig(
            classical_config="invalid"
        )  # type: ignore[arg-type]


def test_invalid_qaoa_config() -> None:
    with pytest.raises(
        HybridSolverError,
        match="qaoa_config",
    ):
        HybridSolverConfig(
            qaoa_config="invalid"
        )  # type: ignore[arg-type]


def test_invalid_prefer_qaoa_on_tie() -> None:
    with pytest.raises(
        HybridSolverError,
        match="prefer_qaoa_on_tie",
    ):
        HybridSolverConfig(
            prefer_qaoa_on_tie=1
        )  # type: ignore[arg-type]


def test_invalid_require_feasible_solution() -> None:
    with pytest.raises(
        HybridSolverError,
        match="require_feasible_solution",
    ):
        HybridSolverConfig(
            require_feasible_solution=1
        )  # type: ignore[arg-type]


# ==============================================================
# Solver construction
# ==============================================================


def test_solver_default_config() -> None:
    solver = HybridPortfolioSolver()

    assert isinstance(
        solver.config,
        HybridSolverConfig,
    )


def test_solver_custom_config() -> None:
    config = HybridSolverConfig(
        objective_tolerance=1e-6
    )

    solver = HybridPortfolioSolver(
        config
    )

    assert solver.config is config


def test_solver_rejects_invalid_config() -> None:
    with pytest.raises(
        HybridSolverError,
        match="HybridSolverConfig",
    ):
        HybridPortfolioSolver(
            "invalid"
        )  # type: ignore[arg-type]


def test_solver_rejects_invalid_problem() -> None:
    solver = HybridPortfolioSolver()

    with pytest.raises(
        HybridSolverError,
        match="PortfolioQUBOProblem",
    ):
        solver.solve(
            "invalid"
        )  # type: ignore[arg-type]


# ==============================================================
# Candidate selection
# ==============================================================


def test_classical_selected_when_objective_is_better() -> None:
    solver = HybridPortfolioSolver()

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            1,
        ),
        objective_value=-0.10,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "classical_exact"


def test_qaoa_selected_when_objective_is_better() -> None:
    solver = HybridPortfolioSolver()

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.10,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            1,
        ),
        objective_value=-0.20,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "qaoa"


def test_classical_selected_on_tie_by_default() -> None:
    solver = HybridPortfolioSolver()

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "classical_exact"


def test_qaoa_selected_on_tie_when_configured() -> None:
    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            prefer_qaoa_on_tie=True
        )
    )

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "qaoa"


def test_feasible_classical_preferred_over_better_infeasible_qaoa() -> None:
    solver = HybridPortfolioSolver()

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.10,
        feasible=True,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            0,
        ),
        objective_value=-1.00,
        feasible=False,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "classical_exact"


def test_infeasible_qaoa_can_win_when_feasibility_not_required() -> None:
    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            require_feasible_solution=False
        )
    )

    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.10,
        feasible=True,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            0,
        ),
        objective_value=-1.00,
        feasible=False,
    )

    selected = solver._select_candidate(
        classical_solution=classical,
        qaoa_solution=qaoa,
    )

    assert selected == "qaoa"


# ==============================================================
# Comparison
# ==============================================================


def test_comparison_objective_gap() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            1,
        ),
        objective_value=-0.15,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    assert isinstance(
        comparison,
        HybridComparison,
    )

    assert (
        comparison.absolute_objective_gap
        == pytest.approx(
            0.05
        )
    )

    assert (
        comparison.relative_objective_gap
        == pytest.approx(
            0.25
        )
    )


def test_comparison_same_binary_solution() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    assert (
        comparison.same_binary_solution
        is True
    )

    assert (
        comparison
        .same_objective_within_tolerance
        is True
    )


def test_comparison_different_binary_solution() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            1,
        ),
        objective_value=-0.15,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    assert (
        comparison.same_binary_solution
        is False
    )


def test_comparison_constraint_violation() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
        feasible=True,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            0,
        ),
        objective_value=-0.15,
        feasible=False,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    assert (
        comparison
        .classical_constraint_violation
        == pytest.approx(
            0.0
        )
    )

    assert (
        comparison
        .qaoa_constraint_violation
        == pytest.approx(
            1.0
        )
    )

    assert comparison.classical_feasible is True
    assert comparison.qaoa_feasible is False


def test_comparison_rejects_invalid_selected_source() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    with pytest.raises(
        HybridSolverError,
        match="selected_source",
    ):
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="invalid",
        )


# ==============================================================
# Hybrid solution construction
# ==============================================================


def test_build_hybrid_solution() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            0,
            1,
        ),
        objective_value=-0.15,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    hybrid = (
        HybridPortfolioSolver
        ._build_hybrid_solution(
            source_solution=classical,
            classical_solution=classical,
            qaoa_solution=qaoa,
            comparison=comparison,
            runtime_seconds=0.10,
        )
    )

    assert (
        hybrid.solver_type
        == QuantumSolverType.HYBRID
    )

    assert (
        hybrid.backend
        == QuantumExecutionBackend.SIMULATOR
    )

    assert (
        hybrid.binary_solution
        == classical.binary_solution
    )

    assert (
        hybrid.objective_value
        == pytest.approx(
            classical.objective_value
        )
    )


def test_hybrid_solution_metadata() -> None:
    classical = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        backend=(
            QuantumExecutionBackend.CLASSICAL
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    qaoa = build_solution(
        solver_type=(
            QuantumSolverType.QAOA
        ),
        backend=(
            QuantumExecutionBackend.SIMULATOR
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    comparison = (
        HybridPortfolioSolver.compare_solutions(
            classical_solution=classical,
            qaoa_solution=qaoa,
            selected_source="classical_exact",
        )
    )

    hybrid = (
        HybridPortfolioSolver
        ._build_hybrid_solution(
            source_solution=classical,
            classical_solution=classical,
            qaoa_solution=qaoa,
            comparison=comparison,
            runtime_seconds=0.10,
        )
    )

    assert (
        hybrid.metadata["algorithm"]
        == "hybrid_classical_quantum"
    )

    assert (
        hybrid.metadata["selected_source"]
        == "classical_exact"
    )

    assert (
        hybrid.metadata["simulator_first"]
        is True
    )


# ==============================================================
# End-to-end hybrid execution
# ==============================================================


def test_end_to_end_hybrid_solve() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            classical_config=(
                ClassicalExactSolverConfig(
                    max_variables=10
                )
            ),
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            ),
        )
    )

    result = solver.solve(
        problem
    )

    assert isinstance(
        result,
        HybridSolveResult,
    )

    assert (
        result.solution.solver_type
        == QuantumSolverType.HYBRID
    )

    assert (
        result.classical_solution.solver_type
        == QuantumSolverType.CLASSICAL_EXACT
    )

    assert (
        result.qaoa_solution.solver_type
        == QuantumSolverType.QAOA
    )


def test_end_to_end_hybrid_objective_is_exact_qubo_energy() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    expected = problem.energy(
        result.solution.binary_solution
    )

    assert (
        result.solution.objective_value
        == pytest.approx(
            expected
        )
    )


def test_end_to_end_classical_is_reference_optimum() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    assert (
        result.classical_solution.objective_value
        <= result.qaoa_solution.objective_value
        + 1e-10
    )


def test_end_to_end_comparison_gap_is_nonnegative() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    assert (
        result.comparison.absolute_objective_gap
        >= 0.0
    )

    assert (
        result.comparison.relative_objective_gap
        >= 0.0
    )


def test_end_to_end_runtime_is_nonnegative() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    assert result.runtime_seconds >= 0.0
    assert result.solution.runtime_seconds >= 0.0


def test_end_to_end_metadata() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            ),
            metadata={
                "experiment": (
                    "hybrid_test"
                ),
            },
        )
    )

    result = solver.solve(
        problem
    )

    assert (
        result.metadata["algorithm"]
        == "hybrid_classical_quantum"
    )

    assert (
        result.metadata["simulator_first"]
        is True
    )

    assert (
        result.metadata["identical_qubo_problem"]
        is True
    )

    assert (
        result.metadata["experiment"]
        == "hybrid_test"
    )


def test_end_to_end_selected_solution_matches_selected_source() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    if (
        result.comparison.selected_source
        == "classical_exact"
    ):
        expected = (
            result.classical_solution
        )
    else:
        expected = (
            result.qaoa_solution
        )

    assert (
        result.solution.binary_solution
        == expected.binary_solution
    )

    assert (
        result.solution.objective_value
        == pytest.approx(
            expected.objective_value
        )
    )


def test_end_to_end_solution_is_completed() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    assert (
        result.solution.status
        == QuantumSolutionStatus.COMPLETED
    )


def test_end_to_end_binary_solution_is_valid() -> None:
    problem = build_problem()

    solver = HybridPortfolioSolver(
        HybridSolverConfig(
            qaoa_config=(
                QAOASolverConfig(
                    reps=1,
                    shots=256,
                    maxiter=10,
                    seed=42,
                )
            )
        )
    )

    result = solver.solve(
        problem
    )

    assert len(
        result.solution.binary_solution
    ) == problem.variable_count

    assert set(
        result.solution.binary_solution
    ).issubset(
        {
            0,
            1,
        }
    )