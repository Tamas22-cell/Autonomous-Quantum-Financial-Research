"""
Hybrid Classical-Quantum Solver for the v5.0.8 Quantum Portfolio Lab.

The hybrid solver combines:

1. QAOA candidate generation
2. Classical exact benchmarking
3. Deterministic candidate comparison
4. Objective-gap measurement
5. Constraint-violation comparison
6. Hybrid solution selection
7. Complete audit metadata

The classical exact solver acts as the reference solver for small
simulator-first research problems. QAOA operates on the identical
PortfolioQUBOProblem, ensuring that datasets, constraints and objective
functions remain directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from quantum_portfolio_lab.classical_solver import (
    ClassicalExactQUBOSolver,
    ClassicalExactSolverConfig,
)
from quantum_portfolio_lab.qaoa_solver import (
    QAOAPortfolioSolver,
    QAOASolverConfig,
)
from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QuantumExecutionBackend,
    QuantumPortfolioModelError,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


class HybridSolverError(QuantumPortfolioModelError):
    """
    Raised when hybrid classical-quantum optimization cannot
    be executed safely.
    """


@dataclass(frozen=True)
class HybridSolverConfig:
    """
    Configuration for the hybrid classical-quantum solver.

    Parameters
    ----------
    classical_config:
        Configuration used by the exact classical reference solver.

    qaoa_config:
        Configuration used by the QAOA candidate solver.

    objective_tolerance:
        Numerical tolerance used when comparing objective values.

    prefer_qaoa_on_tie:
        If True, a QAOA candidate is selected when its objective is
        equal to the classical candidate within objective_tolerance.

        If False, the deterministic classical candidate is selected.

    require_feasible_solution:
        If True, feasible candidates are always preferred over
        infeasible candidates.

    metadata:
        Optional user-defined metadata copied into the hybrid result.
    """

    classical_config: ClassicalExactSolverConfig = field(
        default_factory=ClassicalExactSolverConfig
    )
    qaoa_config: QAOASolverConfig = field(
        default_factory=QAOASolverConfig
    )
    objective_tolerance: float = 1e-10
    prefer_qaoa_on_tie: bool = False
    require_feasible_solution: bool = True
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.classical_config,
            ClassicalExactSolverConfig,
        ):
            raise HybridSolverError(
                "classical_config must be a "
                "ClassicalExactSolverConfig."
            )

        if not isinstance(
            self.qaoa_config,
            QAOASolverConfig,
        ):
            raise HybridSolverError(
                "qaoa_config must be a QAOASolverConfig."
            )

        try:
            tolerance = float(
                self.objective_tolerance
            )
        except (TypeError, ValueError) as exc:
            raise HybridSolverError(
                "objective_tolerance must be a finite "
                "non-negative number."
            ) from exc

        if not np.isfinite(
            tolerance
        ):
            raise HybridSolverError(
                "objective_tolerance must be finite."
            )

        if tolerance < 0.0:
            raise HybridSolverError(
                "objective_tolerance cannot be negative."
            )

        if not isinstance(
            self.prefer_qaoa_on_tie,
            bool,
        ):
            raise HybridSolverError(
                "prefer_qaoa_on_tie must be a bool."
            )

        if not isinstance(
            self.require_feasible_solution,
            bool,
        ):
            raise HybridSolverError(
                "require_feasible_solution must be a bool."
            )

        object.__setattr__(
            self,
            "objective_tolerance",
            tolerance,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


@dataclass(frozen=True)
class HybridComparison:
    """
    Audit record comparing the classical and QAOA candidates.
    """

    classical_objective: float
    qaoa_objective: float

    absolute_objective_gap: float
    relative_objective_gap: float

    classical_constraint_violation: float
    qaoa_constraint_violation: float

    classical_feasible: bool
    qaoa_feasible: bool

    same_binary_solution: bool
    same_objective_within_tolerance: bool

    selected_source: str

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        numeric_fields = (
            "classical_objective",
            "qaoa_objective",
            "absolute_objective_gap",
            "relative_objective_gap",
            "classical_constraint_violation",
            "qaoa_constraint_violation",
        )

        for field_name in numeric_fields:
            value = float(
                getattr(
                    self,
                    field_name,
                )
            )

            if not np.isfinite(
                value
            ):
                raise HybridSolverError(
                    f"{field_name} must be finite."
                )

            if (
                field_name
                in {
                    "absolute_objective_gap",
                    "relative_objective_gap",
                    "classical_constraint_violation",
                    "qaoa_constraint_violation",
                }
                and value < 0.0
            ):
                raise HybridSolverError(
                    f"{field_name} cannot be negative."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        if self.selected_source not in {
            "classical_exact",
            "qaoa",
        }:
            raise HybridSolverError(
                "selected_source must be either "
                "'classical_exact' or 'qaoa'."
            )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


@dataclass(frozen=True)
class HybridSolveResult:
    """
    Complete result of a hybrid classical-quantum experiment.
    """

    solution: QuantumPortfolioSolution

    classical_solution: QuantumPortfolioSolution
    qaoa_solution: QuantumPortfolioSolution

    comparison: HybridComparison

    runtime_seconds: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.solution,
            QuantumPortfolioSolution,
        ):
            raise HybridSolverError(
                "solution must be a QuantumPortfolioSolution."
            )

        if not isinstance(
            self.classical_solution,
            QuantumPortfolioSolution,
        ):
            raise HybridSolverError(
                "classical_solution must be a "
                "QuantumPortfolioSolution."
            )

        if not isinstance(
            self.qaoa_solution,
            QuantumPortfolioSolution,
        ):
            raise HybridSolverError(
                "qaoa_solution must be a "
                "QuantumPortfolioSolution."
            )

        if not isinstance(
            self.comparison,
            HybridComparison,
        ):
            raise HybridSolverError(
                "comparison must be a HybridComparison."
            )

        runtime = float(
            self.runtime_seconds
        )

        if not np.isfinite(
            runtime
        ):
            raise HybridSolverError(
                "runtime_seconds must be finite."
            )

        if runtime < 0.0:
            raise HybridSolverError(
                "runtime_seconds cannot be negative."
            )

        object.__setattr__(
            self,
            "runtime_seconds",
            runtime,
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


class HybridPortfolioSolver:
    """
    Hybrid classical-quantum portfolio solver.

    Workflow
    --------
    PortfolioQUBOProblem
        ->
    Classical Exact Reference
        +
    QAOA Candidate
        ->
    Candidate Comparison
        ->
    Hybrid Selection
        ->
    Audited QuantumPortfolioSolution
    """

    def __init__(
        self,
        config: HybridSolverConfig | None = None,
    ) -> None:
        if config is None:
            config = HybridSolverConfig()

        if not isinstance(
            config,
            HybridSolverConfig,
        ):
            raise HybridSolverError(
                "config must be a HybridSolverConfig."
            )

        self.config = config

        self.classical_solver = (
            ClassicalExactQUBOSolver(
                config.classical_config
            )
        )

        self.qaoa_solver = (
            QAOAPortfolioSolver(
                config.qaoa_config
            )
        )

    def solve(
        self,
        problem: PortfolioQUBOProblem,
    ) -> HybridSolveResult:
        """
        Execute the complete hybrid experiment.
        """

        self._validate_problem(
            problem
        )

        start_time = perf_counter()

        classical_solution = (
            self.classical_solver.solve(
                problem
            )
        )

        qaoa_solution = (
            self.qaoa_solver.solve(
                problem
            )
        )

        selected_source = (
            self._select_candidate(
                classical_solution=classical_solution,
                qaoa_solution=qaoa_solution,
            )
        )

        comparison = self.compare_solutions(
            classical_solution=classical_solution,
            qaoa_solution=qaoa_solution,
            selected_source=selected_source,
            objective_tolerance=(
                self.config.objective_tolerance
            ),
        )

        if selected_source == "qaoa":
            source_solution = qaoa_solution
        else:
            source_solution = classical_solution

        runtime_seconds = (
            perf_counter()
            - start_time
        )

        hybrid_solution = (
            self._build_hybrid_solution(
                source_solution=source_solution,
                classical_solution=classical_solution,
                qaoa_solution=qaoa_solution,
                comparison=comparison,
                runtime_seconds=runtime_seconds,
            )
        )

        metadata = dict(
            self.config.metadata
        )

        metadata.update(
            {
                "algorithm": (
                    "hybrid_classical_quantum"
                ),
                "classical_solver": (
                    "exhaustive_enumeration"
                ),
                "quantum_solver": (
                    "qaoa"
                ),
                "selected_source": (
                    selected_source
                ),
                "simulator_first": True,
                "identical_qubo_problem": True,
                "objective_tolerance": (
                    self.config.objective_tolerance
                ),
                "prefer_qaoa_on_tie": (
                    self.config.prefer_qaoa_on_tie
                ),
                "require_feasible_solution": (
                    self.config.require_feasible_solution
                ),
            }
        )

        return HybridSolveResult(
            solution=hybrid_solution,
            classical_solution=classical_solution,
            qaoa_solution=qaoa_solution,
            comparison=comparison,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    def _select_candidate(
        self,
        *,
        classical_solution: QuantumPortfolioSolution,
        qaoa_solution: QuantumPortfolioSolution,
    ) -> str:
        """
        Select the candidate used by the hybrid result.

        Selection priority:

        1. Completed status
        2. Feasibility, if configured
        3. Objective value
        4. Configured deterministic tie-break
        """

        classical_completed = (
            classical_solution.status
            == QuantumSolutionStatus.COMPLETED
        )

        qaoa_completed = (
            qaoa_solution.status
            == QuantumSolutionStatus.COMPLETED
        )

        if (
            classical_completed
            and not qaoa_completed
        ):
            return "classical_exact"

        if (
            qaoa_completed
            and not classical_completed
        ):
            return "qaoa"

        if (
            not classical_completed
            and not qaoa_completed
        ):
            raise HybridSolverError(
                "Neither classical nor QAOA solver "
                "returned a completed solution."
            )

        if self.config.require_feasible_solution:
            if (
                classical_solution.feasible
                and not qaoa_solution.feasible
            ):
                return "classical_exact"

            if (
                qaoa_solution.feasible
                and not classical_solution.feasible
            ):
                return "qaoa"

        classical_value = float(
            classical_solution.objective_value
        )

        qaoa_value = float(
            qaoa_solution.objective_value
        )

        difference = (
            qaoa_value
            - classical_value
        )

        tolerance = (
            self.config.objective_tolerance
        )

        if difference < -tolerance:
            return "qaoa"

        if difference > tolerance:
            return "classical_exact"

        if self.config.prefer_qaoa_on_tie:
            return "qaoa"

        return "classical_exact"

    @staticmethod
    def compare_solutions(
        *,
        classical_solution: QuantumPortfolioSolution,
        qaoa_solution: QuantumPortfolioSolution,
        selected_source: str,
        objective_tolerance: float = 1e-10,
    ) -> HybridComparison:
        """
        Compare classical and QAOA solutions using the same
        QUBO objective.
        """

        if not isinstance(
            classical_solution,
            QuantumPortfolioSolution,
        ):
            raise HybridSolverError(
                "classical_solution must be a "
                "QuantumPortfolioSolution."
            )

        if not isinstance(
            qaoa_solution,
            QuantumPortfolioSolution,
        ):
            raise HybridSolverError(
                "qaoa_solution must be a "
                "QuantumPortfolioSolution."
            )

        if selected_source not in {
            "classical_exact",
            "qaoa",
        }:
            raise HybridSolverError(
                "selected_source must be either "
                "'classical_exact' or 'qaoa'."
            )

        tolerance = float(
            objective_tolerance
        )

        if (
            not np.isfinite(
                tolerance
            )
            or tolerance < 0.0
        ):
            raise HybridSolverError(
                "objective_tolerance must be a finite "
                "non-negative number."
            )

        classical_value = float(
            classical_solution.objective_value
        )

        qaoa_value = float(
            qaoa_solution.objective_value
        )

        if not np.isfinite(
            classical_value
        ):
            raise HybridSolverError(
                "classical objective must be finite."
            )

        if not np.isfinite(
            qaoa_value
        ):
            raise HybridSolverError(
                "QAOA objective must be finite."
            )

        absolute_gap = abs(
            qaoa_value
            - classical_value
        )

        denominator = max(
            abs(
                classical_value
            ),
            1e-12,
        )

        relative_gap = (
            absolute_gap
            / denominator
        )

        classical_violation = float(
            classical_solution
            .total_constraint_violation
        )

        qaoa_violation = float(
            qaoa_solution
            .total_constraint_violation
        )

        same_binary_solution = (
            tuple(
                int(value)
                for value
                in classical_solution.binary_solution
            )
            ==
            tuple(
                int(value)
                for value
                in qaoa_solution.binary_solution
            )
        )

        same_objective = (
            absolute_gap
            <= tolerance
        )

        return HybridComparison(
            classical_objective=(
                classical_value
            ),
            qaoa_objective=(
                qaoa_value
            ),
            absolute_objective_gap=(
                absolute_gap
            ),
            relative_objective_gap=(
                relative_gap
            ),
            classical_constraint_violation=(
                classical_violation
            ),
            qaoa_constraint_violation=(
                qaoa_violation
            ),
            classical_feasible=(
                classical_solution.feasible
            ),
            qaoa_feasible=(
                qaoa_solution.feasible
            ),
            same_binary_solution=(
                same_binary_solution
            ),
            same_objective_within_tolerance=(
                same_objective
            ),
            selected_source=(
                selected_source
            ),
            metadata={
                "reference_solver": (
                    "classical_exact"
                ),
                "candidate_solver": (
                    "qaoa"
                ),
                "objective_tolerance": (
                    tolerance
                ),
            },
        )

    @staticmethod
    def _build_hybrid_solution(
        *,
        source_solution: QuantumPortfolioSolution,
        classical_solution: QuantumPortfolioSolution,
        qaoa_solution: QuantumPortfolioSolution,
        comparison: HybridComparison,
        runtime_seconds: float,
    ) -> QuantumPortfolioSolution:
        """
        Convert the selected candidate into a HYBRID solution
        while preserving the selected candidate's portfolio.
        """

        metadata = dict(
            source_solution.metadata
        )

        metadata.update(
            {
                "algorithm": (
                    "hybrid_classical_quantum"
                ),
                "selected_source": (
                    comparison.selected_source
                ),
                "classical_objective": (
                    classical_solution.objective_value
                ),
                "qaoa_objective": (
                    qaoa_solution.objective_value
                ),
                "absolute_objective_gap": (
                    comparison.absolute_objective_gap
                ),
                "relative_objective_gap": (
                    comparison.relative_objective_gap
                ),
                "same_binary_solution": (
                    comparison.same_binary_solution
                ),
                "same_objective_within_tolerance": (
                    comparison
                    .same_objective_within_tolerance
                ),
                "classical_feasible": (
                    comparison.classical_feasible
                ),
                "qaoa_feasible": (
                    comparison.qaoa_feasible
                ),
                "simulator_first": True,
            }
        )

        return QuantumPortfolioSolution(
            solver_type=(
                QuantumSolverType.HYBRID
            ),
            backend=(
                QuantumExecutionBackend.SIMULATOR
            ),
            status=(
                source_solution.status
            ),
            binary_solution=tuple(
                int(value)
                for value
                in source_solution.binary_solution
            ),
            objective_value=float(
                source_solution.objective_value
            ),
            selected_assets=tuple(
                source_solution.selected_assets
            ),
            weights=dict(
                source_solution.weights
            ),
            runtime_seconds=float(
                runtime_seconds
            ),
            constraint_diagnostics=tuple(
                source_solution
                .constraint_diagnostics
            ),
            metadata=metadata,
        )

    @staticmethod
    def _validate_problem(
        problem: PortfolioQUBOProblem,
    ) -> None:
        if not isinstance(
            problem,
            PortfolioQUBOProblem,
        ):
            raise HybridSolverError(
                "problem must be a PortfolioQUBOProblem."
            )

        if problem.variable_count <= 0:
            raise HybridSolverError(
                "problem must contain at least one "
                "QUBO variable."
            )

        if not np.all(
            np.isfinite(
                problem.qubo_matrix
            )
        ):
            raise HybridSolverError(
                "problem QUBO matrix must contain "
                "finite values."
            )

        if not np.isfinite(
            float(
                problem.offset
            )
        ):
            raise HybridSolverError(
                "problem offset must be finite."
            )


__all__ = [
    "HybridComparison",
    "HybridPortfolioSolver",
    "HybridSolveResult",
    "HybridSolverConfig",
    "HybridSolverError",
]