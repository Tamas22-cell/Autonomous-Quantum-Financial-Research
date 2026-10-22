"""
Classical QUBO solvers for the v5.0.8 Quantum Portfolio Lab.

This module provides the classical reference layer used for
fair comparison with QAOA and hybrid quantum experiments.

The exact solver exhaustively evaluates every binary state and
therefore returns the global optimum for sufficiently small
QUBO problems.

The exact classical solution is intended to serve as the
reference solution for metrics such as:

- solution quality
- optimality gap
- constraint violations
- stability
- QAOA approximation quality

Exhaustive enumeration scales as O(2^n), so the exact solver
must only be used for problem sizes where exhaustive search is
computationally reasonable.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .qubo_models import (
    PortfolioQUBOProblem,
    QUBOConstraintDiagnostic,
    QuantumExecutionBackend,
    QuantumPortfolioModelError,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


class ClassicalSolverError(QuantumPortfolioModelError):
    """
    Raised when classical QUBO optimization fails.
    """


@dataclass(frozen=True)
class ClassicalExactSolverConfig:
    """
    Configuration for exhaustive classical QUBO optimization.

    max_variables protects the system from accidentally
    attempting an impractically large 2^n enumeration.
    """

    max_variables: int = 24
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.max_variables <= 0:
            raise ClassicalSolverError(
                "max_variables must be positive."
            )

        if not np.isfinite(
            self.tie_tolerance
        ):
            raise ClassicalSolverError(
                "tie_tolerance must be finite."
            )

        if self.tie_tolerance < 0.0:
            raise ClassicalSolverError(
                "tie_tolerance cannot be negative."
            )


class ClassicalExactQUBOSolver:
    """
    Exhaustive exact solver for binary QUBO problems.

    Every state in {0, 1}^n is evaluated.

    This guarantees the global QUBO optimum and therefore
    provides the reference baseline for quantum experiments.
    """

    def __init__(
        self,
        config: Optional[
            ClassicalExactSolverConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or ClassicalExactSolverConfig()
        )

    def solve(
        self,
        problem: PortfolioQUBOProblem,
    ) -> QuantumPortfolioSolution:
        """
        Solve a QUBO problem exactly through exhaustive search.
        """

        if not isinstance(
            problem,
            PortfolioQUBOProblem,
        ):
            raise ClassicalSolverError(
                "problem must be a PortfolioQUBOProblem."
            )

        variable_count = (
            problem.variable_count
        )

        if variable_count == 0:
            raise ClassicalSolverError(
                "QUBO problem must contain at least "
                "one variable."
            )

        if (
            variable_count
            > self.config.max_variables
        ):
            raise ClassicalSolverError(
                "Exact classical enumeration refused: "
                f"problem has {variable_count} variables, "
                f"maximum is "
                f"{self.config.max_variables}."
            )

        start_time = time.perf_counter()

        best_state: Optional[
            tuple[int, ...]
        ] = None

        best_energy = float("inf")

        evaluated_states = 0
        tie_count = 0

        for state in itertools.product(
            (0, 1),
            repeat=variable_count,
        ):
            energy = problem.energy(
                state
            )

            evaluated_states += 1

            if (
                energy
                < best_energy
                - self.config.tie_tolerance
            ):
                best_energy = energy
                best_state = tuple(
                    int(value)
                    for value in state
                )
                tie_count = 1

            elif (
                best_state is not None
                and abs(
                    energy
                    - best_energy
                )
                <= self.config.tie_tolerance
            ):
                tie_count += 1

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        if best_state is None:
            raise ClassicalSolverError(
                "Exact solver failed to produce "
                "a solution."
            )

        selected_assets = (
            self._selected_assets(
                problem=problem,
                binary_solution=best_state,
            )
        )

        weights = self._derive_equal_weights(
            problem=problem,
            binary_solution=best_state,
        )

        diagnostics = (
            self._constraint_diagnostics(
                problem=problem,
                binary_solution=best_state,
            )
        )

        metadata = {
            "algorithm": (
                "exhaustive_enumeration"
            ),
            "global_optimum_guaranteed": True,
            "variable_count": variable_count,
            "evaluated_states": (
                evaluated_states
            ),
            "search_space_size": (
                2 ** variable_count
            ),
            "optimal_tie_count": tie_count,
            "target_cardinality": (
                problem.target_cardinality
            ),
        }

        return QuantumPortfolioSolution(
            solver_type=(
                QuantumSolverType.CLASSICAL_EXACT
            ),
            backend=(
                QuantumExecutionBackend.CLASSICAL
            ),
            status=(
                QuantumSolutionStatus.COMPLETED
            ),
            binary_solution=best_state,
            objective_value=float(
                best_energy
            ),
            selected_assets=selected_assets,
            weights=weights,
            runtime_seconds=float(
                runtime_seconds
            ),
            constraint_diagnostics=(
                diagnostics
            ),
            metadata=metadata,
        )

    def solve_all_optima(
        self,
        problem: PortfolioQUBOProblem,
    ) -> tuple[
        QuantumPortfolioSolution,
        ...,
    ]:
        """
        Return every globally optimal state within the
        configured numerical tie tolerance.

        This is useful for stability analysis because multiple
        portfolios may have identical QUBO energy.
        """

        if not isinstance(
            problem,
            PortfolioQUBOProblem,
        ):
            raise ClassicalSolverError(
                "problem must be a PortfolioQUBOProblem."
            )

        variable_count = (
            problem.variable_count
        )

        if variable_count == 0:
            raise ClassicalSolverError(
                "QUBO problem must contain at least "
                "one variable."
            )

        if (
            variable_count
            > self.config.max_variables
        ):
            raise ClassicalSolverError(
                "Exact classical enumeration refused: "
                f"problem has {variable_count} variables, "
                f"maximum is "
                f"{self.config.max_variables}."
            )

        start_time = time.perf_counter()

        energies: list[
            tuple[
                tuple[int, ...],
                float,
            ]
        ] = []

        best_energy = float("inf")

        for state in itertools.product(
            (0, 1),
            repeat=variable_count,
        ):
            binary_state = tuple(
                int(value)
                for value in state
            )

            energy = float(
                problem.energy(
                    binary_state
                )
            )

            energies.append(
                (
                    binary_state,
                    energy,
                )
            )

            if energy < best_energy:
                best_energy = energy

        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        optimal_states = [
            state
            for state, energy
            in energies
            if abs(
                energy
                - best_energy
            )
            <= self.config.tie_tolerance
        ]

        results: list[
            QuantumPortfolioSolution
        ] = []

        for state in optimal_states:
            selected_assets = (
                self._selected_assets(
                    problem=problem,
                    binary_solution=state,
                )
            )

            weights = (
                self._derive_equal_weights(
                    problem=problem,
                    binary_solution=state,
                )
            )

            diagnostics = (
                self._constraint_diagnostics(
                    problem=problem,
                    binary_solution=state,
                )
            )

            results.append(
                QuantumPortfolioSolution(
                    solver_type=(
                        QuantumSolverType.CLASSICAL_EXACT
                    ),
                    backend=(
                        QuantumExecutionBackend.CLASSICAL
                    ),
                    status=(
                        QuantumSolutionStatus.COMPLETED
                    ),
                    binary_solution=state,
                    objective_value=float(
                        best_energy
                    ),
                    selected_assets=(
                        selected_assets
                    ),
                    weights=weights,
                    runtime_seconds=float(
                        runtime_seconds
                    ),
                    constraint_diagnostics=(
                        diagnostics
                    ),
                    metadata={
                        "algorithm": (
                            "exhaustive_enumeration"
                        ),
                        "global_optimum_guaranteed": (
                            True
                        ),
                        "variable_count": (
                            variable_count
                        ),
                        "evaluated_states": (
                            2 ** variable_count
                        ),
                        "search_space_size": (
                            2 ** variable_count
                        ),
                        "optimal_tie_count": (
                            len(
                                optimal_states
                            )
                        ),
                    },
                )
            )

        return tuple(
            results
        )

    @staticmethod
    def _selected_assets(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> tuple[str, ...]:
        """
        Convert selected binary variables into asset symbols.
        """

        selected: list[str] = []

        for variable, value in zip(
            problem.variables,
            binary_solution,
        ):
            if value != 1:
                continue

            if variable.asset_symbol is None:
                continue

            selected.append(
                variable.asset_symbol
            )

        return tuple(
            selected
        )

    @staticmethod
    def _derive_equal_weights(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> dict[str, float]:
        """
        Derive equal weights across selected assets.

        v5.0.8 currently formulates binary portfolio
        selection. Equal weights provide a deterministic
        portfolio representation for selected assets.

        Discrete allocation bits can later extend this
        representation without changing the solver API.
        """

        selected_assets = (
            ClassicalExactQUBOSolver
            ._selected_assets(
                problem=problem,
                binary_solution=(
                    binary_solution
                ),
            )
        )

        weights = {
            symbol: 0.0
            for symbol
            in problem.asset_symbols
        }

        if not selected_assets:
            return weights

        equal_weight = (
            1.0
            / len(
                selected_assets
            )
        )

        for symbol in selected_assets:
            weights[symbol] = float(
                equal_weight
            )

        return weights

    @staticmethod
    def _constraint_diagnostics(
        *,
        problem: PortfolioQUBOProblem,
        binary_solution: Sequence[int],
    ) -> tuple[
        QUBOConstraintDiagnostic,
        ...,
    ]:
        """
        Evaluate constraints represented by the current
        portfolio-selection QUBO.
        """

        diagnostics: list[
            QUBOConstraintDiagnostic
        ] = []

        if (
            problem.target_cardinality
            is not None
        ):
            selected_count = int(
                sum(
                    int(value)
                    for value
                    in binary_solution
                )
            )

            difference = (
                selected_count
                - problem.target_cardinality
            )

            violation = float(
                abs(
                    difference
                )
            )

            penalty = float(
                problem.budget_penalty
                * difference
                * difference
            )

            diagnostics.append(
                QUBOConstraintDiagnostic(
                    name="cardinality",
                    satisfied=(
                        violation == 0.0
                    ),
                    violation=violation,
                    penalty=penalty,
                    metadata={
                        "selected_count": (
                            selected_count
                        ),
                        "target_cardinality": (
                            problem.target_cardinality
                        ),
                    },
                )
            )

        return tuple(
            diagnostics
        )

    @staticmethod
    def objective_gap(
        *,
        candidate_value: float,
        reference_value: float,
    ) -> float:
        """
        Absolute objective gap for minimization problems.

        A globally optimal candidate has gap 0.
        """

        if not np.isfinite(
            candidate_value
        ):
            raise ClassicalSolverError(
                "candidate_value must be finite."
            )

        if not np.isfinite(
            reference_value
        ):
            raise ClassicalSolverError(
                "reference_value must be finite."
            )

        return float(
            abs(
                candidate_value
                - reference_value
            )
        )

    @staticmethod
    def relative_objective_gap(
        *,
        candidate_value: float,
        reference_value: float,
        epsilon: float = 1e-12,
    ) -> float:
        """
        Relative objective gap.

        Useful later for comparing QAOA and hybrid solutions
        with the exact classical reference.
        """

        if not np.isfinite(
            candidate_value
        ):
            raise ClassicalSolverError(
                "candidate_value must be finite."
            )

        if not np.isfinite(
            reference_value
        ):
            raise ClassicalSolverError(
                "reference_value must be finite."
            )

        if not np.isfinite(
            epsilon
        ):
            raise ClassicalSolverError(
                "epsilon must be finite."
            )

        if epsilon <= 0.0:
            raise ClassicalSolverError(
                "epsilon must be positive."
            )

        denominator = max(
            abs(
                reference_value
            ),
            epsilon,
        )

        return float(
            abs(
                candidate_value
                - reference_value
            )
            / denominator
        )


__all__ = [
    "ClassicalExactQUBOSolver",
    "ClassicalExactSolverConfig",
    "ClassicalSolverError",
]