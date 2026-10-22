"""
Core data models for the v5.0.8 Quantum Portfolio Lab.

These models define the shared representation used by:

- QUBO formulation
- Classical QUBO solvers
- QAOA / Qiskit solvers
- Hybrid classical-quantum experiments
- Benchmarking
- Constraint diagnostics
- Stability analysis

The design intentionally keeps the optimization problem
representation independent from any specific solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


class QuantumPortfolioModelError(ValueError):
    """
    Raised when Quantum Portfolio Lab model data is invalid.
    """


class QuantumSolverType(str, Enum):
    """
    Supported solver families.
    """

    CLASSICAL_EXACT = "classical_exact"
    CLASSICAL_HEURISTIC = "classical_heuristic"
    QAOA = "qaoa"
    HYBRID = "hybrid"


class QuantumExecutionBackend(str, Enum):
    """
    Execution environments for quantum experiments.
    """

    CLASSICAL = "classical"
    SIMULATOR = "simulator"
    HARDWARE = "hardware"


class QuantumSolutionStatus(str, Enum):
    """
    Solver execution status.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class QUBOVariable:
    """
    Binary decision variable used by the QUBO model.

    Example:
        x_NVDA = 1  -> asset selected
        x_NVDA = 0  -> asset not selected
    """

    name: str
    index: int
    asset_symbol: Optional[str] = None
    bit_position: Optional[int] = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise QuantumPortfolioModelError(
                "QUBO variable name cannot be empty."
            )

        if self.index < 0:
            raise QuantumPortfolioModelError(
                "QUBO variable index cannot be negative."
            )

        if (
            self.bit_position is not None
            and self.bit_position < 0
        ):
            raise QuantumPortfolioModelError(
                "bit_position cannot be negative."
            )


@dataclass(frozen=True)
class QUBOConstraintDiagnostic:
    """
    Diagnostic information for one portfolio constraint.
    """

    name: str
    satisfied: bool
    violation: float = 0.0
    penalty: float = 0.0
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise QuantumPortfolioModelError(
                "Constraint diagnostic name cannot be empty."
            )

        if not np.isfinite(self.violation):
            raise QuantumPortfolioModelError(
                "Constraint violation must be finite."
            )

        if self.violation < 0.0:
            raise QuantumPortfolioModelError(
                "Constraint violation cannot be negative."
            )

        if not np.isfinite(self.penalty):
            raise QuantumPortfolioModelError(
                "Constraint penalty must be finite."
            )

        if self.penalty < 0.0:
            raise QuantumPortfolioModelError(
                "Constraint penalty cannot be negative."
            )


@dataclass(frozen=True)
class PortfolioQUBOProblem:
    """
    Solver-independent QUBO representation.

    The objective is represented as:

        minimize:

            x^T Q x + offset

    where x is a binary vector.

    The same QUBO matrix must be supplied to classical,
    QAOA and hybrid solvers to enable fair comparison.
    """

    variables: Tuple[QUBOVariable, ...]
    qubo_matrix: np.ndarray
    offset: float = 0.0

    asset_symbols: Tuple[str, ...] = field(
        default_factory=tuple
    )

    expected_returns: Optional[np.ndarray] = None
    covariance_matrix: Optional[np.ndarray] = None

    risk_aversion: float = 1.0
    return_weight: float = 1.0
    budget_penalty: float = 1.0
    constraint_penalty: float = 1.0

    target_cardinality: Optional[int] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        matrix = np.asarray(
            self.qubo_matrix,
            dtype=float,
        )

        if matrix.ndim != 2:
            raise QuantumPortfolioModelError(
                "qubo_matrix must be two-dimensional."
            )

        if matrix.shape[0] != matrix.shape[1]:
            raise QuantumPortfolioModelError(
                "qubo_matrix must be square."
            )

        if matrix.shape[0] != len(self.variables):
            raise QuantumPortfolioModelError(
                "qubo_matrix dimension must match "
                "the number of QUBO variables."
            )

        if not np.all(np.isfinite(matrix)):
            raise QuantumPortfolioModelError(
                "qubo_matrix must contain only finite values."
            )

        if not np.allclose(
            matrix,
            matrix.T,
            atol=1e-10,
        ):
            raise QuantumPortfolioModelError(
                "qubo_matrix must be symmetric."
            )

        if not np.isfinite(self.offset):
            raise QuantumPortfolioModelError(
                "QUBO offset must be finite."
            )

        indices = [
            variable.index
            for variable in self.variables
        ]

        if len(indices) != len(set(indices)):
            raise QuantumPortfolioModelError(
                "QUBO variable indices must be unique."
            )

        expected_indices = list(
            range(len(self.variables))
        )

        if sorted(indices) != expected_indices:
            raise QuantumPortfolioModelError(
                "QUBO variable indices must be contiguous "
                "and start at zero."
            )

        if self.expected_returns is not None:
            expected_returns = np.asarray(
                self.expected_returns,
                dtype=float,
            )

            if expected_returns.ndim != 1:
                raise QuantumPortfolioModelError(
                    "expected_returns must be one-dimensional."
                )

            if (
                len(self.asset_symbols) > 0
                and expected_returns.shape[0]
                != len(self.asset_symbols)
            ):
                raise QuantumPortfolioModelError(
                    "expected_returns length must match "
                    "asset_symbols."
                )

            if not np.all(
                np.isfinite(expected_returns)
            ):
                raise QuantumPortfolioModelError(
                    "expected_returns must contain "
                    "only finite values."
                )

        if self.covariance_matrix is not None:
            covariance = np.asarray(
                self.covariance_matrix,
                dtype=float,
            )

            if covariance.ndim != 2:
                raise QuantumPortfolioModelError(
                    "covariance_matrix must be two-dimensional."
                )

            if covariance.shape[0] != covariance.shape[1]:
                raise QuantumPortfolioModelError(
                    "covariance_matrix must be square."
                )

            if (
                len(self.asset_symbols) > 0
                and covariance.shape[0]
                != len(self.asset_symbols)
            ):
                raise QuantumPortfolioModelError(
                    "covariance_matrix dimension must match "
                    "asset_symbols."
                )

            if not np.all(
                np.isfinite(covariance)
            ):
                raise QuantumPortfolioModelError(
                    "covariance_matrix must contain "
                    "only finite values."
                )

            if not np.allclose(
                covariance,
                covariance.T,
                atol=1e-10,
            ):
                raise QuantumPortfolioModelError(
                    "covariance_matrix must be symmetric."
                )

        for name, value in (
            ("risk_aversion", self.risk_aversion),
            ("return_weight", self.return_weight),
            ("budget_penalty", self.budget_penalty),
            (
                "constraint_penalty",
                self.constraint_penalty,
            ),
        ):
            if not np.isfinite(value):
                raise QuantumPortfolioModelError(
                    f"{name} must be finite."
                )

            if value < 0.0:
                raise QuantumPortfolioModelError(
                    f"{name} cannot be negative."
                )

        if self.target_cardinality is not None:
            if self.target_cardinality <= 0:
                raise QuantumPortfolioModelError(
                    "target_cardinality must be positive."
                )

            if (
                len(self.asset_symbols) > 0
                and self.target_cardinality
                > len(self.asset_symbols)
            ):
                raise QuantumPortfolioModelError(
                    "target_cardinality cannot exceed "
                    "the number of assets."
                )

        object.__setattr__(
            self,
            "qubo_matrix",
            matrix.copy(),
        )

        if self.expected_returns is not None:
            object.__setattr__(
                self,
                "expected_returns",
                np.asarray(
                    self.expected_returns,
                    dtype=float,
                ).copy(),
            )

        if self.covariance_matrix is not None:
            object.__setattr__(
                self,
                "covariance_matrix",
                np.asarray(
                    self.covariance_matrix,
                    dtype=float,
                ).copy(),
            )

    @property
    def variable_count(self) -> int:
        """
        Number of binary variables in the QUBO.
        """

        return len(self.variables)

    def energy(
        self,
        binary_vector: Sequence[int],
    ) -> float:
        """
        Calculate the QUBO energy of a binary solution.
        """

        vector = np.asarray(
            binary_vector,
            dtype=float,
        )

        if vector.ndim != 1:
            raise QuantumPortfolioModelError(
                "binary_vector must be one-dimensional."
            )

        if vector.shape[0] != self.variable_count:
            raise QuantumPortfolioModelError(
                "binary_vector length must match "
                "the number of QUBO variables."
            )

        if not np.all(
            np.logical_or(
                vector == 0.0,
                vector == 1.0,
            )
        ):
            raise QuantumPortfolioModelError(
                "binary_vector must contain only 0 or 1."
            )

        return float(
            vector
            @ self.qubo_matrix
            @ vector
            + self.offset
        )


@dataclass(frozen=True)
class QuantumPortfolioSolution:
    """
    Standardized result returned by every solver.

    This common representation makes classical,
    QAOA and hybrid solutions directly comparable.
    """

    solver_type: QuantumSolverType
    backend: QuantumExecutionBackend
    status: QuantumSolutionStatus

    binary_solution: Tuple[int, ...]
    objective_value: float

    selected_assets: Tuple[str, ...] = field(
        default_factory=tuple
    )

    weights: Mapping[str, float] = field(
        default_factory=dict
    )

    runtime_seconds: float = 0.0

    constraint_diagnostics: Tuple[
        QUBOConstraintDiagnostic,
        ...
    ] = field(
        default_factory=tuple
    )

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not np.isfinite(
            self.objective_value
        ):
            raise QuantumPortfolioModelError(
                "objective_value must be finite."
            )

        if not np.isfinite(
            self.runtime_seconds
        ):
            raise QuantumPortfolioModelError(
                "runtime_seconds must be finite."
            )

        if self.runtime_seconds < 0.0:
            raise QuantumPortfolioModelError(
                "runtime_seconds cannot be negative."
            )

        if any(
            value not in (0, 1)
            for value in self.binary_solution
        ):
            raise QuantumPortfolioModelError(
                "binary_solution must contain "
                "only 0 or 1."
            )

        for symbol, weight in self.weights.items():
            if not symbol:
                raise QuantumPortfolioModelError(
                    "Weight symbol cannot be empty."
                )

            if not np.isfinite(weight):
                raise QuantumPortfolioModelError(
                    "Portfolio weights must be finite."
                )

    @property
    def feasible(self) -> bool:
        """
        True when all recorded constraints are satisfied.
        """

        return all(
            diagnostic.satisfied
            for diagnostic
            in self.constraint_diagnostics
        )

    @property
    def total_constraint_violation(
        self,
    ) -> float:
        """
        Sum of all recorded constraint violations.
        """

        return float(
            sum(
                diagnostic.violation
                for diagnostic
                in self.constraint_diagnostics
            )
        )


@dataclass(frozen=True)
class QuantumBenchmarkMetrics:
    """
    Standard benchmark metrics used for fair solver comparison.
    """

    solver_type: QuantumSolverType
    backend: QuantumExecutionBackend

    objective_value: float
    runtime_seconds: float

    solution_quality: Optional[float] = None
    optimality_gap: Optional[float] = None

    constraint_violation: float = 0.0
    stability_score: Optional[float] = None

    variable_count: int = 0
    repetition_count: int = 1

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        numeric_values = {
            "objective_value": self.objective_value,
            "runtime_seconds": self.runtime_seconds,
            "constraint_violation": (
                self.constraint_violation
            ),
        }

        for name, value in numeric_values.items():
            if not np.isfinite(value):
                raise QuantumPortfolioModelError(
                    f"{name} must be finite."
                )

        if self.runtime_seconds < 0.0:
            raise QuantumPortfolioModelError(
                "runtime_seconds cannot be negative."
            )

        if self.constraint_violation < 0.0:
            raise QuantumPortfolioModelError(
                "constraint_violation cannot be negative."
            )

        if self.solution_quality is not None:
            if not np.isfinite(
                self.solution_quality
            ):
                raise QuantumPortfolioModelError(
                    "solution_quality must be finite."
                )

        if self.optimality_gap is not None:
            if not np.isfinite(
                self.optimality_gap
            ):
                raise QuantumPortfolioModelError(
                    "optimality_gap must be finite."
                )

            if self.optimality_gap < 0.0:
                raise QuantumPortfolioModelError(
                    "optimality_gap cannot be negative."
                )

        if self.stability_score is not None:
            if not np.isfinite(
                self.stability_score
            ):
                raise QuantumPortfolioModelError(
                    "stability_score must be finite."
                )

            if not (
                0.0
                <= self.stability_score
                <= 1.0
            ):
                raise QuantumPortfolioModelError(
                    "stability_score must be between "
                    "0 and 1."
                )

        if self.variable_count < 0:
            raise QuantumPortfolioModelError(
                "variable_count cannot be negative."
            )

        if self.repetition_count <= 0:
            raise QuantumPortfolioModelError(
                "repetition_count must be positive."
            )


@dataclass(frozen=True)
class QuantumBenchmarkResult:
    """
    Result of comparing multiple portfolio solvers
    on the exact same QUBO problem.
    """

    problem: PortfolioQUBOProblem

    solutions: Tuple[
        QuantumPortfolioSolution,
        ...
    ]

    metrics: Tuple[
        QuantumBenchmarkMetrics,
        ...
    ]

    reference_solver: Optional[
        QuantumSolverType
    ] = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.solutions:
            raise QuantumPortfolioModelError(
                "Benchmark must contain at least "
                "one solution."
            )

        if len(self.solutions) != len(
            self.metrics
        ):
            raise QuantumPortfolioModelError(
                "solutions and metrics must have "
                "the same length."
            )


__all__ = [
    "PortfolioQUBOProblem",
    "QUBOConstraintDiagnostic",
    "QUBOVariable",
    "QuantumBenchmarkMetrics",
    "QuantumBenchmarkResult",
    "QuantumExecutionBackend",
    "QuantumPortfolioModelError",
    "QuantumPortfolioSolution",
    "QuantumSolutionStatus",
    "QuantumSolverType",
]