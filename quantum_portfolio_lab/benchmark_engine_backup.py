"""
Benchmark & Metrics Engine for the v5.0.8 Quantum Portfolio Lab.

The benchmark engine evaluates Classical Exact, QAOA and Hybrid
portfolio solvers on identical QUBO problems.

Measured dimensions:

- solution quality
- absolute objective gap
- relative objective gap
- runtime
- feasibility
- constraint violation
- optimal-solution hit rate
- stability across repeated runs
- scaling across increasing problem sizes

The classical exact solution is used as the reference optimum
for problem sizes where exhaustive enumeration is technically
reasonable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from quantum_portfolio_lab.classical_solver import (
    ClassicalExactQUBOSolver,
    ClassicalExactSolverConfig,
)
from quantum_portfolio_lab.hybrid_solver import (
    HybridPortfolioSolver,
    HybridSolveResult,
    HybridSolverConfig,
)
from quantum_portfolio_lab.qaoa_solver import (
    QAOAPortfolioSolver,
    QAOASolverConfig,
)
from quantum_portfolio_lab.qubo_models import (
    PortfolioQUBOProblem,
    QuantumBenchmarkMetrics,
    QuantumBenchmarkResult,
    QuantumPortfolioModelError,
    QuantumPortfolioSolution,
    QuantumSolverType,
)


class BenchmarkEngineError(
    QuantumPortfolioModelError
):
    """
    Raised when benchmark execution or metric calculation fails.
    """


@dataclass(frozen=True)
class BenchmarkEngineConfig:
    """
    Configuration for the Quantum Portfolio benchmark engine.
    """

    repetitions: int = 3

    objective_tolerance: float = 1e-10

    classical_config: ClassicalExactSolverConfig = field(
        default_factory=ClassicalExactSolverConfig
    )

    qaoa_config: QAOASolverConfig = field(
        default_factory=QAOASolverConfig
    )

    hybrid_config: HybridSolverConfig | None = None

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.repetitions,
                bool,
            )
            or not isinstance(
                self.repetitions,
                (int, np.integer),
            )
            or self.repetitions <= 0
        ):
            raise BenchmarkEngineError(
                "repetitions must be a positive integer."
            )

        try:
            tolerance = float(
                self.objective_tolerance
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkEngineError(
                "objective_tolerance must be a finite "
                "non-negative number."
            ) from exc

        if not np.isfinite(
            tolerance
        ):
            raise BenchmarkEngineError(
                "objective_tolerance must be finite."
            )

        if tolerance < 0.0:
            raise BenchmarkEngineError(
                "objective_tolerance cannot be negative."
            )

        if not isinstance(
            self.classical_config,
            ClassicalExactSolverConfig,
        ):
            raise BenchmarkEngineError(
                "classical_config must be a "
                "ClassicalExactSolverConfig."
            )

        if not isinstance(
            self.qaoa_config,
            QAOASolverConfig,
        ):
            raise BenchmarkEngineError(
                "qaoa_config must be a QAOASolverConfig."
            )

        if (
            self.hybrid_config is not None
            and not isinstance(
                self.hybrid_config,
                HybridSolverConfig,
            )
        ):
            raise BenchmarkEngineError(
                "hybrid_config must be a HybridSolverConfig "
                "or None."
            )

        object.__setattr__(
            self,
            "repetitions",
            int(
                self.repetitions
            ),
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
class SolverBenchmarkSummary:
    """
    Aggregated benchmark statistics for one solver.
    """

    solver_type: QuantumSolverType

    run_count: int

    best_objective: float
    worst_objective: float
    mean_objective: float
    objective_std: float

    mean_absolute_gap: float
    mean_relative_gap: float

    best_absolute_gap: float
    worst_absolute_gap: float

    mean_runtime_seconds: float
    runtime_std_seconds: float

    feasibility_rate: float
    mean_constraint_violation: float

    optimal_hit_rate: float

    unique_solution_count: int

    stability_score: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.solver_type,
            QuantumSolverType,
        ):
            raise BenchmarkEngineError(
                "solver_type must be a QuantumSolverType."
            )

        if (
            isinstance(
                self.run_count,
                bool,
            )
            or not isinstance(
                self.run_count,
                (int, np.integer),
            )
            or self.run_count <= 0
        ):
            raise BenchmarkEngineError(
                "run_count must be a positive integer."
            )

        finite_fields = (
            "best_objective",
            "worst_objective",
            "mean_objective",
            "objective_std",
            "mean_absolute_gap",
            "mean_relative_gap",
            "best_absolute_gap",
            "worst_absolute_gap",
            "mean_runtime_seconds",
            "runtime_std_seconds",
            "feasibility_rate",
            "mean_constraint_violation",
            "optimal_hit_rate",
            "stability_score",
        )

        for field_name in finite_fields:
            value = float(
                getattr(
                    self,
                    field_name,
                )
            )

            if not np.isfinite(
                value
            ):
                raise BenchmarkEngineError(
                    f"{field_name} must be finite."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        nonnegative_fields = (
            "objective_std",
            "mean_absolute_gap",
            "mean_relative_gap",
            "best_absolute_gap",
            "worst_absolute_gap",
            "mean_runtime_seconds",
            "runtime_std_seconds",
            "mean_constraint_violation",
        )

        for field_name in nonnegative_fields:
            if getattr(
                self,
                field_name,
            ) < 0.0:
                raise BenchmarkEngineError(
                    f"{field_name} cannot be negative."
                )

        for field_name in (
            "feasibility_rate",
            "optimal_hit_rate",
            "stability_score",
        ):
            value = getattr(
                self,
                field_name,
            )

            if not 0.0 <= value <= 1.0:
                raise BenchmarkEngineError(
                    f"{field_name} must be between 0 and 1."
                )

        if (
            isinstance(
                self.unique_solution_count,
                bool,
            )
            or not isinstance(
                self.unique_solution_count,
                (int, np.integer),
            )
            or self.unique_solution_count <= 0
        ):
            raise BenchmarkEngineError(
                "unique_solution_count must be positive."
            )

        object.__setattr__(
            self,
            "run_count",
            int(
                self.run_count
            ),
        )

        object.__setattr__(
            self,
            "unique_solution_count",
            int(
                self.unique_solution_count
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


@dataclass(frozen=True)
class ScalingBenchmarkPoint:
    """
    One point in a scaling experiment.
    """

    variable_count: int

    solver_type: QuantumSolverType

    runtime_seconds: float

    objective_value: float

    absolute_objective_gap: float

    relative_objective_gap: float

    feasible: bool

    constraint_violation: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.variable_count,
                bool,
            )
            or not isinstance(
                self.variable_count,
                (int, np.integer),
            )
            or self.variable_count <= 0
        ):
            raise BenchmarkEngineError(
                "variable_count must be positive."
            )

        if not isinstance(
            self.solver_type,
            QuantumSolverType,
        ):
            raise BenchmarkEngineError(
                "solver_type must be a QuantumSolverType."
            )

        for field_name in (
            "runtime_seconds",
            "objective_value",
            "absolute_objective_gap",
            "relative_objective_gap",
            "constraint_violation",
        ):
            value = float(
                getattr(
                    self,
                    field_name,
                )
            )

            if not np.isfinite(
                value
            ):
                raise BenchmarkEngineError(
                    f"{field_name} must be finite."
                )

            object.__setattr__(
                self,
                field_name,
                value,
            )

        for field_name in (
            "runtime_seconds",
            "absolute_objective_gap",
            "relative_objective_gap",
            "constraint_violation",
        ):
            if getattr(
                self,
                field_name,
            ) < 0.0:
                raise BenchmarkEngineError(
                    f"{field_name} cannot be negative."
                )

        object.__setattr__(
            self,
            "variable_count",
            int(
                self.variable_count
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


@dataclass(frozen=True)
class PortfolioBenchmarkReport:
    """
    Complete benchmark report for one QUBO problem.
    """

    problem: PortfolioQUBOProblem

    reference_solution: QuantumPortfolioSolution

    classical_runs: tuple[
        QuantumPortfolioSolution,
        ...
    ]

    qaoa_runs: tuple[
        QuantumPortfolioSolution,
        ...
    ]

    hybrid_runs: tuple[
        QuantumPortfolioSolution,
        ...
    ]

    summaries: Mapping[
        QuantumSolverType,
        SolverBenchmarkSummary,
    ]

    runtime_seconds: float

    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.problem,
            PortfolioQUBOProblem,
        ):
            raise BenchmarkEngineError(
                "problem must be a PortfolioQUBOProblem."
            )

        if not isinstance(
            self.reference_solution,
            QuantumPortfolioSolution,
        ):
            raise BenchmarkEngineError(
                "reference_solution must be a "
                "QuantumPortfolioSolution."
            )

        runtime = float(
            self.runtime_seconds
        )

        if (
            not np.isfinite(
                runtime
            )
            or runtime < 0.0
        ):
            raise BenchmarkEngineError(
                "runtime_seconds must be finite and "
                "non-negative."
            )

        object.__setattr__(
            self,
            "runtime_seconds",
            runtime,
        )

        object.__setattr__(
            self,
            "summaries",
            dict(
                self.summaries
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            dict(
                self.metadata
            ),
        )


class QuantumPortfolioBenchmarkEngine:
    """
    Benchmark engine for Classical Exact, QAOA and Hybrid
    portfolio optimization.
    """

    def __init__(
        self,
        config: BenchmarkEngineConfig | None = None,
    ) -> None:
        if config is None:
            config = BenchmarkEngineConfig()

        if not isinstance(
            config,
            BenchmarkEngineConfig,
        ):
            raise BenchmarkEngineError(
                "config must be a BenchmarkEngineConfig."
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

        if config.hybrid_config is None:
            hybrid_config = HybridSolverConfig(
                classical_config=(
                    config.classical_config
                ),
                qaoa_config=(
                    config.qaoa_config
                ),
                objective_tolerance=(
                    config.objective_tolerance
                ),
            )
        else:
            hybrid_config = (
                config.hybrid_config
            )

        self.hybrid_solver = (
            HybridPortfolioSolver(
                hybrid_config
            )
        )

    def benchmark(
        self,
        problem: PortfolioQUBOProblem,
    ) -> PortfolioBenchmarkReport:
        """
        Benchmark all supported solvers on the identical problem.
        """

        self._validate_problem(
            problem
        )

        start_time = perf_counter()

        reference_solution = (
            self.classical_solver.solve(
                problem
            )
        )

        classical_runs: list[
            QuantumPortfolioSolution
        ] = []

        qaoa_runs: list[
            QuantumPortfolioSolution
        ] = []

        hybrid_runs: list[
            QuantumPortfolioSolution
        ] = []

        for _ in range(
            self.config.repetitions
        ):
            classical_runs.append(
                self.classical_solver.solve(
                    problem
                )
            )

            qaoa_runs.append(
                self.qaoa_solver.solve(
                    problem
                )
            )

            hybrid_result = (
                self.hybrid_solver.solve(
                    problem
                )
            )

            hybrid_runs.append(
                hybrid_result.solution
            )

        summaries = {
            QuantumSolverType.CLASSICAL_EXACT: (
                self.summarize_runs(
                    runs=classical_runs,
                    reference_solution=(
                        reference_solution
                    ),
                    objective_tolerance=(
                        self.config.objective_tolerance
                    ),
                )
            ),
            QuantumSolverType.QAOA: (
                self.summarize_runs(
                    runs=qaoa_runs,
                    reference_solution=(
                        reference_solution
                    ),
                    objective_tolerance=(
                        self.config.objective_tolerance
                    ),
                )
            ),
            QuantumSolverType.HYBRID: (
                self.summarize_runs(
                    runs=hybrid_runs,
                    reference_solution=(
                        reference_solution
                    ),
                    objective_tolerance=(
                        self.config.objective_tolerance
                    ),
                )
            ),
        }

        runtime_seconds = (
            perf_counter()
            - start_time
        )

        metadata = dict(
            self.config.metadata
        )

        metadata.update(
            {
                "benchmark_type": (
                    "classical_quantum_hybrid"
                ),
                "reference_solver": (
                    "classical_exact"
                ),
                "repetitions": (
                    self.config.repetitions
                ),
                "variable_count": (
                    problem.variable_count
                ),
                "simulator_first": True,
                "identical_dataset": True,
                "identical_constraints": True,
                "identical_objective": True,
            }
        )

        return PortfolioBenchmarkReport(
            problem=problem,
            reference_solution=(
                reference_solution
            ),
            classical_runs=tuple(
                classical_runs
            ),
            qaoa_runs=tuple(
                qaoa_runs
            ),
            hybrid_runs=tuple(
                hybrid_runs
            ),
            summaries=summaries,
            runtime_seconds=runtime_seconds,
            metadata=metadata,
        )

    @staticmethod
    def summarize_runs(
        *,
        runs: Sequence[
            QuantumPortfolioSolution
        ],
        reference_solution: QuantumPortfolioSolution,
        objective_tolerance: float = 1e-10,
    ) -> SolverBenchmarkSummary:
        """
        Aggregate repeated solver runs against a classical
        reference optimum.
        """

        if not runs:
            raise BenchmarkEngineError(
                "runs cannot be empty."
            )

        if not isinstance(
            reference_solution,
            QuantumPortfolioSolution,
        ):
            raise BenchmarkEngineError(
                "reference_solution must be a "
                "QuantumPortfolioSolution."
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
            raise BenchmarkEngineError(
                "objective_tolerance must be finite "
                "and non-negative."
            )

        for solution in runs:
            if not isinstance(
                solution,
                QuantumPortfolioSolution,
            ):
                raise BenchmarkEngineError(
                    "Every run must be a "
                    "QuantumPortfolioSolution."
                )

        solver_types = {
            solution.solver_type
            for solution in runs
        }

        if len(
            solver_types
        ) != 1:
            raise BenchmarkEngineError(
                "All runs must use the same solver_type."
            )

        solver_type = next(
            iter(
                solver_types
            )
        )

        reference_value = float(
            reference_solution.objective_value
        )

        objectives = [
            float(
                solution.objective_value
            )
            for solution in runs
        ]

        runtimes = [
            float(
                solution.runtime_seconds
            )
            for solution in runs
        ]

        violations = [
            float(
                solution.total_constraint_violation
            )
            for solution in runs
        ]

        feasibility_values = [
            1.0
            if solution.feasible
            else 0.0
            for solution in runs
        ]

        absolute_gaps = [
            abs(
                value
                - reference_value
            )
            for value in objectives
        ]

        denominator = max(
            abs(
                reference_value
            ),
            1e-12,
        )

        relative_gaps = [
            gap
            / denominator
            for gap in absolute_gaps
        ]

        optimal_hits = [
            1.0
            if gap <= tolerance
            else 0.0
            for gap in absolute_gaps
        ]

        unique_solutions = {
            tuple(
                int(value)
                for value
                in solution.binary_solution
            )
            for solution in runs
        }

        run_count = len(
            runs
        )

        unique_solution_count = len(
            unique_solutions
        )

        if run_count == 1:
            stability_score = 1.0
        else:
            stability_score = (
                1.0
                - (
                    unique_solution_count
                    - 1
                )
                / (
                    run_count
                    - 1
                )
            )

            stability_score = max(
                0.0,
                min(
                    1.0,
                    stability_score,
                ),
            )

        return SolverBenchmarkSummary(
            solver_type=solver_type,
            run_count=run_count,
            best_objective=min(
                objectives
            ),
            worst_objective=max(
                objectives
            ),
            mean_objective=mean(
                objectives
            ),
            objective_std=(
                pstdev(
                    objectives
                )
                if run_count > 1
                else 0.0
            ),
            mean_absolute_gap=mean(
                absolute_gaps
            ),
            mean_relative_gap=mean(
                relative_gaps
            ),
            best_absolute_gap=min(
                absolute_gaps
            ),
            worst_absolute_gap=max(
                absolute_gaps
            ),
            mean_runtime_seconds=mean(
                runtimes
            ),
            runtime_std_seconds=(
                pstdev(
                    runtimes
                )
                if run_count > 1
                else 0.0
            ),
            feasibility_rate=mean(
                feasibility_values
            ),
            mean_constraint_violation=mean(
                violations
            ),
            optimal_hit_rate=mean(
                optimal_hits
            ),
            unique_solution_count=(
                unique_solution_count
            ),
            stability_score=(
                stability_score
            ),
            metadata={
                "reference_solver": (
                    reference_solution
                    .solver_type.value
                ),
                "reference_objective": (
                    reference_value
                ),
                "objective_tolerance": (
                    tolerance
                ),
            },
        )

    def benchmark_scaling(
        self,
        problems: Sequence[
            PortfolioQUBOProblem
        ],
    ) -> tuple[
        ScalingBenchmarkPoint,
        ...
    ]:
        """
        Benchmark solver behavior as problem size increases.

        Each problem is evaluated using Classical Exact, QAOA
        and Hybrid solvers against its own classical reference.
        """

        if not problems:
            raise BenchmarkEngineError(
                "problems cannot be empty."
            )

        points: list[
            ScalingBenchmarkPoint
        ] = []

        for problem in problems:
            self._validate_problem(
                problem
            )

            reference = (
                self.classical_solver.solve(
                    problem
                )
            )

            solutions: list[
                QuantumPortfolioSolution
            ] = []

            classical_solution = (
                self.classical_solver.solve(
                    problem
                )
            )

            solutions.append(
                classical_solution
            )

            qaoa_solution = (
                self.qaoa_solver.solve(
                    problem
                )
            )

            solutions.append(
                qaoa_solution
            )

            hybrid_result: HybridSolveResult = (
                self.hybrid_solver.solve(
                    problem
                )
            )

            solutions.append(
                hybrid_result.solution
            )

            for solution in solutions:
                absolute_gap = abs(
                    float(
                        solution.objective_value
                    )
                    - float(
                        reference.objective_value
                    )
                )

                denominator = max(
                    abs(
                        float(
                            reference.objective_value
                        )
                    ),
                    1e-12,
                )

                relative_gap = (
                    absolute_gap
                    / denominator
                )

                points.append(
                    ScalingBenchmarkPoint(
                        variable_count=(
                            problem.variable_count
                        ),
                        solver_type=(
                            solution.solver_type
                        ),
                        runtime_seconds=(
                            solution.runtime_seconds
                        ),
                        objective_value=(
                            solution.objective_value
                        ),
                        absolute_objective_gap=(
                            absolute_gap
                        ),
                        relative_objective_gap=(
                            relative_gap
                        ),
                        feasible=(
                            solution.feasible
                        ),
                        constraint_violation=(
                            solution
                            .total_constraint_violation
                        ),
                        metadata={
                            "reference_objective": (
                                reference
                                .objective_value
                            ),
                            "reference_solver": (
                                "classical_exact"
                            ),
                        },
                    )
                )

        return tuple(
            points
        )

    @staticmethod
    def to_quantum_benchmark_result(
        *,
        report: PortfolioBenchmarkReport,
    ) -> QuantumBenchmarkResult:
        """
        Convert the richer benchmark report into the common
        QuantumBenchmarkResult model used by the Quantum
        Portfolio Lab.
        """

        if not isinstance(
            report,
            PortfolioBenchmarkReport,
        ):
            raise BenchmarkEngineError(
                "report must be a "
                "PortfolioBenchmarkReport."
            )

        solutions: list[
            QuantumPortfolioSolution
        ] = []

        solutions.extend(
            report.classical_runs
        )

        solutions.extend(
            report.qaoa_runs
        )

        solutions.extend(
            report.hybrid_runs
        )

        metrics: list[
            QuantumBenchmarkMetrics
        ] = []

        for solver_type, summary in (
            report.summaries.items()
        ):
            metrics.append(
                QuantumBenchmarkMetrics(
                    solver_type=solver_type,
                    solution_quality=(
                        1.0
                        / (
                            1.0
                            + summary
                            .mean_absolute_gap
                        )
                    ),
                    runtime_seconds=(
                        summary
                        .mean_runtime_seconds
                    ),
                    constraint_violation=(
                        summary
                        .mean_constraint_violation
                    ),
                    stability=(
                        summary.stability_score
                    ),
                    metadata={
                        "mean_objective": (
                            summary.mean_objective
                        ),
                        "mean_absolute_gap": (
                            summary
                            .mean_absolute_gap
                        ),
                        "mean_relative_gap": (
                            summary
                            .mean_relative_gap
                        ),
                        "feasibility_rate": (
                            summary
                            .feasibility_rate
                        ),
                        "optimal_hit_rate": (
                            summary
                            .optimal_hit_rate
                        ),
                    },
                )
            )

        metadata = dict(
            report.metadata
        )

        metadata[
            "converted_from"
        ] = "PortfolioBenchmarkReport"

        return QuantumBenchmarkResult(
            problem=report.problem,
            solutions=tuple(
                solutions
            ),
            metrics=tuple(
                metrics
            ),
            reference_solver=(
                QuantumSolverType.CLASSICAL_EXACT
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
            raise BenchmarkEngineError(
                "problem must be a PortfolioQUBOProblem."
            )

        if problem.variable_count <= 0:
            raise BenchmarkEngineError(
                "problem must contain at least one variable."
            )

        if not np.all(
            np.isfinite(
                problem.qubo_matrix
            )
        ):
            raise BenchmarkEngineError(
                "problem QUBO matrix must be finite."
            )

        if not np.isfinite(
            float(
                problem.offset
            )
        ):
            raise BenchmarkEngineError(
                "problem offset must be finite."
            )


__all__ = [
    "BenchmarkEngineConfig",
    "BenchmarkEngineError",
    "PortfolioBenchmarkReport",
    "QuantumPortfolioBenchmarkEngine",
    "ScalingBenchmarkPoint",
    "SolverBenchmarkSummary",
]