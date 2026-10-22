"""
Tests for the v5.0.8 Quantum Portfolio Lab Benchmark & Metrics Engine.

Coverage:
- benchmark configuration
- configuration validation
- solver benchmark summaries
- objective-gap metrics
- feasibility and constraint metrics
- optimal hit rate
- stability
- end-to-end Classical vs QAOA vs Hybrid benchmark
- scaling benchmark
- common QuantumBenchmarkResult conversion
"""

from __future__ import annotations

import numpy as np
import pytest

from quantum_portfolio_lab.benchmark_engine import (
    BenchmarkEngineConfig,
    BenchmarkEngineError,
    PortfolioBenchmarkReport,
    QuantumPortfolioBenchmarkEngine,
    ScalingBenchmarkPoint,
    SolverBenchmarkSummary,
)
from quantum_portfolio_lab.classical_solver import (
    ClassicalExactSolverConfig,
)
from quantum_portfolio_lab.hybrid_solver import (
    HybridSolverConfig,
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
    QUBOConstraintDiagnostic,
    QuantumBenchmarkResult,
    QuantumExecutionBackend,
    QuantumPortfolioSolution,
    QuantumSolutionStatus,
    QuantumSolverType,
)


def build_problem(
    asset_count: int = 3,
) -> PortfolioQUBOProblem:
    if asset_count == 2:
        symbols = (
            "SPY",
            "GLD",
        )

        expected_returns = np.array(
            [
                0.08,
                0.04,
            ],
            dtype=float,
        )

        covariance = np.array(
            [
                [0.040, 0.010],
                [0.010, 0.020],
            ],
            dtype=float,
        )

        cardinality = 1

    elif asset_count == 3:
        symbols = (
            "SPY",
            "GLD",
            "TLT",
        )

        expected_returns = np.array(
            [
                0.08,
                0.04,
                0.03,
            ],
            dtype=float,
        )

        covariance = np.array(
            [
                [0.040, 0.010, -0.005],
                [0.010, 0.020, 0.002],
                [-0.005, 0.002, 0.015],
            ],
            dtype=float,
        )

        cardinality = 2

    elif asset_count == 4:
        symbols = (
            "SPY",
            "GLD",
            "TLT",
            "QQQ",
        )

        expected_returns = np.array(
            [
                0.08,
                0.04,
                0.03,
                0.10,
            ],
            dtype=float,
        )

        covariance = np.array(
            [
                [0.040, 0.010, -0.005, 0.025],
                [0.010, 0.020, 0.002, 0.008],
                [-0.005, 0.002, 0.015, -0.004],
                [0.025, 0.008, -0.004, 0.060],
            ],
            dtype=float,
        )

        cardinality = 2

    else:
        raise ValueError(
            "Unsupported test asset_count."
        )

    request = PortfolioSelectionQUBORequest(
        asset_symbols=symbols,
        expected_returns=expected_returns,
        covariance_matrix=covariance,
        target_cardinality=cardinality,
        risk_aversion=1.5,
        return_weight=2.0,
        budget_penalty=10.0,
        constraint_penalty=5.0,
        metadata={
            "dataset": (
                f"benchmark_{asset_count}_assets"
            ),
        },
    )

    return PortfolioQUBOBuilder().build(
        request
    )


def build_solution(
    *,
    solver_type: QuantumSolverType,
    binary_solution: tuple[int, ...],
    objective_value: float,
    runtime_seconds: float = 0.01,
    feasible: bool = True,
) -> QuantumPortfolioSolution:
    if (
        solver_type
        == QuantumSolverType.CLASSICAL_EXACT
    ):
        backend = (
            QuantumExecutionBackend.CLASSICAL
        )
    else:
        backend = (
            QuantumExecutionBackend.SIMULATOR
        )

    symbols = (
        "SPY",
        "GLD",
        "TLT",
        "QQQ",
    )

    selected_assets = tuple(
        symbol
        for symbol, selected
        in zip(
            symbols,
            binary_solution,
        )
        if selected == 1
    )

    if selected_assets:
        weight = (
            1.0
            / len(
                selected_assets
            )
        )

        weights = {
            symbol: weight
            for symbol
            in selected_assets
        }

    else:
        weights = {}

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
    )

    return QuantumPortfolioSolution(
        solver_type=solver_type,
        backend=backend,
        status=QuantumSolutionStatus.COMPLETED,
        binary_solution=binary_solution,
        objective_value=objective_value,
        selected_assets=selected_assets,
        weights=weights,
        runtime_seconds=runtime_seconds,
        constraint_diagnostics=(
            diagnostic,
        ),
        metadata={
            "benchmark_test": True,
        },
    )


# ==============================================================
# Configuration
# ==============================================================


def test_default_config() -> None:
    config = BenchmarkEngineConfig()

    assert config.repetitions == 3

    assert (
        config.objective_tolerance
        == pytest.approx(
            1e-10
        )
    )

    assert isinstance(
        config.classical_config,
        ClassicalExactSolverConfig,
    )

    assert isinstance(
        config.qaoa_config,
        QAOASolverConfig,
    )

    assert config.hybrid_config is None


def test_custom_config() -> None:
    hybrid_config = HybridSolverConfig(
        objective_tolerance=1e-6
    )

    config = BenchmarkEngineConfig(
        repetitions=5,
        objective_tolerance=1e-6,
        classical_config=(
            ClassicalExactSolverConfig(
                max_variables=12
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
        hybrid_config=hybrid_config,
        metadata={
            "experiment": "benchmark",
        },
    )

    assert config.repetitions == 5

    assert (
        config.classical_config.max_variables
        == 12
    )

    assert config.qaoa_config.reps == 2

    assert (
        config.hybrid_config
        is hybrid_config
    )

    assert (
        config.metadata["experiment"]
        == "benchmark"
    )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_invalid_repetitions(
    value,
) -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="repetitions",
    ):
        BenchmarkEngineConfig(
            repetitions=value
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
        BenchmarkEngineError,
        match="objective_tolerance",
    ):
        BenchmarkEngineConfig(
            objective_tolerance=value
        )


def test_invalid_classical_config() -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="classical_config",
    ):
        BenchmarkEngineConfig(
            classical_config="invalid"
        )  # type: ignore[arg-type]


def test_invalid_qaoa_config() -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="qaoa_config",
    ):
        BenchmarkEngineConfig(
            qaoa_config="invalid"
        )  # type: ignore[arg-type]


def test_invalid_hybrid_config() -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="hybrid_config",
    ):
        BenchmarkEngineConfig(
            hybrid_config="invalid"
        )  # type: ignore[arg-type]


# ==============================================================
# Engine construction
# ==============================================================


def test_engine_default_config() -> None:
    engine = (
        QuantumPortfolioBenchmarkEngine()
    )

    assert isinstance(
        engine.config,
        BenchmarkEngineConfig,
    )


def test_engine_custom_config() -> None:
    config = BenchmarkEngineConfig(
        repetitions=1
    )

    engine = (
        QuantumPortfolioBenchmarkEngine(
            config
        )
    )

    assert engine.config is config


def test_engine_rejects_invalid_config() -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="BenchmarkEngineConfig",
    ):
        QuantumPortfolioBenchmarkEngine(
            "invalid"
        )  # type: ignore[arg-type]


def test_engine_rejects_invalid_problem() -> None:
    engine = (
        QuantumPortfolioBenchmarkEngine(
            BenchmarkEngineConfig(
                repetitions=1
            )
        )
    )

    with pytest.raises(
        BenchmarkEngineError,
        match="PortfolioQUBOProblem",
    ):
        engine.benchmark(
            "invalid"
        )  # type: ignore[arg-type]


# ==============================================================
# Summary metrics
# ==============================================================


def test_summary_single_run() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=[
                reference
            ],
            reference_solution=reference,
        )
    )

    assert isinstance(
        summary,
        SolverBenchmarkSummary,
    )

    assert summary.run_count == 1

    assert (
        summary.mean_objective
        == pytest.approx(
            -0.20
        )
    )

    assert (
        summary.mean_absolute_gap
        == pytest.approx(
            0.0
        )
    )

    assert (
        summary.optimal_hit_rate
        == pytest.approx(
            1.0
        )
    )

    assert (
        summary.stability_score
        == pytest.approx(
            1.0
        )
    )


def test_summary_objective_metrics() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                0,
                1,
            ),
            objective_value=-0.15,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                0,
                1,
                1,
            ),
            objective_value=-0.10,
        ),
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert (
        summary.best_objective
        == pytest.approx(
            -0.20
        )
    )

    assert (
        summary.worst_objective
        == pytest.approx(
            -0.10
        )
    )

    assert (
        summary.mean_objective
        == pytest.approx(
            -0.15
        )
    )

    assert (
        summary.mean_absolute_gap
        == pytest.approx(
            0.05
        )
    )

    assert (
        summary.mean_relative_gap
        == pytest.approx(
            0.25
        )
    )


def test_summary_runtime_metrics() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
            runtime_seconds=0.10,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
            runtime_seconds=0.20,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
            runtime_seconds=0.30,
        ),
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert (
        summary.mean_runtime_seconds
        == pytest.approx(
            0.20
        )
    )

    assert (
        summary.runtime_std_seconds
        > 0.0
    )


def test_summary_feasibility_rate() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
            feasible=True,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                0,
                0,
            ),
            objective_value=1.0,
            feasible=False,
        ),
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert (
        summary.feasibility_rate
        == pytest.approx(
            0.5
        )
    )

    assert (
        summary.mean_constraint_violation
        == pytest.approx(
            0.5
        )
    )


def test_summary_optimal_hit_rate() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                0,
                1,
            ),
            objective_value=-0.15,
        ),
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert (
        summary.optimal_hit_rate
        == pytest.approx(
            2.0 / 3.0
        )
    )


def test_summary_stability_identical_runs() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        )
        for _ in range(
            4
        )
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert summary.unique_solution_count == 1

    assert (
        summary.stability_score
        == pytest.approx(
            1.0
        )
    )


def test_summary_stability_all_unique() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                0,
                1,
            ),
            objective_value=-0.15,
        ),
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                0,
                1,
                1,
            ),
            objective_value=-0.10,
        ),
    ]

    summary = (
        QuantumPortfolioBenchmarkEngine
        .summarize_runs(
            runs=runs,
            reference_solution=reference,
        )
    )

    assert summary.unique_solution_count == 3

    assert (
        summary.stability_score
        == pytest.approx(
            0.0
        )
    )


def test_summary_rejects_empty_runs() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    with pytest.raises(
        BenchmarkEngineError,
        match="runs cannot be empty",
    ):
        (
            QuantumPortfolioBenchmarkEngine
            .summarize_runs(
                runs=[],
                reference_solution=reference,
            )
        )


def test_summary_rejects_mixed_solver_types() -> None:
    reference = build_solution(
        solver_type=(
            QuantumSolverType.CLASSICAL_EXACT
        ),
        binary_solution=(
            1,
            1,
            0,
        ),
        objective_value=-0.20,
    )

    runs = [
        reference,
        build_solution(
            solver_type=(
                QuantumSolverType.QAOA
            ),
            binary_solution=(
                1,
                1,
                0,
            ),
            objective_value=-0.20,
        ),
    ]

    with pytest.raises(
        BenchmarkEngineError,
        match="same solver_type",
    ):
        (
            QuantumPortfolioBenchmarkEngine
            .summarize_runs(
                runs=runs,
                reference_solution=reference,
            )
        )


# ==============================================================
# End-to-end benchmark
# ==============================================================


def build_fast_engine(
    repetitions: int = 1,
) -> QuantumPortfolioBenchmarkEngine:
    config = BenchmarkEngineConfig(
        repetitions=repetitions,
        classical_config=(
            ClassicalExactSolverConfig(
                max_variables=10
            )
        ),
        qaoa_config=(
            QAOASolverConfig(
                reps=1,
                shots=128,
                maxiter=5,
                seed=42,
            )
        ),
        metadata={
            "test_mode": True,
        },
    )

    return QuantumPortfolioBenchmarkEngine(
        config
    )


def test_end_to_end_benchmark() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    assert isinstance(
        report,
        PortfolioBenchmarkReport,
    )

    assert report.problem is problem

    assert (
        report.reference_solution.solver_type
        == QuantumSolverType.CLASSICAL_EXACT
    )


def test_end_to_end_run_counts() -> None:
    problem = build_problem()

    engine = build_fast_engine(
        repetitions=2
    )

    report = engine.benchmark(
        problem
    )

    assert len(
        report.classical_runs
    ) == 2

    assert len(
        report.qaoa_runs
    ) == 2

    assert len(
        report.hybrid_runs
    ) == 2


def test_end_to_end_summary_types() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    assert (
        QuantumSolverType.CLASSICAL_EXACT
        in report.summaries
    )

    assert (
        QuantumSolverType.QAOA
        in report.summaries
    )

    assert (
        QuantumSolverType.HYBRID
        in report.summaries
    )


def test_end_to_end_classical_zero_gap() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    classical_summary = (
        report.summaries[
            QuantumSolverType.CLASSICAL_EXACT
        ]
    )

    assert (
        classical_summary.mean_absolute_gap
        == pytest.approx(
            0.0
        )
    )

    assert (
        classical_summary.optimal_hit_rate
        == pytest.approx(
            1.0
        )
    )


def test_end_to_end_metadata() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    assert (
        report.metadata["benchmark_type"]
        == "classical_quantum_hybrid"
    )

    assert (
        report.metadata["reference_solver"]
        == "classical_exact"
    )

    assert (
        report.metadata["simulator_first"]
        is True
    )

    assert (
        report.metadata["identical_dataset"]
        is True
    )

    assert (
        report.metadata["identical_constraints"]
        is True
    )

    assert (
        report.metadata["identical_objective"]
        is True
    )

    assert (
        report.metadata["test_mode"]
        is True
    )


def test_end_to_end_runtime() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    assert report.runtime_seconds >= 0.0


# ==============================================================
# Scaling
# ==============================================================


def test_scaling_benchmark() -> None:
    problems = [
        build_problem(
            2
        ),
        build_problem(
            3
        ),
    ]

    engine = build_fast_engine()

    points = engine.benchmark_scaling(
        problems
    )

    assert len(
        points
    ) == 6

    assert all(
        isinstance(
            point,
            ScalingBenchmarkPoint,
        )
        for point in points
    )


def test_scaling_contains_all_solver_types() -> None:
    problems = [
        build_problem(
            2
        ),
    ]

    engine = build_fast_engine()

    points = engine.benchmark_scaling(
        problems
    )

    solver_types = {
        point.solver_type
        for point in points
    }

    assert solver_types == {
        QuantumSolverType.CLASSICAL_EXACT,
        QuantumSolverType.QAOA,
        QuantumSolverType.HYBRID,
    }


def test_scaling_variable_counts() -> None:
    problems = [
        build_problem(
            2
        ),
        build_problem(
            3
        ),
        build_problem(
            4
        ),
    ]

    engine = build_fast_engine()

    points = engine.benchmark_scaling(
        problems
    )

    counts = {
        point.variable_count
        for point in points
    }

    assert counts == {
        2,
        3,
        4,
    }


def test_scaling_gaps_nonnegative() -> None:
    problems = [
        build_problem(
            2
        ),
    ]

    engine = build_fast_engine()

    points = engine.benchmark_scaling(
        problems
    )

    for point in points:
        assert (
            point.absolute_objective_gap
            >= 0.0
        )

        assert (
            point.relative_objective_gap
            >= 0.0
        )

        assert (
            point.runtime_seconds
            >= 0.0
        )

        assert (
            point.constraint_violation
            >= 0.0
        )


def test_scaling_rejects_empty_problem_list() -> None:
    engine = build_fast_engine()

    with pytest.raises(
        BenchmarkEngineError,
        match="problems cannot be empty",
    ):
        engine.benchmark_scaling(
            []
        )


# ==============================================================
# Common benchmark model conversion
# ==============================================================


def test_convert_to_quantum_benchmark_result() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    result = (
        engine.to_quantum_benchmark_result(
            report=report
        )
    )

    assert isinstance(
        result,
        QuantumBenchmarkResult,
    )

    assert result.problem is problem

    assert (
        result.reference_solver
        == QuantumSolverType.CLASSICAL_EXACT
    )


def test_conversion_contains_all_solutions() -> None:
    problem = build_problem()

    engine = build_fast_engine(
        repetitions=2
    )

    report = engine.benchmark(
        problem
    )

    result = (
        engine.to_quantum_benchmark_result(
            report=report
        )
    )

    assert len(
        result.solutions
    ) == 6


def test_conversion_contains_three_metrics() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    result = (
        engine.to_quantum_benchmark_result(
            report=report
        )
    )

    assert len(
        result.metrics
    ) == 3


def test_conversion_metadata() -> None:
    problem = build_problem()

    engine = build_fast_engine()

    report = engine.benchmark(
        problem
    )

    result = (
        engine.to_quantum_benchmark_result(
            report=report
        )
    )

    assert (
        result.metadata["converted_from"]
        == "PortfolioBenchmarkReport"
    )


def test_conversion_rejects_invalid_report() -> None:
    with pytest.raises(
        BenchmarkEngineError,
        match="PortfolioBenchmarkReport",
    ):
        (
            QuantumPortfolioBenchmarkEngine
            .to_quantum_benchmark_result(
                report="invalid"
            )
        )  # type: ignore[arg-type]