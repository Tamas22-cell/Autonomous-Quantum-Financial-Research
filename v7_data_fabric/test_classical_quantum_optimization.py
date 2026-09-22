import pytest

from .classical_quantum_optimization import (
    ClassicalQuantumOptimizer,
    OptimizationError,
    OptimizationMethod,
    OptimizationProblem,
    OptimizationStatus,
)


def make_problem() -> OptimizationProblem:
    return OptimizationProblem(
        asset_ids=("BTC", "ETH", "SPY"),
        expected_returns=(0.20, 0.14, 0.08),
        risk_scores=(0.40, 0.30, 0.10),
        max_weight=0.70,
    )


def test_version_and_safety_flags() -> None:
    optimizer = ClassicalQuantumOptimizer()

    assert optimizer.VERSION == "v7.0.10"
    assert optimizer.RESEARCH_ONLY is True
    assert optimizer.REAL_MONEY_EXECUTION is False
    assert optimizer.BROKER_CONNECTION is False
    assert optimizer.QUANTUM_HARDWARE_EXECUTION is False


def test_problem_validation() -> None:
    with pytest.raises(OptimizationError):
        OptimizationProblem(
            asset_ids=(),
            expected_returns=(),
            risk_scores=(),
        )

    with pytest.raises(OptimizationError):
        OptimizationProblem(
            asset_ids=("BTC", "BTC"),
            expected_returns=(0.2, 0.1),
            risk_scores=(0.3, 0.2),
        )


def test_classical_optimization_completes() -> None:
    optimizer = ClassicalQuantumOptimizer()

    result = optimizer.optimize(
        make_problem(),
        method=OptimizationMethod.CLASSICAL,
    )

    assert result.status == OptimizationStatus.COMPLETED
    assert result.method == OptimizationMethod.CLASSICAL
    assert result.research_only is True
    assert result.real_money_execution is False
    assert abs(result.invested_weight - 1.0) < 1e-9


def test_weight_limits_are_respected() -> None:
    optimizer = ClassicalQuantumOptimizer()

    result = optimizer.optimize(make_problem())

    assert all(
        weight <= 0.70 + 1e-9
        for weight in result.weights.values()
    )
    assert all(
        weight >= 0.0
        for weight in result.weights.values()
    )


def test_expected_return_and_risk_are_calculated() -> None:
    optimizer = ClassicalQuantumOptimizer()
    result = optimizer.optimize(make_problem())

    assert result.expected_return > 0.0
    assert result.risk_score >= 0.0
    assert result.objective_score == pytest.approx(
        result.expected_return - result.risk_score
    )


def test_quantum_simulation_is_research_only() -> None:
    optimizer = ClassicalQuantumOptimizer()

    result = optimizer.optimize(
        make_problem(),
        method=OptimizationMethod.QUANTUM_SIMULATION,
    )

    assert result.status == OptimizationStatus.COMPLETED
    assert result.method == (
        OptimizationMethod.QUANTUM_SIMULATION
    )
    assert result.real_money_execution is False
    assert optimizer.QUANTUM_HARDWARE_EXECUTION is False


def test_invalid_method_is_rejected() -> None:
    optimizer = ClassicalQuantumOptimizer()

    with pytest.raises(OptimizationError):
        optimizer.optimize(
            make_problem(),
            method="QUANTUM",
        )


def test_history_and_summary() -> None:
    optimizer = ClassicalQuantumOptimizer()

    optimizer.optimize(make_problem())
    optimizer.optimize(
        make_problem(),
        method=OptimizationMethod.QUANTUM_SIMULATION,
    )

    assert len(optimizer.history()) == 2

    summary = optimizer.summary()

    assert summary["version"] == "v7.0.10"
    assert summary["optimization_count"] == 2
    assert summary["has_last_result"] is True
    assert summary["research_only"] is True
    assert summary["automatic_trade_execution"] is False


def test_snapshot_contains_safety_state() -> None:
    optimizer = ClassicalQuantumOptimizer()

    snapshot = optimizer.snapshot()

    assert snapshot["version"] == "v7.0.10"
    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False
    assert snapshot["quantum_hardware_execution"] is False
    assert snapshot["history_count"] == 0