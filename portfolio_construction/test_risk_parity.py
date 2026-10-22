"""
Tests for the v5.0.7 Risk Parity Portfolio Construction Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.models import (
    AssetDefinition,
    ConstraintType,
    CovarianceMatrix,
    ExpectedReturnEstimate,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationStatus,
    PortfolioAllocation,
    PortfolioConstraint,
)

from portfolio_construction.risk_parity import (
    RiskParityError,
    RiskParityOptimizer,
)


# ==============================================================
# Helpers
# ==============================================================


def build_assets() -> list[AssetDefinition]:
    return [
        AssetDefinition(
            symbol="SPY",
            asset_class="equity",
            sector="broad_market",
        ),
        AssetDefinition(
            symbol="NVDA",
            asset_class="equity",
            sector="technology",
        ),
        AssetDefinition(
            symbol="GLD",
            asset_class="commodity",
            sector="gold",
        ),
    ]


def build_expected_returns() -> list[ExpectedReturnEstimate]:
    return [
        ExpectedReturnEstimate(
            symbol="SPY",
            expected_return=0.08,
        ),
        ExpectedReturnEstimate(
            symbol="NVDA",
            expected_return=0.15,
        ),
        ExpectedReturnEstimate(
            symbol="GLD",
            expected_return=0.05,
        ),
    ]


def build_covariance() -> CovarianceMatrix:
    return CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [
                0.0400,
                0.0300,
                0.0050,
            ],
            [
                0.0300,
                0.0900,
                0.0020,
            ],
            [
                0.0050,
                0.0020,
                0.0225,
            ],
        ],
        annualized=True,
    )


def build_request(
    *,
    constraints: list[PortfolioConstraint] | None = None,
    current_allocation: PortfolioAllocation | None = None,
    method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
) -> OptimizationRequest:
    return OptimizationRequest(
        request_id="risk_parity_test",
        method=method,
        assets=build_assets(),
        expected_returns=build_expected_returns(),
        covariance=build_covariance(),
        constraints=constraints or [],
        current_allocation=current_allocation,
        risk_free_rate=0.02,
    )


# ==============================================================
# Basic optimization
# ==============================================================


def test_risk_parity_completes_successfully() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.status == OptimizationStatus.COMPLETED
    assert result.allocation is not None
    assert result.risk_metrics is not None
    assert result.objective_value is not None
    assert result.iterations is not None
    assert result.iterations >= 1


def test_weights_sum_to_one() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.allocation is not None

    assert sum(
        result.allocation.weights.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_portfolio_is_long_only() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.allocation is not None

    for weight in result.allocation.weights.values():
        assert weight >= 0.0


# ==============================================================
# Equal Risk Contribution
# ==============================================================


def test_equal_risk_contribution() -> None:
    optimizer = RiskParityOptimizer(
        learning_rate=0.05,
        max_iterations=30_000,
        tolerance=1e-7,
    )

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    risk_contributions = (
        result.risk_metrics.metadata[
            "percentage_risk_contribution"
        ]
    )

    values = np.asarray(
        [
            risk_contributions["SPY"],
            risk_contributions["NVDA"],
            risk_contributions["GLD"],
        ],
        dtype=float,
    )

    target = np.asarray(
        [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ]
    )

    assert np.allclose(
        values,
        target,
        atol=1e-4,
    )


def test_risk_contributions_sum_to_one() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    contributions = (
        result.risk_metrics.metadata[
            "percentage_risk_contribution"
        ]
    )

    assert sum(
        contributions.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ==============================================================
# Custom Risk Budget
# ==============================================================


def test_custom_risk_budget() -> None:
    optimizer = RiskParityOptimizer(
        learning_rate=0.05,
        max_iterations=30_000,
        tolerance=1e-7,
    )

    target_budgets = {
        "SPY": 0.50,
        "NVDA": 0.20,
        "GLD": 0.30,
    }

    result = optimizer.optimize(
        build_request(),
        risk_budgets=target_budgets,
    )

    assert result.risk_metrics is not None

    achieved = (
        result.risk_metrics.metadata[
            "percentage_risk_contribution"
        ]
    )

    assert achieved["SPY"] == pytest.approx(
        0.50,
        abs=1e-4,
    )

    assert achieved["NVDA"] == pytest.approx(
        0.20,
        abs=1e-4,
    )

    assert achieved["GLD"] == pytest.approx(
        0.30,
        abs=1e-4,
    )


def test_custom_risk_budget_is_normalized() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request(),
        risk_budgets={
            "SPY": 5.0,
            "NVDA": 2.0,
            "GLD": 3.0,
        },
    )

    targets = result.diagnostics[
        "target_risk_budgets"
    ]

    assert targets["SPY"] == pytest.approx(
        0.50,
        abs=1e-6,
    )

    assert targets["NVDA"] == pytest.approx(
        0.20,
        abs=1e-6,
    )

    assert targets["GLD"] == pytest.approx(
        0.30,
        abs=1e-6,
    )


# ==============================================================
# Risk contribution analytics
# ==============================================================


def test_component_risk_contribution_sums_to_volatility() -> None:
    optimizer = RiskParityOptimizer()

    covariance = np.asarray(
        build_covariance().matrix,
        dtype=float,
    )

    weights = np.asarray(
        [
            0.40,
            0.20,
            0.40,
        ],
        dtype=float,
    )

    component = (
        optimizer.component_risk_contribution(
            weights,
            covariance,
        )
    )

    variance = (
        optimizer.portfolio_variance(
            weights,
            covariance,
        )
    )

    volatility = np.sqrt(
        variance
    )

    assert np.sum(
        component
    ) == pytest.approx(
        volatility,
        abs=1e-8,
    )


def test_percentage_risk_contribution_sums_to_one() -> None:
    optimizer = RiskParityOptimizer()

    covariance = np.asarray(
        build_covariance().matrix,
        dtype=float,
    )

    weights = np.asarray(
        [
            0.40,
            0.20,
            0.40,
        ],
        dtype=float,
    )

    contribution = (
        optimizer.percentage_risk_contribution(
            weights,
            covariance,
        )
    )

    assert np.sum(
        contribution
    ) == pytest.approx(
        1.0,
        abs=1e-8,
    )


# ==============================================================
# Position constraints
# ==============================================================


def test_maximum_position_constraint_is_respected() -> None:
    optimizer = RiskParityOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="GLD",
            maximum=0.30,
        ),
    ]

    result = optimizer.optimize(
        build_request(
            constraints=constraints
        )
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["GLD"]
        <= 0.30 + 1e-6
    )


def test_minimum_position_constraint_is_respected() -> None:
    optimizer = RiskParityOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            minimum=0.30,
        ),
    ]

    result = optimizer.optimize(
        build_request(
            constraints=constraints
        )
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["NVDA"]
        >= 0.30 - 1e-6
    )


# ==============================================================
# Portfolio metrics
# ==============================================================


def test_risk_metrics_are_generated() -> None:
    optimizer = RiskParityOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    metrics = result.risk_metrics

    assert metrics is not None

    assert metrics.expected_return is not None
    assert metrics.volatility is not None
    assert metrics.sharpe_ratio is not None

    assert metrics.volatility > 0.0

    assert metrics.gross_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert metrics.net_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ==============================================================
# Turnover
# ==============================================================


def test_turnover_is_calculated() -> None:
    optimizer = RiskParityOptimizer()

    current = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "NVDA": 0.30,
            "GLD": 0.20,
        }
    )

    current.validate()

    result = optimizer.optimize(
        build_request(
            current_allocation=current
        )
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.turnover is not None
    assert result.risk_metrics.turnover >= 0.0


# ==============================================================
# Error handling
# ==============================================================


def test_wrong_optimization_method_raises_error() -> None:
    optimizer = RiskParityOptimizer()

    request = build_request(
        method=OptimizationMethod.MEAN_VARIANCE
    )

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            request
        )


def test_missing_risk_budget_symbol_raises_error() -> None:
    optimizer = RiskParityOptimizer()

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            build_request(),
            risk_budgets={
                "SPY": 0.50,
                "NVDA": 0.50,
            },
        )


def test_unknown_risk_budget_symbol_raises_error() -> None:
    optimizer = RiskParityOptimizer()

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            build_request(),
            risk_budgets={
                "SPY": 0.30,
                "NVDA": 0.30,
                "GLD": 0.30,
                "BTC": 0.10,
            },
        )


def test_negative_risk_budget_raises_error() -> None:
    optimizer = RiskParityOptimizer()

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            build_request(),
            risk_budgets={
                "SPY": 0.50,
                "NVDA": -0.10,
                "GLD": 0.60,
            },
        )


def test_infeasible_minimum_constraints_raise_error() -> None:
    optimizer = RiskParityOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="SPY",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="GLD",
            minimum=0.50,
        ),
    ]

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_infeasible_maximum_constraints_raise_error() -> None:
    optimizer = RiskParityOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="SPY",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="GLD",
            maximum=0.20,
        ),
    ]

    with pytest.raises(
        RiskParityError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_invalid_optimizer_configuration_raises_error() -> None:
    with pytest.raises(
        RiskParityError
    ):
        RiskParityOptimizer(
            learning_rate=0.0
        )

    with pytest.raises(
        RiskParityError
    ):
        RiskParityOptimizer(
            max_iterations=0
        )

    with pytest.raises(
        RiskParityError
    ):
        RiskParityOptimizer(
            tolerance=0.0
        )

    with pytest.raises(
        RiskParityError
    ):
        RiskParityOptimizer(
            epsilon=0.0
        )