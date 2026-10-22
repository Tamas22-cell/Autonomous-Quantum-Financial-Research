"""
Tests for the v5.0.7 CVaR Portfolio Construction Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.cvar_optimizer import (
    CVaROptimizer,
    CVaROptimizerError,
)

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


def build_scenarios() -> np.ndarray:
    """
    Scenario matrix:

        rows    = market scenarios
        columns = SPY, NVDA, GLD

    NVDA has the strongest upside but also the
    largest downside tail.
    """

    return np.asarray(
        [
            [0.020, 0.040, 0.005],
            [0.015, 0.030, 0.002],
            [0.010, 0.025, 0.004],
            [0.025, 0.050, -0.005],
            [0.005, 0.015, 0.008],
            [-0.010, -0.025, 0.010],
            [-0.015, -0.040, 0.015],
            [-0.020, -0.060, 0.020],
            [-0.025, -0.080, 0.025],
            [-0.030, -0.100, 0.030],
            [0.012, 0.035, 0.003],
            [0.018, 0.045, -0.002],
            [0.007, 0.020, 0.006],
            [-0.008, -0.020, 0.012],
            [-0.012, -0.035, 0.018],
            [0.022, 0.055, -0.004],
            [0.014, 0.028, 0.004],
            [-0.018, -0.050, 0.022],
            [-0.035, -0.120, 0.035],
            [-0.040, -0.150, 0.040],
        ],
        dtype=float,
    )


def build_request(
    *,
    constraints: list[PortfolioConstraint] | None = None,
    current_allocation: PortfolioAllocation | None = None,
    method: OptimizationMethod = OptimizationMethod.CVAR,
    target_return: float | None = None,
) -> OptimizationRequest:
    return OptimizationRequest(
        request_id="cvar_test",
        method=method,
        assets=build_assets(),
        expected_returns=build_expected_returns(),
        covariance=build_covariance(),
        constraints=constraints or [],
        current_allocation=current_allocation,
        risk_free_rate=0.02,
        target_return=target_return,
    )


# ==============================================================
# VaR / CVaR analytics
# ==============================================================


def test_value_at_risk_is_calculated() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.80
    )

    losses = np.asarray(
        [
            -0.02,
            -0.01,
            0.00,
            0.01,
            0.02,
            0.05,
            0.08,
            0.10,
            0.15,
            0.20,
        ],
        dtype=float,
    )

    value = optimizer.value_at_risk(
        losses
    )

    assert np.isfinite(value)


def test_cvar_is_not_less_than_var() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.80
    )

    losses = np.asarray(
        [
            -0.02,
            -0.01,
            0.00,
            0.01,
            0.02,
            0.05,
            0.08,
            0.10,
            0.15,
            0.20,
        ],
        dtype=float,
    )

    var_value = optimizer.value_at_risk(
        losses
    )

    cvar_value = (
        optimizer.conditional_value_at_risk(
            losses
        )
    )

    assert cvar_value >= var_value


def test_known_cvar_tail_average() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.80
    )

    losses = np.asarray(
        [
            0.00,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
        dtype=float,
    )

    cvar_value = (
        optimizer.conditional_value_at_risk(
            losses
        )
    )

    assert cvar_value >= 0.08
    assert cvar_value <= 0.09


# ==============================================================
# Portfolio scenario returns
# ==============================================================


def test_portfolio_scenario_returns() -> None:
    optimizer = CVaROptimizer()

    scenarios = np.asarray(
        [
            [0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00],
            [0.00, 0.00, 0.10],
        ],
        dtype=float,
    )

    weights = np.asarray(
        [
            0.50,
            0.30,
            0.20,
        ],
        dtype=float,
    )

    returns = (
        optimizer.portfolio_scenario_returns(
            weights,
            scenarios,
        )
    )

    assert returns[0] == pytest.approx(
        0.05
    )

    assert returns[1] == pytest.approx(
        0.03
    )

    assert returns[2] == pytest.approx(
        0.02
    )


# ==============================================================
# Basic optimization
# ==============================================================


def test_cvar_optimization_completes() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    result = optimizer.optimize(
        build_request(),
        build_scenarios(),
    )

    assert (
        result.status
        == OptimizationStatus.COMPLETED
    )

    assert result.allocation is not None
    assert result.risk_metrics is not None
    assert result.objective_value is not None
    assert result.iterations is not None


def test_cvar_weights_sum_to_one() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    result = optimizer.optimize(
        build_request(),
        build_scenarios(),
    )

    assert result.allocation is not None

    assert sum(
        result.allocation.weights.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_cvar_portfolio_is_long_only() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    result = optimizer.optimize(
        build_request(),
        build_scenarios(),
    )

    assert result.allocation is not None

    for weight in (
        result.allocation.weights.values()
    ):
        assert weight >= 0.0


def test_cvar_metrics_are_generated() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    result = optimizer.optimize(
        build_request(),
        build_scenarios(),
    )

    metrics = result.risk_metrics

    assert metrics is not None
    assert metrics.expected_return is not None
    assert metrics.volatility is not None
    assert metrics.sharpe_ratio is not None
    assert metrics.var is not None
    assert metrics.cvar is not None

    assert metrics.cvar >= metrics.var

    assert metrics.gross_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert metrics.net_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ==============================================================
# Tail-risk behavior
# ==============================================================


def test_cvar_optimization_reduces_tail_risk_vs_nvda_only() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90,
        return_weight=0.0,
    )

    scenarios = build_scenarios()

    result = optimizer.optimize(
        build_request(),
        scenarios,
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.cvar is not None

    nvda_weights = np.asarray(
        [
            0.0,
            1.0,
            0.0,
        ],
        dtype=float,
    )

    nvda_returns = (
        optimizer.portfolio_scenario_returns(
            nvda_weights,
            scenarios,
        )
    )

    nvda_losses = -nvda_returns

    nvda_cvar = (
        optimizer.conditional_value_at_risk(
            nvda_losses
        )
    )

    assert (
        result.risk_metrics.cvar
        <= nvda_cvar + 1e-6
    )


# ==============================================================
# Position constraints
# ==============================================================


def test_maximum_position_constraint_is_respected() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="GLD",
            maximum=0.40,
        ),
    ]

    result = optimizer.optimize(
        build_request(
            constraints=constraints
        ),
        build_scenarios(),
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["GLD"]
        <= 0.40 + 1e-6
    )


def test_minimum_position_constraint_is_respected() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="SPY",
            minimum=0.30,
        ),
    ]

    result = optimizer.optimize(
        build_request(
            constraints=constraints
        ),
        build_scenarios(),
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["SPY"]
        >= 0.30 - 1e-6
    )


# ==============================================================
# Return-aware CVaR
# ==============================================================


def test_return_weight_changes_portfolio() -> None:
    request = build_request()
    scenarios = build_scenarios()

    pure_tail_risk = CVaROptimizer(
        confidence_level=0.90,
        return_weight=0.0,
    )

    return_aware = CVaROptimizer(
        confidence_level=0.90,
        return_weight=1.0,
    )

    first = pure_tail_risk.optimize(
        request,
        scenarios,
    )

    second = return_aware.optimize(
        request,
        scenarios,
    )

    assert first.allocation is not None
    assert second.allocation is not None

    first_weights = np.asarray(
        [
            first.allocation.weights["SPY"],
            first.allocation.weights["NVDA"],
            first.allocation.weights["GLD"],
        ]
    )

    second_weights = np.asarray(
        [
            second.allocation.weights["SPY"],
            second.allocation.weights["NVDA"],
            second.allocation.weights["GLD"],
        ]
    )

    assert not np.allclose(
        first_weights,
        second_weights,
        atol=1e-4,
    )


def test_target_return_influences_allocation() -> None:
    scenarios = build_scenarios()

    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    no_target = optimizer.optimize(
        build_request(),
        scenarios,
    )

    target = optimizer.optimize(
        build_request(
            target_return=0.09
        ),
        scenarios,
    )

    assert no_target.risk_metrics is not None
    assert target.risk_metrics is not None

    assert (
        no_target.risk_metrics.expected_return
        is not None
    )

    assert (
        target.risk_metrics.expected_return
        is not None
    )

    assert (
        target.risk_metrics.expected_return
        >= no_target.risk_metrics.expected_return
        - 1e-6
    )


# ==============================================================
# Turnover
# ==============================================================


def test_turnover_is_calculated() -> None:
    current = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "NVDA": 0.30,
            "GLD": 0.20,
        }
    )

    current.validate()

    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    result = optimizer.optimize(
        build_request(
            current_allocation=current
        ),
        build_scenarios(),
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.turnover is not None
    assert result.risk_metrics.turnover >= 0.0


# ==============================================================
# Diagnostics
# ==============================================================


def test_scenario_diagnostics_are_generated() -> None:
    optimizer = CVaROptimizer(
        confidence_level=0.90
    )

    scenarios = build_scenarios()

    result = optimizer.optimize(
        build_request(),
        scenarios,
    )

    assert (
        result.diagnostics["scenario_count"]
        == scenarios.shape[0]
    )

    assert (
        result.diagnostics["confidence_level"]
        == pytest.approx(0.90)
    )

    assert result.risk_metrics is not None

    assert (
        result.risk_metrics.metadata[
            "scenario_count"
        ]
        == scenarios.shape[0]
    )


# ==============================================================
# Error handling
# ==============================================================


def test_wrong_method_raises_error() -> None:
    optimizer = CVaROptimizer()

    request = build_request(
        method=OptimizationMethod.MEAN_VARIANCE
    )

    with pytest.raises(
        CVaROptimizerError
    ):
        optimizer.optimize(
            request,
            build_scenarios(),
        )


def test_invalid_scenario_shape_raises_error() -> None:
    optimizer = CVaROptimizer()

    invalid_scenarios = np.asarray(
        [
            [0.01, 0.02],
            [-0.01, -0.02],
        ],
        dtype=float,
    )

    with pytest.raises(
        CVaROptimizerError
    ):
        optimizer.optimize(
            build_request(),
            invalid_scenarios,
        )


def test_one_dimensional_scenarios_raise_error() -> None:
    optimizer = CVaROptimizer()

    invalid_scenarios = np.asarray(
        [
            0.01,
            -0.02,
            0.03,
        ],
        dtype=float,
    )

    with pytest.raises(
        CVaROptimizerError
    ):
        optimizer.optimize(
            build_request(),
            invalid_scenarios,
        )


def test_non_finite_scenarios_raise_error() -> None:
    optimizer = CVaROptimizer()

    scenarios = build_scenarios()

    scenarios[0, 0] = np.nan

    with pytest.raises(
        CVaROptimizerError
    ):
        optimizer.optimize(
            build_request(),
            scenarios,
        )


def test_infeasible_minimum_constraints_raise_error() -> None:
    optimizer = CVaROptimizer()

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
        CVaROptimizerError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            ),
            build_scenarios(),
        )


def test_infeasible_maximum_constraints_raise_error() -> None:
    optimizer = CVaROptimizer()

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
        CVaROptimizerError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            ),
            build_scenarios(),
        )


def test_empty_losses_raise_error() -> None:
    optimizer = CVaROptimizer()

    with pytest.raises(
        CVaROptimizerError
    ):
        optimizer.value_at_risk(
            np.asarray(
                [],
                dtype=float,
            )
        )


def test_invalid_optimizer_configuration_raises_error() -> None:
    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            confidence_level=1.0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            confidence_level=0.0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            return_weight=-1.0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            learning_rate=0.0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            max_iterations=0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            tolerance=0.0
        )

    with pytest.raises(
        CVaROptimizerError
    ):
        CVaROptimizer(
            finite_difference_step=0.0
        )