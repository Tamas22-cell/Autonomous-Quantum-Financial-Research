"""
Tests for the v5.0.7 Robust Portfolio Optimization Engine.
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

from portfolio_construction.robust_optimizer import (
    RobustOptimizerError,
    RobustPortfolioOptimizer,
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
            uncertainty=0.01,
            confidence=0.90,
        ),
        ExpectedReturnEstimate(
            symbol="NVDA",
            expected_return=0.15,
            uncertainty=0.06,
            confidence=0.50,
        ),
        ExpectedReturnEstimate(
            symbol="GLD",
            expected_return=0.05,
            uncertainty=0.005,
            confidence=0.95,
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
    method: OptimizationMethod = OptimizationMethod.ROBUST,
    target_return: float | None = None,
) -> OptimizationRequest:
    return OptimizationRequest(
        request_id="robust_optimizer_test",
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
# Basic optimization
# ==============================================================


def test_robust_optimization_completes_successfully() -> None:
    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert (
        result.status
        == OptimizationStatus.COMPLETED
    )

    assert result.allocation is not None
    assert result.risk_metrics is not None
    assert result.objective_value is not None
    assert result.iterations is not None
    assert result.iterations >= 1


def test_robust_weights_sum_to_one() -> None:
    optimizer = RobustPortfolioOptimizer()

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


def test_robust_portfolio_is_long_only() -> None:
    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.allocation is not None

    for weight in (
        result.allocation.weights.values()
    ):
        assert weight >= 0.0


# ==============================================================
# Robust expected returns
# ==============================================================


def test_robust_returns_are_lower_than_nominal_returns() -> None:
    optimizer = RobustPortfolioOptimizer(
        uncertainty_aversion=1.0
    )

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    metadata = (
        result.risk_metrics.metadata
    )

    nominal = metadata[
        "nominal_expected_returns"
    ]

    robust = metadata[
        "robust_expected_returns"
    ]

    assert robust["SPY"] < nominal["SPY"]
    assert robust["NVDA"] < nominal["NVDA"]
    assert robust["GLD"] < nominal["GLD"]


def test_robust_return_formula_is_correct() -> None:
    optimizer = RobustPortfolioOptimizer(
        uncertainty_aversion=1.0
    )

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    metadata = (
        result.risk_metrics.metadata
    )

    robust = metadata[
        "robust_expected_returns"
    ]

    effective_uncertainty = metadata[
        "effective_uncertainties"
    ]

    nominal = metadata[
        "nominal_expected_returns"
    ]

    assert robust["SPY"] == pytest.approx(
        nominal["SPY"]
        - effective_uncertainty["SPY"],
        abs=1e-10,
    )

    assert robust["NVDA"] == pytest.approx(
        nominal["NVDA"]
        - effective_uncertainty["NVDA"],
        abs=1e-10,
    )

    assert robust["GLD"] == pytest.approx(
        nominal["GLD"]
        - effective_uncertainty["GLD"],
        abs=1e-10,
    )


# ==============================================================
# Confidence-aware uncertainty
# ==============================================================


def test_lower_confidence_increases_effective_uncertainty() -> None:
    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None

    metadata = (
        result.risk_metrics.metadata
    )

    uncertainty = metadata[
        "uncertainties"
    ]

    effective = metadata[
        "effective_uncertainties"
    ]

    assert (
        effective["NVDA"]
        > uncertainty["NVDA"]
    )

    assert (
        effective["SPY"]
        > uncertainty["SPY"]
    )

    assert (
        effective["GLD"]
        > uncertainty["GLD"]
    )


def test_confidence_penalty_formula() -> None:
    optimizer = RobustPortfolioOptimizer()

    uncertainties = np.asarray(
        [
            0.10,
            0.10,
            0.10,
        ],
        dtype=float,
    )

    confidences = np.asarray(
        [
            1.0,
            0.5,
            0.0,
        ],
        dtype=float,
    )

    effective = (
        optimizer._effective_uncertainties(
            uncertainties=uncertainties,
            confidences=confidences,
        )
    )

    assert effective[0] == pytest.approx(
        0.10
    )

    assert effective[1] == pytest.approx(
        0.15
    )

    assert effective[2] == pytest.approx(
        0.20
    )


# ==============================================================
# Uncertainty aversion behavior
# ==============================================================


def test_higher_uncertainty_aversion_changes_allocation() -> None:
    request = build_request()

    low_aversion = RobustPortfolioOptimizer(
        uncertainty_aversion=0.0
    )

    high_aversion = RobustPortfolioOptimizer(
        uncertainty_aversion=3.0
    )

    first = low_aversion.optimize(
        request
    )

    second = high_aversion.optimize(
        request
    )

    assert first.allocation is not None
    assert second.allocation is not None

    first_weights = np.asarray(
        [
            first.allocation.weights["SPY"],
            first.allocation.weights["NVDA"],
            first.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    second_weights = np.asarray(
        [
            second.allocation.weights["SPY"],
            second.allocation.weights["NVDA"],
            second.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    assert not np.allclose(
        first_weights,
        second_weights,
        atol=1e-4,
    )


def test_high_uncertainty_aversion_reduces_nvda_weight() -> None:
    request = build_request()

    low_aversion = RobustPortfolioOptimizer(
        uncertainty_aversion=0.0
    )

    high_aversion = RobustPortfolioOptimizer(
        uncertainty_aversion=3.0
    )

    first = low_aversion.optimize(
        request
    )

    second = high_aversion.optimize(
        request
    )

    assert first.allocation is not None
    assert second.allocation is not None

    assert (
        second.allocation.weights["NVDA"]
        <= first.allocation.weights["NVDA"]
        + 1e-6
    )


# ==============================================================
# Risk aversion behavior
# ==============================================================


def test_risk_aversion_changes_allocation() -> None:
    request = build_request()

    low_risk_aversion = RobustPortfolioOptimizer(
        risk_aversion=0.5
    )

    high_risk_aversion = RobustPortfolioOptimizer(
        risk_aversion=10.0
    )

    first = low_risk_aversion.optimize(
        request
    )

    second = high_risk_aversion.optimize(
        request
    )

    assert first.allocation is not None
    assert second.allocation is not None

    first_weights = np.asarray(
        [
            first.allocation.weights["SPY"],
            first.allocation.weights["NVDA"],
            first.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    second_weights = np.asarray(
        [
            second.allocation.weights["SPY"],
            second.allocation.weights["NVDA"],
            second.allocation.weights["GLD"],
        ],
        dtype=float,
    )

    assert not np.allclose(
        first_weights,
        second_weights,
        atol=1e-4,
    )


# ==============================================================
# Position constraints
# ==============================================================


def test_maximum_position_constraint_is_respected() -> None:
    optimizer = RobustPortfolioOptimizer()

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
        )
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["GLD"]
        <= 0.40 + 1e-6
    )


def test_minimum_position_constraint_is_respected() -> None:
    optimizer = RobustPortfolioOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            minimum=0.25,
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
        >= 0.25 - 1e-6
    )


def test_multiple_position_constraints_are_respected() -> None:
    optimizer = RobustPortfolioOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="SPY",
            minimum=0.20,
            maximum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="NVDA",
            maximum=0.35,
        ),
        PortfolioConstraint(
            constraint_type=ConstraintType.POSITION,
            target="GLD",
            minimum=0.20,
        ),
    ]

    result = optimizer.optimize(
        build_request(
            constraints=constraints
        )
    )

    assert result.allocation is not None

    weights = result.allocation.weights

    assert weights["SPY"] >= 0.20 - 1e-6
    assert weights["SPY"] <= 0.50 + 1e-6
    assert weights["NVDA"] <= 0.35 + 1e-6
    assert weights["GLD"] >= 0.20 - 1e-6


# ==============================================================
# Target return
# ==============================================================


def test_target_return_influences_allocation() -> None:
    optimizer = RobustPortfolioOptimizer(
        uncertainty_aversion=1.0
    )

    no_target = optimizer.optimize(
        build_request()
    )

    target = optimizer.optimize(
        build_request(
            target_return=0.10
        )
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
# Risk metrics
# ==============================================================


def test_risk_metrics_are_generated() -> None:
    optimizer = RobustPortfolioOptimizer()

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

    assert (
        "robust_expected_return"
        in metrics.metadata
    )


def test_robust_expected_return_is_not_above_nominal() -> None:
    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.expected_return is not None

    robust_return = (
        result.risk_metrics.metadata[
            "robust_expected_return"
        ]
    )

    assert (
        robust_return
        <= result.risk_metrics.expected_return
        + 1e-12
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

    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request(
            current_allocation=current
        )
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.turnover is not None

    assert (
        result.risk_metrics.turnover
        >= 0.0
    )


# ==============================================================
# Diagnostics
# ==============================================================


def test_uncertainty_penalty_is_reported() -> None:
    optimizer = RobustPortfolioOptimizer()

    result = optimizer.optimize(
        build_request()
    )

    assert (
        "uncertainty_penalty"
        in result.diagnostics
    )

    assert (
        result.diagnostics[
            "uncertainty_penalty"
        ]
        >= 0.0
    )


# ==============================================================
# Error handling
# ==============================================================


def test_wrong_method_raises_error() -> None:
    optimizer = RobustPortfolioOptimizer()

    request = build_request(
        method=OptimizationMethod.MEAN_VARIANCE
    )

    with pytest.raises(
        RobustOptimizerError
    ):
        optimizer.optimize(
            request
        )


def test_infeasible_minimum_constraints_raise_error() -> None:
    optimizer = RobustPortfolioOptimizer()

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
        RobustOptimizerError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_infeasible_maximum_constraints_raise_error() -> None:
    optimizer = RobustPortfolioOptimizer()

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
        RobustOptimizerError
    ):
        optimizer.optimize(
            build_request(
                constraints=constraints
            )
        )


def test_non_positive_semidefinite_covariance_raises_error() -> None:
    optimizer = RobustPortfolioOptimizer()

    covariance = CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

    request = OptimizationRequest(
        request_id="invalid_covariance",
        method=OptimizationMethod.ROBUST,
        assets=build_assets(),
        expected_returns=build_expected_returns(),
        covariance=covariance,
    )

    with pytest.raises(
        RobustOptimizerError
    ):
        optimizer.optimize(
            request
        )


def test_invalid_optimizer_configuration_raises_error() -> None:
    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            risk_aversion=-1.0
        )

    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            uncertainty_aversion=-1.0
        )

    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            learning_rate=0.0
        )

    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            max_iterations=0
        )

    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            tolerance=0.0
        )

    with pytest.raises(
        RobustOptimizerError
    ):
        RobustPortfolioOptimizer(
            target_return_penalty=-1.0
        )