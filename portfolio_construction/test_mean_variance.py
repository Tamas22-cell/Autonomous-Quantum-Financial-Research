"""
Tests for the v5.0.7 Mean-Variance and
Minimum-Variance Portfolio Construction Engine.
"""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_construction.mean_variance import (
    MeanVarianceError,
    MeanVarianceOptimizer,
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
# Fixtures / helpers
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
            uncertainty=0.02,
            confidence=0.80,
        ),
        ExpectedReturnEstimate(
            symbol="NVDA",
            expected_return=0.15,
            uncertainty=0.05,
            confidence=0.70,
        ),
        ExpectedReturnEstimate(
            symbol="GLD",
            expected_return=0.05,
            uncertainty=0.015,
            confidence=0.85,
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
    method: OptimizationMethod,
    constraints: list[PortfolioConstraint] | None = None,
    current_allocation: PortfolioAllocation | None = None,
) -> OptimizationRequest:
    return OptimizationRequest(
        request_id="test_request",
        method=method,
        assets=build_assets(),
        expected_returns=build_expected_returns(),
        covariance=build_covariance(),
        constraints=constraints or [],
        current_allocation=current_allocation,
        risk_free_rate=0.02,
    )


# ==============================================================
# Mean-Variance tests
# ==============================================================


def test_mean_variance_completes_successfully() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    result = optimizer.optimize(
        request
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


def test_mean_variance_weights_sum_to_one() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    total_weight = sum(
        result.allocation.weights.values()
    )

    assert total_weight == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_mean_variance_is_long_only() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    for weight in (
        result.allocation.weights.values()
    ):
        assert weight >= 0.0


def test_mean_variance_returns_risk_metrics() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    metrics = result.risk_metrics

    assert metrics is not None

    assert metrics.expected_return is not None

    assert metrics.volatility is not None

    assert metrics.volatility >= 0.0

    assert metrics.sharpe_ratio is not None

    assert metrics.gross_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert metrics.net_exposure == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ==============================================================
# Minimum-Variance tests
# ==============================================================


def test_minimum_variance_completes_successfully() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    assert (
        result.status
        == OptimizationStatus.COMPLETED
    )

    assert result.allocation is not None

    assert result.risk_metrics is not None


def test_minimum_variance_weights_sum_to_one() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    total_weight = sum(
        result.allocation.weights.values()
    )

    assert total_weight == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_minimum_variance_reduces_volatility_vs_equal_weight() -> None:
    optimizer = MeanVarianceOptimizer(
        learning_rate=0.05,
        max_iterations=20_000,
    )

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE
    )

    result = optimizer.optimize(
        request
    )

    assert result.risk_metrics is not None
    assert result.risk_metrics.volatility is not None

    covariance = np.asarray(
        build_covariance().matrix,
        dtype=float,
    )

    equal_weights = np.asarray(
        [
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
        ],
        dtype=float,
    )

    equal_variance = float(
        equal_weights.T.dot(
            covariance
        ).dot(equal_weights)
    )

    equal_volatility = float(
        np.sqrt(equal_variance)
    )

    assert (
        result.risk_metrics.volatility
        <= equal_volatility + 1e-6
    )


# ==============================================================
# Position constraint tests
# ==============================================================


def test_maximum_position_constraint_is_respected() -> None:
    optimizer = MeanVarianceOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            maximum=0.25,
        ),
    ]

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE,
        constraints=constraints,
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["NVDA"]
        <= 0.25 + 1e-6
    )


def test_minimum_position_constraint_is_respected() -> None:
    optimizer = MeanVarianceOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            minimum=0.30,
        ),
    ]

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE,
        constraints=constraints,
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    assert (
        result.allocation.weights["GLD"]
        >= 0.30 - 1e-6
    )


def test_multiple_position_constraints_are_respected() -> None:
    optimizer = MeanVarianceOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            minimum=0.20,
            maximum=0.60,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            maximum=0.25,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            minimum=0.20,
        ),
    ]

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE,
        constraints=constraints,
    )

    result = optimizer.optimize(
        request
    )

    assert result.allocation is not None

    weights = result.allocation.weights

    assert 0.20 - 1e-6 <= weights["SPY"]
    assert weights["SPY"] <= 0.60 + 1e-6

    assert weights["NVDA"] <= 0.25 + 1e-6

    assert weights["GLD"] >= 0.20 - 1e-6

    assert sum(
        weights.values()
    ) == pytest.approx(
        1.0,
        abs=1e-6,
    )


# ==============================================================
# Current allocation / turnover
# ==============================================================


def test_turnover_is_calculated() -> None:
    optimizer = MeanVarianceOptimizer()

    current_allocation = PortfolioAllocation(
        weights={
            "SPY": 0.50,
            "NVDA": 0.30,
            "GLD": 0.20,
        }
    )

    current_allocation.validate()

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE,
        current_allocation=(
            current_allocation
        ),
    )

    result = optimizer.optimize(
        request
    )

    assert result.risk_metrics is not None

    assert (
        result.risk_metrics.turnover
        is not None
    )

    assert (
        result.risk_metrics.turnover
        >= 0.0
    )


# ==============================================================
# Objective behavior
# ==============================================================


def test_higher_risk_aversion_changes_allocation() -> None:
    request = build_request(
        OptimizationMethod.MEAN_VARIANCE
    )

    low_risk_aversion = (
        MeanVarianceOptimizer(
            risk_aversion=0.25
        )
    )

    high_risk_aversion = (
        MeanVarianceOptimizer(
            risk_aversion=5.0
        )
    )

    low_result = (
        low_risk_aversion.optimize(
            request
        )
    )

    high_result = (
        high_risk_aversion.optimize(
            request
        )
    )

    assert low_result.allocation is not None
    assert high_result.allocation is not None

    low_weights = np.asarray(
        [
            low_result.allocation.weights[
                symbol
            ]
            for symbol in [
                "SPY",
                "NVDA",
                "GLD",
            ]
        ]
    )

    high_weights = np.asarray(
        [
            high_result.allocation.weights[
                symbol
            ]
            for symbol in [
                "SPY",
                "NVDA",
                "GLD",
            ]
        ]
    )

    assert not np.allclose(
        low_weights,
        high_weights,
        atol=1e-4,
    )


# ==============================================================
# Error handling
# ==============================================================


def test_unsupported_method_raises_error() -> None:
    optimizer = MeanVarianceOptimizer()

    request = build_request(
        OptimizationMethod.RISK_PARITY
    )

    with pytest.raises(
        MeanVarianceError
    ):
        optimizer.optimize(
            request
        )


def test_infeasible_minimum_constraints_raise_error() -> None:
    optimizer = MeanVarianceOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            minimum=0.50,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            minimum=0.50,
        ),
    ]

    request = build_request(
        OptimizationMethod.MEAN_VARIANCE,
        constraints=constraints,
    )

    with pytest.raises(
        MeanVarianceError
    ):
        optimizer.optimize(
            request
        )


def test_infeasible_maximum_constraints_raise_error() -> None:
    optimizer = MeanVarianceOptimizer()

    constraints = [
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="SPY",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="NVDA",
            maximum=0.20,
        ),
        PortfolioConstraint(
            constraint_type=(
                ConstraintType.POSITION
            ),
            target="GLD",
            maximum=0.20,
        ),
    ]

    request = build_request(
        OptimizationMethod.MINIMUM_VARIANCE,
        constraints=constraints,
    )

    with pytest.raises(
        MeanVarianceError
    ):
        optimizer.optimize(
            request
        )


def test_non_positive_semidefinite_covariance_raises_error() -> None:
    optimizer = MeanVarianceOptimizer()

    invalid_covariance = CovarianceMatrix(
        symbols=[
            "SPY",
            "NVDA",
            "GLD",
        ],
        matrix=[
            [
                1.0,
                2.0,
                0.0,
            ],
            [
                2.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
    )

    request = OptimizationRequest(
        request_id="invalid_covariance",
        method=(
            OptimizationMethod.MEAN_VARIANCE
        ),
        assets=build_assets(),
        expected_returns=(
            build_expected_returns()
        ),
        covariance=invalid_covariance,
    )

    with pytest.raises(
        MeanVarianceError
    ):
        optimizer.optimize(
            request
        )


def test_invalid_optimizer_configuration_raises_error() -> None:
    with pytest.raises(
        MeanVarianceError
    ):
        MeanVarianceOptimizer(
            risk_aversion=0.0
        )

    with pytest.raises(
        MeanVarianceError
    ):
        MeanVarianceOptimizer(
            learning_rate=0.0
        )

    with pytest.raises(
        MeanVarianceError
    ):
        MeanVarianceOptimizer(
            max_iterations=0
        )

    with pytest.raises(
        MeanVarianceError
    ):
        MeanVarianceOptimizer(
            tolerance=0.0
        )