"""
Mean-Variance and Minimum-Variance optimizers for the
v5.0.7 Portfolio Construction Engine.

This module provides:

- Long-only Mean-Variance Optimization
- Long-only Minimum-Variance Optimization
- Expected portfolio return
- Portfolio volatility
- Sharpe ratio
- Constraint-aware weight bounds
- Deterministic projected-gradient optimization
- Standard OptimizationResult output

The implementation uses NumPy only and does not require
an external mathematical optimization package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import (
    ConstraintType,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PortfolioAllocation,
    PortfolioRiskMetrics,
    generate_id,
)


class MeanVarianceError(RuntimeError):
    """
    Raised when portfolio optimization cannot be completed safely.
    """


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


class MeanVarianceOptimizer:
    """
    Long-only portfolio optimizer.

    Supported methods:

    - OptimizationMethod.MEAN_VARIANCE
    - OptimizationMethod.MINIMUM_VARIANCE

    The optimizer uses projected gradient descent.

    After each gradient step the portfolio is projected back
    into the feasible region defined by:

    - full investment
    - long-only weights
    - per-asset minimum weights
    - per-asset maximum weights
    """

    def __init__(
        self,
        *,
        risk_aversion: float = 1.0,
        learning_rate: float = 0.05,
        max_iterations: int = 10_000,
        tolerance: float = 1e-10,
    ) -> None:
        if risk_aversion <= 0.0:
            raise MeanVarianceError(
                "risk_aversion must be positive."
            )

        if learning_rate <= 0.0:
            raise MeanVarianceError(
                "learning_rate must be positive."
            )

        if max_iterations < 1:
            raise MeanVarianceError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0:
            raise MeanVarianceError(
                "tolerance must be positive."
            )

        self.risk_aversion = float(
            risk_aversion
        )

        self.learning_rate = float(
            learning_rate
        )

        self.max_iterations = int(
            max_iterations
        )

        self.tolerance = float(
            tolerance
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def optimize(
        self,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Optimize one portfolio request.
        """

        started_at = utc_now()

        if request.method not in {
            OptimizationMethod.MEAN_VARIANCE,
            OptimizationMethod.MINIMUM_VARIANCE,
        }:
            raise MeanVarianceError(
                "MeanVarianceOptimizer supports only "
                "MEAN_VARIANCE and MINIMUM_VARIANCE."
            )

        try:
            symbols = [
                asset.symbol
                for asset in request.assets
            ]

            expected_returns = (
                self._expected_return_vector(
                    request
                )
            )

            covariance = np.asarray(
                request.covariance.matrix,
                dtype=float,
            )

            self._validate_inputs(
                symbols=symbols,
                expected_returns=expected_returns,
                covariance=covariance,
            )

            minimum_weights, maximum_weights = (
                self._build_weight_bounds(
                    request=request,
                    symbols=symbols,
                )
            )

            initial_weights = (
                self._initial_weights(
                    request=request,
                    symbols=symbols,
                    minimum_weights=minimum_weights,
                    maximum_weights=maximum_weights,
                )
            )

            if (
                request.method
                == OptimizationMethod.MINIMUM_VARIANCE
            ):
                weights, iterations = (
                    self._optimize_minimum_variance(
                        covariance=covariance,
                        initial_weights=initial_weights,
                        minimum_weights=minimum_weights,
                        maximum_weights=maximum_weights,
                    )
                )

            else:
                weights, iterations = (
                    self._optimize_mean_variance(
                        expected_returns=expected_returns,
                        covariance=covariance,
                        initial_weights=initial_weights,
                        minimum_weights=minimum_weights,
                        maximum_weights=maximum_weights,
                    )
                )

            allocation = PortfolioAllocation(
                weights={
                    symbol: float(weight)
                    for symbol, weight in zip(
                        symbols,
                        weights,
                    )
                },
                cash_weight=0.0,
                gross_exposure=float(
                    np.sum(
                        np.abs(weights)
                    )
                ),
                net_exposure=float(
                    np.sum(weights)
                ),
                metadata={
                    "long_only": True,
                },
            )

            allocation.validate(
                tolerance=1e-6
            )

            risk_metrics = (
                self._calculate_risk_metrics(
                    weights=weights,
                    expected_returns=expected_returns,
                    covariance=covariance,
                    risk_free_rate=(
                        request.risk_free_rate
                    ),
                    request=request,
                    symbols=symbols,
                )
            )

            objective_value = (
                self._objective_value(
                    method=request.method,
                    weights=weights,
                    expected_returns=expected_returns,
                    covariance=covariance,
                )
            )

            completed_at = utc_now()

            return OptimizationResult(
                result_id=generate_id(
                    "portfolio_result"
                ),
                request_id=request.request_id,
                method=request.method,
                status=(
                    OptimizationStatus.COMPLETED
                ),
                allocation=allocation,
                risk_metrics=risk_metrics,
                objective_value=float(
                    objective_value
                ),
                solver_name=(
                    "projected_gradient_numpy"
                ),
                iterations=iterations,
                message=(
                    "Portfolio optimization "
                    "completed successfully."
                ),
                diagnostics={
                    "risk_aversion": (
                        self.risk_aversion
                    ),
                    "learning_rate": (
                        self.learning_rate
                    ),
                    "tolerance": (
                        self.tolerance
                    ),
                    "minimum_weights": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            minimum_weights,
                        )
                    },
                    "maximum_weights": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            maximum_weights,
                        )
                    },
                },
                started_at=started_at,
                completed_at=completed_at,
            )

        except (
            MeanVarianceError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise MeanVarianceError(
                f"Portfolio optimization failed: {exc}"
            ) from exc

    # ==========================================================
    # Input preparation
    # ==========================================================

    @staticmethod
    def _expected_return_vector(
        request: OptimizationRequest,
    ) -> np.ndarray:
        """
        Build expected-return vector using asset ordering.
        """

        estimates = {
            estimate.symbol: (
                estimate.expected_return
            )
            for estimate
            in request.expected_returns
        }

        return np.asarray(
            [
                estimates[asset.symbol]
                for asset in request.assets
            ],
            dtype=float,
        )

    @staticmethod
    def _validate_inputs(
        *,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        """
        Validate optimizer numerical inputs.
        """

        asset_count = len(symbols)

        if asset_count < 2:
            raise MeanVarianceError(
                "At least two assets are required."
            )

        if expected_returns.shape != (
            asset_count,
        ):
            raise MeanVarianceError(
                "Expected return vector has "
                "incorrect dimensions."
            )

        if covariance.shape != (
            asset_count,
            asset_count,
        ):
            raise MeanVarianceError(
                "Covariance matrix has "
                "incorrect dimensions."
            )

        if not np.all(
            np.isfinite(expected_returns)
        ):
            raise MeanVarianceError(
                "Expected returns contain "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(covariance)
        ):
            raise MeanVarianceError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise MeanVarianceError(
                "Covariance matrix must "
                "be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(eigenvalues) < -1e-8:
            raise MeanVarianceError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==========================================================
    # Constraint handling
    # ==========================================================

    @staticmethod
    def _build_weight_bounds(
        *,
        request: OptimizationRequest,
        symbols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build per-asset lower and upper weight bounds.

        POSITION, MIN_WEIGHT and MAX_WEIGHT constraints
        targeting an asset symbol are supported here.
        """

        size = len(symbols)

        minimum_weights = np.zeros(
            size,
            dtype=float,
        )

        maximum_weights = np.ones(
            size,
            dtype=float,
        )

        symbol_index = {
            symbol: index
            for index, symbol
            in enumerate(symbols)
        }

        for constraint in request.constraints:
            if not constraint.enabled:
                continue

            if constraint.target not in symbol_index:
                continue

            if constraint.constraint_type not in {
                ConstraintType.POSITION,
                ConstraintType.MIN_WEIGHT,
                ConstraintType.MAX_WEIGHT,
            }:
                continue

            index = symbol_index[
                constraint.target
            ]

            if constraint.minimum is not None:
                minimum_weights[index] = max(
                    minimum_weights[index],
                    float(
                        constraint.minimum
                    ),
                )

            if constraint.maximum is not None:
                maximum_weights[index] = min(
                    maximum_weights[index],
                    float(
                        constraint.maximum
                    ),
                )

        if np.any(
            minimum_weights < 0.0
        ):
            raise MeanVarianceError(
                "Long-only minimum weights "
                "cannot be negative."
            )

        if np.any(
            maximum_weights > 1.0
        ):
            raise MeanVarianceError(
                "Long-only maximum weights "
                "cannot exceed 1.0."
            )

        if np.any(
            minimum_weights
            > maximum_weights
        ):
            raise MeanVarianceError(
                "Minimum weight exceeds "
                "maximum weight."
            )

        if (
            float(
                np.sum(minimum_weights)
            )
            > 1.0 + 1e-10
        ):
            raise MeanVarianceError(
                "Minimum weight constraints "
                "are infeasible."
            )

        if (
            float(
                np.sum(maximum_weights)
            )
            < 1.0 - 1e-10
        ):
            raise MeanVarianceError(
                "Maximum weight constraints "
                "are infeasible."
            )

        return (
            minimum_weights,
            maximum_weights,
        )

    # ==========================================================
    # Initial portfolio
    # ==========================================================

    def _initial_weights(
        self,
        *,
        request: OptimizationRequest,
        symbols: List[str],
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Create optimizer starting weights.
        """

        if request.current_allocation is not None:
            raw_weights = np.asarray(
                [
                    request.current_allocation.weights.get(
                        symbol,
                        0.0,
                    )
                    for symbol in symbols
                ],
                dtype=float,
            )

            if (
                float(
                    np.sum(raw_weights)
                )
                > 0.0
            ):
                return self._project_weights(
                    raw_weights,
                    minimum_weights,
                    maximum_weights,
                )

        equal_weights = np.full(
            len(symbols),
            1.0 / len(symbols),
            dtype=float,
        )

        return self._project_weights(
            equal_weights,
            minimum_weights,
            maximum_weights,
        )

    # ==========================================================
    # Mean-Variance
    # ==========================================================

    def _optimize_mean_variance(
        self,
        *,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        initial_weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Maximize:

            expected return
            -
            risk_aversion * portfolio variance

        using projected gradient ascent.
        """

        weights = initial_weights.copy()

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            gradient = (
                expected_returns
                - (
                    2.0
                    * self.risk_aversion
                    * covariance.dot(weights)
                )
            )

            candidate = (
                weights
                + self.learning_rate
                * gradient
            )

            candidate = self._project_weights(
                candidate,
                minimum_weights,
                maximum_weights,
            )

            difference = np.linalg.norm(
                candidate - weights,
                ord=1,
            )

            weights = candidate

            if difference <= self.tolerance:
                return weights, iteration

        return weights, self.max_iterations

    # ==========================================================
    # Minimum Variance
    # ==========================================================

    def _optimize_minimum_variance(
        self,
        *,
        covariance: np.ndarray,
        initial_weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Minimize portfolio variance using
        projected gradient descent.
        """

        weights = initial_weights.copy()

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            gradient = (
                2.0
                * covariance.dot(weights)
            )

            candidate = (
                weights
                - self.learning_rate
                * gradient
            )

            candidate = self._project_weights(
                candidate,
                minimum_weights,
                maximum_weights,
            )

            difference = np.linalg.norm(
                candidate - weights,
                ord=1,
            )

            weights = candidate

            if difference <= self.tolerance:
                return weights, iteration

        return weights, self.max_iterations

    # ==========================================================
    # Projection
    # ==========================================================

    @staticmethod
    def _project_weights(
        weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Project weights onto the bounded probability simplex.

        Conditions:

            sum(weights) = 1
            minimum <= weight <= maximum
        """

        if (
            float(
                np.sum(minimum_weights)
            )
            > 1.0 + 1e-12
        ):
            raise MeanVarianceError(
                "Infeasible minimum weights."
            )

        if (
            float(
                np.sum(maximum_weights)
            )
            < 1.0 - 1e-12
        ):
            raise MeanVarianceError(
                "Infeasible maximum weights."
            )

        lower = float(
            np.min(
                weights
                - maximum_weights
            )
        )

        upper = float(
            np.max(
                weights
                - minimum_weights
            )
        )

        projected = weights.copy()

        for _ in range(200):
            midpoint = (
                lower + upper
            ) / 2.0

            projected = np.clip(
                weights - midpoint,
                minimum_weights,
                maximum_weights,
            )

            total = float(
                np.sum(projected)
            )

            if abs(
                total - 1.0
            ) <= 1e-12:
                break

            if total > 1.0:
                lower = midpoint
            else:
                upper = midpoint

        projected = np.clip(
            projected,
            minimum_weights,
            maximum_weights,
        )

        residual = (
            1.0
            - float(
                np.sum(projected)
            )
        )

        if abs(residual) > 1e-10:
            if residual > 0.0:
                capacity = (
                    maximum_weights
                    - projected
                )
            else:
                capacity = (
                    projected
                    - minimum_weights
                )

            available = np.where(
                capacity > 1e-12
            )[0]

            for index in available:
                if abs(residual) <= 1e-12:
                    break

                adjustment = min(
                    abs(residual),
                    capacity[index],
                )

                if residual > 0.0:
                    projected[index] += adjustment
                    residual -= adjustment
                else:
                    projected[index] -= adjustment
                    residual += adjustment

        if abs(
            float(
                np.sum(projected)
            )
            - 1.0
        ) > 1e-8:
            raise MeanVarianceError(
                "Unable to project portfolio "
                "onto feasible weight space."
            )

        return projected

    # ==========================================================
    # Portfolio analytics
    # ==========================================================

    @staticmethod
    def _portfolio_return(
        weights: np.ndarray,
        expected_returns: np.ndarray,
    ) -> float:
        """
        Expected portfolio return.
        """

        return float(
            weights.dot(
                expected_returns
            )
        )

    @staticmethod
    def _portfolio_variance(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Portfolio variance.
        """

        variance = float(
            weights.T.dot(
                covariance
            ).dot(weights)
        )

        return max(
            variance,
            0.0,
        )

    @classmethod
    def _portfolio_volatility(
        cls,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Portfolio volatility.
        """

        return float(
            np.sqrt(
                cls._portfolio_variance(
                    weights,
                    covariance,
                )
            )
        )

    def _calculate_risk_metrics(
        self,
        *,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        risk_free_rate: float,
        request: OptimizationRequest,
        symbols: List[str],
    ) -> PortfolioRiskMetrics:
        """
        Calculate standard portfolio analytics.
        """

        expected_return = (
            self._portfolio_return(
                weights,
                expected_returns,
            )
        )

        volatility = (
            self._portfolio_volatility(
                weights,
                covariance,
            )
        )

        if volatility > 0.0:
            sharpe_ratio: Optional[float] = (
                expected_return
                - risk_free_rate
            ) / volatility
        else:
            sharpe_ratio = None

        turnover = self._calculate_turnover(
            request=request,
            symbols=symbols,
            new_weights=weights,
        )

        return PortfolioRiskMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            turnover=turnover,
            gross_exposure=float(
                np.sum(
                    np.abs(weights)
                )
            ),
            net_exposure=float(
                np.sum(weights)
            ),
            metadata={
                "risk_free_rate": float(
                    risk_free_rate
                ),
            },
        )

    @staticmethod
    def _calculate_turnover(
        *,
        request: OptimizationRequest,
        symbols: List[str],
        new_weights: np.ndarray,
    ) -> Optional[float]:
        """
        Calculate one-way portfolio turnover.

        turnover = 0.5 * sum(abs(new - old))
        """

        if request.current_allocation is None:
            return None

        old_weights = np.asarray(
            [
                request.current_allocation.weights.get(
                    symbol,
                    0.0,
                )
                for symbol in symbols
            ],
            dtype=float,
        )

        return float(
            0.5
            * np.sum(
                np.abs(
                    new_weights
                    - old_weights
                )
            )
        )

    # ==========================================================
    # Objective
    # ==========================================================

    def _objective_value(
        self,
        *,
        method: OptimizationMethod,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Return optimizer objective value.
        """

        variance = (
            self._portfolio_variance(
                weights,
                covariance,
            )
        )

        if (
            method
            == OptimizationMethod.MINIMUM_VARIANCE
        ):
            return variance

        expected_return = (
            self._portfolio_return(
                weights,
                expected_returns,
            )
        )

        return (
            expected_return
            - self.risk_aversion
            * variance
        )


__all__ = [
    "MeanVarianceError",
    "MeanVarianceOptimizer",
]