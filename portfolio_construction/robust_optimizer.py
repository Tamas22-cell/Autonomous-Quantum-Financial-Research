"""
Robust Portfolio Optimizer for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Expected-return uncertainty adjustment
- Confidence-aware return estimates
- Robust mean-variance optimization
- Uncertainty-aversion control
- Risk-aversion control
- Long-only allocation
- Position minimum / maximum constraints
- Target-return support
- Portfolio expected return
- Robust expected return
- Volatility
- Sharpe ratio
- Turnover calculation
- Standard OptimizationResult output

The implementation uses NumPy only.
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


class RobustOptimizerError(RuntimeError):
    """
    Raised when robust portfolio optimization
    cannot be completed safely.
    """


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


class RobustPortfolioOptimizer:
    """
    Long-only robust mean-variance optimizer.

    The optimizer reduces expected-return estimates
    according to their uncertainty.

    Base robust return:

        robust_mu_i =
            mu_i - uncertainty_aversion * uncertainty_i

    When confidence is available:

        effective_uncertainty_i =
            uncertainty_i * (1 + (1 - confidence_i))

    Therefore lower-confidence estimates receive a
    larger uncertainty penalty.

    Objective:

        robust_expected_return
        - risk_aversion * portfolio_variance

    If a target return is supplied, portfolios below
    the target receive an additional quadratic penalty.
    """

    def __init__(
        self,
        *,
        risk_aversion: float = 3.0,
        uncertainty_aversion: float = 1.0,
        learning_rate: float = 0.05,
        max_iterations: int = 20_000,
        tolerance: float = 1e-9,
        target_return_penalty: float = 1000.0,
    ) -> None:
        if risk_aversion < 0.0:
            raise RobustOptimizerError(
                "risk_aversion cannot be negative."
            )

        if uncertainty_aversion < 0.0:
            raise RobustOptimizerError(
                "uncertainty_aversion cannot be negative."
            )

        if learning_rate <= 0.0:
            raise RobustOptimizerError(
                "learning_rate must be positive."
            )

        if max_iterations < 1:
            raise RobustOptimizerError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0:
            raise RobustOptimizerError(
                "tolerance must be positive."
            )

        if target_return_penalty < 0.0:
            raise RobustOptimizerError(
                "target_return_penalty cannot be negative."
            )

        self.risk_aversion = float(
            risk_aversion
        )

        self.uncertainty_aversion = float(
            uncertainty_aversion
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

        self.target_return_penalty = float(
            target_return_penalty
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def optimize(
        self,
        request: OptimizationRequest,
    ) -> OptimizationResult:
        """
        Run robust portfolio optimization.
        """

        started_at = utc_now()

        if request.method != OptimizationMethod.ROBUST:
            raise RobustOptimizerError(
                "RobustPortfolioOptimizer requires "
                "OptimizationMethod.ROBUST."
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

            uncertainties = (
                self._uncertainty_vector(
                    request
                )
            )

            confidences = (
                self._confidence_vector(
                    request
                )
            )

            effective_uncertainties = (
                self._effective_uncertainties(
                    uncertainties=uncertainties,
                    confidences=confidences,
                )
            )

            robust_returns = (
                self._robust_return_vector(
                    expected_returns=expected_returns,
                    effective_uncertainties=(
                        effective_uncertainties
                    ),
                )
            )

            covariance = np.asarray(
                request.covariance.matrix,
                dtype=float,
            )

            self._validate_inputs(
                symbols=symbols,
                expected_returns=expected_returns,
                uncertainties=uncertainties,
                confidences=confidences,
                robust_returns=robust_returns,
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

            weights, iterations = (
                self._optimize_weights(
                    request=request,
                    expected_returns=expected_returns,
                    robust_returns=robust_returns,
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
                    "robust_optimized": True,
                    "risk_aversion": (
                        self.risk_aversion
                    ),
                    "uncertainty_aversion": (
                        self.uncertainty_aversion
                    ),
                },
            )

            allocation.validate(
                tolerance=1e-6
            )

            expected_return = float(
                weights.dot(
                    expected_returns
                )
            )

            robust_expected_return = float(
                weights.dot(
                    robust_returns
                )
            )

            portfolio_variance = float(
                weights.T.dot(
                    covariance
                ).dot(weights)
            )

            portfolio_variance = max(
                portfolio_variance,
                0.0,
            )

            volatility = float(
                np.sqrt(
                    portfolio_variance
                )
            )

            if volatility > 0.0:
                sharpe_ratio: Optional[float] = (
                    expected_return
                    - request.risk_free_rate
                ) / volatility
            else:
                sharpe_ratio = None

            turnover = self._calculate_turnover(
                request=request,
                symbols=symbols,
                new_weights=weights,
            )

            risk_metrics = PortfolioRiskMetrics(
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
                    "robust_expected_return": (
                        robust_expected_return
                    ),
                    "risk_aversion": (
                        self.risk_aversion
                    ),
                    "uncertainty_aversion": (
                        self.uncertainty_aversion
                    ),
                    "nominal_expected_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            expected_returns,
                        )
                    },
                    "robust_expected_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            robust_returns,
                        )
                    },
                    "uncertainties": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            uncertainties,
                        )
                    },
                    "effective_uncertainties": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            effective_uncertainties,
                        )
                    },
                    "confidences": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            confidences,
                        )
                    },
                },
            )

            objective_value = (
                self._objective(
                    weights=weights,
                    expected_returns=expected_returns,
                    robust_returns=robust_returns,
                    covariance=covariance,
                    target_return=request.target_return,
                )
            )

            completed_at = utc_now()

            return OptimizationResult(
                result_id=generate_id(
                    "portfolio_result"
                ),
                request_id=request.request_id,
                method=OptimizationMethod.ROBUST,
                status=OptimizationStatus.COMPLETED,
                allocation=allocation,
                risk_metrics=risk_metrics,
                objective_value=float(
                    objective_value
                ),
                solver_name=(
                    "projected_robust_mean_variance_numpy"
                ),
                iterations=iterations,
                message=(
                    "Robust portfolio optimization "
                    "completed successfully."
                ),
                diagnostics={
                    "risk_aversion": (
                        self.risk_aversion
                    ),
                    "uncertainty_aversion": (
                        self.uncertainty_aversion
                    ),
                    "learning_rate": (
                        self.learning_rate
                    ),
                    "target_return_penalty": (
                        self.target_return_penalty
                    ),
                    "nominal_expected_return": (
                        expected_return
                    ),
                    "robust_expected_return": (
                        robust_expected_return
                    ),
                    "uncertainty_penalty": float(
                        expected_return
                        - robust_expected_return
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
            RobustOptimizerError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise RobustOptimizerError(
                "Robust portfolio optimization "
                f"failed: {exc}"
            ) from exc

    # ==========================================================
    # Input vectors
    # ==========================================================

    @staticmethod
    def _expected_return_vector(
        request: OptimizationRequest,
    ) -> np.ndarray:
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
    def _uncertainty_vector(
        request: OptimizationRequest,
    ) -> np.ndarray:
        estimates = {
            estimate.symbol: (
                estimate.uncertainty
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
    def _confidence_vector(
        request: OptimizationRequest,
    ) -> np.ndarray:
        estimates = {
            estimate.symbol: (
                1.0
                if estimate.confidence is None
                else estimate.confidence
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
    def _effective_uncertainties(
        *,
        uncertainties: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Increase uncertainty penalty when confidence falls.

        confidence = 1.0:
            multiplier = 1.0

        confidence = 0.5:
            multiplier = 1.5

        confidence = 0.0:
            multiplier = 2.0
        """

        confidence_penalty = (
            1.0
            + (
                1.0
                - confidences
            )
        )

        return (
            uncertainties
            * confidence_penalty
        )

    def _robust_return_vector(
        self,
        *,
        expected_returns: np.ndarray,
        effective_uncertainties: np.ndarray,
    ) -> np.ndarray:
        return (
            expected_returns
            - self.uncertainty_aversion
            * effective_uncertainties
        )

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_inputs(
        *,
        symbols: List[str],
        expected_returns: np.ndarray,
        uncertainties: np.ndarray,
        confidences: np.ndarray,
        robust_returns: np.ndarray,
        covariance: np.ndarray,
    ) -> None:
        size = len(symbols)

        if size < 2:
            raise RobustOptimizerError(
                "At least two assets are required."
            )

        vectors = {
            "expected_returns": expected_returns,
            "uncertainties": uncertainties,
            "confidences": confidences,
            "robust_returns": robust_returns,
        }

        for name, vector in vectors.items():
            if vector.shape != (size,):
                raise RobustOptimizerError(
                    f"{name} has incorrect dimensions."
                )

            if not np.all(
                np.isfinite(vector)
            ):
                raise RobustOptimizerError(
                    f"{name} contains non-finite values."
                )

        if np.any(
            uncertainties < 0.0
        ):
            raise RobustOptimizerError(
                "Uncertainties cannot be negative."
            )

        if np.any(
            confidences < 0.0
        ) or np.any(
            confidences > 1.0
        ):
            raise RobustOptimizerError(
                "Confidences must be between 0 and 1."
            )

        if covariance.shape != (
            size,
            size,
        ):
            raise RobustOptimizerError(
                "Covariance matrix has incorrect dimensions."
            )

        if not np.all(
            np.isfinite(covariance)
        ):
            raise RobustOptimizerError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise RobustOptimizerError(
                "Covariance matrix must be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(
            eigenvalues
        ) < -1e-8:
            raise RobustOptimizerError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==========================================================
    # Weight constraints
    # ==========================================================

    @staticmethod
    def _build_weight_bounds(
        *,
        request: OptimizationRequest,
        symbols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            raise RobustOptimizerError(
                "Minimum weights cannot be negative."
            )

        if np.any(
            maximum_weights > 1.0
        ):
            raise RobustOptimizerError(
                "Maximum weights cannot exceed 1.0."
            )

        if np.any(
            minimum_weights
            > maximum_weights
        ):
            raise RobustOptimizerError(
                "Minimum weight exceeds maximum weight."
            )

        if float(
            np.sum(
                minimum_weights
            )
        ) > 1.0 + 1e-10:
            raise RobustOptimizerError(
                "Minimum weight constraints "
                "are infeasible."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-10:
            raise RobustOptimizerError(
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
        if request.current_allocation is not None:
            weights = np.asarray(
                [
                    request.current_allocation.weights.get(
                        symbol,
                        0.0,
                    )
                    for symbol in symbols
                ],
                dtype=float,
            )

            if float(
                np.sum(weights)
            ) > 0.0:
                return self._project_weights(
                    weights,
                    minimum_weights,
                    maximum_weights,
                )

        weights = np.full(
            len(symbols),
            1.0 / len(symbols),
            dtype=float,
        )

        return self._project_weights(
            weights,
            minimum_weights,
            maximum_weights,
        )

    # ==========================================================
    # Objective
    # ==========================================================

    def _objective(
        self,
        *,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        robust_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: Optional[float],
    ) -> float:
        """
        Objective is returned as a minimization value.

        Lower is better.
        """

        robust_expected_return = float(
            weights.dot(
                robust_returns
            )
        )

        variance = float(
            weights.T.dot(
                covariance
            ).dot(weights)
        )

        objective = (
            self.risk_aversion
            * variance
            - robust_expected_return
        )

        if target_return is not None:
            nominal_return = float(
                weights.dot(
                    expected_returns
                )
            )

            if nominal_return < target_return:
                shortfall = (
                    target_return
                    - nominal_return
                )

                objective += (
                    self.target_return_penalty
                    * shortfall
                    * shortfall
                )

        return float(
            objective
        )

    # ==========================================================
    # Analytical gradient
    # ==========================================================

    def _gradient(
        self,
        *,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        robust_returns: np.ndarray,
        covariance: np.ndarray,
        target_return: Optional[float],
    ) -> np.ndarray:
        gradient = (
            2.0
            * self.risk_aversion
            * covariance.dot(weights)
            - robust_returns
        )

        if target_return is not None:
            nominal_return = float(
                weights.dot(
                    expected_returns
                )
            )

            if nominal_return < target_return:
                shortfall = (
                    target_return
                    - nominal_return
                )

                gradient = (
                    gradient
                    - 2.0
                    * self.target_return_penalty
                    * shortfall
                    * expected_returns
                )

        return gradient

    # ==========================================================
    # Optimization
    # ==========================================================

    def _optimize_weights(
        self,
        *,
        request: OptimizationRequest,
        expected_returns: np.ndarray,
        robust_returns: np.ndarray,
        covariance: np.ndarray,
        initial_weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        weights = initial_weights.copy()

        previous_objective = (
            self._objective(
                weights=weights,
                expected_returns=expected_returns,
                robust_returns=robust_returns,
                covariance=covariance,
                target_return=request.target_return,
            )
        )

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            gradient = self._gradient(
                weights=weights,
                expected_returns=expected_returns,
                robust_returns=robust_returns,
                covariance=covariance,
                target_return=request.target_return,
            )

            local_rate = (
                self.learning_rate
            )

            accepted = False

            candidate = weights.copy()
            candidate_objective = (
                previous_objective
            )

            for _ in range(25):
                candidate = (
                    weights
                    - local_rate
                    * gradient
                )

                candidate = (
                    self._project_weights(
                        candidate,
                        minimum_weights,
                        maximum_weights,
                    )
                )

                candidate_objective = (
                    self._objective(
                        weights=candidate,
                        expected_returns=expected_returns,
                        robust_returns=robust_returns,
                        covariance=covariance,
                        target_return=request.target_return,
                    )
                )

                if (
                    candidate_objective
                    <= previous_objective
                    + 1e-14
                ):
                    accepted = True
                    break

                local_rate *= 0.5

            if not accepted:
                return (
                    weights,
                    iteration,
                )

            weight_change = float(
                np.linalg.norm(
                    candidate - weights,
                    ord=1,
                )
            )

            objective_change = abs(
                candidate_objective
                - previous_objective
            )

            weights = candidate
            previous_objective = (
                candidate_objective
            )

            if (
                weight_change
                <= self.tolerance
                or objective_change
                <= self.tolerance
            ):
                return (
                    weights,
                    iteration,
                )

        return (
            weights,
            self.max_iterations,
        )

    # ==========================================================
    # Bounded simplex projection
    # ==========================================================

    @staticmethod
    def _project_weights(
        weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        if float(
            np.sum(
                minimum_weights
            )
        ) > 1.0 + 1e-12:
            raise RobustOptimizerError(
                "Infeasible minimum weights."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-12:
            raise RobustOptimizerError(
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

        if abs(
            residual
        ) > 1e-10:
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
                if abs(
                    residual
                ) <= 1e-12:
                    break

                adjustment = min(
                    abs(residual),
                    float(
                        capacity[index]
                    ),
                )

                if residual > 0.0:
                    projected[index] += (
                        adjustment
                    )

                    residual -= adjustment

                else:
                    projected[index] -= (
                        adjustment
                    )

                    residual += adjustment

        if abs(
            float(
                np.sum(projected)
            )
            - 1.0
        ) > 1e-8:
            raise RobustOptimizerError(
                "Unable to project portfolio "
                "onto feasible weight space."
            )

        return projected

    # ==========================================================
    # Turnover
    # ==========================================================

    @staticmethod
    def _calculate_turnover(
        *,
        request: OptimizationRequest,
        symbols: List[str],
        new_weights: np.ndarray,
    ) -> Optional[float]:
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


__all__ = [
    "RobustOptimizerError",
    "RobustPortfolioOptimizer",
]