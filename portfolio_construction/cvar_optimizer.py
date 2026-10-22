"""
CVaR Portfolio Optimizer for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Historical / scenario-based portfolio loss modelling
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR / Expected Shortfall)
- Long-only CVaR minimization
- Return-aware CVaR optimization
- Position minimum / maximum constraints
- Target-return support
- Portfolio expected return
- Volatility
- Sharpe ratio
- Turnover calculation
- Standard OptimizationResult output

The implementation uses NumPy only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

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


class CVaROptimizerError(RuntimeError):
    """
    Raised when CVaR optimization cannot be completed safely.
    """


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


class CVaROptimizer:
    """
    Scenario-based long-only CVaR optimizer.

    CVaR represents the average loss in the worst tail
    of the portfolio loss distribution.

    Example:

        confidence_level = 0.95

    means CVaR is calculated from approximately the worst
    5 percent of portfolio outcomes.

    The optimizer supports two modes:

    1. Pure CVaR minimization
    2. CVaR minimization with expected-return reward

    Objective:

        CVaR - return_weight * expected_return

    A positive return_weight therefore rewards portfolios
    with higher expected return while still penalizing
    tail risk.
    """

    def __init__(
        self,
        *,
        confidence_level: float = 0.95,
        return_weight: float = 0.0,
        learning_rate: float = 0.02,
        max_iterations: int = 10_000,
        tolerance: float = 1e-8,
        finite_difference_step: float = 1e-5,
    ) -> None:
        if not 0.0 < confidence_level < 1.0:
            raise CVaROptimizerError(
                "confidence_level must be between 0 and 1."
            )

        if return_weight < 0.0:
            raise CVaROptimizerError(
                "return_weight cannot be negative."
            )

        if learning_rate <= 0.0:
            raise CVaROptimizerError(
                "learning_rate must be positive."
            )

        if max_iterations < 1:
            raise CVaROptimizerError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0:
            raise CVaROptimizerError(
                "tolerance must be positive."
            )

        if finite_difference_step <= 0.0:
            raise CVaROptimizerError(
                "finite_difference_step must be positive."
            )

        self.confidence_level = float(
            confidence_level
        )

        self.return_weight = float(
            return_weight
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

        self.finite_difference_step = float(
            finite_difference_step
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def optimize(
        self,
        request: OptimizationRequest,
        scenario_returns: np.ndarray,
    ) -> OptimizationResult:
        """
        Run CVaR portfolio optimization.

        scenario_returns shape:

            number_of_scenarios x number_of_assets

        Example:

            [
                [ 0.01,  0.02, -0.01],
                [-0.03, -0.06,  0.01],
                [ 0.02,  0.04,  0.00],
                ...
            ]
        """

        started_at = utc_now()

        if request.method != OptimizationMethod.CVAR:
            raise CVaROptimizerError(
                "CVaROptimizer requires "
                "OptimizationMethod.CVAR."
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

            scenarios = np.asarray(
                scenario_returns,
                dtype=float,
            )

            self._validate_inputs(
                symbols=symbols,
                expected_returns=expected_returns,
                covariance=covariance,
                scenario_returns=scenarios,
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
                self._optimize_cvar(
                    request=request,
                    scenario_returns=scenarios,
                    expected_returns=expected_returns,
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
                    "cvar_optimized": True,
                    "confidence_level": (
                        self.confidence_level
                    ),
                },
            )

            allocation.validate(
                tolerance=1e-6
            )

            portfolio_returns = (
                self.portfolio_scenario_returns(
                    weights,
                    scenarios,
                )
            )

            portfolio_losses = (
                -portfolio_returns
            )

            var_value = self.value_at_risk(
                portfolio_losses
            )

            cvar_value = (
                self.conditional_value_at_risk(
                    portfolio_losses
                )
            )

            expected_return = float(
                weights.dot(
                    expected_returns
                )
            )

            variance = float(
                weights.T.dot(
                    covariance
                ).dot(weights)
            )

            variance = max(
                variance,
                0.0,
            )

            volatility = float(
                np.sqrt(
                    variance
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
                var=float(
                    var_value
                ),
                cvar=float(
                    cvar_value
                ),
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
                    "confidence_level": (
                        self.confidence_level
                    ),
                    "scenario_count": int(
                        scenarios.shape[0]
                    ),
                    "worst_scenario_return": float(
                        np.min(
                            portfolio_returns
                        )
                    ),
                    "best_scenario_return": float(
                        np.max(
                            portfolio_returns
                        )
                    ),
                    "mean_scenario_return": float(
                        np.mean(
                            portfolio_returns
                        )
                    ),
                },
            )

            objective_value = (
                self._objective(
                    weights=weights,
                    scenario_returns=scenarios,
                    expected_returns=expected_returns,
                    target_return=request.target_return,
                )
            )

            completed_at = utc_now()

            return OptimizationResult(
                result_id=generate_id(
                    "portfolio_result"
                ),
                request_id=request.request_id,
                method=OptimizationMethod.CVAR,
                status=OptimizationStatus.COMPLETED,
                allocation=allocation,
                risk_metrics=risk_metrics,
                objective_value=float(
                    objective_value
                ),
                solver_name=(
                    "projected_cvar_numpy"
                ),
                iterations=iterations,
                message=(
                    "CVaR optimization "
                    "completed successfully."
                ),
                diagnostics={
                    "confidence_level": (
                        self.confidence_level
                    ),
                    "return_weight": (
                        self.return_weight
                    ),
                    "learning_rate": (
                        self.learning_rate
                    ),
                    "finite_difference_step": (
                        self.finite_difference_step
                    ),
                    "scenario_count": int(
                        scenarios.shape[0]
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
            CVaROptimizerError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise CVaROptimizerError(
                f"CVaR optimization failed: {exc}"
            ) from exc

    # ==========================================================
    # Expected return vector
    # ==========================================================

    @staticmethod
    def _expected_return_vector(
        request: OptimizationRequest,
    ) -> np.ndarray:
        """
        Build expected return vector using asset ordering.
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

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_inputs(
        *,
        symbols: List[str],
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        scenario_returns: np.ndarray,
    ) -> None:
        """
        Validate numerical inputs.
        """

        asset_count = len(symbols)

        if asset_count < 2:
            raise CVaROptimizerError(
                "At least two assets are required."
            )

        if expected_returns.shape != (
            asset_count,
        ):
            raise CVaROptimizerError(
                "Expected return vector has "
                "incorrect dimensions."
            )

        if covariance.shape != (
            asset_count,
            asset_count,
        ):
            raise CVaROptimizerError(
                "Covariance matrix has "
                "incorrect dimensions."
            )

        if scenario_returns.ndim != 2:
            raise CVaROptimizerError(
                "scenario_returns must be "
                "a two-dimensional matrix."
            )

        if scenario_returns.shape[1] != asset_count:
            raise CVaROptimizerError(
                "Scenario asset count does not "
                "match portfolio asset count."
            )

        if scenario_returns.shape[0] < 2:
            raise CVaROptimizerError(
                "At least two scenarios are required."
            )

        if not np.all(
            np.isfinite(
                scenario_returns
            )
        ):
            raise CVaROptimizerError(
                "Scenario returns contain "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(
                expected_returns
            )
        ):
            raise CVaROptimizerError(
                "Expected returns contain "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(
                covariance
            )
        ):
            raise CVaROptimizerError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise CVaROptimizerError(
                "Covariance matrix must "
                "be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(
            eigenvalues
        ) < -1e-8:
            raise CVaROptimizerError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==========================================================
    # VaR / CVaR analytics
    # ==========================================================

    def value_at_risk(
        self,
        losses: np.ndarray,
    ) -> float:
        """
        Calculate empirical Value at Risk.

        Input values represent losses, so larger positive
        numbers correspond to worse outcomes.
        """

        values = np.asarray(
            losses,
            dtype=float,
        )

        if values.ndim != 1:
            raise CVaROptimizerError(
                "losses must be one-dimensional."
            )

        if values.size == 0:
            raise CVaROptimizerError(
                "losses cannot be empty."
            )

        if not np.all(
            np.isfinite(values)
        ):
            raise CVaROptimizerError(
                "losses contain non-finite values."
            )

        return float(
            np.quantile(
                values,
                self.confidence_level,
                method="linear",
            )
        )

    def conditional_value_at_risk(
        self,
        losses: np.ndarray,
    ) -> float:
        """
        Calculate empirical CVaR / Expected Shortfall.

        CVaR is the average loss greater than or equal
        to the VaR threshold.
        """

        values = np.asarray(
            losses,
            dtype=float,
        )

        var_value = self.value_at_risk(
            values
        )

        tail_losses = values[
            values >= var_value
        ]

        if tail_losses.size == 0:
            return float(
                var_value
            )

        return float(
            np.mean(
                tail_losses
            )
        )

    @staticmethod
    def portfolio_scenario_returns(
        weights: np.ndarray,
        scenario_returns: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate portfolio return under every scenario.
        """

        return scenario_returns.dot(
            weights
        )

    # ==========================================================
    # Constraints
    # ==========================================================

    @staticmethod
    def _build_weight_bounds(
        *,
        request: OptimizationRequest,
        symbols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build long-only asset weight bounds.
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
            raise CVaROptimizerError(
                "Minimum weights cannot be negative."
            )

        if np.any(
            maximum_weights > 1.0
        ):
            raise CVaROptimizerError(
                "Maximum weights cannot exceed 1.0."
            )

        if np.any(
            minimum_weights
            > maximum_weights
        ):
            raise CVaROptimizerError(
                "Minimum weight exceeds maximum weight."
            )

        if float(
            np.sum(
                minimum_weights
            )
        ) > 1.0 + 1e-10:
            raise CVaROptimizerError(
                "Minimum weight constraints "
                "are infeasible."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-10:
            raise CVaROptimizerError(
                "Maximum weight constraints "
                "are infeasible."
            )

        return (
            minimum_weights,
            maximum_weights,
        )

    # ==========================================================
    # Initial weights
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
        Create feasible starting portfolio.
        """

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
    # Objective
    # ==========================================================

    def _objective(
        self,
        *,
        weights: np.ndarray,
        scenario_returns: np.ndarray,
        expected_returns: np.ndarray,
        target_return: Optional[float],
    ) -> float:
        """
        CVaR optimization objective.

        objective =
            CVaR
            - return_weight * expected_return
            + target-return penalty
        """

        portfolio_returns = (
            self.portfolio_scenario_returns(
                weights,
                scenario_returns,
            )
        )

        losses = (
            -portfolio_returns
        )

        cvar_value = (
            self.conditional_value_at_risk(
                losses
            )
        )

        expected_return = float(
            weights.dot(
                expected_returns
            )
        )

        objective = (
            cvar_value
            - self.return_weight
            * expected_return
        )

        if (
            target_return is not None
            and expected_return < target_return
        ):
            shortfall = (
                target_return
                - expected_return
            )

            objective += (
                1000.0
                * shortfall
                * shortfall
            )

        return float(
            objective
        )

    # ==========================================================
    # Optimization
    # ==========================================================

    def _optimize_cvar(
        self,
        *,
        request: OptimizationRequest,
        scenario_returns: np.ndarray,
        expected_returns: np.ndarray,
        initial_weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Minimize CVaR objective using projected
        finite-difference gradient descent.
        """

        weights = initial_weights.copy()

        previous_objective = (
            self._objective(
                weights=weights,
                scenario_returns=scenario_returns,
                expected_returns=expected_returns,
                target_return=request.target_return,
            )
        )

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            gradient = (
                self._finite_difference_gradient(
                    weights=weights,
                    scenario_returns=scenario_returns,
                    expected_returns=expected_returns,
                    target_return=request.target_return,
                    minimum_weights=minimum_weights,
                    maximum_weights=maximum_weights,
                )
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

            candidate_objective = (
                self._objective(
                    weights=candidate,
                    scenario_returns=scenario_returns,
                    expected_returns=expected_returns,
                    target_return=request.target_return,
                )
            )

            if (
                candidate_objective
                > previous_objective
            ):
                local_rate = (
                    self.learning_rate
                )

                improved = False

                for _ in range(20):
                    local_rate *= 0.5

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
                            scenario_returns=scenario_returns,
                            expected_returns=expected_returns,
                            target_return=request.target_return,
                        )
                    )

                    if (
                        candidate_objective
                        <= previous_objective
                    ):
                        improved = True
                        break

                if not improved:
                    return (
                        weights,
                        iteration,
                    )

            difference = float(
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
                difference <= self.tolerance
                or objective_change <= self.tolerance
            ):
                return (
                    weights,
                    iteration,
                )

        return (
            weights,
            self.max_iterations,
        )

    def _finite_difference_gradient(
        self,
        *,
        weights: np.ndarray,
        scenario_returns: np.ndarray,
        expected_returns: np.ndarray,
        target_return: Optional[float],
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate objective gradient numerically.
        """

        gradient = np.zeros_like(
            weights
        )

        step = self.finite_difference_step

        for index in range(
            len(weights)
        ):
            positive = weights.copy()
            negative = weights.copy()

            positive[index] += step
            negative[index] -= step

            positive = self._project_weights(
                positive,
                minimum_weights,
                maximum_weights,
            )

            negative = self._project_weights(
                negative,
                minimum_weights,
                maximum_weights,
            )

            positive_objective = (
                self._objective(
                    weights=positive,
                    scenario_returns=scenario_returns,
                    expected_returns=expected_returns,
                    target_return=target_return,
                )
            )

            negative_objective = (
                self._objective(
                    weights=negative,
                    scenario_returns=scenario_returns,
                    expected_returns=expected_returns,
                    target_return=target_return,
                )
            )

            distance = float(
                np.linalg.norm(
                    positive - negative
                )
            )

            if distance <= 1e-14:
                gradient[index] = 0.0
            else:
                gradient[index] = (
                    positive_objective
                    - negative_objective
                ) / (
                    2.0 * step
                )

        return gradient

    # ==========================================================
    # Weight projection
    # ==========================================================

    @staticmethod
    def _project_weights(
        weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Project weights onto bounded simplex.

        Conditions:

            sum(weights) = 1
            minimum <= weights <= maximum
        """

        if float(
            np.sum(
                minimum_weights
            )
        ) > 1.0 + 1e-12:
            raise CVaROptimizerError(
                "Infeasible minimum weights."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-12:
            raise CVaROptimizerError(
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
            raise CVaROptimizerError(
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
        """
        Calculate one-way turnover.

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


__all__ = [
    "CVaROptimizer",
    "CVaROptimizerError",
]