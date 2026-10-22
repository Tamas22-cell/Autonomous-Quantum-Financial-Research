"""
Risk Parity Engine for the v5.0.7 Portfolio Construction Engine.

Capabilities:

- Equal Risk Contribution portfolio construction
- Custom risk-budget support
- Marginal Risk Contribution
- Component Risk Contribution
- Percentage Risk Contribution
- Long-only portfolio construction
- Position minimum / maximum constraints
- Deterministic iterative optimization
- Standard OptimizationResult output
- Portfolio expected return
- Volatility
- Sharpe ratio
- Turnover calculation

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


class RiskParityError(RuntimeError):
    """
    Raised when Risk Parity optimization cannot
    be completed safely.
    """


def utc_now() -> datetime:
    """
    Return timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


class RiskParityOptimizer:
    """
    Long-only Risk Parity optimizer.

    The objective is to allocate portfolio weights so that
    assets contribute approximately equal amounts of total
    portfolio risk.

    Custom risk budgets may also be supplied.

    Example:

        risk_budgets = {
            "SPY": 0.40,
            "NVDA": 0.20,
            "GLD": 0.40,
        }

    If no custom risk budget is supplied, Equal Risk
    Contribution is used automatically.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iterations: int = 20_000,
        tolerance: float = 1e-8,
        epsilon: float = 1e-12,
    ) -> None:
        if learning_rate <= 0.0:
            raise RiskParityError(
                "learning_rate must be positive."
            )

        if max_iterations < 1:
            raise RiskParityError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0:
            raise RiskParityError(
                "tolerance must be positive."
            )

        if epsilon <= 0.0:
            raise RiskParityError(
                "epsilon must be positive."
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

        self.epsilon = float(
            epsilon
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def optimize(
        self,
        request: OptimizationRequest,
        *,
        risk_budgets: Optional[
            Dict[str, float]
        ] = None,
    ) -> OptimizationResult:
        """
        Run Risk Parity optimization.
        """

        started_at = utc_now()

        if (
            request.method
            != OptimizationMethod.RISK_PARITY
        ):
            raise RiskParityError(
                "RiskParityOptimizer requires "
                "OptimizationMethod.RISK_PARITY."
            )

        try:
            symbols = [
                asset.symbol
                for asset in request.assets
            ]

            covariance = np.asarray(
                request.covariance.matrix,
                dtype=float,
            )

            expected_returns = (
                self._expected_return_vector(
                    request
                )
            )

            self._validate_inputs(
                symbols=symbols,
                covariance=covariance,
                expected_returns=expected_returns,
            )

            target_budgets = (
                self._build_risk_budgets(
                    symbols=symbols,
                    risk_budgets=risk_budgets,
                )
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
                self._optimize_risk_parity(
                    covariance=covariance,
                    initial_weights=initial_weights,
                    target_budgets=target_budgets,
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
                    "risk_parity": True,
                },
            )

            allocation.validate(
                tolerance=1e-6
            )

            portfolio_variance = (
                self.portfolio_variance(
                    weights,
                    covariance,
                )
            )

            portfolio_volatility = float(
                np.sqrt(
                    portfolio_variance
                )
            )

            marginal_risk = (
                self.marginal_risk_contribution(
                    weights,
                    covariance,
                )
            )

            component_risk = (
                self.component_risk_contribution(
                    weights,
                    covariance,
                )
            )

            percentage_risk = (
                self.percentage_risk_contribution(
                    weights,
                    covariance,
                )
            )

            expected_return = float(
                weights.dot(
                    expected_returns
                )
            )

            if portfolio_volatility > self.epsilon:
                sharpe_ratio: Optional[float] = (
                    expected_return
                    - request.risk_free_rate
                ) / portfolio_volatility
            else:
                sharpe_ratio = None

            turnover = self._calculate_turnover(
                request=request,
                symbols=symbols,
                new_weights=weights,
            )

            risk_metrics = PortfolioRiskMetrics(
                expected_return=expected_return,
                volatility=portfolio_volatility,
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
                        request.risk_free_rate
                    ),
                    "marginal_risk_contribution": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            marginal_risk,
                        )
                    },
                    "component_risk_contribution": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            component_risk,
                        )
                    },
                    "percentage_risk_contribution": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            percentage_risk,
                        )
                    },
                    "target_risk_budgets": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            target_budgets,
                        )
                    },
                },
            )

            objective_value = (
                self._risk_budget_error(
                    weights=weights,
                    covariance=covariance,
                    target_budgets=target_budgets,
                )
            )

            completed_at = utc_now()

            return OptimizationResult(
                result_id=generate_id(
                    "portfolio_result"
                ),
                request_id=request.request_id,
                method=(
                    OptimizationMethod.RISK_PARITY
                ),
                status=(
                    OptimizationStatus.COMPLETED
                ),
                allocation=allocation,
                risk_metrics=risk_metrics,
                objective_value=float(
                    objective_value
                ),
                solver_name=(
                    "iterative_risk_budget_numpy"
                ),
                iterations=iterations,
                message=(
                    "Risk Parity optimization "
                    "completed successfully."
                ),
                diagnostics={
                    "learning_rate": (
                        self.learning_rate
                    ),
                    "tolerance": (
                        self.tolerance
                    ),
                    "target_risk_budgets": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            target_budgets,
                        )
                    },
                    "achieved_risk_budgets": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            percentage_risk,
                        )
                    },
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
            RiskParityError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise RiskParityError(
                f"Risk Parity optimization failed: {exc}"
            ) from exc

    # ==========================================================
    # Input preparation
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

    @staticmethod
    def _validate_inputs(
        *,
        symbols: List[str],
        covariance: np.ndarray,
        expected_returns: np.ndarray,
    ) -> None:
        """
        Validate numerical Risk Parity inputs.
        """

        size = len(symbols)

        if size < 2:
            raise RiskParityError(
                "At least two assets are required."
            )

        if covariance.shape != (
            size,
            size,
        ):
            raise RiskParityError(
                "Covariance matrix has "
                "incorrect dimensions."
            )

        if expected_returns.shape != (
            size,
        ):
            raise RiskParityError(
                "Expected return vector has "
                "incorrect dimensions."
            )

        if not np.all(
            np.isfinite(covariance)
        ):
            raise RiskParityError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(expected_returns)
        ):
            raise RiskParityError(
                "Expected returns contain "
                "non-finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise RiskParityError(
                "Covariance matrix must "
                "be symmetric."
            )

        diagonal = np.diag(
            covariance
        )

        if np.any(
            diagonal <= 0.0
        ):
            raise RiskParityError(
                "Every asset must have "
                "positive variance."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(
            eigenvalues
        ) < -1e-8:
            raise RiskParityError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==========================================================
    # Risk budgets
    # ==========================================================

    @staticmethod
    def _build_risk_budgets(
        *,
        symbols: List[str],
        risk_budgets: Optional[
            Dict[str, float]
        ],
    ) -> np.ndarray:
        """
        Create normalized risk-budget vector.

        Equal budgets are used when none are supplied.
        """

        size = len(symbols)

        if risk_budgets is None:
            return np.full(
                size,
                1.0 / size,
                dtype=float,
            )

        unknown_symbols = (
            set(risk_budgets)
            - set(symbols)
        )

        if unknown_symbols:
            raise RiskParityError(
                "Unknown symbols in risk budget: "
                + ", ".join(
                    sorted(
                        unknown_symbols
                    )
                )
            )

        missing_symbols = (
            set(symbols)
            - set(risk_budgets)
        )

        if missing_symbols:
            raise RiskParityError(
                "Missing risk budgets for: "
                + ", ".join(
                    sorted(
                        missing_symbols
                    )
                )
            )

        budgets = np.asarray(
            [
                risk_budgets[symbol]
                for symbol in symbols
            ],
            dtype=float,
        )

        if not np.all(
            np.isfinite(budgets)
        ):
            raise RiskParityError(
                "Risk budgets contain "
                "non-finite values."
            )

        if np.any(
            budgets <= 0.0
        ):
            raise RiskParityError(
                "Risk budgets must be positive."
            )

        total = float(
            np.sum(budgets)
        )

        if total <= 0.0:
            raise RiskParityError(
                "Risk budget total must be positive."
            )

        return budgets / total

    # ==========================================================
    # Position constraints
    # ==============================================================

    @staticmethod
    def _build_weight_bounds(
        *,
        request: OptimizationRequest,
        symbols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build asset-level long-only bounds.
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
            raise RiskParityError(
                "Minimum weights cannot be negative."
            )

        if np.any(
            maximum_weights > 1.0
        ):
            raise RiskParityError(
                "Maximum weights cannot exceed 1.0."
            )

        if np.any(
            minimum_weights
            > maximum_weights
        ):
            raise RiskParityError(
                "Minimum weight exceeds maximum weight."
            )

        if (
            float(
                np.sum(minimum_weights)
            )
            > 1.0 + 1e-10
        ):
            raise RiskParityError(
                "Minimum weight constraints "
                "are infeasible."
            )

        if (
            float(
                np.sum(maximum_weights)
            )
            < 1.0 - 1e-10
        ):
            raise RiskParityError(
                "Maximum weight constraints "
                "are infeasible."
            )

        return (
            minimum_weights,
            maximum_weights,
        )

    # ==========================================================
    # Initial weights
    # ==============================================================

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

        inverse_volatility = (
            self._inverse_volatility_weights(
                request=request
            )
        )

        return self._project_weights(
            inverse_volatility,
            minimum_weights,
            maximum_weights,
        )

    @staticmethod
    def _inverse_volatility_weights(
        *,
        request: OptimizationRequest,
    ) -> np.ndarray:
        """
        Generate inverse-volatility starting weights.
        """

        covariance = np.asarray(
            request.covariance.matrix,
            dtype=float,
        )

        volatility = np.sqrt(
            np.diag(
                covariance
            )
        )

        inverse_volatility = (
            1.0 / volatility
        )

        return (
            inverse_volatility
            / np.sum(
                inverse_volatility
            )
        )

    # ==========================================================
    # Risk Parity optimizer
    # ==============================================================

    def _optimize_risk_parity(
        self,
        *,
        covariance: np.ndarray,
        initial_weights: np.ndarray,
        target_budgets: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Iteratively adjust weights toward target
        percentage risk contributions.

        The multiplicative update is stable for
        long-only Risk Parity portfolios.
        """

        weights = initial_weights.copy()

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            current_budgets = (
                self.percentage_risk_contribution(
                    weights,
                    covariance,
                )
            )

            error = (
                current_budgets
                - target_budgets
            )

            maximum_error = float(
                np.max(
                    np.abs(error)
                )
            )

            if maximum_error <= self.tolerance:
                return (
                    weights,
                    iteration,
                )

            adjustment = np.exp(
                -self.learning_rate
                * error
                / np.maximum(
                    target_budgets,
                    self.epsilon,
                )
            )

            candidate = (
                weights
                * adjustment
            )

            candidate = (
                self._project_weights(
                    candidate,
                    minimum_weights,
                    maximum_weights,
                )
            )

            difference = float(
                np.linalg.norm(
                    candidate - weights,
                    ord=1,
                )
            )

            weights = candidate

            if difference <= self.tolerance:
                return (
                    weights,
                    iteration,
                )

        return (
            weights,
            self.max_iterations,
        )

    # ==========================================================
    # Risk contribution analytics
    # ==============================================================

    @staticmethod
    def portfolio_variance(
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> float:
        """
        Calculate portfolio variance.
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

    def marginal_risk_contribution(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Marginal contribution of each asset to
        portfolio volatility.
        """

        variance = (
            self.portfolio_variance(
                weights,
                covariance,
            )
        )

        volatility = float(
            np.sqrt(
                variance
            )
        )

        if volatility <= self.epsilon:
            raise RiskParityError(
                "Portfolio volatility is too small "
                "for risk contribution analysis."
            )

        return (
            covariance.dot(weights)
            / volatility
        )

    def component_risk_contribution(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Component Risk Contribution.

        CRC_i = weight_i * MRC_i
        """

        marginal = (
            self.marginal_risk_contribution(
                weights,
                covariance,
            )
        )

        return (
            weights
            * marginal
        )

    def percentage_risk_contribution(
        self,
        weights: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        Percentage contribution of each asset
        to total portfolio volatility.
        """

        component = (
            self.component_risk_contribution(
                weights,
                covariance,
            )
        )

        total = float(
            np.sum(component)
        )

        if abs(total) <= self.epsilon:
            raise RiskParityError(
                "Total risk contribution is too small."
            )

        return (
            component
            / total
        )

    # ==========================================================
    # Risk budget error
    # ==============================================================

    def _risk_budget_error(
        self,
        *,
        weights: np.ndarray,
        covariance: np.ndarray,
        target_budgets: np.ndarray,
    ) -> float:
        """
        Mean squared error between target and achieved
        percentage risk contributions.
        """

        achieved = (
            self.percentage_risk_contribution(
                weights,
                covariance,
            )
        )

        error = (
            achieved
            - target_budgets
        )

        return float(
            np.mean(
                np.square(error)
            )
        )

    # ==========================================================
    # Weight projection
    # ==============================================================

    @staticmethod
    def _project_weights(
        weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Project portfolio onto bounded simplex.

        Conditions:

            sum(weights) = 1
            minimum <= weights <= maximum
        """

        if (
            float(
                np.sum(minimum_weights)
            )
            > 1.0 + 1e-12
        ):
            raise RiskParityError(
                "Infeasible minimum weights."
            )

        if (
            float(
                np.sum(maximum_weights)
            )
            < 1.0 - 1e-12
        ):
            raise RiskParityError(
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
            raise RiskParityError(
                "Unable to project portfolio "
                "onto feasible weight space."
            )

        return projected

    # ==========================================================
    # Turnover
    # ==============================================================

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


__all__ = [
    "RiskParityError",
    "RiskParityOptimizer",
]