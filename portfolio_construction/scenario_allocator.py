"""
Scenario-Aware Portfolio Allocation Engine for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Probability-weighted market scenarios
- Regime-aware expected returns
- Scenario-adjusted asset return estimates
- Scenario dispersion / uncertainty penalties
- Scenario-aware mean-variance allocation
- Worst-scenario awareness
- Long-only allocation
- Position minimum / maximum constraints
- Target-return support
- Portfolio expected return
- Scenario-adjusted expected return
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
    MarketScenario,
    OptimizationMethod,
    OptimizationRequest,
    OptimizationResult,
    OptimizationStatus,
    PortfolioAllocation,
    PortfolioRiskMetrics,
    generate_id,
)


class ScenarioAllocatorError(RuntimeError):
    """
    Raised when scenario-aware allocation cannot
    be completed safely.
    """


def utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


class ScenarioAwareAllocator:
    """
    Probability-weighted scenario-aware allocator.

    Each MarketScenario may contain:

        probability
        regime
        asset_returns

    Scenario probabilities are normalized automatically.

    Scenario expected return for asset i:

        scenario_mu_i =
            sum(probability_s * return_s,i)

    The final blended return estimate is:

        blended_mu_i =
            (1 - scenario_weight) * nominal_mu_i
            + scenario_weight * scenario_mu_i

    Scenario dispersion is calculated as the weighted
    standard deviation of scenario returns.

    The optimization objective is:

        risk_aversion * portfolio_variance
        - blended_expected_return
        + scenario_risk_aversion * scenario_variance
        + downside_aversion * worst_scenario_penalty

    Lower objective values are better.
    """

    def __init__(
        self,
        *,
        scenario_weight: float = 0.60,
        risk_aversion: float = 3.0,
        scenario_risk_aversion: float = 1.0,
        downside_aversion: float = 1.0,
        learning_rate: float = 0.05,
        max_iterations: int = 20_000,
        tolerance: float = 1e-9,
        target_return_penalty: float = 1000.0,
    ) -> None:
        if not 0.0 <= scenario_weight <= 1.0:
            raise ScenarioAllocatorError(
                "scenario_weight must be between 0 and 1."
            )

        if risk_aversion < 0.0:
            raise ScenarioAllocatorError(
                "risk_aversion cannot be negative."
            )

        if scenario_risk_aversion < 0.0:
            raise ScenarioAllocatorError(
                "scenario_risk_aversion cannot be negative."
            )

        if downside_aversion < 0.0:
            raise ScenarioAllocatorError(
                "downside_aversion cannot be negative."
            )

        if learning_rate <= 0.0:
            raise ScenarioAllocatorError(
                "learning_rate must be positive."
            )

        if max_iterations < 1:
            raise ScenarioAllocatorError(
                "max_iterations must be at least 1."
            )

        if tolerance <= 0.0:
            raise ScenarioAllocatorError(
                "tolerance must be positive."
            )

        if target_return_penalty < 0.0:
            raise ScenarioAllocatorError(
                "target_return_penalty cannot be negative."
            )

        self.scenario_weight = float(
            scenario_weight
        )

        self.risk_aversion = float(
            risk_aversion
        )

        self.scenario_risk_aversion = float(
            scenario_risk_aversion
        )

        self.downside_aversion = float(
            downside_aversion
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
        Run scenario-aware portfolio allocation.
        """

        started_at = utc_now()

        if (
            request.method
            != OptimizationMethod.SCENARIO_AWARE
        ):
            raise ScenarioAllocatorError(
                "ScenarioAwareAllocator requires "
                "OptimizationMethod.SCENARIO_AWARE."
            )

        try:
            symbols = [
                asset.symbol
                for asset in request.assets
            ]

            if not request.scenarios:
                raise ScenarioAllocatorError(
                    "Scenario-aware allocation "
                    "requires at least one scenario."
                )

            nominal_returns = (
                self._expected_return_vector(
                    request
                )
            )

            covariance = np.asarray(
                request.covariance.matrix,
                dtype=float,
            )

            probabilities = (
                self._scenario_probabilities(
                    request.scenarios
                )
            )

            scenario_matrix = (
                self._scenario_return_matrix(
                    scenarios=request.scenarios,
                    symbols=symbols,
                )
            )

            self._validate_inputs(
                symbols=symbols,
                nominal_returns=nominal_returns,
                covariance=covariance,
                probabilities=probabilities,
                scenario_matrix=scenario_matrix,
            )

            scenario_expected_returns = (
                self._scenario_expected_returns(
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=probabilities,
                )
            )

            blended_returns = (
                self._blended_expected_returns(
                    nominal_returns=(
                        nominal_returns
                    ),
                    scenario_returns=(
                        scenario_expected_returns
                    ),
                )
            )

            scenario_covariance = (
                self._scenario_covariance(
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=probabilities,
                    expected_returns=(
                        scenario_expected_returns
                    ),
                )
            )

            scenario_dispersion = np.sqrt(
                np.maximum(
                    np.diag(
                        scenario_covariance
                    ),
                    0.0,
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
                self._optimize_weights(
                    request=request,
                    nominal_returns=(
                        nominal_returns
                    ),
                    blended_returns=(
                        blended_returns
                    ),
                    covariance=covariance,
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=(
                        probabilities
                    ),
                    scenario_covariance=(
                        scenario_covariance
                    ),
                    initial_weights=(
                        initial_weights
                    ),
                    minimum_weights=(
                        minimum_weights
                    ),
                    maximum_weights=(
                        maximum_weights
                    ),
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
                    "scenario_aware": True,
                    "scenario_weight": (
                        self.scenario_weight
                    ),
                },
            )

            allocation.validate(
                tolerance=1e-6
            )

            nominal_portfolio_return = float(
                weights.dot(
                    nominal_returns
                )
            )

            scenario_portfolio_return = float(
                weights.dot(
                    scenario_expected_returns
                )
            )

            blended_portfolio_return = float(
                weights.dot(
                    blended_returns
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
                    nominal_portfolio_return
                    - request.risk_free_rate
                ) / volatility
            else:
                sharpe_ratio = None

            portfolio_scenario_returns = (
                scenario_matrix.dot(
                    weights
                )
            )

            weighted_scenario_variance = float(
                np.sum(
                    probabilities
                    * np.square(
                        portfolio_scenario_returns
                        - scenario_portfolio_return
                    )
                )
            )

            scenario_volatility = float(
                np.sqrt(
                    max(
                        weighted_scenario_variance,
                        0.0,
                    )
                )
            )

            worst_scenario_index = int(
                np.argmin(
                    portfolio_scenario_returns
                )
            )

            best_scenario_index = int(
                np.argmax(
                    portfolio_scenario_returns
                )
            )

            worst_scenario = (
                request.scenarios[
                    worst_scenario_index
                ]
            )

            best_scenario = (
                request.scenarios[
                    best_scenario_index
                ]
            )

            turnover = self._calculate_turnover(
                request=request,
                symbols=symbols,
                new_weights=weights,
            )

            risk_metrics = PortfolioRiskMetrics(
                expected_return=(
                    nominal_portfolio_return
                ),
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
                    "scenario_expected_return": (
                        scenario_portfolio_return
                    ),
                    "blended_expected_return": (
                        blended_portfolio_return
                    ),
                    "scenario_volatility": (
                        scenario_volatility
                    ),
                    "worst_scenario_return": float(
                        portfolio_scenario_returns[
                            worst_scenario_index
                        ]
                    ),
                    "best_scenario_return": float(
                        portfolio_scenario_returns[
                            best_scenario_index
                        ]
                    ),
                    "worst_scenario_id": (
                        worst_scenario.scenario_id
                    ),
                    "worst_scenario_name": (
                        worst_scenario.name
                    ),
                    "worst_scenario_regime": (
                        worst_scenario.regime
                    ),
                    "best_scenario_id": (
                        best_scenario.scenario_id
                    ),
                    "best_scenario_name": (
                        best_scenario.name
                    ),
                    "best_scenario_regime": (
                        best_scenario.regime
                    ),
                    "scenario_asset_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            scenario_expected_returns,
                        )
                    },
                    "blended_asset_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            blended_returns,
                        )
                    },
                    "scenario_dispersion": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            scenario_dispersion,
                        )
                    },
                },
            )

            objective_value = (
                self._objective(
                    weights=weights,
                    nominal_returns=(
                        nominal_returns
                    ),
                    blended_returns=(
                        blended_returns
                    ),
                    covariance=covariance,
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=(
                        probabilities
                    ),
                    scenario_covariance=(
                        scenario_covariance
                    ),
                    target_return=(
                        request.target_return
                    ),
                )
            )

            completed_at = utc_now()

            return OptimizationResult(
                result_id=generate_id(
                    "portfolio_result"
                ),
                request_id=request.request_id,
                method=(
                    OptimizationMethod.SCENARIO_AWARE
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
                    "projected_scenario_aware_numpy"
                ),
                iterations=iterations,
                message=(
                    "Scenario-aware portfolio "
                    "allocation completed successfully."
                ),
                diagnostics={
                    "scenario_count": len(
                        request.scenarios
                    ),
                    "scenario_weight": (
                        self.scenario_weight
                    ),
                    "risk_aversion": (
                        self.risk_aversion
                    ),
                    "scenario_risk_aversion": (
                        self.scenario_risk_aversion
                    ),
                    "downside_aversion": (
                        self.downside_aversion
                    ),
                    "scenario_probabilities": {
                        scenario.scenario_id: float(
                            probability
                        )
                        for scenario, probability
                        in zip(
                            request.scenarios,
                            probabilities,
                        )
                    },
                    "scenario_expected_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            scenario_expected_returns,
                        )
                    },
                    "blended_expected_returns": {
                        symbol: float(value)
                        for symbol, value in zip(
                            symbols,
                            blended_returns,
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
            ScenarioAllocatorError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            raise ScenarioAllocatorError(
                "Scenario-aware allocation "
                f"failed: {exc}"
            ) from exc

    # ==========================================================
    # Expected returns
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

    # ==========================================================
    # Scenario probabilities
    # ==========================================================

    @staticmethod
    def _scenario_probabilities(
        scenarios: List[MarketScenario],
    ) -> np.ndarray:
        """
        Build normalized scenario probabilities.

        If every scenario probability is None,
        equal probabilities are used.

        If probabilities are supplied, every scenario
        must have a probability.
        """

        if not scenarios:
            raise ScenarioAllocatorError(
                "At least one scenario is required."
            )

        supplied = [
            scenario.probability
            for scenario in scenarios
        ]

        if all(
            probability is None
            for probability in supplied
        ):
            return np.full(
                len(scenarios),
                1.0 / len(scenarios),
                dtype=float,
            )

        if any(
            probability is None
            for probability in supplied
        ):
            raise ScenarioAllocatorError(
                "Either all scenario probabilities "
                "must be supplied or all must be None."
            )

        probabilities = np.asarray(
            supplied,
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise ScenarioAllocatorError(
                "Scenario probabilities contain "
                "non-finite values."
            )

        if np.any(
            probabilities < 0.0
        ):
            raise ScenarioAllocatorError(
                "Scenario probabilities "
                "cannot be negative."
            )

        total = float(
            np.sum(
                probabilities
            )
        )

        if total <= 0.0:
            raise ScenarioAllocatorError(
                "Scenario probability total "
                "must be positive."
            )

        return (
            probabilities
            / total
        )

    # ==========================================================
    # Scenario matrix
    # ==========================================================

    @staticmethod
    def _scenario_return_matrix(
        *,
        scenarios: List[MarketScenario],
        symbols: List[str],
    ) -> np.ndarray:
        rows: List[List[float]] = []

        for scenario in scenarios:
            missing = (
                set(symbols)
                - set(
                    scenario.asset_returns
                )
            )

            if missing:
                raise ScenarioAllocatorError(
                    "Scenario "
                    f"'{scenario.scenario_id}' "
                    "is missing asset returns for: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

            row = [
                float(
                    scenario.asset_returns[
                        symbol
                    ]
                )
                for symbol in symbols
            ]

            rows.append(
                row
            )

        return np.asarray(
            rows,
            dtype=float,
        )

    # ==========================================================
    # Scenario analytics
    # ==========================================================

    @staticmethod
    def _scenario_expected_returns(
        *,
        scenario_matrix: np.ndarray,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        return probabilities.dot(
            scenario_matrix
        )

    def _blended_expected_returns(
        self,
        *,
        nominal_returns: np.ndarray,
        scenario_returns: np.ndarray,
    ) -> np.ndarray:
        return (
            (
                1.0
                - self.scenario_weight
            )
            * nominal_returns
            + self.scenario_weight
            * scenario_returns
        )

    @staticmethod
    def _scenario_covariance(
        *,
        scenario_matrix: np.ndarray,
        probabilities: np.ndarray,
        expected_returns: np.ndarray,
    ) -> np.ndarray:
        centered = (
            scenario_matrix
            - expected_returns
        )

        weighted = (
            centered
            * probabilities[:, None]
        )

        covariance = (
            centered.T.dot(
                weighted
            )
        )

        return covariance

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_inputs(
        *,
        symbols: List[str],
        nominal_returns: np.ndarray,
        covariance: np.ndarray,
        probabilities: np.ndarray,
        scenario_matrix: np.ndarray,
    ) -> None:
        size = len(symbols)

        if size < 2:
            raise ScenarioAllocatorError(
                "At least two assets are required."
            )

        if nominal_returns.shape != (
            size,
        ):
            raise ScenarioAllocatorError(
                "Nominal expected-return vector "
                "has incorrect dimensions."
            )

        if covariance.shape != (
            size,
            size,
        ):
            raise ScenarioAllocatorError(
                "Covariance matrix has "
                "incorrect dimensions."
            )

        if scenario_matrix.ndim != 2:
            raise ScenarioAllocatorError(
                "Scenario matrix must be "
                "two-dimensional."
            )

        if scenario_matrix.shape[1] != size:
            raise ScenarioAllocatorError(
                "Scenario matrix asset count "
                "does not match portfolio."
            )

        if scenario_matrix.shape[0] != len(
            probabilities
        ):
            raise ScenarioAllocatorError(
                "Scenario probability count "
                "does not match scenario count."
            )

        if not np.all(
            np.isfinite(
                nominal_returns
            )
        ):
            raise ScenarioAllocatorError(
                "Nominal returns contain "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(
                covariance
            )
        ):
            raise ScenarioAllocatorError(
                "Covariance matrix contains "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(
                scenario_matrix
            )
        ):
            raise ScenarioAllocatorError(
                "Scenario matrix contains "
                "non-finite values."
            )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):
            raise ScenarioAllocatorError(
                "Probabilities contain "
                "non-finite values."
            )

        if not np.allclose(
            covariance,
            covariance.T,
            atol=1e-10,
        ):
            raise ScenarioAllocatorError(
                "Covariance matrix must "
                "be symmetric."
            )

        eigenvalues = np.linalg.eigvalsh(
            covariance
        )

        if np.min(
            eigenvalues
        ) < -1e-8:
            raise ScenarioAllocatorError(
                "Covariance matrix must be "
                "positive semi-definite."
            )

    # ==========================================================
    # Weight bounds
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
            raise ScenarioAllocatorError(
                "Minimum weights cannot be negative."
            )

        if np.any(
            maximum_weights > 1.0
        ):
            raise ScenarioAllocatorError(
                "Maximum weights cannot exceed 1.0."
            )

        if np.any(
            minimum_weights
            > maximum_weights
        ):
            raise ScenarioAllocatorError(
                "Minimum weight exceeds maximum weight."
            )

        if float(
            np.sum(
                minimum_weights
            )
        ) > 1.0 + 1e-10:
            raise ScenarioAllocatorError(
                "Minimum weight constraints "
                "are infeasible."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-10:
            raise ScenarioAllocatorError(
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
        nominal_returns: np.ndarray,
        blended_returns: np.ndarray,
        covariance: np.ndarray,
        scenario_matrix: np.ndarray,
        probabilities: np.ndarray,
        scenario_covariance: np.ndarray,
        target_return: Optional[float],
    ) -> float:
        blended_portfolio_return = float(
            weights.dot(
                blended_returns
            )
        )

        nominal_portfolio_return = float(
            weights.dot(
                nominal_returns
            )
        )

        base_variance = float(
            weights.T.dot(
                covariance
            ).dot(weights)
        )

        scenario_variance = float(
            weights.T.dot(
                scenario_covariance
            ).dot(weights)
        )

        scenario_portfolio_returns = (
            scenario_matrix.dot(
                weights
            )
        )

        worst_scenario_return = float(
            np.min(
                scenario_portfolio_returns
            )
        )

        downside_penalty = max(
            -worst_scenario_return,
            0.0,
        )

        objective = (
            self.risk_aversion
            * base_variance
            - blended_portfolio_return
            + self.scenario_risk_aversion
            * scenario_variance
            + self.downside_aversion
            * downside_penalty
            * downside_penalty
        )

        if (
            target_return is not None
            and nominal_portfolio_return
            < target_return
        ):
            shortfall = (
                target_return
                - nominal_portfolio_return
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
    # Numerical gradient
    # ==========================================================

    def _gradient(
        self,
        *,
        weights: np.ndarray,
        nominal_returns: np.ndarray,
        blended_returns: np.ndarray,
        covariance: np.ndarray,
        scenario_matrix: np.ndarray,
        probabilities: np.ndarray,
        scenario_covariance: np.ndarray,
        target_return: Optional[float],
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Numerical gradient is used because the
        worst-scenario term is non-smooth.
        """

        step = 1e-6

        gradient = np.zeros_like(
            weights
        )

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
                    nominal_returns=(
                        nominal_returns
                    ),
                    blended_returns=(
                        blended_returns
                    ),
                    covariance=covariance,
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=(
                        probabilities
                    ),
                    scenario_covariance=(
                        scenario_covariance
                    ),
                    target_return=(
                        target_return
                    ),
                )
            )

            negative_objective = (
                self._objective(
                    weights=negative,
                    nominal_returns=(
                        nominal_returns
                    ),
                    blended_returns=(
                        blended_returns
                    ),
                    covariance=covariance,
                    scenario_matrix=(
                        scenario_matrix
                    ),
                    probabilities=(
                        probabilities
                    ),
                    scenario_covariance=(
                        scenario_covariance
                    ),
                    target_return=(
                        target_return
                    ),
                )
            )

            gradient[index] = (
                positive_objective
                - negative_objective
            ) / (
                2.0 * step
            )

        return gradient

    # ==========================================================
    # Optimization
    # ==========================================================

    def _optimize_weights(
        self,
        *,
        request: OptimizationRequest,
        nominal_returns: np.ndarray,
        blended_returns: np.ndarray,
        covariance: np.ndarray,
        scenario_matrix: np.ndarray,
        probabilities: np.ndarray,
        scenario_covariance: np.ndarray,
        initial_weights: np.ndarray,
        minimum_weights: np.ndarray,
        maximum_weights: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        weights = initial_weights.copy()

        previous_objective = (
            self._objective(
                weights=weights,
                nominal_returns=nominal_returns,
                blended_returns=blended_returns,
                covariance=covariance,
                scenario_matrix=scenario_matrix,
                probabilities=probabilities,
                scenario_covariance=(
                    scenario_covariance
                ),
                target_return=request.target_return,
            )
        )

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            gradient = self._gradient(
                weights=weights,
                nominal_returns=nominal_returns,
                blended_returns=blended_returns,
                covariance=covariance,
                scenario_matrix=scenario_matrix,
                probabilities=probabilities,
                scenario_covariance=(
                    scenario_covariance
                ),
                target_return=request.target_return,
                minimum_weights=minimum_weights,
                maximum_weights=maximum_weights,
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
                        nominal_returns=(
                            nominal_returns
                        ),
                        blended_returns=(
                            blended_returns
                        ),
                        covariance=covariance,
                        scenario_matrix=(
                            scenario_matrix
                        ),
                        probabilities=(
                            probabilities
                        ),
                        scenario_covariance=(
                            scenario_covariance
                        ),
                        target_return=(
                            request.target_return
                        ),
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
    # Weight projection
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
            raise ScenarioAllocatorError(
                "Infeasible minimum weights."
            )

        if float(
            np.sum(
                maximum_weights
            )
        ) < 1.0 - 1e-12:
            raise ScenarioAllocatorError(
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
            raise ScenarioAllocatorError(
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
    "ScenarioAllocatorError",
    "ScenarioAwareAllocator",
]