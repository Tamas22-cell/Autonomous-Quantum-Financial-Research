"""
Drawdown-Aware De-Risking Engine for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Current drawdown calculation
- Maximum drawdown calculation
- Drawdown trigger detection
- Progressive risk reduction
- DrawdownPolicy integration
- Recovery-threshold handling
- Risk multiplier calculation
- Portfolio de-risking
- Automatic cash allocation
- Long-only portfolio preservation
- Exposure diagnostics
- Peak / trough analytics

The engine is optimizer-independent and can be applied
after any portfolio optimization method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .models import (
    DrawdownPolicy,
    PortfolioAllocation,
)


class DrawdownEngineError(RuntimeError):
    """
    Raised when drawdown analysis or de-risking fails.
    """


@dataclass(frozen=True)
class DrawdownMetrics:
    """
    Portfolio drawdown analytics.
    """

    current_value: float
    peak_value: float
    current_drawdown: float
    maximum_drawdown: float
    peak_index: int
    trough_index: int
    observations: int
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class DeRiskingResult:
    """
    Result of drawdown-aware portfolio de-risking.
    """

    original_allocation: PortfolioAllocation
    adjusted_allocation: PortfolioAllocation
    metrics: DrawdownMetrics
    risk_multiplier: float
    triggered: bool
    recovering: bool
    risk_reduction: float
    cash_increase: float
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


class DrawdownAwareDeRiskingEngine:
    """
    Drawdown-aware portfolio risk controller.

    The engine applies a DrawdownPolicy to a portfolio.

    Before trigger:

        risk_multiplier = 1.0

    At trigger:

        risk_multiplier = target_risk_multiplier

    Beyond trigger:

        risk exposure is reduced progressively toward
        minimum_risk_multiplier.

    The progressive reduction uses:

        excess_drawdown =
            current_drawdown - trigger_drawdown

        stress_fraction =
            excess_drawdown
            / (1 - trigger_drawdown)

    Then:

        multiplier =
            target_multiplier
            - stress_fraction
            * (
                target_multiplier
                - minimum_multiplier
            )

    The multiplier is always bounded between:

        minimum_risk_multiplier
        and
        1.0

    De-risked capital is moved to cash.
    """

    def __init__(
        self,
        *,
        tolerance: float = 1e-10,
    ) -> None:
        if tolerance < 0.0:
            raise DrawdownEngineError(
                "tolerance cannot be negative."
            )

        self.tolerance = float(
            tolerance
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def analyze(
        self,
        values: Sequence[float],
    ) -> DrawdownMetrics:
        """
        Analyze a portfolio equity/value series.
        """

        array = self._validate_value_series(
            values
        )

        running_peaks = np.maximum.accumulate(
            array
        )

        drawdowns = (
            running_peaks - array
        ) / running_peaks

        current_value = float(
            array[-1]
        )

        peak_value = float(
            running_peaks[-1]
        )

        current_drawdown = float(
            drawdowns[-1]
        )

        trough_index = int(
            np.argmax(
                drawdowns
            )
        )

        maximum_drawdown = float(
            drawdowns[
                trough_index
            ]
        )

        peak_index = int(
            np.argmax(
                array[
                    : trough_index + 1
                ]
            )
        )

        return DrawdownMetrics(
            current_value=current_value,
            peak_value=peak_value,
            current_drawdown=current_drawdown,
            maximum_drawdown=maximum_drawdown,
            peak_index=peak_index,
            trough_index=trough_index,
            observations=len(array),
            metadata={
                "running_peak": peak_value,
                "current_loss_from_peak": float(
                    peak_value
                    - current_value
                ),
            },
        )

    def apply(
        self,
        *,
        allocation: PortfolioAllocation,
        values: Sequence[float],
        policy: DrawdownPolicy,
    ) -> DeRiskingResult:
        """
        Analyze drawdown and apply the configured
        de-risking policy.
        """

        self._validate_allocation(
            allocation
        )

        metrics = self.analyze(
            values
        )

        recovering = self._is_recovering(
            values=values,
            metrics=metrics,
            policy=policy,
        )

        risk_multiplier = (
            self.calculate_risk_multiplier(
                current_drawdown=(
                    metrics.current_drawdown
                ),
                policy=policy,
                recovering=recovering,
            )
        )

        triggered = (
            risk_multiplier
            < 1.0 - self.tolerance
        )

        adjusted_allocation = (
            self.apply_risk_multiplier(
                allocation=allocation,
                risk_multiplier=risk_multiplier,
            )
        )

        original_invested_weight = float(
            sum(
                allocation.weights.values()
            )
        )

        adjusted_invested_weight = float(
            sum(
                adjusted_allocation.weights.values()
            )
        )

        risk_reduction = max(
            original_invested_weight
            - adjusted_invested_weight,
            0.0,
        )

        cash_increase = (
            adjusted_allocation.cash_weight
            - allocation.cash_weight
        )

        return DeRiskingResult(
            original_allocation=allocation,
            adjusted_allocation=(
                adjusted_allocation
            ),
            metrics=metrics,
            risk_multiplier=float(
                risk_multiplier
            ),
            triggered=triggered,
            recovering=recovering,
            risk_reduction=float(
                risk_reduction
            ),
            cash_increase=float(
                cash_increase
            ),
            metadata={
                "trigger_drawdown": (
                    policy.trigger_drawdown
                ),
                "target_risk_multiplier": (
                    policy.target_risk_multiplier
                ),
                "minimum_risk_multiplier": (
                    policy.minimum_risk_multiplier
                ),
                "recovery_threshold": (
                    policy.recovery_threshold
                ),
                "original_invested_weight": (
                    original_invested_weight
                ),
                "adjusted_invested_weight": (
                    adjusted_invested_weight
                ),
                "original_cash_weight": float(
                    allocation.cash_weight
                ),
                "adjusted_cash_weight": float(
                    adjusted_allocation.cash_weight
                ),
            },
        )

    # ==========================================================
    # Drawdown calculations
    # ==========================================================

    @staticmethod
    def calculate_drawdown_series(
        values: Sequence[float],
    ) -> np.ndarray:
        """
        Calculate drawdown at every observation.
        """

        array = (
            DrawdownAwareDeRiskingEngine
            ._validate_value_series(
                values
            )
        )

        running_peaks = np.maximum.accumulate(
            array
        )

        return (
            running_peaks - array
        ) / running_peaks

    @staticmethod
    def calculate_current_drawdown(
        values: Sequence[float],
    ) -> float:
        """
        Calculate drawdown at the latest observation.
        """

        drawdowns = (
            DrawdownAwareDeRiskingEngine
            .calculate_drawdown_series(
                values
            )
        )

        return float(
            drawdowns[-1]
        )

    @staticmethod
    def calculate_maximum_drawdown(
        values: Sequence[float],
    ) -> float:
        """
        Calculate maximum historical drawdown.
        """

        drawdowns = (
            DrawdownAwareDeRiskingEngine
            .calculate_drawdown_series(
                values
            )
        )

        return float(
            np.max(
                drawdowns
            )
        )

    # ==========================================================
    # Risk multiplier
    # ==========================================================

    def calculate_risk_multiplier(
        self,
        *,
        current_drawdown: float,
        policy: DrawdownPolicy,
        recovering: bool = False,
    ) -> float:
        """
        Convert current drawdown into a portfolio
        risk multiplier.
        """

        if not np.isfinite(
            current_drawdown
        ):
            raise DrawdownEngineError(
                "current_drawdown must be finite."
            )

        if (
            current_drawdown < 0.0
            or current_drawdown > 1.0
        ):
            raise DrawdownEngineError(
                "current_drawdown must be "
                "between 0 and 1."
            )

        trigger = float(
            policy.trigger_drawdown
        )

        if (
            current_drawdown
            < trigger - self.tolerance
        ):
            return 1.0

        if recovering:
            if (
                policy.recovery_threshold
                is not None
                and current_drawdown
                <= policy.recovery_threshold
                + self.tolerance
            ):
                return 1.0

        target = float(
            policy.target_risk_multiplier
        )

        minimum = float(
            policy.minimum_risk_multiplier
        )

        if trigger >= 1.0:
            return max(
                minimum,
                min(
                    target,
                    1.0,
                ),
            )

        excess_drawdown = max(
            current_drawdown - trigger,
            0.0,
        )

        stress_range = max(
            1.0 - trigger,
            self.tolerance,
        )

        stress_fraction = min(
            excess_drawdown
            / stress_range,
            1.0,
        )

        multiplier = (
            target
            - stress_fraction
            * (
                target
                - minimum
            )
        )

        return float(
            np.clip(
                multiplier,
                minimum,
                1.0,
            )
        )

    # ==========================================================
    # Portfolio adjustment
    # ==========================================================

    def apply_risk_multiplier(
        self,
        *,
        allocation: PortfolioAllocation,
        risk_multiplier: float,
    ) -> PortfolioAllocation:
        """
        Scale invested portfolio exposure and move
        released capital into cash.
        """

        self._validate_allocation(
            allocation
        )

        if not np.isfinite(
            risk_multiplier
        ):
            raise DrawdownEngineError(
                "risk_multiplier must be finite."
            )

        if (
            risk_multiplier < 0.0
            or risk_multiplier > 1.0
        ):
            raise DrawdownEngineError(
                "risk_multiplier must be "
                "between 0 and 1."
            )

        adjusted_weights = {
            symbol: float(
                weight
                * risk_multiplier
            )
            for symbol, weight
            in allocation.weights.items()
        }

        adjusted_invested_weight = float(
            sum(
                adjusted_weights.values()
            )
        )

        adjusted_cash_weight = (
            1.0
            - adjusted_invested_weight
        )

        adjusted = PortfolioAllocation(
            weights=adjusted_weights,
            cash_weight=float(
                adjusted_cash_weight
            ),
            gross_exposure=float(
                sum(
                    abs(weight)
                    for weight
                    in adjusted_weights.values()
                )
            ),
            net_exposure=float(
                adjusted_invested_weight
            ),
            metadata={
                **allocation.metadata,
                "drawdown_adjusted": True,
                "risk_multiplier": float(
                    risk_multiplier
                ),
            },
        )

        adjusted.validate(
            tolerance=1e-6
        )

        return adjusted

    # ==========================================================
    # Recovery logic
    # ==========================================================

    def _is_recovering(
        self,
        *,
        values: Sequence[float],
        metrics: DrawdownMetrics,
        policy: DrawdownPolicy,
    ) -> bool:
        """
        Detect whether the portfolio is recovering
        after a historical drawdown.

        Recovery means:

        - a drawdown has occurred,
        - the latest value is above the maximum-drawdown trough,
        - current drawdown is smaller than maximum drawdown.
        """

        if policy.recovery_threshold is None:
            return False

        array = self._validate_value_series(
            values
        )

        if len(array) < 2:
            return False

        if (
            metrics.maximum_drawdown
            <= self.tolerance
        ):
            return False

        trough_value = float(
            array[
                metrics.trough_index
            ]
        )

        current_value = float(
            array[-1]
        )

        return bool(
            current_value
            > trough_value
            + self.tolerance
            and metrics.current_drawdown
            < metrics.maximum_drawdown
            - self.tolerance
        )

    # ==========================================================
    # Validation
    # ==========================================================

    @staticmethod
    def _validate_value_series(
        values: Sequence[float],
    ) -> np.ndarray:
        array = np.asarray(
            values,
            dtype=float,
        )

        if array.ndim != 1:
            raise DrawdownEngineError(
                "Portfolio value series must "
                "be one-dimensional."
            )

        if len(array) < 1:
            raise DrawdownEngineError(
                "Portfolio value series "
                "cannot be empty."
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise DrawdownEngineError(
                "Portfolio value series contains "
                "non-finite values."
            )

        if np.any(
            array <= 0.0
        ):
            raise DrawdownEngineError(
                "Portfolio values must "
                "be positive."
            )

        return array

    @staticmethod
    def _validate_allocation(
        allocation: PortfolioAllocation,
    ) -> None:
        if not allocation.weights:
            raise DrawdownEngineError(
                "Portfolio allocation cannot be empty."
            )

        if not np.isfinite(
            float(
                allocation.cash_weight
            )
        ):
            raise DrawdownEngineError(
                "cash_weight must be finite."
            )

        if allocation.cash_weight < 0.0:
            raise DrawdownEngineError(
                "cash_weight cannot be negative."
            )

        for symbol, weight in (
            allocation.weights.items()
        ):
            if not symbol:
                raise DrawdownEngineError(
                    "Portfolio symbol cannot be empty."
                )

            if not np.isfinite(
                float(weight)
            ):
                raise DrawdownEngineError(
                    f"Weight for {symbol} "
                    "must be finite."
                )

            if weight < 0.0:
                raise DrawdownEngineError(
                    "Drawdown engine currently "
                    "supports long-only portfolios."
                )

        total_weight = (
            float(
                sum(
                    allocation.weights.values()
                )
            )
            + float(
                allocation.cash_weight
            )
        )

        if abs(
            total_weight - 1.0
        ) > 1e-6:
            raise DrawdownEngineError(
                "Portfolio weights plus cash "
                "must sum to 1.0."
            )


__all__ = [
    "DeRiskingResult",
    "DrawdownAwareDeRiskingEngine",
    "DrawdownEngineError",
    "DrawdownMetrics",
]