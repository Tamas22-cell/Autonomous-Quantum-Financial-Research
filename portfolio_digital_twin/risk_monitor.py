
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Optional, Tuple

from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)


class RiskMonitorError(ValueError):
    """Invalid risk monitor configuration or portfolio state."""


@dataclass(frozen=True)
class RiskLimits:
    max_single_position_weight: float = 0.40
    max_gross_exposure: float = 1.00
    max_drawdown: float = 0.20
    min_cash_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "max_single_position_weight",
            "max_gross_exposure",
            "max_drawdown",
            "min_cash_weight",
        ):
            value = getattr(self, name)

            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise RiskMonitorError(
                    f"{name} must be finite and between 0 and 1."
                )


@dataclass(frozen=True)
class RiskReport:
    nav: float
    cash_weight: float
    position_weights: Dict[str, float]
    largest_position_symbol: Optional[str]
    largest_position_weight: float
    gross_exposure: float
    net_exposure: float
    drawdown: float
    violations: Tuple[str, ...]
    within_limits: bool


class PortfolioRiskMonitor:
    """
    Read-only risk monitoring for the Portfolio Digital Twin.

    Drawdown is reported as a positive loss magnitude:
    a 20% decline is represented as 0.20.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
    ) -> None:
        self.limits = limits or RiskLimits()

    def evaluate(
        self,
        shadow: ShadowSimulationEngine,
    ) -> RiskReport:
        nav = shadow.nav

        if not isfinite(nav) or nav <= 0.0:
            raise RiskMonitorError(
                "Portfolio NAV must be finite and positive."
            )

        cash = shadow.cash

        if not isfinite(cash) or cash < 0.0:
            raise RiskMonitorError(
                "Portfolio cash must be finite and non-negative."
            )

        position_weights: Dict[str, float] = {}

        for symbol, position_value in shadow.positions.items():
            if not isfinite(position_value) or position_value < 0.0:
                raise RiskMonitorError(
                    f"Invalid position value for {symbol}."
                )

            position_weights[symbol] = position_value / nav

        cash_weight = cash / nav

        gross_exposure = sum(
            abs(weight)
            for weight in position_weights.values()
        )

        net_exposure = sum(position_weights.values())

        if position_weights:
            largest_position_symbol = max(
                position_weights,
                key=position_weights.get,
            )

            largest_position_weight = position_weights[
                largest_position_symbol
            ]
        else:
            largest_position_symbol = None
            largest_position_weight = 0.0

        raw_drawdown = shadow.drawdown

        if not isfinite(raw_drawdown):
            raise RiskMonitorError(
                "Portfolio drawdown must be finite."
            )

        # Shadow Simulation stores drawdown as a negative return.
        # Risk Monitor reports the positive magnitude of that loss.
        drawdown = max(0.0, -raw_drawdown)

        violations = []

        if (
            largest_position_weight
            > self.limits.max_single_position_weight
        ):
            violations.append(
                "SINGLE_POSITION_CONCENTRATION"
            )

        if (
            gross_exposure
            > self.limits.max_gross_exposure + 1e-9
        ):
            violations.append(
                "GROSS_EXPOSURE_LIMIT"
            )

        if drawdown > self.limits.max_drawdown + 1e-9:
            violations.append(
                "MAX_DRAWDOWN"
            )

        if cash_weight < self.limits.min_cash_weight:
            violations.append(
                "MIN_CASH_WEIGHT"
            )

        return RiskReport(
            nav=nav,
            cash_weight=cash_weight,
            position_weights=position_weights,
            largest_position_symbol=largest_position_symbol,
            largest_position_weight=largest_position_weight,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            drawdown=drawdown,
            violations=tuple(violations),
            within_limits=len(violations) == 0,
        )


__all__ = [
    "PortfolioRiskMonitor",
    "RiskLimits",
    "RiskMonitorError",
    "RiskReport",
]