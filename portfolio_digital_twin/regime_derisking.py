
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Dict, Optional, Tuple

from portfolio_digital_twin.risk_monitor import (
    PortfolioRiskMonitor,
    RiskReport,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)


class DeRiskingError(ValueError):
    """Invalid market regime or de-risking configuration."""


class MarketRegime(str, Enum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"


@dataclass(frozen=True)
class DeRiskingConfig:
    risk_on_max_exposure: float = 1.00
    neutral_max_exposure: float = 0.85
    risk_off_max_exposure: float = 0.60
    crisis_max_exposure: float = 0.30
    drawdown_trigger: float = 0.15
    drawdown_max_exposure: float = 0.50
    minimum_trade_notional: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "risk_on_max_exposure",
            "neutral_max_exposure",
            "risk_off_max_exposure",
            "crisis_max_exposure",
            "drawdown_trigger",
            "drawdown_max_exposure",
        ):
            value = getattr(self, name)

            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise DeRiskingError(
                    f"{name} must be finite and between 0 and 1."
                )

        if (
            not isfinite(self.minimum_trade_notional)
            or self.minimum_trade_notional < 0.0
        ):
            raise DeRiskingError(
                "minimum_trade_notional must be finite "
                "and non-negative."
            )

        if not (
            self.risk_on_max_exposure
            >= self.neutral_max_exposure
            >= self.risk_off_max_exposure
            >= self.crisis_max_exposure
        ):
            raise DeRiskingError(
                "Exposure limits must decrease "
                "from RISK_ON to CRISIS."
            )


@dataclass(frozen=True)
class PlannedSale:
    symbol: str
    notional: float


@dataclass(frozen=True)
class DeRiskingPlan:
    regime: MarketRegime
    nav: float
    current_exposure: float
    target_max_exposure: float
    required_reduction: float
    planned_sales: Tuple[PlannedSale, ...]
    reasons: Tuple[str, ...]
    action_required: bool


class RegimeDeRiskingEngine:
    """
    Read-only, regime-aware de-risking planner.

    Uses current long-only position values to prepare
    proportional virtual sales.

    Does not execute trades or modify portfolio state.
    """

    def __init__(
        self,
        config: Optional[DeRiskingConfig] = None,
        risk_monitor: Optional[PortfolioRiskMonitor] = None,
    ) -> None:
        self.config = config or DeRiskingConfig()
        self.risk_monitor = (
            risk_monitor or PortfolioRiskMonitor()
        )

    def plan(
        self,
        shadow: ShadowSimulationEngine,
        regime: MarketRegime,
    ) -> DeRiskingPlan:
        if not isinstance(regime, MarketRegime):
            raise DeRiskingError(
                "regime must be a MarketRegime value."
            )

        report: RiskReport = self.risk_monitor.evaluate(
            shadow
        )

        nav = report.nav
        current_exposure = report.gross_exposure

        regime_limits: Dict[MarketRegime, float] = {
            MarketRegime.RISK_ON:
                self.config.risk_on_max_exposure,
            MarketRegime.NEUTRAL:
                self.config.neutral_max_exposure,
            MarketRegime.RISK_OFF:
                self.config.risk_off_max_exposure,
            MarketRegime.CRISIS:
                self.config.crisis_max_exposure,
        }

        target_max_exposure = regime_limits[regime]
        reasons = []

        if current_exposure > target_max_exposure + 1e-9:
            reasons.append("REGIME_EXPOSURE_LIMIT")

        if report.drawdown >= self.config.drawdown_trigger:
            target_max_exposure = min(
                target_max_exposure,
                self.config.drawdown_max_exposure,
            )

            if current_exposure > target_max_exposure + 1e-9:
                reasons.append("DRAWDOWN_DERISKING")

        required_reduction = max(
            0.0,
            (
                current_exposure
                - target_max_exposure
            ) * nav,
        )

        if required_reduction < self.config.minimum_trade_notional:
            required_reduction = 0.0

        planned_sales = []

        if required_reduction > 0.0 and current_exposure > 0.0:
            total_position_value = sum(
                shadow.positions.values()
            )

            for symbol, position_value in shadow.positions.items():
                if position_value <= 0.0:
                    continue

                proportion = (
                    position_value / total_position_value
                )

                sale_notional = (
                    required_reduction * proportion
                )

                if sale_notional >= self.config.minimum_trade_notional:
                    planned_sales.append(
                        PlannedSale(
                            symbol=symbol,
                            notional=sale_notional,
                        )
                    )

        action_required = bool(planned_sales)

        return DeRiskingPlan(
            regime=regime,
            nav=nav,
            current_exposure=current_exposure,
            target_max_exposure=target_max_exposure,
            required_reduction=required_reduction,
            planned_sales=tuple(planned_sales),
            reasons=tuple(reasons),
            action_required=action_required,
        )


__all__ = [
    "DeRiskingConfig",
    "DeRiskingError",
    "DeRiskingPlan",
    "MarketRegime",
    "PlannedSale",
    "RegimeDeRiskingEngine",
]