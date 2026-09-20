
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Mapping, Optional, Tuple

from portfolio_digital_twin.risk_monitor import (
    PortfolioRiskMonitor,
    RiskLimits,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
)


class StressTestingError(ValueError):
    """Invalid stress scenario or portfolio state."""


@dataclass(frozen=True)
class StressScenario:
    name: str
    asset_returns: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise StressTestingError(
                "Scenario name cannot be empty."
            )

        for symbol, asset_return in self.asset_returns.items():
            if not symbol or not symbol.strip():
                raise StressTestingError(
                    "Scenario asset symbol cannot be empty."
                )

            if (
                not isfinite(asset_return)
                or asset_return < -1.0
            ):
                raise StressTestingError(
                    f"Invalid scenario return for {symbol}."
                )


@dataclass(frozen=True)
class StressTestResult:
    scenario_name: str
    nav_before: float
    nav_after: float
    pnl: float
    pnl_pct: float
    cash_after: float
    positions_after: Dict[str, float]
    drawdown_after: float
    risk_violations: Tuple[str, ...]
    within_limits: bool


class PortfolioStressTestingEngine:
    """
    Read-only scenario stress testing for the Portfolio Digital Twin.

    Scenario returns are applied to current position values.
    Unspecified assets receive a zero return.
    Cash is assumed unchanged.

    The original ShadowSimulationEngine is never updated.
    """

    def __init__(
        self,
        risk_limits: Optional[RiskLimits] = None,
    ) -> None:
        self.risk_monitor = PortfolioRiskMonitor(
            limits=risk_limits
        )

    def run(
        self,
        shadow: ShadowSimulationEngine,
        scenario: StressScenario,
    ) -> StressTestResult:
        nav_before = shadow.nav

        if not isfinite(nav_before) or nav_before <= 0.0:
            raise StressTestingError(
                "Portfolio NAV must be finite and positive."
            )

        cash_after = shadow.cash

        if not isfinite(cash_after) or cash_after < 0.0:
            raise StressTestingError(
                "Portfolio cash must be finite and non-negative."
            )

        positions_after: Dict[str, float] = {}

        for symbol, position_value in shadow.positions.items():
            if (
                not isfinite(position_value)
                or position_value < 0.0
            ):
                raise StressTestingError(
                    f"Invalid position value for {symbol}."
                )

            asset_return = scenario.asset_returns.get(
                symbol,
                0.0,
            )

            positions_after[symbol] = (
                position_value * (1.0 + asset_return)
            )

        nav_after = cash_after + sum(
            positions_after.values()
        )

        if not isfinite(nav_after) or nav_after <= 0.0:
            raise StressTestingError(
                "Stress scenario produces non-positive NAV."
            )

        pnl = nav_after - nav_before
        pnl_pct = pnl / nav_before

        # Evaluate the stressed portfolio without changing the
        # original engine or its snapshot history.
        stressed_shadow = object.__new__(
            ShadowSimulationEngine
        )

        stressed_shadow.cash = cash_after
        stressed_shadow.positions = positions_after

        # The risk monitor reads NAV and drawdown from the shadow
        # engine. Build an independent, minimal stressed history
        # using the original peak NAV and the scenario NAV.
        peak_nav = max(
            snapshot.nav
            for snapshot in shadow.history
        )

        if not isfinite(peak_nav) or peak_nav <= 0.0:
            raise StressTestingError(
                "Portfolio history contains an invalid peak NAV."
            )

        from portfolio_digital_twin.shadow_simulation import (
            ShadowSnapshot,
        )

        stressed_shadow.history = [
            ShadowSnapshot(
                timestamp=shadow.history[-1].timestamp,
                nav=peak_nav,
                cash=peak_nav,
                positions={},
                weights={},
                pnl=0.0,
                pnl_pct=0.0,
                drawdown=0.0,
                gross_exposure=0.0,
                net_exposure=0.0,
            ),
            ShadowSnapshot(
                timestamp=shadow.history[-1].timestamp,
                nav=nav_after,
                cash=cash_after,
                positions=dict(positions_after),
                weights={
                    symbol: value / nav_after
                    for symbol, value in positions_after.items()
                },
                pnl=pnl,
                pnl_pct=pnl_pct,
                drawdown=min(
                    0.0,
                    nav_after / peak_nav - 1.0,
                ),
                gross_exposure=sum(
                    positions_after.values()
                ) / nav_after,
                net_exposure=sum(
                    positions_after.values()
                ) / nav_after,
            ),
        ]

        stressed_shadow.initial_capital = (
            shadow.initial_capital
        )

        risk_report = self.risk_monitor.evaluate(
            stressed_shadow
        )

        return StressTestResult(
            scenario_name=scenario.name,
            nav_before=nav_before,
            nav_after=nav_after,
            pnl=pnl,
            pnl_pct=pnl_pct,
            cash_after=cash_after,
            positions_after=dict(positions_after),
            drawdown_after=risk_report.drawdown,
            risk_violations=risk_report.violations,
            within_limits=risk_report.within_limits,
        )


__all__ = [
    "PortfolioStressTestingEngine",
    "StressScenario",
    "StressTestResult",
    "StressTestingError",
]