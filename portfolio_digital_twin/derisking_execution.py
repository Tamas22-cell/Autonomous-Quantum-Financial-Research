
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Dict, Mapping, Optional, Tuple

from portfolio_digital_twin.regime_derisking import (
    DeRiskingPlan,
)
from portfolio_digital_twin.risk_monitor import (
    PortfolioRiskMonitor,
    RiskReport,
)
from portfolio_digital_twin.trade_execution import (
    TradeExecutionEngine,
    TradeExecutionError,
    TradeExecutionResult,
    VirtualTrade,
)


class DeRiskingExecutionError(ValueError):
    """A de-risking plan cannot be executed safely."""


@dataclass(frozen=True)
class AssetExecutionData:
    daily_volume_notional: float
    volatility: float = 0.02


@dataclass(frozen=True)
class DeRiskingExecutionResult:
    nav_before: float
    nav_after: float
    exposure_before: float
    exposure_after: float
    target_max_exposure: float
    estimated_execution_cost: float
    actual_execution_cost: float
    executed_notional: float
    trades: Tuple[TradeExecutionResult, ...]
    risk_report: RiskReport
    target_reached: bool


class DeRiskingExecutionEngine:
    """
    Executes a de-risking plan inside the virtual portfolio.

    All planned sales are prevalidated before execution.
    This is a simulation only: no real broker orders.

    The engine rejects plans based on a stale portfolio NAV.
    """

    def __init__(
        self,
        trade_engine: TradeExecutionEngine,
        risk_monitor: Optional[PortfolioRiskMonitor] = None,
    ) -> None:
        self.trade_engine = trade_engine
        self.risk_monitor = (
            risk_monitor or PortfolioRiskMonitor()
        )

    def execute(
        self,
        plan: DeRiskingPlan,
        market_data: Mapping[str, AssetExecutionData],
    ) -> DeRiskingExecutionResult:
        shadow = self.trade_engine.shadow

        report_before = self.risk_monitor.evaluate(shadow)

        if not isfinite(plan.nav) or not isfinite(
            plan.target_max_exposure
        ):
            raise DeRiskingExecutionError(
                "Invalid de-risking plan."
            )

        if abs(plan.nav - report_before.nav) > 1e-6:
            raise DeRiskingExecutionError(
                "Stale de-risking plan: portfolio NAV changed."
            )

        if (
            abs(
                plan.current_exposure
                - report_before.gross_exposure
            )
            > 1e-9
        ):
            raise DeRiskingExecutionError(
                "Stale de-risking plan: exposure changed."
            )

        if not plan.action_required or not plan.planned_sales:
            return DeRiskingExecutionResult(
                nav_before=report_before.nav,
                nav_after=report_before.nav,
                exposure_before=report_before.gross_exposure,
                exposure_after=report_before.gross_exposure,
                target_max_exposure=plan.target_max_exposure,
                estimated_execution_cost=0.0,
                actual_execution_cost=0.0,
                executed_notional=0.0,
                trades=(),
                risk_report=report_before,
                target_reached=(
                    report_before.gross_exposure
                    <= plan.target_max_exposure + 1e-9
                ),
            )

        prepared_trades = []
        estimated_execution_cost = 0.0
        total_notional = 0.0
        symbol_totals: Dict[str, float] = {}

        # Validate the complete plan before modifying the portfolio.
        for sale in plan.planned_sales:
            if (
                not isfinite(sale.notional)
                or sale.notional <= 0.0
            ):
                raise DeRiskingExecutionError(
                    "Planned sale notional must be positive."
                )

            if sale.symbol not in shadow.positions:
                raise DeRiskingExecutionError(
                    f"Unknown portfolio symbol: {sale.symbol}"
                )

            if sale.symbol not in market_data:
                raise DeRiskingExecutionError(
                    f"Missing market data for {sale.symbol}"
                )

            data = market_data[sale.symbol]

            if (
                not isfinite(data.daily_volume_notional)
                or data.daily_volume_notional <= 0.0
                or not isfinite(data.volatility)
                or data.volatility < 0.0
            ):
                raise DeRiskingExecutionError(
                    f"Invalid market data for {sale.symbol}"
                )

            symbol_totals[sale.symbol] = (
                symbol_totals.get(sale.symbol, 0.0)
                + sale.notional
            )

            if (
                symbol_totals[sale.symbol]
                > shadow.positions[sale.symbol] + 1e-9
            ):
                raise DeRiskingExecutionError(
                    f"Insufficient virtual position: {sale.symbol}"
                )

            trade = VirtualTrade(
                symbol=sale.symbol,
                side="SELL",
                notional=sale.notional,
                daily_volume_notional=data.daily_volume_notional,
                volatility=data.volatility,
            )

            cost = self.trade_engine.cost_engine.estimate(
                symbol=sale.symbol,
                order_notional=sale.notional,
                daily_volume_notional=data.daily_volume_notional,
                volatility=data.volatility,
            )

            if cost.liquidity_warning:
                raise DeRiskingExecutionError(
                    f"Liquidity limit exceeded: {sale.symbol}"
                )

            if cost.total_cost > sale.notional + 1e-9:
                raise DeRiskingExecutionError(
                    f"Execution costs exceed proceeds: {sale.symbol}"
                )

            estimated_execution_cost += cost.total_cost
            total_notional += sale.notional
            prepared_trades.append(trade)

        # Prevalidate the total cost and resulting NAV.
        if (
            report_before.nav - estimated_execution_cost
            <= 0.0
        ):
            raise DeRiskingExecutionError(
                "Execution costs produce non-positive NAV."
            )

        # Execute only after every planned sale passes validation.
        results = []

        for trade in prepared_trades:
            try:
                result = self.trade_engine.execute(trade)
            except TradeExecutionError as exc:
                raise DeRiskingExecutionError(
                    f"Virtual execution failed: {exc}"
                ) from exc

            results.append(result)

        report_after = self.risk_monitor.evaluate(shadow)

        actual_execution_cost = sum(
            result.execution_cost.total_cost
            for result in results
        )

        return DeRiskingExecutionResult(
            nav_before=report_before.nav,
            nav_after=report_after.nav,
            exposure_before=report_before.gross_exposure,
            exposure_after=report_after.gross_exposure,
            target_max_exposure=plan.target_max_exposure,
            estimated_execution_cost=estimated_execution_cost,
            actual_execution_cost=actual_execution_cost,
            executed_notional=total_notional,
            trades=tuple(results),
            risk_report=report_after,
            target_reached=(
                report_after.gross_exposure
                <= plan.target_max_exposure + 1e-9
            ),
        )


__all__ = [
    "AssetExecutionData",
    "DeRiskingExecutionEngine",
    "DeRiskingExecutionError",
    "DeRiskingExecutionResult",
]