
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import List, Optional

from portfolio_digital_twin.execution_costs import (
    ExecutionCostEngine,
    ExecutionCostReport,
)
from portfolio_digital_twin.shadow_simulation import (
    ShadowSimulationEngine,
    ShadowSnapshot,
)


class TradeExecutionError(ValueError):
    """Invalid virtual trade or execution failure."""


@dataclass(frozen=True)
class VirtualTrade:
    symbol: str
    side: str
    notional: float
    daily_volume_notional: float
    volatility: float = 0.02


@dataclass(frozen=True)
class TradeExecutionResult:
    trade_id: int
    timestamp: datetime
    symbol: str
    side: str
    notional: float
    execution_cost: ExecutionCostReport
    cash_before: float
    cash_after: float
    nav_before: float
    nav_after: float
    turnover: float
    snapshot: ShadowSnapshot


@dataclass
class TradeExecutionEngine:
    """
    v6.0.4 Portfolio Digital Twin.

    Cash-funded, long-only virtual execution.

    Integrates:
        ShadowSimulationEngine
        ExecutionCostEngine

    No broker connection or real-money execution.
    """

    shadow: ShadowSimulationEngine

    cost_engine: ExecutionCostEngine = field(
        default_factory=ExecutionCostEngine
    )

    trade_history: List[TradeExecutionResult] = field(
        default_factory=list,
        init=False,
    )

    cumulative_turnover: float = field(
        default=0.0,
        init=False,
    )

    cumulative_execution_cost: float = field(
        default=0.0,
        init=False,
    )

    def execute(
        self,
        trade: VirtualTrade,
    ) -> TradeExecutionResult:

        self._validate_trade(trade)

        symbol = trade.symbol
        side = trade.side.upper()
        notional = trade.notional

        cash_before = self.shadow.cash
        nav_before = self.shadow.nav

        if nav_before <= 0:
            raise TradeExecutionError(
                "Portfolio NAV must be positive."
            )

        if symbol not in self.shadow.positions:
            raise TradeExecutionError(
                f"Unknown portfolio symbol: {symbol}"
            )

        cost = self.cost_engine.estimate(
            symbol=symbol,
            order_notional=notional,
            daily_volume_notional=(
                trade.daily_volume_notional
            ),
            volatility=trade.volatility,
        )

        if cost.liquidity_warning:
            raise TradeExecutionError(
                "Trade exceeds maximum "
                "liquidity participation rate."
            )

        current_position = (
            self.shadow.positions[symbol]
        )

        if side == "BUY":
            total_cash_required = (
                notional + cost.total_cost
            )

            if total_cash_required > cash_before + 1e-9:
                raise TradeExecutionError(
                    "Insufficient virtual cash."
                )

            new_cash = (
                cash_before - total_cash_required
            )

            new_position = (
                current_position + notional
            )

        else:
            if notional > current_position + 1e-9:
                raise TradeExecutionError(
                    "Insufficient virtual position."
                )

            new_cash = (
                cash_before
                + notional
                - cost.total_cost
            )

            if new_cash < -1e-9:
                raise TradeExecutionError(
                    "Execution costs exceed available proceeds."
                )

            new_position = (
                current_position - notional
            )

        # Commit only after all validations succeed.
        self.shadow.cash = max(0.0, new_cash)

        self.shadow.positions[symbol] = max(
            0.0,
            new_position,
        )

        snapshot = (
            self.shadow._record_snapshot()
        )

        turnover = notional / nav_before

        self.cumulative_turnover += turnover

        self.cumulative_execution_cost += (
            cost.total_cost
        )

        result = TradeExecutionResult(
            trade_id=len(self.trade_history) + 1,
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            notional=notional,
            execution_cost=cost,
            cash_before=cash_before,
            cash_after=self.shadow.cash,
            nav_before=nav_before,
            nav_after=self.shadow.nav,
            turnover=turnover,
            snapshot=snapshot,
        )

        self.trade_history.append(result)

        return result

    @staticmethod
    def _validate_trade(
        trade: VirtualTrade,
    ) -> None:

        if not trade.symbol or not trade.symbol.strip():
            raise TradeExecutionError(
                "Trade symbol cannot be empty."
            )

        if trade.side.upper() not in {"BUY", "SELL"}:
            raise TradeExecutionError(
                "Trade side must be BUY or SELL."
            )

        if (
            not isfinite(trade.notional)
            or trade.notional <= 0
        ):
            raise TradeExecutionError(
                "Trade notional must be finite and positive."
            )

        if (
            not isfinite(trade.daily_volume_notional)
            or trade.daily_volume_notional <= 0
        ):
            raise TradeExecutionError(
                "Daily volume must be finite and positive."
            )

        if (
            not isfinite(trade.volatility)
            or trade.volatility < 0
        ):
            raise TradeExecutionError(
                "Volatility must be finite and non-negative."
            )


__all__ = [
    "TradeExecutionEngine",
    "TradeExecutionError",
    "TradeExecutionResult",
    "VirtualTrade",
]