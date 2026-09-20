
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Optional


class ExecutionCostError(ValueError):
    """Invalid execution cost model input."""


@dataclass(frozen=True)
class ExecutionCostConfig:
    """
    Execution cost parameters.

    All basis-point values use:
    1 bps = 0.0001 = 0.01%.
    """

    commission_bps: float = 2.0
    spread_bps: float = 5.0
    base_slippage_bps: float = 3.0
    impact_coefficient: float = 0.10
    max_participation_rate: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "commission_bps",
            "spread_bps",
            "base_slippage_bps",
            "impact_coefficient",
        ):
            value = getattr(self, name)

            if not isfinite(value) or value < 0:
                raise ExecutionCostError(
                    f"{name} must be finite and non-negative."
                )

        if (
            not isfinite(self.max_participation_rate)
            or not 0 < self.max_participation_rate <= 1
        ):
            raise ExecutionCostError(
                "max_participation_rate must be in (0, 1]."
            )


@dataclass(frozen=True)
class ExecutionCostReport:
    symbol: str
    order_notional: float
    commission: float
    spread_cost: float
    slippage_cost: float
    market_impact_cost: float
    total_cost: float
    total_cost_bps: float
    participation_rate: float
    liquidity_warning: bool


class ExecutionCostEngine:
    """
    v6.0.4 Portfolio Digital Twin.

    Deterministic virtual execution cost model.

    No real orders or broker connections.
    """

    def __init__(
        self,
        config: Optional[ExecutionCostConfig] = None,
    ) -> None:
        self.config = config or ExecutionCostConfig()

    def estimate(
        self,
        *,
        symbol: str,
        order_notional: float,
        daily_volume_notional: float,
        volatility: float = 0.02,
    ) -> ExecutionCostReport:
        """
        Estimate transaction costs.

        order_notional:
            Absolute order value in portfolio currency.

        daily_volume_notional:
            Daily traded value in the same currency.

        volatility:
            Decimal volatility estimate.
            Example: 0.02 = 2%.

        Spread cost assumes spread_bps is the
        full bid-ask spread, so the model charges
        half the spread per execution.

        Market impact uses a simplified
        square-root participation model.
        """

        if not symbol or not symbol.strip():
            raise ExecutionCostError(
                "symbol cannot be empty."
            )

        for name, value in (
            ("order_notional", order_notional),
            ("daily_volume_notional", daily_volume_notional),
            ("volatility", volatility),
        ):
            if not isfinite(value) or value < 0:
                raise ExecutionCostError(
                    f"{name} must be finite and non-negative."
                )

        if daily_volume_notional <= 0:
            raise ExecutionCostError(
                "daily_volume_notional must be positive."
            )

        participation = (
            order_notional / daily_volume_notional
        )

        liquidity_warning = (
            participation
            > self.config.max_participation_rate
        )

        commission = (
            order_notional
            * self.config.commission_bps
            / 10_000
        )

        spread_cost = (
            order_notional
            * self.config.spread_bps
            / 20_000
        )

        slippage_cost = (
            order_notional
            * self.config.base_slippage_bps
            / 10_000
        )

        market_impact_cost = (
            order_notional
            * self.config.impact_coefficient
            * volatility
            * sqrt(participation)
        )

        total_cost = (
            commission
            + spread_cost
            + slippage_cost
            + market_impact_cost
        )

        total_cost_bps = (
            total_cost / order_notional * 10_000
            if order_notional > 0
            else 0.0
        )

        return ExecutionCostReport(
            symbol=symbol,
            order_notional=order_notional,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            market_impact_cost=market_impact_cost,
            total_cost=total_cost,
            total_cost_bps=total_cost_bps,
            participation_rate=participation,
            liquidity_warning=liquidity_warning,
        )


__all__ = [
    "ExecutionCostConfig",
    "ExecutionCostEngine",
    "ExecutionCostError",
    "ExecutionCostReport",
]