
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isfinite
from typing import Dict, List, Mapping

from portfolio_construction.models import PortfolioAllocation


@dataclass(frozen=True)
class ShadowSnapshot:
    timestamp: datetime
    nav: float
    cash: float
    positions: Dict[str, float]
    weights: Dict[str, float]
    pnl: float
    pnl_pct: float
    drawdown: float
    gross_exposure: float
    net_exposure: float


@dataclass
class ShadowSimulationEngine:
    """
    v6.0.4 Portfolio Digital Twin.

    Long-only virtual portfolio simulation.
    No real trading or broker connections.
    """

    initial_capital: float
    allocation: PortfolioAllocation

    cash: float = field(init=False)
    positions: Dict[str, float] = field(init=False)

    history: List[ShadowSnapshot] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            not isfinite(self.initial_capital)
            or self.initial_capital <= 0
        ):
            raise ValueError(
                "initial_capital must be finite and positive."
            )

        self.allocation.validate()

        self.cash = (
            self.initial_capital
            * self.allocation.cash_weight
        )

        self.positions = {
            symbol: self.initial_capital * weight
            for symbol, weight
            in self.allocation.weights.items()
        }

        self._record_snapshot()

    @property
    def nav(self) -> float:
        return self.cash + sum(self.positions.values())

    @property
    def pnl(self) -> float:
        return self.nav - self.initial_capital

    @property
    def pnl_pct(self) -> float:
        return self.nav / self.initial_capital - 1.0

    @property
    def drawdown(self) -> float:
        peak = max(
            [self.nav]
            + [snapshot.nav for snapshot in self.history]
        )

        return self.nav / peak - 1.0

    def update(
        self,
        returns: Mapping[str, float],
    ) -> ShadowSnapshot:
        """
        Apply one period of asset returns.

        Example:
        0.05 = +5%
        -0.10 = -10%
        """

        if set(returns) != set(self.positions):
            raise ValueError(
                "Returns must contain exactly "
                "the portfolio symbols."
            )

        for symbol, value in returns.items():
            if not isfinite(value) or value < -1.0:
                raise ValueError(
                    f"Invalid return for {symbol}: {value}"
                )

        for symbol, value in returns.items():
            self.positions[symbol] *= 1.0 + value

        return self._record_snapshot()

    def _record_snapshot(self) -> ShadowSnapshot:
        nav = self.nav

        weights = {
            symbol: value / nav if nav > 0 else 0.0
            for symbol, value in self.positions.items()
        }

        exposure = sum(
            abs(value)
            for value in self.positions.values()
        )

        snapshot = ShadowSnapshot(
            timestamp=datetime.now(timezone.utc),
            nav=nav,
            cash=self.cash,
            positions=dict(self.positions),
            weights=weights,
            pnl=self.pnl,
            pnl_pct=self.pnl_pct,
            drawdown=(
                self.drawdown if self.history else 0.0
            ),
            gross_exposure=(
                exposure / nav if nav > 0 else 0.0
            ),
            net_exposure=(
                sum(self.positions.values()) / nav
                if nav > 0 else 0.0
            ),
        )

        self.history.append(snapshot)

        return snapshot