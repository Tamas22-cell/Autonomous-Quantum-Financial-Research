from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


class ShadowFundDigitalTwinError(ValueError):
    """Shadow fund and digital twin error."""


class VirtualOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ScenarioSeverity(str, Enum):
    NONE = "NONE"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ShadowFundDigitalTwinError(
            f"{name} must be a non-empty string."
        )


def _require_positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ShadowFundDigitalTwinError(
            f"{name} must be a finite positive number."
        )


def _require_non_negative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0.0
    ):
        raise ShadowFundDigitalTwinError(
            f"{name} must be a finite non-negative number."
        )


@dataclass(frozen=True)
class VirtualFill:
    fill_id: str
    asset_id: str
    side: VirtualOrderSide
    quantity: float
    price: float
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    impact_cost: float = 0.0
    executed_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_text(self.fill_id, "fill_id")
        _require_text(self.asset_id, "asset_id")

        if not isinstance(
            self.side,
            VirtualOrderSide,
        ):
            raise ShadowFundDigitalTwinError(
                "side must be a VirtualOrderSide."
            )

        _require_positive(self.quantity, "quantity")
        _require_positive(self.price, "price")

        for name, value in (
            ("commission", self.commission),
            ("spread_cost", self.spread_cost),
            ("slippage_cost", self.slippage_cost),
            ("impact_cost", self.impact_cost),
        ):
            _require_non_negative(value, name)

        if (
            not isinstance(self.executed_at, datetime)
            or self.executed_at.tzinfo is None
            or self.executed_at.utcoffset() is None
        ):
            raise ShadowFundDigitalTwinError(
                "executed_at must be timezone-aware."
            )

        if not isinstance(self.metadata, dict):
            raise ShadowFundDigitalTwinError(
                "metadata must be a dictionary."
            )

    @property
    def gross_notional(self) -> float:
        return self.quantity * self.price

    @property
    def total_cost(self) -> float:
        return (
            self.commission
            + self.spread_cost
            + self.slippage_cost
            + self.impact_cost
        )

    @property
    def net_cash_flow(self) -> float:
        if self.side == VirtualOrderSide.BUY:
            return -self.gross_notional - self.total_cost

        return self.gross_notional - self.total_cost


@dataclass(frozen=True)
class ShadowPosition:
    asset_id: str
    quantity: float
    average_price: float
    market_price: float
    realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.asset_id, "asset_id")
        _require_non_negative(self.quantity, "quantity")
        _require_non_negative(
            self.average_price,
            "average_price",
        )
        _require_positive(
            self.market_price,
            "market_price",
        )

        if (
            isinstance(self.realized_pnl, bool)
            or not isinstance(
                self.realized_pnl,
                (int, float),
            )
            or not isfinite(self.realized_pnl)
        ):
            raise ShadowFundDigitalTwinError(
                "realized_pnl must be finite."
            )

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def unrealized_pnl(self) -> float:
        return (
            self.quantity
            * (self.market_price - self.average_price)
        )

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass(frozen=True)
class StressScenario:
    scenario_id: str
    name: str
    price_shocks: Mapping[str, float]
    liquidity_factor: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        _require_text(self.scenario_id, "scenario_id")
        _require_text(self.name, "name")

        if not isinstance(
            self.price_shocks,
            Mapping,
        ):
            raise ShadowFundDigitalTwinError(
                "price_shocks must be a mapping."
            )

        for asset_id, shock in self.price_shocks.items():
            _require_text(asset_id, "asset_id")

            if (
                isinstance(shock, bool)
                or not isinstance(shock, (int, float))
                or not isfinite(shock)
                or shock <= -1.0
            ):
                raise ShadowFundDigitalTwinError(
                    "Price shocks must be finite and "
                    "greater than -1."
                )

        if (
            isinstance(self.liquidity_factor, bool)
            or not isinstance(
                self.liquidity_factor,
                (int, float),
            )
            or not isfinite(self.liquidity_factor)
            or not 0.0 < self.liquidity_factor <= 1.0
        ):
            raise ShadowFundDigitalTwinError(
                "liquidity_factor must be between 0 and 1."
            )

        if not isinstance(self.description, str):
            raise ShadowFundDigitalTwinError(
                "description must be a string."
            )


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    scenario_name: str
    starting_value: float
    stressed_value: float
    pnl_change: float
    pnl_change_ratio: float
    severity: ScenarioSeverity
    liquidity_factor: float
    affected_assets: Tuple[str, ...]

    def __post_init__(self) -> None:
        _require_text(self.scenario_id, "scenario_id")
        _require_text(
            self.scenario_name,
            "scenario_name",
        )

        _require_non_negative(
            self.starting_value,
            "starting_value",
        )
        _require_non_negative(
            self.stressed_value,
            "stressed_value",
        )

        if (
            isinstance(self.pnl_change, bool)
            or not isinstance(
                self.pnl_change,
                (int, float),
            )
            or not isfinite(self.pnl_change)
        ):
            raise ShadowFundDigitalTwinError(
                "pnl_change must be finite."
            )

        if (
            isinstance(self.pnl_change_ratio, bool)
            or not isinstance(
                self.pnl_change_ratio,
                (int, float),
            )
            or not isfinite(self.pnl_change_ratio)
        ):
            raise ShadowFundDigitalTwinError(
                "pnl_change_ratio must be finite."
            )

        if not isinstance(
            self.severity,
            ScenarioSeverity,
        ):
            raise ShadowFundDigitalTwinError(
                "severity must be a ScenarioSeverity."
            )

        if (
            isinstance(self.liquidity_factor, bool)
            or not isinstance(
                self.liquidity_factor,
                (int, float),
            )
            or not isfinite(self.liquidity_factor)
            or not 0.0 < self.liquidity_factor <= 1.0
        ):
            raise ShadowFundDigitalTwinError(
                "liquidity_factor must be between 0 and 1."
            )

        if not isinstance(
            self.affected_assets,
            tuple,
        ):
            raise ShadowFundDigitalTwinError(
                "affected_assets must be a tuple."
            )


@dataclass(frozen=True)
class TwinSnapshot:
    snapshot_id: str
    captured_at: datetime
    cash_balance: float
    positions: Tuple[ShadowPosition, ...]
    total_market_value: float
    total_equity: float
    cumulative_costs: float
    cumulative_realized_pnl: float
    cumulative_unrealized_pnl: float

    def __post_init__(self) -> None:
        _require_text(self.snapshot_id, "snapshot_id")

        if (
            not isinstance(self.captured_at, datetime)
            or self.captured_at.tzinfo is None
            or self.captured_at.utcoffset() is None
        ):
            raise ShadowFundDigitalTwinError(
                "captured_at must be timezone-aware."
            )

        _require_non_negative(
            self.cash_balance,
            "cash_balance",
        )
        _require_non_negative(
            self.total_market_value,
            "total_market_value",
        )
        _require_non_negative(
            self.total_equity,
            "total_equity",
        )
        _require_non_negative(
            self.cumulative_costs,
            "cumulative_costs",
        )

        for name, value in (
            (
                "cumulative_realized_pnl",
                self.cumulative_realized_pnl,
            ),
            (
                "cumulative_unrealized_pnl",
                self.cumulative_unrealized_pnl,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ShadowFundDigitalTwinError(
                    f"{name} must be finite."
                )


class ShadowFundDigitalTwin:
    """
    v7.0.9 research-only shadow fund and digital twin.

    All fills are virtual. The component reconstructs portfolio
    state, models transaction costs and runs stress scenarios.
    It never connects to a broker and never deploys real capital.
    """

    VERSION = "v7.0.9"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    BROKER_CONNECTION = False

    def __init__(
        self,
        *,
        initial_cash: float = 100_000.0,
    ) -> None:
        _require_non_negative(
            initial_cash,
            "initial_cash",
        )

        self._initial_cash = initial_cash
        self._cash_balance = initial_cash
        self._fills: Dict[str, VirtualFill] = {}
        self._positions: Dict[str, ShadowPosition] = {}
        self._snapshots: Dict[str, TwinSnapshot] = {}
        self._scenario_results: Dict[
            str,
            ScenarioResult,
        ] = {}
        self._cumulative_costs = 0.0

    @property
    def initial_cash(self) -> float:
        return self._initial_cash

    @property
    def cash_balance(self) -> float:
        return self._cash_balance

    @property
    def fill_count(self) -> int:
        return len(self._fills)

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)

    @property
    def scenario_count(self) -> int:
        return len(self._scenario_results)

    @property
    def cumulative_costs(self) -> float:
        return self._cumulative_costs

    def record_virtual_fill(
        self,
        fill: VirtualFill,
    ) -> VirtualFill:
        if not isinstance(fill, VirtualFill):
            raise ShadowFundDigitalTwinError(
                "fill must be a VirtualFill."
            )

        if fill.fill_id in self._fills:
            raise ShadowFundDigitalTwinError(
                "Duplicate fill ID: "
                f"{fill.fill_id}"
            )

        position = self._positions.get(
            fill.asset_id,
            ShadowPosition(
                asset_id=fill.asset_id,
                quantity=0.0,
                average_price=fill.price,
                market_price=fill.price,
            ),
        )

        if fill.side == VirtualOrderSide.BUY:
            new_quantity = (
                position.quantity + fill.quantity
            )
            new_average_price = (
                (
                    position.quantity
                    * position.average_price
                )
                + fill.gross_notional
                + fill.total_cost
            ) / new_quantity

        else:
            if fill.quantity > position.quantity:
                raise ShadowFundDigitalTwinError(
                    "Cannot sell more than the "
                    "virtual position quantity."
                )

            new_quantity = (
                position.quantity - fill.quantity
            )
            new_average_price = position.average_price

        self._cash_balance += fill.net_cash_flow

        if self._cash_balance < -1e-9:
            raise ShadowFundDigitalTwinError(
                "Virtual cash balance cannot become negative."
            )

        realized_pnl = position.realized_pnl

        if fill.side == VirtualOrderSide.SELL:
            realized_pnl += (
                fill.quantity
                * (fill.price - position.average_price)
                - fill.total_cost
            )

        self._positions[fill.asset_id] = ShadowPosition(
            asset_id=fill.asset_id,
            quantity=new_quantity,
            average_price=new_average_price,
            market_price=fill.price,
            realized_pnl=realized_pnl,
        )

        self._fills[fill.fill_id] = fill
        self._cumulative_costs += fill.total_cost

        return fill

    def mark_to_market(
        self,
        prices: Mapping[str, float],
    ) -> Tuple[ShadowPosition, ...]:
        if not isinstance(prices, Mapping):
            raise ShadowFundDigitalTwinError(
                "prices must be a mapping."
            )

        updated: Dict[str, ShadowPosition] = {}

        for asset_id, position in self._positions.items():
            price = prices.get(
                asset_id,
                position.market_price,
            )
            _require_positive(price, "price")

            updated[asset_id] = ShadowPosition(
                asset_id=position.asset_id,
                quantity=position.quantity,
                average_price=position.average_price,
                market_price=price,
                realized_pnl=position.realized_pnl,
            )

        self._positions = updated
        return tuple(updated.values())

    def snapshot(
        self,
        snapshot_id: str,
    ) -> TwinSnapshot:
        _require_text(snapshot_id, "snapshot_id")

        if snapshot_id in self._snapshots:
            raise ShadowFundDigitalTwinError(
                "Duplicate snapshot ID: "
                f"{snapshot_id}"
            )

        positions = tuple(self._positions.values())
        market_value = sum(
            position.market_value
            for position in positions
        )
        realized_pnl = sum(
            position.realized_pnl
            for position in positions
        )
        unrealized_pnl = sum(
            position.unrealized_pnl
            for position in positions
        )

        snapshot = TwinSnapshot(
            snapshot_id=snapshot_id,
            captured_at=utc_now(),
            cash_balance=self._cash_balance,
            positions=positions,
            total_market_value=market_value,
            total_equity=(
                self._cash_balance + market_value
            ),
            cumulative_costs=self._cumulative_costs,
            cumulative_realized_pnl=realized_pnl,
            cumulative_unrealized_pnl=unrealized_pnl,
        )

        self._snapshots[snapshot_id] = snapshot
        return snapshot

    def run_scenario(
        self,
        scenario: StressScenario,
    ) -> ScenarioResult:
        if not isinstance(
            scenario,
            StressScenario,
        ):
            raise ShadowFundDigitalTwinError(
                "scenario must be a StressScenario."
            )

        if scenario.scenario_id in self._scenario_results:
            raise ShadowFundDigitalTwinError(
                "Duplicate scenario ID: "
                f"{scenario.scenario_id}"
            )

        positions = tuple(self._positions.values())
        starting_value = sum(
            position.market_value
            for position in positions
        )

        stressed_value = 0.0
        affected_assets = []

        for position in positions:
            shock = scenario.price_shocks.get(
                position.asset_id,
                0.0,
            )
            stressed_price = (
                position.market_price * (1.0 + shock)
            )
            stressed_value += (
                position.quantity * stressed_price
            )

            if shock != 0.0:
                affected_assets.append(
                    position.asset_id
                )

        stressed_value *= scenario.liquidity_factor
        pnl_change = stressed_value - starting_value

        if starting_value > 0.0:
            pnl_change_ratio = (
                pnl_change / starting_value
            )
        else:
            pnl_change_ratio = 0.0

        if pnl_change_ratio <= -0.30:
            severity = ScenarioSeverity.CRITICAL
        elif pnl_change_ratio <= -0.15:
            severity = ScenarioSeverity.HIGH
        elif pnl_change_ratio < 0.0:
            severity = ScenarioSeverity.MODERATE
        else:
            severity = ScenarioSeverity.NONE

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            starting_value=starting_value,
            stressed_value=stressed_value,
            pnl_change=pnl_change,
            pnl_change_ratio=pnl_change_ratio,
            severity=severity,
            liquidity_factor=scenario.liquidity_factor,
            affected_assets=tuple(affected_assets),
        )

        self._scenario_results[
            scenario.scenario_id
        ] = result

        return result

    def get_position(
        self,
        asset_id: str,
    ) -> ShadowPosition:
        _require_text(asset_id, "asset_id")

        try:
            return self._positions[asset_id]
        except KeyError as exc:
            raise ShadowFundDigitalTwinError(
                f"Unknown asset ID: {asset_id}"
            ) from exc

    def list_positions(
        self,
    ) -> Tuple[ShadowPosition, ...]:
        return tuple(self._positions.values())

    def list_fills(
        self,
    ) -> Tuple[VirtualFill, ...]:
        return tuple(self._fills.values())

    def list_snapshots(
        self,
    ) -> Tuple[TwinSnapshot, ...]:
        return tuple(self._snapshots.values())

    def list_scenario_results(
        self,
    ) -> Tuple[ScenarioResult, ...]:
        return tuple(
            self._scenario_results.values()
        )

    def summary(self) -> Dict[str, Any]:
        latest_equity = self._cash_balance + sum(
            position.market_value
            for position in self._positions.values()
        )

        return {
            "version": self.VERSION,
            "initial_cash": self._initial_cash,
            "cash_balance": self._cash_balance,
            "latest_equity": latest_equity,
            "fill_count": self.fill_count,
            "snapshot_count": self.snapshot_count,
            "scenario_count": self.scenario_count,
            "cumulative_costs": self._cumulative_costs,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "broker_connection": (
                self.BROKER_CONNECTION
            ),
        }