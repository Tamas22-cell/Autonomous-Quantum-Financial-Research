from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Dict, List, Optional

from research_control_center.models import (
    AllocationMode,
    ShadowFundSnapshot,
    utc_now,
)


class ShadowFundDashboardConfig:
    """
    Configuration for the v5.0.10 Shadow Fund Dashboard.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        max_gross_exposure: float = 1.50,
        max_net_exposure: float = 1.00,
        max_single_position_weight: float = 0.25,
        max_drawdown_limit: float = 0.20,
    ) -> None:
        numeric_values = {
            "initial_capital": initial_capital,
            "max_gross_exposure": max_gross_exposure,
            "max_net_exposure": max_net_exposure,
            "max_single_position_weight": max_single_position_weight,
            "max_drawdown_limit": max_drawdown_limit,
        }

        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be positive"
                )

        if max_single_position_weight > 1.0:
            raise ValueError(
                "max_single_position_weight "
                "cannot exceed 1.0"
            )

        if max_drawdown_limit > 1.0:
            raise ValueError(
                "max_drawdown_limit cannot exceed 1.0"
            )

        self.initial_capital = float(
            initial_capital
        )
        self.max_gross_exposure = float(
            max_gross_exposure
        )
        self.max_net_exposure = float(
            max_net_exposure
        )
        self.max_single_position_weight = float(
            max_single_position_weight
        )
        self.max_drawdown_limit = float(
            max_drawdown_limit
        )


class ShadowFundDashboard:
    """
    Research Control Center v5.0.10
    Shadow Fund / Portfolio / Risk Dashboard.

    Responsibilities:
    - maintain shadow positions
    - track allocated capital and cash
    - track P&L and portfolio return
    - calculate gross and net exposure
    - track asset / sector / factor exposure
    - track attribution
    - monitor drawdown
    - detect risk-limit breaches
    - generate immutable snapshots
    - preserve audit history

    Research-only / paper portfolio component.
    No broker or real-money execution is permitted.
    """

    def __init__(
        self,
        config: Optional[
            ShadowFundDashboardConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or ShadowFundDashboardConfig()
        )

        self._positions: Dict[str, float] = {}
        self._asset_exposure: Dict[str, float] = {}
        self._sector_exposure: Dict[str, float] = {}
        self._factor_exposure: Dict[str, float] = {}
        self._attribution: Dict[str, float] = {}

        self._pnl = 0.0
        self._max_drawdown = 0.0

        self._audit_log: List[dict] = []

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_name(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{name} cannot be empty"
            )

        return value

    @staticmethod
    def _validate_number(
        value: float,
        name: str,
    ) -> float:
        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"{name} must be numeric"
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    @staticmethod
    def _validate_mapping(
        values: Dict[str, float],
        name: str,
    ) -> Dict[str, float]:
        if not isinstance(values, dict):
            raise TypeError(
                f"{name} must be a dictionary"
            )

        result: Dict[str, float] = {}

        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"{name} keys must be strings"
                )

            clean_key = key.strip()

            if not clean_key:
                raise ValueError(
                    f"{name} keys cannot be empty"
                )

            result[clean_key] = (
                ShadowFundDashboard
                ._validate_number(
                    value,
                    f"{name}[{clean_key}]",
                )
            )

        return result

    def _audit(
        self,
        event: str,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "details": deepcopy(
                    details or {}
                ),
                "research_only": True,
                "real_money_execution": False,
            }
        )

    # =========================================================
    # POSITIONS
    # =========================================================

    def set_position(
        self,
        asset: str,
        notional_value: float,
    ) -> None:
        asset = self._validate_name(
            asset,
            "asset",
        )

        notional_value = self._validate_number(
            notional_value,
            "notional_value",
        )

        if notional_value == 0.0:
            self._positions.pop(
                asset,
                None,
            )
        else:
            self._positions[
                asset
            ] = notional_value

        self._audit(
            "SHADOW_POSITION_UPDATED",
            {
                "asset": asset,
                "notional_value": notional_value,
            },
        )

    def remove_position(
        self,
        asset: str,
    ) -> None:
        asset = self._validate_name(
            asset,
            "asset",
        )

        previous = self._positions.pop(
            asset,
            None,
        )

        self._audit(
            "SHADOW_POSITION_REMOVED",
            {
                "asset": asset,
                "previous_notional": previous,
            },
        )

    def positions(self) -> Dict[str, float]:
        return deepcopy(
            self._positions
        )

    # =========================================================
    # CAPITAL / EXPOSURE
    # =========================================================

    def allocated_capital(self) -> float:
        return sum(
            abs(value)
            for value
            in self._positions.values()
        )

    def cash(self) -> float:
        return (
            self.config.initial_capital
            - self.allocated_capital()
        )

    def gross_exposure(self) -> float:
        if self.config.initial_capital == 0.0:
            return 0.0

        return (
            self.allocated_capital()
            / self.config.initial_capital
        )

    def net_notional(self) -> float:
        return sum(
            self._positions.values()
        )

    def net_exposure(self) -> float:
        if self.config.initial_capital == 0.0:
            return 0.0

        return (
            self.net_notional()
            / self.config.initial_capital
        )

    def position_weights(
        self,
    ) -> Dict[str, float]:
        capital = self.config.initial_capital

        return {
            asset: notional / capital
            for asset, notional
            in self._positions.items()
        }

    # =========================================================
    # P&L
    # =========================================================

    def set_pnl(
        self,
        pnl: float,
    ) -> None:
        pnl = self._validate_number(
            pnl,
            "pnl",
        )

        self._pnl = pnl

        self._audit(
            "SHADOW_PNL_UPDATED",
            {
                "pnl": pnl,
                "return_pct": self.return_pct(),
            },
        )

    def add_pnl(
        self,
        pnl_delta: float,
    ) -> None:
        pnl_delta = self._validate_number(
            pnl_delta,
            "pnl_delta",
        )

        self._pnl += pnl_delta

        self._audit(
            "SHADOW_PNL_ADDED",
            {
                "pnl_delta": pnl_delta,
                "total_pnl": self._pnl,
                "return_pct": self.return_pct(),
            },
        )

    def pnl(self) -> float:
        return self._pnl

    def return_pct(self) -> float:
        return (
            self._pnl
            / self.config.initial_capital
        )

    # =========================================================
    # DRAWDOWN
    # =========================================================

    def set_max_drawdown(
        self,
        drawdown: float,
    ) -> None:
        drawdown = self._validate_number(
            drawdown,
            "drawdown",
        )

        if drawdown < 0.0:
            drawdown = abs(drawdown)

        if drawdown > 1.0:
            raise ValueError(
                "drawdown cannot exceed 1.0"
            )

        self._max_drawdown = drawdown

        self._audit(
            "SHADOW_DRAWDOWN_UPDATED",
            {
                "max_drawdown": drawdown,
            },
        )

    def max_drawdown(self) -> float:
        return self._max_drawdown

    # =========================================================
    # EXPOSURE BREAKDOWNS
    # =========================================================

    def set_asset_exposure(
        self,
        exposures: Dict[str, float],
    ) -> None:
        self._asset_exposure = (
            self._validate_mapping(
                exposures,
                "asset_exposure",
            )
        )

        self._audit(
            "ASSET_EXPOSURE_UPDATED",
            {
                "asset_exposure": deepcopy(
                    self._asset_exposure
                ),
            },
        )

    def set_sector_exposure(
        self,
        exposures: Dict[str, float],
    ) -> None:
        self._sector_exposure = (
            self._validate_mapping(
                exposures,
                "sector_exposure",
            )
        )

        self._audit(
            "SECTOR_EXPOSURE_UPDATED",
            {
                "sector_exposure": deepcopy(
                    self._sector_exposure
                ),
            },
        )

    def set_factor_exposure(
        self,
        exposures: Dict[str, float],
    ) -> None:
        self._factor_exposure = (
            self._validate_mapping(
                exposures,
                "factor_exposure",
            )
        )

        self._audit(
            "FACTOR_EXPOSURE_UPDATED",
            {
                "factor_exposure": deepcopy(
                    self._factor_exposure
                ),
            },
        )

    # =========================================================
    # ATTRIBUTION
    # =========================================================

    def set_attribution(
        self,
        attribution: Dict[str, float],
    ) -> None:
        self._attribution = (
            self._validate_mapping(
                attribution,
                "attribution",
            )
        )

        self._audit(
            "ATTRIBUTION_UPDATED",
            {
                "attribution": deepcopy(
                    self._attribution
                ),
            },
        )

    def add_attribution(
        self,
        source: str,
        pnl_contribution: float,
    ) -> None:
        source = self._validate_name(
            source,
            "source",
        )

        pnl_contribution = (
            self._validate_number(
                pnl_contribution,
                "pnl_contribution",
            )
        )

        self._attribution[source] = (
            self._attribution.get(
                source,
                0.0,
            )
            + pnl_contribution
        )

        self._audit(
            "ATTRIBUTION_ADDED",
            {
                "source": source,
                "pnl_contribution": (
                    pnl_contribution
                ),
                "total_source_attribution": (
                    self._attribution[source]
                ),
            },
        )

    # =========================================================
    # RISK
    # =========================================================

    def risk_report(self) -> dict:
        gross = self.gross_exposure()
        net = self.net_exposure()

        position_weights = (
            self.position_weights()
        )

        gross_breach = (
            gross
            > self.config.max_gross_exposure
        )

        net_breach = (
            abs(net)
            > self.config.max_net_exposure
        )

        drawdown_breach = (
            self._max_drawdown
            > self.config.max_drawdown_limit
        )

        position_breaches = {
            asset: weight
            for asset, weight
            in position_weights.items()
            if abs(weight)
            > self.config.max_single_position_weight
        }

        breaches: List[str] = []

        if gross_breach:
            breaches.append(
                "MAX_GROSS_EXPOSURE"
            )

        if net_breach:
            breaches.append(
                "MAX_NET_EXPOSURE"
            )

        if drawdown_breach:
            breaches.append(
                "MAX_DRAWDOWN"
            )

        if position_breaches:
            breaches.append(
                "MAX_SINGLE_POSITION"
            )

        return {
            "gross_exposure": gross,
            "net_exposure": net,
            "max_drawdown": (
                self._max_drawdown
            ),
            "position_weights": (
                position_weights
            ),
            "gross_exposure_breach": (
                gross_breach
            ),
            "net_exposure_breach": (
                net_breach
            ),
            "drawdown_breach": (
                drawdown_breach
            ),
            "position_breaches": (
                position_breaches
            ),
            "breaches": breaches,
            "breach_count": len(breaches),
            "within_limits": (
                len(breaches) == 0
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(
        self,
    ) -> ShadowFundSnapshot:
        return ShadowFundSnapshot(
            total_capital=(
                self.config.initial_capital
            ),
            allocated_capital=(
                self.allocated_capital()
            ),
            cash=self.cash(),
            pnl=self._pnl,
            return_pct=self.return_pct(),
            gross_exposure=(
                self.gross_exposure()
            ),
            net_exposure=(
                self.net_exposure()
            ),
            max_drawdown=(
                self._max_drawdown
            ),
            positions=deepcopy(
                self._positions
            ),
            asset_exposure=deepcopy(
                self._asset_exposure
            ),
            sector_exposure=deepcopy(
                self._sector_exposure
            ),
            factor_exposure=deepcopy(
                self._factor_exposure
            ),
            attribution=deepcopy(
                self._attribution
            ),
            mode=AllocationMode.SHADOW,
            metadata={
                "dashboard": (
                    "v5.0.10-shadow-fund"
                ),
                "research_only": True,
                "paper_allocation": True,
                "real_money_execution": False,
                "broker_order_allowed": False,
            },
        )

    # =========================================================
    # DASHBOARD VIEW
    # =========================================================

    def dashboard(self) -> dict:
        snapshot = self.snapshot()
        risk = self.risk_report()

        return {
            "capital": {
                "initial_capital": (
                    snapshot.total_capital
                ),
                "allocated_capital": (
                    snapshot.allocated_capital
                ),
                "cash": snapshot.cash,
            },
            "performance": {
                "pnl": snapshot.pnl,
                "return_pct": (
                    snapshot.return_pct
                ),
                "max_drawdown": (
                    snapshot.max_drawdown
                ),
            },
            "exposure": {
                "gross": (
                    snapshot.gross_exposure
                ),
                "net": (
                    snapshot.net_exposure
                ),
                "positions": deepcopy(
                    snapshot.positions
                ),
                "asset": deepcopy(
                    snapshot.asset_exposure
                ),
                "sector": deepcopy(
                    snapshot.sector_exposure
                ),
                "factor": deepcopy(
                    snapshot.factor_exposure
                ),
            },
            "attribution": deepcopy(
                snapshot.attribution
            ),
            "risk": risk,
            "generated_at": utc_now(),
            "mode": AllocationMode.SHADOW.value,
            "research_only": True,
            "paper_allocation": True,
            "real_money_execution": False,
            "broker_order_allowed": False,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        risk = self.risk_report()

        return {
            "positions": len(
                self._positions
            ),
            "allocated_capital": (
                self.allocated_capital()
            ),
            "cash": self.cash(),
            "pnl": self._pnl,
            "return_pct": (
                self.return_pct()
            ),
            "gross_exposure": (
                self.gross_exposure()
            ),
            "net_exposure": (
                self.net_exposure()
            ),
            "max_drawdown": (
                self._max_drawdown
            ),
            "risk_breaches": (
                risk["breach_count"]
            ),
            "within_risk_limits": (
                risk["within_limits"]
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # AUDIT
    # =========================================================

    def audit_log(self) -> List[dict]:
        return deepcopy(
            self._audit_log
        )