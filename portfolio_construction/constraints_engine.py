"""
Institutional Portfolio Constraints Engine for the
v5.0.7 Portfolio Construction Engine.

Capabilities:

- Position minimum / maximum constraints
- Sector exposure constraints
- Asset-class exposure constraints
- Factor exposure constraints
- Liquidity constraints
- Turnover constraints
- Gross leverage constraints
- Cash constraints
- Constraint validation
- Portfolio compliance reports
- Constraint violation diagnostics

The engine is optimizer-independent and can validate
allocations produced by any portfolio construction method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .models import (
    AssetDefinition,
    ConstraintType,
    LeverageConstraint,
    PortfolioAllocation,
    PortfolioConstraint,
    TurnoverConstraint,
)


class ConstraintsEngineError(RuntimeError):
    """
    Raised when portfolio constraint processing fails.
    """


@dataclass(frozen=True)
class ConstraintViolation:
    """
    Description of one portfolio constraint violation.
    """

    constraint_type: ConstraintType
    target: str
    actual: float
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    amount: float = 0.0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintReport:
    """
    Complete portfolio constraint validation result.
    """

    compliant: bool
    violations: List[ConstraintViolation] = field(
        default_factory=list
    )
    exposures: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    turnover: Optional[float] = None
    gross_exposure: Optional[float] = None
    net_exposure: Optional[float] = None
    cash_weight: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def violation_count(self) -> int:
        return len(self.violations)


class PortfolioConstraintsEngine:
    """
    Institutional portfolio constraint validation engine.

    The engine calculates portfolio exposures and validates
    them against configured constraints.

    Supported constraint categories:

    POSITION
        Individual asset weight.

    MIN_WEIGHT / MAX_WEIGHT
        Individual asset weight bounds.

    SECTOR
        Aggregate exposure to a sector.

    ASSET_CLASS
        Aggregate exposure to an asset class.

    FACTOR
        Weighted portfolio factor exposure.

    LIQUIDITY
        Minimum weighted-average liquidity score.

    TURNOVER
        Maximum portfolio turnover.

    LEVERAGE
        Maximum gross portfolio exposure.

    CASH
        Portfolio cash-weight bounds.
    """

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
    ) -> None:
        if tolerance < 0.0:
            raise ConstraintsEngineError(
                "tolerance cannot be negative."
            )

        self.tolerance = float(tolerance)

    # ==========================================================
    # Public API
    # ==========================================================

    def validate(
        self,
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        constraints: Optional[List[PortfolioConstraint]] = None,
        current_allocation: Optional[PortfolioAllocation] = None,
        turnover_constraint: Optional[TurnoverConstraint] = None,
        leverage_constraint: Optional[LeverageConstraint] = None,
    ) -> ConstraintReport:
        """
        Validate a portfolio allocation against all supplied
        institutional portfolio constraints.
        """

        constraints = constraints or []

        self._validate_assets(
            assets=assets
        )

        self._validate_allocation_symbols(
            allocation=allocation,
            assets=assets,
        )

        weights = self._weight_vector(
            allocation=allocation,
            assets=assets,
        )

        position_exposures = self.position_exposures(
            allocation=allocation,
            assets=assets,
        )

        sector_exposures = self.sector_exposures(
            allocation=allocation,
            assets=assets,
        )

        asset_class_exposures = self.asset_class_exposures(
            allocation=allocation,
            assets=assets,
        )

        factor_exposures = self.factor_exposures(
            allocation=allocation,
            assets=assets,
        )

        weighted_liquidity = self.weighted_liquidity_score(
            allocation=allocation,
            assets=assets,
        )

        gross_exposure = self.gross_exposure(
            allocation=allocation
        )

        net_exposure = self.net_exposure(
            allocation=allocation
        )

        turnover = self.calculate_turnover(
            allocation=allocation,
            current_allocation=current_allocation,
            assets=assets,
        )

        violations: List[ConstraintViolation] = []

        for constraint in constraints:
            if not constraint.enabled:
                continue

            violation = self._evaluate_constraint(
                constraint=constraint,
                allocation=allocation,
                assets=assets,
                position_exposures=position_exposures,
                sector_exposures=sector_exposures,
                asset_class_exposures=asset_class_exposures,
                factor_exposures=factor_exposures,
                weighted_liquidity=weighted_liquidity,
                turnover=turnover,
                gross_exposure=gross_exposure,
            )

            if violation is not None:
                violations.append(
                    violation
                )

        if turnover_constraint is not None:
            turnover_violation = (
                self._evaluate_turnover_constraint(
                    turnover=turnover,
                    constraint=turnover_constraint,
                )
            )

            if turnover_violation is not None:
                violations.append(
                    turnover_violation
                )

        if leverage_constraint is not None:
            leverage_violation = (
                self._evaluate_leverage_constraint(
                    gross_exposure=gross_exposure,
                    constraint=leverage_constraint,
                )
            )

            if leverage_violation is not None:
                violations.append(
                    leverage_violation
                )

        exposures: Dict[str, Dict[str, float]] = {
            "position": position_exposures,
            "sector": sector_exposures,
            "asset_class": asset_class_exposures,
            "factor": factor_exposures,
        }

        if weighted_liquidity is not None:
            exposures["liquidity"] = {
                "weighted_average": (
                    weighted_liquidity
                )
            }

        return ConstraintReport(
            compliant=(
                len(violations) == 0
            ),
            violations=violations,
            exposures=exposures,
            turnover=turnover,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            cash_weight=float(
                allocation.cash_weight
            ),
            metadata={
                "asset_count": len(assets),
                "constraint_count": len(
                    [
                        constraint
                        for constraint in constraints
                        if constraint.enabled
                    ]
                ),
                "weights": {
                    asset.symbol: float(
                        weight
                    )
                    for asset, weight in zip(
                        assets,
                        weights,
                    )
                },
            },
        )

    # ==========================================================
    # Exposure calculations
    # ==========================================================

    @staticmethod
    def position_exposures(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> Dict[str, float]:
        return {
            asset.symbol: float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )
            for asset in assets
        }

    @staticmethod
    def sector_exposures(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> Dict[str, float]:
        exposures: Dict[str, float] = {}

        for asset in assets:
            if asset.sector is None:
                continue

            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            exposures[asset.sector] = (
                exposures.get(
                    asset.sector,
                    0.0,
                )
                + weight
            )

        return exposures

    @staticmethod
    def asset_class_exposures(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> Dict[str, float]:
        exposures: Dict[str, float] = {}

        for asset in assets:
            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            exposures[asset.asset_class] = (
                exposures.get(
                    asset.asset_class,
                    0.0,
                )
                + weight
            )

        return exposures

    @staticmethod
    def factor_exposures(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> Dict[str, float]:
        exposures: Dict[str, float] = {}

        for asset in assets:
            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            for factor, exposure in (
                asset.factor_exposures.items()
            ):
                exposures[factor] = (
                    exposures.get(
                        factor,
                        0.0,
                    )
                    + weight
                    * float(exposure)
                )

        return exposures

    @staticmethod
    def weighted_liquidity_score(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> Optional[float]:
        """
        Calculate weighted-average liquidity.

        Assets without a liquidity score are excluded from
        the denominator rather than treated as zero.
        """

        weighted_sum = 0.0
        covered_weight = 0.0

        for asset in assets:
            if asset.liquidity_score is None:
                continue

            weight = float(
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
            )

            weighted_sum += (
                weight
                * float(
                    asset.liquidity_score
                )
            )

            covered_weight += weight

        if covered_weight <= 0.0:
            return None

        return float(
            weighted_sum
            / covered_weight
        )

    @staticmethod
    def gross_exposure(
        *,
        allocation: PortfolioAllocation,
    ) -> float:
        """
        Calculate gross invested exposure.
        """

        return float(
            sum(
                abs(float(weight))
                for weight
                in allocation.weights.values()
            )
        )

    @staticmethod
    def net_exposure(
        *,
        allocation: PortfolioAllocation,
    ) -> float:
        """
        Calculate net invested exposure.
        """

        return float(
            sum(
                float(weight)
                for weight
                in allocation.weights.values()
            )
        )

    # ==========================================================
    # Turnover
    # ==========================================================

    @staticmethod
    def calculate_turnover(
        *,
        allocation: PortfolioAllocation,
        current_allocation: Optional[
            PortfolioAllocation
        ],
        assets: List[AssetDefinition],
    ) -> Optional[float]:
        """
        One-way portfolio turnover:

            0.5 * sum(abs(new_weight - old_weight))
        """

        if current_allocation is None:
            return None

        symbols = [
            asset.symbol
            for asset in assets
        ]

        new_weights = np.asarray(
            [
                allocation.weights.get(
                    symbol,
                    0.0,
                )
                for symbol in symbols
            ],
            dtype=float,
        )

        old_weights = np.asarray(
            [
                current_allocation.weights.get(
                    symbol,
                    0.0,
                )
                for symbol in symbols
            ],
            dtype=float,
        )

        new_cash = float(
            allocation.cash_weight
        )

        old_cash = float(
            current_allocation.cash_weight
        )

        invested_turnover = float(
            np.sum(
                np.abs(
                    new_weights
                    - old_weights
                )
            )
        )

        cash_turnover = abs(
            new_cash
            - old_cash
        )

        return float(
            0.5
            * (
                invested_turnover
                + cash_turnover
            )
        )

    # ==========================================================
    # Constraint evaluation
    # ==========================================================

    def _evaluate_constraint(
        self,
        *,
        constraint: PortfolioConstraint,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
        position_exposures: Dict[str, float],
        sector_exposures: Dict[str, float],
        asset_class_exposures: Dict[str, float],
        factor_exposures: Dict[str, float],
        weighted_liquidity: Optional[float],
        turnover: Optional[float],
        gross_exposure: float,
    ) -> Optional[ConstraintViolation]:
        constraint_type = (
            constraint.constraint_type
        )

        if constraint_type in {
            ConstraintType.POSITION,
            ConstraintType.MIN_WEIGHT,
            ConstraintType.MAX_WEIGHT,
        }:
            if (
                constraint.target
                not in position_exposures
            ):
                raise ConstraintsEngineError(
                    "Unknown position constraint "
                    f"target: {constraint.target}"
                )

            actual = position_exposures[
                constraint.target
            ]

        elif (
            constraint_type
            == ConstraintType.SECTOR
        ):
            actual = sector_exposures.get(
                constraint.target,
                0.0,
            )

        elif (
            constraint_type
            == ConstraintType.ASSET_CLASS
        ):
            actual = asset_class_exposures.get(
                constraint.target,
                0.0,
            )

        elif (
            constraint_type
            == ConstraintType.FACTOR
        ):
            actual = factor_exposures.get(
                constraint.target,
                0.0,
            )

        elif (
            constraint_type
            == ConstraintType.LIQUIDITY
        ):
            if weighted_liquidity is None:
                raise ConstraintsEngineError(
                    "Liquidity constraint cannot "
                    "be evaluated because no asset "
                    "has a liquidity score."
                )

            actual = weighted_liquidity

        elif (
            constraint_type
            == ConstraintType.TURNOVER
        ):
            if turnover is None:
                raise ConstraintsEngineError(
                    "Turnover constraint requires "
                    "current_allocation."
                )

            actual = turnover

        elif (
            constraint_type
            == ConstraintType.LEVERAGE
        ):
            actual = gross_exposure

        elif (
            constraint_type
            == ConstraintType.CASH
        ):
            actual = float(
                allocation.cash_weight
            )

        else:
            raise ConstraintsEngineError(
                "Unsupported constraint type: "
                f"{constraint_type}"
            )

        minimum = constraint.minimum
        maximum = constraint.maximum

        if (
            constraint_type
            == ConstraintType.MIN_WEIGHT
            and minimum is None
        ):
            raise ConstraintsEngineError(
                "MIN_WEIGHT constraint requires "
                "minimum."
            )

        if (
            constraint_type
            == ConstraintType.MAX_WEIGHT
            and maximum is None
        ):
            raise ConstraintsEngineError(
                "MAX_WEIGHT constraint requires "
                "maximum."
            )

        return self._check_bounds(
            constraint_type=constraint_type,
            target=constraint.target,
            actual=float(actual),
            minimum=minimum,
            maximum=maximum,
        )

    def _check_bounds(
        self,
        *,
        constraint_type: ConstraintType,
        target: str,
        actual: float,
        minimum: Optional[float],
        maximum: Optional[float],
    ) -> Optional[ConstraintViolation]:
        if (
            minimum is not None
            and actual
            < float(minimum)
            - self.tolerance
        ):
            amount = (
                float(minimum)
                - actual
            )

            return ConstraintViolation(
                constraint_type=(
                    constraint_type
                ),
                target=target,
                actual=actual,
                minimum=float(minimum),
                maximum=(
                    None
                    if maximum is None
                    else float(maximum)
                ),
                amount=float(amount),
                message=(
                    f"{constraint_type.value} "
                    f"constraint '{target}' is "
                    f"below minimum: "
                    f"{actual:.6f} < "
                    f"{float(minimum):.6f}."
                ),
            )

        if (
            maximum is not None
            and actual
            > float(maximum)
            + self.tolerance
        ):
            amount = (
                actual
                - float(maximum)
            )

            return ConstraintViolation(
                constraint_type=(
                    constraint_type
                ),
                target=target,
                actual=actual,
                minimum=(
                    None
                    if minimum is None
                    else float(minimum)
                ),
                maximum=float(maximum),
                amount=float(amount),
                message=(
                    f"{constraint_type.value} "
                    f"constraint '{target}' is "
                    f"above maximum: "
                    f"{actual:.6f} > "
                    f"{float(maximum):.6f}."
                ),
            )

        return None

    # ==========================================================
    # Dedicated turnover constraint
    # ==========================================================

    def _evaluate_turnover_constraint(
        self,
        *,
        turnover: Optional[float],
        constraint: TurnoverConstraint,
    ) -> Optional[ConstraintViolation]:
        if turnover is None:
            raise ConstraintsEngineError(
                "TurnoverConstraint requires "
                "current_allocation."
            )

        if (
            turnover
            <= constraint.maximum_turnover
            + self.tolerance
        ):
            return None

        return ConstraintViolation(
            constraint_type=(
                ConstraintType.TURNOVER
            ),
            target="portfolio",
            actual=float(turnover),
            maximum=float(
                constraint.maximum_turnover
            ),
            amount=float(
                turnover
                - constraint.maximum_turnover
            ),
            message=(
                "Portfolio turnover exceeds "
                "maximum turnover: "
                f"{turnover:.6f} > "
                f"{constraint.maximum_turnover:.6f}."
            ),
        )

    # ==========================================================
    # Dedicated leverage constraint
    # ==========================================================

    def _evaluate_leverage_constraint(
        self,
        *,
        gross_exposure: float,
        constraint: LeverageConstraint,
    ) -> Optional[ConstraintViolation]:
        if (
            gross_exposure
            <= constraint.maximum_gross_exposure
            + self.tolerance
        ):
            return None

        return ConstraintViolation(
            constraint_type=(
                ConstraintType.LEVERAGE
            ),
            target="portfolio",
            actual=float(
                gross_exposure
            ),
            maximum=float(
                constraint.maximum_gross_exposure
            ),
            amount=float(
                gross_exposure
                - constraint.maximum_gross_exposure
            ),
            message=(
                "Portfolio gross exposure exceeds "
                "maximum leverage: "
                f"{gross_exposure:.6f} > "
                f"{constraint.maximum_gross_exposure:.6f}."
            ),
        )

    # ==========================================================
    # Validation helpers
    # ==========================================================

    @staticmethod
    def _validate_assets(
        *,
        assets: List[AssetDefinition],
    ) -> None:
        if not assets:
            raise ConstraintsEngineError(
                "At least one asset is required."
            )

        symbols = [
            asset.symbol
            for asset in assets
        ]

        if len(
            set(symbols)
        ) != len(symbols):
            raise ConstraintsEngineError(
                "Asset symbols must be unique."
            )

    @staticmethod
    def _validate_allocation_symbols(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> None:
        known_symbols = {
            asset.symbol
            for asset in assets
        }

        unknown_symbols = (
            set(
                allocation.weights
            )
            - known_symbols
        )

        if unknown_symbols:
            raise ConstraintsEngineError(
                "Allocation contains unknown "
                "symbols: "
                + ", ".join(
                    sorted(
                        unknown_symbols
                    )
                )
            )

        if not np.isfinite(
            float(
                allocation.cash_weight
            )
        ):
            raise ConstraintsEngineError(
                "cash_weight must be finite."
            )

        for symbol, weight in (
            allocation.weights.items()
        ):
            if not np.isfinite(
                float(weight)
            ):
                raise ConstraintsEngineError(
                    "Allocation weight for "
                    f"{symbol} must be finite."
                )

    @staticmethod
    def _weight_vector(
        *,
        allocation: PortfolioAllocation,
        assets: List[AssetDefinition],
    ) -> np.ndarray:
        return np.asarray(
            [
                allocation.weights.get(
                    asset.symbol,
                    0.0,
                )
                for asset in assets
            ],
            dtype=float,
        )


__all__ = [
    "ConstraintReport",
    "ConstraintViolation",
    "ConstraintsEngineError",
    "PortfolioConstraintsEngine",
]