"""
v5.0.9 - Autonomous Research Loop
Shadow Allocation Engine.

Creates controlled paper/shadow allocations for research candidates
that have:

1. passed strategy validation with PROMOTE
2. passed the independent Risk Governor with APPROVE or REDUCE

This module NEVER:
- sends broker orders
- allocates real capital
- bypasses validation
- bypasses the Risk Governor
- modifies production strategy code

Shadow allocations exist only for observation, measurement,
forward validation and later meta-learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from autonomous_research_loop.models import (
    RiskAssessment,
    RiskDecision,
    ShadowAllocation,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)


@dataclass(frozen=True)
class ShadowAllocatorConfig:
    """
    Configuration for shadow allocation.

    shadow_capital is simulated research capital only.
    """

    shadow_capital: float = 100_000.0

    maximum_shadow_weight: float = 0.10
    minimum_shadow_weight: float = 0.001

    require_entry_price: bool = False

    def __post_init__(self) -> None:
        if self.shadow_capital <= 0.0:
            raise ValueError(
                "shadow_capital must be greater than zero."
            )

        if not (
            0.0
            <= self.minimum_shadow_weight
            <= 1.0
        ):
            raise ValueError(
                "minimum_shadow_weight must be between "
                "0.0 and 1.0."
            )

        if not (
            0.0
            < self.maximum_shadow_weight
            <= 1.0
        ):
            raise ValueError(
                "maximum_shadow_weight must be greater than "
                "0.0 and at most 1.0."
            )

        if (
            self.minimum_shadow_weight
            > self.maximum_shadow_weight
        ):
            raise ValueError(
                "minimum_shadow_weight cannot exceed "
                "maximum_shadow_weight."
            )


class ShadowAllocator:
    """
    Create paper/shadow allocations after all upstream gates pass.

    Required chain:

    StrategyCandidate
            |
            v
    ValidationResult(PROMOTE)
            |
            v
    RiskAssessment(APPROVE/REDUCE)
            |
            v
    ShadowAllocation
    """

    def __init__(
        self,
        config: Optional[ShadowAllocatorConfig] = None,
    ) -> None:
        self.config = config or ShadowAllocatorConfig()

    @staticmethod
    def _clamp(
        value: Any,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return minimum

        return max(
            minimum,
            min(
                maximum,
                numeric,
            ),
        )

    def validate_gate_chain(
        self,
        candidate: StrategyCandidate,
        validation: ValidationResult,
        risk: RiskAssessment,
    ) -> None:
        """
        Verify that all objects belong to the same candidate
        and that every mandatory gate has passed.
        """

        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise TypeError(
                "ShadowAllocator requires a StrategyCandidate."
            )

        if not isinstance(
            validation,
            ValidationResult,
        ):
            raise TypeError(
                "validation must be a ValidationResult."
            )

        if not isinstance(
            risk,
            RiskAssessment,
        ):
            raise TypeError(
                "risk must be a RiskAssessment."
            )

        if (
            validation.strategy_id
            != candidate.strategy_id
        ):
            raise ValueError(
                "ValidationResult does not belong to "
                "the supplied StrategyCandidate."
            )

        if (
            risk.strategy_id
            != candidate.strategy_id
        ):
            raise ValueError(
                "RiskAssessment does not belong to "
                "the supplied StrategyCandidate."
            )

        if (
            validation.decision
            != ValidationDecision.PROMOTE
        ):
            raise ValueError(
                "Shadow allocation requires "
                "ValidationDecision.PROMOTE."
            )

        if risk.decision not in {
            RiskDecision.APPROVE,
            RiskDecision.REDUCE,
        }:
            raise ValueError(
                "Shadow allocation requires RiskDecision.APPROVE "
                "or RiskDecision.REDUCE."
            )

        if (
            risk.approved_position_size
            <= 0.0
        ):
            raise ValueError(
                "Approved shadow position size must "
                "be greater than zero."
            )

    def calculate_weight(
        self,
        candidate: StrategyCandidate,
        risk: RiskAssessment,
    ) -> float:
        """
        Final shadow weight is bounded by:

        - candidate requested position size
        - Risk Governor approved position size
        - global shadow allocation maximum
        """

        candidate_weight = self._clamp(
            candidate.position_size
        )

        risk_weight = self._clamp(
            risk.approved_position_size
        )

        weight = min(
            candidate_weight,
            risk_weight,
            self.config.maximum_shadow_weight,
        )

        if (
            weight
            < self.config.minimum_shadow_weight
        ):
            raise ValueError(
                "Final shadow weight is below "
                "minimum_shadow_weight."
            )

        return round(
            weight,
            6,
        )

    def calculate_notional(
        self,
        weight: float,
    ) -> float:
        """
        Convert shadow portfolio weight into simulated notional.
        """

        return round(
            self.config.shadow_capital
            * weight,
            2,
        )

    def normalize_entry_price(
        self,
        entry_price: Optional[float],
    ) -> Optional[float]:
        """
        Validate optional observation entry price.
        """

        if entry_price is None:
            if self.config.require_entry_price:
                raise ValueError(
                    "entry_price is required by configuration."
                )

            return None

        try:
            price = float(
                entry_price
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "entry_price must be numeric."
            ) from error

        if price <= 0.0:
            raise ValueError(
                "entry_price must be greater than zero."
            )

        return price

    def allocate(
        self,
        candidate: StrategyCandidate,
        validation: ValidationResult,
        risk: RiskAssessment,
        entry_price: Optional[float] = None,
    ) -> ShadowAllocation:
        """
        Create one controlled ShadowAllocation.
        """

        self.validate_gate_chain(
            candidate,
            validation,
            risk,
        )

        weight = self.calculate_weight(
            candidate,
            risk,
        )

        notional = self.calculate_notional(
            weight
        )

        normalized_entry_price = (
            self.normalize_entry_price(
                entry_price
            )
        )

        asset = (
            candidate.asset
            or "MARKET"
        )

        return ShadowAllocation(
            strategy_id=candidate.strategy_id,
            asset=asset,
            weight=weight,
            notional_value=notional,
            entry_price=normalized_entry_price,
            metadata={
                "allocator":
                    "v5.0.9-shadow-allocator",
                "mode":
                    "SHADOW",
                "research_only":
                    True,
                "paper_allocation":
                    True,
                "real_money_execution":
                    False,
                "broker_order_allowed":
                    False,
                "shadow_capital":
                    self.config.shadow_capital,
                "candidate_position_size":
                    candidate.position_size,
                "risk_approved_position_size":
                    risk.approved_position_size,
                "validation_decision":
                    validation.decision.value,
                "validation_score":
                    validation.score,
                "risk_decision":
                    risk.decision.value,
                "risk_score":
                    risk.risk_score,
                "direction":
                    candidate.parameters.get(
                        "direction",
                        "NEUTRAL",
                    ),
                "horizon":
                    candidate.horizon,
                "requires_forward_observation":
                    True,
                "eligible_for_real_execution":
                    False,
            },
        )

    def __call__(
        self,
        candidate: StrategyCandidate,
        validation: ValidationResult,
        risk: RiskAssessment,
        entry_price: Optional[float] = None,
    ) -> ShadowAllocation:
        """
        Callable interface for AutonomousResearchLoop dependency
        injection.
        """

        return self.allocate(
            candidate,
            validation,
            risk,
            entry_price,
        )