"""
v5.0.9 - Autonomous Research Loop
Shadow Outcome Observer.

Evaluates completed shadow allocations and converts their
forward performance into ResearchOutcome objects.

This module is research-only:
- no broker execution
- no real-money allocation
- no automatic live promotion
- no source-code self-modification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from autonomous_research_loop.models import (
    ResearchOutcome,
    ShadowAllocation,
    StrategyCandidate,
)


@dataclass(frozen=True)
class OutcomeObserverConfig:
    """
    Configuration for shadow outcome evaluation.
    """

    minimum_success_return: float = 0.0
    minimum_success_alpha: float = 0.0
    maximum_acceptable_drawdown: float = 0.20

    require_positive_alpha: bool = True
    require_positive_return: bool = True

    def __post_init__(self) -> None:
        if self.maximum_acceptable_drawdown < 0.0:
            raise ValueError(
                "maximum_acceptable_drawdown cannot be negative."
            )

        if self.maximum_acceptable_drawdown > 1.0:
            raise ValueError(
                "maximum_acceptable_drawdown cannot exceed 1.0."
            )


class OutcomeObserver:
    """
    Evaluate the forward result of a shadow allocation.

    The observer receives already-observed prices.
    It does not execute trades.
    """

    def __init__(
        self,
        config: Optional[OutcomeObserverConfig] = None,
    ) -> None:
        self.config = config or OutcomeObserverConfig()

    @staticmethod
    def _to_float(
        value: Any,
        field_name: str,
    ) -> float:
        """
        Convert input to float with a clear validation error.
        """

        try:
            return float(value)

        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} must be numeric."
            ) from error

    @staticmethod
    def _normalize_direction(
        candidate: StrategyCandidate,
    ) -> str:
        """
        Read and normalize strategy direction.
        """

        direction = str(
            candidate.parameters.get(
                "direction",
                "NEUTRAL",
            )
        ).upper()

        if direction not in {
            "LONG",
            "SHORT",
            "NEUTRAL",
        }:
            return "NEUTRAL"

        return direction

    def calculate_return(
        self,
        entry_price: float,
        exit_price: float,
        direction: str,
    ) -> float:
        """
        Calculate strategy return.

        LONG:
            (exit - entry) / entry

        SHORT:
            (entry - exit) / entry

        NEUTRAL:
            0.0
        """

        entry = self._to_float(
            entry_price,
            "entry_price",
        )

        exit_value = self._to_float(
            exit_price,
            "exit_price",
        )

        if entry <= 0.0:
            raise ValueError(
                "entry_price must be greater than zero."
            )

        if exit_value <= 0.0:
            raise ValueError(
                "exit_price must be greater than zero."
            )

        normalized_direction = str(
            direction
        ).upper()

        if normalized_direction == "LONG":
            result = (
                exit_value - entry
            ) / entry

        elif normalized_direction == "SHORT":
            result = (
                entry - exit_value
            ) / entry

        elif normalized_direction == "NEUTRAL":
            result = 0.0

        else:
            raise ValueError(
                "direction must be LONG, SHORT or NEUTRAL."
            )

        return round(
            result,
            8,
        )

    def calculate_benchmark_return(
        self,
        benchmark_entry_price: Optional[float],
        benchmark_exit_price: Optional[float],
    ) -> float:
        """
        Calculate benchmark buy-and-hold return.

        When no benchmark is supplied, the benchmark return
        defaults to zero.
        """

        if (
            benchmark_entry_price is None
            and benchmark_exit_price is None
        ):
            return 0.0

        if (
            benchmark_entry_price is None
            or benchmark_exit_price is None
        ):
            raise ValueError(
                "Both benchmark entry and exit prices "
                "must be supplied together."
            )

        entry = self._to_float(
            benchmark_entry_price,
            "benchmark_entry_price",
        )

        exit_value = self._to_float(
            benchmark_exit_price,
            "benchmark_exit_price",
        )

        if entry <= 0.0:
            raise ValueError(
                "benchmark_entry_price must be greater than zero."
            )

        if exit_value <= 0.0:
            raise ValueError(
                "benchmark_exit_price must be greater than zero."
            )

        result = (
            exit_value - entry
        ) / entry

        return round(
            result,
            8,
        )

    @staticmethod
    def calculate_alpha(
        strategy_return: float,
        benchmark_return: float,
    ) -> float:
        """
        Calculate simple forward excess return.
        """

        return round(
            strategy_return - benchmark_return,
            8,
        )

    def normalize_drawdown(
        self,
        max_drawdown: Any,
    ) -> float:
        """
        Normalize drawdown into a positive decimal.

        Examples:
            -0.08 -> 0.08
             0.08 -> 0.08
        """

        drawdown = abs(
            self._to_float(
                max_drawdown,
                "max_drawdown",
            )
        )

        if drawdown > 1.0:
            raise ValueError(
                "max_drawdown cannot exceed 1.0."
            )

        return round(
            drawdown,
            8,
        )

    def determine_success(
        self,
        strategy_return: float,
        alpha: float,
        max_drawdown: float,
    ) -> bool:
        """
        Determine whether the research observation
        passes configured success criteria.
        """

        if (
            max_drawdown
            > self.config.maximum_acceptable_drawdown
        ):
            return False

        if (
            self.config.require_positive_return
            and strategy_return
            <= self.config.minimum_success_return
        ):
            return False

        if (
            self.config.require_positive_alpha
            and alpha
            <= self.config.minimum_success_alpha
        ):
            return False

        return True

    def observe(
        self,
        candidate: StrategyCandidate,
        allocation: ShadowAllocation,
        exit_price: float,
        max_drawdown: float,
        benchmark_entry_price: Optional[float] = None,
        benchmark_exit_price: Optional[float] = None,
    ) -> ResearchOutcome:
        """
        Evaluate a completed shadow observation.

        ResearchOutcome model fields used:
        - strategy_id
        - pnl
        - return_pct
        - benchmark_return_pct
        - alpha
        - hit
        - metadata

        Additional shadow-specific values are stored
        inside metadata.
        """

        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise TypeError(
                "OutcomeObserver requires a StrategyCandidate."
            )

        if not isinstance(
            allocation,
            ShadowAllocation,
        ):
            raise TypeError(
                "allocation must be a ShadowAllocation."
            )

        if (
            allocation.strategy_id
            != candidate.strategy_id
        ):
            raise ValueError(
                "ShadowAllocation does not belong to "
                "the supplied StrategyCandidate."
            )

        if allocation.entry_price is None:
            raise ValueError(
                "ShadowAllocation must contain an entry_price "
                "before forward outcome observation."
            )

        direction = self._normalize_direction(
            candidate
        )

        strategy_return = self.calculate_return(
            entry_price=allocation.entry_price,
            exit_price=exit_price,
            direction=direction,
        )

        benchmark_return = (
            self.calculate_benchmark_return(
                benchmark_entry_price=
                benchmark_entry_price,
                benchmark_exit_price=
                benchmark_exit_price,
            )
        )

        alpha = self.calculate_alpha(
            strategy_return=strategy_return,
            benchmark_return=benchmark_return,
        )

        normalized_drawdown = (
            self.normalize_drawdown(
                max_drawdown
            )
        )

        success = self.determine_success(
            strategy_return=strategy_return,
            alpha=alpha,
            max_drawdown=normalized_drawdown,
        )

        pnl = round(
            allocation.notional_value
            * strategy_return,
            2,
        )

        return ResearchOutcome(
            strategy_id=candidate.strategy_id,
            pnl=pnl,
            return_pct=strategy_return,
            benchmark_return_pct=benchmark_return,
            alpha=alpha,
            hit=success,
            metadata={
                "observer":
                    "v5.0.9-outcome-observer",

                "mode":
                    "SHADOW_OBSERVATION",

                "shadow_allocation_id":
                    allocation.allocation_id,

                "asset":
                    candidate.asset,

                "direction":
                    direction,

                "horizon":
                    candidate.horizon,

                "entry_price":
                    allocation.entry_price,

                "exit_price":
                    float(exit_price),

                "shadow_weight":
                    allocation.weight,

                "shadow_notional":
                    allocation.notional_value,

                "max_drawdown":
                    normalized_drawdown,

                "success":
                    success,

                "benchmark_entry_price":
                    benchmark_entry_price,

                "benchmark_exit_price":
                    benchmark_exit_price,

                "minimum_success_return":
                    self.config.minimum_success_return,

                "minimum_success_alpha":
                    self.config.minimum_success_alpha,

                "maximum_acceptable_drawdown":
                    self.config.maximum_acceptable_drawdown,

                "research_only":
                    True,

                "paper_observation":
                    True,

                "real_money_execution":
                    False,

                "eligible_for_automatic_live_promotion":
                    False,
            },
        )

    def __call__(
        self,
        candidate: StrategyCandidate,
        allocation: ShadowAllocation,
        exit_price: float,
        max_drawdown: float,
        benchmark_entry_price: Optional[float] = None,
        benchmark_exit_price: Optional[float] = None,
    ) -> ResearchOutcome:
        """
        Callable dependency-injection interface.
        """

        return self.observe(
            candidate=candidate,
            allocation=allocation,
            exit_price=exit_price,
            max_drawdown=max_drawdown,
            benchmark_entry_price=benchmark_entry_price,
            benchmark_exit_price=benchmark_exit_price,
        )