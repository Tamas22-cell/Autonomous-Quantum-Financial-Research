"""
v5.0.9 - Autonomous Research Loop
Strategy Generator.

Transforms a structured research Hypothesis into a deterministic,
auditable StrategyCandidate that can be passed to the validation layer.

The generator does NOT:
- execute trades
- allocate real capital
- bypass validation
- bypass the Risk Governor
- modify source code

Its only responsibility is to convert a hypothesis into a candidate
strategy specification suitable for controlled research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from autonomous_research_loop.models import (
    Hypothesis,
    StrategyCandidate,
)


@dataclass(frozen=True)
class StrategyGeneratorConfig:
    """
    Conservative defaults for research strategy generation.

    Position sizes represent research sizing only.
    Real-money execution is outside the Autonomous Research Loop.
    """

    default_position_size: float = 0.05
    low_confidence_position_size: float = 0.02
    high_confidence_position_size: float = 0.08

    low_confidence_threshold: float = 0.40
    high_confidence_threshold: float = 0.75

    default_stop_loss: float = 0.05
    default_take_profit: float = 0.10

    maximum_position_size: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.default_position_size,
            self.low_confidence_position_size,
            self.high_confidence_position_size,
            self.low_confidence_threshold,
            self.high_confidence_threshold,
            self.default_stop_loss,
            self.default_take_profit,
            self.maximum_position_size,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                "Strategy generator configuration values must "
                "be between 0.0 and 1.0."
            )

        if (
            self.low_confidence_threshold
            >= self.high_confidence_threshold
        ):
            raise ValueError(
                "low_confidence_threshold must be lower than "
                "high_confidence_threshold."
            )

        if (
            self.low_confidence_position_size
            > self.maximum_position_size
        ):
            raise ValueError(
                "low_confidence_position_size exceeds "
                "maximum_position_size."
            )

        if (
            self.default_position_size
            > self.maximum_position_size
        ):
            raise ValueError(
                "default_position_size exceeds "
                "maximum_position_size."
            )

        if (
            self.high_confidence_position_size
            > self.maximum_position_size
        ):
            raise ValueError(
                "high_confidence_position_size exceeds "
                "maximum_position_size."
            )


class StrategyGenerator:
    """
    Convert a research hypothesis into a StrategyCandidate.

    The implementation is deterministic so identical hypotheses
    generate equivalent research rules.

    Later versions may add:
    - LLM strategy proposals
    - causal feature selection
    - parameter search
    - Alpha Lifecycle templates
    - regime-specific strategy libraries
    - portfolio-aware candidate generation

    Those systems should still pass through this controlled boundary.
    """

    def __init__(
        self,
        config: Optional[StrategyGeneratorConfig] = None,
    ) -> None:
        self.config = config or StrategyGeneratorConfig()

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    @staticmethod
    def _normalize_regime(value: Any) -> str:
        regime = str(
            value or "MIXED"
        ).strip().upper()

        aliases = {
            "RISK_ON": "RISK-ON",
            "RISK ON": "RISK-ON",
            "RISK_OFF": "RISK-OFF",
            "RISK OFF": "RISK-OFF",
            "NEUTRAL": "MIXED",
        }

        return aliases.get(
            regime,
            regime,
        )

    @staticmethod
    def _normalize_horizon(value: Any) -> str:
        horizon = str(
            value or "7D"
        ).strip().upper()

        if not horizon:
            return "7D"

        return horizon

    def calculate_position_size(
        self,
        hypothesis: Hypothesis,
    ) -> float:
        """
        Determine research position size from hypothesis confidence.

        This is only candidate sizing.
        The independent Risk Governor may reduce it further or veto it.
        """

        confidence = self._clamp(
            hypothesis.confidence
        )

        if (
            confidence
            < self.config.low_confidence_threshold
        ):
            position_size = (
                self.config.low_confidence_position_size
            )

        elif (
            confidence
            >= self.config.high_confidence_threshold
        ):
            position_size = (
                self.config.high_confidence_position_size
            )

        else:
            position_size = (
                self.config.default_position_size
            )

        return round(
            min(
                position_size,
                self.config.maximum_position_size,
            ),
            6,
        )

    def determine_direction(
        self,
        hypothesis: Hypothesis,
    ) -> str:
        """
        Infer a research direction from the hypothesis source metadata.

        Direction is intentionally conservative:
        LONG, SHORT or NEUTRAL.
        """

        metadata = hypothesis.metadata

        decision = str(
            metadata.get(
                "source_decision",
                metadata.get(
                    "decision",
                    "",
                ),
            )
        ).strip().upper()

        if decision == "BUY":
            return "LONG"

        if decision == "AVOID":
            return "SHORT"

        statement = hypothesis.statement.lower()

        positive_terms = (
            "positive predictive information",
            "positive performance",
            "risk-on",
            "bullish",
        )

        negative_terms = (
            "negative predictive information",
            "negative performance",
            "risk-off",
            "bearish",
            "downside risk",
        )

        positive_score = sum(
            term in statement
            for term in positive_terms
        )

        negative_score = sum(
            term in statement
            for term in negative_terms
        )

        if positive_score > negative_score:
            return "LONG"

        if negative_score > positive_score:
            return "SHORT"

        return "NEUTRAL"

    def build_entry_rule(
        self,
        hypothesis: Hypothesis,
        direction: str,
    ) -> str:
        """
        Create an explicit research entry rule.
        """

        asset = hypothesis.asset or "target asset"

        regime = self._normalize_regime(
            hypothesis.regime
        )

        confidence = self._clamp(
            hypothesis.confidence
        )

        if direction == "LONG":
            return (
                f"Enter LONG research exposure in {asset} only when "
                f"the hypothesis remains active, regime={regime}, "
                f"and hypothesis confidence is at least "
                f"{confidence:.3f}."
            )

        if direction == "SHORT":
            return (
                f"Enter SHORT research exposure in {asset} only when "
                f"the hypothesis remains active, regime={regime}, "
                f"and hypothesis confidence is at least "
                f"{confidence:.3f}."
            )

        return (
            f"Do not initiate directional exposure in {asset}; "
            f"observe the hypothesis until a measurable directional "
            f"condition emerges."
        )

    def build_exit_rule(
        self,
        hypothesis: Hypothesis,
        direction: str,
    ) -> str:
        """
        Create a deterministic exit condition.
        """

        horizon = self._normalize_horizon(
            hypothesis.horizon
        )

        if direction == "NEUTRAL":
            return (
                f"End observation after the {horizon} research "
                f"horizon or when the hypothesis is invalidated."
            )

        return (
            f"Exit at the end of the {horizon} research horizon, "
            f"on hypothesis invalidation, at "
            f"{self.config.default_stop_loss:.1%} adverse movement, "
            f"or at {self.config.default_take_profit:.1%} "
            f"favorable movement."
        )

    def build_stop_condition(
        self,
        hypothesis: Hypothesis,
        direction: str,
    ) -> str:
        """
        Define hard research invalidation conditions.
        """

        regime = self._normalize_regime(
            hypothesis.regime
        )

        if direction == "NEUTRAL":
            return (
                "Stop observation if the underlying research "
                "hypothesis becomes invalid or required market "
                "data becomes unavailable."
            )

        return (
            f"Invalidate the candidate if the hypothesis fails, "
            f"the expected {regime} regime condition disappears, "
            f"required data becomes stale or unavailable, or the "
            f"independent Risk Governor issues REDUCE, VETO, "
            f"or SHUTDOWN."
        )

    def build_parameters(
        self,
        hypothesis: Hypothesis,
        direction: str,
    ) -> Dict[str, Any]:
        """
        Produce explicit strategy parameters for validation.
        """

        return {
            "direction": direction,
            "research_only": True,
            "real_money_execution": False,
            "hypothesis_confidence": self._clamp(
                hypothesis.confidence
            ),
            "regime": self._normalize_regime(
                hypothesis.regime
            ),
            "horizon": self._normalize_horizon(
                hypothesis.horizon
            ),
            "stop_loss": self.config.default_stop_loss,
            "take_profit": self.config.default_take_profit,
            "maximum_position_size":
                self.config.maximum_position_size,
            "requires_validation": True,
            "requires_risk_gate": True,
            "shadow_before_trust": True,
        }

    def generate(
        self,
        hypothesis: Hypothesis,
    ) -> StrategyCandidate:
        """
        Generate one controlled StrategyCandidate.
        """

        if not isinstance(
            hypothesis,
            Hypothesis,
        ):
            raise TypeError(
                "StrategyGenerator requires a Hypothesis."
            )

        direction = self.determine_direction(
            hypothesis
        )

        position_size = self.calculate_position_size(
            hypothesis
        )

        asset = hypothesis.asset or "MARKET"

        candidate = StrategyCandidate(
            hypothesis_id=hypothesis.hypothesis_id,
            name=(
                f"{asset} "
                f"{direction} "
                f"{self._normalize_horizon(hypothesis.horizon)} "
                f"research candidate"
            ),
            entry_rule=self.build_entry_rule(
                hypothesis,
                direction,
            ),
            exit_rule=self.build_exit_rule(
                hypothesis,
                direction,
            ),
            asset=hypothesis.asset,
            horizon=self._normalize_horizon(
                hypothesis.horizon
            ),
            position_size=position_size,
            stop_condition=self.build_stop_condition(
                hypothesis,
                direction,
            ),
            parameters=self.build_parameters(
                hypothesis,
                direction,
            ),
            metadata={
                "generator":
                    "v5.0.9-strategy-generator",
                "source_hypothesis_confidence":
                    self._clamp(
                        hypothesis.confidence
                    ),
                "source_regime":
                    self._normalize_regime(
                        hypothesis.regime
                    ),
                "source_hypothesis_statement":
                    hypothesis.statement,
                "source_hypothesis_metadata":
                    dict(
                        hypothesis.metadata
                    ),
            },
        )

        return candidate

    def __call__(
        self,
        hypothesis: Hypothesis,
    ) -> StrategyCandidate:
        """
        Allows direct injection into ResearchLoopDependencies
        as strategy_generator.
        """

        return self.generate(
            hypothesis
        )