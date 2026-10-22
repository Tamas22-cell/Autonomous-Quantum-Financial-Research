"""
v5.0.9 - Autonomous Research Loop
Hypothesis Engine.

Transforms a prioritized ResearchTask into a structured,
testable market hypothesis.

Inputs may originate from:
- nine-agent Orchestrator
- regime change
- uncertainty
- performance gap
- alpha decay
- risk event
- manual research request

The engine does not trade, modify source code, or promote strategies.
It only creates auditable research hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from autonomous_research_loop.models import (
    Hypothesis,
    ResearchTask,
    ResearchTrigger,
)


@dataclass(frozen=True)
class HypothesisEngineConfig:
    """
    Configuration for hypothesis generation.
    """

    minimum_confidence: float = 0.10
    maximum_confidence: float = 0.95

    uncertainty_penalty: float = 0.35
    conflict_penalty: float = 0.20

    regime_change_bonus: float = 0.05
    agent_signal_bonus: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.minimum_confidence,
            self.maximum_confidence,
            self.uncertainty_penalty,
            self.conflict_penalty,
            self.regime_change_bonus,
            self.agent_signal_bonus,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                "Hypothesis configuration values must "
                "be between 0.0 and 1.0."
            )

        if self.minimum_confidence > self.maximum_confidence:
            raise ValueError(
                "minimum_confidence cannot exceed "
                "maximum_confidence."
            )


class HypothesisEngine:
    """
    Generate deterministic research hypotheses.

    The first v5.0.9 implementation is intentionally deterministic.
    Later an LLM/research layer can propose richer hypotheses while
    this engine remains the validation and normalization boundary.
    """

    def __init__(
        self,
        config: Optional[HypothesisEngineConfig] = None,
    ) -> None:
        self.config = config or HypothesisEngineConfig()

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _safe_list(value: Any) -> List[str]:
        if value is None:
            return []

        if isinstance(value, (list, tuple)):
            return [
                str(item)
                for item in value
                if item is not None
            ]

        return [str(value)]

    @staticmethod
    def _normalize_decision(value: Any) -> str:
        decision = str(value or "HOLD").strip().upper()

        if decision not in {
            "BUY",
            "HOLD",
            "AVOID",
        }:
            return "HOLD"

        return decision

    @staticmethod
    def _normalize_regime(value: Any) -> str:
        regime = str(value or "MIXED").strip().upper()

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

    def calculate_confidence(
        self,
        task: ResearchTask,
    ) -> float:
        """
        Estimate initial hypothesis confidence.

        Confidence uses existing orchestrator evidence where available.

        It is reduced by:
        - research uncertainty
        - signal conflict

        It receives only small trigger-specific bonuses.
        """

        metadata = task.metadata

        average_confidence = self._clamp(
            metadata.get(
                "average_confidence",
                0.50,
            )
        )

        consensus_strength = self._clamp(
            metadata.get(
                "consensus_strength",
                0.50,
            )
        )

        signal_conflict = self._clamp(
            metadata.get(
                "signal_conflict",
                0.0,
            )
        )

        uncertainty = self._clamp(
            task.uncertainty_score
        )

        base_confidence = (
            average_confidence * 0.60
            + consensus_strength * 0.40
        )

        confidence = (
            base_confidence
            - uncertainty
            * self.config.uncertainty_penalty
            - signal_conflict
            * self.config.conflict_penalty
        )

        if task.trigger == ResearchTrigger.REGIME_CHANGE:
            confidence += self.config.regime_change_bonus

        elif task.trigger == ResearchTrigger.AGENT_SIGNAL:
            confidence += self.config.agent_signal_bonus

        confidence = max(
            self.config.minimum_confidence,
            min(
                self.config.maximum_confidence,
                confidence,
            ),
        )

        return round(confidence, 6)

    def build_statement(
        self,
        task: ResearchTask,
    ) -> str:
        """
        Build a testable hypothesis statement.
        """

        metadata = task.metadata

        asset = task.asset or "market"
        horizon = task.horizon or "research horizon"

        regime = self._normalize_regime(
            task.regime
        )

        decision = self._normalize_decision(
            metadata.get("decision")
        )

        market_score = self._clamp(
            metadata.get(
                "market_score",
                0.50,
            )
        )

        if task.trigger == ResearchTrigger.REGIME_CHANGE:
            return (
                f"{asset} exhibits a measurable change in "
                f"risk-adjusted performance during the {regime} "
                f"regime over the {horizon} horizon relative to "
                f"the preceding regime."
            )

        if task.trigger == ResearchTrigger.RISK_EVENT:
            return (
                f"{asset} experiences materially different downside "
                f"risk and drawdown behavior during the detected "
                f"risk event over the {horizon} horizon."
            )

        if task.trigger == ResearchTrigger.UNCERTAINTY:
            return (
                f"Elevated disagreement and uncertainty in the "
                f"multi-agent system precede a measurable change "
                f"in {asset} volatility or directional performance "
                f"over the {horizon} horizon."
            )

        if task.trigger == ResearchTrigger.PERFORMANCE_GAP:
            return (
                f"Current disagreement between specialist signals "
                f"contains predictive information for {asset} "
                f"performance over the {horizon} horizon."
            )

        if task.trigger == ResearchTrigger.ALPHA_DECAY:
            return (
                f"The previously observed predictive relationship "
                f"for {asset} has weakened materially over the "
                f"{horizon} horizon and requires re-estimation."
            )

        direction = {
            "BUY": "positive",
            "AVOID": "negative",
            "HOLD": "non-directional or weak",
        }[decision]

        return (
            f"The current multi-agent signal with market score "
            f"{market_score:.3f} contains measurable {direction} "
            f"predictive information for {asset} over the "
            f"{horizon} horizon."
        )

    def build_evidence(
        self,
        task: ResearchTask,
    ) -> List[str]:
        """
        Extract evidence from the research task metadata.
        """

        metadata = task.metadata

        evidence: List[str] = []

        positive_signals = self._safe_list(
            metadata.get(
                "top_positive_signals"
            )
        )

        top_risks = self._safe_list(
            metadata.get(
                "top_risks"
            )
        )

        for signal in positive_signals:
            evidence.append(
                f"Positive signal: {signal}"
            )

        for risk in top_risks:
            evidence.append(
                f"Risk signal: {risk}"
            )

        market_interpretation = metadata.get(
            "market_interpretation"
        )

        if market_interpretation:
            evidence.append(
                "Market interpretation: "
                f"{market_interpretation}"
            )

        evidence.append(
            "Research trigger: "
            f"{task.trigger.value}"
        )

        evidence.append(
            "Research uncertainty score: "
            f"{self._clamp(task.uncertainty_score):.3f}"
        )

        evidence.append(
            "Performance-gap score: "
            f"{self._clamp(task.performance_gap_score):.3f}"
        )

        return evidence

    def build_assumptions(
        self,
        task: ResearchTask,
    ) -> List[str]:
        """
        Explicit assumptions make the hypothesis falsifiable
        and easier to audit.
        """

        assumptions = [
            (
                "Input agent outputs are available and sufficiently "
                "recent for the selected research horizon."
            ),
            (
                "The market snapshot does not contain look-ahead "
                "information."
            ),
            (
                "Validation will use data not used to formulate "
                "the hypothesis whenever possible."
            ),
            (
                "Transaction costs, slippage and liquidity effects "
                "must be considered before promotion."
            ),
        ]

        if task.regime:
            assumptions.append(
                "Market-regime classification remains consistent "
                "with the platform regime methodology."
            )

        return assumptions

    def generate(
        self,
        task: ResearchTask,
    ) -> Hypothesis:
        """
        Generate one structured hypothesis from a ResearchTask.
        """

        if not isinstance(task, ResearchTask):
            raise TypeError(
                "HypothesisEngine requires a ResearchTask."
            )

        statement = self.build_statement(task)

        confidence = self.calculate_confidence(task)

        hypothesis = Hypothesis(
            task_id=task.task_id,
            statement=statement,
            asset=task.asset,
            regime=task.regime,
            horizon=task.horizon,
            confidence=confidence,
            evidence=self.build_evidence(task),
            assumptions=self.build_assumptions(task),
            metadata={
                "generator": "v5.0.9-hypothesis-engine",
                "research_trigger": task.trigger.value,
                "source_priority": task.priority.value,
                "source_uncertainty":
                    self._clamp(
                        task.uncertainty_score
                    ),
                "source_performance_gap":
                    self._clamp(
                        task.performance_gap_score
                    ),
            },
        )

        return hypothesis

    def __call__(
        self,
        task: ResearchTask,
    ) -> Hypothesis:
        """
        Allows HypothesisEngine to be injected directly into
        ResearchLoopDependencies as hypothesis_generator.
        """

        return self.generate(task)