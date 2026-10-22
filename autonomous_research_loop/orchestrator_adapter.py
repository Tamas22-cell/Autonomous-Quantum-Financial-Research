"""
v5.0.9 - Autonomous Research Loop
Orchestrator Adapter.

Connects the existing nine-agent Orchestrator to the autonomous
research system without modifying the production orchestrator.

Existing flow:

9 Specialist Agents
        |
        v
   Orchestrator
        |
        v
Structured Market Snapshot
        |
        v
OrchestratorResearchAdapter
        |
        v
ResearchTask
        |
        v
Autonomous Research Loop

The adapter translates existing market intelligence into research
triggers. It does not execute trades and does not modify source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from autonomous_research_loop.models import (
    ResearchPriority,
    ResearchTask,
    ResearchTrigger,
)


@dataclass(frozen=True)
class OrchestratorAdapterConfig:
    """
    Threshold configuration for converting orchestrator output
    into autonomous research tasks.
    """

    uncertainty_threshold: float = 0.40
    conflict_threshold: float = 0.35

    risk_score_threshold: float = 0.40
    performance_gap_threshold: float = 0.30

    regime_change_priority_boost: float = 0.20
    risk_event_priority_boost: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.uncertainty_threshold,
            self.conflict_threshold,
            self.risk_score_threshold,
            self.performance_gap_threshold,
            self.regime_change_priority_boost,
            self.risk_event_priority_boost,
        )

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError(
                "All adapter configuration values must "
                "be between 0.0 and 1.0."
            )


class OrchestratorResearchAdapter:
    """
    Translate existing Orchestrator output into research tasks.

    Expected Orchestrator fields include:

    - agents
    - market_score
    - decision
    - market_regime
    - market_interpretation
    - bullish_agents
    - neutral_agents
    - bearish_agents
    - bullish_pct
    - neutral_pct
    - bearish_pct
    - consensus_strength
    - average_confidence
    - signal_conflict
    - risk_override
    - strongest_bullish_agent
    - strongest_bearish_agent
    - top_positive_signals
    - top_risks

    The adapter is deliberately defensive. Missing fields receive
    conservative defaults instead of crashing the research system.
    """

    VALID_REGIMES = {
        "RISK-ON",
        "RISK-OFF",
        "MIXED",
    }

    def __init__(
        self,
        config: Optional[OrchestratorAdapterConfig] = None,
    ) -> None:
        self.config = config or OrchestratorAdapterConfig()

    @staticmethod
    def _clamp(value: Any) -> float:
        """
        Convert a value to float and clamp it into [0, 1].
        """

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _percentage_to_ratio(value: Any) -> float:
        """
        Convert percentage-like values into [0, 1].

        Examples:
            55 -> 0.55
            55.0 -> 0.55
            0.55 -> 0.55
        """

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if numeric > 1.0:
            numeric /= 100.0

        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        """
        Normalize an arbitrary value into a list.
        """

        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        return [value]

    @staticmethod
    def _normalize_regime(value: Any) -> str:
        """
        Normalize market regime text.
        """

        regime = str(value or "MIXED").strip().upper()

        aliases = {
            "RISK_ON": "RISK-ON",
            "RISK ON": "RISK-ON",
            "RISK_OFF": "RISK-OFF",
            "RISK OFF": "RISK-OFF",
            "NEUTRAL": "MIXED",
        }

        return aliases.get(regime, regime)

    def calculate_uncertainty(
        self,
        snapshot: Dict[str, Any],
    ) -> float:
        """
        Estimate research uncertainty from existing orchestrator data.

        Higher uncertainty is associated with:
        - lower confidence
        - weaker consensus
        - stronger signal conflict
        """

        confidence = self._clamp(
            snapshot.get("average_confidence", 0.50)
        )

        consensus = self._clamp(
            snapshot.get("consensus_strength", 0.50)
        )

        conflict = self._clamp(
            snapshot.get("signal_conflict", 0.0)
        )

        confidence_uncertainty = 1.0 - confidence
        consensus_uncertainty = 1.0 - consensus

        uncertainty = (
            confidence_uncertainty * 0.40
            + consensus_uncertainty * 0.35
            + conflict * 0.25
        )

        return round(
            self._clamp(uncertainty),
            6,
        )

    def calculate_performance_gap(
        self,
        snapshot: Dict[str, Any],
    ) -> float:
        """
        Estimate disagreement/performance-gap pressure.

        v5.0.9 later connects this field to historical validation
        and Alpha Lifecycle performance. For now the adapter uses
        observable disagreement inside the current market snapshot.
        """

        bullish = self._percentage_to_ratio(
            snapshot.get("bullish_pct", 0.0)
        )

        bearish = self._percentage_to_ratio(
            snapshot.get("bearish_pct", 0.0)
        )

        neutral = self._percentage_to_ratio(
            snapshot.get("neutral_pct", 0.0)
        )

        conflict = self._clamp(
            snapshot.get("signal_conflict", 0.0)
        )

        directional_balance = 1.0 - abs(
            bullish - bearish
        )

        disagreement = (
            directional_balance * 0.45
            + neutral * 0.20
            + conflict * 0.35
        )

        return round(
            self._clamp(disagreement),
            6,
        )

    def detect_trigger(
        self,
        snapshot: Dict[str, Any],
        previous_regime: Optional[str] = None,
    ) -> ResearchTrigger:
        """
        Select the strongest research trigger.

        Trigger precedence:

        1. Risk event
        2. Regime change
        3. High uncertainty
        4. Performance gap
        5. Agent signal
        """

        risk_override = bool(
            snapshot.get("risk_override", False)
        )

        top_risks = self._safe_list(
            snapshot.get("top_risks")
        )

        bearish_pct = self._percentage_to_ratio(
            snapshot.get("bearish_pct", 0.0)
        )

        current_regime = self._normalize_regime(
            snapshot.get("market_regime")
        )

        normalized_previous_regime = (
            self._normalize_regime(previous_regime)
            if previous_regime is not None
            else None
        )

        uncertainty = self.calculate_uncertainty(
            snapshot
        )

        performance_gap = self.calculate_performance_gap(
            snapshot
        )

        if risk_override:
            return ResearchTrigger.RISK_EVENT

        if (
            current_regime == "RISK-OFF"
            and top_risks
            and bearish_pct >= self.config.risk_score_threshold
        ):
            return ResearchTrigger.RISK_EVENT

        if (
            normalized_previous_regime is not None
            and current_regime
            != normalized_previous_regime
        ):
            return ResearchTrigger.REGIME_CHANGE

        if (
            uncertainty
            >= self.config.uncertainty_threshold
        ):
            return ResearchTrigger.UNCERTAINTY

        if (
            performance_gap
            >= self.config.performance_gap_threshold
        ):
            return ResearchTrigger.PERFORMANCE_GAP

        return ResearchTrigger.AGENT_SIGNAL

    def build_description(
        self,
        snapshot: Dict[str, Any],
        trigger: ResearchTrigger,
    ) -> str:
        """
        Build an auditable research-task description.
        """

        decision = str(
            snapshot.get("decision", "UNKNOWN")
        )

        regime = self._normalize_regime(
            snapshot.get("market_regime")
        )

        market_score = self._clamp(
            snapshot.get("market_score", 0.50)
        )

        confidence = self._clamp(
            snapshot.get("average_confidence", 0.50)
        )

        consensus = self._clamp(
            snapshot.get("consensus_strength", 0.50)
        )

        return (
            f"Autonomous research triggered by "
            f"{trigger.value}. "
            f"Current decision={decision}, "
            f"regime={regime}, "
            f"market_score={market_score:.3f}, "
            f"average_confidence={confidence:.3f}, "
            f"consensus_strength={consensus:.3f}."
        )

    def create_research_task(
        self,
        snapshot: Dict[str, Any],
        *,
        asset: Optional[str] = None,
        horizon: Optional[str] = None,
        previous_regime: Optional[str] = None,
    ) -> ResearchTask:
        """
        Convert one Orchestrator market snapshot into a ResearchTask.
        """

        if not isinstance(snapshot, dict):
            raise TypeError(
                "Orchestrator snapshot must be a dictionary."
            )

        trigger = self.detect_trigger(
            snapshot,
            previous_regime=previous_regime,
        )

        uncertainty = self.calculate_uncertainty(
            snapshot
        )

        performance_gap = self.calculate_performance_gap(
            snapshot
        )

        regime = self._normalize_regime(
            snapshot.get("market_regime")
        )

        decision = str(
            snapshot.get("decision", "UNKNOWN")
        ).upper()

        task = ResearchTask(
            title=(
                f"{asset or 'MARKET'} "
                f"{trigger.value} research"
            ),
            description=self.build_description(
                snapshot,
                trigger,
            ),
            trigger=trigger,
            asset=asset,
            regime=regime,
            horizon=horizon,
            uncertainty_score=uncertainty,
            performance_gap_score=performance_gap,
            priority=ResearchPriority.MEDIUM,
            metadata={
                "source": "orchestrator",
                "decision": decision,
                "market_score": self._clamp(
                    snapshot.get(
                        "market_score",
                        0.50,
                    )
                ),
                "average_confidence": self._clamp(
                    snapshot.get(
                        "average_confidence",
                        0.50,
                    )
                ),
                "consensus_strength": self._clamp(
                    snapshot.get(
                        "consensus_strength",
                        0.50,
                    )
                ),
                "signal_conflict": self._clamp(
                    snapshot.get(
                        "signal_conflict",
                        0.0,
                    )
                ),
                "risk_override": bool(
                    snapshot.get(
                        "risk_override",
                        False,
                    )
                ),
                "bullish_pct": snapshot.get(
                    "bullish_pct",
                    0.0,
                ),
                "neutral_pct": snapshot.get(
                    "neutral_pct",
                    0.0,
                ),
                "bearish_pct": snapshot.get(
                    "bearish_pct",
                    0.0,
                ),
                "strongest_bullish_agent":
                    snapshot.get(
                        "strongest_bullish_agent"
                    ),
                "strongest_bearish_agent":
                    snapshot.get(
                        "strongest_bearish_agent"
                    ),
                "top_positive_signals":
                    self._safe_list(
                        snapshot.get(
                            "top_positive_signals"
                        )
                    ),
                "top_risks":
                    self._safe_list(
                        snapshot.get(
                            "top_risks"
                        )
                    ),
                "market_interpretation":
                    snapshot.get(
                        "market_interpretation"
                    ),
                "agent_count": len(
                    self._safe_list(
                        snapshot.get("agents")
                    )
                ),
            },
        )

        return task

    def run_orchestrator(
        self,
        orchestrator: Any,
        *,
        asset: Optional[str] = None,
        horizon: Optional[str] = None,
        previous_regime: Optional[str] = None,
    ) -> ResearchTask:
        """
        Execute an existing Orchestrator instance and translate
        its output into a ResearchTask.

        The adapter requires only a callable run() method.
        """

        run_method = getattr(
            orchestrator,
            "run",
            None,
        )

        if not callable(run_method):
            raise TypeError(
                "Orchestrator must expose a callable run() method."
            )

        snapshot = run_method()

        if not isinstance(snapshot, dict):
            raise TypeError(
                "Orchestrator.run() must return a dictionary."
            )

        return self.create_research_task(
            snapshot,
            asset=asset,
            horizon=horizon,
            previous_regime=previous_regime,
        )