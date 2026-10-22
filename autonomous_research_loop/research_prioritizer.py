"""
v5.0.9 - Autonomous Research Loop
Research Prioritizer.

Determines which research tasks should be investigated first.

Priority is driven by:
- uncertainty
- performance gaps
- regime changes
- alpha decay
- risk events
- agent-generated signals

The prioritizer does not execute research.
It only scores and ranks research tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from autonomous_research_loop.models import (
    ResearchPriority,
    ResearchStatus,
    ResearchTask,
    ResearchTrigger,
)


@dataclass(frozen=True)
class PrioritizationConfig:
    """
    Configuration for autonomous research prioritization.
    """

    uncertainty_weight: float = 0.45
    performance_gap_weight: float = 0.35
    trigger_weight: float = 0.20

    high_threshold: float = 0.70
    critical_threshold: float = 0.90
    medium_threshold: float = 0.40

    def __post_init__(self) -> None:
        weights = (
            self.uncertainty_weight,
            self.performance_gap_weight,
            self.trigger_weight,
        )

        if any(weight < 0.0 for weight in weights):
            raise ValueError("Prioritization weights cannot be negative.")

        total_weight = sum(weights)

        if total_weight <= 0.0:
            raise ValueError(
                "At least one prioritization weight must be positive."
            )

        thresholds = (
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

        if not all(0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError(
                "Priority thresholds must be between 0.0 and 1.0."
            )

        if not (
            self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):
            raise ValueError(
                "Priority thresholds must be ordered "
                "MEDIUM <= HIGH <= CRITICAL."
            )


class ResearchPrioritizer:
    """
    Score and rank autonomous research tasks.

    Scores are normalized into the [0, 1] interval.

    Risk and regime-change events receive higher trigger scores because
    they can materially change portfolio suitability or invalidate
    previously successful research.
    """

    TRIGGER_SCORES = {
        ResearchTrigger.MANUAL: 0.30,
        ResearchTrigger.AGENT_SIGNAL: 0.50,
        ResearchTrigger.UNCERTAINTY: 0.65,
        ResearchTrigger.PERFORMANCE_GAP: 0.75,
        ResearchTrigger.ALPHA_DECAY: 0.80,
        ResearchTrigger.REGIME_CHANGE: 0.90,
        ResearchTrigger.RISK_EVENT: 1.00,
    }

    def __init__(
        self,
        config: PrioritizationConfig | None = None,
    ) -> None:
        self.config = config or PrioritizationConfig()

    @staticmethod
    def _clamp(value: float) -> float:
        """
        Clamp a numeric value into [0, 1].
        """
        return max(0.0, min(1.0, float(value)))

    def trigger_score(
        self,
        trigger: ResearchTrigger,
    ) -> float:
        """
        Return the configured importance of a research trigger.
        """
        return self.TRIGGER_SCORES.get(trigger, 0.0)

    def score_task(
        self,
        task: ResearchTask,
    ) -> float:
        """
        Calculate normalized priority score for a research task.
        """

        uncertainty = self._clamp(task.uncertainty_score)
        performance_gap = self._clamp(task.performance_gap_score)
        trigger = self.trigger_score(task.trigger)

        config = self.config

        weighted_score = (
            uncertainty * config.uncertainty_weight
            + performance_gap * config.performance_gap_weight
            + trigger * config.trigger_weight
        )

        total_weight = (
            config.uncertainty_weight
            + config.performance_gap_weight
            + config.trigger_weight
        )

        score = weighted_score / total_weight

        return round(self._clamp(score), 6)

    def priority_from_score(
        self,
        score: float,
    ) -> ResearchPriority:
        """
        Convert numeric priority score into a priority class.
        """

        score = self._clamp(score)

        if score >= self.config.critical_threshold:
            return ResearchPriority.CRITICAL

        if score >= self.config.high_threshold:
            return ResearchPriority.HIGH

        if score >= self.config.medium_threshold:
            return ResearchPriority.MEDIUM

        return ResearchPriority.LOW

    def prioritize(
        self,
        task: ResearchTask,
    ) -> float:
        """
        Score one task and update its priority/status.
        """

        score = self.score_task(task)

        task.priority = self.priority_from_score(score)
        task.update_status(ResearchStatus.PRIORITIZED)

        task.metadata["priority_score"] = score
        task.metadata["trigger_score"] = self.trigger_score(task.trigger)

        return score

    def rank_tasks(
        self,
        tasks: Iterable[ResearchTask],
    ) -> List[Tuple[ResearchTask, float]]:
        """
        Prioritize and rank tasks from highest to lowest score.
        """

        ranked: List[Tuple[ResearchTask, float]] = []

        for task in tasks:
            score = self.prioritize(task)
            ranked.append((task, score))

        ranked.sort(
            key=lambda item: (
                item[1],
                item[0].created_at,
                item[0].task_id,
            ),
            reverse=True,
        )

        return ranked

    def select_next(
        self,
        tasks: Iterable[ResearchTask],
    ) -> ResearchTask | None:
        """
        Select the highest-priority research task.

        Returns None when no tasks are available.
        """

        ranked = self.rank_tasks(tasks)

        if not ranked:
            return None

        return ranked[0][0]