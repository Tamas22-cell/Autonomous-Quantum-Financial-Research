"""
v5.0.9 - Autonomous Research Loop
Learning Feedback Engine.

Transforms observed ResearchOutcome objects into structured
LearningFeedback.

The feedback can later be consumed by:
- Meta-Learning
- Alpha Lifecycle
- Agent Tournament
- Autonomous Research Loop
- new Research Tasks

Safety:
- no live trading
- no broker execution
- no source-code modification
- no automatic production promotion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from autonomous_research_loop.models import (
    LearningFeedback,
    ResearchOutcome,
)


@dataclass(frozen=True)
class FeedbackEngineConfig:
    """
    Configuration for research feedback generation.
    """

    strong_performance_score: float = 0.75
    weak_performance_score: float = 0.40

    strong_alpha: float = 0.05
    weak_alpha: float = 0.0

    acceptable_drawdown: float = 0.20

    create_task_on_failure: bool = True
    create_task_on_weak_performance: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.strong_performance_score <= 1.0:
            raise ValueError(
                "strong_performance_score must be between 0 and 1."
            )

        if not 0.0 <= self.weak_performance_score <= 1.0:
            raise ValueError(
                "weak_performance_score must be between 0 and 1."
            )

        if (
            self.weak_performance_score
            > self.strong_performance_score
        ):
            raise ValueError(
                "weak_performance_score cannot exceed "
                "strong_performance_score."
            )

        if not 0.0 <= self.acceptable_drawdown <= 1.0:
            raise ValueError(
                "acceptable_drawdown must be between 0 and 1."
            )


class FeedbackEngine:
    """
    Convert ResearchOutcome into LearningFeedback.

    Performance score combines:
    - observed return
    - alpha
    - drawdown quality
    - hit/miss outcome

    The result is bounded to [0, 1].
    """

    def __init__(
        self,
        config: Optional[FeedbackEngineConfig] = None,
    ) -> None:
        self.config = config or FeedbackEngineConfig()

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    @staticmethod
    def _get_drawdown(
        outcome: ResearchOutcome,
    ) -> float:
        """
        Read normalized drawdown from outcome metadata.
        """

        value = outcome.metadata.get(
            "max_drawdown",
            0.0,
        )

        try:
            drawdown = abs(
                float(value)
            )

        except (TypeError, ValueError):
            return 0.0

        return min(
            drawdown,
            1.0,
        )

    def calculate_performance_score(
        self,
        outcome: ResearchOutcome,
    ) -> float:
        """
        Produce a normalized research-performance score.

        Components:
        - hit quality:       35%
        - return quality:    25%
        - alpha quality:     25%
        - drawdown quality:  15%
        """

        if not isinstance(
            outcome,
            ResearchOutcome,
        ):
            raise TypeError(
                "FeedbackEngine requires a ResearchOutcome."
            )

        hit_score = (
            1.0
            if outcome.hit is True
            else 0.0
        )

        return_score = self._clamp(
            0.5
            + float(outcome.return_pct) * 5.0
        )

        alpha_value = (
            0.0
            if outcome.alpha is None
            else float(outcome.alpha)
        )

        alpha_score = self._clamp(
            0.5
            + alpha_value * 5.0
        )

        drawdown = self._get_drawdown(
            outcome
        )

        if self.config.acceptable_drawdown == 0.0:
            drawdown_score = (
                1.0
                if drawdown == 0.0
                else 0.0
            )

        else:
            drawdown_score = self._clamp(
                1.0
                - (
                    drawdown
                    / self.config.acceptable_drawdown
                )
            )

        score = (
            hit_score * 0.35
            + return_score * 0.25
            + alpha_score * 0.25
            + drawdown_score * 0.15
        )

        return round(
            self._clamp(score),
            4,
        )

    def build_lessons(
        self,
        outcome: ResearchOutcome,
        performance_score: float,
    ) -> List[str]:
        """
        Generate deterministic research lessons.
        """

        lessons: List[str] = []

        drawdown = self._get_drawdown(
            outcome
        )

        alpha_value = (
            0.0
            if outcome.alpha is None
            else float(outcome.alpha)
        )

        if outcome.hit is True:
            lessons.append(
                "Shadow observation satisfied the configured "
                "research success criteria."
            )
        else:
            lessons.append(
                "Shadow observation failed at least one configured "
                "research success criterion."
            )

        if outcome.return_pct > 0.0:
            lessons.append(
                "The candidate produced a positive forward return."
            )

        elif outcome.return_pct < 0.0:
            lessons.append(
                "The candidate produced a negative forward return."
            )

        else:
            lessons.append(
                "The candidate produced no positive forward return."
            )

        if alpha_value > self.config.strong_alpha:
            lessons.append(
                "The candidate generated strong positive excess return."
            )

        elif alpha_value > self.config.weak_alpha:
            lessons.append(
                "The candidate generated positive excess return."
            )

        elif alpha_value < 0.0:
            lessons.append(
                "The candidate underperformed its benchmark."
            )

        else:
            lessons.append(
                "The candidate generated no positive excess return."
            )

        if drawdown > self.config.acceptable_drawdown:
            lessons.append(
                "Observed drawdown exceeded the acceptable "
                "research threshold."
            )

        elif drawdown > (
            self.config.acceptable_drawdown * 0.75
        ):
            lessons.append(
                "Observed drawdown was elevated relative to "
                "the configured research threshold."
            )

        else:
            lessons.append(
                "Observed drawdown remained within the "
                "configured research threshold."
            )

        if (
            performance_score
            >= self.config.strong_performance_score
        ):
            lessons.append(
                "Overall shadow performance was strong."
            )

        elif (
            performance_score
            < self.config.weak_performance_score
        ):
            lessons.append(
                "Overall shadow performance was weak."
            )

        else:
            lessons.append(
                "Overall shadow performance was mixed."
            )

        return lessons

    def build_recommended_actions(
        self,
        outcome: ResearchOutcome,
        performance_score: float,
    ) -> List[str]:
        """
        Produce safe research-only follow-up actions.
        """

        actions: List[str] = []

        drawdown = self._get_drawdown(
            outcome
        )

        alpha_value = (
            0.0
            if outcome.alpha is None
            else float(outcome.alpha)
        )

        if (
            outcome.hit is True
            and performance_score
            >= self.config.strong_performance_score
        ):
            actions.append(
                "Retain the candidate in research tracking "
                "for additional out-of-sample validation."
            )

            actions.append(
                "Evaluate robustness across additional regimes, "
                "assets and observation windows."
            )

        else:
            actions.append(
                "Review the candidate assumptions and "
                "validation evidence."
            )

        if alpha_value <= 0.0:
            actions.append(
                "Investigate benchmark underperformance and "
                "reassess the alpha hypothesis."
            )

        if drawdown > self.config.acceptable_drawdown:
            actions.append(
                "Reassess risk constraints, sizing and "
                "drawdown sensitivity."
            )

        if (
            performance_score
            < self.config.weak_performance_score
        ):
            actions.append(
                "Generate a new research hypothesis addressing "
                "the observed performance weakness."
            )

        actions.append(
            "Keep all follow-up activity in research or "
            "shadow mode until independently validated."
        )

        return actions

    def should_create_new_task(
        self,
        outcome: ResearchOutcome,
        performance_score: float,
    ) -> bool:
        """
        Decide whether observed evidence should generate
        another research task.
        """

        if (
            self.config.create_task_on_failure
            and outcome.hit is not True
        ):
            return True

        if (
            self.config.create_task_on_weak_performance
            and performance_score
            < self.config.weak_performance_score
        ):
            return True

        return False

    def generate(
        self,
        outcome: ResearchOutcome,
    ) -> LearningFeedback:
        """
        Create LearningFeedback from ResearchOutcome.
        """

        if not isinstance(
            outcome,
            ResearchOutcome,
        ):
            raise TypeError(
                "FeedbackEngine requires a ResearchOutcome."
            )

        performance_score = (
            self.calculate_performance_score(
                outcome
            )
        )

        lessons = self.build_lessons(
            outcome,
            performance_score,
        )

        recommended_actions = (
            self.build_recommended_actions(
                outcome,
                performance_score,
            )
        )

        create_new_task = (
            self.should_create_new_task(
                outcome,
                performance_score,
            )
        )

        return LearningFeedback(
            strategy_id=outcome.strategy_id,
            outcome_id=outcome.outcome_id,
            performance_score=performance_score,
            lessons=lessons,
            recommended_actions=recommended_actions,
            create_new_research_task=create_new_task,
            metadata={
                "engine":
                    "v5.0.9-feedback-engine",

                "source":
                    "SHADOW_OUTCOME",

                "outcome_hit":
                    outcome.hit,

                "return_pct":
                    outcome.return_pct,

                "alpha":
                    outcome.alpha,

                "max_drawdown":
                    self._get_drawdown(
                        outcome
                    ),

                "research_only":
                    True,

                "real_money_execution":
                    False,

                "automatic_live_promotion":
                    False,

                "feeds_meta_learning":
                    True,

                "feeds_alpha_lifecycle":
                    True,

                "feeds_agent_tournament":
                    True,
            },
        )

    def __call__(
        self,
        outcome: ResearchOutcome,
    ) -> LearningFeedback:
        """
        Callable dependency-injection interface.
        """

        return self.generate(
            outcome
        )