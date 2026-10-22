"""
Causal Research Engine - Event Study Engine
Version: v5.0.5

Provides event-study and intervention-style causal analysis.

The engine compares pre-event and post-event outcomes and produces
a structured EventStudyResult for the causal evidence layer.
"""

from __future__ import annotations

from statistics import mean
from typing import Iterable, Optional

from causal_research_engine.models import EventStudyResult


class EventStudyEngineError(Exception):
    """Base exception for event-study errors."""


class EventStudyEngine:
    """
    Performs event-study style analysis.

    Core responsibilities:
    - calculate pre-event mean
    - calculate post-event mean
    - estimate abnormal effect
    - estimate cumulative abnormal effect
    - optionally assess statistical significance
    """

    def analyze(
        self,
        *,
        event_name: str,
        pre_event_values: Iterable[float],
        post_event_values: Iterable[float],
        benchmark_effect: Optional[float] = None,
        p_value: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> EventStudyResult:
        """
        Analyze one event.

        If benchmark_effect is supplied, the abnormal effect is:

            post_mean - pre_mean - benchmark_effect

        Otherwise:

            post_mean - pre_mean
        """

        if not event_name or not event_name.strip():
            raise EventStudyEngineError(
                "event_name is required."
            )

        pre_values = self._validate_values(
            pre_event_values,
            "pre_event_values",
        )

        post_values = self._validate_values(
            post_event_values,
            "post_event_values",
        )

        pre_mean = mean(pre_values)
        post_mean = mean(post_values)

        raw_effect = post_mean - pre_mean

        if benchmark_effect is not None:
            abnormal_effect = (
                raw_effect - float(benchmark_effect)
            )
        else:
            abnormal_effect = raw_effect

        return EventStudyResult(
            event_name=event_name.strip(),
            pre_event_mean=pre_mean,
            post_event_mean=post_mean,
            abnormal_effect=abnormal_effect,
            cumulative_abnormal_effect=abnormal_effect,
            p_value=p_value,
            confidence=confidence,
            notes=notes,
        )

    def intervention_effect(
        self,
        *,
        event_name: str,
        pre_event_values: Iterable[float],
        post_event_values: Iterable[float],
        benchmark_effect: float = 0.0,
        p_value: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> EventStudyResult:
        """
        Estimate an intervention-style effect.

        This is equivalent to comparing the post-event mean with
        the counterfactual benchmark-adjusted pre-event level.
        """

        return self.analyze(
            event_name=event_name,
            pre_event_values=pre_event_values,
            post_event_values=post_event_values,
            benchmark_effect=benchmark_effect,
            p_value=p_value,
            confidence=confidence,
            notes=notes,
        )

    def cumulative_effect(
        self,
        *,
        event_name: str,
        abnormal_returns: Iterable[float],
        p_value: Optional[float] = None,
        confidence: float = 0.0,
        notes: str = "",
    ) -> EventStudyResult:
        """
        Calculate cumulative abnormal effect from an event window.
        """

        values = self._validate_values(
            abnormal_returns,
            "abnormal_returns",
        )

        cumulative = sum(values)

        return EventStudyResult(
            event_name=event_name.strip(),
            abnormal_effect=mean(values),
            cumulative_abnormal_effect=cumulative,
            p_value=p_value,
            confidence=confidence,
            notes=notes,
        )

    @staticmethod
    def _validate_values(
        values: Iterable[float],
        field_name: str,
    ) -> list[float]:
        """
        Validate and normalize numeric observations.
        """

        if values is None:
            raise EventStudyEngineError(
                f"{field_name} is required."
            )

        result: list[float] = []

        for value in values:
            try:
                result.append(float(value))
            except (TypeError, ValueError) as exc:
                raise EventStudyEngineError(
                    f"{field_name} contains a non-numeric value: "
                    f"{value!r}"
                ) from exc

        if not result:
            raise EventStudyEngineError(
                f"{field_name} cannot be empty."
            )

        return result