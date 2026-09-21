
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from math import isfinite
from typing import Optional, Tuple

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
    GraphStoreError,
)
from financial_knowledge_graph.models import (
    HypothesisStatus,
    ObservedOutcome,
    _require_aware_datetime,
)


class ExpectedDirection(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class OutcomeAssessment(str, Enum):
    ALIGNED = "ALIGNED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class OutcomeEvaluation:
    """
    Comparison between an expected direction and an observed result.

    ALIGNED does not establish a causal relationship.
    """

    relationship_id: str
    outcome_id: str
    expected_direction: ExpectedDirection
    observed_value: float
    assessment: OutcomeAssessment
    hypothesis_status: HypothesisStatus
    evidence_ids: Tuple[str, ...]
    evaluated_at: datetime


class FinancialOutcomeEvaluator:
    """
    Evaluate a recorded hypothesis against an observed outcome.

    The caller must explicitly provide the expected direction.
    The evaluation does not mutate the graph or its hypothesis status.
    """

    def __init__(
        self,
        graph: FinancialKnowledgeGraphStore,
    ) -> None:
        if not isinstance(
            graph,
            FinancialKnowledgeGraphStore,
        ):
            raise GraphStoreError(
                "graph must be a FinancialKnowledgeGraphStore."
            )

        self._graph = graph

    def evaluate(
        self,
        relationship_id: str,
        outcome_id: str,
        *,
        expected_direction: ExpectedDirection,
        neutral_tolerance: float = 0.0,
        evaluated_at: Optional[datetime] = None,
    ) -> OutcomeEvaluation:
        """
        Compare an expected direction with an observed value.

        neutral_tolerance defines the absolute-value interval
        considered neutral, in the same unit as the outcome.

        The outcome must explicitly reference the relationship.
        """

        if not isinstance(
            expected_direction,
            ExpectedDirection,
        ):
            raise GraphStoreError(
                "expected_direction must be "
                "an ExpectedDirection value."
            )

        if (
            isinstance(neutral_tolerance, bool)
            or not isinstance(
                neutral_tolerance,
                (int, float),
            )
            or not isfinite(neutral_tolerance)
            or neutral_tolerance < 0
        ):
            raise GraphStoreError(
                "neutral_tolerance must be "
                "a finite non-negative number."
            )

        relationship = self._graph.get_relationship(
            relationship_id
        )
        outcome = self._graph.get_outcome(outcome_id)

        if (
            relationship_id
            not in outcome.related_relationship_ids
        ):
            raise GraphStoreError(
                "Outcome does not explicitly reference "
                f"relationship ID: {relationship_id}"
            )

        if (
            outcome.observed_at
            < relationship.validity.valid_from
        ):
            raise GraphStoreError(
                "Outcome was observed before "
                "the relationship validity period."
            )

        if evaluated_at is None:
            evaluated_at = outcome.observed_at

        _require_aware_datetime(
            evaluated_at,
            "evaluated_at",
        )

        if evaluated_at < outcome.observed_at:
            raise GraphStoreError(
                "evaluated_at cannot be earlier "
                "than outcome.observed_at."
            )

        assessment = self._assess_direction(
            expected_direction,
            outcome,
            neutral_tolerance,
        )

        evidence_ids = tuple(
            dict.fromkeys(
                relationship.evidence_ids
                + outcome.evidence_ids
            )
        )

        return OutcomeEvaluation(
            relationship_id=relationship_id,
            outcome_id=outcome_id,
            expected_direction=expected_direction,
            observed_value=outcome.value,
            assessment=assessment,
            hypothesis_status=relationship.hypothesis_status,
            evidence_ids=evidence_ids,
            evaluated_at=evaluated_at,
        )

    @staticmethod
    def _assess_direction(
        expected_direction: ExpectedDirection,
        outcome: ObservedOutcome,
        neutral_tolerance: float,
    ) -> OutcomeAssessment:
        value = outcome.value

        if abs(value) <= neutral_tolerance:
            observed_direction = ExpectedDirection.NEUTRAL
        elif value > 0:
            observed_direction = ExpectedDirection.POSITIVE
        else:
            observed_direction = ExpectedDirection.NEGATIVE

        if observed_direction == ExpectedDirection.NEUTRAL:
            if expected_direction == ExpectedDirection.NEUTRAL:
                return OutcomeAssessment.ALIGNED

            return OutcomeAssessment.INCONCLUSIVE

        if expected_direction == ExpectedDirection.NEUTRAL:
            return OutcomeAssessment.CONTRADICTED

        if observed_direction == expected_direction:
            return OutcomeAssessment.ALIGNED

        return OutcomeAssessment.CONTRADICTED


__all__ = [
    "ExpectedDirection",
    "FinancialOutcomeEvaluator",
    "OutcomeAssessment",
    "OutcomeEvaluation",
]