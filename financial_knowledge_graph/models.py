
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Optional, Tuple


class KnowledgeGraphValidationError(ValueError):
    """Invalid financial knowledge graph data."""


class NodeType(str, Enum):
    MACRO_EVENT = "MACRO_EVENT"
    ECONOMIC_MECHANISM = "ECONOMIC_MECHANISM"
    MARKET_VARIABLE = "MARKET_VARIABLE"
    ASSET = "ASSET"
    PORTFOLIO = "PORTFOLIO"
    OUTCOME = "OUTCOME"


class RelationshipType(str, Enum):
    TRIGGERS = "TRIGGERS"
    INFLUENCES = "INFLUENCES"
    AFFECTS = "AFFECTS"
    CONTAINS = "CONTAINS"
    PRODUCES = "PRODUCES"


class EvidenceType(str, Enum):
    AGENT_RESEARCH = "AGENT_RESEARCH"
    MARKET_DATA = "MARKET_DATA"
    ECONOMIC_DATA = "ECONOMIC_DATA"
    NEWS = "NEWS"
    BACKTEST = "BACKTEST"
    OBSERVED_OUTCOME = "OBSERVED_OUTCOME"
    OTHER = "OTHER"


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeGraphValidationError(
            f"{name} must be a non-empty string."
        )


def _require_aware_datetime(
    value: datetime,
    name: str,
) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise KnowledgeGraphValidationError(
            f"{name} must be a timezone-aware datetime."
        )


def _require_confidence(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise KnowledgeGraphValidationError(
            "confidence must be a finite number between 0 and 1."
        )


@dataclass(frozen=True)
class ValidityPeriod:
    """
    Historical validity interval.

    valid_from is inclusive.
    valid_until is exclusive when provided.
    """

    valid_from: datetime
    valid_until: Optional[datetime] = None

    def __post_init__(self) -> None:
        _require_aware_datetime(
            self.valid_from,
            "valid_from",
        )

        if self.valid_until is not None:
            _require_aware_datetime(
                self.valid_until,
                "valid_until",
            )

            if self.valid_until <= self.valid_from:
                raise KnowledgeGraphValidationError(
                    "valid_until must be later than valid_from."
                )

    def contains(self, moment: datetime) -> bool:
        _require_aware_datetime(moment, "moment")

        if moment < self.valid_from:
            return False

        return (
            self.valid_until is None
            or moment < self.valid_until
        )


@dataclass(frozen=True)
class KnowledgeNode:
    node_id: str
    node_type: NodeType
    label: str
    description: str = ""
    attributes: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _require_text(self.node_id, "node_id")
        _require_text(self.label, "label")

        if not isinstance(self.node_type, NodeType):
            raise KnowledgeGraphValidationError(
                "node_type must be a NodeType value."
            )

        if not isinstance(self.description, str):
            raise KnowledgeGraphValidationError(
                "description must be a string."
            )

        if not isinstance(self.attributes, Mapping):
            raise KnowledgeGraphValidationError(
                "attributes must be a mapping."
            )

        object.__setattr__(
            self,
            "attributes",
            dict(self.attributes),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_type: EvidenceType
    source: str
    summary: str
    observed_at: datetime
    reference: Optional[str] = None

    def __post_init__(self) -> None:
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.source, "source")
        _require_text(self.summary, "summary")
        _require_aware_datetime(
            self.observed_at,
            "observed_at",
        )

        if not isinstance(self.evidence_type, EvidenceType):
            raise KnowledgeGraphValidationError(
                "evidence_type must be an EvidenceType value."
            )

        if self.reference is not None:
            _require_text(self.reference, "reference")


@dataclass(frozen=True)
class KnowledgeRelationship:
    relationship_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    confidence: float
    validity: ValidityPeriod
    evidence_ids: Tuple[str, ...] = ()
    hypothesis_status: HypothesisStatus = (
        HypothesisStatus.PROPOSED
    )
    agent_id: Optional[str] = None
    rationale: str = ""

    def __post_init__(self) -> None:
        _require_text(
            self.relationship_id,
            "relationship_id",
        )
        _require_text(
            self.source_node_id,
            "source_node_id",
        )
        _require_text(
            self.target_node_id,
            "target_node_id",
        )
        _require_confidence(self.confidence)

        if self.source_node_id == self.target_node_id:
            raise KnowledgeGraphValidationError(
                "A relationship cannot link a node to itself."
            )

        if not isinstance(
            self.relationship_type,
            RelationshipType,
        ):
            raise KnowledgeGraphValidationError(
                "relationship_type must be "
                "a RelationshipType value."
            )

        if not isinstance(self.validity, ValidityPeriod):
            raise KnowledgeGraphValidationError(
                "validity must be a ValidityPeriod."
            )

        if not isinstance(
            self.hypothesis_status,
            HypothesisStatus,
        ):
            raise KnowledgeGraphValidationError(
                "hypothesis_status must be "
                "a HypothesisStatus value."
            )

        if not isinstance(self.evidence_ids, tuple):
            raise KnowledgeGraphValidationError(
                "evidence_ids must be a tuple."
            )

        for evidence_id in self.evidence_ids:
            _require_text(evidence_id, "evidence_id")

        if len(set(self.evidence_ids)) != len(
            self.evidence_ids
        ):
            raise KnowledgeGraphValidationError(
                "Duplicate evidence IDs are not allowed."
            )

        if self.agent_id is not None:
            _require_text(self.agent_id, "agent_id")

        if not isinstance(self.rationale, str):
            raise KnowledgeGraphValidationError(
                "rationale must be a string."
            )


@dataclass(frozen=True)
class ObservedOutcome:
    """
    Records an observed result without asserting causation.
    """

    outcome_id: str
    outcome_node_id: str
    observed_at: datetime
    value: float
    unit: str
    evidence_ids: Tuple[str, ...] = ()
    related_relationship_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.outcome_id, "outcome_id")
        _require_text(
            self.outcome_node_id,
            "outcome_node_id",
        )
        _require_aware_datetime(
            self.observed_at,
            "observed_at",
        )
        _require_text(self.unit, "unit")

        if (
            isinstance(self.value, bool)
            or not isinstance(self.value, (int, float))
            or not isfinite(self.value)
        ):
            raise KnowledgeGraphValidationError(
                "value must be a finite number."
            )

        for name, identifiers in (
            ("evidence_ids", self.evidence_ids),
            (
                "related_relationship_ids",
                self.related_relationship_ids,
            ),
        ):
            if not isinstance(identifiers, tuple):
                raise KnowledgeGraphValidationError(
                    f"{name} must be a tuple."
                )

            for identifier in identifiers:
                _require_text(identifier, name)

            if len(set(identifiers)) != len(identifiers):
                raise KnowledgeGraphValidationError(
                    f"{name} cannot contain duplicates."
                )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "Evidence",
    "EvidenceType",
    "HypothesisStatus",
    "KnowledgeGraphValidationError",
    "KnowledgeNode",
    "KnowledgeRelationship",
    "NodeType",
    "ObservedOutcome",
    "RelationshipType",
    "ValidityPeriod",
    "utc_now",
]