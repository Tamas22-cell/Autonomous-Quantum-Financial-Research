
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from financial_knowledge_graph.models import (
    Evidence,
    KnowledgeGraphValidationError,
    KnowledgeNode,
    KnowledgeRelationship,
    NodeType,
    ObservedOutcome,
    RelationshipType,
    _require_aware_datetime,
)


class GraphStoreError(KnowledgeGraphValidationError):
    """Invalid graph operation or reference."""


class FinancialKnowledgeGraphStore:
    """
    In-memory financial knowledge graph.

    Stores nodes, evidence, relationships and observed outcomes.
    Validates references before inserting records.

    This version does not yet persist data to disk.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._evidence: Dict[str, Evidence] = {}
        self._relationships: Dict[
            str, KnowledgeRelationship
        ] = {}
        self._outcomes: Dict[str, ObservedOutcome] = {}

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence)

    @property
    def relationship_count(self) -> int:
        return len(self._relationships)

    @property
    def outcome_count(self) -> int:
        return len(self._outcomes)

    def add_node(self, node: KnowledgeNode) -> None:
        if not isinstance(node, KnowledgeNode):
            raise GraphStoreError(
                "node must be a KnowledgeNode."
            )

        if node.node_id in self._nodes:
            raise GraphStoreError(
                f"Duplicate node ID: {node.node_id}"
            )

        self._nodes[node.node_id] = node

    def add_evidence(self, evidence: Evidence) -> None:
        if not isinstance(evidence, Evidence):
            raise GraphStoreError(
                "evidence must be an Evidence."
            )

        if evidence.evidence_id in self._evidence:
            raise GraphStoreError(
                f"Duplicate evidence ID: {evidence.evidence_id}"
            )

        self._evidence[evidence.evidence_id] = evidence

    def add_relationship(
        self,
        relationship: KnowledgeRelationship,
    ) -> None:
        if not isinstance(
            relationship,
            KnowledgeRelationship,
        ):
            raise GraphStoreError(
                "relationship must be a KnowledgeRelationship."
            )

        if (
            relationship.relationship_id
            in self._relationships
        ):
            raise GraphStoreError(
                "Duplicate relationship ID: "
                f"{relationship.relationship_id}"
            )

        self._require_node(
            relationship.source_node_id
        )
        self._require_node(
            relationship.target_node_id
        )

        for evidence_id in relationship.evidence_ids:
            self._require_evidence(evidence_id)

        self._relationships[
            relationship.relationship_id
        ] = relationship

    def add_outcome(
        self,
        outcome: ObservedOutcome,
    ) -> None:
        if not isinstance(outcome, ObservedOutcome):
            raise GraphStoreError(
                "outcome must be an ObservedOutcome."
            )

        if outcome.outcome_id in self._outcomes:
            raise GraphStoreError(
                f"Duplicate outcome ID: {outcome.outcome_id}"
            )

        outcome_node = self._require_node(
            outcome.outcome_node_id
        )

        if outcome_node.node_type != NodeType.OUTCOME:
            raise GraphStoreError(
                "Observed outcome must reference "
                "an OUTCOME node."
            )

        for evidence_id in outcome.evidence_ids:
            self._require_evidence(evidence_id)

        for relationship_id in (
            outcome.related_relationship_ids
        ):
            self._require_relationship(
                relationship_id
            )

        self._outcomes[outcome.outcome_id] = outcome

    def get_node(self, node_id: str) -> KnowledgeNode:
        return self._require_node(node_id)

    def get_evidence(
        self,
        evidence_id: str,
    ) -> Evidence:
        return self._require_evidence(evidence_id)

    def get_relationship(
        self,
        relationship_id: str,
    ) -> KnowledgeRelationship:
        return self._require_relationship(
            relationship_id
        )

    def get_outcome(
        self,
        outcome_id: str,
    ) -> ObservedOutcome:
        try:
            return self._outcomes[outcome_id]
        except (KeyError, TypeError) as exc:
            raise GraphStoreError(
                f"Unknown outcome ID: {outcome_id}"
            ) from exc

    def list_nodes(
        self,
        node_type: Optional[NodeType] = None,
    ) -> Tuple[KnowledgeNode, ...]:
        if (
            node_type is not None
            and not isinstance(node_type, NodeType)
        ):
            raise GraphStoreError(
                "node_type must be a NodeType value."
            )

        return tuple(
            node
            for node in self._nodes.values()
            if (
                node_type is None
                or node.node_type == node_type
            )
        )

    def list_evidence(self) -> Tuple[Evidence, ...]:
        return tuple(self._evidence.values())

    def list_relationships(
        self,
        *,
        source_node_id: Optional[str] = None,
        target_node_id: Optional[str] = None,
        relationship_type: Optional[
            RelationshipType
        ] = None,
        valid_at: Optional[datetime] = None,
    ) -> Tuple[KnowledgeRelationship, ...]:
        if (
            relationship_type is not None
            and not isinstance(
                relationship_type,
                RelationshipType,
            )
        ):
            raise GraphStoreError(
                "relationship_type must be "
                "a RelationshipType value."
            )

        if valid_at is not None:
            _require_aware_datetime(
                valid_at,
                "valid_at",
            )

        results: List[KnowledgeRelationship] = []

        for relationship in (
            self._relationships.values()
        ):
            if (
                source_node_id is not None
                and relationship.source_node_id
                != source_node_id
            ):
                continue

            if (
                target_node_id is not None
                and relationship.target_node_id
                != target_node_id
            ):
                continue

            if (
                relationship_type is not None
                and relationship.relationship_type
                != relationship_type
            ):
                continue

            if (
                valid_at is not None
                and not relationship.validity.contains(
                    valid_at
                )
            ):
                continue

            results.append(relationship)

        return tuple(results)

    def list_outcomes(
        self,
        *,
        outcome_node_id: Optional[str] = None,
    ) -> Tuple[ObservedOutcome, ...]:
        return tuple(
            outcome
            for outcome in self._outcomes.values()
            if (
                outcome_node_id is None
                or outcome.outcome_node_id
                == outcome_node_id
            )
        )

    def _require_node(
        self,
        node_id: str,
    ) -> KnowledgeNode:
        try:
            return self._nodes[node_id]
        except (KeyError, TypeError) as exc:
            raise GraphStoreError(
                f"Unknown node ID: {node_id}"
            ) from exc

    def _require_evidence(
        self,
        evidence_id: str,
    ) -> Evidence:
        try:
            return self._evidence[evidence_id]
        except (KeyError, TypeError) as exc:
            raise GraphStoreError(
                f"Unknown evidence ID: {evidence_id}"
            ) from exc

    def _require_relationship(
        self,
        relationship_id: str,
    ) -> KnowledgeRelationship:
        try:
            return self._relationships[relationship_id]
        except (KeyError, TypeError) as exc:
            raise GraphStoreError(
                "Unknown relationship ID: "
                f"{relationship_id}"
            ) from exc


__all__ = [
    "FinancialKnowledgeGraphStore",
    "GraphStoreError",
]