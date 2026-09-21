from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
)
from financial_knowledge_graph.models import (
    HypothesisStatus,
    KnowledgeRelationship,
    RelationshipType,
    ValidityPeriod,
)


class RelationshipManagerError(RuntimeError):
    """Base error for v7 research relationship operations."""


class RelationshipApprovalRequiredError(
    RelationshipManagerError
):
    """Raised when a relationship lacks explicit approval."""


class ResearchRelationshipManager:
    """
    Manages explicit research relationships in the Knowledge Graph.

    Relationships are never inferred automatically.
    Every relationship requires explicit approval.
    """

    VERSION = "v7.0.3"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False

    def __init__(
        self,
        *,
        store: Optional[
            FinancialKnowledgeGraphStore
        ] = None,
    ) -> None:
        self.store = (
            store
            or FinancialKnowledgeGraphStore()
        )

    def add_relationship(
        self,
        *,
        relationship_id: str,
        source_node_id: str,
        target_node_id: str,
        relationship_type: RelationshipType,
        confidence: float,
        valid_from: datetime,
        valid_until: Optional[datetime] = None,
        evidence_ids: Tuple[str, ...] = (),
        hypothesis_status: HypothesisStatus = (
            HypothesisStatus.PROPOSED
        ),
        agent_id: Optional[str] = None,
        rationale: str = "",
        approved: bool = False,
    ) -> KnowledgeRelationship:
        if approved is not True:
            raise RelationshipApprovalRequiredError(
                "Explicit approval is required before "
                "a research relationship can be stored."
            )

        relationship = KnowledgeRelationship(
            relationship_id=relationship_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relationship_type=relationship_type,
            confidence=confidence,
            validity=ValidityPeriod(
                valid_from=valid_from,
                valid_until=valid_until,
            ),
            evidence_ids=tuple(evidence_ids),
            hypothesis_status=hypothesis_status,
            agent_id=agent_id,
            rationale=rationale,
        )

        self.store.add_relationship(
            relationship
        )

        return relationship

    def get_relationship(
        self,
        relationship_id: str,
    ) -> KnowledgeRelationship:
        return self.store.get_relationship(
            relationship_id
        )

    def list_relationships(
        self,
    ) -> Tuple[KnowledgeRelationship, ...]:
        return self.store.list_relationships()

    def relationship_count(self) -> int:
        return self.store.relationship_count

    def summary(self) -> Dict[str, Any]:
        return {
            "component": "research_relationship_manager",
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "relationship_count": (
                self.relationship_count()
            ),
        }