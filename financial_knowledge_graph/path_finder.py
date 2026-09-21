
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Set, Tuple

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
    GraphStoreError,
)
from financial_knowledge_graph.models import (
    HypothesisStatus,
    KnowledgeRelationship,
    _require_aware_datetime,
)


@dataclass(frozen=True)
class GraphPath:
    """
    A directed chain of graph relationships.

    A path represents recorded hypotheses and observations.
    It is not proof of causation.
    """

    node_ids: Tuple[str, ...]
    relationship_ids: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    hypothesis_statuses: Tuple[HypothesisStatus, ...]

    @property
    def length(self) -> int:
        return len(self.relationship_ids)


class FinancialGraphPathFinder:
    """Find directed paths through a financial knowledge graph."""

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

    def find_paths(
        self,
        source_node_id: str,
        target_node_id: str,
        *,
        max_depth: int = 5,
        valid_at: Optional[datetime] = None,
        max_paths: int = 100,
    ) -> Tuple[GraphPath, ...]:
        """
        Find simple directed paths from source to target.

        max_depth:
            Maximum number of relationships in a path.

        valid_at:
            If provided, every relationship must be valid
            at this timezone-aware datetime.

        max_paths:
            Maximum number of paths returned.

        Paths are ordered by increasing length, then by
        relationship IDs for deterministic results.
        """
        if isinstance(max_depth, bool) or not isinstance(
            max_depth, int
        ) or max_depth < 1:
            raise GraphStoreError(
                "max_depth must be a positive integer."
            )

        if isinstance(max_paths, bool) or not isinstance(
            max_paths, int
        ) or max_paths < 1:
            raise GraphStoreError(
                "max_paths must be a positive integer."
            )

        if valid_at is not None:
            _require_aware_datetime(valid_at, "valid_at")

        # Validate both endpoints before searching.
        self._graph.get_node(source_node_id)
        self._graph.get_node(target_node_id)

        if source_node_id == target_node_id:
            return ()

        adjacency: dict[
            str, List[KnowledgeRelationship]
        ] = {}

        for relationship in self._graph.list_relationships(
            valid_at=valid_at
        ):
            adjacency.setdefault(
                relationship.source_node_id, []
            ).append(relationship)

        for outgoing in adjacency.values():
            outgoing.sort(
                key=lambda relationship: relationship.relationship_id
            )

        found: List[GraphPath] = []

        def visit(
            current_node_id: str,
            node_ids: Tuple[str, ...],
            relationships: Tuple[
                KnowledgeRelationship, ...
            ],
            visited: Set[str],
        ) -> None:
            if len(relationships) >= max_depth:
                return

            for relationship in adjacency.get(
                current_node_id, ()
            ):
                next_node_id = relationship.target_node_id

                # Simple paths: do not revisit nodes.
                if next_node_id in visited:
                    continue

                next_nodes = node_ids + (next_node_id,)
                next_relationships = (
                    relationships + (relationship,)
                )

                if next_node_id == target_node_id:
                    evidence_ids = tuple(
                        dict.fromkeys(
                            evidence_id
                            for edge in next_relationships
                            for evidence_id in edge.evidence_ids
                        )
                    )

                    found.append(
                        GraphPath(
                            node_ids=next_nodes,
                            relationship_ids=tuple(
                                edge.relationship_id
                                for edge in next_relationships
                            ),
                            evidence_ids=evidence_ids,
                            hypothesis_statuses=tuple(
                                edge.hypothesis_status
                                for edge in next_relationships
                            ),
                        )
                    )
                    continue

                visit(
                    next_node_id,
                    next_nodes,
                    next_relationships,
                    visited | {next_node_id},
                )

        visit(
            source_node_id,
            (source_node_id,),
            (),
            {source_node_id},
        )

        found.sort(
            key=lambda path: (
                path.length,
                path.relationship_ids,
            )
        )

        return tuple(found[:max_paths])


__all__ = [
    "FinancialGraphPathFinder",
    "GraphPath",
]