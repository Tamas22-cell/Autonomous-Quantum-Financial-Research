from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


class LineageEngineError(RuntimeError):
    """Base exception for lineage failures."""


class LineageValidationError(LineageEngineError):
    """Raised when lineage input is invalid."""


class LineageNodeNotFoundError(LineageEngineError):
    """Raised when a lineage node cannot be found."""


class LineageEdgeAlreadyExistsError(LineageEngineError):
    """Raised when an identical lineage edge already exists."""


class LineageCycleError(LineageEngineError):
    """Raised when an edge would create a directed cycle."""


class LineageNodeType(str, Enum):
    DATASET = "DATASET"
    CONTRACT = "CONTRACT"
    MODEL_VERSION = "MODEL_VERSION"
    PROMPT_VERSION = "PROMPT_VERSION"
    POLICY_VERSION = "POLICY_VERSION"
    RUN = "RUN"
    EXPERIMENT = "EXPERIMENT"
    ARTIFACT = "ARTIFACT"
    CHECKPOINT = "CHECKPOINT"
    AGENT = "AGENT"
    HYPOTHESIS = "HYPOTHESIS"
    STRATEGY = "STRATEGY"
    VALIDATION = "VALIDATION"
    OTHER = "OTHER"


class LineageRelation(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    USES = "USES"
    PRODUCED_BY = "PRODUCED_BY"
    PRODUCES = "PRODUCES"
    GENERATED_FROM = "GENERATED_FROM"
    VALIDATED_BY = "VALIDATED_BY"
    DEPENDS_ON = "DEPENDS_ON"
    VERSION_OF = "VERSION_OF"
    PARENT_OF = "PARENT_OF"
    RECOVERED_FROM = "RECOVERED_FROM"
    CHECKPOINT_OF = "CHECKPOINT_OF"
    RELATED_TO = "RELATED_TO"


@dataclass
class LineageNode:
    node_id: str
    node_type: LineageNodeType
    label: Optional[str] = None
    fingerprint: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.node_id, str)
            or not self.node_id.strip()
        ):
            raise LineageValidationError(
                "node_id must be a non-empty string"
            )

        self.node_id = self.node_id.strip()

        if not isinstance(
            self.node_type,
            LineageNodeType,
        ):
            raise TypeError(
                "node_type must be LineageNodeType"
            )

        if self.label is not None:
            if (
                not isinstance(self.label, str)
                or not self.label.strip()
            ):
                raise LineageValidationError(
                    "label must be a non-empty string"
                )

            self.label = self.label.strip()

        if self.fingerprint is not None:
            if (
                not isinstance(self.fingerprint, str)
                or not self.fingerprint.strip()
            ):
                raise LineageValidationError(
                    "fingerprint must be a "
                    "non-empty string"
                )

            self.fingerprint = (
                self.fingerprint.strip()
            )

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


@dataclass
class LineageEdge:
    source_id: str
    target_id: str
    relation: LineageRelation
    edge_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "target_id",
        ):
            value = getattr(
                self,
                field_name,
            )

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise LineageValidationError(
                    f"{field_name} must be a "
                    f"non-empty string"
                )

            setattr(
                self,
                field_name,
                value.strip(),
            )

        if self.source_id == self.target_id:
            raise LineageValidationError(
                "self-referencing lineage edges "
                "are not allowed"
            )

        if not isinstance(
            self.relation,
            LineageRelation,
        ):
            raise TypeError(
                "relation must be LineageRelation"
            )

        if self.edge_id is None:
            fingerprint = stable_hash(
                {
                    "source_id": self.source_id,
                    "target_id": self.target_id,
                    "relation": self.relation.value,
                }
            )

            self.edge_id = (
                f"lineage_{fingerprint[:32]}"
            )

        elif (
            not isinstance(self.edge_id, str)
            or not self.edge_id.strip()
        ):
            raise LineageValidationError(
                "edge_id must be a non-empty string"
            )
        else:
            self.edge_id = self.edge_id.strip()

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


@dataclass
class LineagePath:
    node_ids: List[str]
    edge_ids: List[str]
    relations: List[LineageRelation]
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.node_ids,
            list,
        ):
            raise TypeError(
                "node_ids must be a list"
            )

        if not isinstance(
            self.edge_ids,
            list,
        ):
            raise TypeError(
                "edge_ids must be a list"
            )

        if not isinstance(
            self.relations,
            list,
        ):
            raise TypeError(
                "relations must be a list"
            )

        if len(self.node_ids) == 0:
            raise LineageValidationError(
                "path must contain at least one node"
            )

        if len(self.edge_ids) != max(
            0,
            len(self.node_ids) - 1,
        ):
            raise LineageValidationError(
                "edge count does not match path"
            )

        if len(self.relations) != len(
            self.edge_ids
        ):
            raise LineageValidationError(
                "relation count does not match edges"
            )

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    @property
    def length(self) -> int:
        return len(self.edge_ids)


@dataclass
class LineageSnapshot:
    total_nodes: int
    total_edges: int
    root_nodes: int
    leaf_nodes: int
    node_type_counts: Dict[str, int]
    relation_counts: Dict[str, int]
    created_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


class LineageEngine:
    """
    Directed research-lineage graph.

    Tracks provenance across:
    datasets -> versions -> runs -> experiments -> artifacts.

    The engine supports:
    - deterministic edges,
    - ancestry and descendants,
    - direct parents and children,
    - shortest lineage paths,
    - cycle protection,
    - node/edge filtering,
    - lineage fingerprints,
    - graph snapshots,
    - full audit history.

    This component stores research provenance only.
    It does not authorize or execute live trades.
    """

    VERSION = "v6.0.1"

    def __init__(self) -> None:
        self._nodes: Dict[
            str,
            LineageNode,
        ] = {}

        self._node_order: List[str] = []

        self._edges: Dict[
            str,
            LineageEdge,
        ] = {}

        self._edge_order: List[str] = []

        self._outgoing: Dict[
            str,
            List[str],
        ] = {}

        self._incoming: Dict[
            str,
            List[str],
        ] = {}

        self._audit_events: List[
            Dict[str, Any]
        ] = []

    # ========================================================
    # Audit
    # ========================================================

    def _audit(
        self,
        event: str,
        **details: Any,
    ) -> None:
        self._audit_events.append(
            {
                "event": event,
                "created_at": utc_now(),
                "details": details,
            }
        )

    # ========================================================
    # Nodes
    # ========================================================

    def add_node(
        self,
        node: LineageNode,
    ) -> LineageNode:
        if not isinstance(
            node,
            LineageNode,
        ):
            raise TypeError(
                "node must be LineageNode"
            )

        if node.node_id in self._nodes:
            existing = self._nodes[
                node.node_id
            ]

            if (
                existing.node_type
                != node.node_type
            ):
                raise LineageValidationError(
                    "existing node has a different "
                    "node type"
                )

            return existing

        self._nodes[
            node.node_id
        ] = node

        self._node_order.append(
            node.node_id
        )

        self._outgoing.setdefault(
            node.node_id,
            [],
        )

        self._incoming.setdefault(
            node.node_id,
            [],
        )

        self._audit(
            "LINEAGE_NODE_ADDED",
            node_id=node.node_id,
            node_type=node.node_type.value,
            fingerprint=node.fingerprint,
        )

        return node

    def create_node(
        self,
        *,
        node_id: str,
        node_type: LineageNodeType,
        label: Optional[str] = None,
        fingerprint: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> LineageNode:
        return self.add_node(
            LineageNode(
                node_id=node_id,
                node_type=node_type,
                label=label,
                fingerprint=fingerprint,
                metadata=dict(
                    metadata or {}
                ),
            )
        )

    def get_node(
        self,
        node_id: str,
    ) -> LineageNode:
        if (
            not isinstance(node_id, str)
            or not node_id.strip()
        ):
            raise LineageValidationError(
                "node_id must be a non-empty string"
            )

        node = self._nodes.get(
            node_id.strip()
        )

        if node is None:
            raise LineageNodeNotFoundError(
                f"Lineage node not found: "
                f"{node_id}"
            )

        return node

    def has_node(
        self,
        node_id: str,
    ) -> bool:
        try:
            self.get_node(node_id)
        except LineageNodeNotFoundError:
            return False

        return True

    def nodes(
        self,
    ) -> List[LineageNode]:
        return [
            self._nodes[node_id]
            for node_id
            in self._node_order
        ]

    def nodes_by_type(
        self,
        node_type: LineageNodeType,
    ) -> List[LineageNode]:
        if not isinstance(
            node_type,
            LineageNodeType,
        ):
            raise TypeError(
                "node_type must be LineageNodeType"
            )

        return [
            node
            for node in self.nodes()
            if node.node_type == node_type
        ]

    # ========================================================
    # Edges
    # ========================================================

    def _validate_edge_nodes(
        self,
        edge: LineageEdge,
    ) -> None:
        if not self.has_node(
            edge.source_id
        ):
            raise LineageNodeNotFoundError(
                f"Source node not found: "
                f"{edge.source_id}"
            )

        if not self.has_node(
            edge.target_id
        ):
            raise LineageNodeNotFoundError(
                f"Target node not found: "
                f"{edge.target_id}"
            )

    def _would_create_cycle(
        self,
        source_id: str,
        target_id: str,
    ) -> bool:
        if source_id == target_id:
            return True

        visited: Set[str] = set()
        queue = deque([target_id])

        while queue:
            current = queue.popleft()

            if current == source_id:
                return True

            if current in visited:
                continue

            visited.add(current)

            for edge_id in self._outgoing.get(
                current,
                [],
            ):
                edge = self._edges[
                    edge_id
                ]

                queue.append(
                    edge.target_id
                )

        return False

    def add_edge(
        self,
        edge: LineageEdge,
        *,
        allow_cycle: bool = False,
    ) -> LineageEdge:
        if not isinstance(
            edge,
            LineageEdge,
        ):
            raise TypeError(
                "edge must be LineageEdge"
            )

        self._validate_edge_nodes(
            edge
        )

        edge_id = edge.edge_id

        if edge_id is None:
            raise LineageEngineError(
                "edge_id generation failed"
            )

        if edge_id in self._edges:
            raise LineageEdgeAlreadyExistsError(
                f"Lineage edge already exists: "
                f"{edge_id}"
            )

        if (
            not allow_cycle
            and self._would_create_cycle(
                edge.source_id,
                edge.target_id,
            )
        ):
            raise LineageCycleError(
                "lineage edge would create "
                "a directed cycle"
            )

        self._edges[
            edge_id
        ] = edge

        self._edge_order.append(
            edge_id
        )

        self._outgoing[
            edge.source_id
        ].append(edge_id)

        self._incoming[
            edge.target_id
        ].append(edge_id)

        self._audit(
            "LINEAGE_EDGE_ADDED",
            edge_id=edge_id,
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation.value,
        )

        return edge

    def connect(
        self,
        source_id: str,
        target_id: str,
        relation: LineageRelation,
        *,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        allow_cycle: bool = False,
    ) -> LineageEdge:
        edge = LineageEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            metadata=dict(
                metadata or {}
            ),
        )

        return self.add_edge(
            edge,
            allow_cycle=allow_cycle,
        )

    def get_edge(
        self,
        edge_id: str,
    ) -> LineageEdge:
        if (
            not isinstance(edge_id, str)
            or not edge_id.strip()
        ):
            raise LineageValidationError(
                "edge_id must be a non-empty string"
            )

        edge = self._edges.get(
            edge_id.strip()
        )

        if edge is None:
            raise LineageEngineError(
                f"Lineage edge not found: "
                f"{edge_id}"
            )

        return edge

    def edges(
        self,
    ) -> List[LineageEdge]:
        return [
            self._edges[edge_id]
            for edge_id
            in self._edge_order
        ]

    def edges_by_relation(
        self,
        relation: LineageRelation,
    ) -> List[LineageEdge]:
        if not isinstance(
            relation,
            LineageRelation,
        ):
            raise TypeError(
                "relation must be LineageRelation"
            )

        return [
            edge
            for edge in self.edges()
            if edge.relation == relation
        ]

    # ========================================================
    # Direct graph navigation
    # ========================================================

    def outgoing_edges(
        self,
        node_id: str,
    ) -> List[LineageEdge]:
        self.get_node(node_id)

        return [
            self._edges[edge_id]
            for edge_id
            in self._outgoing.get(
                node_id,
                [],
            )
        ]

    def incoming_edges(
        self,
        node_id: str,
    ) -> List[LineageEdge]:
        self.get_node(node_id)

        return [
            self._edges[edge_id]
            for edge_id
            in self._incoming.get(
                node_id,
                [],
            )
        ]

    def children(
        self,
        node_id: str,
    ) -> List[LineageNode]:
        return [
            self._nodes[
                edge.target_id
            ]
            for edge in self.outgoing_edges(
                node_id
            )
        ]

    def parents(
        self,
        node_id: str,
    ) -> List[LineageNode]:
        return [
            self._nodes[
                edge.source_id
            ]
            for edge in self.incoming_edges(
                node_id
            )
        ]

    # ========================================================
    # Transitive graph navigation
    # ========================================================

    def descendants(
        self,
        node_id: str,
    ) -> List[LineageNode]:
        self.get_node(node_id)

        visited: Set[str] = set()
        queue = deque([node_id])
        result: List[LineageNode] = []

        while queue:
            current = queue.popleft()

            for edge in self.outgoing_edges(
                current
            ):
                child_id = edge.target_id

                if child_id in visited:
                    continue

                visited.add(child_id)

                result.append(
                    self._nodes[
                        child_id
                    ]
                )

                queue.append(
                    child_id
                )

        return result

    def ancestors(
        self,
        node_id: str,
    ) -> List[LineageNode]:
        self.get_node(node_id)

        visited: Set[str] = set()
        queue = deque([node_id])
        result: List[LineageNode] = []

        while queue:
            current = queue.popleft()

            for edge in self.incoming_edges(
                current
            ):
                parent_id = edge.source_id

                if parent_id in visited:
                    continue

                visited.add(parent_id)

                result.append(
                    self._nodes[
                        parent_id
                    ]
                )

                queue.append(
                    parent_id
                )

        return result

    # ========================================================
    # Path finding
    # ========================================================

    def find_path(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[LineagePath]:
        self.get_node(source_id)
        self.get_node(target_id)

        if source_id == target_id:
            return LineagePath(
                node_ids=[source_id],
                edge_ids=[],
                relations=[],
            )

        queue = deque(
            [
                (
                    source_id,
                    [source_id],
                    [],
                    [],
                )
            ]
        )

        visited: Set[str] = {
            source_id
        }

        while queue:
            (
                current,
                node_path,
                edge_path,
                relation_path,
            ) = queue.popleft()

            for edge in self.outgoing_edges(
                current
            ):
                next_id = edge.target_id

                if next_id in visited:
                    continue

                next_nodes = (
                    node_path + [next_id]
                )

                next_edges = (
                    edge_path
                    + [edge.edge_id]
                )

                next_relations = (
                    relation_path
                    + [edge.relation]
                )

                if next_id == target_id:
                    return LineagePath(
                        node_ids=next_nodes,
                        edge_ids=next_edges,
                        relations=next_relations,
                    )

                visited.add(next_id)

                queue.append(
                    (
                        next_id,
                        next_nodes,
                        next_edges,
                        next_relations,
                    )
                )

        return None

    # ========================================================
    # Provenance helpers
    # ========================================================

    def roots(
        self,
    ) -> List[LineageNode]:
        return [
            node
            for node in self.nodes()
            if len(
                self._incoming.get(
                    node.node_id,
                    [],
                )
            )
            == 0
        ]

    def leaves(
        self,
    ) -> List[LineageNode]:
        return [
            node
            for node in self.nodes()
            if len(
                self._outgoing.get(
                    node.node_id,
                    [],
                )
            )
            == 0
        ]

    def lineage_fingerprint(
        self,
        node_id: str,
    ) -> str:
        node = self.get_node(
            node_id
        )

        ancestors = sorted(
            ancestor.node_id
            for ancestor in self.ancestors(
                node_id
            )
        )

        incoming = sorted(
            (
                edge.source_id,
                edge.target_id,
                edge.relation.value,
            )
            for edge in self.incoming_edges(
                node_id
            )
        )

        return stable_hash(
            {
                "node_id": node.node_id,
                "node_type": (
                    node.node_type.value
                ),
                "fingerprint": (
                    node.fingerprint
                ),
                "ancestors": ancestors,
                "incoming": incoming,
            }
        )

    def full_graph_fingerprint(
        self,
    ) -> str:
        nodes = sorted(
            (
                node.node_id,
                node.node_type.value,
                node.fingerprint,
            )
            for node in self.nodes()
        )

        edges = sorted(
            (
                edge.source_id,
                edge.target_id,
                edge.relation.value,
            )
            for edge in self.edges()
        )

        return stable_hash(
            {
                "nodes": nodes,
                "edges": edges,
            }
        )

    # ========================================================
    # Reporting
    # ========================================================

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": (
                    item["created_at"]
                ),
                "details": dict(
                    item["details"]
                ),
            }
            for item in self._audit_events
        ]

    def snapshot(
        self,
    ) -> LineageSnapshot:
        node_type_counts = {
            node_type.value: 0
            for node_type
            in LineageNodeType
        }

        relation_counts = {
            relation.value: 0
            for relation
            in LineageRelation
        }

        for node in self.nodes():
            node_type_counts[
                node.node_type.value
            ] += 1

        for edge in self.edges():
            relation_counts[
                edge.relation.value
            ] += 1

        return LineageSnapshot(
            total_nodes=self.node_count(),
            total_edges=self.edge_count(),
            root_nodes=len(
                self.roots()
            ),
            leaf_nodes=len(
                self.leaves()
            ),
            node_type_counts=(
                node_type_counts
            ),
            relation_counts=(
                relation_counts
            ),
            metadata={
                "engine_version": self.VERSION,
                "graph_fingerprint": (
                    self.full_graph_fingerprint()
                ),
            },
        )

    def node_report(
        self,
        node_id: str,
    ) -> Dict[str, Any]:
        node = self.get_node(
            node_id
        )

        parents = self.parents(
            node_id
        )

        children = self.children(
            node_id
        )

        ancestors = self.ancestors(
            node_id
        )

        descendants = self.descendants(
            node_id
        )

        return {
            "node_id": node.node_id,
            "node_type": (
                node.node_type.value
            ),
            "label": node.label,
            "fingerprint": (
                node.fingerprint
            ),
            "parent_ids": [
                item.node_id
                for item in parents
            ],
            "child_ids": [
                item.node_id
                for item in children
            ],
            "ancestor_ids": [
                item.node_id
                for item in ancestors
            ],
            "descendant_ids": [
                item.node_id
                for item in descendants
            ],
            "lineage_fingerprint": (
                self.lineage_fingerprint(
                    node_id
                )
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def summary(
        self,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "component": "lineage_engine",
            "version": self.VERSION,
            "total_nodes": (
                snapshot.total_nodes
            ),
            "total_edges": (
                snapshot.total_edges
            ),
            "root_nodes": (
                snapshot.root_nodes
            ),
            "leaf_nodes": (
                snapshot.leaf_nodes
            ),
            "node_type_counts": dict(
                snapshot.node_type_counts
            ),
            "relation_counts": dict(
                snapshot.relation_counts
            ),
            "graph_fingerprint": (
                snapshot.metadata[
                    "graph_fingerprint"
                ]
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }