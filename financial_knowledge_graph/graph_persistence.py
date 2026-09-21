
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
)
from financial_knowledge_graph.models import (
    Evidence,
    EvidenceType,
    HypothesisStatus,
    KnowledgeNode,
    KnowledgeRelationship,
    NodeType,
    ObservedOutcome,
    RelationshipType,
    ValidityPeriod,
)


SCHEMA_VERSION = 1


class GraphPersistenceError(ValueError):
    """Invalid graph file or unsuccessful persistence operation."""


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, tuple):
        return [_encode(item) for item in value]

    if isinstance(value, list):
        return [_encode(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _encode(item)
            for key, item in value.items()
        }

    if isinstance(value, Path):
        return str(value)

    return value


def _parse_datetime(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise GraphPersistenceError(
            f"{field_name} must be an ISO datetime string."
        )

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise GraphPersistenceError(
            f"Invalid datetime in {field_name}."
        ) from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GraphPersistenceError(
            f"{field_name} must include a timezone."
        )

    return parsed


def _parse_optional_datetime(
    value: Any,
    field_name: str,
) -> datetime | None:
    if value is None:
        return None

    return _parse_datetime(value, field_name)


def _require_mapping(
    value: Any,
    field_name: str,
) -> dict:
    if not isinstance(value, dict):
        raise GraphPersistenceError(
            f"{field_name} must be an object."
        )

    return value


def _require_list(
    value: Any,
    field_name: str,
) -> list:
    if not isinstance(value, list):
        raise GraphPersistenceError(
            f"{field_name} must be an array."
        )

    return value


def _require_string_tuple(
    value: Any,
    field_name: str,
) -> tuple[str, ...]:
    items = _require_list(value, field_name)

    if not all(isinstance(item, str) for item in items):
        raise GraphPersistenceError(
            f"{field_name} must contain strings."
        )

    return tuple(items)


def graph_to_dict(
    graph: FinancialKnowledgeGraphStore,
) -> dict[str, Any]:
    """Convert a graph into a JSON-compatible dictionary."""

    if not isinstance(graph, FinancialKnowledgeGraphStore):
        raise GraphPersistenceError(
            "graph must be a FinancialKnowledgeGraphStore."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            _encode(asdict(node))
            for node in graph.list_nodes()
        ],
        "evidence": [
            _encode(asdict(item))
            for item in graph.list_evidence()
        ],
        "relationships": [
            _encode(asdict(relationship))
            for relationship in graph.list_relationships()
        ],
        "outcomes": [
            _encode(asdict(outcome))
            for outcome in graph.list_outcomes()
        ],
    }


def graph_from_dict(
    payload: dict[str, Any],
) -> FinancialKnowledgeGraphStore:
    """
    Rebuild a graph using the existing model and store validation.

    Load order preserves references:
    nodes -> evidence -> relationships -> outcomes.
    """

    data = _require_mapping(payload, "payload")

    if (
        type(data.get("schema_version")) is not int
        or data["schema_version"] != SCHEMA_VERSION
    ):
        raise GraphPersistenceError(
            "Unsupported graph schema version."
        )

    nodes = _require_list(data.get("nodes"), "nodes")
    evidence_items = _require_list(
        data.get("evidence"),
        "evidence",
    )
    relationships = _require_list(
        data.get("relationships"),
        "relationships",
    )
    outcomes = _require_list(
        data.get("outcomes"),
        "outcomes",
    )

    graph = FinancialKnowledgeGraphStore()

    try:
        for raw in nodes:
            item = _require_mapping(raw, "node")

            graph.add_node(
                KnowledgeNode(
                    node_id=item["node_id"],
                    node_type=NodeType(item["node_type"]),
                    label=item["label"],
                    description=item.get("description", ""),
                    attributes=item.get("attributes", {}),
                )
            )

        for raw in evidence_items:
            item = _require_mapping(raw, "evidence item")

            graph.add_evidence(
                Evidence(
                    evidence_id=item["evidence_id"],
                    evidence_type=EvidenceType(
                        item["evidence_type"]
                    ),
                    source=item["source"],
                    summary=item["summary"],
                    observed_at=_parse_datetime(
                        item["observed_at"],
                        "evidence.observed_at",
                    ),
                    reference=item.get("reference"),
                )
            )

        for raw in relationships:
            item = _require_mapping(raw, "relationship")
            validity = _require_mapping(
                item["validity"],
                "relationship.validity",
            )

            graph.add_relationship(
                KnowledgeRelationship(
                    relationship_id=item["relationship_id"],
                    source_node_id=item["source_node_id"],
                    target_node_id=item["target_node_id"],
                    relationship_type=RelationshipType(
                        item["relationship_type"]
                    ),
                    confidence=item["confidence"],
                    validity=ValidityPeriod(
                        valid_from=_parse_datetime(
                            validity["valid_from"],
                            "validity.valid_from",
                        ),
                        valid_until=_parse_optional_datetime(
                            validity.get("valid_until"),
                            "validity.valid_until",
                        ),
                    ),
                    evidence_ids=_require_string_tuple(
                        item.get("evidence_ids", []),
                        "relationship.evidence_ids",
                    ),
                    hypothesis_status=HypothesisStatus(
                        item.get(
                            "hypothesis_status",
                            HypothesisStatus.PROPOSED.value,
                        )
                    ),
                    agent_id=item.get("agent_id"),
                    rationale=item.get("rationale", ""),
                )
            )

        for raw in outcomes:
            item = _require_mapping(raw, "outcome")

            graph.add_outcome(
                ObservedOutcome(
                    outcome_id=item["outcome_id"],
                    outcome_node_id=item["outcome_node_id"],
                    observed_at=_parse_datetime(
                        item["observed_at"],
                        "outcome.observed_at",
                    ),
                    value=item["value"],
                    unit=item["unit"],
                    evidence_ids=_require_string_tuple(
                        item.get("evidence_ids", []),
                        "outcome.evidence_ids",
                    ),
                    related_relationship_ids=(
                        _require_string_tuple(
                            item.get(
                                "related_relationship_ids",
                                [],
                            ),
                            "outcome.related_relationship_ids",
                        )
                    ),
                )
            )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise GraphPersistenceError(
            f"Invalid graph payload: {exc}"
        ) from exc

    return graph


def save_graph(
    graph: FinancialKnowledgeGraphStore,
    file_path: str | Path,
) -> Path:
    """
    Atomically save a graph as UTF-8 JSON.

    A temporary file is written in the destination directory,
    then replaces the destination after successful serialization.
    """

    payload = graph_to_dict(graph)
    destination = Path(file_path)
    temporary_path: Path | None = None

    try:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.write("\n")

        os.replace(temporary_path, destination)
        temporary_path = None

    except (OSError, TypeError, ValueError) as exc:
        raise GraphPersistenceError(
            f"Could not save graph: {exc}"
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return destination


def load_graph(
    file_path: str | Path,
) -> FinancialKnowledgeGraphStore:
    """Read a saved JSON graph and validate its contents."""

    source = Path(file_path)

    try:
        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GraphPersistenceError(
            f"Could not read graph: {exc}"
        ) from exc

    return graph_from_dict(payload)


__all__ = [
    "GraphPersistenceError",
    "SCHEMA_VERSION",
    "graph_from_dict",
    "graph_to_dict",
    "load_graph",
    "save_graph",
]