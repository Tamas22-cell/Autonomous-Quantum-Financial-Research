
import json
from datetime import datetime, timedelta, timezone

import pytest

from financial_knowledge_graph.graph_persistence import (
    GraphPersistenceError,
    SCHEMA_VERSION,
    graph_from_dict,
    graph_to_dict,
    load_graph,
    save_graph,
)
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


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def empty_graph():
    return FinancialKnowledgeGraphStore()


@pytest.fixture
def populated_graph():
    graph = FinancialKnowledgeGraphStore()

    graph.add_node(
        KnowledgeNode(
            node_id="macro-001",
            node_type=NodeType.MACRO_EVENT,
            label="Inflation surprise",
            description="Unexpected inflation increase",
            attributes={"region": "US", "impact": 0.5},
        )
    )

    graph.add_node(
        KnowledgeNode(
            node_id="outcome-001",
            node_type=NodeType.OUTCOME,
            label="Portfolio return",
        )
    )

    graph.add_evidence(
        Evidence(
            evidence_id="evidence-001",
            evidence_type=EvidenceType.ECONOMIC_DATA,
            source="Economic dataset",
            summary="Inflation observation",
            observed_at=NOW,
            reference="dataset:inflation",
        )
    )

    graph.add_evidence(
        Evidence(
            evidence_id="evidence-002",
            evidence_type=EvidenceType.MARKET_DATA,
            source="Market dataset",
            summary="Portfolio return observation",
            observed_at=NOW + timedelta(days=1),
        )
    )

    graph.add_relationship(
        KnowledgeRelationship(
            relationship_id="relationship-001",
            source_node_id="macro-001",
            target_node_id="outcome-001",
            relationship_type=RelationshipType.INFLUENCES,
            confidence=0.75,
            validity=ValidityPeriod(
                valid_from=NOW,
                valid_until=NOW + timedelta(days=30),
            ),
            evidence_ids=("evidence-001",),
            hypothesis_status=HypothesisStatus.PROPOSED,
            agent_id="macro-agent",
            rationale="Inflation may affect portfolio returns.",
        )
    )

    graph.add_outcome(
        ObservedOutcome(
            outcome_id="observation-001",
            outcome_node_id="outcome-001",
            observed_at=NOW + timedelta(days=1),
            value=-0.04,
            unit="portfolio_return",
            evidence_ids=("evidence-002",),
            related_relationship_ids=("relationship-001",),
        )
    )

    return graph


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_empty_graph_to_dict(empty_graph):
    payload = graph_to_dict(empty_graph)

    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "nodes": [],
        "evidence": [],
        "relationships": [],
        "outcomes": [],
    }


def test_graph_to_dict_rejects_invalid_graph():
    with pytest.raises(GraphPersistenceError):
        graph_to_dict("not-a-graph")


def test_populated_graph_to_dict_is_json_serializable(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)

    encoded = json.dumps(payload)

    assert isinstance(encoded, str)
    assert len(payload["nodes"]) == 2
    assert len(payload["evidence"]) == 2
    assert len(payload["relationships"]) == 1
    assert len(payload["outcomes"]) == 1


def test_graph_to_dict_preserves_node_attributes(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)

    assert payload["nodes"][0]["attributes"] == {
        "region": "US",
        "impact": 0.5,
    }


def test_graph_to_dict_serializes_datetimes(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)

    assert (
        payload["evidence"][0]["observed_at"]
        == NOW.isoformat()
    )
    assert (
        payload["relationships"][0]["validity"]["valid_from"]
        == NOW.isoformat()
    )


def test_graph_to_dict_serializes_enums(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)

    assert payload["nodes"][0]["node_type"] == "MACRO_EVENT"
    assert (
        payload["relationships"][0]["relationship_type"]
        == "INFLUENCES"
    )


def test_empty_graph_round_trip(empty_graph):
    restored = graph_from_dict(graph_to_dict(empty_graph))

    assert isinstance(restored, FinancialKnowledgeGraphStore)
    assert restored.node_count == 0
    assert restored.evidence_count == 0
    assert restored.relationship_count == 0
    assert restored.outcome_count == 0


def test_populated_graph_round_trip(populated_graph):
    original_payload = graph_to_dict(populated_graph)
    restored = graph_from_dict(original_payload)

    assert graph_to_dict(restored) == original_payload


def test_restored_graph_is_independent(populated_graph):
    restored = graph_from_dict(graph_to_dict(populated_graph))

    restored.add_node(
        KnowledgeNode(
            node_id="new-node",
            node_type=NodeType.ASSET,
            label="New asset",
        )
    )

    assert restored.node_count == populated_graph.node_count + 1
    assert populated_graph.node_count == 2


def test_round_trip_preserves_relationship_fields(
    populated_graph,
):
    restored = graph_from_dict(graph_to_dict(populated_graph))
    relationship = restored.get_relationship("relationship-001")

    assert relationship.confidence == pytest.approx(0.75)
    assert relationship.agent_id == "macro-agent"
    assert relationship.hypothesis_status == HypothesisStatus.PROPOSED
    assert relationship.evidence_ids == ("evidence-001",)
    assert relationship.validity.valid_from == NOW
    assert relationship.validity.valid_until == (
        NOW + timedelta(days=30)
    )


def test_round_trip_preserves_outcome_fields(
    populated_graph,
):
    restored = graph_from_dict(graph_to_dict(populated_graph))
    outcome = restored.get_outcome("observation-001")

    assert outcome.value == pytest.approx(-0.04)
    assert outcome.evidence_ids == ("evidence-002",)
    assert outcome.related_relationship_ids == (
        "relationship-001",
    )


def test_save_and_load_empty_graph(tmp_path, empty_graph):
    destination = tmp_path / "empty_graph.json"

    saved_path = save_graph(empty_graph, destination)
    restored = load_graph(destination)

    assert saved_path == destination
    assert destination.exists()
    assert restored.node_count == 0


def test_save_and_load_populated_graph(
    tmp_path,
    populated_graph,
):
    destination = tmp_path / "knowledge_graph.json"

    save_graph(populated_graph, destination)
    restored = load_graph(destination)

    assert graph_to_dict(restored) == graph_to_dict(
        populated_graph
    )


def test_save_creates_parent_directories(
    tmp_path,
    empty_graph,
):
    destination = (
        tmp_path / "nested" / "research" / "graph.json"
    )

    save_graph(empty_graph, destination)

    assert destination.exists()


def test_saved_file_contains_valid_utf8_json(
    tmp_path,
    empty_graph,
):
    destination = tmp_path / "graph.json"

    save_graph(empty_graph, destination)

    payload = json.loads(
        destination.read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == SCHEMA_VERSION


def test_save_overwrites_existing_graph(
    tmp_path,
    empty_graph,
    populated_graph,
):
    destination = tmp_path / "graph.json"

    save_graph(empty_graph, destination)
    save_graph(populated_graph, destination)

    restored = load_graph(destination)

    assert restored.node_count == 2
    assert restored.relationship_count == 1


def test_save_leaves_no_temporary_file(
    tmp_path,
    empty_graph,
):
    destination = tmp_path / "graph.json"

    save_graph(empty_graph, destination)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "graph.json"
    ]


def test_load_missing_file_raises_error(tmp_path):
    with pytest.raises(
        GraphPersistenceError,
        match="Could not read graph",
    ):
        load_graph(tmp_path / "missing.json")


def test_load_invalid_json_raises_error(tmp_path):
    destination = tmp_path / "broken.json"
    destination.write_text(
        "{invalid-json",
        encoding="utf-8",
    )

    with pytest.raises(
        GraphPersistenceError,
        match="Could not read graph",
    ):
        load_graph(destination)


def test_load_rejects_non_object_json(tmp_path):
    destination = tmp_path / "array.json"
    destination.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(GraphPersistenceError):
        load_graph(destination)


@pytest.mark.parametrize(
    "version",
    [0, 2, "1", True, None],
)
def test_reject_unsupported_schema_version(
    empty_graph,
    version,
):
    payload = graph_to_dict(empty_graph)
    payload["schema_version"] = version

    with pytest.raises(
        GraphPersistenceError,
        match="Unsupported graph schema version",
    ):
        graph_from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["nodes", "evidence", "relationships", "outcomes"],
)
def test_reject_missing_sections(empty_graph, field):
    payload = graph_to_dict(empty_graph)
    del payload[field]

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["nodes", "evidence", "relationships", "outcomes"],
)
def test_reject_non_list_sections(empty_graph, field):
    payload = graph_to_dict(empty_graph)
    payload[field] = {}

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_invalid_node_type(populated_graph):
    payload = graph_to_dict(populated_graph)
    payload["nodes"][0]["node_type"] = "INVALID"

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_duplicate_node_id(populated_graph):
    payload = graph_to_dict(populated_graph)
    payload["nodes"].append(dict(payload["nodes"][0]))

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_missing_relationship_source(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["relationships"][0]["source_node_id"] = "missing"

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_missing_relationship_evidence(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["relationships"][0]["evidence_ids"] = [
        "missing-evidence"
    ]

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_missing_outcome_relationship(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["outcomes"][0]["related_relationship_ids"] = [
        "missing-relationship"
    ]

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_invalid_evidence_datetime(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["evidence"][0]["observed_at"] = "not-a-date"

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_timezone_naive_datetime(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["evidence"][0]["observed_at"] = (
        "2026-09-20T12:00:00"
    )

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_invalid_relationship_validity(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["relationships"][0]["validity"]["valid_until"] = (
        "2026-09-19T12:00:00+00:00"
    )

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_invalid_outcome_value(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["outcomes"][0]["value"] = float("nan")

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_non_string_relationship_evidence_id(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["relationships"][0]["evidence_ids"] = [123]

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)


def test_reject_non_string_outcome_relationship_id(
    populated_graph,
):
    payload = graph_to_dict(populated_graph)
    payload["outcomes"][0]["related_relationship_ids"] = [123]

    with pytest.raises(GraphPersistenceError):
        graph_from_dict(payload)