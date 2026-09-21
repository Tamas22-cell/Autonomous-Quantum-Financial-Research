
from datetime import datetime, timedelta, timezone

import pytest

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
    GraphStoreError,
)
from financial_knowledge_graph.models import (
    Evidence,
    EvidenceType,
    HypothesisStatus,
    KnowledgeGraphValidationError,
    KnowledgeNode,
    KnowledgeRelationship,
    NodeType,
    ObservedOutcome,
    RelationshipType,
    ValidityPeriod,
)


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    return FinancialKnowledgeGraphStore()


@pytest.fixture
def macro_node():
    return KnowledgeNode(
        node_id="macro-001",
        node_type=NodeType.MACRO_EVENT,
        label="Inflation surprise",
    )


@pytest.fixture
def mechanism_node():
    return KnowledgeNode(
        node_id="mechanism-001",
        node_type=NodeType.ECONOMIC_MECHANISM,
        label="Interest rate expectations",
    )


@pytest.fixture
def outcome_node():
    return KnowledgeNode(
        node_id="outcome-node-001",
        node_type=NodeType.OUTCOME,
        label="Portfolio return",
    )


@pytest.fixture
def evidence():
    return Evidence(
        evidence_id="evidence-001",
        evidence_type=EvidenceType.ECONOMIC_DATA,
        source="Economic data release",
        summary="Reported inflation exceeded expectations.",
        observed_at=NOW,
    )


@pytest.fixture
def relationship():
    return KnowledgeRelationship(
        relationship_id="relationship-001",
        source_node_id="macro-001",
        target_node_id="mechanism-001",
        relationship_type=RelationshipType.TRIGGERS,
        confidence=0.75,
        validity=ValidityPeriod(valid_from=NOW),
        evidence_ids=("evidence-001",),
        hypothesis_status=HypothesisStatus.PROPOSED,
        agent_id="macro-agent",
    )


@pytest.fixture
def populated_graph(
    graph,
    macro_node,
    mechanism_node,
    evidence,
    relationship,
):
    graph.add_node(macro_node)
    graph.add_node(mechanism_node)
    graph.add_evidence(evidence)
    graph.add_relationship(relationship)
    return graph


def test_new_graph_is_empty(graph):
    assert graph.node_count == 0
    assert graph.evidence_count == 0
    assert graph.relationship_count == 0
    assert graph.outcome_count == 0


def test_add_and_get_node(graph, macro_node):
    graph.add_node(macro_node)

    assert graph.node_count == 1
    assert graph.get_node("macro-001") == macro_node


def test_reject_duplicate_node(graph, macro_node):
    graph.add_node(macro_node)

    with pytest.raises(GraphStoreError, match="Duplicate node ID"):
        graph.add_node(macro_node)

    assert graph.node_count == 1


def test_reject_invalid_node_type(graph):
    with pytest.raises(GraphStoreError):
        graph.add_node("not-a-node")


def test_unknown_node_raises_error(graph):
    with pytest.raises(GraphStoreError, match="Unknown node ID"):
        graph.get_node("missing-node")


def test_list_nodes(graph, macro_node, mechanism_node):
    graph.add_node(macro_node)
    graph.add_node(mechanism_node)

    assert graph.list_nodes() == (
        macro_node,
        mechanism_node,
    )


def test_list_nodes_filters_by_type(
    graph,
    macro_node,
    mechanism_node,
):
    graph.add_node(macro_node)
    graph.add_node(mechanism_node)

    assert graph.list_nodes(
        NodeType.MACRO_EVENT
    ) == (macro_node,)


def test_list_nodes_rejects_invalid_filter(graph):
    with pytest.raises(GraphStoreError):
        graph.list_nodes("MACRO_EVENT")


def test_add_and_get_evidence(graph, evidence):
    graph.add_evidence(evidence)

    assert graph.evidence_count == 1
    assert graph.get_evidence("evidence-001") == evidence
    assert graph.list_evidence() == (evidence,)


def test_reject_duplicate_evidence(graph, evidence):
    graph.add_evidence(evidence)

    with pytest.raises(
        GraphStoreError,
        match="Duplicate evidence ID",
    ):
        graph.add_evidence(evidence)

    assert graph.evidence_count == 1


def test_reject_invalid_evidence(graph):
    with pytest.raises(GraphStoreError):
        graph.add_evidence("not-evidence")


def test_unknown_evidence_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown evidence ID",
    ):
        graph.get_evidence("missing-evidence")


def test_add_and_get_relationship(
    populated_graph,
    relationship,
):
    assert populated_graph.relationship_count == 1

    assert populated_graph.get_relationship(
        "relationship-001"
    ) == relationship


def test_reject_duplicate_relationship(
    populated_graph,
    relationship,
):
    with pytest.raises(
        GraphStoreError,
        match="Duplicate relationship ID",
    ):
        populated_graph.add_relationship(relationship)

    assert populated_graph.relationship_count == 1


def test_reject_invalid_relationship(graph):
    with pytest.raises(GraphStoreError):
        graph.add_relationship("not-a-relationship")


def test_reject_relationship_with_missing_source(
    graph,
    mechanism_node,
    relationship,
    evidence,
):
    graph.add_node(mechanism_node)
    graph.add_evidence(evidence)

    with pytest.raises(
        GraphStoreError,
        match="Unknown node ID: macro-001",
    ):
        graph.add_relationship(relationship)

    assert graph.relationship_count == 0


def test_reject_relationship_with_missing_target(
    graph,
    macro_node,
    relationship,
    evidence,
):
    graph.add_node(macro_node)
    graph.add_evidence(evidence)

    with pytest.raises(
        GraphStoreError,
        match="Unknown node ID: mechanism-001",
    ):
        graph.add_relationship(relationship)

    assert graph.relationship_count == 0


def test_reject_relationship_with_missing_evidence(
    graph,
    macro_node,
    mechanism_node,
    relationship,
):
    graph.add_node(macro_node)
    graph.add_node(mechanism_node)

    with pytest.raises(
        GraphStoreError,
        match="Unknown evidence ID: evidence-001",
    ):
        graph.add_relationship(relationship)

    assert graph.relationship_count == 0


def test_unknown_relationship_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown relationship ID",
    ):
        graph.get_relationship("missing-relationship")


def test_list_relationships_by_source(
    populated_graph,
    relationship,
):
    assert populated_graph.list_relationships(
        source_node_id="macro-001"
    ) == (relationship,)

    assert populated_graph.list_relationships(
        source_node_id="other-node"
    ) == ()


def test_list_relationships_by_target(
    populated_graph,
    relationship,
):
    assert populated_graph.list_relationships(
        target_node_id="mechanism-001"
    ) == (relationship,)


def test_list_relationships_by_type(
    populated_graph,
    relationship,
):
    assert populated_graph.list_relationships(
        relationship_type=RelationshipType.TRIGGERS
    ) == (relationship,)

    assert populated_graph.list_relationships(
        relationship_type=RelationshipType.AFFECTS
    ) == ()


def test_list_relationships_rejects_invalid_type(
    populated_graph,
):
    with pytest.raises(GraphStoreError):
        populated_graph.list_relationships(
            relationship_type="TRIGGERS"
        )


def test_list_relationships_filters_by_validity(
    populated_graph,
    relationship,
):
    assert populated_graph.list_relationships(
        valid_at=NOW
    ) == (relationship,)

    assert populated_graph.list_relationships(
        valid_at=NOW - timedelta(seconds=1)
    ) == ()


def test_list_relationships_rejects_naive_datetime(
    populated_graph,
):
    with pytest.raises(KnowledgeGraphValidationError):
        populated_graph.list_relationships(
            valid_at=datetime(2026, 9, 20)
        )


def test_add_and_get_outcome(
    populated_graph,
    outcome_node,
):
    populated_graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=-0.045,
        unit="portfolio_return",
        evidence_ids=("evidence-001",),
        related_relationship_ids=("relationship-001",),
    )

    populated_graph.add_outcome(outcome)

    assert populated_graph.outcome_count == 1
    assert populated_graph.get_outcome(
        "outcome-001"
    ) == outcome
    assert populated_graph.list_outcomes() == (outcome,)


def test_reject_duplicate_outcome(
    graph,
    outcome_node,
):
    graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
    )

    graph.add_outcome(outcome)

    with pytest.raises(
        GraphStoreError,
        match="Duplicate outcome ID",
    ):
        graph.add_outcome(outcome)

    assert graph.outcome_count == 1


def test_reject_invalid_outcome(graph):
    with pytest.raises(GraphStoreError):
        graph.add_outcome("not-an-outcome")


def test_reject_outcome_with_missing_node(graph):
    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="missing-node",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
    )

    with pytest.raises(
        GraphStoreError,
        match="Unknown node ID",
    ):
        graph.add_outcome(outcome)


def test_reject_outcome_with_wrong_node_type(
    graph,
    macro_node,
):
    graph.add_node(macro_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="macro-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
    )

    with pytest.raises(
        GraphStoreError,
        match="OUTCOME node",
    ):
        graph.add_outcome(outcome)


def test_reject_outcome_with_missing_evidence(
    graph,
    outcome_node,
):
    graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(
        GraphStoreError,
        match="Unknown evidence ID",
    ):
        graph.add_outcome(outcome)


def test_reject_outcome_with_missing_relationship(
    graph,
    outcome_node,
):
    graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
        related_relationship_ids=("missing-relationship",),
    )

    with pytest.raises(
        GraphStoreError,
        match="Unknown relationship ID",
    ):
        graph.add_outcome(outcome)


def test_unknown_outcome_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown outcome ID",
    ):
        graph.get_outcome("missing-outcome")


def test_list_outcomes_filters_by_node(
    graph,
    outcome_node,
):
    graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
    )

    graph.add_outcome(outcome)

    assert graph.list_outcomes(
        outcome_node_id="outcome-node-001"
    ) == (outcome,)

    assert graph.list_outcomes(
        outcome_node_id="another-outcome-node"
    ) == ()


def test_failed_relationship_insert_does_not_change_count(
    graph,
    macro_node,
    mechanism_node,
    relationship,
):
    graph.add_node(macro_node)
    graph.add_node(mechanism_node)

    with pytest.raises(GraphStoreError):
        graph.add_relationship(relationship)

    assert graph.relationship_count == 0


def test_failed_outcome_insert_does_not_change_count(
    graph,
    outcome_node,
):
    graph.add_node(outcome_node)

    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=0.02,
        unit="portfolio_return",
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(GraphStoreError):
        graph.add_outcome(outcome)

    assert graph.outcome_count == 0