
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone

import pytest

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
    utc_now,
)


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def validity():
    return ValidityPeriod(valid_from=NOW)


@pytest.fixture
def evidence():
    return Evidence(
        evidence_id="evidence-001",
        evidence_type=EvidenceType.ECONOMIC_DATA,
        source="Economic data release",
        summary="Reported inflation increased.",
        observed_at=NOW,
        reference="dataset:inflation-001",
    )


@pytest.fixture
def relationship(validity):
    return KnowledgeRelationship(
        relationship_id="relationship-001",
        source_node_id="macro-event-001",
        target_node_id="mechanism-001",
        relationship_type=RelationshipType.TRIGGERS,
        confidence=0.75,
        validity=validity,
        evidence_ids=("evidence-001",),
        hypothesis_status=HypothesisStatus.PROPOSED,
        agent_id="macro-agent",
        rationale="Higher inflation may affect rate expectations.",
    )


def test_create_macro_event_node():
    node = KnowledgeNode(
        node_id="macro-event-001",
        node_type=NodeType.MACRO_EVENT,
        label="Inflation surprise",
        description="Inflation exceeded market expectations.",
        attributes={"country": "US"},
    )

    assert node.node_id == "macro-event-001"
    assert node.node_type == NodeType.MACRO_EVENT
    assert node.attributes["country"] == "US"


def test_all_six_node_types_are_available():
    assert len(NodeType) == 6

    assert {
        node_type.value for node_type in NodeType
    } == {
        "MACRO_EVENT",
        "ECONOMIC_MECHANISM",
        "MARKET_VARIABLE",
        "ASSET",
        "PORTFOLIO",
        "OUTCOME",
    }


def test_node_requires_nonempty_id():
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeNode(
            node_id=" ",
            node_type=NodeType.ASSET,
            label="SPY",
        )


def test_node_requires_valid_type():
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeNode(
            node_id="asset-001",
            node_type="ASSET",
            label="SPY",
        )


def test_node_rejects_invalid_attributes():
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeNode(
            node_id="asset-001",
            node_type=NodeType.ASSET,
            label="SPY",
            attributes=["invalid"],
        )


def test_node_attributes_are_copied():
    attributes = {"symbol": "SPY"}

    node = KnowledgeNode(
        node_id="asset-001",
        node_type=NodeType.ASSET,
        label="S&P 500 ETF",
        attributes=attributes,
    )

    attributes["symbol"] = "BTC"

    assert node.attributes["symbol"] == "SPY"


def test_node_fields_are_frozen():
    node = KnowledgeNode(
        node_id="asset-001",
        node_type=NodeType.ASSET,
        label="SPY",
    )

    with pytest.raises(FrozenInstanceError):
        node.label = "BTC"


def test_validity_contains_start(validity):
    assert validity.contains(NOW) is True


def test_validity_excludes_time_before_start(validity):
    assert validity.contains(
        NOW - timedelta(seconds=1)
    ) is False


def test_validity_excludes_end():
    period = ValidityPeriod(
        valid_from=NOW,
        valid_until=NOW + timedelta(days=1),
    )

    assert period.contains(NOW) is True
    assert period.contains(
        NOW + timedelta(hours=12)
    ) is True
    assert period.contains(
        NOW + timedelta(days=1)
    ) is False


def test_validity_rejects_end_before_start():
    with pytest.raises(KnowledgeGraphValidationError):
        ValidityPeriod(
            valid_from=NOW,
            valid_until=NOW - timedelta(seconds=1),
        )


def test_validity_rejects_equal_start_and_end():
    with pytest.raises(KnowledgeGraphValidationError):
        ValidityPeriod(
            valid_from=NOW,
            valid_until=NOW,
        )


def test_validity_rejects_naive_datetime():
    with pytest.raises(KnowledgeGraphValidationError):
        ValidityPeriod(
            valid_from=datetime(2026, 9, 20),
        )


def test_evidence_creation(evidence):
    assert evidence.evidence_id == "evidence-001"
    assert evidence.evidence_type == EvidenceType.ECONOMIC_DATA
    assert evidence.reference == "dataset:inflation-001"


def test_evidence_requires_source():
    with pytest.raises(KnowledgeGraphValidationError):
        Evidence(
            evidence_id="evidence-001",
            evidence_type=EvidenceType.NEWS,
            source=" ",
            summary="Market announcement.",
            observed_at=NOW,
        )


def test_evidence_requires_timezone_aware_observation():
    with pytest.raises(KnowledgeGraphValidationError):
        Evidence(
            evidence_id="evidence-001",
            evidence_type=EvidenceType.NEWS,
            source="News",
            summary="Market announcement.",
            observed_at=datetime(2026, 9, 20),
        )


def test_relationship_creation(relationship):
    assert relationship.confidence == pytest.approx(0.75)
    assert relationship.agent_id == "macro-agent"
    assert relationship.evidence_ids == ("evidence-001",)
    assert relationship.hypothesis_status == HypothesisStatus.PROPOSED


@pytest.mark.parametrize(
    "confidence",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_relationship_rejects_invalid_confidence(
    validity,
    confidence,
):
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeRelationship(
            relationship_id="relationship-001",
            source_node_id="macro-event-001",
            target_node_id="mechanism-001",
            relationship_type=RelationshipType.TRIGGERS,
            confidence=confidence,
            validity=validity,
        )


def test_relationship_accepts_confidence_boundaries(validity):
    for confidence in (0.0, 1.0):
        relationship = KnowledgeRelationship(
            relationship_id=f"relationship-{confidence}",
            source_node_id="macro-event-001",
            target_node_id="mechanism-001",
            relationship_type=RelationshipType.TRIGGERS,
            confidence=confidence,
            validity=validity,
        )

        assert relationship.confidence == confidence


def test_relationship_rejects_self_link(validity):
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeRelationship(
            relationship_id="relationship-001",
            source_node_id="asset-001",
            target_node_id="asset-001",
            relationship_type=RelationshipType.AFFECTS,
            confidence=0.5,
            validity=validity,
        )


def test_relationship_rejects_duplicate_evidence(validity):
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeRelationship(
            relationship_id="relationship-001",
            source_node_id="macro-event-001",
            target_node_id="mechanism-001",
            relationship_type=RelationshipType.TRIGGERS,
            confidence=0.5,
            validity=validity,
            evidence_ids=("evidence-001", "evidence-001"),
        )


def test_relationship_rejects_invalid_status(validity):
    with pytest.raises(KnowledgeGraphValidationError):
        KnowledgeRelationship(
            relationship_id="relationship-001",
            source_node_id="macro-event-001",
            target_node_id="mechanism-001",
            relationship_type=RelationshipType.TRIGGERS,
            confidence=0.5,
            validity=validity,
            hypothesis_status="SUPPORTED",
        )


def test_outcome_creation():
    outcome = ObservedOutcome(
        outcome_id="outcome-001",
        outcome_node_id="outcome-node-001",
        observed_at=NOW,
        value=-0.045,
        unit="portfolio_return",
        evidence_ids=("evidence-001",),
        related_relationship_ids=("relationship-001",),
    )

    assert outcome.value == pytest.approx(-0.045)
    assert outcome.unit == "portfolio_return"
    assert outcome.related_relationship_ids == (
        "relationship-001",
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), True],
)
def test_outcome_rejects_invalid_value(value):
    with pytest.raises(KnowledgeGraphValidationError):
        ObservedOutcome(
            outcome_id="outcome-001",
            outcome_node_id="outcome-node-001",
            observed_at=NOW,
            value=value,
            unit="portfolio_return",
        )


def test_outcome_rejects_duplicate_relationship_ids():
    with pytest.raises(KnowledgeGraphValidationError):
        ObservedOutcome(
            outcome_id="outcome-001",
            outcome_node_id="outcome-node-001",
            observed_at=NOW,
            value=0.02,
            unit="portfolio_return",
            related_relationship_ids=(
                "relationship-001",
                "relationship-001",
            ),
        )


def test_outcome_rejects_naive_datetime():
    with pytest.raises(KnowledgeGraphValidationError):
        ObservedOutcome(
            outcome_id="outcome-001",
            outcome_node_id="outcome-node-001",
            observed_at=datetime(2026, 9, 20),
            value=0.02,
            unit="portfolio_return",
        )


def test_utc_now_is_timezone_aware():
    current_time = utc_now()

    assert current_time.tzinfo is not None
    assert current_time.utcoffset() == timedelta(0)