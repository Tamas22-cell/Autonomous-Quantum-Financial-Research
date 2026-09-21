
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
from financial_knowledge_graph.outcome_evaluator import (
    ExpectedDirection,
    FinancialOutcomeEvaluator,
    OutcomeAssessment,
    OutcomeEvaluation,
)


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    store = FinancialKnowledgeGraphStore()

    store.add_node(
        KnowledgeNode(
            node_id="macro-001",
            node_type=NodeType.MACRO_EVENT,
            label="Inflation surprise",
        )
    )
    store.add_node(
        KnowledgeNode(
            node_id="outcome-node-001",
            node_type=NodeType.OUTCOME,
            label="Portfolio return",
        )
    )

    for evidence_id in ("e1", "e2", "e3"):
        store.add_evidence(
            Evidence(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.MARKET_DATA,
                source="market-data",
                summary=f"Evidence {evidence_id}",
                observed_at=NOW,
            )
        )

    store.add_relationship(
        KnowledgeRelationship(
            relationship_id="r1",
            source_node_id="macro-001",
            target_node_id="outcome-node-001",
            relationship_type=RelationshipType.INFLUENCES,
            confidence=0.7,
            validity=ValidityPeriod(valid_from=NOW),
            evidence_ids=("e1", "e2"),
            hypothesis_status=HypothesisStatus.PROPOSED,
            agent_id="macro-agent",
        )
    )

    return store


def add_outcome(
    graph,
    *,
    outcome_id="o1",
    value=-0.04,
    observed_at=None,
    related_relationship_ids=("r1",),
    evidence_ids=("e2", "e3"),
):
    outcome = ObservedOutcome(
        outcome_id=outcome_id,
        outcome_node_id="outcome-node-001",
        observed_at=observed_at or NOW + timedelta(days=1),
        value=value,
        unit="portfolio_return",
        evidence_ids=evidence_ids,
        related_relationship_ids=related_relationship_ids,
    )

    graph.add_outcome(outcome)
    return outcome


def test_evaluator_requires_graph():
    with pytest.raises(GraphStoreError):
        FinancialOutcomeEvaluator("not-a-graph")


def test_negative_prediction_aligned_with_negative_outcome(graph):
    add_outcome(graph, value=-0.04)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert isinstance(result, OutcomeEvaluation)
    assert result.assessment == OutcomeAssessment.ALIGNED
    assert result.observed_value == pytest.approx(-0.04)
    assert result.expected_direction == ExpectedDirection.NEGATIVE


def test_positive_prediction_aligned_with_positive_outcome(graph):
    add_outcome(graph, value=0.03)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.POSITIVE,
    )

    assert result.assessment == OutcomeAssessment.ALIGNED


def test_positive_prediction_contradicted_by_negative_outcome(graph):
    add_outcome(graph, value=-0.03)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.POSITIVE,
    )

    assert result.assessment == OutcomeAssessment.CONTRADICTED


def test_negative_prediction_contradicted_by_positive_outcome(graph):
    add_outcome(graph, value=0.03)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.assessment == OutcomeAssessment.CONTRADICTED


def test_neutral_prediction_aligned_with_zero_outcome(graph):
    add_outcome(graph, value=0.0)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEUTRAL,
    )

    assert result.assessment == OutcomeAssessment.ALIGNED


def test_non_neutral_prediction_inconclusive_for_zero_outcome(graph):
    add_outcome(graph, value=0.0)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.assessment == OutcomeAssessment.INCONCLUSIVE


def test_neutral_prediction_contradicted_by_nonzero_outcome(graph):
    add_outcome(graph, value=0.03)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEUTRAL,
    )

    assert result.assessment == OutcomeAssessment.CONTRADICTED


def test_tolerance_treats_small_value_as_neutral(graph):
    add_outcome(graph, value=0.002)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEUTRAL,
        neutral_tolerance=0.005,
    )

    assert result.assessment == OutcomeAssessment.ALIGNED


def test_tolerance_boundary_is_inclusive(graph):
    add_outcome(graph, value=-0.005)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEUTRAL,
        neutral_tolerance=0.005,
    )

    assert result.assessment == OutcomeAssessment.ALIGNED


def test_tolerance_does_not_hide_large_value(graph):
    add_outcome(graph, value=0.02)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
        neutral_tolerance=0.005,
    )

    assert result.assessment == OutcomeAssessment.CONTRADICTED


@pytest.mark.parametrize(
    "tolerance",
    [-0.1, float("nan"), float("inf"), True, "0.1"],
)
def test_invalid_tolerance_raises_error(graph, tolerance):
    add_outcome(graph)

    with pytest.raises(GraphStoreError):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
            neutral_tolerance=tolerance,
        )


def test_invalid_expected_direction_raises_error(graph):
    add_outcome(graph)

    with pytest.raises(GraphStoreError):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction="NEGATIVE",
        )


def test_unknown_relationship_raises_error(graph):
    add_outcome(graph)

    with pytest.raises(
        GraphStoreError,
        match="Unknown relationship ID",
    ):
        FinancialOutcomeEvaluator(graph).evaluate(
            "missing-relationship",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
        )


def test_unknown_outcome_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown outcome ID",
    ):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "missing-outcome",
            expected_direction=ExpectedDirection.NEGATIVE,
        )


def test_outcome_must_explicitly_reference_relationship(graph):
    add_outcome(
        graph,
        related_relationship_ids=(),
    )

    with pytest.raises(
        GraphStoreError,
        match="does not explicitly reference",
    ):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
        )


def test_outcome_cannot_precede_relationship_validity(graph):
    add_outcome(
        graph,
        observed_at=NOW - timedelta(days=1),
    )

    with pytest.raises(
        GraphStoreError,
        match="before the relationship validity period",
    ):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
        )


def test_evaluated_at_defaults_to_outcome_time(graph):
    outcome = add_outcome(graph)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.evaluated_at == outcome.observed_at


def test_evaluated_at_can_be_later_than_outcome(graph):
    outcome = add_outcome(graph)
    later = outcome.observed_at + timedelta(hours=2)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
        evaluated_at=later,
    )

    assert result.evaluated_at == later


def test_evaluated_at_cannot_precede_outcome(graph):
    outcome = add_outcome(graph)
    earlier = outcome.observed_at - timedelta(seconds=1)

    with pytest.raises(
        GraphStoreError,
        match="cannot be earlier",
    ):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
            evaluated_at=earlier,
        )


def test_evaluated_at_must_be_timezone_aware(graph):
    add_outcome(graph)

    with pytest.raises(KnowledgeGraphValidationError):
        FinancialOutcomeEvaluator(graph).evaluate(
            "r1",
            "o1",
            expected_direction=ExpectedDirection.NEGATIVE,
            evaluated_at=datetime(2026, 9, 22),
        )


def test_evidence_ids_are_combined_without_duplicates(graph):
    add_outcome(
        graph,
        evidence_ids=("e2", "e3"),
    )

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.evidence_ids == ("e1", "e2", "e3")


def test_evaluation_preserves_original_hypothesis_status(graph):
    add_outcome(graph, value=-0.04)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.hypothesis_status == HypothesisStatus.PROPOSED
    assert (
        graph.get_relationship("r1").hypothesis_status
        == HypothesisStatus.PROPOSED
    )


def test_evaluation_does_not_modify_graph(graph):
    add_outcome(graph)

    before = (
        graph.node_count,
        graph.evidence_count,
        graph.relationship_count,
        graph.outcome_count,
    )

    FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    after = (
        graph.node_count,
        graph.evidence_count,
        graph.relationship_count,
        graph.outcome_count,
    )

    assert after == before


def test_evaluation_preserves_relationship_and_outcome_ids(graph):
    add_outcome(graph)

    result = FinancialOutcomeEvaluator(graph).evaluate(
        "r1",
        "o1",
        expected_direction=ExpectedDirection.NEGATIVE,
    )

    assert result.relationship_id == "r1"
    assert result.outcome_id == "o1"