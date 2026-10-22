"""
Tests for the v5.0.6 Audit History and Rollback Engine.

These tests validate the complete governance lifecycle:

Promotion Decision
    ->
Approval
    ->
Champion Switch
    ->
Audit History
    ->
Rollback Proposal
    ->
Rollback Execution
    ->
Previous Champion Restoration
"""

import pytest

from agent_tournament.audit_rollback import (
    AuditRollbackEngine,
    AuditRollbackError,
)
from agent_tournament.models import (
    DecisionType,
    PromotionDecision,
    generate_id,
)


def create_promotion_decision(
    *,
    approved: bool = True,
    champion_id: str = "champion",
    challenger_id: str = "challenger",
) -> PromotionDecision:
    """
    Create a valid promotion decision for governance tests.
    """

    return PromotionDecision(
        decision_id=generate_id("decision"),
        tournament_id="tournament_001",
        decision_type=DecisionType.PROMOTE_CHALLENGER,
        current_champion_id=champion_id,
        proposed_champion_id=challenger_id,
        reason="Challenger outperformed champion.",
        champion_score=0.70,
        challenger_score=0.85,
        approved=approved,
        metadata={
            "asset": "BTC",
            "regime": "high_volatility",
            "horizon": "7D",
        },
    )


def test_initial_champion_is_registered() -> None:
    """
    Engine must initialize with a known champion and audit event.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    assert engine.current_champion_id == "champion"

    assert engine.champion_history == [
        "champion"
    ]

    assert len(engine.audit_records) == 1

    assert (
        engine.audit_records[0].event_type
        == "champion_initialized"
    )


def test_empty_initial_champion_is_rejected() -> None:
    """
    Empty champion IDs must not be accepted.
    """

    with pytest.raises(AuditRollbackError):
        AuditRollbackEngine(
            initial_champion_id=""
        )


def test_governance_decision_can_be_registered() -> None:
    """
    Promotion decisions must be stored and audited.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    audit_record = engine.register_decision(
        decision
    )

    stored = engine.get_decision(
        decision.decision_id
    )

    assert stored.decision_id == decision.decision_id

    assert (
        audit_record.event_type
        == "governance_decision_registered"
    )

    assert len(engine.audit_records) == 2


def test_duplicate_decision_is_rejected() -> None:
    """
    Same governance decision must not be registered twice.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    engine.register_decision(
        decision
    )

    with pytest.raises(AuditRollbackError):
        engine.register_decision(
            decision
        )


def test_unapproved_promotion_cannot_execute() -> None:
    """
    Promotion must require explicit approval.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision(
        approved=False
    )

    engine.register_decision(
        decision
    )

    with pytest.raises(AuditRollbackError):
        engine.execute_promotion(
            decision.decision_id
        )

    assert engine.current_champion_id == "champion"


def test_approved_promotion_switches_champion() -> None:
    """
    Approved challenger must become the active champion.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision(
        approved=True
    )

    engine.register_decision(
        decision
    )

    new_champion = engine.execute_promotion(
        decision.decision_id
    )

    assert new_champion == "challenger"

    assert (
        engine.current_champion_id
        == "challenger"
    )

    assert engine.champion_history == [
        "champion",
        "challenger",
    ]

    promotion_events = engine.events_by_type(
        "champion_promoted"
    )

    assert len(promotion_events) == 1

    assert (
        promotion_events[0].previous_state[
            "champion_id"
        ]
        == "champion"
    )

    assert (
        promotion_events[0].new_state[
            "champion_id"
        ]
        == "challenger"
    )


def test_keep_champion_decision_cannot_execute_promotion() -> None:
    """
    KEEP_CHAMPION decisions must never trigger a promotion.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = PromotionDecision(
        decision_id=generate_id("decision"),
        tournament_id="tournament_keep",
        decision_type=DecisionType.KEEP_CHAMPION,
        current_champion_id="champion",
        proposed_champion_id="champion",
        reason="Challenger did not meet thresholds.",
        champion_score=0.85,
        challenger_score=0.70,
        approved=True,
    )

    engine.register_decision(
        decision
    )

    with pytest.raises(AuditRollbackError):
        engine.execute_promotion(
            decision.decision_id
        )

    assert engine.current_champion_id == "champion"


def test_rollback_proposal_is_created() -> None:
    """
    Executed promotion must support a rollback proposal.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    engine.register_decision(
        decision
    )

    engine.execute_promotion(
        decision.decision_id
    )

    rollback = engine.create_rollback(
        decision_id=decision.decision_id,
        reason="Post-promotion degradation detected.",
    )

    assert rollback.decision_id == decision.decision_id

    assert (
        rollback.previous_champion_id
        == "challenger"
    )

    assert (
        rollback.replacement_champion_id
        == "champion"
    )

    assert rollback.executed is False

    assert len(engine.rollback_records) == 1

    rollback_events = engine.events_by_type(
        "rollback_proposed"
    )

    assert len(rollback_events) == 1


def test_rollback_restores_previous_champion() -> None:
    """
    Rollback must restore the champion that existed before promotion.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    engine.register_decision(
        decision
    )

    engine.execute_promotion(
        decision.decision_id
    )

    assert (
        engine.current_champion_id
        == "challenger"
    )

    rollback = engine.create_rollback(
        decision_id=decision.decision_id,
        reason="Challenger degraded after deployment.",
    )

    executed = engine.execute_rollback(
        rollback.rollback_id
    )

    assert executed.executed is True

    assert executed.executed_at is not None

    assert (
        engine.current_champion_id
        == "champion"
    )

    assert engine.champion_history == [
        "champion",
        "challenger",
        "champion",
    ]

    rollback_events = engine.events_by_type(
        "rollback_executed"
    )

    assert len(rollback_events) == 1

    assert (
        rollback_events[0].previous_state[
            "champion_id"
        ]
        == "challenger"
    )

    assert (
        rollback_events[0].new_state[
            "champion_id"
        ]
        == "champion"
    )


def test_rollback_cannot_execute_twice() -> None:
    """
    One rollback record must be executable only once.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    engine.register_decision(
        decision
    )

    engine.execute_promotion(
        decision.decision_id
    )

    rollback = engine.create_rollback(
        decision_id=decision.decision_id,
        reason="Regression detected.",
    )

    engine.execute_rollback(
        rollback.rollback_id
    )

    with pytest.raises(AuditRollbackError):
        engine.execute_rollback(
            rollback.rollback_id
        )


def test_unknown_decision_is_rejected() -> None:
    """
    Unknown decision IDs must raise an explicit error.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    with pytest.raises(AuditRollbackError):
        engine.get_decision(
            "unknown_decision"
        )


def test_unknown_rollback_is_rejected() -> None:
    """
    Unknown rollback IDs must raise an explicit error.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    with pytest.raises(AuditRollbackError):
        engine.get_rollback(
            "unknown_rollback"
        )


def test_governance_snapshot_contains_complete_state() -> None:
    """
    Governance snapshot must summarize the current state.
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision()

    engine.register_decision(
        decision
    )

    engine.execute_promotion(
        decision.decision_id
    )

    rollback = engine.create_rollback(
        decision_id=decision.decision_id,
        reason="Controlled rollback test.",
    )

    engine.execute_rollback(
        rollback.rollback_id
    )

    snapshot = engine.governance_snapshot()

    assert (
        snapshot["current_champion_id"]
        == "champion"
    )

    assert snapshot["champion_history"] == [
        "champion",
        "challenger",
        "champion",
    ]

    assert snapshot["decision_count"] == 1

    assert snapshot["rollback_count"] == 1

    assert (
        snapshot["executed_rollback_count"]
        == 1
    )

    assert (
        snapshot["audit_record_count"]
        >= 5
    )


def test_complete_promotion_and_rollback_lifecycle() -> None:
    """
    End-to-end governance lifecycle:

    champion
        ->
    challenger promoted
        ->
    challenger active
        ->
    rollback proposed
        ->
    rollback executed
        ->
    original champion restored
    """

    engine = AuditRollbackEngine(
        initial_champion_id="champion"
    )

    decision = create_promotion_decision(
        approved=True
    )

    engine.register_decision(
        decision
    )

    promoted_id = engine.execute_promotion(
        decision.decision_id
    )

    assert promoted_id == "challenger"

    assert (
        engine.current_champion_id
        == "challenger"
    )

    rollback = engine.create_rollback(
        decision_id=decision.decision_id,
        reason=(
            "Live performance fell below "
            "deployment threshold."
        ),
    )

    assert rollback.executed is False

    engine.execute_rollback(
        rollback.rollback_id
    )

    assert rollback.executed is True

    assert (
        engine.current_champion_id
        == "champion"
    )

    event_types = [
        record.event_type
        for record in engine.audit_records
    ]

    assert "champion_initialized" in event_types

    assert (
        "governance_decision_registered"
        in event_types
    )

    assert "champion_promoted" in event_types

    assert "rollback_proposed" in event_types

    assert "rollback_executed" in event_types