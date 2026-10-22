"""
Audit History and Rollback Engine for the v5.0.6 Agent Tournament System.

Responsibilities:

- Immutable-style audit history
- Promotion / governance decision logging
- Champion transition logging
- Rollback proposal creation
- Explicit rollback execution
- Previous champion restoration
- Complete traceability of governance events

The module does not directly modify AgentCompetitor objects.
Instead, it maintains a controlled champion-state registry and
produces AuditRecord / RollbackRecord objects defined in models.py.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    AuditRecord,
    DecisionType,
    PromotionDecision,
    RollbackRecord,
    generate_id,
)


class AuditRollbackError(Exception):
    """
    Raised when an audit or rollback operation is invalid.
    """


def utc_now() -> datetime:
    """
    Return timezone-aware UTC time.
    """

    return datetime.now(timezone.utc)


class AuditRollbackEngine:
    """
    Governance audit and rollback manager.

    The engine tracks:

        current champion
        champion history
        promotion decisions
        audit events
        rollback proposals
        executed rollbacks

    A rollback is always explicit. Creating a RollbackRecord does not
    automatically change the active champion.
    """

    def __init__(
        self,
        initial_champion_id: str,
    ) -> None:
        if not initial_champion_id:
            raise AuditRollbackError(
                "initial_champion_id cannot be empty."
            )

        self._current_champion_id = (
            initial_champion_id
        )

        self._champion_history: List[str] = [
            initial_champion_id
        ]

        self._audit_records: List[
            AuditRecord
        ] = []

        self._rollback_records: Dict[
            str,
            RollbackRecord
        ] = {}

        self._decisions: Dict[
            str,
            PromotionDecision
        ] = {}

        self._record_event(
            event_type="champion_initialized",
            entity_type="competitor",
            entity_id=initial_champion_id,
            description=(
                "Initial champion registered."
            ),
            previous_state=None,
            new_state={
                "champion_id": (
                    initial_champion_id
                )
            },
        )

    # --------------------------------------------------------------
    # Champion state
    # --------------------------------------------------------------

    @property
    def current_champion_id(
        self,
    ) -> str:
        """
        Return the currently active champion.
        """

        return self._current_champion_id

    @property
    def champion_history(
        self,
    ) -> List[str]:
        """
        Return champion transition history.
        """

        return list(
            self._champion_history
        )

    # --------------------------------------------------------------
    # Audit
    # --------------------------------------------------------------

    def _record_event(
        self,
        event_type: str,
        entity_type: str,
        entity_id: str,
        description: str,
        previous_state: Optional[
            Dict[str, Any]
        ] = None,
        new_state: Optional[
            Dict[str, Any]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> AuditRecord:
        """
        Create and store one audit event.
        """

        if not event_type:
            raise AuditRollbackError(
                "event_type cannot be empty."
            )

        if not entity_type:
            raise AuditRollbackError(
                "entity_type cannot be empty."
            )

        if not entity_id:
            raise AuditRollbackError(
                "entity_id cannot be empty."
            )

        record = AuditRecord(
            audit_id=generate_id(
                "audit"
            ),
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            previous_state=previous_state,
            new_state=new_state,
            metadata=metadata or {},
        )

        self._audit_records.append(
            record
        )

        return record

    @property
    def audit_records(
        self,
    ) -> List[AuditRecord]:
        """
        Return a copy of the complete audit history.
        """

        return list(
            self._audit_records
        )

    def get_audit_record(
        self,
        audit_id: str,
    ) -> AuditRecord:
        """
        Retrieve one audit event.
        """

        for record in self._audit_records:
            if record.audit_id == audit_id:
                return record

        raise AuditRollbackError(
            "Unknown audit record: "
            f"{audit_id}"
        )

    # --------------------------------------------------------------
    # Decision registration
    # --------------------------------------------------------------

    def register_decision(
        self,
        decision: PromotionDecision,
    ) -> AuditRecord:
        """
        Register a PromotionDecision in the audit system.
        """

        if not decision.decision_id:
            raise AuditRollbackError(
                "Decision must have a decision_id."
            )

        if (
            decision.decision_id
            in self._decisions
        ):
            raise AuditRollbackError(
                "Decision already registered: "
                f"{decision.decision_id}"
            )

        self._decisions[
            decision.decision_id
        ] = decision

        return self._record_event(
            event_type=(
                "governance_decision_registered"
            ),
            entity_type=(
                "promotion_decision"
            ),
            entity_id=(
                decision.decision_id
            ),
            description=(
                "Governance decision registered "
                "for audit."
            ),
            previous_state=None,
            new_state={
                "decision_type": (
                    decision.decision_type.value
                ),
                "current_champion_id": (
                    decision.current_champion_id
                ),
                "proposed_champion_id": (
                    decision.proposed_champion_id
                ),
                "approved": (
                    decision.approved
                ),
            },
            metadata={
                "tournament_id": (
                    decision.tournament_id
                ),
                "reason": (
                    decision.reason
                ),
            },
        )

    def get_decision(
        self,
        decision_id: str,
    ) -> PromotionDecision:
        """
        Retrieve a registered governance decision.
        """

        if decision_id not in self._decisions:
            raise AuditRollbackError(
                "Unknown governance decision: "
                f"{decision_id}"
            )

        return self._decisions[
            decision_id
        ]

    # --------------------------------------------------------------
    # Promotion execution
    # --------------------------------------------------------------

    def execute_promotion(
        self,
        decision_id: str,
    ) -> str:
        """
        Execute an approved PROMOTE_CHALLENGER decision.

        Returns the new champion ID.
        """

        decision = self.get_decision(
            decision_id
        )

        if (
            decision.decision_type
            != DecisionType.PROMOTE_CHALLENGER
        ):
            raise AuditRollbackError(
                "Only PROMOTE_CHALLENGER decisions "
                "can execute a promotion."
            )

        if not decision.approved:
            raise AuditRollbackError(
                "Promotion decision must be approved "
                "before execution."
            )

        if (
            decision.current_champion_id
            != self._current_champion_id
        ):
            raise AuditRollbackError(
                "Promotion decision champion does not "
                "match the currently active champion."
            )

        new_champion_id = (
            decision.proposed_champion_id
        )

        if not new_champion_id:
            raise AuditRollbackError(
                "Promotion decision does not contain "
                "a proposed champion."
            )

        if (
            new_champion_id
            == self._current_champion_id
        ):
            raise AuditRollbackError(
                "Proposed champion is already active."
            )

        previous_champion_id = (
            self._current_champion_id
        )

        self._current_champion_id = (
            new_champion_id
        )

        self._champion_history.append(
            new_champion_id
        )

        self._record_event(
            event_type="champion_promoted",
            entity_type="competitor",
            entity_id=new_champion_id,
            description=(
                "Approved challenger promoted "
                "to champion."
            ),
            previous_state={
                "champion_id": (
                    previous_champion_id
                )
            },
            new_state={
                "champion_id": (
                    new_champion_id
                )
            },
            metadata={
                "decision_id": (
                    decision.decision_id
                ),
                "tournament_id": (
                    decision.tournament_id
                ),
            },
        )

        return new_champion_id

    # --------------------------------------------------------------
    # Rollback proposal
    # --------------------------------------------------------------

    def create_rollback(
        self,
        decision_id: str,
        reason: str,
    ) -> RollbackRecord:
        """
        Create a rollback proposal for a previously
        executed promotion.

        Creating the proposal does not execute it.
        """

        if not reason:
            raise AuditRollbackError(
                "Rollback reason cannot be empty."
            )

        decision = self.get_decision(
            decision_id
        )

        if (
            decision.decision_type
            != DecisionType.PROMOTE_CHALLENGER
        ):
            raise AuditRollbackError(
                "Rollback requires a "
                "PROMOTE_CHALLENGER decision."
            )

        promoted_champion_id = (
            decision.proposed_champion_id
        )

        previous_champion_id = (
            decision.current_champion_id
        )

        if not promoted_champion_id:
            raise AuditRollbackError(
                "Promotion decision has no "
                "promoted champion."
            )

        if (
            self._current_champion_id
            != promoted_champion_id
        ):
            raise AuditRollbackError(
                "The promoted competitor is not "
                "the currently active champion."
            )

        rollback = RollbackRecord(
            rollback_id=generate_id(
                "rollback"
            ),
            decision_id=(
                decision.decision_id
            ),
            previous_champion_id=(
                promoted_champion_id
            ),
            replacement_champion_id=(
                previous_champion_id
            ),
            reason=reason,
            executed=False,
        )

        self._rollback_records[
            rollback.rollback_id
        ] = rollback

        self._record_event(
            event_type=(
                "rollback_proposed"
            ),
            entity_type=(
                "rollback"
            ),
            entity_id=(
                rollback.rollback_id
            ),
            description=(
                "Rollback proposal created."
            ),
            previous_state={
                "champion_id": (
                    promoted_champion_id
                )
            },
            new_state={
                "proposed_champion_id": (
                    previous_champion_id
                )
            },
            metadata={
                "decision_id": (
                    decision.decision_id
                ),
                "reason": reason,
            },
        )

        return rollback

    # --------------------------------------------------------------
    # Rollback execution
    # --------------------------------------------------------------

    def execute_rollback(
        self,
        rollback_id: str,
    ) -> RollbackRecord:
        """
        Execute an existing rollback proposal.
        """

        rollback = self.get_rollback(
            rollback_id
        )

        if rollback.executed:
            raise AuditRollbackError(
                "Rollback has already been executed."
            )

        if (
            self._current_champion_id
            != rollback.previous_champion_id
        ):
            raise AuditRollbackError(
                "Current champion does not match "
                "the rollback source champion."
            )

        previous_champion_id = (
            self._current_champion_id
        )

        replacement_champion_id = (
            rollback.replacement_champion_id
        )

        self._current_champion_id = (
            replacement_champion_id
        )

        self._champion_history.append(
            replacement_champion_id
        )

        rollback.executed = True
        rollback.executed_at = utc_now()

        self._record_event(
            event_type="rollback_executed",
            entity_type="rollback",
            entity_id=rollback.rollback_id,
            description=(
                "Champion rollback executed."
            ),
            previous_state={
                "champion_id": (
                    previous_champion_id
                )
            },
            new_state={
                "champion_id": (
                    replacement_champion_id
                )
            },
            metadata={
                "decision_id": (
                    rollback.decision_id
                ),
                "reason": (
                    rollback.reason
                ),
            },
        )

        return rollback

    # --------------------------------------------------------------
    # Rollback retrieval
    # --------------------------------------------------------------

    def get_rollback(
        self,
        rollback_id: str,
    ) -> RollbackRecord:
        """
        Retrieve one rollback record.
        """

        if (
            rollback_id
            not in self._rollback_records
        ):
            raise AuditRollbackError(
                "Unknown rollback record: "
                f"{rollback_id}"
            )

        return self._rollback_records[
            rollback_id
        ]

    @property
    def rollback_records(
        self,
    ) -> List[RollbackRecord]:
        """
        Return all rollback records.
        """

        return list(
            self._rollback_records.values()
        )

    # --------------------------------------------------------------
    # History queries
    # --------------------------------------------------------------

    def events_by_type(
        self,
        event_type: str,
    ) -> List[AuditRecord]:
        """
        Return audit records matching an event type.
        """

        return [
            record
            for record in self._audit_records
            if record.event_type
            == event_type
        ]

    def events_for_entity(
        self,
        entity_id: str,
    ) -> List[AuditRecord]:
        """
        Return audit records associated with an entity.
        """

        return [
            record
            for record in self._audit_records
            if record.entity_id
            == entity_id
        ]

    # --------------------------------------------------------------
    # Snapshot / export
    # --------------------------------------------------------------

    def governance_snapshot(
        self,
    ) -> Dict[str, Any]:
        """
        Return the current governance state.
        """

        return {
            "current_champion_id": (
                self._current_champion_id
            ),
            "champion_history": list(
                self._champion_history
            ),
            "decision_count": len(
                self._decisions
            ),
            "audit_record_count": len(
                self._audit_records
            ),
            "rollback_count": len(
                self._rollback_records
            ),
            "executed_rollback_count": sum(
                1
                for rollback
                in self._rollback_records.values()
                if rollback.executed
            ),
        }

    def export_audit_history(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Export audit history as serializable dictionaries.
        """

        return [
            asdict(record)
            for record in self._audit_records
        ]


__all__ = [
    "AuditRollbackEngine",
    "AuditRollbackError",
]
