from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


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


class RunManagerError(RuntimeError):
    """Base exception for run-management failures."""


class RunValidationError(RunManagerError):
    """Raised when run data is invalid."""


class RunAlreadyExistsError(RunManagerError):
    """Raised when a run already exists."""


class RunNotFoundError(RunManagerError):
    """Raised when a requested run cannot be found."""


class RunStateTransitionError(RunManagerError):
    """Raised when an invalid run-state transition is requested."""


class ManagedRunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    RECOVERING = "RECOVERING"
    RECOVERED = "RECOVERED"
    CANCELLED = "CANCELLED"


TERMINAL_STATUSES = {
    ManagedRunStatus.COMPLETED,
    ManagedRunStatus.PARTIAL,
    ManagedRunStatus.FAILED,
    ManagedRunStatus.TIMED_OUT,
    ManagedRunStatus.RECOVERED,
    ManagedRunStatus.CANCELLED,
}


ALLOWED_TRANSITIONS = {
    ManagedRunStatus.CREATED: {
        ManagedRunStatus.RUNNING,
        ManagedRunStatus.CANCELLED,
        ManagedRunStatus.FAILED,
    },
    ManagedRunStatus.RUNNING: {
        ManagedRunStatus.COMPLETED,
        ManagedRunStatus.PARTIAL,
        ManagedRunStatus.FAILED,
        ManagedRunStatus.TIMED_OUT,
        ManagedRunStatus.CANCELLED,
    },
    ManagedRunStatus.FAILED: {
        ManagedRunStatus.RECOVERING,
    },
    ManagedRunStatus.TIMED_OUT: {
        ManagedRunStatus.RECOVERING,
    },
    ManagedRunStatus.PARTIAL: {
        ManagedRunStatus.RECOVERING,
    },
    ManagedRunStatus.RECOVERING: {
        ManagedRunStatus.RECOVERED,
        ManagedRunStatus.FAILED,
        ManagedRunStatus.TIMED_OUT,
    },
    ManagedRunStatus.COMPLETED: set(),
    ManagedRunStatus.RECOVERED: set(),
    ManagedRunStatus.CANCELLED: set(),
}


@dataclass
class RunIdentity:
    """
    Inputs that define a reproducible research run.

    The deterministic run ID is derived from these fields only.
    """

    research_name: str
    configuration_fingerprint: str
    dataset_fingerprints: List[str] = field(
        default_factory=list
    )
    parameters: Dict[str, Any] = field(
        default_factory=dict
    )
    seed: Optional[int] = None
    namespace: str = "research"
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.research_name, str)
            or not self.research_name.strip()
        ):
            raise RunValidationError(
                "research_name must be a non-empty string"
            )

        self.research_name = self.research_name.strip()

        if (
            not isinstance(
                self.configuration_fingerprint,
                str,
            )
            or not self.configuration_fingerprint.strip()
        ):
            raise RunValidationError(
                "configuration_fingerprint must be "
                "a non-empty string"
            )

        self.configuration_fingerprint = (
            self.configuration_fingerprint.strip()
        )

        if not isinstance(
            self.dataset_fingerprints,
            list,
        ):
            raise TypeError(
                "dataset_fingerprints must be a list"
            )

        for fingerprint in self.dataset_fingerprints:
            if (
                not isinstance(fingerprint, str)
                or not fingerprint.strip()
            ):
                raise RunValidationError(
                    "every dataset fingerprint must "
                    "be a non-empty string"
                )

        self.dataset_fingerprints = [
            fingerprint.strip()
            for fingerprint
            in self.dataset_fingerprints
        ]

        if not isinstance(self.parameters, dict):
            raise TypeError(
                "parameters must be a dictionary"
            )

        if self.seed is not None:
            if (
                isinstance(self.seed, bool)
                or not isinstance(self.seed, int)
            ):
                raise TypeError(
                    "seed must be an integer"
                )

        if (
            not isinstance(self.namespace, str)
            or not self.namespace.strip()
        ):
            raise RunValidationError(
                "namespace must be a non-empty string"
            )

        self.namespace = self.namespace.strip()

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "research_name": self.research_name,
            "configuration_fingerprint": (
                self.configuration_fingerprint
            ),
            "dataset_fingerprints": sorted(
                self.dataset_fingerprints
            ),
            "parameters": self.parameters,
            "seed": self.seed,
        }

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            self.canonical_payload()
        )

    @property
    def reproducible_run_id(self) -> str:
        return f"run_{self.fingerprint[:32]}"


@dataclass
class RunTransition:
    run_id: str
    previous_status: ManagedRunStatus
    new_status: ManagedRunStatus
    reason: Optional[str] = None
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


@dataclass
class ManagedRun:
    run_id: str
    identity: RunIdentity
    status: ManagedRunStatus = ManagedRunStatus.CREATED
    attempt: int = 1
    parent_run_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_fingerprint: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )
    transitions: List[RunTransition] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise RunValidationError(
                "run_id must be a non-empty string"
            )

        self.run_id = self.run_id.strip()

        if not isinstance(
            self.identity,
            RunIdentity,
        ):
            raise TypeError(
                "identity must be a RunIdentity"
            )

        if not isinstance(
            self.status,
            ManagedRunStatus,
        ):
            raise TypeError(
                "status must be a ManagedRunStatus"
            )

        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
        ):
            raise TypeError(
                "attempt must be an integer"
            )

        if self.attempt < 1:
            raise RunValidationError(
                "attempt must be at least 1"
            )

        if self.parent_run_id is not None:
            if (
                not isinstance(
                    self.parent_run_id,
                    str,
                )
                or not self.parent_run_id.strip()
            ):
                raise RunValidationError(
                    "parent_run_id must be a "
                    "non-empty string"
                )

            self.parent_run_id = (
                self.parent_run_id.strip()
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
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def duration_seconds(self) -> float:
        if self.started_at is None:
            return 0.0

        end = self.completed_at or utc_now()

        return max(
            0.0,
            (end - self.started_at).total_seconds(),
        )


@dataclass
class RunManagerSnapshot:
    total_runs: int
    status_counts: Dict[str, int]
    reproducible_runs: int
    recovery_runs: int
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


class RunManager:
    """
    Reproducible research-run manager.

    Provides:
    - deterministic run IDs,
    - configuration + dataset fingerprint binding,
    - lifecycle state management,
    - transition validation,
    - run attempts,
    - recovery lineage,
    - result fingerprints,
    - audit history,
    - reproducibility checks.

    This component manages research execution metadata only.
    It cannot authorize or execute real-money trades.
    """

    VERSION = "v6.0.1"

    def __init__(self) -> None:
        self._runs: Dict[str, ManagedRun] = {}
        self._run_order: List[str] = []
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
    # Run ID
    # ========================================================

    @staticmethod
    def generate_run_id(
        identity: RunIdentity,
    ) -> str:
        if not isinstance(
            identity,
            RunIdentity,
        ):
            raise TypeError(
                "identity must be a RunIdentity"
            )

        return identity.reproducible_run_id

    # ========================================================
    # Creation
    # ========================================================

    def create_run(
        self,
        identity: RunIdentity,
        *,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ManagedRun:
        if not isinstance(
            identity,
            RunIdentity,
        ):
            raise TypeError(
                "identity must be a RunIdentity"
            )

        run_id = self.generate_run_id(
            identity
        )

        if run_id in self._runs:
            raise RunAlreadyExistsError(
                f"Run already exists: {run_id}"
            )

        run = ManagedRun(
            run_id=run_id,
            identity=identity,
            metadata=dict(metadata or {}),
        )

        self._runs[run_id] = run
        self._run_order.append(run_id)

        self._audit(
            "RUN_CREATED",
            run_id=run.run_id,
            research_name=(
                identity.research_name
            ),
            identity_fingerprint=(
                identity.fingerprint
            ),
            attempt=run.attempt,
        )

        return run

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        run_id: str,
    ) -> ManagedRun:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise RunValidationError(
                "run_id must be a non-empty string"
            )

        run = self._runs.get(
            run_id.strip()
        )

        if run is None:
            raise RunNotFoundError(
                f"Run not found: {run_id}"
            )

        return run

    def exists(
        self,
        run_id: str,
    ) -> bool:
        try:
            self.get(run_id)
        except RunNotFoundError:
            return False

        return True

    def runs(self) -> List[ManagedRun]:
        return [
            self._runs[run_id]
            for run_id in self._run_order
        ]

    def runs_by_status(
        self,
        status: ManagedRunStatus,
    ) -> List[ManagedRun]:
        if not isinstance(
            status,
            ManagedRunStatus,
        ):
            raise TypeError(
                "status must be a ManagedRunStatus"
            )

        return [
            run
            for run in self.runs()
            if run.status == status
        ]

    # ========================================================
    # Lifecycle
    # ========================================================

    def transition(
        self,
        run_id: str,
        new_status: ManagedRunStatus,
        *,
        reason: Optional[str] = None,
    ) -> ManagedRun:
        if not isinstance(
            new_status,
            ManagedRunStatus,
        ):
            raise TypeError(
                "new_status must be a ManagedRunStatus"
            )

        run = self.get(run_id)

        previous_status = run.status

        allowed = ALLOWED_TRANSITIONS[
            previous_status
        ]

        if new_status not in allowed:
            raise RunStateTransitionError(
                f"Invalid run transition: "
                f"{previous_status.value} -> "
                f"{new_status.value}"
            )

        now = utc_now()

        if new_status == ManagedRunStatus.RUNNING:
            if run.started_at is None:
                run.started_at = now

        if new_status in TERMINAL_STATUSES:
            run.completed_at = now

        if (
            new_status
            == ManagedRunStatus.RECOVERING
        ):
            run.completed_at = None

        run.status = new_status

        transition = RunTransition(
            run_id=run.run_id,
            previous_status=previous_status,
            new_status=new_status,
            reason=reason,
        )

        run.transitions.append(
            transition
        )

        self._audit(
            "RUN_TRANSITION",
            run_id=run.run_id,
            previous_status=(
                previous_status.value
            ),
            new_status=new_status.value,
            reason=reason,
        )

        return run

    def start(
        self,
        run_id: str,
    ) -> ManagedRun:
        return self.transition(
            run_id,
            ManagedRunStatus.RUNNING,
        )

    def complete(
        self,
        run_id: str,
        *,
        result: Any = None,
    ) -> ManagedRun:
        run = self.get(run_id)

        run.result_fingerprint = stable_hash(
            result
        )

        return self.transition(
            run_id,
            ManagedRunStatus.COMPLETED,
        )

    def mark_partial(
        self,
        run_id: str,
        *,
        result: Any = None,
        reason: Optional[str] = None,
    ) -> ManagedRun:
        run = self.get(run_id)

        run.result_fingerprint = stable_hash(
            result
        )

        return self.transition(
            run_id,
            ManagedRunStatus.PARTIAL,
            reason=reason,
        )

    def fail(
        self,
        run_id: str,
        *,
        error: str,
    ) -> ManagedRun:
        if (
            not isinstance(error, str)
            or not error.strip()
        ):
            raise RunValidationError(
                "error must be a non-empty string"
            )

        run = self.get(run_id)

        run.error = error.strip()

        return self.transition(
            run_id,
            ManagedRunStatus.FAILED,
            reason=run.error,
        )

    def timeout(
        self,
        run_id: str,
        *,
        reason: Optional[str] = None,
    ) -> ManagedRun:
        return self.transition(
            run_id,
            ManagedRunStatus.TIMED_OUT,
            reason=reason,
        )

    def cancel(
        self,
        run_id: str,
        *,
        reason: Optional[str] = None,
    ) -> ManagedRun:
        return self.transition(
            run_id,
            ManagedRunStatus.CANCELLED,
            reason=reason,
        )

    # ========================================================
    # Recovery
    # ========================================================

    def begin_recovery(
        self,
        run_id: str,
    ) -> ManagedRun:
        return self.transition(
            run_id,
            ManagedRunStatus.RECOVERING,
            reason="Recovery started",
        )

    def mark_recovered(
        self,
        run_id: str,
        *,
        result: Any = None,
    ) -> ManagedRun:
        run = self.get(run_id)

        run.result_fingerprint = stable_hash(
            result
        )

        return self.transition(
            run_id,
            ManagedRunStatus.RECOVERED,
            reason="Recovery completed",
        )

    def create_recovery_run(
        self,
        run_id: str,
        *,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ManagedRun:
        parent = self.get(run_id)

        if parent.status not in {
            ManagedRunStatus.FAILED,
            ManagedRunStatus.TIMED_OUT,
            ManagedRunStatus.PARTIAL,
        }:
            raise RunStateTransitionError(
                "Recovery run can only be created "
                "from FAILED, TIMED_OUT or PARTIAL runs"
            )

        attempt = parent.attempt + 1

        recovery_id = (
            f"{parent.run_id}_retry_{attempt}"
        )

        if recovery_id in self._runs:
            raise RunAlreadyExistsError(
                f"Recovery run already exists: "
                f"{recovery_id}"
            )

        recovery = ManagedRun(
            run_id=recovery_id,
            identity=parent.identity,
            status=ManagedRunStatus.CREATED,
            attempt=attempt,
            parent_run_id=parent.run_id,
            metadata=dict(metadata or {}),
        )

        self._runs[
            recovery.run_id
        ] = recovery

        self._run_order.append(
            recovery.run_id
        )

        self._audit(
            "RECOVERY_RUN_CREATED",
            run_id=recovery.run_id,
            parent_run_id=parent.run_id,
            attempt=recovery.attempt,
        )

        return recovery

    def parent_of(
        self,
        run_id: str,
    ) -> Optional[ManagedRun]:
        run = self.get(run_id)

        if run.parent_run_id is None:
            return None

        return self.get(
            run.parent_run_id
        )

    def recovery_runs(
        self,
        run_id: str,
    ) -> List[ManagedRun]:
        self.get(run_id)

        return [
            run
            for run in self.runs()
            if run.parent_run_id == run_id
        ]

    # ========================================================
    # Reproducibility
    # ========================================================

    def is_reproducible(
        self,
        run_id: str,
        identity: RunIdentity,
    ) -> bool:
        if not isinstance(
            identity,
            RunIdentity,
        ):
            raise TypeError(
                "identity must be a RunIdentity"
            )

        run = self.get(run_id)

        return (
            run.identity.fingerprint
            == identity.fingerprint
        )

    def verify_result(
        self,
        run_id: str,
        result: Any,
    ) -> bool:
        run = self.get(run_id)

        if run.result_fingerprint is None:
            return False

        return (
            run.result_fingerprint
            == stable_hash(result)
        )

    # ========================================================
    # Reporting
    # ========================================================

    def count(self) -> int:
        return len(self._runs)

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item[
                    "created_at"
                ],
                "details": dict(
                    item["details"]
                ),
            }
            for item in self._audit_events
        ]

    def snapshot(
        self,
    ) -> RunManagerSnapshot:
        status_counts = {
            status.value: 0
            for status in ManagedRunStatus
        }

        for run in self.runs():
            status_counts[
                run.status.value
            ] += 1

        recovery_count = sum(
            1
            for run in self.runs()
            if run.parent_run_id is not None
        )

        return RunManagerSnapshot(
            total_runs=self.count(),
            status_counts=status_counts,
            reproducible_runs=self.count(),
            recovery_runs=recovery_count,
            metadata={
                "manager_version": self.VERSION,
            },
        )

    def run_report(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        run = self.get(run_id)

        return {
            "run_id": run.run_id,
            "research_name": (
                run.identity.research_name
            ),
            "status": run.status.value,
            "attempt": run.attempt,
            "parent_run_id": (
                run.parent_run_id
            ),
            "identity_fingerprint": (
                run.identity.fingerprint
            ),
            "configuration_fingerprint": (
                run.identity
                .configuration_fingerprint
            ),
            "dataset_fingerprints": list(
                run.identity
                .dataset_fingerprints
            ),
            "result_fingerprint": (
                run.result_fingerprint
            ),
            "error": run.error,
            "duration_seconds": (
                run.duration_seconds
            ),
            "transition_count": len(
                run.transitions
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def summary(
        self,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "component": "run_manager",
            "version": self.VERSION,
            "total_runs": (
                snapshot.total_runs
            ),
            "status_counts": dict(
                snapshot.status_counts
            ),
            "reproducible_runs": (
                snapshot.reproducible_runs
            ),
            "recovery_runs": (
                snapshot.recovery_runs
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }