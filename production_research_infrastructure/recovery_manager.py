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


class RecoveryManagerError(RuntimeError):
    """Base exception for recovery-management failures."""


class RecoveryValidationError(RecoveryManagerError):
    """Raised when recovery input is invalid."""


class RecoveryNotFoundError(RecoveryManagerError):
    """Raised when a recovery record cannot be found."""


class CheckpointNotFoundError(RecoveryManagerError):
    """Raised when a checkpoint cannot be found."""


class RecoveryStateError(RecoveryManagerError):
    """Raised for invalid recovery state transitions."""


class RecoveryStatus(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    RECOVERING = "RECOVERING"
    RECOVERED = "RECOVERED"
    FAILED = "FAILED"
    EXHAUSTED = "EXHAUSTED"
    CANCELLED = "CANCELLED"


class FailureType(str, Enum):
    UNKNOWN = "UNKNOWN"
    EXCEPTION = "EXCEPTION"
    TIMEOUT = "TIMEOUT"
    DATA_ERROR = "DATA_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INFRASTRUCTURE_ERROR = "INFRASTRUCTURE_ERROR"


TERMINAL_RECOVERY_STATUSES = {
    RecoveryStatus.RECOVERED,
    RecoveryStatus.EXHAUSTED,
    RecoveryStatus.CANCELLED,
}


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 60.0
    retry_on: List[FailureType] = field(
        default_factory=lambda: list(FailureType)
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
        ):
            raise TypeError(
                "max_attempts must be an integer"
            )

        if self.max_attempts < 1:
            raise RecoveryValidationError(
                "max_attempts must be at least 1"
            )

        for name, value in (
            (
                "base_delay_seconds",
                self.base_delay_seconds,
            ),
            (
                "backoff_multiplier",
                self.backoff_multiplier,
            ),
            (
                "max_delay_seconds",
                self.max_delay_seconds,
            ),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(
                    value,
                    (int, float),
                )
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            if value < 0:
                raise RecoveryValidationError(
                    f"{name} cannot be negative"
                )

        if self.backoff_multiplier < 1:
            raise RecoveryValidationError(
                "backoff_multiplier must be at least 1"
            )

        if (
            self.max_delay_seconds
            < self.base_delay_seconds
        ):
            raise RecoveryValidationError(
                "max_delay_seconds cannot be smaller "
                "than base_delay_seconds"
            )

        if not isinstance(self.retry_on, list):
            raise TypeError(
                "retry_on must be a list"
            )

        for failure_type in self.retry_on:
            if not isinstance(
                failure_type,
                FailureType,
            ):
                raise TypeError(
                    "retry_on entries must be FailureType"
                )

        self.base_delay_seconds = float(
            self.base_delay_seconds
        )
        self.backoff_multiplier = float(
            self.backoff_multiplier
        )
        self.max_delay_seconds = float(
            self.max_delay_seconds
        )

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    def delay_for_attempt(
        self,
        attempt: int,
    ) -> float:
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
        ):
            raise TypeError(
                "attempt must be an integer"
            )

        if attempt < 1:
            raise RecoveryValidationError(
                "attempt must be at least 1"
            )

        delay = (
            self.base_delay_seconds
            * (
                self.backoff_multiplier
                ** (attempt - 1)
            )
        )

        return min(
            delay,
            self.max_delay_seconds,
        )

    def allows(
        self,
        failure_type: FailureType,
    ) -> bool:
        if not isinstance(
            failure_type,
            FailureType,
        ):
            raise TypeError(
                "failure_type must be FailureType"
            )

        return failure_type in self.retry_on


@dataclass
class RecoveryCheckpoint:
    run_id: str
    stage: str
    state: Any
    sequence: int
    checkpoint_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise RecoveryValidationError(
                "run_id must be a non-empty string"
            )

        self.run_id = self.run_id.strip()

        if (
            not isinstance(self.stage, str)
            or not self.stage.strip()
        ):
            raise RecoveryValidationError(
                "stage must be a non-empty string"
            )

        self.stage = self.stage.strip()

        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
        ):
            raise TypeError(
                "sequence must be an integer"
            )

        if self.sequence < 1:
            raise RecoveryValidationError(
                "sequence must be at least 1"
            )

        if self.checkpoint_id is None:
            fingerprint = stable_hash(
                {
                    "run_id": self.run_id,
                    "stage": self.stage,
                    "sequence": self.sequence,
                    "state": self.state,
                }
            )

            self.checkpoint_id = (
                f"checkpoint_{fingerprint[:32]}"
            )

        elif (
            not isinstance(self.checkpoint_id, str)
            or not self.checkpoint_id.strip()
        ):
            raise RecoveryValidationError(
                "checkpoint_id must be a non-empty string"
            )
        else:
            self.checkpoint_id = (
                self.checkpoint_id.strip()
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
    def state_fingerprint(self) -> str:
        return stable_hash(self.state)


@dataclass
class RecoveryAttempt:
    attempt: int
    started_at: datetime = field(
        default_factory=utc_now
    )
    completed_at: Optional[datetime] = None
    success: Optional[bool] = None
    error: Optional[str] = None
    checkpoint_id: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.attempt, bool)
            or not isinstance(self.attempt, int)
        ):
            raise TypeError(
                "attempt must be an integer"
            )

        if self.attempt < 1:
            raise RecoveryValidationError(
                "attempt must be at least 1"
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
class RecoveryRecord:
    run_id: str
    failure_type: FailureType
    error: str
    retry_policy: RetryPolicy
    status: RecoveryStatus = RecoveryStatus.CREATED
    recovery_id: Optional[str] = None
    checkpoint_id: Optional[str] = None
    attempts: List[RecoveryAttempt] = field(
        default_factory=list
    )
    created_at: datetime = field(
        default_factory=utc_now
    )
    updated_at: datetime = field(
        default_factory=utc_now
    )
    recovered_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or not self.run_id.strip()
        ):
            raise RecoveryValidationError(
                "run_id must be a non-empty string"
            )

        self.run_id = self.run_id.strip()

        if not isinstance(
            self.failure_type,
            FailureType,
        ):
            raise TypeError(
                "failure_type must be FailureType"
            )

        if (
            not isinstance(self.error, str)
            or not self.error.strip()
        ):
            raise RecoveryValidationError(
                "error must be a non-empty string"
            )

        self.error = self.error.strip()

        if not isinstance(
            self.retry_policy,
            RetryPolicy,
        ):
            raise TypeError(
                "retry_policy must be RetryPolicy"
            )

        if not isinstance(
            self.status,
            RecoveryStatus,
        ):
            raise TypeError(
                "status must be RecoveryStatus"
            )

        if self.recovery_id is None:
            fingerprint = stable_hash(
                {
                    "run_id": self.run_id,
                    "failure_type": (
                        self.failure_type.value
                    ),
                    "error": self.error,
                    "created_at": (
                        self.created_at.isoformat()
                    ),
                }
            )

            self.recovery_id = (
                f"recovery_{fingerprint[:32]}"
            )

        elif (
            not isinstance(self.recovery_id, str)
            or not self.recovery_id.strip()
        ):
            raise RecoveryValidationError(
                "recovery_id must be a non-empty string"
            )
        else:
            self.recovery_id = (
                self.recovery_id.strip()
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
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def terminal(self) -> bool:
        return (
            self.status
            in TERMINAL_RECOVERY_STATUSES
        )

    @property
    def can_retry(self) -> bool:
        return (
            self.retry_policy.allows(
                self.failure_type
            )
            and self.attempt_count
            < self.retry_policy.max_attempts
            and not self.terminal
        )


@dataclass
class RecoveryManagerSnapshot:
    total_recoveries: int
    ready_recoveries: int
    recovering: int
    recovered: int
    failed: int
    exhausted: int
    cancelled: int
    checkpoints: int
    total_attempts: int
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


class RecoveryManager:
    """
    Failure recovery infrastructure for research runs.

    Provides:
    - retry policies,
    - exponential backoff calculation,
    - failure classification,
    - deterministic checkpoints,
    - checkpoint restoration,
    - recovery attempts,
    - retry exhaustion,
    - recovery audit history.

    The manager coordinates research recovery metadata only.
    It does not execute broker orders or authorize live trading.
    """

    VERSION = "v6.0.1"

    def __init__(
        self,
        *,
        default_policy: Optional[
            RetryPolicy
        ] = None,
    ) -> None:
        if (
            default_policy is not None
            and not isinstance(
                default_policy,
                RetryPolicy,
            )
        ):
            raise TypeError(
                "default_policy must be RetryPolicy"
            )

        self.default_policy = (
            default_policy
            or RetryPolicy()
        )

        self._recoveries: Dict[
            str,
            RecoveryRecord,
        ] = {}

        self._recovery_order: List[str] = []

        self._checkpoints: Dict[
            str,
            RecoveryCheckpoint,
        ] = {}

        self._checkpoint_order: List[
            str
        ] = []

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
    # Checkpoints
    # ========================================================

    def create_checkpoint(
        self,
        run_id: str,
        stage: str,
        state: Any,
        *,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> RecoveryCheckpoint:
        existing = self.checkpoints_for_run(
            run_id
        )

        sequence = len(existing) + 1

        checkpoint = RecoveryCheckpoint(
            run_id=run_id,
            stage=stage,
            state=state,
            sequence=sequence,
            metadata=dict(metadata or {}),
        )

        checkpoint_id = (
            checkpoint.checkpoint_id
        )

        if checkpoint_id is None:
            raise RecoveryManagerError(
                "checkpoint_id generation failed"
            )

        self._checkpoints[
            checkpoint_id
        ] = checkpoint

        self._checkpoint_order.append(
            checkpoint_id
        )

        self._audit(
            "CHECKPOINT_CREATED",
            checkpoint_id=checkpoint_id,
            run_id=checkpoint.run_id,
            stage=checkpoint.stage,
            sequence=checkpoint.sequence,
            state_fingerprint=(
                checkpoint.state_fingerprint
            ),
        )

        return checkpoint

    def get_checkpoint(
        self,
        checkpoint_id: str,
    ) -> RecoveryCheckpoint:
        if (
            not isinstance(
                checkpoint_id,
                str,
            )
            or not checkpoint_id.strip()
        ):
            raise RecoveryValidationError(
                "checkpoint_id must be a "
                "non-empty string"
            )

        checkpoint = self._checkpoints.get(
            checkpoint_id.strip()
        )

        if checkpoint is None:
            raise CheckpointNotFoundError(
                f"Checkpoint not found: "
                f"{checkpoint_id}"
            )

        return checkpoint

    def checkpoints(
        self,
    ) -> List[RecoveryCheckpoint]:
        return [
            self._checkpoints[
                checkpoint_id
            ]
            for checkpoint_id
            in self._checkpoint_order
        ]

    def checkpoints_for_run(
        self,
        run_id: str,
    ) -> List[RecoveryCheckpoint]:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise RecoveryValidationError(
                "run_id must be a non-empty string"
            )

        run_id = run_id.strip()

        return [
            checkpoint
            for checkpoint
            in self.checkpoints()
            if checkpoint.run_id == run_id
        ]

    def latest_checkpoint(
        self,
        run_id: str,
    ) -> Optional[RecoveryCheckpoint]:
        checkpoints = (
            self.checkpoints_for_run(
                run_id
            )
        )

        if not checkpoints:
            return None

        return max(
            checkpoints,
            key=lambda item: item.sequence,
        )

    def restore_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Any:
        checkpoint = self.get_checkpoint(
            checkpoint_id
        )

        self._audit(
            "CHECKPOINT_RESTORED",
            checkpoint_id=(
                checkpoint.checkpoint_id
            ),
            run_id=checkpoint.run_id,
            stage=checkpoint.stage,
        )

        return checkpoint.state

    # ========================================================
    # Recovery creation
    # ========================================================

    def create_recovery(
        self,
        run_id: str,
        *,
        failure_type: FailureType,
        error: str,
        retry_policy: Optional[
            RetryPolicy
        ] = None,
        checkpoint_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> RecoveryRecord:
        policy = (
            retry_policy
            or self.default_policy
        )

        if not isinstance(
            policy,
            RetryPolicy,
        ):
            raise TypeError(
                "retry_policy must be RetryPolicy"
            )

        if checkpoint_id is not None:
            checkpoint = (
                self.get_checkpoint(
                    checkpoint_id
                )
            )

            if (
                checkpoint.run_id
                != run_id.strip()
            ):
                raise RecoveryValidationError(
                    "checkpoint does not belong "
                    "to the requested run"
                )

        recovery = RecoveryRecord(
            run_id=run_id,
            failure_type=failure_type,
            error=error,
            retry_policy=policy,
            checkpoint_id=checkpoint_id,
            metadata=dict(metadata or {}),
        )

        if policy.allows(failure_type):
            recovery.status = (
                RecoveryStatus.READY
            )
        else:
            recovery.status = (
                RecoveryStatus.EXHAUSTED
            )

        recovery_id = recovery.recovery_id

        if recovery_id is None:
            raise RecoveryManagerError(
                "recovery_id generation failed"
            )

        self._recoveries[
            recovery_id
        ] = recovery

        self._recovery_order.append(
            recovery_id
        )

        self._audit(
            "RECOVERY_CREATED",
            recovery_id=recovery_id,
            run_id=recovery.run_id,
            failure_type=(
                recovery.failure_type.value
            ),
            status=recovery.status.value,
            checkpoint_id=(
                recovery.checkpoint_id
            ),
        )

        return recovery

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        recovery_id: str,
    ) -> RecoveryRecord:
        if (
            not isinstance(recovery_id, str)
            or not recovery_id.strip()
        ):
            raise RecoveryValidationError(
                "recovery_id must be a "
                "non-empty string"
            )

        recovery = self._recoveries.get(
            recovery_id.strip()
        )

        if recovery is None:
            raise RecoveryNotFoundError(
                f"Recovery not found: "
                f"{recovery_id}"
            )

        return recovery

    def exists(
        self,
        recovery_id: str,
    ) -> bool:
        try:
            self.get(recovery_id)
        except RecoveryNotFoundError:
            return False

        return True

    def recoveries(
        self,
    ) -> List[RecoveryRecord]:
        return [
            self._recoveries[
                recovery_id
            ]
            for recovery_id
            in self._recovery_order
        ]

    def recoveries_for_run(
        self,
        run_id: str,
    ) -> List[RecoveryRecord]:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise RecoveryValidationError(
                "run_id must be a non-empty string"
            )

        run_id = run_id.strip()

        return [
            recovery
            for recovery
            in self.recoveries()
            if recovery.run_id == run_id
        ]

    # ========================================================
    # Recovery attempts
    # ========================================================

    def begin_attempt(
        self,
        recovery_id: str,
    ) -> RecoveryAttempt:
        recovery = self.get(
            recovery_id
        )

        if recovery.status not in {
            RecoveryStatus.READY,
            RecoveryStatus.FAILED,
        }:
            raise RecoveryStateError(
                f"Cannot begin attempt from "
                f"{recovery.status.value}"
            )

        if not recovery.can_retry:
            recovery.status = (
                RecoveryStatus.EXHAUSTED
            )
            recovery.updated_at = utc_now()

            raise RecoveryStateError(
                "Recovery attempts exhausted"
            )

        attempt_number = (
            recovery.attempt_count + 1
        )

        attempt = RecoveryAttempt(
            attempt=attempt_number,
            checkpoint_id=(
                recovery.checkpoint_id
            ),
        )

        recovery.attempts.append(
            attempt
        )

        recovery.status = (
            RecoveryStatus.RECOVERING
        )

        recovery.updated_at = utc_now()

        self._audit(
            "RECOVERY_ATTEMPT_STARTED",
            recovery_id=recovery_id,
            run_id=recovery.run_id,
            attempt=attempt_number,
            delay_seconds=(
                recovery.retry_policy
                .delay_for_attempt(
                    attempt_number
                )
            ),
            checkpoint_id=(
                attempt.checkpoint_id
            ),
        )

        return attempt

    def complete_attempt(
        self,
        recovery_id: str,
        *,
        result: Any = None,
    ) -> RecoveryRecord:
        recovery = self.get(
            recovery_id
        )

        if (
            recovery.status
            != RecoveryStatus.RECOVERING
            or not recovery.attempts
        ):
            raise RecoveryStateError(
                "No active recovery attempt"
            )

        attempt = recovery.attempts[-1]

        if attempt.success is not None:
            raise RecoveryStateError(
                "Recovery attempt already completed"
            )

        now = utc_now()

        attempt.success = True
        attempt.completed_at = now
        attempt.metadata[
            "result_fingerprint"
        ] = stable_hash(result)

        recovery.status = (
            RecoveryStatus.RECOVERED
        )

        recovery.recovered_at = now
        recovery.updated_at = now

        self._audit(
            "RECOVERY_SUCCEEDED",
            recovery_id=recovery_id,
            run_id=recovery.run_id,
            attempt=attempt.attempt,
            result_fingerprint=(
                attempt.metadata[
                    "result_fingerprint"
                ]
            ),
        )

        return recovery

    def fail_attempt(
        self,
        recovery_id: str,
        *,
        error: str,
    ) -> RecoveryRecord:
        if (
            not isinstance(error, str)
            or not error.strip()
        ):
            raise RecoveryValidationError(
                "error must be a non-empty string"
            )

        recovery = self.get(
            recovery_id
        )

        if (
            recovery.status
            != RecoveryStatus.RECOVERING
            or not recovery.attempts
        ):
            raise RecoveryStateError(
                "No active recovery attempt"
            )

        attempt = recovery.attempts[-1]

        if attempt.success is not None:
            raise RecoveryStateError(
                "Recovery attempt already completed"
            )

        attempt.success = False
        attempt.error = error.strip()
        attempt.completed_at = utc_now()

        recovery.error = error.strip()
        recovery.updated_at = utc_now()

        if (
            recovery.attempt_count
            >= recovery.retry_policy.max_attempts
        ):
            recovery.status = (
                RecoveryStatus.EXHAUSTED
            )
        else:
            recovery.status = (
                RecoveryStatus.FAILED
            )

        self._audit(
            "RECOVERY_ATTEMPT_FAILED",
            recovery_id=recovery_id,
            run_id=recovery.run_id,
            attempt=attempt.attempt,
            error=attempt.error,
            status=recovery.status.value,
        )

        return recovery

    def cancel(
        self,
        recovery_id: str,
        *,
        reason: Optional[str] = None,
    ) -> RecoveryRecord:
        recovery = self.get(
            recovery_id
        )

        if recovery.terminal:
            raise RecoveryStateError(
                "Terminal recovery cannot be cancelled"
            )

        recovery.status = (
            RecoveryStatus.CANCELLED
        )

        recovery.updated_at = utc_now()

        if reason is not None:
            recovery.metadata[
                "cancellation_reason"
            ] = reason

        self._audit(
            "RECOVERY_CANCELLED",
            recovery_id=recovery_id,
            run_id=recovery.run_id,
            reason=reason,
        )

        return recovery

    # ========================================================
    # Retry helpers
    # ========================================================

    def next_delay(
        self,
        recovery_id: str,
    ) -> Optional[float]:
        recovery = self.get(
            recovery_id
        )

        if not recovery.can_retry:
            return None

        next_attempt = (
            recovery.attempt_count + 1
        )

        return (
            recovery.retry_policy
            .delay_for_attempt(
                next_attempt
            )
        )

    # ========================================================
    # Reporting
    # ========================================================

    def count(self) -> int:
        return len(self._recoveries)

    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

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
    ) -> RecoveryManagerSnapshot:
        counts = {
            status: 0
            for status in RecoveryStatus
        }

        total_attempts = 0

        for recovery in self.recoveries():
            counts[
                recovery.status
            ] += 1

            total_attempts += (
                recovery.attempt_count
            )

        return RecoveryManagerSnapshot(
            total_recoveries=self.count(),
            ready_recoveries=counts[
                RecoveryStatus.READY
            ],
            recovering=counts[
                RecoveryStatus.RECOVERING
            ],
            recovered=counts[
                RecoveryStatus.RECOVERED
            ],
            failed=counts[
                RecoveryStatus.FAILED
            ],
            exhausted=counts[
                RecoveryStatus.EXHAUSTED
            ],
            cancelled=counts[
                RecoveryStatus.CANCELLED
            ],
            checkpoints=(
                self.checkpoint_count()
            ),
            total_attempts=total_attempts,
            metadata={
                "manager_version": self.VERSION,
            },
        )

    def recovery_report(
        self,
        recovery_id: str,
    ) -> Dict[str, Any]:
        recovery = self.get(
            recovery_id
        )

        return {
            "recovery_id": (
                recovery.recovery_id
            ),
            "run_id": recovery.run_id,
            "failure_type": (
                recovery.failure_type.value
            ),
            "status": recovery.status.value,
            "error": recovery.error,
            "checkpoint_id": (
                recovery.checkpoint_id
            ),
            "attempt_count": (
                recovery.attempt_count
            ),
            "max_attempts": (
                recovery.retry_policy
                .max_attempts
            ),
            "can_retry": (
                recovery.can_retry
            ),
            "next_delay_seconds": (
                self.next_delay(
                    recovery_id
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
            "component": "recovery_manager",
            "version": self.VERSION,
            "total_recoveries": (
                snapshot.total_recoveries
            ),
            "ready_recoveries": (
                snapshot.ready_recoveries
            ),
            "recovering": (
                snapshot.recovering
            ),
            "recovered": (
                snapshot.recovered
            ),
            "failed": snapshot.failed,
            "exhausted": (
                snapshot.exhausted
            ),
            "cancelled": (
                snapshot.cancelled
            ),
            "checkpoints": (
                snapshot.checkpoints
            ),
            "total_attempts": (
                snapshot.total_attempts
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }