from __future__ import annotations

from datetime import timedelta

import pytest

from production_research_infrastructure.recovery_manager import (
    CheckpointNotFoundError,
    FailureType,
    RecoveryCheckpoint,
    RecoveryManager,
    RecoveryNotFoundError,
    RecoveryStateError,
    RecoveryStatus,
    RecoveryValidationError,
    RetryPolicy,
    utc_now,
)


def make_policy(
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    max_delay_seconds: float = 60.0,
    retry_on=None,
) -> RetryPolicy:
    if retry_on is None:
        retry_on = list(FailureType)

    return RetryPolicy(
        max_attempts=max_attempts,
        base_delay_seconds=base_delay_seconds,
        backoff_multiplier=backoff_multiplier,
        max_delay_seconds=max_delay_seconds,
        retry_on=retry_on,
    )


def make_recovery(
    manager: RecoveryManager,
    *,
    run_id: str = "run_test_001",
    failure_type: FailureType = FailureType.EXCEPTION,
    error: str = "Research failure",
    retry_policy: RetryPolicy | None = None,
    checkpoint_id: str | None = None,
):
    return manager.create_recovery(
        run_id,
        failure_type=failure_type,
        error=error,
        retry_policy=retry_policy,
        checkpoint_id=checkpoint_id,
    )


# ============================================================
# RetryPolicy
# ============================================================


def test_retry_policy_defaults() -> None:
    policy = RetryPolicy()

    assert policy.max_attempts == 3
    assert policy.base_delay_seconds == 1.0
    assert policy.backoff_multiplier == 2.0
    assert policy.max_delay_seconds == 60.0


def test_retry_policy_research_safety() -> None:
    policy = RetryPolicy()

    assert policy.metadata["research_only"] is True
    assert policy.metadata["real_money_execution"] is False


def test_retry_policy_rejects_zero_attempts() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(max_attempts=0)


def test_retry_policy_rejects_negative_attempts() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(max_attempts=-1)


def test_retry_policy_rejects_bool_attempts() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(max_attempts=True)


def test_retry_policy_rejects_float_attempts() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(max_attempts=2.5)


def test_retry_policy_rejects_negative_base_delay() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(base_delay_seconds=-1)


def test_retry_policy_rejects_boolean_base_delay() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(base_delay_seconds=True)


def test_retry_policy_rejects_backoff_below_one() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(backoff_multiplier=0.5)


def test_retry_policy_rejects_boolean_backoff() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(backoff_multiplier=True)


def test_retry_policy_rejects_negative_max_delay() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(
            base_delay_seconds=0,
            max_delay_seconds=-1,
        )


def test_retry_policy_rejects_max_delay_below_base() -> None:
    with pytest.raises(RecoveryValidationError):
        RetryPolicy(
            base_delay_seconds=10,
            max_delay_seconds=5,
        )


def test_retry_policy_requires_retry_list() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(
            retry_on="EXCEPTION"
        )


def test_retry_policy_requires_failure_types() -> None:
    with pytest.raises(TypeError):
        RetryPolicy(
            retry_on=["EXCEPTION"]
        )


def test_delay_first_attempt() -> None:
    policy = make_policy()

    assert policy.delay_for_attempt(1) == 1.0


def test_delay_second_attempt() -> None:
    policy = make_policy()

    assert policy.delay_for_attempt(2) == 2.0


def test_delay_third_attempt() -> None:
    policy = make_policy()

    assert policy.delay_for_attempt(3) == 4.0


def test_delay_respects_maximum() -> None:
    policy = make_policy(
        base_delay_seconds=10,
        backoff_multiplier=10,
        max_delay_seconds=50,
    )

    assert policy.delay_for_attempt(3) == 50.0


def test_delay_rejects_zero_attempt() -> None:
    policy = make_policy()

    with pytest.raises(RecoveryValidationError):
        policy.delay_for_attempt(0)


def test_delay_rejects_boolean_attempt() -> None:
    policy = make_policy()

    with pytest.raises(TypeError):
        policy.delay_for_attempt(True)


def test_policy_allows_failure() -> None:
    policy = make_policy(
        retry_on=[
            FailureType.TIMEOUT,
            FailureType.EXCEPTION,
        ]
    )

    assert policy.allows(FailureType.TIMEOUT)


def test_policy_disallows_failure() -> None:
    policy = make_policy(
        retry_on=[FailureType.TIMEOUT]
    )

    assert not policy.allows(
        FailureType.DATA_ERROR
    )


def test_policy_allows_requires_enum() -> None:
    policy = make_policy()

    with pytest.raises(TypeError):
        policy.allows("TIMEOUT")


# ============================================================
# RecoveryCheckpoint
# ============================================================


def test_checkpoint_creation() -> None:
    checkpoint = RecoveryCheckpoint(
        run_id="run_001",
        stage="agent_execution",
        state={"completed_agents": 5},
        sequence=1,
    )

    assert checkpoint.run_id == "run_001"
    assert checkpoint.stage == "agent_execution"
    assert checkpoint.sequence == 1


def test_checkpoint_has_deterministic_id() -> None:
    first = RecoveryCheckpoint(
        run_id="run_001",
        stage="validation",
        state={"score": 0.7},
        sequence=1,
    )

    second = RecoveryCheckpoint(
        run_id="run_001",
        stage="validation",
        state={"score": 0.7},
        sequence=1,
    )

    assert first.checkpoint_id == second.checkpoint_id


def test_checkpoint_state_dict_order_independent() -> None:
    first = RecoveryCheckpoint(
        run_id="run_001",
        stage="validation",
        state={"a": 1, "b": 2},
        sequence=1,
    )

    second = RecoveryCheckpoint(
        run_id="run_001",
        stage="validation",
        state={"b": 2, "a": 1},
        sequence=1,
    )

    assert (
        first.state_fingerprint
        == second.state_fingerprint
    )


def test_checkpoint_rejects_empty_run_id() -> None:
    with pytest.raises(RecoveryValidationError):
        RecoveryCheckpoint(
            run_id=" ",
            stage="stage",
            state={},
            sequence=1,
        )


def test_checkpoint_trims_run_id() -> None:
    checkpoint = RecoveryCheckpoint(
        run_id="  run_001  ",
        stage="stage",
        state={},
        sequence=1,
    )

    assert checkpoint.run_id == "run_001"


def test_checkpoint_rejects_empty_stage() -> None:
    with pytest.raises(RecoveryValidationError):
        RecoveryCheckpoint(
            run_id="run_001",
            stage=" ",
            state={},
            sequence=1,
        )


def test_checkpoint_trims_stage() -> None:
    checkpoint = RecoveryCheckpoint(
        run_id="run_001",
        stage="  validation  ",
        state={},
        sequence=1,
    )

    assert checkpoint.stage == "validation"


def test_checkpoint_rejects_zero_sequence() -> None:
    with pytest.raises(RecoveryValidationError):
        RecoveryCheckpoint(
            run_id="run_001",
            stage="stage",
            state={},
            sequence=0,
        )


def test_checkpoint_rejects_bool_sequence() -> None:
    with pytest.raises(TypeError):
        RecoveryCheckpoint(
            run_id="run_001",
            stage="stage",
            state={},
            sequence=True,
        )


def test_checkpoint_custom_id() -> None:
    checkpoint = RecoveryCheckpoint(
        run_id="run_001",
        stage="stage",
        state={},
        sequence=1,
        checkpoint_id="checkpoint_custom",
    )

    assert (
        checkpoint.checkpoint_id
        == "checkpoint_custom"
    )


def test_checkpoint_rejects_empty_custom_id() -> None:
    with pytest.raises(RecoveryValidationError):
        RecoveryCheckpoint(
            run_id="run_001",
            stage="stage",
            state={},
            sequence=1,
            checkpoint_id=" ",
        )


def test_checkpoint_research_safety() -> None:
    checkpoint = RecoveryCheckpoint(
        run_id="run_001",
        stage="stage",
        state={},
        sequence=1,
    )

    assert checkpoint.metadata["research_only"] is True
    assert (
        checkpoint.metadata["real_money_execution"]
        is False
    )


# ============================================================
# RecoveryManager basics
# ============================================================


def test_manager_version() -> None:
    assert RecoveryManager.VERSION == "v6.0.1"


def test_manager_starts_empty() -> None:
    manager = RecoveryManager()

    assert manager.count() == 0
    assert manager.checkpoint_count() == 0
    assert manager.recoveries() == []
    assert manager.checkpoints() == []
    assert manager.audit_log() == []


def test_manager_uses_default_policy() -> None:
    manager = RecoveryManager()

    assert isinstance(
        manager.default_policy,
        RetryPolicy,
    )


def test_manager_accepts_custom_default_policy() -> None:
    policy = make_policy(
        max_attempts=5
    )

    manager = RecoveryManager(
        default_policy=policy
    )

    assert manager.default_policy is policy


def test_manager_rejects_invalid_default_policy() -> None:
    with pytest.raises(TypeError):
        RecoveryManager(
            default_policy="invalid"
        )


# ============================================================
# Checkpoint manager
# ============================================================


def test_create_checkpoint() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_001",
        "market_data",
        {"loaded": True},
    )

    assert manager.checkpoint_count() == 1
    assert checkpoint.sequence == 1


def test_checkpoint_sequence_increments_per_run() -> None:
    manager = RecoveryManager()

    first = manager.create_checkpoint(
        "run_001",
        "stage_1",
        {"step": 1},
    )

    second = manager.create_checkpoint(
        "run_001",
        "stage_2",
        {"step": 2},
    )

    assert first.sequence == 1
    assert second.sequence == 2


def test_checkpoint_sequence_independent_between_runs() -> None:
    manager = RecoveryManager()

    first = manager.create_checkpoint(
        "run_001",
        "stage",
        {},
    )

    second = manager.create_checkpoint(
        "run_002",
        "stage",
        {},
    )

    assert first.sequence == 1
    assert second.sequence == 1


def test_get_checkpoint() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_001",
        "stage",
        {"value": 1},
    )

    assert manager.get_checkpoint(
        checkpoint.checkpoint_id
    ) is checkpoint


def test_get_missing_checkpoint_raises() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        CheckpointNotFoundError
    ):
        manager.get_checkpoint(
            "checkpoint_missing"
        )


def test_get_checkpoint_rejects_empty_id() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        RecoveryValidationError
    ):
        manager.get_checkpoint(" ")


def test_checkpoints_preserve_creation_order() -> None:
    manager = RecoveryManager()

    first = manager.create_checkpoint(
        "run_001",
        "one",
        {},
    )

    second = manager.create_checkpoint(
        "run_001",
        "two",
        {},
    )

    assert manager.checkpoints() == [
        first,
        second,
    ]


def test_checkpoints_for_run() -> None:
    manager = RecoveryManager()

    first = manager.create_checkpoint(
        "run_001",
        "one",
        {},
    )

    manager.create_checkpoint(
        "run_002",
        "other",
        {},
    )

    second = manager.create_checkpoint(
        "run_001",
        "two",
        {},
    )

    assert manager.checkpoints_for_run(
        "run_001"
    ) == [
        first,
        second,
    ]


def test_checkpoints_for_run_rejects_empty_id() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        RecoveryValidationError
    ):
        manager.checkpoints_for_run(" ")


def test_latest_checkpoint() -> None:
    manager = RecoveryManager()

    manager.create_checkpoint(
        "run_001",
        "first",
        {},
    )

    latest = manager.create_checkpoint(
        "run_001",
        "second",
        {},
    )

    assert manager.latest_checkpoint(
        "run_001"
    ) is latest


def test_latest_checkpoint_none() -> None:
    manager = RecoveryManager()

    assert manager.latest_checkpoint(
        "run_001"
    ) is None


def test_restore_checkpoint() -> None:
    manager = RecoveryManager()

    state = {
        "completed_agents": [
            "macro",
            "crypto",
        ]
    }

    checkpoint = manager.create_checkpoint(
        "run_001",
        "agents",
        state,
    )

    restored = manager.restore_checkpoint(
        checkpoint.checkpoint_id
    )

    assert restored == state


# ============================================================
# Recovery creation
# ============================================================


def test_create_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert manager.count() == 1
    assert recovery.status == RecoveryStatus.READY
    assert recovery.attempt_count == 0


def test_create_recovery_research_safety() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert recovery.metadata["research_only"] is True
    assert (
        recovery.metadata["real_money_execution"]
        is False
    )


def test_create_recovery_with_checkpoint() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_001",
        "agents",
        {"done": 4},
    )

    recovery = make_recovery(
        manager,
        run_id="run_001",
        checkpoint_id=checkpoint.checkpoint_id,
    )

    assert (
        recovery.checkpoint_id
        == checkpoint.checkpoint_id
    )


def test_checkpoint_must_belong_to_run() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_other",
        "stage",
        {},
    )

    with pytest.raises(
        RecoveryValidationError
    ):
        make_recovery(
            manager,
            run_id="run_001",
            checkpoint_id=checkpoint.checkpoint_id,
        )


def test_non_retryable_failure_is_exhausted() -> None:
    policy = make_policy(
        retry_on=[
            FailureType.TIMEOUT
        ]
    )

    manager = RecoveryManager()

    recovery = make_recovery(
        manager,
        failure_type=FailureType.DATA_ERROR,
        retry_policy=policy,
    )

    assert (
        recovery.status
        == RecoveryStatus.EXHAUSTED
    )

    assert recovery.can_retry is False


def test_recovery_has_generated_id() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert recovery.recovery_id.startswith(
        "recovery_"
    )


# ============================================================
# Retrieval
# ============================================================


def test_get_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert manager.get(
        recovery.recovery_id
    ) is recovery


def test_get_missing_recovery_raises() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        RecoveryNotFoundError
    ):
        manager.get(
            "recovery_missing"
        )


def test_get_recovery_rejects_empty_id() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        RecoveryValidationError
    ):
        manager.get(" ")


def test_exists_true() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert manager.exists(
        recovery.recovery_id
    )


def test_exists_false() -> None:
    manager = RecoveryManager()

    assert not manager.exists(
        "recovery_missing"
    )


def test_recoveries_preserve_order() -> None:
    manager = RecoveryManager()

    first = make_recovery(
        manager,
        run_id="run_001",
    )

    second = make_recovery(
        manager,
        run_id="run_002",
    )

    assert manager.recoveries() == [
        first,
        second,
    ]


def test_recoveries_for_run() -> None:
    manager = RecoveryManager()

    first = make_recovery(
        manager,
        run_id="run_001",
        error="first",
    )

    make_recovery(
        manager,
        run_id="run_002",
    )

    second = make_recovery(
        manager,
        run_id="run_001",
        error="second",
    )

    assert manager.recoveries_for_run(
        "run_001"
    ) == [
        first,
        second,
    ]


def test_recoveries_for_run_rejects_empty_id() -> None:
    manager = RecoveryManager()

    with pytest.raises(
        RecoveryValidationError
    ):
        manager.recoveries_for_run(" ")


# ============================================================
# Recovery attempts
# ============================================================


def test_begin_attempt() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    attempt = manager.begin_attempt(
        recovery.recovery_id
    )

    assert attempt.attempt == 1
    assert (
        recovery.status
        == RecoveryStatus.RECOVERING
    )


def test_begin_attempt_uses_checkpoint() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_001",
        "stage",
        {},
    )

    recovery = make_recovery(
        manager,
        run_id="run_001",
        checkpoint_id=checkpoint.checkpoint_id,
    )

    attempt = manager.begin_attempt(
        recovery.recovery_id
    )

    assert (
        attempt.checkpoint_id
        == checkpoint.checkpoint_id
    )


def test_begin_attempt_from_recovering_rejected() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    with pytest.raises(
        RecoveryStateError
    ):
        manager.begin_attempt(
            recovery.recovery_id
        )


def test_complete_attempt() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id,
        result={"status": "ok"},
    )

    assert (
        recovery.status
        == RecoveryStatus.RECOVERED
    )

    assert recovery.recovered_at is not None
    assert recovery.attempts[0].success is True
    assert (
        recovery.attempts[0].completed_at
        is not None
    )


def test_complete_attempt_stores_result_fingerprint() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id,
        result={"decision": "HOLD"},
    )

    assert (
        "result_fingerprint"
        in recovery.attempts[0].metadata
    )


def test_complete_without_active_attempt_rejected() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    with pytest.raises(
        RecoveryStateError
    ):
        manager.complete_attempt(
            recovery.recovery_id
        )


def test_fail_attempt() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="temporary failure",
    )

    assert (
        recovery.status
        == RecoveryStatus.FAILED
    )

    assert (
        recovery.attempts[0].success
        is False
    )


def test_fail_attempt_updates_error() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="new error",
    )

    assert recovery.error == "new error"


def test_fail_attempt_rejects_empty_error() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    with pytest.raises(
        RecoveryValidationError
    ):
        manager.fail_attempt(
            recovery.recovery_id,
            error=" ",
        )


def test_fail_without_active_attempt_rejected() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    with pytest.raises(
        RecoveryStateError
    ):
        manager.fail_attempt(
            recovery.recovery_id,
            error="failure",
        )


def test_second_attempt_after_failure() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="first failure",
    )

    second = manager.begin_attempt(
        recovery.recovery_id
    )

    assert second.attempt == 2


def test_attempt_exhaustion() -> None:
    policy = make_policy(
        max_attempts=2
    )

    manager = RecoveryManager()

    recovery = make_recovery(
        manager,
        retry_policy=policy,
    )

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure one",
    )

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure two",
    )

    assert (
        recovery.status
        == RecoveryStatus.EXHAUSTED
    )

    assert recovery.can_retry is False


def test_recovered_is_terminal() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    assert recovery.terminal is True


def test_ready_is_not_terminal() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert recovery.terminal is False


# ============================================================
# Cancellation
# ============================================================


def test_cancel_ready_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.cancel(
        recovery.recovery_id,
        reason="manual stop",
    )

    assert (
        recovery.status
        == RecoveryStatus.CANCELLED
    )

    assert (
        recovery.metadata[
            "cancellation_reason"
        ]
        == "manual stop"
    )


def test_cancel_recovering_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.cancel(
        recovery.recovery_id
    )

    assert (
        recovery.status
        == RecoveryStatus.CANCELLED
    )


def test_cancel_terminal_recovery_rejected() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    with pytest.raises(
        RecoveryStateError
    ):
        manager.cancel(
            recovery.recovery_id
        )


# ============================================================
# Delay
# ============================================================


def test_next_delay_before_first_attempt() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    assert manager.next_delay(
        recovery.recovery_id
    ) == 1.0


def test_next_delay_after_failed_attempt() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure",
    )

    assert manager.next_delay(
        recovery.recovery_id
    ) == 2.0


def test_next_delay_none_after_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    assert manager.next_delay(
        recovery.recovery_id
    ) is None


def test_next_delay_none_when_exhausted() -> None:
    policy = make_policy(
        max_attempts=1
    )

    manager = RecoveryManager()

    recovery = make_recovery(
        manager,
        retry_policy=policy,
    )

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure",
    )

    assert manager.next_delay(
        recovery.recovery_id
    ) is None


# ============================================================
# Audit
# ============================================================


def test_checkpoint_creation_audited() -> None:
    manager = RecoveryManager()

    manager.create_checkpoint(
        "run_001",
        "stage",
        {},
    )

    assert any(
        event["event"]
        == "CHECKPOINT_CREATED"
        for event in manager.audit_log()
    )


def test_checkpoint_restore_audited() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_001",
        "stage",
        {},
    )

    manager.restore_checkpoint(
        checkpoint.checkpoint_id
    )

    assert any(
        event["event"]
        == "CHECKPOINT_RESTORED"
        for event in manager.audit_log()
    )


def test_recovery_creation_audited() -> None:
    manager = RecoveryManager()

    make_recovery(manager)

    assert any(
        event["event"]
        == "RECOVERY_CREATED"
        for event in manager.audit_log()
    )


def test_attempt_start_audited() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    assert any(
        event["event"]
        == "RECOVERY_ATTEMPT_STARTED"
        for event in manager.audit_log()
    )


def test_success_audited() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    assert any(
        event["event"]
        == "RECOVERY_SUCCEEDED"
        for event in manager.audit_log()
    )


def test_failure_audited() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure",
    )

    assert any(
        event["event"]
        == "RECOVERY_ATTEMPT_FAILED"
        for event in manager.audit_log()
    )


def test_cancel_audited() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.cancel(
        recovery.recovery_id
    )

    assert any(
        event["event"]
        == "RECOVERY_CANCELLED"
        for event in manager.audit_log()
    )


# ============================================================
# Snapshot
# ============================================================


def test_empty_snapshot() -> None:
    manager = RecoveryManager()

    snapshot = manager.snapshot()

    assert snapshot.total_recoveries == 0
    assert snapshot.ready_recoveries == 0
    assert snapshot.recovering == 0
    assert snapshot.recovered == 0
    assert snapshot.failed == 0
    assert snapshot.exhausted == 0
    assert snapshot.cancelled == 0
    assert snapshot.checkpoints == 0
    assert snapshot.total_attempts == 0


def test_snapshot_counts_ready() -> None:
    manager = RecoveryManager()

    make_recovery(manager)

    snapshot = manager.snapshot()

    assert snapshot.total_recoveries == 1
    assert snapshot.ready_recoveries == 1


def test_snapshot_counts_recovering() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    snapshot = manager.snapshot()

    assert snapshot.recovering == 1
    assert snapshot.total_attempts == 1


def test_snapshot_counts_recovered() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    snapshot = manager.snapshot()

    assert snapshot.recovered == 1


def test_snapshot_counts_failed() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure",
    )

    snapshot = manager.snapshot()

    assert snapshot.failed == 1


def test_snapshot_counts_exhausted() -> None:
    policy = make_policy(
        max_attempts=1
    )

    manager = RecoveryManager()

    recovery = make_recovery(
        manager,
        retry_policy=policy,
    )

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.fail_attempt(
        recovery.recovery_id,
        error="failure",
    )

    snapshot = manager.snapshot()

    assert snapshot.exhausted == 1


def test_snapshot_counts_cancelled() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.cancel(
        recovery.recovery_id
    )

    snapshot = manager.snapshot()

    assert snapshot.cancelled == 1


def test_snapshot_counts_checkpoints() -> None:
    manager = RecoveryManager()

    manager.create_checkpoint(
        "run_001",
        "one",
        {},
    )

    manager.create_checkpoint(
        "run_001",
        "two",
        {},
    )

    snapshot = manager.snapshot()

    assert snapshot.checkpoints == 2


def test_snapshot_research_safety() -> None:
    manager = RecoveryManager()

    snapshot = manager.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Reports
# ============================================================


def test_recovery_report() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    report = manager.recovery_report(
        recovery.recovery_id
    )

    assert (
        report["recovery_id"]
        == recovery.recovery_id
    )
    assert report["run_id"] == "run_test_001"
    assert report["failure_type"] == "EXCEPTION"
    assert report["status"] == "READY"
    assert report["attempt_count"] == 0
    assert report["max_attempts"] == 3
    assert report["can_retry"] is True
    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


def test_recovery_report_after_success() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id,
        result={"ok": True},
    )

    report = manager.recovery_report(
        recovery.recovery_id
    )

    assert report["status"] == "RECOVERED"
    assert report["attempt_count"] == 1
    assert report["can_retry"] is False
    assert (
        report["next_delay_seconds"]
        is None
    )


def test_summary_empty() -> None:
    manager = RecoveryManager()

    summary = manager.summary()

    assert (
        summary["component"]
        == "recovery_manager"
    )
    assert summary["version"] == "v6.0.1"
    assert summary["total_recoveries"] == 0
    assert summary["checkpoints"] == 0
    assert summary["total_attempts"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_after_recovery() -> None:
    manager = RecoveryManager()

    recovery = make_recovery(manager)

    manager.begin_attempt(
        recovery.recovery_id
    )

    manager.complete_attempt(
        recovery.recovery_id
    )

    summary = manager.summary()

    assert summary["total_recoveries"] == 1
    assert summary["recovered"] == 1
    assert summary["total_attempts"] == 1


# ============================================================
# Full checkpoint recovery flow
# ============================================================


def test_full_checkpoint_recovery_flow() -> None:
    manager = RecoveryManager()

    checkpoint = manager.create_checkpoint(
        "run_full_test",
        "agent_execution",
        {
            "completed_agents": [
                "macro",
                "stock",
                "crypto",
            ],
            "remaining_agents": [
                "risk",
                "technical",
            ],
        },
    )

    recovery = manager.create_recovery(
        "run_full_test",
        failure_type=(
            FailureType.AGENT_ERROR
        ),
        error="Technical agent failed",
        checkpoint_id=(
            checkpoint.checkpoint_id
        ),
    )

    restored_state = (
        manager.restore_checkpoint(
            checkpoint.checkpoint_id
        )
    )

    assert (
        restored_state[
            "completed_agents"
        ]
        == [
            "macro",
            "stock",
            "crypto",
        ]
    )

    attempt = manager.begin_attempt(
        recovery.recovery_id
    )

    assert attempt.attempt == 1

    manager.complete_attempt(
        recovery.recovery_id,
        result={
            "recovered_agents": [
                "technical",
                "risk",
            ]
        },
    )

    assert (
        recovery.status
        == RecoveryStatus.RECOVERED
    )
    assert recovery.terminal is True