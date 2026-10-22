from __future__ import annotations

import pytest

from production_research_infrastructure.run_manager import (
    ManagedRunStatus,
    RunAlreadyExistsError,
    RunIdentity,
    RunManager,
    RunNotFoundError,
    RunStateTransitionError,
    RunValidationError,
)


def make_identity(
    *,
    research_name: str = "market_research",
    configuration_fingerprint: str = "config_abc123",
    dataset_fingerprints=None,
    parameters=None,
    seed: int | None = 42,
    namespace: str = "research",
) -> RunIdentity:
    if dataset_fingerprints is None:
        dataset_fingerprints = [
            "dataset_market_001",
            "dataset_macro_001",
        ]

    if parameters is None:
        parameters = {
            "asset": "BTC",
            "horizon": "1D",
        }

    return RunIdentity(
        research_name=research_name,
        configuration_fingerprint=configuration_fingerprint,
        dataset_fingerprints=dataset_fingerprints,
        parameters=parameters,
        seed=seed,
        namespace=namespace,
    )


# ============================================================
# RunIdentity
# ============================================================


def test_identity_creation() -> None:
    identity = make_identity()

    assert identity.research_name == "market_research"
    assert identity.seed == 42
    assert identity.namespace == "research"


def test_identity_trims_research_name() -> None:
    identity = make_identity(
        research_name="  macro_research  "
    )

    assert identity.research_name == "macro_research"


def test_identity_rejects_empty_research_name() -> None:
    with pytest.raises(RunValidationError):
        make_identity(
            research_name=" "
        )


def test_identity_rejects_empty_configuration_fingerprint() -> None:
    with pytest.raises(RunValidationError):
        make_identity(
            configuration_fingerprint=" "
        )


def test_identity_trims_configuration_fingerprint() -> None:
    identity = make_identity(
        configuration_fingerprint="  abc123  "
    )

    assert (
        identity.configuration_fingerprint
        == "abc123"
    )


def test_identity_requires_dataset_list() -> None:
    with pytest.raises(TypeError):
        make_identity(
            dataset_fingerprints="invalid"
        )


def test_identity_rejects_empty_dataset_fingerprint() -> None:
    with pytest.raises(RunValidationError):
        make_identity(
            dataset_fingerprints=[
                "dataset_one",
                " ",
            ]
        )


def test_identity_trims_dataset_fingerprints() -> None:
    identity = make_identity(
        dataset_fingerprints=[
            "  dataset_one  ",
            " dataset_two ",
        ]
    )

    assert identity.dataset_fingerprints == [
        "dataset_one",
        "dataset_two",
    ]


def test_identity_requires_parameters_dict() -> None:
    with pytest.raises(TypeError):
        make_identity(
            parameters=["invalid"]
        )


def test_identity_accepts_none_seed() -> None:
    identity = make_identity(
        seed=None
    )

    assert identity.seed is None


def test_identity_rejects_boolean_seed() -> None:
    with pytest.raises(TypeError):
        make_identity(
            seed=True
        )


def test_identity_rejects_float_seed() -> None:
    with pytest.raises(TypeError):
        make_identity(
            seed=1.5
        )


def test_identity_rejects_empty_namespace() -> None:
    with pytest.raises(RunValidationError):
        make_identity(
            namespace=" "
        )


def test_identity_trims_namespace() -> None:
    identity = make_identity(
        namespace="  quant_lab  "
    )

    assert identity.namespace == "quant_lab"


def test_identity_research_safety() -> None:
    identity = make_identity()

    assert identity.metadata["research_only"] is True
    assert (
        identity.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Deterministic fingerprint / run ID
# ============================================================


def test_identity_fingerprint_is_stable() -> None:
    first = make_identity()
    second = make_identity()

    assert first.fingerprint == second.fingerprint


def test_reproducible_run_id_is_stable() -> None:
    first = make_identity()
    second = make_identity()

    assert (
        first.reproducible_run_id
        == second.reproducible_run_id
    )


def test_reproducible_run_id_prefix() -> None:
    identity = make_identity()

    assert identity.reproducible_run_id.startswith(
        "run_"
    )


def test_dataset_order_does_not_change_run_id() -> None:
    first = make_identity(
        dataset_fingerprints=[
            "dataset_a",
            "dataset_b",
        ]
    )

    second = make_identity(
        dataset_fingerprints=[
            "dataset_b",
            "dataset_a",
        ]
    )

    assert (
        first.reproducible_run_id
        == second.reproducible_run_id
    )


def test_parameter_order_does_not_change_run_id() -> None:
    first = make_identity(
        parameters={
            "asset": "BTC",
            "horizon": "1D",
        }
    )

    second = make_identity(
        parameters={
            "horizon": "1D",
            "asset": "BTC",
        }
    )

    assert (
        first.reproducible_run_id
        == second.reproducible_run_id
    )


def test_configuration_change_changes_run_id() -> None:
    first = make_identity(
        configuration_fingerprint="config_a"
    )

    second = make_identity(
        configuration_fingerprint="config_b"
    )

    assert (
        first.reproducible_run_id
        != second.reproducible_run_id
    )


def test_dataset_change_changes_run_id() -> None:
    first = make_identity(
        dataset_fingerprints=["dataset_a"]
    )

    second = make_identity(
        dataset_fingerprints=["dataset_b"]
    )

    assert (
        first.reproducible_run_id
        != second.reproducible_run_id
    )


def test_seed_change_changes_run_id() -> None:
    first = make_identity(seed=1)
    second = make_identity(seed=2)

    assert (
        first.reproducible_run_id
        != second.reproducible_run_id
    )


def test_parameter_change_changes_run_id() -> None:
    first = make_identity(
        parameters={"asset": "BTC"}
    )

    second = make_identity(
        parameters={"asset": "ETH"}
    )

    assert (
        first.reproducible_run_id
        != second.reproducible_run_id
    )


# ============================================================
# RunManager basics
# ============================================================


def test_manager_version() -> None:
    assert RunManager.VERSION == "v6.0.1"


def test_manager_starts_empty() -> None:
    manager = RunManager()

    assert manager.count() == 0
    assert manager.runs() == []
    assert manager.audit_log() == []


def test_generate_run_id() -> None:
    identity = make_identity()

    run_id = RunManager.generate_run_id(
        identity
    )

    assert run_id == identity.reproducible_run_id


def test_generate_run_id_requires_identity() -> None:
    with pytest.raises(TypeError):
        RunManager.generate_run_id(
            "invalid"
        )


# ============================================================
# Run creation
# ============================================================


def test_create_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert run.status == ManagedRunStatus.CREATED
    assert run.attempt == 1
    assert manager.count() == 1


def test_create_run_uses_reproducible_id() -> None:
    manager = RunManager()
    identity = make_identity()

    run = manager.create_run(identity)

    assert (
        run.run_id
        == identity.reproducible_run_id
    )


def test_create_run_research_safety() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert run.metadata["research_only"] is True
    assert (
        run.metadata["real_money_execution"]
        is False
    )


def test_create_run_custom_metadata() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity(),
        metadata={
            "source": "research_engine"
        },
    )

    assert (
        run.metadata["source"]
        == "research_engine"
    )


def test_create_duplicate_run_rejected() -> None:
    manager = RunManager()
    identity = make_identity()

    manager.create_run(identity)

    with pytest.raises(
        RunAlreadyExistsError
    ):
        manager.create_run(identity)


def test_create_run_requires_identity() -> None:
    manager = RunManager()

    with pytest.raises(TypeError):
        manager.create_run(
            "invalid"
        )


# ============================================================
# Retrieval
# ============================================================


def test_get_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert manager.get(
        run.run_id
    ) is run


def test_get_missing_run_raises() -> None:
    manager = RunManager()

    with pytest.raises(RunNotFoundError):
        manager.get(
            "run_missing"
        )


def test_get_rejects_empty_id() -> None:
    manager = RunManager()

    with pytest.raises(RunValidationError):
        manager.get(" ")


def test_exists_true() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert manager.exists(
        run.run_id
    )


def test_exists_false() -> None:
    manager = RunManager()

    assert not manager.exists(
        "run_missing"
    )


def test_runs_preserve_creation_order() -> None:
    manager = RunManager()

    first = manager.create_run(
        make_identity(
            research_name="first"
        )
    )

    second = manager.create_run(
        make_identity(
            research_name="second"
        )
    )

    assert manager.runs() == [
        first,
        second,
    ]


# ============================================================
# Lifecycle
# ============================================================


def test_start_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    assert run.status == ManagedRunStatus.RUNNING
    assert run.started_at is not None


def test_complete_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.complete(
        run.run_id,
        result={"decision": "HOLD"},
    )

    assert run.status == ManagedRunStatus.COMPLETED
    assert run.completed_at is not None
    assert run.result_fingerprint is not None


def test_partial_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.mark_partial(
        run.run_id,
        result={"completed": 8},
        reason="One agent failed",
    )

    assert run.status == ManagedRunStatus.PARTIAL
    assert run.result_fingerprint is not None


def test_fail_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.fail(
        run.run_id,
        error="Research execution failed",
    )

    assert run.status == ManagedRunStatus.FAILED
    assert (
        run.error
        == "Research execution failed"
    )


def test_fail_rejects_empty_error() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    with pytest.raises(RunValidationError):
        manager.fail(
            run.run_id,
            error=" ",
        )


def test_timeout_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.timeout(run.run_id)

    assert run.status == ManagedRunStatus.TIMED_OUT


def test_cancel_created_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.cancel(run.run_id)

    assert run.status == ManagedRunStatus.CANCELLED


def test_cancel_running_run() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.cancel(run.run_id)

    assert run.status == ManagedRunStatus.CANCELLED


def test_invalid_transition_rejected() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    with pytest.raises(
        RunStateTransitionError
    ):
        manager.complete(run.run_id)


def test_completed_run_cannot_restart() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    with pytest.raises(
        RunStateTransitionError
    ):
        manager.start(run.run_id)


def test_transition_requires_status_enum() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    with pytest.raises(TypeError):
        manager.transition(
            run.run_id,
            "RUNNING"
        )


def test_transition_history_recorded() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    assert len(run.transitions) == 2

    assert (
        run.transitions[0].previous_status
        == ManagedRunStatus.CREATED
    )

    assert (
        run.transitions[0].new_status
        == ManagedRunStatus.RUNNING
    )


def test_terminal_property_completed() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    assert run.terminal is True


def test_terminal_property_running_false() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    assert run.terminal is False


def test_duration_non_negative() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    assert run.duration_seconds >= 0.0


# ============================================================
# Recovery
# ============================================================


def test_begin_recovery_from_failed() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.fail(
        run.run_id,
        error="failure",
    )

    manager.begin_recovery(
        run.run_id
    )

    assert (
        run.status
        == ManagedRunStatus.RECOVERING
    )


def test_begin_recovery_from_timeout() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.timeout(run.run_id)

    manager.begin_recovery(
        run.run_id
    )

    assert (
        run.status
        == ManagedRunStatus.RECOVERING
    )


def test_begin_recovery_from_partial() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.mark_partial(run.run_id)

    manager.begin_recovery(
        run.run_id
    )

    assert (
        run.status
        == ManagedRunStatus.RECOVERING
    )


def test_mark_recovered() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.fail(
        run.run_id,
        error="failure",
    )

    manager.begin_recovery(
        run.run_id
    )

    manager.mark_recovered(
        run.run_id,
        result={"recovered": True},
    )

    assert (
        run.status
        == ManagedRunStatus.RECOVERED
    )

    assert run.result_fingerprint is not None


def test_create_recovery_run_from_failed() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    recovery = manager.create_recovery_run(
        parent.run_id
    )

    assert recovery.attempt == 2
    assert (
        recovery.parent_run_id
        == parent.run_id
    )

    assert recovery.status == ManagedRunStatus.CREATED


def test_recovery_run_id_contains_retry() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    recovery = manager.create_recovery_run(
        parent.run_id
    )

    assert recovery.run_id == (
        f"{parent.run_id}_retry_2"
    )


def test_recovery_run_from_completed_rejected() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    with pytest.raises(
        RunStateTransitionError
    ):
        manager.create_recovery_run(
            run.run_id
        )


def test_duplicate_recovery_run_rejected() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    manager.create_recovery_run(
        parent.run_id
    )

    with pytest.raises(
        RunAlreadyExistsError
    ):
        manager.create_recovery_run(
            parent.run_id
        )


def test_parent_of_recovery_run() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    recovery = manager.create_recovery_run(
        parent.run_id
    )

    assert manager.parent_of(
        recovery.run_id
    ) is parent


def test_parent_of_root_returns_none() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert manager.parent_of(
        run.run_id
    ) is None


def test_recovery_runs() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    recovery = manager.create_recovery_run(
        parent.run_id
    )

    assert manager.recovery_runs(
        parent.run_id
    ) == [recovery]


# ============================================================
# Reproducibility
# ============================================================


def test_is_reproducible_true() -> None:
    manager = RunManager()
    identity = make_identity()

    run = manager.create_run(identity)

    assert manager.is_reproducible(
        run.run_id,
        make_identity(),
    )


def test_is_reproducible_false() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    different = make_identity(
        seed=999
    )

    assert not manager.is_reproducible(
        run.run_id,
        different,
    )


def test_is_reproducible_requires_identity() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    with pytest.raises(TypeError):
        manager.is_reproducible(
            run.run_id,
            "invalid"
        )


def test_verify_result_true() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    result = {
        "decision": "HOLD",
        "score": 0.55,
    }

    manager.complete(
        run.run_id,
        result=result,
    )

    assert manager.verify_result(
        run.run_id,
        result,
    )


def test_verify_result_false_for_changed_result() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.complete(
        run.run_id,
        result={"score": 1},
    )

    assert not manager.verify_result(
        run.run_id,
        {"score": 2},
    )


def test_verify_result_false_without_result() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    assert not manager.verify_result(
        run.run_id,
        {"anything": True},
    )


# ============================================================
# Filtering
# ============================================================


def test_runs_by_status() -> None:
    manager = RunManager()

    first = manager.create_run(
        make_identity(
            research_name="one"
        )
    )

    second = manager.create_run(
        make_identity(
            research_name="two"
        )
    )

    manager.start(first.run_id)
    manager.complete(first.run_id)

    manager.start(second.run_id)

    completed = manager.runs_by_status(
        ManagedRunStatus.COMPLETED
    )

    running = manager.runs_by_status(
        ManagedRunStatus.RUNNING
    )

    assert completed == [first]
    assert running == [second]


def test_runs_by_status_requires_enum() -> None:
    manager = RunManager()

    with pytest.raises(TypeError):
        manager.runs_by_status(
            "COMPLETED"
        )


# ============================================================
# Audit
# ============================================================


def test_create_run_creates_audit_event() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    audit = manager.audit_log()

    assert (
        audit[0]["event"]
        == "RUN_CREATED"
    )

    assert (
        audit[0]["details"]["run_id"]
        == run.run_id
    )


def test_transition_creates_audit_event() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    assert any(
        item["event"] == "RUN_TRANSITION"
        for item in manager.audit_log()
    )


def test_recovery_creation_creates_audit_event() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.fail(
        run.run_id,
        error="failure",
    )

    manager.create_recovery_run(
        run.run_id
    )

    assert any(
        item["event"]
        == "RECOVERY_RUN_CREATED"
        for item in manager.audit_log()
    )


# ============================================================
# Snapshot
# ============================================================


def test_empty_snapshot() -> None:
    manager = RunManager()

    snapshot = manager.snapshot()

    assert snapshot.total_runs == 0
    assert snapshot.reproducible_runs == 0
    assert snapshot.recovery_runs == 0


def test_snapshot_status_counts() -> None:
    manager = RunManager()

    first = manager.create_run(
        make_identity(
            research_name="first"
        )
    )

    second = manager.create_run(
        make_identity(
            research_name="second"
        )
    )

    manager.start(first.run_id)
    manager.complete(first.run_id)

    manager.start(second.run_id)

    snapshot = manager.snapshot()

    assert snapshot.total_runs == 2

    assert (
        snapshot.status_counts["COMPLETED"]
        == 1
    )

    assert (
        snapshot.status_counts["RUNNING"]
        == 1
    )


def test_snapshot_counts_recovery_runs() -> None:
    manager = RunManager()

    parent = manager.create_run(
        make_identity()
    )

    manager.start(parent.run_id)

    manager.fail(
        parent.run_id,
        error="failure",
    )

    manager.create_recovery_run(
        parent.run_id
    )

    snapshot = manager.snapshot()

    assert snapshot.total_runs == 2
    assert snapshot.recovery_runs == 1


def test_snapshot_research_safety() -> None:
    manager = RunManager()

    snapshot = manager.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Reports
# ============================================================


def test_run_report() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)

    manager.complete(
        run.run_id,
        result={"signal": "HOLD"},
    )

    report = manager.run_report(
        run.run_id
    )

    assert report["run_id"] == run.run_id
    assert report["status"] == "COMPLETED"
    assert report["attempt"] == 1
    assert report["result_fingerprint"] is not None


def test_run_report_contains_configuration() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity(
            configuration_fingerprint="config_xyz"
        )
    )

    report = manager.run_report(
        run.run_id
    )

    assert (
        report["configuration_fingerprint"]
        == "config_xyz"
    )


def test_run_report_research_safety() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    report = manager.run_report(
        run.run_id
    )

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


def test_summary_empty() -> None:
    manager = RunManager()

    summary = manager.summary()

    assert summary["component"] == "run_manager"
    assert summary["version"] == "v6.0.1"
    assert summary["total_runs"] == 0
    assert summary["recovery_runs"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_after_runs() -> None:
    manager = RunManager()

    run = manager.create_run(
        make_identity()
    )

    manager.start(run.run_id)
    manager.complete(run.run_id)

    summary = manager.summary()

    assert summary["total_runs"] == 1
    assert summary["reproducible_runs"] == 1

    assert (
        summary["status_counts"]["COMPLETED"]
        == 1
    )


# ============================================================
# Result fingerprint determinism
# ============================================================


def test_result_fingerprint_dict_order_independent() -> None:
    first_manager = RunManager()
    second_manager = RunManager()

    first_run = first_manager.create_run(
        make_identity()
    )

    second_run = second_manager.create_run(
        make_identity()
    )

    first_manager.start(
        first_run.run_id
    )

    second_manager.start(
        second_run.run_id
    )

    first_manager.complete(
        first_run.run_id,
        result={
            "decision": "HOLD",
            "score": 0.5,
        },
    )

    second_manager.complete(
        second_run.run_id,
        result={
            "score": 0.5,
            "decision": "HOLD",
        },
    )

    assert (
        first_run.result_fingerprint
        == second_run.result_fingerprint
    )