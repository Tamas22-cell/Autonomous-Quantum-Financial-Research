from __future__ import annotations

import pytest

from production_research_infrastructure.artifact_registry import (
    ArtifactAlreadyExistsError,
    ArtifactNotFoundError,
    ArtifactRecord,
    ArtifactReference,
    ArtifactRegistry,
    ArtifactRegistryError,
    ArtifactStatus,
    ArtifactType,
    ArtifactValidationError,
    stable_hash,
)


def make_artifact(
    registry: ArtifactRegistry,
    *,
    name: str = "research_report",
    artifact_type: ArtifactType = ArtifactType.RESEARCH_REPORT,
    content=None,
    run_id: str | None = "run_001",
    experiment_id: str | None = None,
    dataset_ids=None,
    version_ids=None,
    references=None,
):
    if content is None:
        content = {"decision": "HOLD", "score": 0.55}

    return registry.create(
        name=name,
        artifact_type=artifact_type,
        content=content,
        run_id=run_id,
        experiment_id=experiment_id,
        dataset_ids=dataset_ids,
        version_ids=version_ids,
        references=references,
    )


# ============================================================
# ArtifactReference
# ============================================================


def test_reference_creation() -> None:
    reference = ArtifactReference(
        reference_type="dataset",
        reference_id="dataset_001",
    )

    assert reference.reference_type == "dataset"
    assert reference.reference_id == "dataset_001"
    assert reference.relation == "RELATED_TO"


def test_reference_trims_values() -> None:
    reference = ArtifactReference(
        reference_type=" dataset ",
        reference_id=" dataset_001 ",
        relation=" USES ",
    )

    assert reference.reference_type == "dataset"
    assert reference.reference_id == "dataset_001"
    assert reference.relation == "USES"


@pytest.mark.parametrize(
    "field_name,kwargs",
    [
        (
            "reference_type",
            {
                "reference_type": " ",
                "reference_id": "id",
            },
        ),
        (
            "reference_id",
            {
                "reference_type": "dataset",
                "reference_id": " ",
            },
        ),
        (
            "relation",
            {
                "reference_type": "dataset",
                "reference_id": "id",
                "relation": " ",
            },
        ),
    ],
)
def test_reference_rejects_empty_values(
    field_name,
    kwargs,
) -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactReference(**kwargs)


def test_reference_safety_metadata() -> None:
    reference = ArtifactReference(
        reference_type="run",
        reference_id="run_001",
    )

    assert reference.metadata["research_only"] is True
    assert (
        reference.metadata["real_money_execution"]
        is False
    )


# ============================================================
# ArtifactRecord
# ============================================================


def test_artifact_record_creation() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"value": 1},
    )

    assert artifact.name == "report"
    assert artifact.status == ArtifactStatus.CREATED
    assert artifact.artifact_id.startswith("artifact_")


def test_artifact_name_trimmed() -> None:
    artifact = ArtifactRecord(
        name=" report ",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
    )

    assert artifact.name == "report"


def test_artifact_rejects_empty_name() -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactRecord(
            name=" ",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
        )


def test_artifact_requires_artifact_type_enum() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type="RESEARCH_REPORT",
            content={},
        )


def test_artifact_requires_status_enum() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            status="CREATED",
        )


@pytest.mark.parametrize(
    "field_name",
    ["run_id", "experiment_id"],
)
def test_optional_ids_reject_empty_strings(
    field_name,
) -> None:
    kwargs = {
        "name": "report",
        "artifact_type": ArtifactType.RESEARCH_REPORT,
        "content": {},
        field_name: " ",
    }

    with pytest.raises(ArtifactValidationError):
        ArtifactRecord(**kwargs)


def test_optional_ids_are_trimmed() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        run_id=" run_001 ",
        experiment_id=" experiment_001 ",
    )

    assert artifact.run_id == "run_001"
    assert artifact.experiment_id == "experiment_001"


def test_dataset_ids_must_be_list() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            dataset_ids="dataset_001",
        )


def test_version_ids_must_be_list() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            version_ids="version_001",
        )


def test_references_must_be_list() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            references="invalid",
        )


def test_dataset_ids_reject_empty_value() -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            dataset_ids=["dataset_001", " "],
        )


def test_version_ids_reject_empty_value() -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            version_ids=["version_001", " "],
        )


def test_dataset_and_version_ids_trimmed() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        dataset_ids=[" dataset_001 "],
        version_ids=[" version_001 "],
    )

    assert artifact.dataset_ids == ["dataset_001"]
    assert artifact.version_ids == ["version_001"]


def test_references_require_reference_objects() -> None:
    with pytest.raises(TypeError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            references=["invalid"],
        )


def test_custom_artifact_id() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        artifact_id="artifact_custom",
    )

    assert artifact.artifact_id == "artifact_custom"


def test_empty_custom_artifact_id_rejected() -> None:
    with pytest.raises(ArtifactValidationError):
        ArtifactRecord(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={},
            artifact_id=" ",
        )


def test_artifact_id_is_deterministic() -> None:
    first = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"a": 1, "b": 2},
        run_id="run_001",
        dataset_ids=["dataset_001"],
        version_ids=["version_001"],
    )

    second = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"b": 2, "a": 1},
        run_id="run_001",
        dataset_ids=["dataset_001"],
        version_ids=["version_001"],
    )

    assert first.artifact_id == second.artifact_id


def test_artifact_id_independent_of_dataset_order() -> None:
    first = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        dataset_ids=["dataset_b", "dataset_a"],
    )

    second = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        dataset_ids=["dataset_a", "dataset_b"],
    )

    assert first.artifact_id == second.artifact_id


def test_content_fingerprint_deterministic() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"b": 2, "a": 1},
    )

    assert artifact.content_fingerprint == stable_hash(
        {"a": 1, "b": 2}
    )


def test_lineage_fingerprint_available() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
        run_id="run_001",
        dataset_ids=["dataset_001"],
        version_ids=["version_001"],
    )

    assert len(artifact.lineage_fingerprint) == 64


def test_artifact_safety_metadata() -> None:
    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
    )

    assert artifact.metadata["research_only"] is True
    assert (
        artifact.metadata["real_money_execution"]
        is False
    )


@pytest.mark.parametrize(
    "artifact_type",
    list(ArtifactType),
)
def test_all_artifact_types_supported(
    artifact_type,
) -> None:
    artifact = ArtifactRecord(
        name=f"artifact_{artifact_type.value}",
        artifact_type=artifact_type,
        content={},
    )

    assert artifact.artifact_type == artifact_type


# ============================================================
# Registry basics
# ============================================================


def test_registry_version() -> None:
    assert ArtifactRegistry.VERSION == "v6.0.1"


def test_registry_starts_empty() -> None:
    registry = ArtifactRegistry()

    assert registry.count() == 0
    assert registry.artifacts() == []
    assert registry.audit_log() == []


def test_register_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={},
    )

    result = registry.register(artifact)

    assert result is artifact
    assert registry.count() == 1


def test_register_requires_artifact_record() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(TypeError):
        registry.register("invalid")


def test_duplicate_artifact_rejected() -> None:
    registry = ArtifactRegistry()

    artifact = ArtifactRecord(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"value": 1},
    )

    registry.register(artifact)

    with pytest.raises(ArtifactAlreadyExistsError):
        registry.register(artifact)


def test_duplicate_deterministic_artifact_rejected() -> None:
    registry = ArtifactRegistry()

    registry.create(
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"value": 1},
        run_id="run_001",
    )

    with pytest.raises(ArtifactAlreadyExistsError):
        registry.create(
            name="report",
            artifact_type=ArtifactType.RESEARCH_REPORT,
            content={"value": 1},
            run_id="run_001",
        )


def test_create_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    assert registry.count() == 1
    assert artifact.name == "research_report"


def test_create_with_all_lineage_fields() -> None:
    registry = ArtifactRegistry()

    reference = ArtifactReference(
        reference_type="strategy",
        reference_id="strategy_001",
        relation="GENERATED_FROM",
    )

    artifact = registry.create(
        name="backtest",
        artifact_type=ArtifactType.BACKTEST_RESULT,
        content={"sharpe": 1.2},
        run_id="run_001",
        experiment_id="experiment_001",
        dataset_ids=["dataset_001"],
        version_ids=["model_v1", "prompt_v2"],
        references=[reference],
    )

    assert artifact.run_id == "run_001"
    assert artifact.experiment_id == "experiment_001"
    assert artifact.dataset_ids == ["dataset_001"]
    assert artifact.version_ids == [
        "model_v1",
        "prompt_v2",
    ]
    assert artifact.references == [reference]


# ============================================================
# Retrieval
# ============================================================


def test_get_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    assert registry.get(artifact.artifact_id) is artifact


def test_get_missing_artifact() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactNotFoundError):
        registry.get("artifact_missing")


def test_get_rejects_empty_id() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.get(" ")


def test_exists_true() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    assert registry.exists(artifact.artifact_id)


def test_exists_false() -> None:
    registry = ArtifactRegistry()

    assert not registry.exists("artifact_missing")


def test_artifacts_preserve_registration_order() -> None:
    registry = ArtifactRegistry()

    first = make_artifact(
        registry,
        name="first",
        content={"value": 1},
    )

    second = make_artifact(
        registry,
        name="second",
        content={"value": 2},
    )

    assert registry.artifacts() == [first, second]


# ============================================================
# Filtering
# ============================================================


def test_filter_by_type() -> None:
    registry = ArtifactRegistry()

    report = make_artifact(
        registry,
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
    )

    make_artifact(
        registry,
        name="metric",
        artifact_type=ArtifactType.METRIC,
    )

    assert registry.by_type(
        ArtifactType.RESEARCH_REPORT
    ) == [report]


def test_filter_by_type_requires_enum() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(TypeError):
        registry.by_type("RESEARCH_REPORT")


def test_filter_by_status() -> None:
    registry = ArtifactRegistry()

    created = make_artifact(
        registry,
        name="created",
        content={"id": 1},
    )

    verified = make_artifact(
        registry,
        name="verified",
        content={"id": 2},
    )

    registry.verify(verified.artifact_id)

    assert registry.by_status(
        ArtifactStatus.CREATED
    ) == [created]

    assert registry.by_status(
        ArtifactStatus.VERIFIED
    ) == [verified]


def test_filter_by_status_requires_enum() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(TypeError):
        registry.by_status("CREATED")


def test_artifacts_for_run() -> None:
    registry = ArtifactRegistry()

    first = make_artifact(
        registry,
        name="one",
        run_id="run_001",
        content={"id": 1},
    )

    make_artifact(
        registry,
        name="other",
        run_id="run_002",
        content={"id": 2},
    )

    second = make_artifact(
        registry,
        name="two",
        run_id="run_001",
        content={"id": 3},
    )

    assert registry.for_run("run_001") == [
        first,
        second,
    ]


def test_for_run_rejects_empty_id() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.for_run(" ")


def test_artifacts_for_experiment() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(
        registry,
        experiment_id="experiment_001",
    )

    assert registry.for_experiment(
        "experiment_001"
    ) == [artifact]


def test_for_experiment_rejects_empty_id() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.for_experiment(" ")


def test_artifacts_using_dataset() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(
        registry,
        dataset_ids=[
            "dataset_001",
            "dataset_002",
        ],
    )

    assert registry.using_dataset(
        "dataset_002"
    ) == [artifact]


def test_using_dataset_rejects_empty_id() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.using_dataset(" ")


def test_artifacts_using_version() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(
        registry,
        version_ids=[
            "model_v1",
            "prompt_v1",
        ],
    )

    assert registry.using_version(
        "model_v1"
    ) == [artifact]


def test_using_version_rejects_empty_id() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.using_version(" ")


# ============================================================
# References
# ============================================================


def test_add_reference() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    reference = ArtifactReference(
        reference_type="strategy",
        reference_id="strategy_001",
        relation="GENERATED_FROM",
    )

    result = registry.add_reference(
        artifact.artifact_id,
        reference,
    )

    assert result is artifact
    assert artifact.references == [reference]


def test_add_reference_requires_reference_object() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    with pytest.raises(TypeError):
        registry.add_reference(
            artifact.artifact_id,
            "invalid",
        )


def test_references_to() -> None:
    registry = ArtifactRegistry()

    reference = ArtifactReference(
        reference_type="hypothesis",
        reference_id="hypothesis_001",
    )

    artifact = make_artifact(
        registry,
        references=[reference],
    )

    assert registry.references_to(
        "hypothesis_001"
    ) == [artifact]


def test_references_to_empty_id_rejected() -> None:
    registry = ArtifactRegistry()

    with pytest.raises(ArtifactValidationError):
        registry.references_to(" ")


def test_lineage_fingerprint_changes_after_reference() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    before = artifact.lineage_fingerprint

    registry.add_reference(
        artifact.artifact_id,
        ArtifactReference(
            reference_type="strategy",
            reference_id="strategy_001",
        ),
    )

    after = artifact.lineage_fingerprint

    assert before != after


# ============================================================
# Lifecycle
# ============================================================


def test_verify_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.verify(artifact.artifact_id)

    assert artifact.status == ArtifactStatus.VERIFIED
    assert artifact.verified_at is not None


def test_verify_invalidated_artifact_rejected() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.invalidate(artifact.artifact_id)

    with pytest.raises(ArtifactRegistryError):
        registry.verify(artifact.artifact_id)


def test_verify_archived_artifact_rejected() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.archive(artifact.artifact_id)

    with pytest.raises(ArtifactRegistryError):
        registry.verify(artifact.artifact_id)


def test_invalidate_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.invalidate(
        artifact.artifact_id,
        reason="stale dataset",
    )

    assert artifact.status == ArtifactStatus.INVALIDATED
    assert artifact.invalidated_at is not None
    assert (
        artifact.metadata["invalidation_reason"]
        == "stale dataset"
    )


def test_invalidate_archived_artifact_rejected() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.archive(artifact.artifact_id)

    with pytest.raises(ArtifactRegistryError):
        registry.invalidate(artifact.artifact_id)


def test_archive_artifact() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.archive(artifact.artifact_id)

    assert artifact.status == ArtifactStatus.ARCHIVED
    assert artifact.archived_at is not None


def test_archive_invalidated_artifact_allowed() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.invalidate(artifact.artifact_id)
    registry.archive(artifact.artifact_id)

    assert artifact.status == ArtifactStatus.ARCHIVED


# ============================================================
# Integrity
# ============================================================


def test_verify_content_true() -> None:
    registry = ArtifactRegistry()

    content = {
        "sharpe": 1.25,
        "drawdown": -0.08,
    }

    artifact = make_artifact(
        registry,
        content=content,
    )

    assert registry.verify_content(
        artifact.artifact_id,
        {
            "drawdown": -0.08,
            "sharpe": 1.25,
        },
    )


def test_verify_content_false() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(
        registry,
        content={"score": 1},
    )

    assert not registry.verify_content(
        artifact.artifact_id,
        {"score": 2},
    )


# ============================================================
# Audit
# ============================================================


def test_registration_is_audited() -> None:
    registry = ArtifactRegistry()

    make_artifact(registry)

    assert any(
        event["event"] == "ARTIFACT_REGISTERED"
        for event in registry.audit_log()
    )


def test_reference_add_is_audited() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)

    registry.add_reference(
        artifact.artifact_id,
        ArtifactReference(
            reference_type="run",
            reference_id="run_parent",
        ),
    )

    assert any(
        event["event"]
        == "ARTIFACT_REFERENCE_ADDED"
        for event in registry.audit_log()
    )


def test_verify_is_audited() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)
    registry.verify(artifact.artifact_id)

    assert any(
        event["event"] == "ARTIFACT_VERIFIED"
        for event in registry.audit_log()
    )


def test_invalidation_is_audited() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)
    registry.invalidate(artifact.artifact_id)

    assert any(
        event["event"] == "ARTIFACT_INVALIDATED"
        for event in registry.audit_log()
    )


def test_archive_is_audited() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)
    registry.archive(artifact.artifact_id)

    assert any(
        event["event"] == "ARTIFACT_ARCHIVED"
        for event in registry.audit_log()
    )


# ============================================================
# Snapshot
# ============================================================


def test_empty_snapshot() -> None:
    registry = ArtifactRegistry()

    snapshot = registry.snapshot()

    assert snapshot.total_artifacts == 0
    assert snapshot.created_artifacts == 0
    assert snapshot.verified_artifacts == 0
    assert snapshot.invalidated_artifacts == 0
    assert snapshot.archived_artifacts == 0
    assert snapshot.runs_with_artifacts == 0
    assert snapshot.experiments_with_artifacts == 0


def test_snapshot_status_counts() -> None:
    registry = ArtifactRegistry()

    make_artifact(
        registry,
        name="created",
        content={"id": 1},
    )

    verified = make_artifact(
        registry,
        name="verified",
        content={"id": 2},
    )
    registry.verify(verified.artifact_id)

    invalidated = make_artifact(
        registry,
        name="invalidated",
        content={"id": 3},
    )
    registry.invalidate(invalidated.artifact_id)

    archived = make_artifact(
        registry,
        name="archived",
        content={"id": 4},
    )
    registry.archive(archived.artifact_id)

    snapshot = registry.snapshot()

    assert snapshot.total_artifacts == 4
    assert snapshot.created_artifacts == 1
    assert snapshot.verified_artifacts == 1
    assert snapshot.invalidated_artifacts == 1
    assert snapshot.archived_artifacts == 1


def test_snapshot_type_counts() -> None:
    registry = ArtifactRegistry()

    make_artifact(
        registry,
        name="report",
        artifact_type=ArtifactType.RESEARCH_REPORT,
        content={"id": 1},
    )

    make_artifact(
        registry,
        name="metric",
        artifact_type=ArtifactType.METRIC,
        content={"id": 2},
    )

    snapshot = registry.snapshot()

    assert (
        snapshot.artifact_type_counts[
            "RESEARCH_REPORT"
        ]
        == 1
    )
    assert snapshot.artifact_type_counts["METRIC"] == 1


def test_snapshot_unique_runs() -> None:
    registry = ArtifactRegistry()

    make_artifact(
        registry,
        name="one",
        run_id="run_001",
        content={"id": 1},
    )

    make_artifact(
        registry,
        name="two",
        run_id="run_001",
        content={"id": 2},
    )

    make_artifact(
        registry,
        name="three",
        run_id="run_002",
        content={"id": 3},
    )

    assert registry.snapshot().runs_with_artifacts == 2


def test_snapshot_unique_experiments() -> None:
    registry = ArtifactRegistry()

    make_artifact(
        registry,
        name="one",
        experiment_id="experiment_001",
        content={"id": 1},
    )

    make_artifact(
        registry,
        name="two",
        experiment_id="experiment_002",
        content={"id": 2},
    )

    assert (
        registry.snapshot().experiments_with_artifacts
        == 2
    )


def test_snapshot_safety() -> None:
    registry = ArtifactRegistry()

    snapshot = registry.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Reports
# ============================================================


def test_artifact_report() -> None:
    registry = ArtifactRegistry()

    artifact = registry.create(
        name="quant_result",
        artifact_type=ArtifactType.QUANTUM_RESULT,
        content={"energy": -1.25},
        run_id="run_001",
        experiment_id="experiment_001",
        dataset_ids=["dataset_001"],
        version_ids=["model_v1"],
    )

    report = registry.artifact_report(
        artifact.artifact_id
    )

    assert report["artifact_id"] == artifact.artifact_id
    assert report["name"] == "quant_result"
    assert report["artifact_type"] == "QUANTUM_RESULT"
    assert report["status"] == "CREATED"
    assert report["run_id"] == "run_001"
    assert report["experiment_id"] == "experiment_001"
    assert report["dataset_ids"] == ["dataset_001"]
    assert report["version_ids"] == ["model_v1"]
    assert len(report["content_fingerprint"]) == 64
    assert len(report["lineage_fingerprint"]) == 64
    assert report["research_only"] is True
    assert report["real_money_execution"] is False


def test_summary_empty() -> None:
    registry = ArtifactRegistry()

    summary = registry.summary()

    assert summary["component"] == "artifact_registry"
    assert summary["version"] == "v6.0.1"
    assert summary["total_artifacts"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


def test_summary_after_artifacts() -> None:
    registry = ArtifactRegistry()

    artifact = make_artifact(registry)
    registry.verify(artifact.artifact_id)

    summary = registry.summary()

    assert summary["total_artifacts"] == 1
    assert summary["verified_artifacts"] == 1
    assert summary["runs_with_artifacts"] == 1
    assert summary["audit_events"] == 2


# ============================================================
# Institutional lineage scenario
# ============================================================


def test_full_research_artifact_lineage_flow() -> None:
    registry = ArtifactRegistry()

    backtest = registry.create(
        name="btc_regime_backtest",
        artifact_type=ArtifactType.BACKTEST_RESULT,
        content={
            "sharpe_ratio": 1.42,
            "max_drawdown": -0.11,
            "oos_score": 0.74,
        },
        run_id="run_btc_001",
        experiment_id="experiment_btc_001",
        dataset_ids=[
            "dataset_btc_prices_v3",
            "dataset_macro_v2",
        ],
        version_ids=[
            "model_crypto_agent_v4",
            "prompt_crypto_research_v7",
            "policy_risk_v2",
        ],
        references=[
            ArtifactReference(
                reference_type="hypothesis",
                reference_id="hypothesis_btc_001",
                relation="TESTS",
            ),
            ArtifactReference(
                reference_type="strategy",
                reference_id="strategy_btc_001",
                relation="EVALUATES",
            ),
        ],
    )

    assert registry.verify_content(
        backtest.artifact_id,
        {
            "oos_score": 0.74,
            "max_drawdown": -0.11,
            "sharpe_ratio": 1.42,
        },
    )

    registry.verify(backtest.artifact_id)

    report = registry.artifact_report(
        backtest.artifact_id
    )

    assert report["status"] == "VERIFIED"
    assert report["reference_count"] == 2

    assert registry.for_run(
        "run_btc_001"
    ) == [backtest]

    assert registry.for_experiment(
        "experiment_btc_001"
    ) == [backtest]

    assert registry.using_dataset(
        "dataset_macro_v2"
    ) == [backtest]

    assert registry.using_version(
        "policy_risk_v2"
    ) == [backtest]

    assert registry.references_to(
        "strategy_btc_001"
    ) == [backtest]

    snapshot = registry.snapshot()

    assert snapshot.total_artifacts == 1
    assert snapshot.verified_artifacts == 1