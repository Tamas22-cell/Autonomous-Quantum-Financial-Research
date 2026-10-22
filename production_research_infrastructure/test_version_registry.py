from __future__ import annotations

import pytest

from production_research_infrastructure.version_registry import (
    SemanticVersion,
    VersionAlreadyExistsError,
    VersionNotFoundError,
    VersionRecord,
    VersionRegistry,
    VersionRegistryError,
    VersionStatus,
    VersionType,
    VersionValidationError,
)


def make_model(
    *,
    name: str = "market_model",
    version: str = "1.0.0",
    content=None,
    status: VersionStatus = VersionStatus.DRAFT,
    parent_version_id: str | None = None,
) -> VersionRecord:
    if content is None:
        content = {
            "provider": "local",
            "model": "research-model",
            "temperature": 0.1,
        }

    return VersionRecord(
        name=name,
        version=version,
        version_type=VersionType.MODEL,
        content=content,
        status=status,
        parent_version_id=parent_version_id,
    )


# ============================================================
# SemanticVersion
# ============================================================


def test_semantic_version_creation() -> None:
    version = SemanticVersion(1, 2, 3)

    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3


def test_semantic_version_string() -> None:
    assert str(
        SemanticVersion(2, 4, 6)
    ) == "2.4.6"


def test_semantic_version_parse_full() -> None:
    version = SemanticVersion.parse("1.2.3")

    assert version == SemanticVersion(1, 2, 3)


def test_semantic_version_parse_with_v() -> None:
    version = SemanticVersion.parse("v3.1.4")

    assert version == SemanticVersion(3, 1, 4)


def test_semantic_version_parse_major_only() -> None:
    version = SemanticVersion.parse("5")

    assert version == SemanticVersion(5, 0, 0)


def test_semantic_version_parse_major_minor() -> None:
    version = SemanticVersion.parse("5.2")

    assert version == SemanticVersion(5, 2, 0)


def test_semantic_version_rejects_invalid_string() -> None:
    with pytest.raises(VersionValidationError):
        SemanticVersion.parse("1.2.3.4")


def test_semantic_version_rejects_letters() -> None:
    with pytest.raises(VersionValidationError):
        SemanticVersion.parse("one.two.three")


def test_semantic_version_requires_string() -> None:
    with pytest.raises(TypeError):
        SemanticVersion.parse(123)  # type: ignore[arg-type]


def test_semantic_version_rejects_negative_major() -> None:
    with pytest.raises(VersionValidationError):
        SemanticVersion(-1, 0, 0)


def test_semantic_version_rejects_negative_minor() -> None:
    with pytest.raises(VersionValidationError):
        SemanticVersion(1, -1, 0)


def test_semantic_version_rejects_negative_patch() -> None:
    with pytest.raises(VersionValidationError):
        SemanticVersion(1, 0, -1)


def test_semantic_version_rejects_bool() -> None:
    with pytest.raises(TypeError):
        SemanticVersion(True, 0, 0)  # type: ignore[arg-type]


# ============================================================
# VersionRecord
# ============================================================


def test_version_record_creation() -> None:
    record = make_model()

    assert record.name == "market_model"
    assert record.version == "1.0.0"
    assert record.version_type == VersionType.MODEL
    assert record.status == VersionStatus.DRAFT
    assert record.version_id.startswith("version_")


def test_version_record_normalizes_version() -> None:
    record = make_model(
        version="v2.3"
    )

    assert record.version == "2.3.0"


def test_version_record_trims_name() -> None:
    record = make_model(
        name="  macro_model  "
    )

    assert record.name == "macro_model"


def test_version_record_rejects_empty_name() -> None:
    with pytest.raises(VersionValidationError):
        make_model(name=" ")


def test_version_record_rejects_empty_version() -> None:
    with pytest.raises(VersionValidationError):
        make_model(version=" ")


def test_version_record_requires_version_type() -> None:
    with pytest.raises(TypeError):
        VersionRecord(
            name="model",
            version="1.0.0",
            version_type="MODEL",  # type: ignore[arg-type]
            content={},
        )


def test_version_record_requires_status_enum() -> None:
    with pytest.raises(TypeError):
        VersionRecord(
            name="model",
            version="1.0.0",
            version_type=VersionType.MODEL,
            content={},
            status="ACTIVE",  # type: ignore[arg-type]
        )


def test_version_record_rejects_empty_parent_id() -> None:
    with pytest.raises(VersionValidationError):
        make_model(
            parent_version_id=" "
        )


def test_version_record_research_safety() -> None:
    record = make_model()

    assert record.metadata["research_only"] is True
    assert (
        record.metadata["real_money_execution"]
        is False
    )


def test_version_record_fingerprint_stable() -> None:
    one = make_model()
    two = make_model()

    assert one.fingerprint == two.fingerprint


def test_version_record_fingerprint_changes_with_content() -> None:
    one = make_model(
        content={"temperature": 0.1}
    )
    two = make_model(
        content={"temperature": 0.2}
    )

    assert one.fingerprint != two.fingerprint


def test_version_record_fingerprint_changes_with_version() -> None:
    one = make_model(version="1.0.0")
    two = make_model(version="1.1.0")

    assert one.fingerprint != two.fingerprint


# ============================================================
# Registry basics
# ============================================================


def test_registry_starts_empty() -> None:
    registry = VersionRegistry()

    assert registry.count() == 0
    assert registry.list_all() == []
    assert registry.audit_log() == []


def test_registry_version() -> None:
    assert VersionRegistry.VERSION == "v6.0.1"


def test_register_record() -> None:
    registry = VersionRegistry()
    record = make_model()

    result = registry.register(record)

    assert result is record
    assert registry.count() == 1


def test_register_requires_record() -> None:
    registry = VersionRegistry()

    with pytest.raises(TypeError):
        registry.register("invalid")  # type: ignore[arg-type]


def test_duplicate_name_type_version_rejected() -> None:
    registry = VersionRegistry()

    registry.register(
        make_model(version="1.0.0")
    )

    with pytest.raises(
        VersionAlreadyExistsError
    ):
        registry.register(
            make_model(version="1.0.0")
        )


def test_duplicate_version_id_rejected() -> None:
    registry = VersionRegistry()

    first = make_model()
    registry.register(first)

    second = VersionRecord(
        name="other_model",
        version="1.0.0",
        version_type=VersionType.MODEL,
        content={},
        version_id=first.version_id,
    )

    with pytest.raises(
        VersionAlreadyExistsError
    ):
        registry.register(second)


# ============================================================
# Convenience registration
# ============================================================


def test_register_model() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0",
        content={"model": "local"},
    )

    assert record.version_type == VersionType.MODEL
    assert record.version == "1.0.0"


def test_register_prompt() -> None:
    registry = VersionRegistry()

    record = registry.register_prompt(
        name="macro_prompt",
        version="1.0.0",
        content="Analyze macro conditions.",
    )

    assert record.version_type == VersionType.PROMPT


def test_register_research_policy() -> None:
    registry = VersionRegistry()

    record = registry.register_research_policy(
        name="risk_policy",
        version="1.0.0",
        content={
            "max_drawdown": 0.10,
        },
    )

    assert (
        record.version_type
        == VersionType.RESEARCH_POLICY
    )


def test_convenience_registration_metadata() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        metadata={"provider": "local"},
    )

    assert record.metadata["provider"] == "local"
    assert record.metadata["research_only"] is True


# ============================================================
# Retrieval
# ============================================================


def test_get_record_by_id() -> None:
    registry = VersionRegistry()

    record = registry.register(
        make_model()
    )

    assert registry.get(
        record.version_id
    ) is record


def test_get_missing_record_raises() -> None:
    registry = VersionRegistry()

    with pytest.raises(VersionNotFoundError):
        registry.get("version_missing")


def test_get_rejects_empty_id() -> None:
    registry = VersionRegistry()

    with pytest.raises(VersionValidationError):
        registry.get(" ")


def test_get_version() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.2.0",
        content={},
    )

    result = registry.get_version(
        VersionType.MODEL,
        "llm",
        "1.2",
    )

    assert result is record


def test_get_missing_version_raises() -> None:
    registry = VersionRegistry()

    with pytest.raises(VersionNotFoundError):
        registry.get_version(
            VersionType.MODEL,
            "llm",
            "9.0.0",
        )


def test_get_version_requires_type_enum() -> None:
    registry = VersionRegistry()

    with pytest.raises(TypeError):
        registry.get_version(
            "MODEL",  # type: ignore[arg-type]
            "llm",
            "1.0.0",
        )


def test_exists_true() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    assert registry.exists(
        VersionType.MODEL,
        "llm",
        "1.0",
    )


def test_exists_false() -> None:
    registry = VersionRegistry()

    assert not registry.exists(
        VersionType.MODEL,
        "llm",
        "1.0.0",
    )


# ============================================================
# Active version management
# ============================================================


def test_register_active_version() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    assert registry.get_active(
        VersionType.MODEL,
        "llm",
    ) is record


def test_activate_version() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    registry.activate(
        record.version_id
    )

    assert record.status == VersionStatus.ACTIVE


def test_new_active_deprecates_previous_active() -> None:
    registry = VersionRegistry()

    first = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    second = registry.register_model(
        name="llm",
        version="2.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    assert first.status == VersionStatus.DEPRECATED
    assert second.status == VersionStatus.ACTIVE
    assert registry.get_active(
        VersionType.MODEL,
        "llm",
    ) is second


def test_activate_existing_version_deprecates_previous() -> None:
    registry = VersionRegistry()

    first = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    second = registry.register_model(
        name="llm",
        version="2.0.0",
        content={},
    )

    registry.activate(second.version_id)

    assert first.status == VersionStatus.DEPRECATED
    assert second.status == VersionStatus.ACTIVE


def test_deprecate_active_version() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    registry.deprecate(
        record.version_id
    )

    assert record.status == VersionStatus.DEPRECATED
    assert registry.get_active(
        VersionType.MODEL,
        "llm",
    ) is None


def test_archive_active_version() -> None:
    registry = VersionRegistry()

    record = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    registry.archive(
        record.version_id
    )

    assert record.status == VersionStatus.ARCHIVED
    assert registry.get_active(
        VersionType.MODEL,
        "llm",
    ) is None


def test_get_active_missing_returns_none() -> None:
    registry = VersionRegistry()

    assert registry.get_active(
        VersionType.MODEL,
        "missing",
    ) is None


# ============================================================
# Listing
# ============================================================


def test_list_all_preserves_registration_order() -> None:
    registry = VersionRegistry()

    first = registry.register_model(
        name="one",
        version="1.0.0",
        content={},
    )

    second = registry.register_prompt(
        name="two",
        version="1.0.0",
        content="prompt",
    )

    assert registry.list_all() == [
        first,
        second,
    ]


def test_list_by_type() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="model",
        version="1.0.0",
        content={},
    )

    registry.register_prompt(
        name="prompt",
        version="1.0.0",
        content="text",
    )

    models = registry.list_by_type(
        VersionType.MODEL
    )

    assert len(models) == 1
    assert models[0].name == "model"


def test_list_versions_semantic_order() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="llm",
        version="2.0.0",
        content={},
    )

    registry.register_model(
        name="llm",
        version="1.10.0",
        content={},
    )

    registry.register_model(
        name="llm",
        version="1.2.0",
        content={},
    )

    versions = registry.list_versions(
        VersionType.MODEL,
        "llm",
    )

    assert [
        record.version
        for record in versions
    ] == [
        "1.2.0",
        "1.10.0",
        "2.0.0",
    ]


# ============================================================
# Parent / lineage
# ============================================================


def test_register_child_version() -> None:
    registry = VersionRegistry()

    parent = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    child = registry.register_model(
        name="llm",
        version="1.1.0",
        content={},
        parent_version_id=parent.version_id,
    )

    assert (
        child.parent_version_id
        == parent.version_id
    )


def test_missing_parent_rejected() -> None:
    registry = VersionRegistry()

    with pytest.raises(VersionNotFoundError):
        registry.register_model(
            name="llm",
            version="1.1.0",
            content={},
            parent_version_id="missing",
        )


def test_parent_must_match_name() -> None:
    registry = VersionRegistry()

    parent = registry.register_model(
        name="model_a",
        version="1.0.0",
        content={},
    )

    with pytest.raises(VersionValidationError):
        registry.register_model(
            name="model_b",
            version="1.1.0",
            content={},
            parent_version_id=parent.version_id,
        )


def test_parent_must_match_type() -> None:
    registry = VersionRegistry()

    parent = registry.register_prompt(
        name="shared",
        version="1.0.0",
        content="prompt",
    )

    with pytest.raises(VersionValidationError):
        registry.register_model(
            name="shared",
            version="1.1.0",
            content={},
            parent_version_id=parent.version_id,
        )


def test_parent_of() -> None:
    registry = VersionRegistry()

    parent = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    child = registry.register_model(
        name="llm",
        version="1.1.0",
        content={},
        parent_version_id=parent.version_id,
    )

    assert registry.parent_of(
        child.version_id
    ) is parent


def test_parent_of_root_returns_none() -> None:
    registry = VersionRegistry()

    root = registry.register(
        make_model()
    )

    assert registry.parent_of(
        root.version_id
    ) is None


def test_children_of() -> None:
    registry = VersionRegistry()

    parent = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    child_one = registry.register_model(
        name="llm",
        version="1.1.0",
        content={},
        parent_version_id=parent.version_id,
    )

    child_two = registry.register_model(
        name="llm",
        version="1.2.0",
        content={},
        parent_version_id=parent.version_id,
    )

    assert registry.children_of(
        parent.version_id
    ) == [
        child_one,
        child_two,
    ]


def test_ancestry() -> None:
    registry = VersionRegistry()

    v1 = registry.register_model(
        name="llm",
        version="1.0.0",
        content={},
    )

    v2 = registry.register_model(
        name="llm",
        version="1.1.0",
        content={},
        parent_version_id=v1.version_id,
    )

    v3 = registry.register_model(
        name="llm",
        version="1.2.0",
        content={},
        parent_version_id=v2.version_id,
    )

    assert registry.ancestry(
        v3.version_id
    ) == [
        v2,
        v1,
    ]


# ============================================================
# Configuration fingerprints
# ============================================================


def test_configuration_fingerprint_stable() -> None:
    registry = VersionRegistry()

    model = registry.register_model(
        name="llm",
        version="1.0.0",
        content={"model": "research"},
    )

    prompt = registry.register_prompt(
        name="macro",
        version="1.0.0",
        content="Analyze market.",
    )

    first = registry.configuration_fingerprint(
        model_version_ids=[
            model.version_id
        ],
        prompt_version_ids=[
            prompt.version_id
        ],
    )

    second = registry.configuration_fingerprint(
        model_version_ids=[
            model.version_id
        ],
        prompt_version_ids=[
            prompt.version_id
        ],
    )

    assert first == second


def test_configuration_fingerprint_order_independent() -> None:
    registry = VersionRegistry()

    first_model = registry.register_model(
        name="model_a",
        version="1.0.0",
        content={"a": 1},
    )

    second_model = registry.register_model(
        name="model_b",
        version="1.0.0",
        content={"b": 2},
    )

    one = registry.configuration_fingerprint(
        model_version_ids=[
            first_model.version_id,
            second_model.version_id,
        ]
    )

    two = registry.configuration_fingerprint(
        model_version_ids=[
            second_model.version_id,
            first_model.version_id,
        ]
    )

    assert one == two


def test_configuration_fingerprint_changes_with_model() -> None:
    registry = VersionRegistry()

    first = registry.register_model(
        name="llm",
        version="1.0.0",
        content={"temperature": 0.1},
    )

    second = registry.register_model(
        name="llm",
        version="2.0.0",
        content={"temperature": 0.2},
    )

    first_hash = (
        registry.configuration_fingerprint(
            model_version_ids=[
                first.version_id
            ]
        )
    )

    second_hash = (
        registry.configuration_fingerprint(
            model_version_ids=[
                second.version_id
            ]
        )
    )

    assert first_hash != second_hash


def test_model_slot_rejects_prompt_version() -> None:
    registry = VersionRegistry()

    prompt = registry.register_prompt(
        name="prompt",
        version="1.0.0",
        content="text",
    )

    with pytest.raises(VersionValidationError):
        registry.configuration_fingerprint(
            model_version_ids=[
                prompt.version_id
            ]
        )


def test_prompt_slot_rejects_model_version() -> None:
    registry = VersionRegistry()

    model = registry.register_model(
        name="model",
        version="1.0.0",
        content={},
    )

    with pytest.raises(VersionValidationError):
        registry.configuration_fingerprint(
            prompt_version_ids=[
                model.version_id
            ]
        )


def test_policy_slot_rejects_model_version() -> None:
    registry = VersionRegistry()

    model = registry.register_model(
        name="model",
        version="1.0.0",
        content={},
    )

    with pytest.raises(VersionValidationError):
        registry.configuration_fingerprint(
            policy_version_ids=[
                model.version_id
            ]
        )


# ============================================================
# Audit
# ============================================================


def test_registration_creates_audit_event() -> None:
    registry = VersionRegistry()

    record = registry.register(
        make_model()
    )

    audit = registry.audit_log()

    assert audit[0]["event"] == (
        "VERSION_REGISTERED"
    )

    assert (
        audit[0]["details"]["version_id"]
        == record.version_id
    )


def test_activation_creates_audit_event() -> None:
    registry = VersionRegistry()

    record = registry.register(
        make_model()
    )

    registry.activate(
        record.version_id
    )

    assert any(
        item["event"] == "VERSION_ACTIVATED"
        for item in registry.audit_log()
    )


def test_deprecation_creates_audit_event() -> None:
    registry = VersionRegistry()

    record = registry.register(
        make_model()
    )

    registry.deprecate(
        record.version_id
    )

    assert any(
        item["event"] == "VERSION_DEPRECATED"
        for item in registry.audit_log()
    )


def test_archive_creates_audit_event() -> None:
    registry = VersionRegistry()

    record = registry.register(
        make_model()
    )

    registry.archive(
        record.version_id
    )

    assert any(
        item["event"] == "VERSION_ARCHIVED"
        for item in registry.audit_log()
    )


# ============================================================
# Snapshot / summary
# ============================================================


def test_empty_snapshot() -> None:
    registry = VersionRegistry()

    snapshot = registry.snapshot()

    assert snapshot.total_versions == 0
    assert snapshot.model_versions == 0
    assert snapshot.prompt_versions == 0
    assert (
        snapshot.research_policy_versions
        == 0
    )
    assert snapshot.active_versions == 0


def test_snapshot_counts_types() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="model",
        version="1.0.0",
        content={},
    )

    registry.register_prompt(
        name="prompt",
        version="1.0.0",
        content="text",
    )

    registry.register_research_policy(
        name="policy",
        version="1.0.0",
        content={},
    )

    snapshot = registry.snapshot()

    assert snapshot.total_versions == 3
    assert snapshot.model_versions == 1
    assert snapshot.prompt_versions == 1
    assert (
        snapshot.research_policy_versions
        == 1
    )


def test_snapshot_counts_active_versions() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="model",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    registry.register_prompt(
        name="prompt",
        version="1.0.0",
        content="text",
        status=VersionStatus.ACTIVE,
    )

    snapshot = registry.snapshot()

    assert snapshot.active_versions == 2


def test_snapshot_research_safety() -> None:
    registry = VersionRegistry()

    snapshot = registry.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


def test_summary() -> None:
    registry = VersionRegistry()

    registry.register_model(
        name="model",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    summary = registry.summary()

    assert summary["component"] == (
        "version_registry"
    )
    assert summary["version"] == "v6.0.1"
    assert summary["total_versions"] == 1
    assert summary["model_versions"] == 1
    assert summary["active_versions"] == 1
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


# ============================================================
# Isolation between version types
# ============================================================


def test_same_name_and_version_allowed_for_different_types() -> None:
    registry = VersionRegistry()

    model = registry.register_model(
        name="shared",
        version="1.0.0",
        content={"model": True},
    )

    prompt = registry.register_prompt(
        name="shared",
        version="1.0.0",
        content="prompt",
    )

    policy = registry.register_research_policy(
        name="shared",
        version="1.0.0",
        content={"policy": True},
    )

    assert registry.count() == 3

    assert (
        model.version_type
        == VersionType.MODEL
    )

    assert (
        prompt.version_type
        == VersionType.PROMPT
    )

    assert (
        policy.version_type
        == VersionType.RESEARCH_POLICY
    )


def test_active_versions_are_isolated_by_type() -> None:
    registry = VersionRegistry()

    model = registry.register_model(
        name="shared",
        version="1.0.0",
        content={},
        status=VersionStatus.ACTIVE,
    )

    prompt = registry.register_prompt(
        name="shared",
        version="1.0.0",
        content="prompt",
        status=VersionStatus.ACTIVE,
    )

    assert registry.get_active(
        VersionType.MODEL,
        "shared",
    ) is model

    assert registry.get_active(
        VersionType.PROMPT,
        "shared",
    ) is prompt