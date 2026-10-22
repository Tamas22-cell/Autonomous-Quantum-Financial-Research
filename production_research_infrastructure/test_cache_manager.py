from __future__ import annotations

from datetime import timedelta

import pytest

from production_research_infrastructure.cache_manager import (
    CacheEntry,
    CacheEntryNotFoundError,
    CacheKey,
    CacheManager,
    CacheManagerError,
    CacheStatus,
    CacheValidationError,
    utc_now,
)


def make_key(
    *,
    namespace: str = "research",
    payload=None,
    version: str | None = "v1",
) -> CacheKey:
    if payload is None:
        payload = {
            "asset": "BTC",
            "horizon": "1D",
        }

    return CacheKey(
        namespace=namespace,
        payload=payload,
        version=version,
    )


# ============================================================
# CacheKey
# ============================================================


def test_cache_key_creation() -> None:
    key = make_key()

    assert key.namespace == "research"
    assert key.version == "v1"


def test_cache_key_trims_namespace() -> None:
    key = make_key(
        namespace="  research  "
    )

    assert key.namespace == "research"


def test_cache_key_rejects_empty_namespace() -> None:
    with pytest.raises(CacheValidationError):
        make_key(namespace=" ")


def test_cache_key_trims_version() -> None:
    key = make_key(
        version="  v2  "
    )

    assert key.version == "v2"


def test_cache_key_accepts_none_version() -> None:
    key = make_key(
        version=None
    )

    assert key.version is None


def test_cache_key_rejects_empty_version() -> None:
    with pytest.raises(CacheValidationError):
        make_key(version=" ")


def test_cache_key_research_safety() -> None:
    key = make_key()

    assert key.metadata["research_only"] is True
    assert (
        key.metadata["real_money_execution"]
        is False
    )


def test_cache_key_fingerprint_stable() -> None:
    first = make_key()
    second = make_key()

    assert first.fingerprint == second.fingerprint


def test_cache_key_id_stable() -> None:
    first = make_key()
    second = make_key()

    assert first.cache_id == second.cache_id


def test_cache_id_prefix() -> None:
    key = make_key()

    assert key.cache_id.startswith("cache_")


def test_payload_dict_order_does_not_change_key() -> None:
    first = make_key(
        payload={
            "asset": "BTC",
            "horizon": "1D",
        }
    )

    second = make_key(
        payload={
            "horizon": "1D",
            "asset": "BTC",
        }
    )

    assert first.cache_id == second.cache_id


def test_payload_change_changes_key() -> None:
    first = make_key(
        payload={"asset": "BTC"}
    )

    second = make_key(
        payload={"asset": "ETH"}
    )

    assert first.cache_id != second.cache_id


def test_namespace_change_changes_key() -> None:
    first = make_key(
        namespace="macro"
    )

    second = make_key(
        namespace="crypto"
    )

    assert first.cache_id != second.cache_id


def test_version_change_changes_key() -> None:
    first = make_key(
        version="v1"
    )

    second = make_key(
        version="v2"
    )

    assert first.cache_id != second.cache_id


# ============================================================
# CacheEntry
# ============================================================


def test_cache_entry_creation() -> None:
    key = make_key()

    entry = CacheEntry(
        key=key,
        value={"signal": "HOLD"},
    )

    assert entry.key is key
    assert entry.status == CacheStatus.ACTIVE
    assert entry.hit_count == 0


def test_cache_entry_requires_cache_key() -> None:
    with pytest.raises(TypeError):
        CacheEntry(
            key="invalid",
            value={},
        )


def test_cache_entry_requires_status_enum() -> None:
    with pytest.raises(TypeError):
        CacheEntry(
            key=make_key(),
            value={},
            status="ACTIVE",
        )


def test_cache_entry_accepts_integer_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
        ttl_seconds=60,
    )

    assert entry.ttl_seconds == 60.0


def test_cache_entry_accepts_float_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
        ttl_seconds=1.5,
    )

    assert entry.ttl_seconds == 1.5


def test_cache_entry_accepts_none_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
    )

    assert entry.ttl_seconds is None


def test_cache_entry_rejects_zero_ttl() -> None:
    with pytest.raises(CacheValidationError):
        CacheEntry(
            key=make_key(),
            value={},
            ttl_seconds=0,
        )


def test_cache_entry_rejects_negative_ttl() -> None:
    with pytest.raises(CacheValidationError):
        CacheEntry(
            key=make_key(),
            value={},
            ttl_seconds=-1,
        )


def test_cache_entry_rejects_boolean_ttl() -> None:
    with pytest.raises(TypeError):
        CacheEntry(
            key=make_key(),
            value={},
            ttl_seconds=True,
        )


def test_cache_entry_rejects_negative_hit_count() -> None:
    with pytest.raises(CacheValidationError):
        CacheEntry(
            key=make_key(),
            value={},
            hit_count=-1,
        )


def test_cache_entry_rejects_boolean_hit_count() -> None:
    with pytest.raises(TypeError):
        CacheEntry(
            key=make_key(),
            value={},
            hit_count=True,
        )


def test_cache_entry_value_fingerprint_stable() -> None:
    first = CacheEntry(
        key=make_key(),
        value={
            "signal": "HOLD",
            "score": 0.5,
        },
    )

    second = CacheEntry(
        key=make_key(),
        value={
            "score": 0.5,
            "signal": "HOLD",
        },
    )

    assert (
        first.value_fingerprint
        == second.value_fingerprint
    )


def test_cache_entry_expires_at_none_without_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
    )

    assert entry.expires_at is None


def test_cache_entry_expires_at_with_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
        ttl_seconds=60,
    )

    assert entry.expires_at is not None
    assert (
        entry.expires_at
        == entry.created_at
        + timedelta(seconds=60)
    )


def test_cache_entry_not_expired_before_ttl() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
        ttl_seconds=60,
    )

    assert not entry.is_expired(
        now=entry.created_at
        + timedelta(seconds=59)
    )


def test_cache_entry_expired_at_ttl_boundary() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
        ttl_seconds=60,
    )

    assert entry.is_expired(
        now=entry.created_at
        + timedelta(seconds=60)
    )


def test_cache_entry_without_ttl_never_expires() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
    )

    assert not entry.is_expired(
        now=utc_now()
        + timedelta(days=1000)
    )


def test_cache_entry_research_safety() -> None:
    entry = CacheEntry(
        key=make_key(),
        value={},
    )

    assert entry.metadata["research_only"] is True
    assert (
        entry.metadata["real_money_execution"]
        is False
    )


# ============================================================
# CacheManager basics
# ============================================================


def test_manager_version() -> None:
    assert CacheManager.VERSION == "v6.0.1"


def test_manager_starts_empty() -> None:
    manager = CacheManager()

    assert manager.count() == 0
    assert manager.entries() == []
    assert manager.audit_log() == []


def test_create_key_helper() -> None:
    key = CacheManager.create_key(
        "macro",
        {"asset": "SPY"},
        version="v2",
    )

    assert isinstance(key, CacheKey)
    assert key.namespace == "macro"
    assert key.version == "v2"


def test_generate_cache_id() -> None:
    key = make_key()

    assert (
        CacheManager.generate_cache_id(key)
        == key.cache_id
    )


def test_generate_cache_id_requires_key() -> None:
    with pytest.raises(TypeError):
        CacheManager.generate_cache_id(
            "invalid"
        )


# ============================================================
# Set
# ============================================================


def test_set_entry() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"signal": "BUY"},
    )

    assert entry.key is key
    assert entry.value == {"signal": "BUY"}
    assert manager.count() == 1


def test_set_with_ttl() -> None:
    manager = CacheManager()

    entry = manager.set(
        make_key(),
        {"result": 1},
        ttl_seconds=120,
    )

    assert entry.ttl_seconds == 120.0


def test_set_custom_metadata() -> None:
    manager = CacheManager()

    entry = manager.set(
        make_key(),
        {"result": 1},
        metadata={
            "source": "agent"
        },
    )

    assert entry.metadata["source"] == "agent"
    assert entry.metadata["research_only"] is True


def test_set_requires_key() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.set(
            "invalid",
            {},
        )


def test_set_requires_boolean_overwrite() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.set(
            make_key(),
            {},
            overwrite="yes",
        )


def test_set_overwrites_existing_entry() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"value": 1},
    )

    second = manager.set(
        key,
        {"value": 2},
    )

    assert manager.count() == 1
    assert second.value == {"value": 2}


def test_set_no_overwrite_rejects_active_entry() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"value": 1},
    )

    with pytest.raises(CacheManagerError):
        manager.set(
            key,
            {"value": 2},
            overwrite=False,
        )


def test_set_after_invalidation_allowed_without_overwrite() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"value": 1},
    )

    manager.invalidate(key)

    entry = manager.set(
        key,
        {"value": 2},
        overwrite=False,
    )

    assert entry.status == CacheStatus.ACTIVE
    assert entry.value == {"value": 2}


# ============================================================
# Lookup
# ============================================================


def test_lookup_hit() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"signal": "HOLD"},
    )

    result = manager.lookup(key)

    assert result.hit is True
    assert result.value == {"signal": "HOLD"}
    assert result.status == CacheStatus.ACTIVE


def test_lookup_hit_increments_hit_count() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"result": 1},
    )

    manager.lookup(key)
    manager.lookup(key)

    assert entry.hit_count == 2


def test_lookup_updates_last_accessed() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"result": 1},
    )

    assert entry.last_accessed_at is None

    manager.lookup(key)

    assert entry.last_accessed_at is not None


def test_lookup_missing_is_miss() -> None:
    manager = CacheManager()
    key = make_key()

    result = manager.lookup(key)

    assert result.hit is False
    assert result.value is None


def test_lookup_requires_key() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.lookup(
            "invalid"
        )


def test_get_returns_cached_value() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        123,
    )

    assert manager.get(key) == 123


def test_get_returns_default_on_miss() -> None:
    manager = CacheManager()

    assert manager.get(
        make_key(),
        default="fallback",
    ) == "fallback"


def test_get_entry() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
    )

    assert (
        manager.get_entry(key.cache_id)
        is entry
    )


def test_get_entry_missing_raises() -> None:
    manager = CacheManager()

    with pytest.raises(
        CacheEntryNotFoundError
    ):
        manager.get_entry(
            "cache_missing"
        )


def test_get_entry_rejects_empty_id() -> None:
    manager = CacheManager()

    with pytest.raises(CacheValidationError):
        manager.get_entry(" ")


def test_exists_true() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"value": 1},
    )

    assert manager.exists(key)


def test_exists_false_missing() -> None:
    manager = CacheManager()

    assert not manager.exists(
        make_key()
    )


def test_exists_requires_key() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.exists(
            "invalid"
        )


# ============================================================
# Expiration
# ============================================================


def test_expired_entry_becomes_miss() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
        ttl_seconds=1,
    )

    entry.created_at = (
        utc_now()
        - timedelta(seconds=10)
    )

    result = manager.lookup(key)

    assert result.hit is False
    assert result.status == CacheStatus.EXPIRED


def test_expired_entry_status_updated() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
        ttl_seconds=1,
    )

    entry.created_at = (
        utc_now()
        - timedelta(seconds=10)
    )

    manager.get_entry(
        key.cache_id
    )

    assert entry.status == CacheStatus.EXPIRED


def test_expired_entry_does_not_exist_as_active() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
        ttl_seconds=1,
    )

    entry.created_at = (
        utc_now()
        - timedelta(seconds=10)
    )

    assert not manager.exists(key)


def test_expiration_creates_audit_event() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {},
        ttl_seconds=1,
    )

    entry.created_at = (
        utc_now()
        - timedelta(seconds=10)
    )

    manager.lookup(key)

    assert any(
        event["event"] == "CACHE_EXPIRED"
        for event in manager.audit_log()
    )


# ============================================================
# Invalidation
# ============================================================


def test_invalidate_entry() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
    )

    manager.invalidate(
        key,
        reason="dataset updated",
    )

    assert (
        entry.status
        == CacheStatus.INVALIDATED
    )

    assert (
        entry.invalidation_reason
        == "dataset updated"
    )

    assert entry.invalidated_at is not None


def test_invalidated_entry_lookup_is_miss() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(
        key,
        {"value": 1},
    )

    manager.invalidate(key)

    result = manager.lookup(key)

    assert result.hit is False
    assert (
        result.status
        == CacheStatus.INVALIDATED
    )


def test_invalidate_requires_key() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.invalidate(
            "invalid"
        )


def test_invalidate_missing_raises() -> None:
    manager = CacheManager()

    with pytest.raises(
        CacheEntryNotFoundError
    ):
        manager.invalidate(
            make_key()
        )


def test_invalidate_namespace() -> None:
    manager = CacheManager()

    first = make_key(
        namespace="macro",
        payload={"id": 1},
    )

    second = make_key(
        namespace="macro",
        payload={"id": 2},
    )

    third = make_key(
        namespace="crypto",
        payload={"id": 3},
    )

    manager.set(first, 1)
    manager.set(second, 2)
    manager.set(third, 3)

    count = manager.invalidate_namespace(
        "macro",
        reason="refresh",
    )

    assert count == 2
    assert not manager.exists(first)
    assert not manager.exists(second)
    assert manager.exists(third)


def test_invalidate_namespace_rejects_empty() -> None:
    manager = CacheManager()

    with pytest.raises(CacheValidationError):
        manager.invalidate_namespace(" ")


# ============================================================
# Listing
# ============================================================


def test_entries_preserve_insertion_order() -> None:
    manager = CacheManager()

    first_key = make_key(
        payload={"id": 1}
    )

    second_key = make_key(
        payload={"id": 2}
    )

    first = manager.set(
        first_key,
        1,
    )

    second = manager.set(
        second_key,
        2,
    )

    assert manager.entries() == [
        first,
        second,
    ]


def test_entries_by_namespace() -> None:
    manager = CacheManager()

    manager.set(
        make_key(
            namespace="macro",
            payload={"id": 1},
        ),
        1,
    )

    manager.set(
        make_key(
            namespace="crypto",
            payload={"id": 2},
        ),
        2,
    )

    result = manager.entries_by_namespace(
        "macro"
    )

    assert len(result) == 1
    assert (
        result[0].key.namespace
        == "macro"
    )


def test_entries_by_namespace_rejects_empty() -> None:
    manager = CacheManager()

    with pytest.raises(CacheValidationError):
        manager.entries_by_namespace(
            " "
        )


# ============================================================
# Delete / clear
# ============================================================


def test_delete_entry() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        {"value": 1},
    )

    deleted = manager.delete(key)

    assert deleted is entry
    assert manager.count() == 0


def test_delete_missing_raises() -> None:
    manager = CacheManager()

    with pytest.raises(
        CacheEntryNotFoundError
    ):
        manager.delete(
            make_key()
        )


def test_delete_requires_key() -> None:
    manager = CacheManager()

    with pytest.raises(TypeError):
        manager.delete(
            "invalid"
        )


def test_clear() -> None:
    manager = CacheManager()

    manager.set(
        make_key(
            payload={"id": 1}
        ),
        1,
    )

    manager.set(
        make_key(
            payload={"id": 2}
        ),
        2,
    )

    deleted = manager.clear()

    assert deleted == 2
    assert manager.count() == 0
    assert manager.entries() == []


def test_clear_empty_returns_zero() -> None:
    manager = CacheManager()

    assert manager.clear() == 0


# ============================================================
# Audit
# ============================================================


def test_set_creates_audit_event() -> None:
    manager = CacheManager()

    manager.set(
        make_key(),
        {"value": 1},
    )

    assert (
        manager.audit_log()[0]["event"]
        == "CACHE_SET"
    )


def test_hit_creates_audit_event() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)
    manager.lookup(key)

    assert any(
        event["event"] == "CACHE_HIT"
        for event in manager.audit_log()
    )


def test_miss_creates_audit_event() -> None:
    manager = CacheManager()

    manager.lookup(
        make_key()
    )

    assert any(
        event["event"] == "CACHE_MISS"
        for event in manager.audit_log()
    )


def test_invalidation_creates_audit_event() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)
    manager.invalidate(key)

    assert any(
        event["event"]
        == "CACHE_INVALIDATED"
        for event in manager.audit_log()
    )


def test_delete_creates_audit_event() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)
    manager.delete(key)

    assert any(
        event["event"] == "CACHE_DELETED"
        for event in manager.audit_log()
    )


# ============================================================
# Snapshot
# ============================================================


def test_empty_snapshot() -> None:
    manager = CacheManager()

    snapshot = manager.snapshot()

    assert snapshot.total_entries == 0
    assert snapshot.active_entries == 0
    assert snapshot.expired_entries == 0
    assert snapshot.invalidated_entries == 0
    assert snapshot.total_hits == 0
    assert snapshot.total_misses == 0


def test_snapshot_counts_active_entries() -> None:
    manager = CacheManager()

    manager.set(
        make_key(
            payload={"id": 1}
        ),
        1,
    )

    manager.set(
        make_key(
            payload={"id": 2}
        ),
        2,
    )

    snapshot = manager.snapshot()

    assert snapshot.total_entries == 2
    assert snapshot.active_entries == 2


def test_snapshot_counts_invalidated_entries() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)
    manager.invalidate(key)

    snapshot = manager.snapshot()

    assert snapshot.invalidated_entries == 1
    assert snapshot.active_entries == 0


def test_snapshot_counts_expired_entries() -> None:
    manager = CacheManager()
    key = make_key()

    entry = manager.set(
        key,
        1,
        ttl_seconds=1,
    )

    entry.created_at = (
        utc_now()
        - timedelta(seconds=10)
    )

    snapshot = manager.snapshot()

    assert snapshot.expired_entries == 1
    assert snapshot.active_entries == 0


def test_snapshot_counts_hits_and_misses() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)

    manager.lookup(key)
    manager.lookup(key)

    manager.lookup(
        make_key(
            payload={"missing": True}
        )
    )

    snapshot = manager.snapshot()

    assert snapshot.total_hits == 2
    assert snapshot.total_misses == 1


def test_snapshot_research_safety() -> None:
    manager = CacheManager()

    snapshot = manager.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Summary
# ============================================================


def test_summary_empty() -> None:
    manager = CacheManager()

    summary = manager.summary()

    assert summary["component"] == "cache_manager"
    assert summary["version"] == "v6.0.1"
    assert summary["total_entries"] == 0
    assert summary["hit_rate"] == 0.0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_hit_rate() -> None:
    manager = CacheManager()
    key = make_key()

    manager.set(key, 1)

    manager.lookup(key)

    manager.lookup(
        make_key(
            payload={"missing": 1}
        )
    )

    summary = manager.summary()

    assert summary["total_hits"] == 1
    assert summary["total_misses"] == 1
    assert summary["hit_rate"] == 0.5


# ============================================================
# Research reproducibility
# ============================================================


def test_same_research_input_reuses_cache() -> None:
    manager = CacheManager()

    first_key = CacheManager.create_key(
        "agent_research",
        {
            "run_id": "run_123",
            "agent": "macro",
            "dataset": "dataset_001",
        },
        version="v6.0.1",
    )

    manager.set(
        first_key,
        {
            "signal": "HOLD",
            "confidence": 0.75,
        },
    )

    second_key = CacheManager.create_key(
        "agent_research",
        {
            "dataset": "dataset_001",
            "agent": "macro",
            "run_id": "run_123",
        },
        version="v6.0.1",
    )

    result = manager.lookup(
        second_key
    )

    assert result.hit is True

    assert result.value == {
        "signal": "HOLD",
        "confidence": 0.75,
    }


def test_different_dataset_prevents_cache_reuse() -> None:
    manager = CacheManager()

    first = CacheManager.create_key(
        "agent_research",
        {
            "agent": "macro",
            "dataset": "dataset_v1",
        },
    )

    second = CacheManager.create_key(
        "agent_research",
        {
            "agent": "macro",
            "dataset": "dataset_v2",
        },
    )

    manager.set(
        first,
        {"signal": "BUY"},
    )

    assert manager.lookup(
        second
    ).hit is False


def test_different_model_version_prevents_cache_reuse() -> None:
    manager = CacheManager()

    first = CacheManager.create_key(
        "agent_research",
        {
            "agent": "macro"
        },
        version="model_v1",
    )

    second = CacheManager.create_key(
        "agent_research",
        {
            "agent": "macro"
        },
        version="model_v2",
    )

    manager.set(
        first,
        {"signal": "BUY"},
    )

    assert manager.lookup(
        second
    ).hit is False