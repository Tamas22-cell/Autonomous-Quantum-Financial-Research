from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


class CacheManagerError(RuntimeError):
    """Base exception for cache-management failures."""


class CacheValidationError(CacheManagerError):
    """Raised when cache input is invalid."""


class CacheEntryNotFoundError(CacheManagerError):
    """Raised when a requested cache entry cannot be found."""


class CacheStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


@dataclass
class CacheKey:
    namespace: str
    payload: Any
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace.strip()
        ):
            raise CacheValidationError(
                "namespace must be a non-empty string"
            )

        self.namespace = self.namespace.strip()

        if self.version is not None:
            if (
                not isinstance(self.version, str)
                or not self.version.strip()
            ):
                raise CacheValidationError(
                    "version must be a non-empty string"
                )

            self.version = self.version.strip()

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "namespace": self.namespace,
                "version": self.version,
                "payload": self.payload,
            }
        )

    @property
    def cache_id(self) -> str:
        return f"cache_{self.fingerprint[:32]}"


@dataclass
class CacheEntry:
    key: CacheKey
    value: Any
    status: CacheStatus = CacheStatus.ACTIVE
    ttl_seconds: Optional[float] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    last_accessed_at: Optional[datetime] = None
    hit_count: int = 0
    invalidated_at: Optional[datetime] = None
    invalidation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        if not isinstance(
            self.status,
            CacheStatus,
        ):
            raise TypeError(
                "status must be a CacheStatus"
            )

        if self.ttl_seconds is not None:
            if (
                isinstance(self.ttl_seconds, bool)
                or not isinstance(
                    self.ttl_seconds,
                    (int, float),
                )
            ):
                raise TypeError(
                    "ttl_seconds must be numeric"
                )

            if self.ttl_seconds <= 0:
                raise CacheValidationError(
                    "ttl_seconds must be greater than zero"
                )

            self.ttl_seconds = float(
                self.ttl_seconds
            )

        if (
            isinstance(self.hit_count, bool)
            or not isinstance(self.hit_count, int)
        ):
            raise TypeError(
                "hit_count must be an integer"
            )

        if self.hit_count < 0:
            raise CacheValidationError(
                "hit_count cannot be negative"
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
    def cache_id(self) -> str:
        return self.key.cache_id

    @property
    def value_fingerprint(self) -> str:
        return stable_hash(self.value)

    @property
    def expires_at(self) -> Optional[datetime]:
        if self.ttl_seconds is None:
            return None

        return self.created_at + timedelta(
            seconds=self.ttl_seconds
        )

    def is_expired(
        self,
        *,
        now: Optional[datetime] = None,
    ) -> bool:
        if self.ttl_seconds is None:
            return False

        check_time = now or utc_now()

        expires_at = self.expires_at

        if expires_at is None:
            return False

        return check_time >= expires_at

    @property
    def usable(self) -> bool:
        return (
            self.status == CacheStatus.ACTIVE
            and not self.is_expired()
        )


@dataclass
class CacheLookupResult:
    hit: bool
    cache_id: str
    value: Any = None
    status: Optional[CacheStatus] = None
    value_fingerprint: Optional[str] = None
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
class CacheManagerSnapshot:
    total_entries: int
    active_entries: int
    expired_entries: int
    invalidated_entries: int
    total_hits: int
    total_misses: int
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


class CacheManager:
    """
    Deterministic research cache.

    Provides:
    - content-addressed cache keys,
    - deterministic cache IDs,
    - namespace isolation,
    - optional TTL expiration,
    - explicit invalidation,
    - cache hit/miss accounting,
    - value fingerprints,
    - reproducible research result reuse,
    - audit history.

    Cache metadata is research-only and does not authorize
    real-money execution.
    """

    VERSION = "v6.0.1"

    def __init__(self) -> None:
        self._entries: Dict[
            str,
            CacheEntry,
        ] = {}

        self._entry_order: List[str] = []

        self._audit_events: List[
            Dict[str, Any]
        ] = []

        self._total_hits = 0
        self._total_misses = 0

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
    # Key helpers
    # ========================================================

    @staticmethod
    def create_key(
        namespace: str,
        payload: Any,
        *,
        version: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CacheKey:
        return CacheKey(
            namespace=namespace,
            payload=payload,
            version=version,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def generate_cache_id(
        key: CacheKey,
    ) -> str:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        return key.cache_id

    # ========================================================
    # Storage
    # ========================================================

    def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_seconds: Optional[float] = None,
        overwrite: bool = True,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CacheEntry:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        if not isinstance(overwrite, bool):
            raise TypeError(
                "overwrite must be a boolean"
            )

        cache_id = key.cache_id

        existing = self._entries.get(cache_id)

        if (
            existing is not None
            and existing.usable
            and not overwrite
        ):
            raise CacheManagerError(
                f"Active cache entry already exists: "
                f"{cache_id}"
            )

        entry = CacheEntry(
            key=key,
            value=value,
            ttl_seconds=ttl_seconds,
            metadata=dict(metadata or {}),
        )

        is_new = cache_id not in self._entries

        self._entries[cache_id] = entry

        if is_new:
            self._entry_order.append(
                cache_id
            )

        self._audit(
            "CACHE_SET",
            cache_id=cache_id,
            namespace=key.namespace,
            version=key.version,
            value_fingerprint=(
                entry.value_fingerprint
            ),
            ttl_seconds=entry.ttl_seconds,
            overwrite=not is_new,
        )

        return entry

    # ========================================================
    # Lookup
    # ========================================================

    def _refresh_expiration(
        self,
        entry: CacheEntry,
    ) -> None:
        if (
            entry.status == CacheStatus.ACTIVE
            and entry.is_expired()
        ):
            entry.status = CacheStatus.EXPIRED

            self._audit(
                "CACHE_EXPIRED",
                cache_id=entry.cache_id,
                namespace=entry.key.namespace,
            )

    def lookup(
        self,
        key: CacheKey,
    ) -> CacheLookupResult:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        cache_id = key.cache_id

        entry = self._entries.get(cache_id)

        if entry is None:
            self._total_misses += 1

            self._audit(
                "CACHE_MISS",
                cache_id=cache_id,
                namespace=key.namespace,
                reason="NOT_FOUND",
            )

            return CacheLookupResult(
                hit=False,
                cache_id=cache_id,
            )

        self._refresh_expiration(entry)

        if entry.status != CacheStatus.ACTIVE:
            self._total_misses += 1

            self._audit(
                "CACHE_MISS",
                cache_id=cache_id,
                namespace=key.namespace,
                reason=entry.status.value,
            )

            return CacheLookupResult(
                hit=False,
                cache_id=cache_id,
                status=entry.status,
                value_fingerprint=(
                    entry.value_fingerprint
                ),
            )

        entry.hit_count += 1
        entry.last_accessed_at = utc_now()

        self._total_hits += 1

        self._audit(
            "CACHE_HIT",
            cache_id=cache_id,
            namespace=key.namespace,
            hit_count=entry.hit_count,
        )

        return CacheLookupResult(
            hit=True,
            cache_id=cache_id,
            value=entry.value,
            status=entry.status,
            value_fingerprint=(
                entry.value_fingerprint
            ),
        )

    def get(
        self,
        key: CacheKey,
        *,
        default: Any = None,
    ) -> Any:
        result = self.lookup(key)

        if result.hit:
            return result.value

        return default

    def get_entry(
        self,
        cache_id: str,
    ) -> CacheEntry:
        if (
            not isinstance(cache_id, str)
            or not cache_id.strip()
        ):
            raise CacheValidationError(
                "cache_id must be a non-empty string"
            )

        entry = self._entries.get(
            cache_id.strip()
        )

        if entry is None:
            raise CacheEntryNotFoundError(
                f"Cache entry not found: "
                f"{cache_id}"
            )

        self._refresh_expiration(entry)

        return entry

    def exists(
        self,
        key: CacheKey,
    ) -> bool:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        entry = self._entries.get(
            key.cache_id
        )

        if entry is None:
            return False

        self._refresh_expiration(entry)

        return entry.status == CacheStatus.ACTIVE

    # ========================================================
    # Invalidation
    # ========================================================

    def invalidate(
        self,
        key: CacheKey,
        *,
        reason: Optional[str] = None,
    ) -> CacheEntry:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        entry = self.get_entry(
            key.cache_id
        )

        entry.status = CacheStatus.INVALIDATED
        entry.invalidated_at = utc_now()
        entry.invalidation_reason = reason

        self._audit(
            "CACHE_INVALIDATED",
            cache_id=entry.cache_id,
            namespace=entry.key.namespace,
            reason=reason,
        )

        return entry

    def invalidate_namespace(
        self,
        namespace: str,
        *,
        reason: Optional[str] = None,
    ) -> int:
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
        ):
            raise CacheValidationError(
                "namespace must be a non-empty string"
            )

        namespace = namespace.strip()

        count = 0

        for entry in self._entries.values():
            self._refresh_expiration(entry)

            if (
                entry.key.namespace == namespace
                and entry.status
                == CacheStatus.ACTIVE
            ):
                entry.status = (
                    CacheStatus.INVALIDATED
                )
                entry.invalidated_at = utc_now()
                entry.invalidation_reason = reason
                count += 1

        self._audit(
            "CACHE_NAMESPACE_INVALIDATED",
            namespace=namespace,
            invalidated_count=count,
            reason=reason,
        )

        return count

    # ========================================================
    # Listing
    # ========================================================

    def entries(
        self,
    ) -> List[CacheEntry]:
        result = []

        for cache_id in self._entry_order:
            entry = self._entries[
                cache_id
            ]

            self._refresh_expiration(entry)
            result.append(entry)

        return result

    def entries_by_namespace(
        self,
        namespace: str,
    ) -> List[CacheEntry]:
        if (
            not isinstance(namespace, str)
            or not namespace.strip()
        ):
            raise CacheValidationError(
                "namespace must be a non-empty string"
            )

        namespace = namespace.strip()

        return [
            entry
            for entry in self.entries()
            if entry.key.namespace
            == namespace
        ]

    # ========================================================
    # Removal
    # ========================================================

    def delete(
        self,
        key: CacheKey,
    ) -> CacheEntry:
        if not isinstance(key, CacheKey):
            raise TypeError(
                "key must be a CacheKey"
            )

        cache_id = key.cache_id

        entry = self._entries.get(
            cache_id
        )

        if entry is None:
            raise CacheEntryNotFoundError(
                f"Cache entry not found: "
                f"{cache_id}"
            )

        del self._entries[cache_id]

        self._entry_order = [
            item
            for item in self._entry_order
            if item != cache_id
        ]

        self._audit(
            "CACHE_DELETED",
            cache_id=cache_id,
            namespace=key.namespace,
        )

        return entry

    def clear(
        self,
    ) -> int:
        count = len(self._entries)

        self._entries.clear()
        self._entry_order.clear()

        self._audit(
            "CACHE_CLEARED",
            deleted_count=count,
        )

        return count

    # ========================================================
    # Reporting
    # ========================================================

    def count(self) -> int:
        return len(self._entries)

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item["created_at"],
                "details": dict(
                    item["details"]
                ),
            }
            for item in self._audit_events
        ]

    def snapshot(
        self,
    ) -> CacheManagerSnapshot:
        active = 0
        expired = 0
        invalidated = 0

        for entry in self.entries():
            if entry.status == CacheStatus.ACTIVE:
                active += 1
            elif entry.status == CacheStatus.EXPIRED:
                expired += 1
            elif (
                entry.status
                == CacheStatus.INVALIDATED
            ):
                invalidated += 1

        return CacheManagerSnapshot(
            total_entries=self.count(),
            active_entries=active,
            expired_entries=expired,
            invalidated_entries=invalidated,
            total_hits=self._total_hits,
            total_misses=self._total_misses,
            metadata={
                "manager_version": self.VERSION,
            },
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()

        requests = (
            snapshot.total_hits
            + snapshot.total_misses
        )

        hit_rate = (
            snapshot.total_hits / requests
            if requests
            else 0.0
        )

        return {
            "component": "cache_manager",
            "version": self.VERSION,
            "total_entries": (
                snapshot.total_entries
            ),
            "active_entries": (
                snapshot.active_entries
            ),
            "expired_entries": (
                snapshot.expired_entries
            ),
            "invalidated_entries": (
                snapshot.invalidated_entries
            ),
            "total_hits": (
                snapshot.total_hits
            ),
            "total_misses": (
                snapshot.total_misses
            ),
            "hit_rate": hit_rate,
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }