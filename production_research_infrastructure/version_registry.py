from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


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


class VersionRegistryError(RuntimeError):
    """Base exception for version registry failures."""


class VersionValidationError(VersionRegistryError):
    """Raised when version information is invalid."""


class VersionAlreadyExistsError(VersionRegistryError):
    """Raised when a duplicate version is registered."""


class VersionNotFoundError(VersionRegistryError):
    """Raised when a requested version cannot be found."""


class VersionType(str, Enum):
    MODEL = "MODEL"
    PROMPT = "PROMPT"
    RESEARCH_POLICY = "RESEARCH_POLICY"


class VersionStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    ARCHIVED = "ARCHIVED"


@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise TypeError(
                    f"{name} must be an integer"
                )

            if value < 0:
                raise VersionValidationError(
                    f"{name} must be greater than "
                    "or equal to zero"
                )

    @classmethod
    def parse(
        cls,
        version: str,
    ) -> "SemanticVersion":
        if not isinstance(version, str):
            raise TypeError(
                "version must be a string"
            )

        cleaned = version.strip()

        if cleaned.startswith("v"):
            cleaned = cleaned[1:]

        parts = cleaned.split(".")

        if len(parts) not in (1, 2, 3):
            raise VersionValidationError(
                f"Invalid semantic version: "
                f"{version!r}"
            )

        try:
            numbers = [
                int(part)
                for part in parts
            ]
        except ValueError as exc:
            raise VersionValidationError(
                f"Invalid semantic version: "
                f"{version!r}"
            ) from exc

        while len(numbers) < 3:
            numbers.append(0)

        return cls(*numbers)

    def __str__(self) -> str:
        return (
            f"{self.major}."
            f"{self.minor}."
            f"{self.patch}"
        )


@dataclass
class VersionRecord:
    name: str
    version: str
    version_type: VersionType
    content: Any
    status: VersionStatus = VersionStatus.DRAFT
    description: Optional[str] = None
    parent_version_id: Optional[str] = None
    version_id: str = field(
        default_factory=lambda: generate_id("version")
    )
    created_at: datetime = field(
        default_factory=utc_now
    )
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
        ):
            raise VersionValidationError(
                "name must be a non-empty string"
            )

        self.name = self.name.strip()

        if (
            not isinstance(self.version, str)
            or not self.version.strip()
        ):
            raise VersionValidationError(
                "version must be a non-empty string"
            )

        self.version = str(
            SemanticVersion.parse(
                self.version
            )
        )

        if not isinstance(
            self.version_type,
            VersionType,
        ):
            raise TypeError(
                "version_type must be a VersionType"
            )

        if not isinstance(
            self.status,
            VersionStatus,
        ):
            raise TypeError(
                "status must be a VersionStatus"
            )

        if self.parent_version_id is not None:
            if (
                not isinstance(
                    self.parent_version_id,
                    str,
                )
                or not self.parent_version_id.strip()
            ):
                raise VersionValidationError(
                    "parent_version_id must be a "
                    "non-empty string"
                )

            self.parent_version_id = (
                self.parent_version_id.strip()
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
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "version": self.version,
                "version_type": (
                    self.version_type.value
                ),
                "content": self.content,
            }
        )


@dataclass
class VersionRegistrySnapshot:
    total_versions: int
    model_versions: int
    prompt_versions: int
    research_policy_versions: int
    active_versions: int
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


class VersionRegistry:
    """
    Registry for reproducible model, prompt and research-policy
    versioning.

    Supports:
    - semantic versions,
    - immutable version records,
    - model versions,
    - prompt versions,
    - research-policy versions,
    - active-version management,
    - version ancestry,
    - deterministic fingerprints,
    - audit history,
    - reproducible research configuration.

    The registry stores research configuration only.
    It does not authorize real-money trading.
    """

    VERSION = "v6.0.1"

    def __init__(self) -> None:
        self._records_by_id: Dict[
            str,
            VersionRecord,
        ] = {}

        self._versions: Dict[
            tuple[VersionType, str, str],
            str,
        ] = {}

        self._active_versions: Dict[
            tuple[VersionType, str],
            str,
        ] = {}

        self._registration_order: List[str] = []

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
    # Registration
    # ========================================================

    def register(
        self,
        record: VersionRecord,
    ) -> VersionRecord:
        if not isinstance(
            record,
            VersionRecord,
        ):
            raise TypeError(
                "record must be a VersionRecord"
            )

        key = (
            record.version_type,
            record.name,
            record.version,
        )

        if key in self._versions:
            raise VersionAlreadyExistsError(
                f"{record.version_type.value} "
                f"{record.name!r} version "
                f"{record.version!r} already exists"
            )

        if record.version_id in self._records_by_id:
            raise VersionAlreadyExistsError(
                f"version_id already exists: "
                f"{record.version_id}"
            )

        if record.parent_version_id is not None:
            if (
                record.parent_version_id
                not in self._records_by_id
            ):
                raise VersionNotFoundError(
                    "Parent version not found: "
                    f"{record.parent_version_id}"
                )

            parent = self._records_by_id[
                record.parent_version_id
            ]

            if (
                parent.version_type
                != record.version_type
                or parent.name
                != record.name
            ):
                raise VersionValidationError(
                    "Parent version must belong "
                    "to the same name and type"
                )

        self._records_by_id[
            record.version_id
        ] = record

        self._versions[key] = record.version_id

        self._registration_order.append(
            record.version_id
        )

        if record.status == VersionStatus.ACTIVE:
            self._activate_record(record)

        self._audit(
            "VERSION_REGISTERED",
            version_id=record.version_id,
            name=record.name,
            version=record.version,
            version_type=record.version_type.value,
            status=record.status.value,
            fingerprint=record.fingerprint,
        )

        return record

    def register_model(
        self,
        *,
        name: str,
        version: str,
        content: Any,
        status: VersionStatus = VersionStatus.DRAFT,
        description: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionRecord:
        return self.register(
            VersionRecord(
                name=name,
                version=version,
                version_type=VersionType.MODEL,
                content=content,
                status=status,
                description=description,
                parent_version_id=parent_version_id,
                metadata=dict(metadata or {}),
            )
        )

    def register_prompt(
        self,
        *,
        name: str,
        version: str,
        content: Any,
        status: VersionStatus = VersionStatus.DRAFT,
        description: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionRecord:
        return self.register(
            VersionRecord(
                name=name,
                version=version,
                version_type=VersionType.PROMPT,
                content=content,
                status=status,
                description=description,
                parent_version_id=parent_version_id,
                metadata=dict(metadata or {}),
            )
        )

    def register_research_policy(
        self,
        *,
        name: str,
        version: str,
        content: Any,
        status: VersionStatus = VersionStatus.DRAFT,
        description: Optional[str] = None,
        parent_version_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VersionRecord:
        return self.register(
            VersionRecord(
                name=name,
                version=version,
                version_type=(
                    VersionType.RESEARCH_POLICY
                ),
                content=content,
                status=status,
                description=description,
                parent_version_id=parent_version_id,
                metadata=dict(metadata or {}),
            )
        )

    # ========================================================
    # Active version management
    # ========================================================

    def _activate_record(
        self,
        record: VersionRecord,
    ) -> None:
        active_key = (
            record.version_type,
            record.name,
        )

        previous_id = self._active_versions.get(
            active_key
        )

        if (
            previous_id is not None
            and previous_id != record.version_id
        ):
            previous = self._records_by_id[
                previous_id
            ]

            if previous.status == VersionStatus.ACTIVE:
                previous.status = (
                    VersionStatus.DEPRECATED
                )

        record.status = VersionStatus.ACTIVE

        self._active_versions[
            active_key
        ] = record.version_id

    def activate(
        self,
        version_id: str,
    ) -> VersionRecord:
        record = self.get(version_id)

        self._activate_record(record)

        self._audit(
            "VERSION_ACTIVATED",
            version_id=record.version_id,
            name=record.name,
            version=record.version,
            version_type=record.version_type.value,
        )

        return record

    def deprecate(
        self,
        version_id: str,
    ) -> VersionRecord:
        record = self.get(version_id)

        record.status = VersionStatus.DEPRECATED

        active_key = (
            record.version_type,
            record.name,
        )

        if (
            self._active_versions.get(active_key)
            == record.version_id
        ):
            del self._active_versions[
                active_key
            ]

        self._audit(
            "VERSION_DEPRECATED",
            version_id=record.version_id,
            name=record.name,
            version=record.version,
            version_type=record.version_type.value,
        )

        return record

    def archive(
        self,
        version_id: str,
    ) -> VersionRecord:
        record = self.get(version_id)

        record.status = VersionStatus.ARCHIVED

        active_key = (
            record.version_type,
            record.name,
        )

        if (
            self._active_versions.get(active_key)
            == record.version_id
        ):
            del self._active_versions[
                active_key
            ]

        self._audit(
            "VERSION_ARCHIVED",
            version_id=record.version_id,
            name=record.name,
            version=record.version,
            version_type=record.version_type.value,
        )

        return record

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        version_id: str,
    ) -> VersionRecord:
        if (
            not isinstance(version_id, str)
            or not version_id.strip()
        ):
            raise VersionValidationError(
                "version_id must be a non-empty string"
            )

        record = self._records_by_id.get(
            version_id.strip()
        )

        if record is None:
            raise VersionNotFoundError(
                f"Version not found: {version_id}"
            )

        return record

    def get_version(
        self,
        version_type: VersionType,
        name: str,
        version: str,
    ) -> VersionRecord:
        if not isinstance(
            version_type,
            VersionType,
        ):
            raise TypeError(
                "version_type must be a VersionType"
            )

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise VersionValidationError(
                "name must be a non-empty string"
            )

        normalized_version = str(
            SemanticVersion.parse(version)
        )

        key = (
            version_type,
            name.strip(),
            normalized_version,
        )

        version_id = self._versions.get(key)

        if version_id is None:
            raise VersionNotFoundError(
                f"{version_type.value} "
                f"{name!r} version "
                f"{normalized_version!r} not found"
            )

        return self._records_by_id[
            version_id
        ]

    def get_active(
        self,
        version_type: VersionType,
        name: str,
    ) -> Optional[VersionRecord]:
        if not isinstance(
            version_type,
            VersionType,
        ):
            raise TypeError(
                "version_type must be a VersionType"
            )

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise VersionValidationError(
                "name must be a non-empty string"
            )

        version_id = self._active_versions.get(
            (
                version_type,
                name.strip(),
            )
        )

        if version_id is None:
            return None

        return self._records_by_id[
            version_id
        ]

    def exists(
        self,
        version_type: VersionType,
        name: str,
        version: str,
    ) -> bool:
        try:
            self.get_version(
                version_type,
                name,
                version,
            )
        except VersionNotFoundError:
            return False

        return True

    # ========================================================
    # Listing
    # ========================================================

    def list_all(
        self,
    ) -> List[VersionRecord]:
        return [
            self._records_by_id[
                version_id
            ]
            for version_id
            in self._registration_order
        ]

    def list_by_type(
        self,
        version_type: VersionType,
    ) -> List[VersionRecord]:
        if not isinstance(
            version_type,
            VersionType,
        ):
            raise TypeError(
                "version_type must be a VersionType"
            )

        return [
            record
            for record in self.list_all()
            if (
                record.version_type
                == version_type
            )
        ]

    def list_versions(
        self,
        version_type: VersionType,
        name: str,
    ) -> List[VersionRecord]:
        if not isinstance(
            version_type,
            VersionType,
        ):
            raise TypeError(
                "version_type must be a VersionType"
            )

        if (
            not isinstance(name, str)
            or not name.strip()
        ):
            raise VersionValidationError(
                "name must be a non-empty string"
            )

        name = name.strip()

        records = [
            record
            for record in self.list_all()
            if (
                record.version_type
                == version_type
                and record.name == name
            )
        ]

        return sorted(
            records,
            key=lambda record: (
                SemanticVersion.parse(
                    record.version
                ).major,
                SemanticVersion.parse(
                    record.version
                ).minor,
                SemanticVersion.parse(
                    record.version
                ).patch,
            ),
        )

    # ========================================================
    # Lineage
    # ========================================================

    def parent_of(
        self,
        version_id: str,
    ) -> Optional[VersionRecord]:
        record = self.get(version_id)

        if record.parent_version_id is None:
            return None

        return self.get(
            record.parent_version_id
        )

    def children_of(
        self,
        version_id: str,
    ) -> List[VersionRecord]:
        self.get(version_id)

        return [
            record
            for record in self.list_all()
            if (
                record.parent_version_id
                == version_id
            )
        ]

    def ancestry(
        self,
        version_id: str,
    ) -> List[VersionRecord]:
        current = self.get(version_id)

        lineage: List[VersionRecord] = []

        visited = set()

        while current.parent_version_id is not None:
            if current.version_id in visited:
                raise VersionRegistryError(
                    "Version lineage cycle detected"
                )

            visited.add(
                current.version_id
            )

            parent = self.get(
                current.parent_version_id
            )

            lineage.append(parent)
            current = parent

        return lineage

    # ========================================================
    # Reproducible configuration
    # ========================================================

    def configuration_fingerprint(
        self,
        *,
        model_version_ids: Optional[
            List[str]
        ] = None,
        prompt_version_ids: Optional[
            List[str]
        ] = None,
        policy_version_ids: Optional[
            List[str]
        ] = None,
    ) -> str:
        model_version_ids = list(
            model_version_ids or []
        )
        prompt_version_ids = list(
            prompt_version_ids or []
        )
        policy_version_ids = list(
            policy_version_ids or []
        )

        models = [
            self.get(version_id)
            for version_id
            in model_version_ids
        ]

        prompts = [
            self.get(version_id)
            for version_id
            in prompt_version_ids
        ]

        policies = [
            self.get(version_id)
            for version_id
            in policy_version_ids
        ]

        for record in models:
            if record.version_type != VersionType.MODEL:
                raise VersionValidationError(
                    f"{record.version_id} is not "
                    "a MODEL version"
                )

        for record in prompts:
            if record.version_type != VersionType.PROMPT:
                raise VersionValidationError(
                    f"{record.version_id} is not "
                    "a PROMPT version"
                )

        for record in policies:
            if (
                record.version_type
                != VersionType.RESEARCH_POLICY
            ):
                raise VersionValidationError(
                    f"{record.version_id} is not "
                    "a RESEARCH_POLICY version"
                )

        payload = {
            "models": sorted(
                record.fingerprint
                for record in models
            ),
            "prompts": sorted(
                record.fingerprint
                for record in prompts
            ),
            "research_policies": sorted(
                record.fingerprint
                for record in policies
            ),
        }

        return stable_hash(payload)

    # ========================================================
    # Reporting
    # ========================================================

    def count(self) -> int:
        return len(
            self._records_by_id
        )

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
    ) -> VersionRegistrySnapshot:
        return VersionRegistrySnapshot(
            total_versions=self.count(),
            model_versions=len(
                self.list_by_type(
                    VersionType.MODEL
                )
            ),
            prompt_versions=len(
                self.list_by_type(
                    VersionType.PROMPT
                )
            ),
            research_policy_versions=len(
                self.list_by_type(
                    VersionType.RESEARCH_POLICY
                )
            ),
            active_versions=len(
                self._active_versions
            ),
            metadata={
                "registry_version": self.VERSION,
            },
        )

    def summary(
        self,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "component": "version_registry",
            "version": self.VERSION,
            "total_versions": (
                snapshot.total_versions
            ),
            "model_versions": (
                snapshot.model_versions
            ),
            "prompt_versions": (
                snapshot.prompt_versions
            ),
            "research_policy_versions": (
                snapshot.research_policy_versions
            ),
            "active_versions": (
                snapshot.active_versions
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }