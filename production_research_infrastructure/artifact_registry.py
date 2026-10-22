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


class ArtifactRegistryError(RuntimeError):
    """Base exception for artifact registry failures."""


class ArtifactValidationError(ArtifactRegistryError):
    """Raised when artifact data is invalid."""


class ArtifactAlreadyExistsError(ArtifactRegistryError):
    """Raised when an artifact already exists."""


class ArtifactNotFoundError(ArtifactRegistryError):
    """Raised when an artifact cannot be found."""


class ArtifactType(str, Enum):
    DATASET = "DATASET"
    MODEL_OUTPUT = "MODEL_OUTPUT"
    AGENT_OUTPUT = "AGENT_OUTPUT"
    RESEARCH_REPORT = "RESEARCH_REPORT"
    BACKTEST_RESULT = "BACKTEST_RESULT"
    VALIDATION_RESULT = "VALIDATION_RESULT"
    PORTFOLIO = "PORTFOLIO"
    QUANTUM_RESULT = "QUANTUM_RESULT"
    CHECKPOINT = "CHECKPOINT"
    LOG = "LOG"
    METRIC = "METRIC"
    OTHER = "OTHER"


class ArtifactStatus(str, Enum):
    CREATED = "CREATED"
    VERIFIED = "VERIFIED"
    INVALIDATED = "INVALIDATED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ArtifactReference:
    reference_type: str
    reference_id: str
    relation: str = "RELATED_TO"
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_type", self.reference_type),
            ("reference_id", self.reference_id),
            ("relation", self.relation),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ArtifactValidationError(
                    f"{name} must be a non-empty string"
                )

        self.reference_type = (
            self.reference_type.strip()
        )
        self.reference_id = (
            self.reference_id.strip()
        )
        self.relation = self.relation.strip()

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


@dataclass
class ArtifactRecord:
    name: str
    artifact_type: ArtifactType
    content: Any
    run_id: Optional[str] = None
    experiment_id: Optional[str] = None
    dataset_ids: List[str] = field(
        default_factory=list
    )
    version_ids: List[str] = field(
        default_factory=list
    )
    references: List[
        ArtifactReference
    ] = field(default_factory=list)
    status: ArtifactStatus = (
        ArtifactStatus.CREATED
    )
    artifact_id: Optional[str] = None
    created_at: datetime = field(
        default_factory=utc_now
    )
    verified_at: Optional[datetime] = None
    invalidated_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
        ):
            raise ArtifactValidationError(
                "name must be a non-empty string"
            )

        self.name = self.name.strip()

        if not isinstance(
            self.artifact_type,
            ArtifactType,
        ):
            raise TypeError(
                "artifact_type must be ArtifactType"
            )

        if not isinstance(
            self.status,
            ArtifactStatus,
        ):
            raise TypeError(
                "status must be ArtifactStatus"
            )

        for field_name in (
            "run_id",
            "experiment_id",
        ):
            value = getattr(
                self,
                field_name,
            )

            if value is not None:
                if (
                    not isinstance(value, str)
                    or not value.strip()
                ):
                    raise ArtifactValidationError(
                        f"{field_name} must be a "
                        f"non-empty string"
                    )

                setattr(
                    self,
                    field_name,
                    value.strip(),
                )

        if not isinstance(
            self.dataset_ids,
            list,
        ):
            raise TypeError(
                "dataset_ids must be a list"
            )

        if not isinstance(
            self.version_ids,
            list,
        ):
            raise TypeError(
                "version_ids must be a list"
            )

        if not isinstance(
            self.references,
            list,
        ):
            raise TypeError(
                "references must be a list"
            )

        self.dataset_ids = (
            self._normalize_ids(
                self.dataset_ids,
                "dataset_ids",
            )
        )

        self.version_ids = (
            self._normalize_ids(
                self.version_ids,
                "version_ids",
            )
        )

        for reference in self.references:
            if not isinstance(
                reference,
                ArtifactReference,
            ):
                raise TypeError(
                    "references must contain "
                    "ArtifactReference objects"
                )

        if self.artifact_id is None:
            fingerprint = stable_hash(
                {
                    "name": self.name,
                    "artifact_type": (
                        self.artifact_type.value
                    ),
                    "content": self.content,
                    "run_id": self.run_id,
                    "experiment_id": (
                        self.experiment_id
                    ),
                    "dataset_ids": sorted(
                        self.dataset_ids
                    ),
                    "version_ids": sorted(
                        self.version_ids
                    ),
                }
            )

            self.artifact_id = (
                f"artifact_{fingerprint[:32]}"
            )

        elif (
            not isinstance(
                self.artifact_id,
                str,
            )
            or not self.artifact_id.strip()
        ):
            raise ArtifactValidationError(
                "artifact_id must be a "
                "non-empty string"
            )
        else:
            self.artifact_id = (
                self.artifact_id.strip()
            )

        self.metadata.setdefault(
            "research_only",
            True,
        )
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    @staticmethod
    def _normalize_ids(
        values: List[str],
        field_name: str,
    ) -> List[str]:
        normalized = []

        for value in values:
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ArtifactValidationError(
                    f"{field_name} must contain "
                    f"non-empty strings"
                )

            normalized.append(
                value.strip()
            )

        return normalized

    @property
    def content_fingerprint(self) -> str:
        return stable_hash(self.content)

    @property
    def lineage_fingerprint(self) -> str:
        references = [
            {
                "reference_type": (
                    reference.reference_type
                ),
                "reference_id": (
                    reference.reference_id
                ),
                "relation": (
                    reference.relation
                ),
            }
            for reference in self.references
        ]

        references.sort(
            key=lambda item: (
                item["reference_type"],
                item["reference_id"],
                item["relation"],
            )
        )

        return stable_hash(
            {
                "artifact_id": (
                    self.artifact_id
                ),
                "run_id": self.run_id,
                "experiment_id": (
                    self.experiment_id
                ),
                "dataset_ids": sorted(
                    self.dataset_ids
                ),
                "version_ids": sorted(
                    self.version_ids
                ),
                "references": references,
            }
        )


@dataclass
class ArtifactRegistrySnapshot:
    total_artifacts: int
    created_artifacts: int
    verified_artifacts: int
    invalidated_artifacts: int
    archived_artifacts: int
    artifact_type_counts: Dict[str, int]
    runs_with_artifacts: int
    experiments_with_artifacts: int
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


class ArtifactRegistry:
    """
    Registry for research artifacts and their provenance.

    Tracks:
    - deterministic artifact identities,
    - content fingerprints,
    - run ownership,
    - experiment ownership,
    - dataset dependencies,
    - model/prompt/policy version dependencies,
    - arbitrary lineage references,
    - verification,
    - invalidation,
    - archival,
    - audit history.

    This registry stores research metadata only.
    """

    VERSION = "v6.0.1"

    def __init__(self) -> None:
        self._artifacts: Dict[
            str,
            ArtifactRecord,
        ] = {}

        self._artifact_order: List[
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
    # Registration
    # ========================================================

    def register(
        self,
        artifact: ArtifactRecord,
    ) -> ArtifactRecord:
        if not isinstance(
            artifact,
            ArtifactRecord,
        ):
            raise TypeError(
                "artifact must be ArtifactRecord"
            )

        artifact_id = artifact.artifact_id

        if artifact_id is None:
            raise ArtifactRegistryError(
                "artifact_id generation failed"
            )

        if artifact_id in self._artifacts:
            raise ArtifactAlreadyExistsError(
                f"Artifact already exists: "
                f"{artifact_id}"
            )

        self._artifacts[
            artifact_id
        ] = artifact

        self._artifact_order.append(
            artifact_id
        )

        self._audit(
            "ARTIFACT_REGISTERED",
            artifact_id=artifact_id,
            name=artifact.name,
            artifact_type=(
                artifact.artifact_type.value
            ),
            run_id=artifact.run_id,
            experiment_id=(
                artifact.experiment_id
            ),
            content_fingerprint=(
                artifact.content_fingerprint
            ),
            lineage_fingerprint=(
                artifact.lineage_fingerprint
            ),
        )

        return artifact

    def create(
        self,
        *,
        name: str,
        artifact_type: ArtifactType,
        content: Any,
        run_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        dataset_ids: Optional[
            List[str]
        ] = None,
        version_ids: Optional[
            List[str]
        ] = None,
        references: Optional[
            List[ArtifactReference]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> ArtifactRecord:
        artifact = ArtifactRecord(
            name=name,
            artifact_type=artifact_type,
            content=content,
            run_id=run_id,
            experiment_id=experiment_id,
            dataset_ids=list(
                dataset_ids or []
            ),
            version_ids=list(
                version_ids or []
            ),
            references=list(
                references or []
            ),
            metadata=dict(
                metadata or {}
            ),
        )

        return self.register(
            artifact
        )

    # ========================================================
    # Retrieval
    # ========================================================

    def get(
        self,
        artifact_id: str,
    ) -> ArtifactRecord:
        if (
            not isinstance(
                artifact_id,
                str,
            )
            or not artifact_id.strip()
        ):
            raise ArtifactValidationError(
                "artifact_id must be a "
                "non-empty string"
            )

        artifact = self._artifacts.get(
            artifact_id.strip()
        )

        if artifact is None:
            raise ArtifactNotFoundError(
                f"Artifact not found: "
                f"{artifact_id}"
            )

        return artifact

    def exists(
        self,
        artifact_id: str,
    ) -> bool:
        try:
            self.get(artifact_id)
        except ArtifactNotFoundError:
            return False

        return True

    def artifacts(
        self,
    ) -> List[ArtifactRecord]:
        return [
            self._artifacts[
                artifact_id
            ]
            for artifact_id
            in self._artifact_order
        ]

    # ========================================================
    # Filtering
    # ========================================================

    def by_type(
        self,
        artifact_type: ArtifactType,
    ) -> List[ArtifactRecord]:
        if not isinstance(
            artifact_type,
            ArtifactType,
        ):
            raise TypeError(
                "artifact_type must be ArtifactType"
            )

        return [
            artifact
            for artifact in self.artifacts()
            if (
                artifact.artifact_type
                == artifact_type
            )
        ]

    def by_status(
        self,
        status: ArtifactStatus,
    ) -> List[ArtifactRecord]:
        if not isinstance(
            status,
            ArtifactStatus,
        ):
            raise TypeError(
                "status must be ArtifactStatus"
            )

        return [
            artifact
            for artifact in self.artifacts()
            if artifact.status == status
        ]

    def for_run(
        self,
        run_id: str,
    ) -> List[ArtifactRecord]:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise ArtifactValidationError(
                "run_id must be a non-empty string"
            )

        run_id = run_id.strip()

        return [
            artifact
            for artifact in self.artifacts()
            if artifact.run_id == run_id
        ]

    def for_experiment(
        self,
        experiment_id: str,
    ) -> List[ArtifactRecord]:
        if (
            not isinstance(
                experiment_id,
                str,
            )
            or not experiment_id.strip()
        ):
            raise ArtifactValidationError(
                "experiment_id must be a "
                "non-empty string"
            )

        experiment_id = (
            experiment_id.strip()
        )

        return [
            artifact
            for artifact in self.artifacts()
            if (
                artifact.experiment_id
                == experiment_id
            )
        ]

    def using_dataset(
        self,
        dataset_id: str,
    ) -> List[ArtifactRecord]:
        if (
            not isinstance(
                dataset_id,
                str,
            )
            or not dataset_id.strip()
        ):
            raise ArtifactValidationError(
                "dataset_id must be a "
                "non-empty string"
            )

        dataset_id = dataset_id.strip()

        return [
            artifact
            for artifact in self.artifacts()
            if dataset_id
            in artifact.dataset_ids
        ]

    def using_version(
        self,
        version_id: str,
    ) -> List[ArtifactRecord]:
        if (
            not isinstance(
                version_id,
                str,
            )
            or not version_id.strip()
        ):
            raise ArtifactValidationError(
                "version_id must be a "
                "non-empty string"
            )

        version_id = version_id.strip()

        return [
            artifact
            for artifact in self.artifacts()
            if version_id
            in artifact.version_ids
        ]

    # ========================================================
    # References
    # ========================================================

    def add_reference(
        self,
        artifact_id: str,
        reference: ArtifactReference,
    ) -> ArtifactRecord:
        if not isinstance(
            reference,
            ArtifactReference,
        ):
            raise TypeError(
                "reference must be "
                "ArtifactReference"
            )

        artifact = self.get(
            artifact_id
        )

        artifact.references.append(
            reference
        )

        self._audit(
            "ARTIFACT_REFERENCE_ADDED",
            artifact_id=artifact_id,
            reference_type=(
                reference.reference_type
            ),
            reference_id=(
                reference.reference_id
            ),
            relation=reference.relation,
        )

        return artifact

    def references_to(
        self,
        reference_id: str,
    ) -> List[ArtifactRecord]:
        if (
            not isinstance(
                reference_id,
                str,
            )
            or not reference_id.strip()
        ):
            raise ArtifactValidationError(
                "reference_id must be a "
                "non-empty string"
            )

        reference_id = (
            reference_id.strip()
        )

        result = []

        for artifact in self.artifacts():
            if any(
                reference.reference_id
                == reference_id
                for reference
                in artifact.references
            ):
                result.append(artifact)

        return result

    # ========================================================
    # Lifecycle
    # ========================================================

    def verify(
        self,
        artifact_id: str,
    ) -> ArtifactRecord:
        artifact = self.get(
            artifact_id
        )

        if (
            artifact.status
            == ArtifactStatus.INVALIDATED
        ):
            raise ArtifactRegistryError(
                "Invalidated artifact cannot "
                "be verified"
            )

        if (
            artifact.status
            == ArtifactStatus.ARCHIVED
        ):
            raise ArtifactRegistryError(
                "Archived artifact cannot "
                "be verified"
            )

        artifact.status = (
            ArtifactStatus.VERIFIED
        )
        artifact.verified_at = utc_now()

        self._audit(
            "ARTIFACT_VERIFIED",
            artifact_id=artifact_id,
            content_fingerprint=(
                artifact.content_fingerprint
            ),
        )

        return artifact

    def invalidate(
        self,
        artifact_id: str,
        *,
        reason: Optional[str] = None,
    ) -> ArtifactRecord:
        artifact = self.get(
            artifact_id
        )

        if (
            artifact.status
            == ArtifactStatus.ARCHIVED
        ):
            raise ArtifactRegistryError(
                "Archived artifact cannot "
                "be invalidated"
            )

        artifact.status = (
            ArtifactStatus.INVALIDATED
        )
        artifact.invalidated_at = (
            utc_now()
        )

        if reason is not None:
            artifact.metadata[
                "invalidation_reason"
            ] = reason

        self._audit(
            "ARTIFACT_INVALIDATED",
            artifact_id=artifact_id,
            reason=reason,
        )

        return artifact

    def archive(
        self,
        artifact_id: str,
    ) -> ArtifactRecord:
        artifact = self.get(
            artifact_id
        )

        artifact.status = (
            ArtifactStatus.ARCHIVED
        )
        artifact.archived_at = utc_now()

        self._audit(
            "ARTIFACT_ARCHIVED",
            artifact_id=artifact_id,
        )

        return artifact

    # ========================================================
    # Integrity
    # ========================================================

    def verify_content(
        self,
        artifact_id: str,
        content: Any,
    ) -> bool:
        artifact = self.get(
            artifact_id
        )

        return (
            artifact.content_fingerprint
            == stable_hash(content)
        )

    # ========================================================
    # Reporting
    # ========================================================

    def count(self) -> int:
        return len(self._artifacts)

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
    ) -> ArtifactRegistrySnapshot:
        status_counts = {
            status: 0
            for status in ArtifactStatus
        }

        type_counts = {
            artifact_type.value: 0
            for artifact_type
            in ArtifactType
        }

        runs = set()
        experiments = set()

        for artifact in self.artifacts():
            status_counts[
                artifact.status
            ] += 1

            type_counts[
                artifact.artifact_type.value
            ] += 1

            if artifact.run_id is not None:
                runs.add(
                    artifact.run_id
                )

            if (
                artifact.experiment_id
                is not None
            ):
                experiments.add(
                    artifact.experiment_id
                )

        return ArtifactRegistrySnapshot(
            total_artifacts=self.count(),
            created_artifacts=status_counts[
                ArtifactStatus.CREATED
            ],
            verified_artifacts=status_counts[
                ArtifactStatus.VERIFIED
            ],
            invalidated_artifacts=(
                status_counts[
                    ArtifactStatus.INVALIDATED
                ]
            ),
            archived_artifacts=status_counts[
                ArtifactStatus.ARCHIVED
            ],
            artifact_type_counts=type_counts,
            runs_with_artifacts=len(runs),
            experiments_with_artifacts=(
                len(experiments)
            ),
            metadata={
                "registry_version": self.VERSION,
            },
        )

    def artifact_report(
        self,
        artifact_id: str,
    ) -> Dict[str, Any]:
        artifact = self.get(
            artifact_id
        )

        return {
            "artifact_id": (
                artifact.artifact_id
            ),
            "name": artifact.name,
            "artifact_type": (
                artifact.artifact_type.value
            ),
            "status": (
                artifact.status.value
            ),
            "run_id": artifact.run_id,
            "experiment_id": (
                artifact.experiment_id
            ),
            "dataset_ids": list(
                artifact.dataset_ids
            ),
            "version_ids": list(
                artifact.version_ids
            ),
            "reference_count": len(
                artifact.references
            ),
            "content_fingerprint": (
                artifact.content_fingerprint
            ),
            "lineage_fingerprint": (
                artifact.lineage_fingerprint
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def summary(
        self,
    ) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "component": (
                "artifact_registry"
            ),
            "version": self.VERSION,
            "total_artifacts": (
                snapshot.total_artifacts
            ),
            "created_artifacts": (
                snapshot.created_artifacts
            ),
            "verified_artifacts": (
                snapshot.verified_artifacts
            ),
            "invalidated_artifacts": (
                snapshot.invalidated_artifacts
            ),
            "archived_artifacts": (
                snapshot.archived_artifacts
            ),
            "artifact_type_counts": dict(
                snapshot.artifact_type_counts
            ),
            "runs_with_artifacts": (
                snapshot.runs_with_artifacts
            ),
            "experiments_with_artifacts": (
                snapshot.experiments_with_artifacts
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }