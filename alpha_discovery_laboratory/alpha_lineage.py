from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    VERSION,
    AlphaDiscoveryExperiment,
    AlphaLineageRecord,
    StrategyCandidate,
    ValidationResult,
    stable_hash,
    utc_now,
)


# ============================================================
# Exceptions
# ============================================================


class AlphaLineageError(Exception):
    """Base exception for alpha lineage operations."""


class AlphaLineageNotFoundError(AlphaLineageError):
    """Raised when a lineage record cannot be found."""


class AlphaExperimentNotFoundError(AlphaLineageError):
    """Raised when an experiment cannot be found."""


class AlphaLineageAlreadyExistsError(AlphaLineageError):
    """Raised when duplicate lineage registration is disallowed."""


class AlphaExperimentAlreadyExistsError(AlphaLineageError):
    """Raised when duplicate experiment registration is disallowed."""


# ============================================================
# Audit Models
# ============================================================


@dataclass(frozen=True)
class AlphaLineageAuditRecord:
    action: str
    object_type: str
    object_id: str

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "action": self.action,
                "object_type": self.object_type,
                "object_id": self.object_id,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
            }
        )


@dataclass(frozen=True)
class AlphaLineageSnapshot:
    lineage_count: int
    experiment_count: int
    audit_count: int

    candidate_ids: List[str]
    experiment_ids: List[str]

    fingerprint: str

    created_at: datetime = field(
        default_factory=utc_now
    )


# ============================================================
# Alpha Lineage Registry
# ============================================================


class AlphaLineageRegistry:
    """
    Reproducible lineage registry for discovered alpha candidates.

    Tracks:

        Dataset
          -> Feature
          -> Feature Combination
          -> Strategy Candidate
          -> Validation Results
          -> Experiment
          -> Research Run

    The registry is intentionally research-only and does not
    perform live or real-money execution.
    """

    VERSION = VERSION

    def __init__(self) -> None:
        self._lineage: Dict[
            str,
            AlphaLineageRecord,
        ] = {}

        self._lineage_fingerprints: Dict[
            str,
            str,
        ] = {}

        self._experiments: Dict[
            str,
            AlphaDiscoveryExperiment,
        ] = {}

        self._experiment_fingerprints: Dict[
            str,
            str,
        ] = {}

        self._audit: List[
            AlphaLineageAuditRecord
        ] = []

    # ========================================================
    # Audit
    # ========================================================

    def _record_audit(
        self,
        *,
        action: str,
        object_type: str,
        object_id: str,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        self._audit.append(
            AlphaLineageAuditRecord(
                action=action,
                object_type=object_type,
                object_id=object_id,
                details=dict(details or {}),
            )
        )

    def audit_log(
        self,
    ) -> List[AlphaLineageAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Experiment Registry
    # ========================================================

    def register_experiment(
        self,
        experiment: AlphaDiscoveryExperiment,
        *,
        allow_existing: bool = True,
    ) -> AlphaDiscoveryExperiment:
        if not isinstance(
            experiment,
            AlphaDiscoveryExperiment,
        ):
            raise AlphaLineageError(
                "experiment must be an "
                "AlphaDiscoveryExperiment instance."
            )

        existing_id = (
            self._experiment_fingerprints.get(
                experiment.fingerprint
            )
        )

        if existing_id is not None:
            if allow_existing:
                return self._experiments[
                    existing_id
                ]

            raise AlphaExperimentAlreadyExistsError(
                "Experiment already exists: "
                f"{existing_id}"
            )

        experiment_id = experiment.experiment_id

        if experiment_id in self._experiments:
            existing = self._experiments[
                experiment_id
            ]

            if (
                existing.fingerprint
                == experiment.fingerprint
            ):
                if allow_existing:
                    return existing

                raise AlphaExperimentAlreadyExistsError(
                    "Experiment already exists: "
                    f"{experiment_id}"
                )

            raise AlphaLineageError(
                "Experiment ID collision detected."
            )

        self._experiments[
            experiment_id
        ] = experiment

        self._experiment_fingerprints[
            experiment.fingerprint
        ] = experiment_id

        self._record_audit(
            action="register",
            object_type="experiment",
            object_id=experiment_id,
            details={
                "name": experiment.name,
                "dataset_ids": list(
                    experiment.dataset_ids
                ),
                "feature_ids": list(
                    experiment.feature_ids
                ),
                "combination_ids": list(
                    experiment.combination_ids
                ),
                "candidate_ids": list(
                    experiment.candidate_ids
                ),
                "run_id": experiment.run_id,
            },
        )

        return experiment

    def create_experiment(
        self,
        *,
        name: str,
        dataset_ids: Sequence[str],
        feature_ids: Optional[
            Sequence[str]
        ] = None,
        combination_ids: Optional[
            Sequence[str]
        ] = None,
        candidate_ids: Optional[
            Sequence[str]
        ] = None,
        accepted_candidate_ids: Optional[
            Sequence[str]
        ] = None,
        rejected_candidate_ids: Optional[
            Sequence[str]
        ] = None,
        configuration: Optional[
            Dict[str, Any]
        ] = None,
        seed: Optional[int] = None,
        run_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        allow_existing: bool = True,
    ) -> AlphaDiscoveryExperiment:
        if not isinstance(name, str) or not name.strip():
            raise AlphaLineageError(
                "name must be a non-empty string."
            )

        if not dataset_ids:
            raise AlphaLineageError(
                "At least one dataset ID is required."
            )

        experiment = AlphaDiscoveryExperiment(
            name=name.strip(),
            dataset_ids=list(dataset_ids),
            feature_ids=list(
                feature_ids or []
            ),
            combination_ids=list(
                combination_ids or []
            ),
            candidate_ids=list(
                candidate_ids or []
            ),
            accepted_candidate_ids=list(
                accepted_candidate_ids or []
            ),
            rejected_candidate_ids=list(
                rejected_candidate_ids or []
            ),
            configuration=dict(
                configuration or {}
            ),
            seed=seed,
            run_id=run_id,
            metadata=dict(metadata or {}),
        )

        return self.register_experiment(
            experiment,
            allow_existing=allow_existing,
        )

    def get_experiment(
        self,
        experiment_id: str,
    ) -> AlphaDiscoveryExperiment:
        try:
            return self._experiments[
                experiment_id
            ]
        except KeyError as exc:
            raise AlphaExperimentNotFoundError(
                "Experiment not found: "
                f"{experiment_id}"
            ) from exc

    def experiment_exists(
        self,
        experiment_id: str,
    ) -> bool:
        return (
            experiment_id
            in self._experiments
        )

    def experiments(
        self,
    ) -> List[AlphaDiscoveryExperiment]:
        return list(
            self._experiments.values()
        )

    def experiment_count(self) -> int:
        return len(self._experiments)

    # ========================================================
    # Lineage Registration
    # ========================================================

    def register_lineage(
        self,
        record: AlphaLineageRecord,
        *,
        allow_existing: bool = True,
    ) -> AlphaLineageRecord:
        if not isinstance(
            record,
            AlphaLineageRecord,
        ):
            raise AlphaLineageError(
                "record must be an "
                "AlphaLineageRecord instance."
            )

        existing_candidate = self._lineage.get(
            record.candidate_id
        )

        if existing_candidate is not None:
            if (
                existing_candidate.fingerprint
                == record.fingerprint
            ):
                if allow_existing:
                    return existing_candidate

                raise AlphaLineageAlreadyExistsError(
                    "Lineage already exists for "
                    f"{record.candidate_id}"
                )

            raise AlphaLineageAlreadyExistsError(
                "A different lineage record already "
                "exists for candidate "
                f"{record.candidate_id}"
            )

        existing_id = (
            self._lineage_fingerprints.get(
                record.fingerprint
            )
        )

        if existing_id is not None:
            if allow_existing:
                return self._lineage[
                    existing_id
                ]

            raise AlphaLineageAlreadyExistsError(
                "Lineage fingerprint already "
                f"registered: {existing_id}"
            )

        self._lineage[
            record.candidate_id
        ] = record

        self._lineage_fingerprints[
            record.fingerprint
        ] = record.candidate_id

        self._record_audit(
            action="register",
            object_type="lineage",
            object_id=record.candidate_id,
            details={
                "feature_ids": list(
                    record.feature_ids
                ),
                "combination_ids": list(
                    record.combination_ids
                ),
                "dataset_ids": list(
                    record.dataset_ids
                ),
                "experiment_id": (
                    record.experiment_id
                ),
                "run_id": record.run_id,
            },
        )

        return record

    # ========================================================
    # Lineage Creation
    # ========================================================

    def create_lineage(
        self,
        *,
        candidate: StrategyCandidate,
        validation_results: Optional[
            Sequence[ValidationResult]
        ] = None,
        dataset_ids: Optional[
            Sequence[str]
        ] = None,
        dataset_fingerprints: Optional[
            Sequence[str]
        ] = None,
        model_version_ids: Optional[
            Sequence[str]
        ] = None,
        prompt_version_ids: Optional[
            Sequence[str]
        ] = None,
        policy_version_ids: Optional[
            Sequence[str]
        ] = None,
        experiment_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        allow_existing: bool = True,
    ) -> AlphaLineageRecord:
        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise AlphaLineageError(
                "candidate must be a "
                "StrategyCandidate instance."
            )

        if (
            experiment_id is not None
            and not self.experiment_exists(
                experiment_id
            )
        ):
            raise AlphaExperimentNotFoundError(
                "Experiment not found: "
                f"{experiment_id}"
            )

        results = list(
            validation_results or []
        )

        for result in results:
            if not isinstance(
                result,
                ValidationResult,
            ):
                raise AlphaLineageError(
                    "validation_results must contain "
                    "ValidationResult instances."
                )

            if (
                result.candidate_id
                != candidate.candidate_id
            ):
                raise AlphaLineageError(
                    "Validation result candidate ID "
                    "does not match candidate."
                )

        record = AlphaLineageRecord(
            candidate_id=(
                candidate.candidate_id
            ),
            feature_ids=list(
                candidate.feature_ids
            ),
            combination_ids=list(
                candidate.combination_ids
            ),
            dataset_ids=list(
                dataset_ids or []
            ),
            dataset_fingerprints=list(
                dataset_fingerprints or []
            ),
            model_version_ids=list(
                model_version_ids or []
            ),
            prompt_version_ids=list(
                prompt_version_ids or []
            ),
            policy_version_ids=list(
                policy_version_ids or []
            ),
            validation_fingerprints=[
                result.fingerprint
                for result in results
            ],
            experiment_id=experiment_id,
            run_id=run_id,
            parent_candidate_ids=list(
                candidate.parent_candidate_ids
            ),
            metadata=dict(metadata or {}),
        )

        return self.register_lineage(
            record,
            allow_existing=allow_existing,
        )

    # ========================================================
    # Lookup
    # ========================================================

    def exists(
        self,
        candidate_id: str,
    ) -> bool:
        return candidate_id in self._lineage

    def get(
        self,
        candidate_id: str,
    ) -> AlphaLineageRecord:
        try:
            return self._lineage[
                candidate_id
            ]
        except KeyError as exc:
            raise AlphaLineageNotFoundError(
                "Lineage not found for candidate: "
                f"{candidate_id}"
            ) from exc

    def get_by_fingerprint(
        self,
        fingerprint: str,
    ) -> AlphaLineageRecord:
        try:
            candidate_id = (
                self._lineage_fingerprints[
                    fingerprint
                ]
            )
        except KeyError as exc:
            raise AlphaLineageNotFoundError(
                "Lineage fingerprint not found: "
                f"{fingerprint}"
            ) from exc

        return self._lineage[
            candidate_id
        ]

    def records(
        self,
    ) -> List[AlphaLineageRecord]:
        return list(
            self._lineage.values()
        )

    def lineage_count(self) -> int:
        return len(self._lineage)

    # ========================================================
    # Queries
    # ========================================================

    def by_experiment(
        self,
        experiment_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if record.experiment_id
            == experiment_id
        ]

    def by_run(
        self,
        run_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if record.run_id == run_id
        ]

    def by_dataset(
        self,
        dataset_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if dataset_id
            in record.dataset_ids
        ]

    def by_feature(
        self,
        feature_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if feature_id
            in record.feature_ids
        ]

    def by_combination(
        self,
        combination_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if combination_id
            in record.combination_ids
        ]

    def children_of(
        self,
        candidate_id: str,
    ) -> List[AlphaLineageRecord]:
        return [
            record
            for record in self._lineage.values()
            if candidate_id
            in record.parent_candidate_ids
        ]

    def parents_of(
        self,
        candidate_id: str,
    ) -> List[AlphaLineageRecord]:
        record = self.get(candidate_id)

        parents: List[
            AlphaLineageRecord
        ] = []

        for parent_id in (
            record.parent_candidate_ids
        ):
            if parent_id in self._lineage:
                parents.append(
                    self._lineage[parent_id]
                )

        return parents

    # ========================================================
    # Reproducibility
    # ========================================================

    def experiment_fingerprint(
        self,
        experiment_id: str,
    ) -> str:
        return self.get_experiment(
            experiment_id
        ).fingerprint

    def lineage_set_fingerprint(
        self,
        candidate_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> str:
        selected_ids = (
            list(candidate_ids)
            if candidate_ids is not None
            else list(self._lineage.keys())
        )

        fingerprints: List[str] = []

        for candidate_id in selected_ids:
            fingerprints.append(
                self.get(
                    candidate_id
                ).fingerprint
            )

        return stable_hash(
            sorted(fingerprints)
        )

    def experiment_set_fingerprint(
        self,
        experiment_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> str:
        selected_ids = (
            list(experiment_ids)
            if experiment_ids is not None
            else list(
                self._experiments.keys()
            )
        )

        fingerprints: List[str] = []

        for experiment_id in selected_ids:
            fingerprints.append(
                self.get_experiment(
                    experiment_id
                ).fingerprint
            )

        return stable_hash(
            sorted(fingerprints)
        )

    def full_fingerprint(self) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "lineage": (
                    self.lineage_set_fingerprint()
                ),
                "experiments": (
                    self.experiment_set_fingerprint()
                ),
            }
        )

    # ========================================================
    # Snapshot / Summary
    # ========================================================

    def snapshot(
        self,
    ) -> AlphaLineageSnapshot:
        return AlphaLineageSnapshot(
            lineage_count=(
                self.lineage_count()
            ),
            experiment_count=(
                self.experiment_count()
            ),
            audit_count=len(self._audit),
            candidate_ids=list(
                self._lineage.keys()
            ),
            experiment_ids=list(
                self._experiments.keys()
            ),
            fingerprint=(
                self.full_fingerprint()
            ),
        )

    def summary(self) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "version": self.VERSION,
            "lineage_count": (
                snapshot.lineage_count
            ),
            "experiment_count": (
                snapshot.experiment_count
            ),
            "audit_count": (
                snapshot.audit_count
            ),
            "fingerprint": (
                snapshot.fingerprint
            ),
            "research_only": True,
            "real_money_execution": False,
        }


# ============================================================
# Public exports
# ============================================================


__all__ = [
    "AlphaLineageError",
    "AlphaLineageNotFoundError",
    "AlphaExperimentNotFoundError",
    "AlphaLineageAlreadyExistsError",
    "AlphaExperimentAlreadyExistsError",
    "AlphaLineageAuditRecord",
    "AlphaLineageSnapshot",
    "AlphaLineageRegistry",
]