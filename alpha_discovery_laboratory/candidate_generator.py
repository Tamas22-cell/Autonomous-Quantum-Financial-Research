from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .feature_discovery import FeatureDiscoveryEngine
from .models import (
    VERSION,
    AlphaValidationError,
    CandidateStatus,
    StrategyCandidate,
    StrategyDirection,
    stable_hash,
    utc_now,
)


# ============================================================
# Exceptions
# ============================================================


class CandidateGenerationError(Exception):
    """Base exception for strategy candidate generation."""


class CandidateAlreadyExistsError(CandidateGenerationError):
    """Raised when a duplicate candidate is disallowed."""


class CandidateNotFoundError(CandidateGenerationError):
    """Raised when a requested candidate cannot be found."""


class CandidateSourceError(CandidateGenerationError):
    """Raised when referenced features or combinations are invalid."""


# ============================================================
# Audit / result models
# ============================================================


@dataclass(frozen=True)
class CandidateGenerationAuditRecord:
    action: str
    candidate_id: str
    timestamp: datetime = field(default_factory=utc_now)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "action": self.action,
                "candidate_id": self.candidate_id,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
            }
        )


@dataclass(frozen=True)
class CandidateGenerationResult:
    candidates: List[StrategyCandidate]
    skipped_duplicates: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_ids": [
                    candidate.candidate_id
                    for candidate in self.candidates
                ],
                "skipped_duplicates": self.skipped_duplicates,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class CandidateGeneratorSnapshot:
    candidate_count: int
    accepted_count: int
    rejected_count: int
    active_count: int
    audit_count: int
    candidate_ids: List[str]
    fingerprint: str
    created_at: datetime = field(default_factory=utc_now)


# ============================================================
# Candidate Generator
# ============================================================


class StrategyCandidateGenerator:
    """
    Generates reproducible alpha / strategy candidates from
    discovered market features and feature combinations.

    The generator does not perform backtesting or live trading.
    It creates research candidates that can later enter the
    validation pipeline:

        Backtest
            -> Causal Validation
            -> Out-of-Sample Validation
            -> Robustness Validation
    """

    VERSION = VERSION

    def __init__(
        self,
        feature_discovery: Optional[
            FeatureDiscoveryEngine
        ] = None,
    ) -> None:
        self._feature_discovery = (
            feature_discovery
            or FeatureDiscoveryEngine()
        )

        self._candidates: Dict[
            str,
            StrategyCandidate,
        ] = {}

        self._fingerprints: Dict[str, str] = {}

        self._audit: List[
            CandidateGenerationAuditRecord
        ] = []

    # ========================================================
    # Properties
    # ========================================================

    @property
    def feature_discovery(
        self,
    ) -> FeatureDiscoveryEngine:
        return self._feature_discovery

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _require_text(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AlphaValidationError(
                f"{field_name} must be a non-empty string."
            )

        return value.strip()

    def _validate_sources(
        self,
        *,
        feature_ids: Sequence[str],
        combination_ids: Sequence[str],
    ) -> None:
        if not feature_ids and not combination_ids:
            raise CandidateSourceError(
                "A strategy candidate requires at least "
                "one feature or feature combination."
            )

        for feature_id in feature_ids:
            if not self._feature_discovery.exists(
                feature_id
            ):
                raise CandidateSourceError(
                    f"Feature not found: {feature_id}"
                )

        for combination_id in combination_ids:
            if not (
                self._feature_discovery
                .combination_exists(combination_id)
            ):
                raise CandidateSourceError(
                    "Feature combination not found: "
                    f"{combination_id}"
                )

    def _validate_candidate(
        self,
        candidate: StrategyCandidate,
    ) -> None:
        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise AlphaValidationError(
                "candidate must be a "
                "StrategyCandidate instance."
            )

        self._require_text(
            candidate.name,
            "candidate.name",
        )

        self._validate_sources(
            feature_ids=candidate.feature_ids,
            combination_ids=(
                candidate.combination_ids
            ),
        )

    # ========================================================
    # Audit
    # ========================================================

    def _record_audit(
        self,
        *,
        action: str,
        candidate_id: str,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        self._audit.append(
            CandidateGenerationAuditRecord(
                action=action,
                candidate_id=candidate_id,
                details=dict(details or {}),
            )
        )

    def audit_log(
        self,
    ) -> List[CandidateGenerationAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Candidate registration
    # ========================================================

    def register(
        self,
        candidate: StrategyCandidate,
        *,
        allow_existing: bool = True,
    ) -> StrategyCandidate:
        self._validate_candidate(candidate)

        existing_id = self._fingerprints.get(
            candidate.fingerprint
        )

        if existing_id is not None:
            if allow_existing:
                return self._candidates[
                    existing_id
                ]

            raise CandidateAlreadyExistsError(
                "Candidate already exists: "
                f"{existing_id}"
            )

        if candidate.candidate_id in self._candidates:
            existing = self._candidates[
                candidate.candidate_id
            ]

            if (
                existing.fingerprint
                == candidate.fingerprint
            ):
                if allow_existing:
                    return existing

                raise CandidateAlreadyExistsError(
                    "Candidate already exists: "
                    f"{candidate.candidate_id}"
                )

            raise CandidateGenerationError(
                "Candidate ID collision detected."
            )

        self._candidates[
            candidate.candidate_id
        ] = candidate

        self._fingerprints[
            candidate.fingerprint
        ] = candidate.candidate_id

        self._record_audit(
            action="register",
            candidate_id=candidate.candidate_id,
            details={
                "name": candidate.name,
                "direction": (
                    candidate.direction.value
                ),
                "feature_ids": list(
                    candidate.feature_ids
                ),
                "combination_ids": list(
                    candidate.combination_ids
                ),
            },
        )

        return candidate

    # ========================================================
    # Candidate creation
    # ========================================================

    def create_candidate(
        self,
        *,
        name: str,
        feature_ids: Optional[
            Sequence[str]
        ] = None,
        combination_ids: Optional[
            Sequence[str]
        ] = None,
        direction: StrategyDirection = (
            StrategyDirection.LONG_SHORT
        ),
        entry_rule: Optional[
            Dict[str, Any]
        ] = None,
        exit_rule: Optional[
            Dict[str, Any]
        ] = None,
        position_rule: Optional[
            Dict[str, Any]
        ] = None,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        hypothesis_id: Optional[str] = None,
        parent_candidate_ids: Optional[
            Sequence[str]
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
        allow_existing: bool = True,
    ) -> StrategyCandidate:
        name = self._require_text(
            name,
            "name",
        )

        if not isinstance(
            direction,
            StrategyDirection,
        ):
            raise AlphaValidationError(
                "direction must be StrategyDirection."
            )

        normalized_feature_ids = list(
            feature_ids or []
        )

        normalized_combination_ids = list(
            combination_ids or []
        )

        self._validate_sources(
            feature_ids=normalized_feature_ids,
            combination_ids=(
                normalized_combination_ids
            ),
        )

        candidate = StrategyCandidate(
            name=name,
            feature_ids=normalized_feature_ids,
            combination_ids=(
                normalized_combination_ids
            ),
            direction=direction,
            entry_rule=dict(entry_rule or {}),
            exit_rule=dict(exit_rule or {}),
            position_rule=dict(
                position_rule or {}
            ),
            parameters=dict(parameters or {}),
            status=CandidateStatus.GENERATED,
            hypothesis_id=hypothesis_id,
            parent_candidate_ids=list(
                parent_candidate_ids or []
            ),
            metadata=dict(metadata or {}),
        )

        return self.register(
            candidate,
            allow_existing=allow_existing,
        )

    # ========================================================
    # Automatic generation from combinations
    # ========================================================

    def generate_from_combinations(
        self,
        *,
        combination_ids: Optional[
            Sequence[str]
        ] = None,
        directions: Optional[
            Sequence[StrategyDirection]
        ] = None,
        name_prefix: str = "alpha",
        entry_rule: Optional[
            Dict[str, Any]
        ] = None,
        exit_rule: Optional[
            Dict[str, Any]
        ] = None,
        position_rule: Optional[
            Dict[str, Any]
        ] = None,
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        hypothesis_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CandidateGenerationResult:
        selected_ids = (
            list(combination_ids)
            if combination_ids is not None
            else [
                combination.combination_id
                for combination
                in (
                    self._feature_discovery
                    .list_combinations()
                )
            ]
        )

        if not selected_ids:
            raise CandidateSourceError(
                "No feature combinations available "
                "for candidate generation."
            )

        selected_directions = list(
            directions
            or [
                StrategyDirection.LONG,
                StrategyDirection.SHORT,
                StrategyDirection.LONG_SHORT,
            ]
        )

        if not selected_directions:
            raise AlphaValidationError(
                "At least one strategy direction "
                "is required."
            )

        for direction in selected_directions:
            if not isinstance(
                direction,
                StrategyDirection,
            ):
                raise AlphaValidationError(
                    "All directions must be "
                    "StrategyDirection values."
                )

        generated: List[
            StrategyCandidate
        ] = []

        skipped_duplicates = 0

        for combination_id in selected_ids:
            combination = (
                self._feature_discovery
                .get_combination(
                    combination_id
                )
            )

            for direction in selected_directions:
                candidate = StrategyCandidate(
                    name=(
                        f"{name_prefix}:"
                        f"{combination.name}:"
                        f"{direction.value}"
                    ),
                    feature_ids=list(
                        combination.feature_ids
                    ),
                    combination_ids=[
                        combination_id
                    ],
                    direction=direction,
                    entry_rule=dict(
                        entry_rule or {}
                    ),
                    exit_rule=dict(
                        exit_rule or {}
                    ),
                    position_rule=dict(
                        position_rule or {}
                    ),
                    parameters=dict(
                        parameters or {}
                    ),
                    status=(
                        CandidateStatus.GENERATED
                    ),
                    hypothesis_id=hypothesis_id,
                    metadata=dict(
                        metadata or {}
                    ),
                )

                existing_id = (
                    self._fingerprints.get(
                        candidate.fingerprint
                    )
                )

                if existing_id is not None:
                    skipped_duplicates += 1
                    continue

                generated.append(
                    self.register(candidate)
                )

        batch_id = stable_hash(
            {
                "combination_ids": selected_ids,
                "directions": [
                    direction.value
                    for direction
                    in selected_directions
                ],
                "candidate_ids": [
                    candidate.candidate_id
                    for candidate in generated
                ],
            }
        )[:16]

        self._record_audit(
            action="batch_generation",
            candidate_id=(
                f"batch_{batch_id}"
            ),
            details={
                "combination_count": len(
                    selected_ids
                ),
                "direction_count": len(
                    selected_directions
                ),
                "generated_count": len(
                    generated
                ),
                "skipped_duplicates": (
                    skipped_duplicates
                ),
            },
        )

        return CandidateGenerationResult(
            candidates=generated,
            skipped_duplicates=(
                skipped_duplicates
            ),
            metadata={
                "combination_ids": (
                    selected_ids
                ),
                "directions": [
                    direction.value
                    for direction
                    in selected_directions
                ],
                **dict(metadata or {}),
            },
        )

    # ========================================================
    # Generation from individual features
    # ========================================================

    def generate_from_features(
        self,
        *,
        feature_ids: Optional[
            Sequence[str]
        ] = None,
        directions: Optional[
            Sequence[StrategyDirection]
        ] = None,
        name_prefix: str = "feature_alpha",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CandidateGenerationResult:
        selected_ids = (
            list(feature_ids)
            if feature_ids is not None
            else [
                feature.feature_id
                for feature
                in (
                    self._feature_discovery
                    .list_features()
                )
            ]
        )

        if not selected_ids:
            raise CandidateSourceError(
                "No features available for "
                "candidate generation."
            )

        for feature_id in selected_ids:
            if not (
                self._feature_discovery
                .exists(feature_id)
            ):
                raise CandidateSourceError(
                    f"Feature not found: {feature_id}"
                )

        selected_directions = list(
            directions
            or [
                StrategyDirection.LONG,
                StrategyDirection.SHORT,
            ]
        )

        generated: List[
            StrategyCandidate
        ] = []

        skipped_duplicates = 0

        for feature_id in selected_ids:
            feature = (
                self._feature_discovery.get(
                    feature_id
                )
            )

            for direction in selected_directions:
                if not isinstance(
                    direction,
                    StrategyDirection,
                ):
                    raise AlphaValidationError(
                        "All directions must be "
                        "StrategyDirection values."
                    )

                candidate = StrategyCandidate(
                    name=(
                        f"{name_prefix}:"
                        f"{feature.name}:"
                        f"{direction.value}"
                    ),
                    feature_ids=[
                        feature_id
                    ],
                    direction=direction,
                    metadata=dict(
                        metadata or {}
                    ),
                )

                existing_id = (
                    self._fingerprints.get(
                        candidate.fingerprint
                    )
                )

                if existing_id is not None:
                    skipped_duplicates += 1
                    continue

                generated.append(
                    self.register(candidate)
                )

        return CandidateGenerationResult(
            candidates=generated,
            skipped_duplicates=(
                skipped_duplicates
            ),
            metadata={
                "feature_ids": selected_ids,
                "directions": [
                    direction.value
                    for direction
                    in selected_directions
                ],
                **dict(metadata or {}),
            },
        )

    # ========================================================
    # Lookup
    # ========================================================

    def exists(
        self,
        candidate_id: str,
    ) -> bool:
        return candidate_id in self._candidates

    def get(
        self,
        candidate_id: str,
    ) -> StrategyCandidate:
        try:
            return self._candidates[
                candidate_id
            ]
        except KeyError as exc:
            raise CandidateNotFoundError(
                f"Candidate not found: "
                f"{candidate_id}"
            ) from exc

    def get_by_fingerprint(
        self,
        fingerprint: str,
    ) -> StrategyCandidate:
        try:
            candidate_id = self._fingerprints[
                fingerprint
            ]
        except KeyError as exc:
            raise CandidateNotFoundError(
                "Candidate fingerprint "
                f"not found: {fingerprint}"
            ) from exc

        return self._candidates[
            candidate_id
        ]

    def list_all(
        self,
    ) -> List[StrategyCandidate]:
        return list(
            self._candidates.values()
        )

    def count(self) -> int:
        return len(self._candidates)

    def by_status(
        self,
        status: CandidateStatus,
    ) -> List[StrategyCandidate]:
        return [
            candidate
            for candidate
            in self._candidates.values()
            if candidate.status == status
        ]

    # ========================================================
    # Candidate lifecycle
    # ========================================================

    def accept(
        self,
        candidate_id: str,
    ) -> StrategyCandidate:
        candidate = self.get(
            candidate_id
        )

        candidate.accept()

        self._record_audit(
            action="accept",
            candidate_id=candidate_id,
        )

        return candidate

    def reject(
        self,
        candidate_id: str,
        *,
        reason,
        details: Optional[str] = None,
    ) -> StrategyCandidate:
        candidate = self.get(
            candidate_id
        )

        candidate.reject(
            reason,
            details,
        )

        self._record_audit(
            action="reject",
            candidate_id=candidate_id,
            details={
                "reason": reason.value,
                "details": details,
            },
        )

        return candidate

    # ========================================================
    # Batch registration
    # ========================================================

    def register_many(
        self,
        candidates: Iterable[
            StrategyCandidate
        ],
        *,
        allow_existing: bool = True,
    ) -> List[StrategyCandidate]:
        registered: List[
            StrategyCandidate
        ] = []

        for candidate in candidates:
            registered.append(
                self.register(
                    candidate,
                    allow_existing=(
                        allow_existing
                    ),
                )
            )

        return registered

    # ========================================================
    # Reproducibility
    # ========================================================

    def candidate_set_fingerprint(
        self,
        candidate_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> str:
        selected_ids = (
            list(candidate_ids)
            if candidate_ids is not None
            else list(
                self._candidates.keys()
            )
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

    def full_fingerprint(self) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "feature_discovery": (
                    self._feature_discovery
                    .full_fingerprint()
                ),
                "candidates": (
                    self.candidate_set_fingerprint()
                ),
            }
        )

    # ========================================================
    # Snapshot / summary
    # ========================================================

    def snapshot(
        self,
    ) -> CandidateGeneratorSnapshot:
        accepted_count = len(
            self.by_status(
                CandidateStatus.ACCEPTED
            )
        )

        rejected_count = len(
            self.by_status(
                CandidateStatus.REJECTED
            )
        )

        active_count = (
            self.count()
            - accepted_count
            - rejected_count
        )

        return CandidateGeneratorSnapshot(
            candidate_count=self.count(),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            active_count=active_count,
            audit_count=len(self._audit),
            candidate_ids=list(
                self._candidates.keys()
            ),
            fingerprint=(
                self.full_fingerprint()
            ),
        )

    def summary(self) -> Dict[str, Any]:
        snapshot = self.snapshot()

        return {
            "version": self.VERSION,
            "candidate_count": (
                snapshot.candidate_count
            ),
            "accepted_count": (
                snapshot.accepted_count
            ),
            "rejected_count": (
                snapshot.rejected_count
            ),
            "active_count": (
                snapshot.active_count
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
    "CandidateGenerationError",
    "CandidateAlreadyExistsError",
    "CandidateNotFoundError",
    "CandidateSourceError",
    "CandidateGenerationAuditRecord",
    "CandidateGenerationResult",
    "CandidateGeneratorSnapshot",
    "StrategyCandidateGenerator",
]