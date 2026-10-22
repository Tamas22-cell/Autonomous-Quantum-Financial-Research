from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    VERSION,
    AlphaValidationError,
    CombinationOperator,
    FeatureCombination,
    FeatureTransform,
    FeatureType,
    MarketFeature,
    stable_hash,
    utc_now,
)


# ============================================================
# Exceptions
# ============================================================


class FeatureDiscoveryError(Exception):
    """Base exception for feature discovery."""


class FeatureAlreadyExistsError(FeatureDiscoveryError):
    """Raised when duplicate registration is disallowed."""


class FeatureNotFoundError(FeatureDiscoveryError):
    """Raised when a requested feature cannot be found."""


class CombinationAlreadyExistsError(FeatureDiscoveryError):
    """Raised when duplicate combination registration is disallowed."""


class CombinationNotFoundError(FeatureDiscoveryError):
    """Raised when a requested combination cannot be found."""


# ============================================================
# Audit / result models
# ============================================================


@dataclass(frozen=True)
class FeatureDiscoveryAuditRecord:
    action: str
    object_type: str
    object_id: str
    timestamp: datetime = field(default_factory=utc_now)
    details: Dict[str, Any] = field(default_factory=dict)

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
class FeatureDiscoveryResult:
    features: List[MarketFeature]
    combinations: List[FeatureCombination]
    skipped_duplicates: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feature_count(self) -> int:
        return len(self.features)

    @property
    def combination_count(self) -> int:
        return len(self.combinations)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "feature_ids": [
                    feature.feature_id
                    for feature in self.features
                ],
                "combination_ids": [
                    combination.combination_id
                    for combination in self.combinations
                ],
                "skipped_duplicates": self.skipped_duplicates,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class FeatureDiscoverySnapshot:
    feature_count: int
    combination_count: int
    audit_count: int
    feature_ids: List[str]
    combination_ids: List[str]
    fingerprint: str
    created_at: datetime = field(default_factory=utc_now)


# ============================================================
# Feature Discovery Engine
# ============================================================


class FeatureDiscoveryEngine:
    """
    Deterministic market-feature and feature-combination discovery.

    Responsibilities:
    - register individual market features
    - create deterministic feature definitions
    - generate pairwise feature combinations
    - reject invalid feature references
    - prevent duplicate discovery records
    - expose reproducible fingerprints
    - maintain a complete audit history

    This module performs research-definition discovery only.
    It does not execute real-money trading.
    """

    VERSION = VERSION

    def __init__(self) -> None:
        self._features: Dict[str, MarketFeature] = {}
        self._feature_fingerprints: Dict[str, str] = {}

        self._combinations: Dict[str, FeatureCombination] = {}
        self._combination_fingerprints: Dict[str, str] = {}

        self._audit: List[FeatureDiscoveryAuditRecord] = []

    # ========================================================
    # Validation helpers
    # ========================================================

    @staticmethod
    def _require_text(value: str, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise AlphaValidationError(
                f"{field_name} must be a non-empty string."
            )
        return value.strip()

    @staticmethod
    def _validate_feature(feature: MarketFeature) -> None:
        if not isinstance(feature, MarketFeature):
            raise AlphaValidationError(
                "feature must be a MarketFeature instance."
            )

        FeatureDiscoveryEngine._require_text(
            feature.name,
            "feature.name",
        )
        FeatureDiscoveryEngine._require_text(
            feature.source,
            "feature.source",
        )

    @staticmethod
    def _validate_combination(
        combination: FeatureCombination,
    ) -> None:
        if not isinstance(
            combination,
            FeatureCombination,
        ):
            raise AlphaValidationError(
                "combination must be a "
                "FeatureCombination instance."
            )

        FeatureDiscoveryEngine._require_text(
            combination.name,
            "combination.name",
        )

        if len(combination.feature_ids) < 2:
            raise AlphaValidationError(
                "A feature combination requires "
                "at least two feature IDs."
            )

    # ========================================================
    # Audit
    # ========================================================

    def _record_audit(
        self,
        *,
        action: str,
        object_type: str,
        object_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._audit.append(
            FeatureDiscoveryAuditRecord(
                action=action,
                object_type=object_type,
                object_id=object_id,
                details=dict(details or {}),
            )
        )

    def audit_log(self) -> List[FeatureDiscoveryAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Feature registration
    # ========================================================

    def register_feature(
        self,
        feature: MarketFeature,
        *,
        allow_existing: bool = True,
    ) -> MarketFeature:
        self._validate_feature(feature)

        existing_id = self._feature_fingerprints.get(
            feature.fingerprint
        )

        if existing_id is not None:
            if allow_existing:
                return self._features[existing_id]

            raise FeatureAlreadyExistsError(
                f"Feature already exists: {existing_id}"
            )

        if feature.feature_id in self._features:
            existing = self._features[feature.feature_id]

            if existing.fingerprint == feature.fingerprint:
                if allow_existing:
                    return existing

                raise FeatureAlreadyExistsError(
                    f"Feature already exists: "
                    f"{feature.feature_id}"
                )

            raise FeatureDiscoveryError(
                "Feature ID collision detected."
            )

        self._features[feature.feature_id] = feature
        self._feature_fingerprints[
            feature.fingerprint
        ] = feature.feature_id

        self._record_audit(
            action="register",
            object_type="feature",
            object_id=feature.feature_id,
            details={
                "name": feature.name,
                "feature_type": feature.feature_type.value,
                "transform": feature.transform.value,
                "source": feature.source,
            },
        )

        return feature

    def create_feature(
        self,
        *,
        name: str,
        feature_type: FeatureType,
        source: str,
        transform: FeatureTransform = FeatureTransform.RAW,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        dataset_id: Optional[str] = None,
        dataset_fingerprint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        allow_existing: bool = True,
    ) -> MarketFeature:
        name = self._require_text(name, "name")
        source = self._require_text(source, "source")

        if not isinstance(feature_type, FeatureType):
            raise AlphaValidationError(
                "feature_type must be FeatureType."
            )

        if not isinstance(transform, FeatureTransform):
            raise AlphaValidationError(
                "transform must be FeatureTransform."
            )

        feature = MarketFeature(
            name=name,
            feature_type=feature_type,
            source=source,
            transform=transform,
            parameters=dict(parameters or {}),
            description=description,
            dataset_id=dataset_id,
            dataset_fingerprint=dataset_fingerprint,
            metadata=dict(metadata or {}),
        )

        return self.register_feature(
            feature,
            allow_existing=allow_existing,
        )

    # ========================================================
    # Feature lookup
    # ========================================================

    def exists(self, feature_id: str) -> bool:
        return feature_id in self._features

    def get(self, feature_id: str) -> MarketFeature:
        try:
            return self._features[feature_id]
        except KeyError as exc:
            raise FeatureNotFoundError(
                f"Feature not found: {feature_id}"
            ) from exc

    def get_by_fingerprint(
        self,
        fingerprint: str,
    ) -> MarketFeature:
        try:
            feature_id = self._feature_fingerprints[
                fingerprint
            ]
        except KeyError as exc:
            raise FeatureNotFoundError(
                "Feature fingerprint not found: "
                f"{fingerprint}"
            ) from exc

        return self._features[feature_id]

    def list_features(self) -> List[MarketFeature]:
        return list(self._features.values())

    def feature_count(self) -> int:
        return len(self._features)

    def features_by_type(
        self,
        feature_type: FeatureType,
    ) -> List[MarketFeature]:
        return [
            feature
            for feature in self._features.values()
            if feature.feature_type == feature_type
        ]

    def features_by_source(
        self,
        source: str,
    ) -> List[MarketFeature]:
        return [
            feature
            for feature in self._features.values()
            if feature.source == source
        ]

    def features_by_dataset(
        self,
        dataset_id: str,
    ) -> List[MarketFeature]:
        return [
            feature
            for feature in self._features.values()
            if feature.dataset_id == dataset_id
        ]

    # ========================================================
    # Combination registration
    # ========================================================

    def register_combination(
        self,
        combination: FeatureCombination,
        *,
        allow_existing: bool = True,
    ) -> FeatureCombination:
        self._validate_combination(combination)

        for feature_id in combination.feature_ids:
            if feature_id not in self._features:
                raise FeatureNotFoundError(
                    "Cannot register combination. "
                    f"Feature not found: {feature_id}"
                )

        existing_id = self._combination_fingerprints.get(
            combination.fingerprint
        )

        if existing_id is not None:
            if allow_existing:
                return self._combinations[existing_id]

            raise CombinationAlreadyExistsError(
                "Feature combination already exists: "
                f"{existing_id}"
            )

        if (
            combination.combination_id
            in self._combinations
        ):
            existing = self._combinations[
                combination.combination_id
            ]

            if (
                existing.fingerprint
                == combination.fingerprint
            ):
                if allow_existing:
                    return existing

                raise CombinationAlreadyExistsError(
                    "Feature combination already exists: "
                    f"{combination.combination_id}"
                )

            raise FeatureDiscoveryError(
                "Combination ID collision detected."
            )

        self._combinations[
            combination.combination_id
        ] = combination

        self._combination_fingerprints[
            combination.fingerprint
        ] = combination.combination_id

        self._record_audit(
            action="register",
            object_type="combination",
            object_id=combination.combination_id,
            details={
                "name": combination.name,
                "feature_ids": list(
                    combination.feature_ids
                ),
                "operator": (
                    combination.operator.value
                ),
            },
        )

        return combination

    def create_combination(
        self,
        *,
        name: str,
        feature_ids: Sequence[str],
        operator: CombinationOperator,
        weights: Optional[Sequence[float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        allow_existing: bool = True,
    ) -> FeatureCombination:
        name = self._require_text(name, "name")

        if not isinstance(
            operator,
            CombinationOperator,
        ):
            raise AlphaValidationError(
                "operator must be CombinationOperator."
            )

        normalized_feature_ids = list(feature_ids)

        if len(normalized_feature_ids) < 2:
            raise AlphaValidationError(
                "At least two feature IDs are required."
            )

        for feature_id in normalized_feature_ids:
            if feature_id not in self._features:
                raise FeatureNotFoundError(
                    f"Feature not found: {feature_id}"
                )

        combination = FeatureCombination(
            name=name,
            feature_ids=normalized_feature_ids,
            operator=operator,
            weights=list(weights or []),
            parameters=dict(parameters or {}),
            description=description,
            metadata=dict(metadata or {}),
        )

        return self.register_combination(
            combination,
            allow_existing=allow_existing,
        )

    # ========================================================
    # Combination lookup
    # ========================================================

    def combination_exists(
        self,
        combination_id: str,
    ) -> bool:
        return combination_id in self._combinations

    def get_combination(
        self,
        combination_id: str,
    ) -> FeatureCombination:
        try:
            return self._combinations[
                combination_id
            ]
        except KeyError as exc:
            raise CombinationNotFoundError(
                "Feature combination not found: "
                f"{combination_id}"
            ) from exc

    def list_combinations(
        self,
    ) -> List[FeatureCombination]:
        return list(self._combinations.values())

    def combination_count(self) -> int:
        return len(self._combinations)

    def combinations_for_feature(
        self,
        feature_id: str,
    ) -> List[FeatureCombination]:
        if feature_id not in self._features:
            raise FeatureNotFoundError(
                f"Feature not found: {feature_id}"
            )

        return [
            combination
            for combination
            in self._combinations.values()
            if feature_id
            in combination.feature_ids
        ]

    # ========================================================
    # Automatic pairwise discovery
    # ========================================================

    def discover_pairwise(
        self,
        *,
        feature_ids: Optional[
            Sequence[str]
        ] = None,
        operators: Optional[
            Sequence[CombinationOperator]
        ] = None,
        name_prefix: str = "discovered",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FeatureDiscoveryResult:
        selected_ids = (
            list(feature_ids)
            if feature_ids is not None
            else list(self._features.keys())
        )

        if len(selected_ids) < 2:
            raise AlphaValidationError(
                "Pairwise discovery requires "
                "at least two features."
            )

        for feature_id in selected_ids:
            if feature_id not in self._features:
                raise FeatureNotFoundError(
                    f"Feature not found: {feature_id}"
                )

        selected_operators = list(
            operators
            or [
                CombinationOperator.SPREAD,
                CombinationOperator.RATIO,
                CombinationOperator.INTERACTION,
            ]
        )

        if not selected_operators:
            raise AlphaValidationError(
                "At least one combination operator "
                "is required."
            )

        for operator in selected_operators:
            if not isinstance(
                operator,
                CombinationOperator,
            ):
                raise AlphaValidationError(
                    "All operators must be "
                    "CombinationOperator values."
                )

        created: List[FeatureCombination] = []
        skipped_duplicates = 0

        for left_id, right_id in combinations(
            selected_ids,
            2,
        ):
            left = self._features[left_id]
            right = self._features[right_id]

            for operator in selected_operators:
                combination = FeatureCombination(
                    name=(
                        f"{name_prefix}:"
                        f"{left.name}:"
                        f"{operator.value}:"
                        f"{right.name}"
                    ),
                    feature_ids=[
                        left_id,
                        right_id,
                    ],
                    operator=operator,
                    metadata=dict(metadata or {}),
                )

                existing_id = (
                    self._combination_fingerprints.get(
                        combination.fingerprint
                    )
                )

                if existing_id is not None:
                    skipped_duplicates += 1
                    continue

                registered = (
                    self.register_combination(
                        combination
                    )
                )
                created.append(registered)

        self._record_audit(
            action="pairwise_discovery",
            object_type="discovery",
            object_id=stable_hash(
                {
                    "feature_ids": selected_ids,
                    "operators": [
                        operator.value
                        for operator
                        in selected_operators
                    ],
                    "created": [
                        item.combination_id
                        for item in created
                    ],
                }
            )[:16],
            details={
                "selected_feature_count": len(
                    selected_ids
                ),
                "created_count": len(created),
                "skipped_duplicates": (
                    skipped_duplicates
                ),
            },
        )

        return FeatureDiscoveryResult(
            features=[
                self._features[feature_id]
                for feature_id in selected_ids
            ],
            combinations=created,
            skipped_duplicates=skipped_duplicates,
            metadata={
                "operators": [
                    operator.value
                    for operator
                    in selected_operators
                ],
                **dict(metadata or {}),
            },
        )

    # ========================================================
    # Batch registration
    # ========================================================

    def register_features(
        self,
        features: Iterable[MarketFeature],
        *,
        allow_existing: bool = True,
    ) -> List[MarketFeature]:
        registered: List[MarketFeature] = []

        for feature in features:
            registered.append(
                self.register_feature(
                    feature,
                    allow_existing=allow_existing,
                )
            )

        return registered

    # ========================================================
    # Reproducibility
    # ========================================================

    def feature_set_fingerprint(
        self,
        feature_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> str:
        selected_ids = (
            list(feature_ids)
            if feature_ids is not None
            else list(self._features.keys())
        )

        fingerprints: List[str] = []

        for feature_id in selected_ids:
            fingerprints.append(
                self.get(feature_id).fingerprint
            )

        return stable_hash(
            sorted(fingerprints)
        )

    def combination_set_fingerprint(
        self,
        combination_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> str:
        selected_ids = (
            list(combination_ids)
            if combination_ids is not None
            else list(self._combinations.keys())
        )

        fingerprints: List[str] = []

        for combination_id in selected_ids:
            fingerprints.append(
                self.get_combination(
                    combination_id
                ).fingerprint
            )

        return stable_hash(
            sorted(fingerprints)
        )

    def full_fingerprint(self) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "features": (
                    self.feature_set_fingerprint()
                ),
                "combinations": (
                    self.combination_set_fingerprint()
                ),
            }
        )

    # ========================================================
    # Snapshot / summary
    # ========================================================

    def snapshot(self) -> FeatureDiscoverySnapshot:
        return FeatureDiscoverySnapshot(
            feature_count=self.feature_count(),
            combination_count=(
                self.combination_count()
            ),
            audit_count=len(self._audit),
            feature_ids=list(
                self._features.keys()
            ),
            combination_ids=list(
                self._combinations.keys()
            ),
            fingerprint=self.full_fingerprint(),
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "feature_count": self.feature_count(),
            "combination_count": (
                self.combination_count()
            ),
            "audit_count": len(self._audit),
            "fingerprint": (
                self.full_fingerprint()
            ),
            "research_only": True,
            "real_money_execution": False,
        }


# ============================================================
# Public exports
# ============================================================


__all__ = [
    "FeatureDiscoveryError",
    "FeatureAlreadyExistsError",
    "FeatureNotFoundError",
    "CombinationAlreadyExistsError",
    "CombinationNotFoundError",
    "FeatureDiscoveryAuditRecord",
    "FeatureDiscoveryResult",
    "FeatureDiscoverySnapshot",
    "FeatureDiscoveryEngine",
]