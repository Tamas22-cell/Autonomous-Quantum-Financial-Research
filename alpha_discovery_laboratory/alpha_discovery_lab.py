from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .alpha_lineage import AlphaLineageRegistry
from .candidate_generator import (
    CandidateGenerationResult,
    StrategyCandidateGenerator,
)
from .feature_discovery import (
    FeatureDiscoveryEngine,
    FeatureDiscoveryResult,
)
from .models import (
    VERSION,
    AlphaDiscoveryExperiment,
    AlphaLineageRecord,
    CandidateStatus,
    CombinationOperator,
    FeatureTransform,
    FeatureType,
    MarketFeature,
    RejectionReason,
    StrategyCandidate,
    StrategyDirection,
    stable_hash,
    utc_now,
)
from .overfit_detector import (
    OverfitDetectionResult,
    OverfitDetector,
    OverfitEvidence,
)
from .validation_pipeline import (
    AlphaValidationPipeline,
    BacktestEvidence,
    CausalEvidence,
    OOSEvidence,
    RobustnessEvidence,
    ValidationPipelineResult,
)


# ============================================================
# Exceptions
# ============================================================


class AlphaDiscoveryLabError(Exception):
    """Base exception for the Alpha Discovery Laboratory."""


class AlphaDiscoveryLabValidationError(AlphaDiscoveryLabError):
    """Raised when laboratory input is invalid."""


class AlphaDiscoveryExperimentError(AlphaDiscoveryLabError):
    """Raised when an experiment cannot be executed."""


# ============================================================
# Research Input
# ============================================================


@dataclass(frozen=True)
class AlphaResearchInput:
    name: str

    dataset_ids: List[str]

    dataset_fingerprints: List[str] = field(
        default_factory=list
    )

    feature_ids: List[str] = field(
        default_factory=list
    )

    combination_ids: List[str] = field(
        default_factory=list
    )

    directions: List[StrategyDirection] = field(
        default_factory=lambda: [
            StrategyDirection.LONG,
            StrategyDirection.SHORT,
            StrategyDirection.LONG_SHORT,
        ]
    )

    model_version_ids: List[str] = field(
        default_factory=list
    )

    prompt_version_ids: List[str] = field(
        default_factory=list
    )

    policy_version_ids: List[str] = field(
        default_factory=list
    )

    seed: Optional[int] = None
    run_id: Optional[str] = None

    configuration: Dict[str, Any] = field(
        default_factory=dict
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise AlphaDiscoveryLabValidationError(
                "name must be a non-empty string."
            )

        if not self.dataset_ids:
            raise AlphaDiscoveryLabValidationError(
                "At least one dataset ID is required."
            )

        if not self.directions:
            raise AlphaDiscoveryLabValidationError(
                "At least one strategy direction is required."
            )

        for direction in self.directions:
            if not isinstance(
                direction,
                StrategyDirection,
            ):
                raise AlphaDiscoveryLabValidationError(
                    "directions must contain "
                    "StrategyDirection values."
                )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "dataset_ids": self.dataset_ids,
                "dataset_fingerprints": (
                    self.dataset_fingerprints
                ),
                "feature_ids": self.feature_ids,
                "combination_ids": (
                    self.combination_ids
                ),
                "directions": [
                    direction.value
                    for direction in self.directions
                ],
                "model_version_ids": (
                    self.model_version_ids
                ),
                "prompt_version_ids": (
                    self.prompt_version_ids
                ),
                "policy_version_ids": (
                    self.policy_version_ids
                ),
                "seed": self.seed,
                "run_id": self.run_id,
                "configuration": self.configuration,
            }
        )


# ============================================================
# Candidate Evidence Bundle
# ============================================================


@dataclass(frozen=True)
class CandidateEvidenceBundle:
    candidate_id: str

    backtest: BacktestEvidence
    causal: CausalEvidence
    oos: OOSEvidence
    robustness: RobustnessEvidence
    overfit: OverfitEvidence

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Candidate Research Result
# ============================================================


@dataclass(frozen=True)
class CandidateResearchResult:
    candidate_id: str

    validation: ValidationPipelineResult

    overfit: Optional[
        OverfitDetectionResult
    ] = None

    lineage: Optional[
        AlphaLineageRecord
    ] = None

    accepted: bool = False

    rejection_reason: Optional[
        RejectionReason
    ] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": self.candidate_id,
                "validation": (
                    self.validation.fingerprint
                ),
                "overfit": (
                    self.overfit.fingerprint
                    if self.overfit
                    else None
                ),
                "lineage": (
                    self.lineage.fingerprint
                    if self.lineage
                    else None
                ),
                "accepted": self.accepted,
                "rejection_reason": (
                    self.rejection_reason.value
                    if self.rejection_reason
                    else None
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Laboratory Result
# ============================================================


@dataclass(frozen=True)
class AlphaDiscoveryLabResult:
    experiment_id: str

    candidate_results: List[
        CandidateResearchResult
    ]

    accepted_candidate_ids: List[str]
    rejected_candidate_ids: List[str]

    started_at: datetime
    completed_at: datetime

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_results)

    @property
    def accepted_count(self) -> int:
        return len(
            self.accepted_candidate_ids
        )

    @property
    def rejected_count(self) -> int:
        return len(
            self.rejected_candidate_ids
        )

    @property
    def acceptance_rate(self) -> float:
        if self.candidate_count == 0:
            return 0.0

        return (
            self.accepted_count
            / self.candidate_count
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "experiment_id": (
                    self.experiment_id
                ),
                "candidate_results": [
                    result.fingerprint
                    for result
                    in self.candidate_results
                ],
                "accepted_candidate_ids": (
                    self.accepted_candidate_ids
                ),
                "rejected_candidate_ids": (
                    self.rejected_candidate_ids
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Audit
# ============================================================


@dataclass(frozen=True)
class AlphaDiscoveryLabAuditRecord:
    action: str
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
                "object_id": self.object_id,
                "timestamp": (
                    self.timestamp.isoformat()
                ),
                "details": self.details,
            }
        )


# ============================================================
# Alpha Discovery Laboratory
# ============================================================


class AlphaDiscoveryLaboratory:
    """
    v6.0.3 Alpha Discovery Laboratory.

    Full research flow:

        Market Features
            ->
        Feature Combinations
            ->
        Strategy Candidates
            ->
        Backtest
            ->
        Causal Validation
            ->
        Out-of-Sample Validation
            ->
        Robustness Validation
            ->
        Overfit / Stability Detection
            ->
        Accept / Reject
            ->
        Alpha Lineage
            ->
        Reproducible Experiment Record

    Research-only architecture.
    No autonomous real-money execution.
    """

    VERSION = VERSION

    def __init__(
        self,
        *,
        feature_discovery: Optional[
            FeatureDiscoveryEngine
        ] = None,
        candidate_generator: Optional[
            StrategyCandidateGenerator
        ] = None,
        validation_pipeline: Optional[
            AlphaValidationPipeline
        ] = None,
        overfit_detector: Optional[
            OverfitDetector
        ] = None,
        lineage_registry: Optional[
            AlphaLineageRegistry
        ] = None,
    ) -> None:
        self.feature_discovery = (
            feature_discovery
            or FeatureDiscoveryEngine()
        )

        if candidate_generator is None:
            candidate_generator = (
                StrategyCandidateGenerator(
                    self.feature_discovery
                )
            )

        self.candidate_generator = (
            candidate_generator
        )

        self.validation_pipeline = (
            validation_pipeline
            or AlphaValidationPipeline()
        )

        self.overfit_detector = (
            overfit_detector
            or OverfitDetector()
        )

        self.lineage_registry = (
            lineage_registry
            or AlphaLineageRegistry()
        )

        self._results: Dict[
            str,
            AlphaDiscoveryLabResult,
        ] = {}

        self._audit: List[
            AlphaDiscoveryLabAuditRecord
        ] = []

    # ========================================================
    # Audit
    # ========================================================

    def _record_audit(
        self,
        *,
        action: str,
        object_id: str,
        details: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        self._audit.append(
            AlphaDiscoveryLabAuditRecord(
                action=action,
                object_id=object_id,
                details=dict(details or {}),
            )
        )

    def audit_log(
        self,
    ) -> List[AlphaDiscoveryLabAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Feature creation
    # ========================================================

    def create_feature(
        self,
        *,
        name: str,
        feature_type: FeatureType,
        source: str,
        transform: FeatureTransform = (
            FeatureTransform.RAW
        ),
        parameters: Optional[
            Dict[str, Any]
        ] = None,
        description: str = "",
        dataset_id: Optional[str] = None,
        dataset_fingerprint: Optional[
            str
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> MarketFeature:
        return self.feature_discovery.create_feature(
            name=name,
            feature_type=feature_type,
            source=source,
            transform=transform,
            parameters=parameters,
            description=description,
            dataset_id=dataset_id,
            dataset_fingerprint=(
                dataset_fingerprint
            ),
            metadata=metadata,
        )

    # ========================================================
    # Feature discovery
    # ========================================================

    def discover_features(
        self,
        *,
        feature_ids: Optional[
            Sequence[str]
        ] = None,
        operators: Optional[
            Sequence[CombinationOperator]
        ] = None,
        name_prefix: str = "discovered",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> FeatureDiscoveryResult:
        result = (
            self.feature_discovery
            .discover_pairwise(
                feature_ids=feature_ids,
                operators=operators,
                name_prefix=name_prefix,
                metadata=metadata,
            )
        )

        self._record_audit(
            action="feature_discovery",
            object_id=result.fingerprint,
            details={
                "feature_count": (
                    result.feature_count
                ),
                "combination_count": (
                    result.combination_count
                ),
                "skipped_duplicates": (
                    result.skipped_duplicates
                ),
            },
        )

        return result

    # ========================================================
    # Candidate generation
    # ========================================================

    def generate_candidates(
        self,
        *,
        combination_ids: Optional[
            Sequence[str]
        ] = None,
        directions: Optional[
            Sequence[StrategyDirection]
        ] = None,
        name_prefix: str = "alpha",
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CandidateGenerationResult:
        result = (
            self.candidate_generator
            .generate_from_combinations(
                combination_ids=(
                    combination_ids
                ),
                directions=directions,
                name_prefix=name_prefix,
                metadata=metadata,
            )
        )

        self._record_audit(
            action="candidate_generation",
            object_id=result.fingerprint,
            details={
                "candidate_count": (
                    result.candidate_count
                ),
                "skipped_duplicates": (
                    result.skipped_duplicates
                ),
            },
        )

        return result

    # ========================================================
    # Single candidate validation
    # ========================================================

    def evaluate_candidate(
        self,
        *,
        candidate: StrategyCandidate,
        evidence: CandidateEvidenceBundle,
        experiment_id: Optional[
            str
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
        run_id: Optional[str] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> CandidateResearchResult:
        if (
            evidence.candidate_id
            != candidate.candidate_id
        ):
            raise AlphaDiscoveryLabValidationError(
                "Evidence candidate ID does not "
                "match candidate."
            )

        validation = (
            self.validation_pipeline.run(
                candidate,
                backtest=evidence.backtest,
                causal=evidence.causal,
                oos=evidence.oos,
                robustness=evidence.robustness,
                metadata=evidence.metadata,
            )
        )

        overfit_result: Optional[
            OverfitDetectionResult
        ] = None

        accepted = validation.accepted

        rejection_reason = (
            validation.rejection_reason
        )

        if accepted:
            overfit_result = (
                self.overfit_detector
                .detect_and_apply(
                    candidate,
                    evidence.overfit,
                )
            )

            if not overfit_result.accepted:
                accepted = False

                rejection_reason = (
                    overfit_result
                    .rejection_reason
                    or RejectionReason.OTHER
                )

        if accepted:
            candidate.set_status(
                CandidateStatus.ACCEPTED
            )
        elif not candidate.rejected:
            candidate.reject(
                rejection_reason
                or RejectionReason.OTHER,
                "Rejected by Alpha Discovery "
                "Laboratory.",
            )

        lineage = (
            self.lineage_registry
            .create_lineage(
                candidate=candidate,
                validation_results=(
                    validation.results
                ),
                dataset_ids=dataset_ids,
                dataset_fingerprints=(
                    dataset_fingerprints
                ),
                model_version_ids=(
                    model_version_ids
                ),
                prompt_version_ids=(
                    prompt_version_ids
                ),
                policy_version_ids=(
                    policy_version_ids
                ),
                experiment_id=experiment_id,
                run_id=run_id,
                metadata={
                    "validation_fingerprint": (
                        validation.fingerprint
                    ),
                    "overfit_fingerprint": (
                        overfit_result.fingerprint
                        if overfit_result
                        else None
                    ),
                    **dict(metadata or {}),
                },
            )
        )

        result = CandidateResearchResult(
            candidate_id=(
                candidate.candidate_id
            ),
            validation=validation,
            overfit=overfit_result,
            lineage=lineage,
            accepted=accepted,
            rejection_reason=(
                rejection_reason
            ),
            metadata=dict(metadata or {}),
        )

        self._record_audit(
            action="candidate_evaluation",
            object_id=(
                candidate.candidate_id
            ),
            details={
                "accepted": accepted,
                "rejection_reason": (
                    rejection_reason.value
                    if rejection_reason
                    else None
                ),
                "result_fingerprint": (
                    result.fingerprint
                ),
            },
        )

        return result

    # ========================================================
    # Experiment creation
    # ========================================================

    def create_experiment(
        self,
        research_input: AlphaResearchInput,
        *,
        candidate_ids: Optional[
            Sequence[str]
        ] = None,
    ) -> AlphaDiscoveryExperiment:
        if not isinstance(
            research_input,
            AlphaResearchInput,
        ):
            raise AlphaDiscoveryLabValidationError(
                "research_input must be "
                "AlphaResearchInput."
            )

        return (
            self.lineage_registry
            .create_experiment(
                name=research_input.name,
                dataset_ids=(
                    research_input.dataset_ids
                ),
                feature_ids=(
                    research_input.feature_ids
                ),
                combination_ids=(
                    research_input
                    .combination_ids
                ),
                candidate_ids=list(
                    candidate_ids or []
                ),
                configuration={
                    **dict(
                        research_input
                        .configuration
                    ),
                    "research_input_fingerprint": (
                        research_input
                        .fingerprint
                    ),
                    "validation_configuration": (
                        self.validation_pipeline
                        .configuration_fingerprint()
                    ),
                    "overfit_configuration": (
                        self.overfit_detector
                        .configuration_fingerprint()
                    ),
                },
                seed=research_input.seed,
                run_id=research_input.run_id,
                metadata=(
                    research_input.metadata
                ),
            )
        )

    # ========================================================
    # Complete experiment
    # ========================================================

    def run_experiment(
        self,
        *,
        research_input: AlphaResearchInput,
        evidence_by_candidate: Dict[
            str,
            CandidateEvidenceBundle,
        ],
    ) -> AlphaDiscoveryLabResult:
        started_at = utc_now()

        candidate_ids = list(
            evidence_by_candidate.keys()
        )

        experiment = self.create_experiment(
            research_input,
            candidate_ids=candidate_ids,
        )

        results: List[
            CandidateResearchResult
        ] = []

        accepted_ids: List[str] = []
        rejected_ids: List[str] = []

        for candidate_id in candidate_ids:
            if not (
                self.candidate_generator
                .exists(candidate_id)
            ):
                raise AlphaDiscoveryExperimentError(
                    "Candidate not registered: "
                    f"{candidate_id}"
                )

            candidate = (
                self.candidate_generator.get(
                    candidate_id
                )
            )

            evidence = (
                evidence_by_candidate[
                    candidate_id
                ]
            )

            result = self.evaluate_candidate(
                candidate=candidate,
                evidence=evidence,
                experiment_id=(
                    experiment.experiment_id
                ),
                dataset_ids=(
                    research_input.dataset_ids
                ),
                dataset_fingerprints=(
                    research_input
                    .dataset_fingerprints
                ),
                model_version_ids=(
                    research_input
                    .model_version_ids
                ),
                prompt_version_ids=(
                    research_input
                    .prompt_version_ids
                ),
                policy_version_ids=(
                    research_input
                    .policy_version_ids
                ),
                run_id=research_input.run_id,
                metadata={
                    "experiment_name": (
                        research_input.name
                    )
                },
            )

            results.append(result)

            if result.accepted:
                accepted_ids.append(
                    candidate_id
                )
            else:
                rejected_ids.append(
                    candidate_id
                )

        experiment.accepted_candidate_ids.extend(
            accepted_ids
        )

        experiment.rejected_candidate_ids.extend(
            rejected_ids
        )

        experiment.completed_at = utc_now()

        final_result = AlphaDiscoveryLabResult(
            experiment_id=(
                experiment.experiment_id
            ),
            candidate_results=results,
            accepted_candidate_ids=(
                accepted_ids
            ),
            rejected_candidate_ids=(
                rejected_ids
            ),
            started_at=started_at,
            completed_at=utc_now(),
            metadata={
                "research_input_fingerprint": (
                    research_input.fingerprint
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

        self._results[
            experiment.experiment_id
        ] = final_result

        self._record_audit(
            action="experiment_completed",
            object_id=(
                experiment.experiment_id
            ),
            details={
                "candidate_count": (
                    final_result
                    .candidate_count
                ),
                "accepted_count": (
                    final_result
                    .accepted_count
                ),
                "rejected_count": (
                    final_result
                    .rejected_count
                ),
                "acceptance_rate": (
                    final_result
                    .acceptance_rate
                ),
                "fingerprint": (
                    final_result
                    .fingerprint
                ),
            },
        )

        return final_result

    # ========================================================
    # Results
    # ========================================================

    def get_result(
        self,
        experiment_id: str,
    ) -> AlphaDiscoveryLabResult:
        try:
            return self._results[
                experiment_id
            ]
        except KeyError as exc:
            raise AlphaDiscoveryExperimentError(
                "Experiment result not found: "
                f"{experiment_id}"
            ) from exc

    def results(
        self,
    ) -> List[AlphaDiscoveryLabResult]:
        return list(
            self._results.values()
        )

    # ========================================================
    # Reproducibility
    # ========================================================

    def configuration_fingerprint(
        self,
    ) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "feature_discovery": (
                    self.feature_discovery
                    .full_fingerprint()
                ),
                "candidate_generator": (
                    self.candidate_generator
                    .full_fingerprint()
                ),
                "validation": (
                    self.validation_pipeline
                    .configuration_fingerprint()
                ),
                "overfit": (
                    self.overfit_detector
                    .configuration_fingerprint()
                ),
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        results = self.results()

        total_candidates = sum(
            result.candidate_count
            for result in results
        )

        total_accepted = sum(
            result.accepted_count
            for result in results
        )

        total_rejected = sum(
            result.rejected_count
            for result in results
        )

        return {
            "version": self.VERSION,
            "experiments": len(results),
            "features": (
                self.feature_discovery
                .feature_count()
            ),
            "feature_combinations": (
                self.feature_discovery
                .combination_count()
            ),
            "strategy_candidates": (
                self.candidate_generator
                .count()
            ),
            "evaluated_candidates": (
                total_candidates
            ),
            "accepted_candidates": (
                total_accepted
            ),
            "rejected_candidates": (
                total_rejected
            ),
            "lineage_records": (
                self.lineage_registry
                .lineage_count()
            ),
            "audit_records": len(
                self._audit
            ),
            "configuration_fingerprint": (
                self.configuration_fingerprint()
            ),
            "research_only": True,
            "real_money_execution": False,
        }


# ============================================================
# Public exports
# ============================================================


__all__ = [
    "AlphaDiscoveryLabError",
    "AlphaDiscoveryLabValidationError",
    "AlphaDiscoveryExperimentError",
    "AlphaResearchInput",
    "CandidateEvidenceBundle",
    "CandidateResearchResult",
    "AlphaDiscoveryLabResult",
    "AlphaDiscoveryLabAuditRecord",
    "AlphaDiscoveryLaboratory",
]