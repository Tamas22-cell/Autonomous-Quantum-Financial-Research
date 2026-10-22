from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

from .models import (
    VERSION,
    AlphaValidationError,
    CandidateStatus,
    PerformanceMetrics,
    RejectionReason,
    StrategyCandidate,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    stable_hash,
    utc_now,
)


# ============================================================
# Exceptions
# ============================================================


class ValidationPipelineError(Exception):
    """Base exception for alpha validation pipeline."""


class ValidationStageError(ValidationPipelineError):
    """Raised when a validation stage cannot be executed."""


class ValidationConfigurationError(ValidationPipelineError):
    """Raised when validation thresholds are invalid."""


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class ValidationThresholds:
    min_sharpe_ratio: float = 0.75
    min_profit_factor: float = 1.05
    max_drawdown: float = 0.30
    min_trade_count: int = 20

    min_causal_score: float = 0.50

    min_oos_sharpe_ratio: float = 0.40
    min_oos_retention: float = 0.40

    min_robustness_score: float = 0.50
    max_parameter_sensitivity: float = 0.50

    max_turnover: Optional[float] = None
    max_execution_cost: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min_trade_count < 0:
            raise ValidationConfigurationError(
                "min_trade_count cannot be negative."
            )

        bounded_values = {
            "max_drawdown": self.max_drawdown,
            "min_causal_score": self.min_causal_score,
            "min_oos_retention": self.min_oos_retention,
            "min_robustness_score": self.min_robustness_score,
            "max_parameter_sensitivity": (
                self.max_parameter_sensitivity
            ),
        }

        for name, value in bounded_values.items():
            if value < 0.0 or value > 1.0:
                raise ValidationConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        if (
            self.max_turnover is not None
            and self.max_turnover < 0
        ):
            raise ValidationConfigurationError(
                "max_turnover cannot be negative."
            )

        if (
            self.max_execution_cost is not None
            and self.max_execution_cost < 0
        ):
            raise ValidationConfigurationError(
                "max_execution_cost cannot be negative."
            )


# ============================================================
# Stage input models
# ============================================================


@dataclass(frozen=True)
class BacktestEvidence:
    metrics: PerformanceMetrics
    data_leakage_detected: bool = False
    lookahead_bias_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CausalEvidence:
    causal_score: float
    stable_effect: bool = True
    confounding_detected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.causal_score < 0.0 or self.causal_score > 1.0:
            raise AlphaValidationError(
                "causal_score must be between 0 and 1."
            )


@dataclass(frozen=True)
class OOSEvidence:
    metrics: PerformanceMetrics
    performance_retention: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.performance_retention < 0.0:
            raise AlphaValidationError(
                "performance_retention cannot be negative."
            )


@dataclass(frozen=True)
class RobustnessEvidence:
    robustness_score: float
    parameter_sensitivity: float
    regime_stable: bool = True
    perturbation_stable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.robustness_score < 0.0
            or self.robustness_score > 1.0
        ):
            raise AlphaValidationError(
                "robustness_score must be between 0 and 1."
            )

        if (
            self.parameter_sensitivity < 0.0
            or self.parameter_sensitivity > 1.0
        ):
            raise AlphaValidationError(
                "parameter_sensitivity must be between 0 and 1."
            )


# ============================================================
# Pipeline result / audit
# ============================================================


@dataclass(frozen=True)
class ValidationPipelineAuditRecord:
    candidate_id: str
    stage: ValidationStage
    status: ValidationStatus
    timestamp: datetime = field(default_factory=utc_now)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": self.candidate_id,
                "stage": self.stage.value,
                "status": self.status.value,
                "timestamp": self.timestamp.isoformat(),
                "details": self.details,
            }
        )


@dataclass(frozen=True)
class ValidationPipelineResult:
    candidate_id: str
    accepted: bool
    results: List[ValidationResult]

    rejection_reason: Optional[RejectionReason] = None
    stopped_at: Optional[ValidationStage] = None

    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed_stage_count(self) -> int:
        return sum(
            result.status == ValidationStatus.PASSED
            for result in self.results
        )

    @property
    def failed_stage_count(self) -> int:
        return sum(
            result.status
            in {
                ValidationStatus.FAILED,
                ValidationStatus.REJECTED,
                ValidationStatus.ERROR,
            }
            for result in self.results
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": self.candidate_id,
                "accepted": self.accepted,
                "results": [
                    result.fingerprint
                    for result in self.results
                ],
                "rejection_reason": (
                    self.rejection_reason.value
                    if self.rejection_reason
                    else None
                ),
                "stopped_at": (
                    self.stopped_at.value
                    if self.stopped_at
                    else None
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Validation Pipeline
# ============================================================


class AlphaValidationPipeline:
    """
    Institutional-style alpha validation pipeline.

    Pipeline:
        Backtest
            -> Causal Validation
            -> Out-of-Sample Validation
            -> Robustness Validation
            -> Accept / Reject

    Candidates fail closed: if a stage fails, later stages are
    not accepted implicitly.

    This engine is research-only and performs no live execution.
    """

    VERSION = VERSION

    STAGE_ORDER = (
        ValidationStage.BACKTEST,
        ValidationStage.CAUSAL,
        ValidationStage.OUT_OF_SAMPLE,
        ValidationStage.ROBUSTNESS,
    )

    def __init__(
        self,
        thresholds: Optional[ValidationThresholds] = None,
    ) -> None:
        self.thresholds = (
            thresholds or ValidationThresholds()
        )

        self._history: Dict[
            str,
            List[ValidationPipelineResult],
        ] = {}

        self._audit: List[
            ValidationPipelineAuditRecord
        ] = []

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _require_candidate(
        candidate: StrategyCandidate,
    ) -> None:
        if not isinstance(candidate, StrategyCandidate):
            raise AlphaValidationError(
                "candidate must be a "
                "StrategyCandidate instance."
            )

    def _record(
        self,
        result: ValidationResult,
    ) -> None:
        self._audit.append(
            ValidationPipelineAuditRecord(
                candidate_id=result.candidate_id,
                stage=result.stage,
                status=result.status,
                details={
                    "score": result.score,
                    "passed_checks": list(
                        result.passed_checks
                    ),
                    "failed_checks": list(
                        result.failed_checks
                    ),
                    "rejection_reason": (
                        result.rejection_reason.value
                        if result.rejection_reason
                        else None
                    ),
                },
            )
        )

    @staticmethod
    def _failed_result(
        *,
        candidate_id: str,
        stage: ValidationStage,
        failed_checks: Sequence[str],
        reason: RejectionReason,
        score: Optional[float] = None,
        metrics: Optional[PerformanceMetrics] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        return ValidationResult(
            candidate_id=candidate_id,
            stage=stage,
            status=ValidationStatus.REJECTED,
            metrics=metrics,
            score=score,
            failed_checks=list(failed_checks),
            rejection_reason=reason,
            details=dict(details or {}),
            completed_at=utc_now(),
        )

    # ========================================================
    # Backtest validation
    # ========================================================

    def validate_backtest(
        self,
        candidate: StrategyCandidate,
        evidence: BacktestEvidence,
    ) -> ValidationResult:
        self._require_candidate(candidate)

        if not isinstance(evidence, BacktestEvidence):
            raise AlphaValidationError(
                "evidence must be BacktestEvidence."
            )

        metrics = evidence.metrics
        passed: List[str] = []
        failed: List[str] = []

        if evidence.data_leakage_detected:
            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.BACKTEST,
                failed_checks=["data_leakage"],
                reason=RejectionReason.DATA_LEAKAGE,
                metrics=metrics,
                details=evidence.metadata,
            )
            self._record(result)
            return result

        passed.append("no_data_leakage")

        if evidence.lookahead_bias_detected:
            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.BACKTEST,
                failed_checks=["lookahead_bias"],
                reason=RejectionReason.LOOKAHEAD_BIAS,
                metrics=metrics,
                details=evidence.metadata,
            )
            self._record(result)
            return result

        passed.append("no_lookahead_bias")

        if metrics.sharpe_ratio >= self.thresholds.min_sharpe_ratio:
            passed.append("sharpe_ratio")
        else:
            failed.append("sharpe_ratio")

        if metrics.profit_factor >= self.thresholds.min_profit_factor:
            passed.append("profit_factor")
        else:
            failed.append("profit_factor")

        if metrics.max_drawdown <= self.thresholds.max_drawdown:
            passed.append("max_drawdown")
        else:
            failed.append("max_drawdown")

        if metrics.trade_count >= self.thresholds.min_trade_count:
            passed.append("trade_count")
        else:
            failed.append("trade_count")

        if (
            self.thresholds.max_turnover is not None
            and metrics.turnover
            > self.thresholds.max_turnover
        ):
            failed.append("turnover")

        if (
            self.thresholds.max_execution_cost is not None
            and metrics.execution_cost
            > self.thresholds.max_execution_cost
        ):
            failed.append("execution_cost")

        if failed:
            if "max_drawdown" in failed:
                reason = RejectionReason.EXCESSIVE_DRAWDOWN
            elif "sharpe_ratio" in failed:
                reason = RejectionReason.LOW_SHARPE
            elif "turnover" in failed:
                reason = RejectionReason.HIGH_TURNOVER
            elif "execution_cost" in failed:
                reason = RejectionReason.HIGH_EXECUTION_COST
            else:
                reason = RejectionReason.NEGATIVE_EXPECTANCY

            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.BACKTEST,
                failed_checks=failed,
                reason=reason,
                score=metrics.sharpe_ratio,
                metrics=metrics,
                details=evidence.metadata,
            )
        else:
            result = ValidationResult(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.BACKTEST,
                status=ValidationStatus.PASSED,
                metrics=metrics,
                score=metrics.sharpe_ratio,
                passed_checks=passed,
                details=dict(evidence.metadata),
                completed_at=utc_now(),
            )

        self._record(result)
        return result

    # ========================================================
    # Causal validation
    # ========================================================

    def validate_causal(
        self,
        candidate: StrategyCandidate,
        evidence: CausalEvidence,
    ) -> ValidationResult:
        self._require_candidate(candidate)

        if not isinstance(evidence, CausalEvidence):
            raise AlphaValidationError(
                "evidence must be CausalEvidence."
            )

        passed: List[str] = []
        failed: List[str] = []

        if evidence.causal_score >= self.thresholds.min_causal_score:
            passed.append("causal_score")
        else:
            failed.append("causal_score")

        if evidence.stable_effect:
            passed.append("stable_effect")
        else:
            failed.append("stable_effect")

        if not evidence.confounding_detected:
            passed.append("no_confounding")
        else:
            failed.append("confounding")

        if failed:
            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.CAUSAL,
                failed_checks=failed,
                reason=RejectionReason.CAUSAL_FAILURE,
                score=evidence.causal_score,
                details=evidence.metadata,
            )
        else:
            result = ValidationResult(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.CAUSAL,
                status=ValidationStatus.PASSED,
                score=evidence.causal_score,
                passed_checks=passed,
                details=dict(evidence.metadata),
                completed_at=utc_now(),
            )

        self._record(result)
        return result

    # ========================================================
    # Out-of-sample validation
    # ========================================================

    def validate_oos(
        self,
        candidate: StrategyCandidate,
        evidence: OOSEvidence,
    ) -> ValidationResult:
        self._require_candidate(candidate)

        if not isinstance(evidence, OOSEvidence):
            raise AlphaValidationError(
                "evidence must be OOSEvidence."
            )

        passed: List[str] = []
        failed: List[str] = []

        metrics = evidence.metrics

        if (
            metrics.sharpe_ratio
            >= self.thresholds.min_oos_sharpe_ratio
        ):
            passed.append("oos_sharpe_ratio")
        else:
            failed.append("oos_sharpe_ratio")

        if (
            evidence.performance_retention
            >= self.thresholds.min_oos_retention
        ):
            passed.append("performance_retention")
        else:
            failed.append("performance_retention")

        if metrics.max_drawdown <= self.thresholds.max_drawdown:
            passed.append("oos_max_drawdown")
        else:
            failed.append("oos_max_drawdown")

        if failed:
            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.OUT_OF_SAMPLE,
                failed_checks=failed,
                reason=RejectionReason.OOS_FAILURE,
                score=evidence.performance_retention,
                metrics=metrics,
                details=evidence.metadata,
            )
        else:
            result = ValidationResult(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.OUT_OF_SAMPLE,
                status=ValidationStatus.PASSED,
                metrics=metrics,
                score=evidence.performance_retention,
                passed_checks=passed,
                details=dict(evidence.metadata),
                completed_at=utc_now(),
            )

        self._record(result)
        return result

    # ========================================================
    # Robustness validation
    # ========================================================

    def validate_robustness(
        self,
        candidate: StrategyCandidate,
        evidence: RobustnessEvidence,
    ) -> ValidationResult:
        self._require_candidate(candidate)

        if not isinstance(evidence, RobustnessEvidence):
            raise AlphaValidationError(
                "evidence must be RobustnessEvidence."
            )

        passed: List[str] = []
        failed: List[str] = []

        if (
            evidence.robustness_score
            >= self.thresholds.min_robustness_score
        ):
            passed.append("robustness_score")
        else:
            failed.append("robustness_score")

        if (
            evidence.parameter_sensitivity
            <= self.thresholds.max_parameter_sensitivity
        ):
            passed.append("parameter_sensitivity")
        else:
            failed.append("parameter_sensitivity")

        if evidence.regime_stable:
            passed.append("regime_stability")
        else:
            failed.append("regime_stability")

        if evidence.perturbation_stable:
            passed.append("perturbation_stability")
        else:
            failed.append("perturbation_stability")

        if failed:
            if "regime_stability" in failed:
                reason = RejectionReason.REGIME_DEPENDENT
            elif "parameter_sensitivity" in failed:
                reason = RejectionReason.OVERFIT
            else:
                reason = RejectionReason.ROBUSTNESS_FAILURE

            result = self._failed_result(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.ROBUSTNESS,
                failed_checks=failed,
                reason=reason,
                score=evidence.robustness_score,
                details=evidence.metadata,
            )
        else:
            result = ValidationResult(
                candidate_id=candidate.candidate_id,
                stage=ValidationStage.ROBUSTNESS,
                status=ValidationStatus.PASSED,
                score=evidence.robustness_score,
                passed_checks=passed,
                details=dict(evidence.metadata),
                completed_at=utc_now(),
            )

        self._record(result)
        return result

    # ========================================================
    # Complete pipeline
    # ========================================================

    def run(
        self,
        candidate: StrategyCandidate,
        *,
        backtest: BacktestEvidence,
        causal: CausalEvidence,
        oos: OOSEvidence,
        robustness: RobustnessEvidence,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ValidationPipelineResult:
        self._require_candidate(candidate)

        started_at = utc_now()
        results: List[ValidationResult] = []

        stages: List[
            tuple[
                CandidateStatus,
                ValidationStage,
                Callable[[], ValidationResult],
            ]
        ] = [
            (
                CandidateStatus.BACKTESTING,
                ValidationStage.BACKTEST,
                lambda: self.validate_backtest(
                    candidate,
                    backtest,
                ),
            ),
            (
                CandidateStatus.CAUSAL_VALIDATION,
                ValidationStage.CAUSAL,
                lambda: self.validate_causal(
                    candidate,
                    causal,
                ),
            ),
            (
                CandidateStatus.OOS_VALIDATION,
                ValidationStage.OUT_OF_SAMPLE,
                lambda: self.validate_oos(
                    candidate,
                    oos,
                ),
            ),
            (
                CandidateStatus.ROBUSTNESS_VALIDATION,
                ValidationStage.ROBUSTNESS,
                lambda: self.validate_robustness(
                    candidate,
                    robustness,
                ),
            ),
        ]

        for candidate_status, stage, validator in stages:
            candidate.set_status(candidate_status)

            try:
                result = validator()
            except Exception as exc:
                error_result = ValidationResult(
                    candidate_id=candidate.candidate_id,
                    stage=stage,
                    status=ValidationStatus.ERROR,
                    failed_checks=["stage_exception"],
                    rejection_reason=RejectionReason.OTHER,
                    details={
                        "exception_type": type(exc).__name__,
                        "exception": str(exc),
                    },
                    completed_at=utc_now(),
                )

                self._record(error_result)
                results.append(error_result)

                candidate.reject(
                    RejectionReason.OTHER,
                    str(exc),
                )

                pipeline_result = ValidationPipelineResult(
                    candidate_id=candidate.candidate_id,
                    accepted=False,
                    results=results,
                    rejection_reason=RejectionReason.OTHER,
                    stopped_at=stage,
                    started_at=started_at,
                    completed_at=utc_now(),
                    metadata=dict(metadata or {}),
                )

                self._store_result(pipeline_result)
                return pipeline_result

            results.append(result)

            if not result.successful:
                reason = (
                    result.rejection_reason
                    or RejectionReason.OTHER
                )

                candidate.reject(
                    reason,
                    (
                        ", ".join(result.failed_checks)
                        if result.failed_checks
                        else None
                    ),
                )

                pipeline_result = ValidationPipelineResult(
                    candidate_id=candidate.candidate_id,
                    accepted=False,
                    results=results,
                    rejection_reason=reason,
                    stopped_at=stage,
                    started_at=started_at,
                    completed_at=utc_now(),
                    metadata=dict(metadata or {}),
                )

                self._store_result(pipeline_result)
                return pipeline_result

        candidate.accept()

        pipeline_result = ValidationPipelineResult(
            candidate_id=candidate.candidate_id,
            accepted=True,
            results=results,
            rejection_reason=None,
            stopped_at=None,
            started_at=started_at,
            completed_at=utc_now(),
            metadata=dict(metadata or {}),
        )

        self._store_result(pipeline_result)
        return pipeline_result

    # ========================================================
    # History
    # ========================================================

    def _store_result(
        self,
        result: ValidationPipelineResult,
    ) -> None:
        self._history.setdefault(
            result.candidate_id,
            [],
        ).append(result)

    def history(
        self,
        candidate_id: Optional[str] = None,
    ) -> List[ValidationPipelineResult]:
        if candidate_id is None:
            return [
                result
                for results in self._history.values()
                for result in results
            ]

        return list(
            self._history.get(
                candidate_id,
                [],
            )
        )

    def latest(
        self,
        candidate_id: str,
    ) -> Optional[ValidationPipelineResult]:
        history = self._history.get(
            candidate_id,
            [],
        )

        if not history:
            return None

        return history[-1]

    def audit_log(
        self,
    ) -> List[ValidationPipelineAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Reproducibility
    # ========================================================

    def configuration_fingerprint(self) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "thresholds": {
                    "min_sharpe_ratio": (
                        self.thresholds.min_sharpe_ratio
                    ),
                    "min_profit_factor": (
                        self.thresholds.min_profit_factor
                    ),
                    "max_drawdown": (
                        self.thresholds.max_drawdown
                    ),
                    "min_trade_count": (
                        self.thresholds.min_trade_count
                    ),
                    "min_causal_score": (
                        self.thresholds.min_causal_score
                    ),
                    "min_oos_sharpe_ratio": (
                        self.thresholds.min_oos_sharpe_ratio
                    ),
                    "min_oos_retention": (
                        self.thresholds.min_oos_retention
                    ),
                    "min_robustness_score": (
                        self.thresholds.min_robustness_score
                    ),
                    "max_parameter_sensitivity": (
                        self.thresholds
                        .max_parameter_sensitivity
                    ),
                    "max_turnover": (
                        self.thresholds.max_turnover
                    ),
                    "max_execution_cost": (
                        self.thresholds.max_execution_cost
                    ),
                },
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        all_results = self.history()

        accepted = sum(
            result.accepted
            for result in all_results
        )

        rejected = len(all_results) - accepted

        return {
            "version": self.VERSION,
            "pipeline_runs": len(all_results),
            "accepted": accepted,
            "rejected": rejected,
            "audit_records": len(self._audit),
            "configuration_fingerprint": (
                self.configuration_fingerprint()
            ),
            "stage_order": [
                stage.value
                for stage in self.STAGE_ORDER
            ],
            "research_only": True,
            "real_money_execution": False,
        }


# ============================================================
# Public exports
# ============================================================


__all__ = [
    "ValidationPipelineError",
    "ValidationStageError",
    "ValidationConfigurationError",
    "ValidationThresholds",
    "BacktestEvidence",
    "CausalEvidence",
    "OOSEvidence",
    "RobustnessEvidence",
    "ValidationPipelineAuditRecord",
    "ValidationPipelineResult",
    "AlphaValidationPipeline",
]