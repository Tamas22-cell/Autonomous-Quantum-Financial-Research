from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    VERSION,
    PerformanceMetrics,
    RejectionReason,
    StrategyCandidate,
    stable_hash,
    utc_now,
)


# ============================================================
# Exceptions
# ============================================================


class OverfitDetectionError(Exception):
    """Base exception for overfit and stability detection."""


class OverfitConfigurationError(OverfitDetectionError):
    """Raised when detector configuration is invalid."""


# ============================================================
# Configuration
# ============================================================


@dataclass(frozen=True)
class OverfitThresholds:
    max_sharpe_degradation: float = 0.60
    max_return_degradation: float = 0.70
    max_drawdown_increase: float = 0.50

    max_parameter_sensitivity: float = 0.50
    max_regime_dispersion: float = 0.60

    min_oos_retention: float = 0.40
    min_stability_score: float = 0.50

    min_regime_count: int = 2
    min_perturbation_count: int = 3

    def __post_init__(self) -> None:
        bounded = {
            "max_sharpe_degradation": (
                self.max_sharpe_degradation
            ),
            "max_return_degradation": (
                self.max_return_degradation
            ),
            "max_drawdown_increase": (
                self.max_drawdown_increase
            ),
            "max_parameter_sensitivity": (
                self.max_parameter_sensitivity
            ),
            "max_regime_dispersion": (
                self.max_regime_dispersion
            ),
            "min_oos_retention": (
                self.min_oos_retention
            ),
            "min_stability_score": (
                self.min_stability_score
            ),
        }

        for name, value in bounded.items():
            if value < 0.0 or value > 1.0:
                raise OverfitConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        if self.min_regime_count < 1:
            raise OverfitConfigurationError(
                "min_regime_count must be at least 1."
            )

        if self.min_perturbation_count < 1:
            raise OverfitConfigurationError(
                "min_perturbation_count must be at least 1."
            )


# ============================================================
# Evidence
# ============================================================


@dataclass(frozen=True)
class ParameterPerturbation:
    name: str
    baseline_value: float
    perturbed_value: float
    performance_score: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class RegimePerformance:
    regime: str
    metrics: PerformanceMetrics

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class OverfitEvidence:
    in_sample: PerformanceMetrics
    out_of_sample: PerformanceMetrics

    parameter_perturbations: List[
        ParameterPerturbation
    ] = field(default_factory=list)

    regime_performance: List[
        RegimePerformance
    ] = field(default_factory=list)

    bootstrap_scores: List[float] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Detection Result
# ============================================================


@dataclass(frozen=True)
class OverfitDetectionResult:
    candidate_id: str

    overfit_detected: bool
    unstable_detected: bool

    stability_score: float

    sharpe_degradation: float
    return_degradation: float
    drawdown_increase: float

    parameter_sensitivity: float
    regime_dispersion: float
    oos_retention: float

    failed_checks: List[str] = field(
        default_factory=list
    )

    passed_checks: List[str] = field(
        default_factory=list
    )

    rejection_reason: Optional[
        RejectionReason
    ] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    @property
    def accepted(self) -> bool:
        return not (
            self.overfit_detected
            or self.unstable_detected
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "candidate_id": self.candidate_id,
                "overfit_detected": (
                    self.overfit_detected
                ),
                "unstable_detected": (
                    self.unstable_detected
                ),
                "stability_score": (
                    self.stability_score
                ),
                "sharpe_degradation": (
                    self.sharpe_degradation
                ),
                "return_degradation": (
                    self.return_degradation
                ),
                "drawdown_increase": (
                    self.drawdown_increase
                ),
                "parameter_sensitivity": (
                    self.parameter_sensitivity
                ),
                "regime_dispersion": (
                    self.regime_dispersion
                ),
                "oos_retention": (
                    self.oos_retention
                ),
                "failed_checks": (
                    self.failed_checks
                ),
                "passed_checks": (
                    self.passed_checks
                ),
                "rejection_reason": (
                    self.rejection_reason.value
                    if self.rejection_reason
                    else None
                ),
            }
        )


@dataclass(frozen=True)
class OverfitAuditRecord:
    candidate_id: str
    result_fingerprint: str
    accepted: bool

    timestamp: datetime = field(
        default_factory=utc_now
    )

    details: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Detector
# ============================================================


class OverfitDetector:
    """
    Detects unstable or potentially overfit alpha candidates.

    The detector compares:
    - in-sample vs out-of-sample performance
    - parameter perturbation sensitivity
    - regime stability
    - bootstrap stability

    It does not perform live trading or real-money execution.
    """

    VERSION = VERSION

    def __init__(
        self,
        thresholds: Optional[
            OverfitThresholds
        ] = None,
    ) -> None:
        self.thresholds = (
            thresholds or OverfitThresholds()
        )

        self._results: Dict[
            str,
            List[OverfitDetectionResult],
        ] = {}

        self._audit: List[
            OverfitAuditRecord
        ] = []

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_degradation(
        baseline: float,
        observed: float,
    ) -> float:
        if baseline <= 0.0:
            if observed >= baseline:
                return 0.0
            return 1.0

        degradation = (
            baseline - observed
        ) / abs(baseline)

        return max(
            0.0,
            min(1.0, degradation),
        )

    @staticmethod
    def _drawdown_increase(
        baseline: float,
        observed: float,
    ) -> float:
        baseline_abs = abs(baseline)
        observed_abs = abs(observed)

        if baseline_abs == 0.0:
            return (
                0.0
                if observed_abs == 0.0
                else 1.0
            )

        increase = (
            observed_abs - baseline_abs
        ) / baseline_abs

        return max(
            0.0,
            min(1.0, increase),
        )

    @staticmethod
    def _mean(
        values: Sequence[float],
    ) -> float:
        if not values:
            return 0.0

        return sum(values) / len(values)

    @classmethod
    def _dispersion(
        cls,
        values: Sequence[float],
    ) -> float:
        if len(values) < 2:
            return 0.0

        mean = cls._mean(values)

        if mean == 0.0:
            return (
                0.0
                if all(
                    value == 0.0
                    for value in values
                )
                else 1.0
            )

        variance = sum(
            (value - mean) ** 2
            for value in values
        ) / len(values)

        standard_deviation = (
            variance ** 0.5
        )

        coefficient = (
            standard_deviation
            / abs(mean)
        )

        return max(
            0.0,
            min(1.0, coefficient),
        )

    @staticmethod
    def _oos_retention(
        in_sample: float,
        out_of_sample: float,
    ) -> float:
        if in_sample <= 0.0:
            return (
                1.0
                if out_of_sample
                >= in_sample
                else 0.0
            )

        return max(
            0.0,
            min(
                1.0,
                out_of_sample
                / in_sample,
            ),
        )

    # ========================================================
    # Sensitivity
    # ========================================================

    def parameter_sensitivity(
        self,
        perturbations: Sequence[
            ParameterPerturbation
        ],
    ) -> float:
        if not perturbations:
            return 0.0

        scores = [
            item.performance_score
            for item in perturbations
        ]

        return self._dispersion(scores)

    def regime_dispersion(
        self,
        regimes: Sequence[
            RegimePerformance
        ],
    ) -> float:
        if not regimes:
            return 0.0

        scores = [
            item.metrics.sharpe_ratio
            for item in regimes
        ]

        return self._dispersion(scores)

    def bootstrap_stability(
        self,
        bootstrap_scores: Sequence[float],
    ) -> float:
        if not bootstrap_scores:
            return 1.0

        dispersion = self._dispersion(
            bootstrap_scores
        )

        return max(
            0.0,
            min(
                1.0,
                1.0 - dispersion,
            ),
        )

    # ========================================================
    # Detection
    # ========================================================

    def detect(
        self,
        candidate: StrategyCandidate,
        evidence: OverfitEvidence,
    ) -> OverfitDetectionResult:
        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise OverfitDetectionError(
                "candidate must be a "
                "StrategyCandidate instance."
            )

        if not isinstance(
            evidence,
            OverfitEvidence,
        ):
            raise OverfitDetectionError(
                "evidence must be an "
                "OverfitEvidence instance."
            )

        in_sample = evidence.in_sample
        out_of_sample = (
            evidence.out_of_sample
        )

        sharpe_degradation = (
            self._safe_degradation(
                in_sample.sharpe_ratio,
                out_of_sample.sharpe_ratio,
            )
        )

        return_degradation = (
            self._safe_degradation(
                in_sample.annualized_return,
                out_of_sample.annualized_return,
            )
        )

        drawdown_increase = (
            self._drawdown_increase(
                in_sample.max_drawdown,
                out_of_sample.max_drawdown,
            )
        )

        parameter_sensitivity = (
            self.parameter_sensitivity(
                evidence.parameter_perturbations
            )
        )

        regime_dispersion = (
            self.regime_dispersion(
                evidence.regime_performance
            )
        )

        oos_retention = (
            self._oos_retention(
                in_sample.sharpe_ratio,
                out_of_sample.sharpe_ratio,
            )
        )

        bootstrap_stability = (
            self.bootstrap_stability(
                evidence.bootstrap_scores
            )
        )

        stability_components = [
            1.0 - sharpe_degradation,
            1.0 - return_degradation,
            1.0 - drawdown_increase,
            1.0 - parameter_sensitivity,
            1.0 - regime_dispersion,
            oos_retention,
            bootstrap_stability,
        ]

        stability_score = max(
            0.0,
            min(
                1.0,
                self._mean(
                    stability_components
                ),
            ),
        )

        passed: List[str] = []
        failed: List[str] = []

        checks = {
            "sharpe_degradation": (
                sharpe_degradation
                <= self.thresholds
                .max_sharpe_degradation
            ),
            "return_degradation": (
                return_degradation
                <= self.thresholds
                .max_return_degradation
            ),
            "drawdown_increase": (
                drawdown_increase
                <= self.thresholds
                .max_drawdown_increase
            ),
            "parameter_sensitivity": (
                parameter_sensitivity
                <= self.thresholds
                .max_parameter_sensitivity
            ),
            "regime_dispersion": (
                regime_dispersion
                <= self.thresholds
                .max_regime_dispersion
            ),
            "oos_retention": (
                oos_retention
                >= self.thresholds
                .min_oos_retention
            ),
            "stability_score": (
                stability_score
                >= self.thresholds
                .min_stability_score
            ),
        }

        for check_name, check_passed in (
            checks.items()
        ):
            if check_passed:
                passed.append(check_name)
            else:
                failed.append(check_name)

        insufficient_perturbations = (
            len(
                evidence.parameter_perturbations
            )
            < self.thresholds
            .min_perturbation_count
        )

        insufficient_regimes = (
            len(evidence.regime_performance)
            < self.thresholds.min_regime_count
        )

        if insufficient_perturbations:
            failed.append(
                "insufficient_parameter_perturbations"
            )

        if insufficient_regimes:
            failed.append(
                "insufficient_regime_coverage"
            )

        overfit_detected = any(
            check in failed
            for check in (
                "sharpe_degradation",
                "return_degradation",
                "parameter_sensitivity",
                "oos_retention",
            )
        )

        unstable_detected = any(
            check in failed
            for check in (
                "drawdown_increase",
                "regime_dispersion",
                "stability_score",
                "insufficient_parameter_perturbations",
                "insufficient_regime_coverage",
            )
        )

        rejection_reason: Optional[
            RejectionReason
        ] = None

        if overfit_detected:
            rejection_reason = (
                RejectionReason.OVERFIT
            )
        elif unstable_detected:
            rejection_reason = (
                RejectionReason.LOW_STABILITY
            )

        result = OverfitDetectionResult(
            candidate_id=(
                candidate.candidate_id
            ),
            overfit_detected=(
                overfit_detected
            ),
            unstable_detected=(
                unstable_detected
            ),
            stability_score=(
                stability_score
            ),
            sharpe_degradation=(
                sharpe_degradation
            ),
            return_degradation=(
                return_degradation
            ),
            drawdown_increase=(
                drawdown_increase
            ),
            parameter_sensitivity=(
                parameter_sensitivity
            ),
            regime_dispersion=(
                regime_dispersion
            ),
            oos_retention=(
                oos_retention
            ),
            failed_checks=failed,
            passed_checks=passed,
            rejection_reason=(
                rejection_reason
            ),
            details={
                "bootstrap_stability": (
                    bootstrap_stability
                ),
                "perturbation_count": len(
                    evidence
                    .parameter_perturbations
                ),
                "regime_count": len(
                    evidence.regime_performance
                ),
                **dict(evidence.metadata),
            },
        )

        self._results.setdefault(
            candidate.candidate_id,
            [],
        ).append(result)

        self._audit.append(
            OverfitAuditRecord(
                candidate_id=(
                    candidate.candidate_id
                ),
                result_fingerprint=(
                    result.fingerprint
                ),
                accepted=result.accepted,
                details={
                    "overfit_detected": (
                        result.overfit_detected
                    ),
                    "unstable_detected": (
                        result.unstable_detected
                    ),
                    "rejection_reason": (
                        rejection_reason.value
                        if rejection_reason
                        else None
                    ),
                },
            )
        )

        return result

    # ========================================================
    # Automatic rejection
    # ========================================================

    def detect_and_apply(
        self,
        candidate: StrategyCandidate,
        evidence: OverfitEvidence,
    ) -> OverfitDetectionResult:
        result = self.detect(
            candidate,
            evidence,
        )

        if not result.accepted:
            candidate.reject(
                result.rejection_reason
                or RejectionReason.OTHER,
                ", ".join(
                    result.failed_checks
                ),
            )

        return result

    # ========================================================
    # History
    # ========================================================

    def history(
        self,
        candidate_id: Optional[
            str
        ] = None,
    ) -> List[OverfitDetectionResult]:
        if candidate_id is None:
            return [
                result
                for results
                in self._results.values()
                for result in results
            ]

        return list(
            self._results.get(
                candidate_id,
                [],
            )
        )

    def latest(
        self,
        candidate_id: str,
    ) -> Optional[
        OverfitDetectionResult
    ]:
        results = self._results.get(
            candidate_id,
            [],
        )

        if not results:
            return None

        return results[-1]

    def audit_log(
        self,
    ) -> List[OverfitAuditRecord]:
        return list(self._audit)

    # ========================================================
    # Reproducibility
    # ========================================================

    def configuration_fingerprint(
        self,
    ) -> str:
        return stable_hash(
            {
                "version": self.VERSION,
                "thresholds": {
                    "max_sharpe_degradation": (
                        self.thresholds
                        .max_sharpe_degradation
                    ),
                    "max_return_degradation": (
                        self.thresholds
                        .max_return_degradation
                    ),
                    "max_drawdown_increase": (
                        self.thresholds
                        .max_drawdown_increase
                    ),
                    "max_parameter_sensitivity": (
                        self.thresholds
                        .max_parameter_sensitivity
                    ),
                    "max_regime_dispersion": (
                        self.thresholds
                        .max_regime_dispersion
                    ),
                    "min_oos_retention": (
                        self.thresholds
                        .min_oos_retention
                    ),
                    "min_stability_score": (
                        self.thresholds
                        .min_stability_score
                    ),
                    "min_regime_count": (
                        self.thresholds
                        .min_regime_count
                    ),
                    "min_perturbation_count": (
                        self.thresholds
                        .min_perturbation_count
                    ),
                },
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    def summary(self) -> Dict[str, Any]:
        results = self.history()

        accepted = sum(
            result.accepted
            for result in results
        )

        rejected = (
            len(results) - accepted
        )

        overfit = sum(
            result.overfit_detected
            for result in results
        )

        unstable = sum(
            result.unstable_detected
            for result in results
        )

        return {
            "version": self.VERSION,
            "evaluations": len(results),
            "accepted": accepted,
            "rejected": rejected,
            "overfit_detected": overfit,
            "unstable_detected": unstable,
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
    "OverfitDetectionError",
    "OverfitConfigurationError",
    "OverfitThresholds",
    "ParameterPerturbation",
    "RegimePerformance",
    "OverfitEvidence",
    "OverfitDetectionResult",
    "OverfitAuditRecord",
    "OverfitDetector",
]