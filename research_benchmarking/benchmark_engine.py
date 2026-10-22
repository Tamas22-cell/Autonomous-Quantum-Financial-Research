from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from research_benchmarking.models import (
    AlphaVerdict,
    BenchmarkReport,
    OverfittingDiagnostics,
    OverfittingRisk,
    RegimeValidationResult,
    SensitivityResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    stable_hash,
)


VERSION = "v6.0.2"


# ============================================================
# Exceptions
# ============================================================


class BenchmarkEngineError(Exception):
    """Raised when the benchmark aggregation cannot be completed."""


# ============================================================
# Stage policy
# ============================================================


@dataclass(frozen=True)
class BenchmarkStagePolicy:
    required_stages: Sequence[ValidationStage] = field(
        default_factory=lambda: (
            ValidationStage.WALK_FORWARD,
            ValidationStage.OUT_OF_SAMPLE,
            ValidationStage.BOOTSTRAP,
            ValidationStage.MONTE_CARLO,
            ValidationStage.STATISTICAL_SIGNIFICANCE,
            ValidationStage.MULTIPLE_TESTING,
            ValidationStage.ROBUSTNESS,
            ValidationStage.SENSITIVITY,
            ValidationStage.REGIME_STABILITY,
            ValidationStage.OVERFITTING_CONTROL,
        )
    )
    minimum_stage_score: float = 0.50
    minimum_pass_ratio: float = 0.80
    require_oos_pass: bool = True
    require_overfitting_pass: bool = True
    require_significance_pass: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stages = tuple(self.required_stages)

        if not stages:
            raise ValidationConfigurationError(
                "required_stages cannot be empty."
            )

        if not all(
            isinstance(stage, ValidationStage)
            for stage in stages
        ):
            raise ValidationConfigurationError(
                "required_stages must contain ValidationStage values."
            )

        if len(set(stages)) != len(stages):
            raise ValidationConfigurationError(
                "required_stages cannot contain duplicates."
            )

        object.__setattr__(
            self,
            "required_stages",
            stages,
        )

        self._validate_ratio(
            self.minimum_stage_score,
            "minimum_stage_score",
        )

        self._validate_ratio(
            self.minimum_pass_ratio,
            "minimum_pass_ratio",
        )

        for name, value in {
            "require_oos_pass": self.require_oos_pass,
            "require_overfitting_pass": (
                self.require_overfitting_pass
            ),
            "require_significance_pass": (
                self.require_significance_pass
            ),
        }.items():
            if not isinstance(value, bool):
                raise ValidationConfigurationError(
                    f"{name} must be bool."
                )

    @staticmethod
    def _validate_ratio(
        value: float,
        name: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValidationConfigurationError(
                f"{name} must be numeric."
            )

        numeric = float(value)

        if not 0.0 <= numeric <= 1.0:
            raise ValidationConfigurationError(
                f"{name} must be between 0 and 1."
            )

        return numeric

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "required_stages": [
                    stage.value
                    for stage in self.required_stages
                ],
                "minimum_stage_score": (
                    self.minimum_stage_score
                ),
                "minimum_pass_ratio": (
                    self.minimum_pass_ratio
                ),
                "require_oos_pass": (
                    self.require_oos_pass
                ),
                "require_overfitting_pass": (
                    self.require_overfitting_pass
                ),
                "require_significance_pass": (
                    self.require_significance_pass
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Stage evidence
# ============================================================


@dataclass(frozen=True)
class StageEvidence:
    stage: ValidationStage
    status: ValidationStatus
    score: float
    passed: bool
    significant: bool
    message: str
    fingerprint: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.stage,
            ValidationStage,
        ):
            raise ValidationConfigurationError(
                "stage must be ValidationStage."
            )

        if not isinstance(
            self.status,
            ValidationStatus,
        ):
            raise ValidationConfigurationError(
                "status must be ValidationStatus."
            )

        if (
            isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not 0.0 <= float(self.score) <= 1.0
        ):
            raise ValidationConfigurationError(
                "score must be between 0 and 1."
            )

        if not isinstance(self.passed, bool):
            raise ValidationConfigurationError(
                "passed must be bool."
            )

        if not isinstance(self.significant, bool):
            raise ValidationConfigurationError(
                "significant must be bool."
            )

        if not isinstance(self.message, str):
            raise ValidationConfigurationError(
                "message must be str."
            )

        if not isinstance(self.fingerprint, str):
            raise ValidationConfigurationError(
                "fingerprint must be str."
            )


# ============================================================
# Benchmark evidence summary
# ============================================================


@dataclass(frozen=True)
class BenchmarkEvidenceSummary:
    evidence: Sequence[StageEvidence]
    required_stage_count: int
    observed_required_stage_count: int
    passed_required_stage_count: int
    missing_stages: Sequence[ValidationStage]
    failed_stages: Sequence[ValidationStage]
    inconclusive_stages: Sequence[ValidationStage]
    average_score: float
    minimum_score: float
    pass_ratio: float
    all_required_stages_present: bool
    all_required_stages_passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        missing = tuple(self.missing_stages)
        failed = tuple(self.failed_stages)
        inconclusive = tuple(
            self.inconclusive_stages
        )

        if not evidence:
            raise ValidationConfigurationError(
                "evidence cannot be empty."
            )

        if not all(
            isinstance(item, StageEvidence)
            for item in evidence
        ):
            raise ValidationConfigurationError(
                "evidence must contain StageEvidence values."
            )

        for name, value in {
            "required_stage_count": self.required_stage_count,
            "observed_required_stage_count": (
                self.observed_required_stage_count
            ),
            "passed_required_stage_count": (
                self.passed_required_stage_count
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be a non-negative integer."
                )

        if self.required_stage_count <= 0:
            raise ValidationConfigurationError(
                "required_stage_count must be positive."
            )

        if (
            self.observed_required_stage_count
            > self.required_stage_count
        ):
            raise ValidationConfigurationError(
                "observed_required_stage_count cannot exceed "
                "required_stage_count."
            )

        if (
            self.passed_required_stage_count
            > self.observed_required_stage_count
        ):
            raise ValidationConfigurationError(
                "passed_required_stage_count cannot exceed "
                "observed_required_stage_count."
            )

        for collection_name, collection in {
            "missing_stages": missing,
            "failed_stages": failed,
            "inconclusive_stages": inconclusive,
        }.items():
            if not all(
                isinstance(stage, ValidationStage)
                for stage in collection
            ):
                raise ValidationConfigurationError(
                    f"{collection_name} must contain "
                    "ValidationStage values."
                )

        for name, value in {
            "average_score": self.average_score,
            "minimum_score": self.minimum_score,
            "pass_ratio": self.pass_ratio,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        if not isinstance(
            self.all_required_stages_present,
            bool,
        ):
            raise ValidationConfigurationError(
                "all_required_stages_present must be bool."
            )

        if not isinstance(
            self.all_required_stages_passed,
            bool,
        ):
            raise ValidationConfigurationError(
                "all_required_stages_passed must be bool."
            )

        object.__setattr__(
            self,
            "evidence",
            evidence,
        )
        object.__setattr__(
            self,
            "missing_stages",
            missing,
        )
        object.__setattr__(
            self,
            "failed_stages",
            failed,
        )
        object.__setattr__(
            self,
            "inconclusive_stages",
            inconclusive,
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "evidence": [
                    {
                        "stage": item.stage.value,
                        "status": item.status.value,
                        "score": item.score,
                        "passed": item.passed,
                        "significant": item.significant,
                        "message": item.message,
                        "fingerprint": item.fingerprint,
                    }
                    for item in self.evidence
                ],
                "required_stage_count": (
                    self.required_stage_count
                ),
                "observed_required_stage_count": (
                    self.observed_required_stage_count
                ),
                "passed_required_stage_count": (
                    self.passed_required_stage_count
                ),
                "missing_stages": [
                    stage.value
                    for stage in self.missing_stages
                ],
                "failed_stages": [
                    stage.value
                    for stage in self.failed_stages
                ],
                "inconclusive_stages": [
                    stage.value
                    for stage in self.inconclusive_stages
                ],
                "average_score": self.average_score,
                "minimum_score": self.minimum_score,
                "pass_ratio": self.pass_ratio,
                "all_required_stages_present": (
                    self.all_required_stages_present
                ),
                "all_required_stages_passed": (
                    self.all_required_stages_passed
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Final benchmark decision
# ============================================================


@dataclass(frozen=True)
class BenchmarkDecision:
    verdict: AlphaVerdict
    overall_score: float
    confidence_score: float
    evidence_quality_score: float
    overfitting_risk: OverfittingRisk
    robust_alpha_supported: bool
    reasons: Sequence[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.verdict,
            AlphaVerdict,
        ):
            raise ValidationConfigurationError(
                "verdict must be AlphaVerdict."
            )

        for name, value in {
            "overall_score": self.overall_score,
            "confidence_score": self.confidence_score,
            "evidence_quality_score": (
                self.evidence_quality_score
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        if not isinstance(
            self.overfitting_risk,
            OverfittingRisk,
        ):
            raise ValidationConfigurationError(
                "overfitting_risk must be OverfittingRisk."
            )

        if not isinstance(
            self.robust_alpha_supported,
            bool,
        ):
            raise ValidationConfigurationError(
                "robust_alpha_supported must be bool."
            )

        reasons = tuple(self.reasons)

        if not all(
            isinstance(reason, str)
            for reason in reasons
        ):
            raise ValidationConfigurationError(
                "reasons must contain strings."
            )

        object.__setattr__(
            self,
            "reasons",
            reasons,
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "verdict": self.verdict.value,
                "overall_score": self.overall_score,
                "confidence_score": (
                    self.confidence_score
                ),
                "evidence_quality_score": (
                    self.evidence_quality_score
                ),
                "overfitting_risk": (
                    self.overfitting_risk.value
                ),
                "robust_alpha_supported": (
                    self.robust_alpha_supported
                ),
                "reasons": list(self.reasons),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Benchmark Engine
# ============================================================


class BenchmarkEngine:
    """
    Central v6.0.2 research benchmark aggregator.

    The engine combines the independent validation layers into
    one evidence-driven benchmark decision.

    It does not treat one good backtest, one p-value, or one
    performance metric as proof of alpha.

    Robust-alpha support requires broad agreement across:
    - walk-forward validation
    - out-of-sample validation
    - bootstrap evidence
    - Monte Carlo evidence
    - statistical significance
    - multiple-testing controls
    - robustness testing
    - parameter sensitivity
    - regime stability
    - anti-overfitting controls
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        policy: Optional[BenchmarkStagePolicy] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.policy = policy or BenchmarkStagePolicy()

    # ========================================================
    # Validation
    # ========================================================

    @staticmethod
    def _validate_results(
        results: Iterable[ValidationResult],
    ) -> List[ValidationResult]:
        if isinstance(
            results,
            (str, bytes, Mapping),
        ):
            raise ValidationConfigurationError(
                "results must be an iterable of ValidationResult."
            )

        try:
            validated = list(results)
        except TypeError as exc:
            raise ValidationConfigurationError(
                "results must be iterable."
            ) from exc

        if not validated:
            raise BenchmarkEngineError(
                "validation results cannot be empty."
            )

        if not all(
            isinstance(item, ValidationResult)
            for item in validated
        ):
            raise ValidationConfigurationError(
                "results must contain ValidationResult values."
            )

        stages = [
            item.stage
            for item in validated
        ]

        if len(set(stages)) != len(stages):
            raise BenchmarkEngineError(
                "duplicate validation stages are not allowed."
            )

        return validated

    # ========================================================
    # Evidence conversion
    # ========================================================

    def build_stage_evidence(
        self,
        result: ValidationResult,
    ) -> StageEvidence:
        if not isinstance(
            result,
            ValidationResult,
        ):
            raise ValidationConfigurationError(
                "result must be ValidationResult."
            )

        passed = (
            result.status == ValidationStatus.PASSED
            and result.score
            >= self.policy.minimum_stage_score
        )

        significant = (
            result.significant_test_count > 0
        )

        return StageEvidence(
            stage=result.stage,
            status=result.status,
            score=float(result.score),
            passed=passed,
            significant=significant,
            message=result.message,
            fingerprint=result.fingerprint,
            metadata={
                "version": VERSION,
                "statistical_test_count": len(
                    result.statistical_tests
                ),
                "significant_test_count": (
                    result.significant_test_count
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Evidence aggregation
    # ========================================================

    def summarize(
        self,
        results: Iterable[ValidationResult],
    ) -> BenchmarkEvidenceSummary:
        validated = self._validate_results(
            results
        )

        evidence = [
            self.build_stage_evidence(result)
            for result in validated
        ]

        by_stage = {
            item.stage: item
            for item in evidence
        }

        required = tuple(
            self.policy.required_stages
        )

        observed_required = [
            stage
            for stage in required
            if stage in by_stage
        ]

        missing = [
            stage
            for stage in required
            if stage not in by_stage
        ]

        passed_required = [
            stage
            for stage in observed_required
            if by_stage[stage].passed
        ]

        failed = [
            stage
            for stage in observed_required
            if by_stage[stage].status
            == ValidationStatus.FAILED
            or not by_stage[stage].passed
        ]

        inconclusive = [
            stage
            for stage in observed_required
            if by_stage[stage].status
            == ValidationStatus.INCONCLUSIVE
        ]

        required_scores = [
            by_stage[stage].score
            for stage in observed_required
        ]

        if required_scores:
            average_score = (
                sum(required_scores)
                / len(required_scores)
            )
            minimum_score = min(
                required_scores
            )
        else:
            average_score = 0.0
            minimum_score = 0.0

        pass_ratio = (
            len(passed_required)
            / len(required)
        )

        all_present = (
            len(missing) == 0
        )

        all_passed = (
            all_present
            and len(passed_required)
            == len(required)
        )

        return BenchmarkEvidenceSummary(
            evidence=evidence,
            required_stage_count=len(required),
            observed_required_stage_count=(
                len(observed_required)
            ),
            passed_required_stage_count=(
                len(passed_required)
            ),
            missing_stages=missing,
            failed_stages=failed,
            inconclusive_stages=inconclusive,
            average_score=max(
                0.0,
                min(1.0, average_score),
            ),
            minimum_score=max(
                0.0,
                min(1.0, minimum_score),
            ),
            pass_ratio=max(
                0.0,
                min(1.0, pass_ratio),
            ),
            all_required_stages_present=(
                all_present
            ),
            all_required_stages_passed=(
                all_passed
            ),
            metadata={
                "version": VERSION,
                "policy_fingerprint": (
                    self.policy.fingerprint
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Overfitting evidence
    # ========================================================

    @staticmethod
    def _extract_overfitting_risk(
        results: Sequence[ValidationResult],
    ) -> tuple[OverfittingRisk, float]:
        for result in results:
            if (
                result.stage
                == ValidationStage.OVERFITTING_CONTROL
            ):
                raw_level = result.diagnostics.get(
                    "risk_level",
                    OverfittingRisk.MODERATE.value,
                )

                try:
                    level = OverfittingRisk(
                        raw_level
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    level = OverfittingRisk.MODERATE

                raw_score = result.diagnostics.get(
                    "overall_risk_score",
                    0.50,
                )

                if (
                    isinstance(raw_score, bool)
                    or not isinstance(
                        raw_score,
                        (int, float),
                    )
                ):
                    risk_score = 0.50
                else:
                    risk_score = max(
                        0.0,
                        min(
                            1.0,
                            float(raw_score),
                        ),
                    )

                return level, risk_score

        return OverfittingRisk.MODERATE, 0.50

    # ========================================================
    # Critical stage checks
    # ========================================================

    def _critical_stage_failures(
        self,
        summary: BenchmarkEvidenceSummary,
    ) -> List[str]:
        evidence = {
            item.stage: item
            for item in summary.evidence
        }

        reasons: List[str] = []

        if self.policy.require_oos_pass:
            oos = evidence.get(
                ValidationStage.OUT_OF_SAMPLE
            )

            if oos is None:
                reasons.append(
                    "Out-of-sample validation is missing."
                )
            elif not oos.passed:
                reasons.append(
                    "Out-of-sample validation did not pass."
                )

        if self.policy.require_overfitting_pass:
            overfitting = evidence.get(
                ValidationStage.OVERFITTING_CONTROL
            )

            if overfitting is None:
                reasons.append(
                    "Anti-overfitting validation is missing."
                )
            elif not overfitting.passed:
                reasons.append(
                    "Anti-overfitting validation did not pass."
                )

        if self.policy.require_significance_pass:
            significance = evidence.get(
                ValidationStage.STATISTICAL_SIGNIFICANCE
            )

            if significance is None:
                reasons.append(
                    "Statistical-significance validation is missing."
                )
            elif not significance.passed:
                reasons.append(
                    "Statistical-significance validation did not pass."
                )

        return reasons

    # ========================================================
    # Decision
    # ========================================================

    def decide(
        self,
        results: Iterable[ValidationResult],
    ) -> BenchmarkDecision:
        validated = self._validate_results(
            results
        )

        summary = self.summarize(
            validated
        )

        overfitting_risk, overfitting_score = (
            self._extract_overfitting_risk(
                validated
            )
        )

        presence_score = (
            summary.observed_required_stage_count
            / summary.required_stage_count
        )

        evidence_quality = (
            0.40 * presence_score
            + 0.35 * summary.pass_ratio
            + 0.25 * summary.minimum_score
        )

        evidence_quality = max(
            0.0,
            min(
                1.0,
                evidence_quality,
            ),
        )

        confidence_score = (
            0.45 * summary.average_score
            + 0.30 * summary.pass_ratio
            + 0.15 * presence_score
            + 0.10 * (
                1.0 - overfitting_score
            )
        )

        confidence_score = max(
            0.0,
            min(
                1.0,
                confidence_score,
            ),
        )

        overall_score = (
            0.45 * summary.average_score
            + 0.25 * summary.pass_ratio
            + 0.15 * evidence_quality
            + 0.15 * (
                1.0 - overfitting_score
            )
        )

        overall_score = max(
            0.0,
            min(
                1.0,
                overall_score,
            ),
        )

        reasons: List[str] = []

        if summary.missing_stages:
            reasons.append(
                "Missing required validation stages: "
                + ", ".join(
                    stage.value
                    for stage
                    in summary.missing_stages
                )
                + "."
            )

        if summary.failed_stages:
            reasons.append(
                "Failed or sub-threshold stages: "
                + ", ".join(
                    stage.value
                    for stage
                    in summary.failed_stages
                )
                + "."
            )

        if summary.inconclusive_stages:
            reasons.append(
                "Inconclusive stages: "
                + ", ".join(
                    stage.value
                    for stage
                    in summary.inconclusive_stages
                )
                + "."
            )

        reasons.extend(
            self._critical_stage_failures(
                summary
            )
        )

        if overfitting_risk in {
            OverfittingRisk.HIGH,
            OverfittingRisk.CRITICAL,
        }:
            reasons.append(
                "Material overfitting risk remains."
            )

        critical_failures = (
            self._critical_stage_failures(
                summary
            )
        )

        robust_alpha_supported = (
            summary.all_required_stages_present
            and summary.pass_ratio
            >= self.policy.minimum_pass_ratio
            and not critical_failures
            and overfitting_risk
            == OverfittingRisk.LOW
            and overall_score >= 0.70
        )

        if robust_alpha_supported:
            verdict = AlphaVerdict.ROBUST_ALPHA

        elif overfitting_risk in {
            OverfittingRisk.HIGH,
            OverfittingRisk.CRITICAL,
        }:
            verdict = AlphaVerdict.OVERFIT_RISK

        elif (
            summary.all_required_stages_present
            and summary.pass_ratio < 0.50
        ):
            verdict = AlphaVerdict.REJECTED

        elif (
            not summary.all_required_stages_present
            or summary.inconclusive_stages
        ):
            verdict = AlphaVerdict.INCONCLUSIVE

        else:
            verdict = AlphaVerdict.WEAK_EVIDENCE

        if not reasons:
            if robust_alpha_supported:
                reasons.append(
                    "Independent validation layers provide "
                    "consistent evidence of generalization, "
                    "robustness and low overfitting risk."
                )
            else:
                reasons.append(
                    "Evidence is not yet sufficient for "
                    "robust-alpha support."
                )

        return BenchmarkDecision(
            verdict=verdict,
            overall_score=overall_score,
            confidence_score=confidence_score,
            evidence_quality_score=evidence_quality,
            overfitting_risk=overfitting_risk,
            robust_alpha_supported=(
                robust_alpha_supported
            ),
            reasons=reasons,
            metadata={
                "version": VERSION,
                "summary_fingerprint": (
                    summary.fingerprint
                ),
                "policy_fingerprint": (
                    self.policy.fingerprint
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Shared BenchmarkReport adapter
    # ========================================================

    def build_report(
        self,
        *,
        research_id: str,
        strategy_name: str,
        validation_results: Iterable[ValidationResult],
        regime_results: Optional[
            Sequence[RegimeValidationResult]
        ] = None,
        sensitivity_results: Optional[
            Sequence[SensitivityResult]
        ] = None,
        overfitting_diagnostics: Optional[
            OverfittingDiagnostics
        ] = None,
    ) -> BenchmarkReport:
        if (
            not isinstance(research_id, str)
            or not research_id.strip()
        ):
            raise ValidationConfigurationError(
                "research_id must be a non-empty string."
            )

        if (
            not isinstance(strategy_name, str)
            or not strategy_name.strip()
        ):
            raise ValidationConfigurationError(
                "strategy_name must be a non-empty string."
            )

        validated = self._validate_results(
            validation_results
        )

        decision = self.decide(
            validated
        )

        regimes = list(
            regime_results or []
        )

        sensitivities = list(
            sensitivity_results or []
        )

        try:
            return BenchmarkReport(
                research_id=research_id.strip(),
                strategy_name=strategy_name.strip(),
                validation_results=validated,
                regime_results=regimes,
                sensitivity_results=sensitivities,
                overfitting_diagnostics=(
                    overfitting_diagnostics
                ),
                verdict=decision.verdict,
                overall_score=decision.overall_score,
                research_only=True,
                real_money_execution=False,
            )
        except TypeError as exc:
            raise BenchmarkEngineError(
                "BenchmarkReport shared-model constructor "
                "does not match the benchmark engine adapter."
            ) from exc

    # ========================================================
    # Complete analysis
    # ========================================================

    def analyze(
        self,
        results: Iterable[ValidationResult],
    ) -> Dict[str, Any]:
        validated = self._validate_results(
            results
        )

        summary = self.summarize(
            validated
        )

        decision = self.decide(
            validated
        )

        return {
            "version": VERSION,
            "summary": summary,
            "decision": decision,
            "summary_fingerprint": (
                summary.fingerprint
            ),
            "decision_fingerprint": (
                decision.fingerprint
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "policy_fingerprint": (
                self.policy.fingerprint
            ),
            "required_stages": [
                stage.value
                for stage
                in self.policy.required_stages
            ],
            "minimum_stage_score": (
                self.policy.minimum_stage_score
            ),
            "minimum_pass_ratio": (
                self.policy.minimum_pass_ratio
            ),
            "critical_controls": {
                "require_oos_pass": (
                    self.policy.require_oos_pass
                ),
                "require_overfitting_pass": (
                    self.policy.require_overfitting_pass
                ),
                "require_significance_pass": (
                    self.policy.require_significance_pass
                ),
            },
            "research_only": True,
            "real_money_execution": False,
        }