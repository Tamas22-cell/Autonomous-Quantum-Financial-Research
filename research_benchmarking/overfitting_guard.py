from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from research_benchmarking.models import (
    OverfittingDiagnostics,
    OverfittingRisk,
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


class OverfittingGuardError(Exception):
    """Raised when anti-overfitting validation cannot be performed."""


# ============================================================
# Helpers
# ============================================================


def _validate_numeric(
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

    if not math.isfinite(numeric):
        raise ValidationConfigurationError(
            f"{name} must be finite."
        )

    return numeric


def _validate_non_negative(
    value: float,
    name: str,
) -> float:
    numeric = _validate_numeric(value, name)

    if numeric < 0.0:
        raise ValidationConfigurationError(
            f"{name} must be non-negative."
        )

    return numeric


def _validate_ratio(
    value: float,
    name: str,
) -> float:
    numeric = _validate_numeric(value, name)

    if not 0.0 <= numeric <= 1.0:
        raise ValidationConfigurationError(
            f"{name} must be between 0 and 1."
        )

    return numeric


def _validate_positive_integer(
    value: int,
    name: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValidationConfigurationError(
            f"{name} must be a positive integer."
        )

    return value


# ============================================================
# IS / OOS degradation
# ============================================================


@dataclass(frozen=True)
class GeneralizationGap:
    in_sample_score: float
    out_of_sample_score: float
    absolute_gap: float
    degradation_ratio: float
    retained_performance_ratio: float
    positive_out_of_sample: bool
    acceptable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "in_sample_score": self.in_sample_score,
            "out_of_sample_score": self.out_of_sample_score,
            "absolute_gap": self.absolute_gap,
            "degradation_ratio": self.degradation_ratio,
            "retained_performance_ratio": self.retained_performance_ratio,
        }.items():
            _validate_numeric(value, name)

        if not isinstance(self.positive_out_of_sample, bool):
            raise ValidationConfigurationError(
                "positive_out_of_sample must be bool."
            )

        if not isinstance(self.acceptable, bool):
            raise ValidationConfigurationError(
                "acceptable must be bool."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "in_sample_score": self.in_sample_score,
                "out_of_sample_score": self.out_of_sample_score,
                "absolute_gap": self.absolute_gap,
                "degradation_ratio": self.degradation_ratio,
                "retained_performance_ratio": (
                    self.retained_performance_ratio
                ),
                "positive_out_of_sample": (
                    self.positive_out_of_sample
                ),
                "acceptable": self.acceptable,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Selection-bias diagnostics
# ============================================================


@dataclass(frozen=True)
class SelectionBiasDiagnostics:
    tested_strategies: int
    selected_rank: int
    best_score: float
    median_score: float
    score_advantage: float
    normalized_advantage: float
    selection_pressure: float
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive_integer(
            self.tested_strategies,
            "tested_strategies",
        )

        _validate_positive_integer(
            self.selected_rank,
            "selected_rank",
        )

        if self.selected_rank > self.tested_strategies:
            raise ValidationConfigurationError(
                "selected_rank cannot exceed tested_strategies."
            )

        for name, value in {
            "best_score": self.best_score,
            "median_score": self.median_score,
            "score_advantage": self.score_advantage,
            "normalized_advantage": self.normalized_advantage,
        }.items():
            _validate_numeric(value, name)

        _validate_ratio(
            self.selection_pressure,
            "selection_pressure",
        )

        _validate_ratio(
            self.risk_score,
            "risk_score",
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "tested_strategies": self.tested_strategies,
                "selected_rank": self.selected_rank,
                "best_score": self.best_score,
                "median_score": self.median_score,
                "score_advantage": self.score_advantage,
                "normalized_advantage": self.normalized_advantage,
                "selection_pressure": self.selection_pressure,
                "risk_score": self.risk_score,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Parameter-instability diagnostics
# ============================================================


@dataclass(frozen=True)
class ParameterInstabilityDiagnostics:
    parameter_count: int
    stable_parameter_count: int
    stable_parameter_ratio: float
    average_stability_score: float
    minimum_stability_score: float
    instability_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_positive_integer(
            self.parameter_count,
            "parameter_count",
        )

        if (
            isinstance(self.stable_parameter_count, bool)
            or not isinstance(self.stable_parameter_count, int)
            or self.stable_parameter_count < 0
        ):
            raise ValidationConfigurationError(
                "stable_parameter_count must be "
                "a non-negative integer."
            )

        if self.stable_parameter_count > self.parameter_count:
            raise ValidationConfigurationError(
                "stable_parameter_count cannot exceed "
                "parameter_count."
            )

        _validate_ratio(
            self.stable_parameter_ratio,
            "stable_parameter_ratio",
        )

        _validate_ratio(
            self.average_stability_score,
            "average_stability_score",
        )

        _validate_ratio(
            self.minimum_stability_score,
            "minimum_stability_score",
        )

        _validate_ratio(
            self.instability_score,
            "instability_score",
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "parameter_count": self.parameter_count,
                "stable_parameter_count": (
                    self.stable_parameter_count
                ),
                "stable_parameter_ratio": (
                    self.stable_parameter_ratio
                ),
                "average_stability_score": (
                    self.average_stability_score
                ),
                "minimum_stability_score": (
                    self.minimum_stability_score
                ),
                "instability_score": self.instability_score,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Anti-overfitting report
# ============================================================


@dataclass(frozen=True)
class OverfittingGuardReport:
    generalization_gap: GeneralizationGap
    selection_bias: SelectionBiasDiagnostics
    parameter_instability: ParameterInstabilityDiagnostics
    multiple_testing_risk: float
    regime_instability_risk: float
    robustness_failure_risk: float
    overall_risk_score: float
    risk_level: OverfittingRisk
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(
            self.generalization_gap,
            GeneralizationGap,
        ):
            raise ValidationConfigurationError(
                "generalization_gap must be GeneralizationGap."
            )

        if not isinstance(
            self.selection_bias,
            SelectionBiasDiagnostics,
        ):
            raise ValidationConfigurationError(
                "selection_bias must be SelectionBiasDiagnostics."
            )

        if not isinstance(
            self.parameter_instability,
            ParameterInstabilityDiagnostics,
        ):
            raise ValidationConfigurationError(
                "parameter_instability must be "
                "ParameterInstabilityDiagnostics."
            )

        for name, value in {
            "multiple_testing_risk": self.multiple_testing_risk,
            "regime_instability_risk": self.regime_instability_risk,
            "robustness_failure_risk": self.robustness_failure_risk,
            "overall_risk_score": self.overall_risk_score,
        }.items():
            _validate_ratio(value, name)

        if not isinstance(self.risk_level, OverfittingRisk):
            raise ValidationConfigurationError(
                "risk_level must be OverfittingRisk."
            )

        if not isinstance(self.passed, bool):
            raise ValidationConfigurationError(
                "passed must be bool."
            )

        if not isinstance(self.reasons, list):
            raise ValidationConfigurationError(
                "reasons must be a list."
            )

        if not all(
            isinstance(reason, str)
            for reason in self.reasons
        ):
            raise ValidationConfigurationError(
                "reasons must contain strings."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "generalization_gap": (
                    self.generalization_gap.fingerprint
                ),
                "selection_bias": (
                    self.selection_bias.fingerprint
                ),
                "parameter_instability": (
                    self.parameter_instability.fingerprint
                ),
                "multiple_testing_risk": (
                    self.multiple_testing_risk
                ),
                "regime_instability_risk": (
                    self.regime_instability_risk
                ),
                "robustness_failure_risk": (
                    self.robustness_failure_risk
                ),
                "overall_risk_score": (
                    self.overall_risk_score
                ),
                "risk_level": self.risk_level.value,
                "passed": self.passed,
                "reasons": self.reasons,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Engine
# ============================================================


class OverfittingGuard:
    """
    Central anti-overfitting control for v6.0.2.

    Combines independent evidence from:

    - in-sample vs out-of-sample degradation
    - strategy-selection pressure
    - parameter instability
    - multiple-testing risk
    - regime instability
    - robustness failures

    This engine does not prove the existence of alpha.
    It estimates whether the available validation evidence
    is consistent with a robust result or carries material
    historical-overfitting risk.
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        maximum_generalization_gap: float = 0.50,
        minimum_retained_performance: float = 0.50,
        maximum_overall_risk: float = 0.50,
        high_risk_threshold: float = 0.70,
        critical_risk_threshold: float = 0.85,
    ) -> None:
        self.config = config or ValidationConfig()

        self.maximum_generalization_gap = _validate_non_negative(
            maximum_generalization_gap,
            "maximum_generalization_gap",
        )

        self.minimum_retained_performance = _validate_ratio(
            minimum_retained_performance,
            "minimum_retained_performance",
        )

        self.maximum_overall_risk = _validate_ratio(
            maximum_overall_risk,
            "maximum_overall_risk",
        )

        self.high_risk_threshold = _validate_ratio(
            high_risk_threshold,
            "high_risk_threshold",
        )

        self.critical_risk_threshold = _validate_ratio(
            critical_risk_threshold,
            "critical_risk_threshold",
        )

        if self.high_risk_threshold < self.maximum_overall_risk:
            raise ValidationConfigurationError(
                "high_risk_threshold cannot be below "
                "maximum_overall_risk."
            )

        if self.critical_risk_threshold < self.high_risk_threshold:
            raise ValidationConfigurationError(
                "critical_risk_threshold cannot be below "
                "high_risk_threshold."
            )

    # ========================================================
    # Generalization gap
    # ========================================================

    def analyze_generalization_gap(
        self,
        *,
        in_sample_score: float,
        out_of_sample_score: float,
    ) -> GeneralizationGap:
        in_sample = _validate_numeric(
            in_sample_score,
            "in_sample_score",
        )

        out_of_sample = _validate_numeric(
            out_of_sample_score,
            "out_of_sample_score",
        )

        absolute_gap = in_sample - out_of_sample

        if math.isclose(
            in_sample,
            0.0,
            abs_tol=1e-15,
        ):
            if math.isclose(
                out_of_sample,
                0.0,
                abs_tol=1e-15,
            ):
                degradation_ratio = 0.0
                retained_ratio = 1.0
            elif out_of_sample > 0.0:
                degradation_ratio = -1.0
                retained_ratio = 1.0
            else:
                degradation_ratio = 1.0
                retained_ratio = 0.0
        else:
            degradation_ratio = (
                absolute_gap
                / abs(in_sample)
            )

            retained_ratio = (
                out_of_sample
                / abs(in_sample)
            )

        positive_oos = out_of_sample > 0.0

        acceptable = (
            positive_oos
            and degradation_ratio
            <= self.maximum_generalization_gap
            and retained_ratio
            >= self.minimum_retained_performance
        )

        return GeneralizationGap(
            in_sample_score=in_sample,
            out_of_sample_score=out_of_sample,
            absolute_gap=absolute_gap,
            degradation_ratio=degradation_ratio,
            retained_performance_ratio=retained_ratio,
            positive_out_of_sample=positive_oos,
            acceptable=acceptable,
            metadata={
                "version": VERSION,
                "maximum_generalization_gap": (
                    self.maximum_generalization_gap
                ),
                "minimum_retained_performance": (
                    self.minimum_retained_performance
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Selection bias
    # ========================================================

    def analyze_selection_bias(
        self,
        strategy_scores: Sequence[float],
        *,
        selected_rank: int = 1,
    ) -> SelectionBiasDiagnostics:
        if not strategy_scores:
            raise OverfittingGuardError(
                "strategy_scores cannot be empty."
            )

        scores = [
            _validate_numeric(
                value,
                f"strategy_scores[{index}]",
            )
            for index, value in enumerate(strategy_scores)
        ]

        tested = len(scores)

        _validate_positive_integer(
            selected_rank,
            "selected_rank",
        )

        if selected_rank > tested:
            raise ValidationConfigurationError(
                "selected_rank cannot exceed "
                "number of tested strategies."
            )

        ordered = sorted(
            scores,
            reverse=True,
        )

        selected_score = ordered[
            selected_rank - 1
        ]

        median_sorted = sorted(scores)
        midpoint = tested // 2

        if tested % 2:
            median_score = median_sorted[midpoint]
        else:
            median_score = (
                median_sorted[midpoint - 1]
                + median_sorted[midpoint]
            ) / 2.0

        score_advantage = (
            selected_score - median_score
        )

        scale = max(
            abs(median_score),
            abs(selected_score),
            1e-12,
        )

        normalized_advantage = (
            score_advantage / scale
        )

        if tested <= 1:
            selection_pressure = 0.0
        else:
            selection_pressure = min(
                1.0,
                math.log10(float(tested))
                / 3.0,
            )

        advantage_component = min(
            1.0,
            max(
                0.0,
                normalized_advantage,
            ),
        )

        rank_component = (
            1.0
            - (
                (selected_rank - 1)
                / max(tested - 1, 1)
            )
        )

        risk_score = (
            0.50 * selection_pressure
            + 0.30 * advantage_component
            + 0.20 * rank_component
        )

        risk_score = max(
            0.0,
            min(1.0, risk_score),
        )

        return SelectionBiasDiagnostics(
            tested_strategies=tested,
            selected_rank=selected_rank,
            best_score=selected_score,
            median_score=median_score,
            score_advantage=score_advantage,
            normalized_advantage=normalized_advantage,
            selection_pressure=selection_pressure,
            risk_score=risk_score,
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Parameter instability
    # ========================================================

    def analyze_parameter_instability(
        self,
        stability_scores: Sequence[float],
        *,
        stable_threshold: float = 0.65,
    ) -> ParameterInstabilityDiagnostics:
        if not stability_scores:
            raise OverfittingGuardError(
                "stability_scores cannot be empty."
            )

        threshold = _validate_ratio(
            stable_threshold,
            "stable_threshold",
        )

        scores = [
            _validate_ratio(
                value,
                f"stability_scores[{index}]",
            )
            for index, value in enumerate(
                stability_scores
            )
        ]

        count = len(scores)

        stable_count = sum(
            1
            for value in scores
            if value >= threshold
        )

        stable_ratio = stable_count / count
        average_score = sum(scores) / count
        minimum_score = min(scores)

        instability_score = (
            0.50 * (1.0 - stable_ratio)
            + 0.30 * (1.0 - average_score)
            + 0.20 * (1.0 - minimum_score)
        )

        instability_score = max(
            0.0,
            min(1.0, instability_score),
        )

        return ParameterInstabilityDiagnostics(
            parameter_count=count,
            stable_parameter_count=stable_count,
            stable_parameter_ratio=stable_ratio,
            average_stability_score=average_score,
            minimum_stability_score=minimum_score,
            instability_score=instability_score,
            metadata={
                "version": VERSION,
                "stable_threshold": threshold,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Risk classification
    # ========================================================

    def classify_risk(
        self,
        risk_score: float,
    ) -> OverfittingRisk:
        score = _validate_ratio(
            risk_score,
            "risk_score",
        )

        if score >= self.critical_risk_threshold:
            return OverfittingRisk.CRITICAL

        if score >= self.high_risk_threshold:
            return OverfittingRisk.HIGH

        if score > self.maximum_overall_risk:
            return OverfittingRisk.MODERATE

        return OverfittingRisk.LOW

    # ========================================================
    # Full evaluation
    # ========================================================

    def evaluate(
        self,
        *,
        in_sample_score: float,
        out_of_sample_score: float,
        strategy_scores: Sequence[float],
        parameter_stability_scores: Sequence[float],
        selected_rank: int = 1,
        multiple_testing_risk: float = 0.0,
        regime_instability_risk: float = 0.0,
        robustness_failure_risk: float = 0.0,
    ) -> OverfittingGuardReport:
        multiple_testing = _validate_ratio(
            multiple_testing_risk,
            "multiple_testing_risk",
        )

        regime_risk = _validate_ratio(
            regime_instability_risk,
            "regime_instability_risk",
        )

        robustness_risk = _validate_ratio(
            robustness_failure_risk,
            "robustness_failure_risk",
        )

        generalization = (
            self.analyze_generalization_gap(
                in_sample_score=in_sample_score,
                out_of_sample_score=out_of_sample_score,
            )
        )

        selection_bias = (
            self.analyze_selection_bias(
                strategy_scores,
                selected_rank=selected_rank,
            )
        )

        parameter_instability = (
            self.analyze_parameter_instability(
                parameter_stability_scores
            )
        )

        generalization_risk = max(
            0.0,
            min(
                1.0,
                max(
                    0.0,
                    generalization.degradation_ratio,
                ),
            ),
        )

        if not generalization.positive_out_of_sample:
            generalization_risk = 1.0

        overall_risk = (
            0.30 * generalization_risk
            + 0.20 * selection_bias.risk_score
            + 0.20 * parameter_instability.instability_score
            + 0.10 * multiple_testing
            + 0.10 * regime_risk
            + 0.10 * robustness_risk
        )

        overall_risk = max(
            0.0,
            min(1.0, overall_risk),
        )

        risk_level = self.classify_risk(
            overall_risk
        )

        reasons: List[str] = []

        if not generalization.acceptable:
            reasons.append(
                "Material in-sample / out-of-sample "
                "generalization gap."
            )

        if selection_bias.risk_score >= 0.50:
            reasons.append(
                "Material strategy-selection pressure."
            )

        if parameter_instability.instability_score >= 0.50:
            reasons.append(
                "Parameter neighborhood instability."
            )

        if multiple_testing >= 0.50:
            reasons.append(
                "Material multiple-testing risk."
            )

        if regime_risk >= 0.50:
            reasons.append(
                "Material regime-instability risk."
            )

        if robustness_risk >= 0.50:
            reasons.append(
                "Material robustness-failure risk."
            )

        passed = (
            self.config.anti_overfitting_enabled
            and generalization.acceptable
            and overall_risk <= self.maximum_overall_risk
            and risk_level == OverfittingRisk.LOW
        )

        return OverfittingGuardReport(
            generalization_gap=generalization,
            selection_bias=selection_bias,
            parameter_instability=parameter_instability,
            multiple_testing_risk=multiple_testing,
            regime_instability_risk=regime_risk,
            robustness_failure_risk=robustness_risk,
            overall_risk_score=overall_risk,
            risk_level=risk_level,
            passed=passed,
            reasons=reasons,
            metadata={
                "version": VERSION,
                "anti_overfitting_enabled": (
                    self.config.anti_overfitting_enabled
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Mapping API
    # ========================================================

    def validate_mapping(
        self,
        data: Mapping[str, Any],
    ) -> OverfittingGuardReport:
        if not isinstance(data, Mapping):
            raise ValidationConfigurationError(
                "data must be a mapping."
            )

        required = {
            "in_sample_score",
            "out_of_sample_score",
            "strategy_scores",
            "parameter_stability_scores",
        }

        missing = required.difference(data.keys())

        if missing:
            raise OverfittingGuardError(
                "Missing required fields: "
                + ", ".join(sorted(missing))
            )

        return self.evaluate(
            in_sample_score=data["in_sample_score"],
            out_of_sample_score=data["out_of_sample_score"],
            strategy_scores=data["strategy_scores"],
            parameter_stability_scores=(
                data["parameter_stability_scores"]
            ),
            selected_rank=data.get(
                "selected_rank",
                1,
            ),
            multiple_testing_risk=data.get(
                "multiple_testing_risk",
                0.0,
            ),
            regime_instability_risk=data.get(
                "regime_instability_risk",
                0.0,
            ),
            robustness_failure_risk=data.get(
                "robustness_failure_risk",
                0.0,
            ),
        )

    # ========================================================
    # Shared model adapter
    # ========================================================

    def to_overfitting_diagnostics(
        self,
        report: OverfittingGuardReport,
    ) -> Optional[OverfittingDiagnostics]:
        """
        Best-effort adapter to the shared v6.0.2 model.

        The anti-overfitting calculation layer remains
        independent if the shared dataclass constructor evolves.
        """

        try:
            return OverfittingDiagnostics(
                in_sample_score=(
                    report.generalization_gap.in_sample_score
                ),
                out_of_sample_score=(
                    report.generalization_gap.out_of_sample_score
                ),
                degradation_ratio=(
                    report.generalization_gap.degradation_ratio
                ),
                parameter_stability=(
                    report.parameter_instability
                    .average_stability_score
                ),
                selection_bias_score=(
                    report.selection_bias.risk_score
                ),
                overfitting_risk=report.risk_level,
                metadata={
                    "version": VERSION,
                    "overall_risk_score": (
                        report.overall_risk_score
                    ),
                    "research_only": True,
                    "real_money_execution": False,
                },
            )
        except TypeError:
            return None

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: OverfittingGuardReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Anti-overfitting controls found no material "
            "historical-overfitting warning."
            if report.passed
            else (
                "Anti-overfitting controls detected "
                "material generalization or selection risk."
            )
        )

        validation_score = max(
            0.0,
            min(
                1.0,
                1.0 - report.overall_risk_score,
            ),
        )

        return ValidationResult(
            stage=ValidationStage.OVERFITTING_CONTROL,
            status=status,
            score=validation_score,
            message=message,
            diagnostics={
                "overall_risk_score": (
                    report.overall_risk_score
                ),
                "risk_level": (
                    report.risk_level.value
                ),
                "generalization_gap": (
                    report.generalization_gap
                    .degradation_ratio
                ),
                "retained_performance_ratio": (
                    report.generalization_gap
                    .retained_performance_ratio
                ),
                "selection_bias_risk": (
                    report.selection_bias.risk_score
                ),
                "parameter_instability": (
                    report.parameter_instability
                    .instability_score
                ),
                "multiple_testing_risk": (
                    report.multiple_testing_risk
                ),
                "regime_instability_risk": (
                    report.regime_instability_risk
                ),
                "robustness_failure_risk": (
                    report.robustness_failure_risk
                ),
                "reasons": list(report.reasons),
                "report_fingerprint": (
                    report.fingerprint
                ),
            },
            metadata={
                "version": VERSION,
                "anti_overfitting_enabled": (
                    self.config.anti_overfitting_enabled
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "anti_overfitting_enabled": (
                self.config.anti_overfitting_enabled
            ),
            "maximum_generalization_gap": (
                self.maximum_generalization_gap
            ),
            "minimum_retained_performance": (
                self.minimum_retained_performance
            ),
            "maximum_overall_risk": (
                self.maximum_overall_risk
            ),
            "high_risk_threshold": (
                self.high_risk_threshold
            ),
            "critical_risk_threshold": (
                self.critical_risk_threshold
            ),
            "risk_components": [
                "generalization_gap",
                "selection_bias",
                "parameter_instability",
                "multiple_testing",
                "regime_instability",
                "robustness_failure",
            ],
            "research_only": True,
            "real_money_execution": False,
        }