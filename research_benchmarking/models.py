from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


VERSION = "v6.0.2"


# ============================================================
# Exceptions
# ============================================================


class ResearchBenchmarkingError(Exception):
    """Base exception for the research benchmarking layer."""


class ValidationConfigurationError(ResearchBenchmarkingError):
    """Raised when a validation configuration is invalid."""


class StatisticalValidationError(ResearchBenchmarkingError):
    """Raised when statistical validation cannot be completed."""


# ============================================================
# Helpers
# ============================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _validate_probability(
    value: float,
    name: str,
    *,
    allow_zero: bool = False,
    allow_one: bool = False,
) -> None:
    if isinstance(value, bool):
        raise ValidationConfigurationError(
            f"{name} must be a numeric probability."
        )

    if not isinstance(value, (int, float)):
        raise ValidationConfigurationError(
            f"{name} must be numeric."
        )

    if not math.isfinite(float(value)):
        raise ValidationConfigurationError(
            f"{name} must be finite."
        )

    lower_valid = value >= 0.0 if allow_zero else value > 0.0
    upper_valid = value <= 1.0 if allow_one else value < 1.0

    if not lower_valid or not upper_valid:
        left = "[" if allow_zero else "("
        right = "]" if allow_one else ")"

        raise ValidationConfigurationError(
            f"{name} must be in {left}0, 1{right}."
        )


def _validate_positive_integer(
    value: int,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationConfigurationError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValidationConfigurationError(
            f"{name} must be greater than zero."
        )


def _validate_non_negative_integer(
    value: int,
    name: str,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationConfigurationError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValidationConfigurationError(
            f"{name} must be non-negative."
        )


def _validate_finite_number(
    value: float,
    name: str,
) -> None:
    if isinstance(value, bool):
        raise ValidationConfigurationError(
            f"{name} must be numeric."
        )

    if not isinstance(value, (int, float)):
        raise ValidationConfigurationError(
            f"{name} must be numeric."
        )

    if not math.isfinite(float(value)):
        raise ValidationConfigurationError(
            f"{name} must be finite."
        )


# ============================================================
# Enums
# ============================================================


class ValidationStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASSED = "PASSED"
    FAILED = "FAILED"
    INCONCLUSIVE = "INCONCLUSIVE"


class ValidationStage(str, Enum):
    WALK_FORWARD = "WALK_FORWARD"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    BOOTSTRAP = "BOOTSTRAP"
    MONTE_CARLO = "MONTE_CARLO"
    STATISTICAL_SIGNIFICANCE = "STATISTICAL_SIGNIFICANCE"
    MULTIPLE_TESTING = "MULTIPLE_TESTING"
    ROBUSTNESS = "ROBUSTNESS"
    SENSITIVITY = "SENSITIVITY"
    REGIME_STABILITY = "REGIME_STABILITY"
    OVERFITTING_CONTROL = "OVERFITTING_CONTROL"


class AlphaVerdict(str, Enum):
    ROBUST_ALPHA = "ROBUST_ALPHA"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    INCONCLUSIVE = "INCONCLUSIVE"
    OVERFIT_RISK = "OVERFIT_RISK"
    REJECTED = "REJECTED"


class OverfittingRisk(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SignificanceLevel(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    NOT_SIGNIFICANT = "NOT_SIGNIFICANT"


class MultipleTestingMethod(str, Enum):
    NONE = "NONE"
    BONFERRONI = "BONFERRONI"
    HOLM = "HOLM"
    BENJAMINI_HOCHBERG = "BENJAMINI_HOCHBERG"


class RegimeType(str, Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


# ============================================================
# Core validation configuration
# ============================================================


@dataclass(frozen=True)
class ValidationConfig:
    confidence_level: float = 0.95
    significance_threshold: float = 0.05

    bootstrap_samples: int = 1000
    monte_carlo_simulations: int = 1000

    walk_forward_windows: int = 5

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    minimum_observations: int = 30

    random_seed: Optional[int] = 42

    multiple_testing_method: MultipleTestingMethod = (
        MultipleTestingMethod.BENJAMINI_HOCHBERG
    )

    anti_overfitting_enabled: bool = True

    research_only: bool = True
    real_money_execution: bool = False

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_probability(
            self.confidence_level,
            "confidence_level",
        )

        _validate_probability(
            self.significance_threshold,
            "significance_threshold",
        )

        _validate_positive_integer(
            self.bootstrap_samples,
            "bootstrap_samples",
        )

        _validate_positive_integer(
            self.monte_carlo_simulations,
            "monte_carlo_simulations",
        )

        _validate_positive_integer(
            self.walk_forward_windows,
            "walk_forward_windows",
        )

        _validate_probability(
            self.train_fraction,
            "train_fraction",
        )

        _validate_probability(
            self.validation_fraction,
            "validation_fraction",
        )

        _validate_probability(
            self.test_fraction,
            "test_fraction",
        )

        _validate_positive_integer(
            self.minimum_observations,
            "minimum_observations",
        )

        if self.random_seed is not None:
            _validate_non_negative_integer(
                self.random_seed,
                "random_seed",
            )

        total_fraction = (
            self.train_fraction
            + self.validation_fraction
            + self.test_fraction
        )

        if not math.isclose(
            total_fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValidationConfigurationError(
                "train_fraction + validation_fraction + "
                "test_fraction must equal 1.0."
            )

        if self.real_money_execution:
            raise ValidationConfigurationError(
                "v6.0.2 Research Benchmarking is research-only. "
                "real_money_execution must remain False."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "confidence_level": self.confidence_level,
                "significance_threshold": (
                    self.significance_threshold
                ),
                "bootstrap_samples": self.bootstrap_samples,
                "monte_carlo_simulations": (
                    self.monte_carlo_simulations
                ),
                "walk_forward_windows": (
                    self.walk_forward_windows
                ),
                "train_fraction": self.train_fraction,
                "validation_fraction": (
                    self.validation_fraction
                ),
                "test_fraction": self.test_fraction,
                "minimum_observations": (
                    self.minimum_observations
                ),
                "random_seed": self.random_seed,
                "multiple_testing_method": (
                    self.multiple_testing_method.value
                ),
                "anti_overfitting_enabled": (
                    self.anti_overfitting_enabled
                ),
                "research_only": self.research_only,
                "real_money_execution": (
                    self.real_money_execution
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Performance metrics
# ============================================================


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    observations: int = 0

    def __post_init__(self) -> None:
        numeric_fields = {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
        }

        for name, value in numeric_fields.items():
            _validate_finite_number(
                value,
                name,
            )

        _validate_non_negative_integer(
            self.observations,
            "observations",
        )

        if not 0.0 <= self.win_rate <= 1.0:
            raise ValidationConfigurationError(
                "win_rate must be between 0 and 1."
            )

        if self.volatility < 0.0:
            raise ValidationConfigurationError(
                "volatility must be non-negative."
            )

        if self.max_drawdown < 0.0:
            raise ValidationConfigurationError(
                "max_drawdown must be non-negative."
            )

        if self.profit_factor < 0.0:
            raise ValidationConfigurationError(
                "profit_factor must be non-negative."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            asdict(self)
        )


# ============================================================
# Confidence intervals
# ============================================================


@dataclass(frozen=True)
class ConfidenceInterval:
    lower: float
    upper: float
    confidence_level: float = 0.95
    estimate: Optional[float] = None
    method: str = "bootstrap"

    def __post_init__(self) -> None:
        _validate_finite_number(
            self.lower,
            "lower",
        )

        _validate_finite_number(
            self.upper,
            "upper",
        )

        _validate_probability(
            self.confidence_level,
            "confidence_level",
        )

        if self.estimate is not None:
            _validate_finite_number(
                self.estimate,
                "estimate",
            )

        if self.lower > self.upper:
            raise ValidationConfigurationError(
                "Confidence interval lower bound "
                "cannot exceed upper bound."
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            asdict(self)
        )


# ============================================================
# Statistical test result
# ============================================================


@dataclass(frozen=True)
class StatisticalTestResult:
    test_name: str
    statistic: float
    p_value: float
    significance_threshold: float = 0.05

    adjusted_p_value: Optional[float] = None
    effect_size: Optional[float] = None

    null_hypothesis: str = ""
    alternative_hypothesis: str = ""

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.test_name.strip():
            raise ValidationConfigurationError(
                "test_name cannot be empty."
            )

        _validate_finite_number(
            self.statistic,
            "statistic",
        )

        _validate_probability(
            self.p_value,
            "p_value",
            allow_zero=True,
            allow_one=True,
        )

        _validate_probability(
            self.significance_threshold,
            "significance_threshold",
        )

        if self.adjusted_p_value is not None:
            _validate_probability(
                self.adjusted_p_value,
                "adjusted_p_value",
                allow_zero=True,
                allow_one=True,
            )

        if self.effect_size is not None:
            _validate_finite_number(
                self.effect_size,
                "effect_size",
            )

    @property
    def effective_p_value(self) -> float:
        if self.adjusted_p_value is not None:
            return self.adjusted_p_value

        return self.p_value

    @property
    def significant(self) -> bool:
        return (
            self.effective_p_value
            <= self.significance_threshold
        )

    @property
    def significance_level(self) -> SignificanceLevel:
        p_value = self.effective_p_value

        if p_value <= 0.01:
            return SignificanceLevel.STRONG

        if p_value <= 0.05:
            return SignificanceLevel.MODERATE

        if p_value <= 0.10:
            return SignificanceLevel.WEAK

        return SignificanceLevel.NOT_SIGNIFICANT

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                **asdict(self),
                "significance_level": (
                    self.significance_level.value
                ),
                "significant": self.significant,
            }
        )


# ============================================================
# Validation stage result
# ============================================================


@dataclass
class ValidationResult:
    stage: ValidationStage
    status: ValidationStatus

    score: Optional[float] = None
    message: str = ""

    metrics: Optional[PerformanceMetrics] = None

    confidence_interval: Optional[
        ConfidenceInterval
    ] = None

    statistical_tests: List[
        StatisticalTestResult
    ] = field(default_factory=list)

    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    started_at: datetime = field(
        default_factory=utc_now
    )

    completed_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.score is not None:
            _validate_finite_number(
                self.score,
                "score",
            )

            if not 0.0 <= self.score <= 1.0:
                raise ValidationConfigurationError(
                    "score must be between 0 and 1."
                )

        if (
            self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValidationConfigurationError(
                "completed_at cannot be before started_at."
            )

    @property
    def successful(self) -> bool:
        return self.status == ValidationStatus.PASSED

    @property
    def significant_test_count(self) -> int:
        return sum(
            1
            for result in self.statistical_tests
            if result.significant
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "stage": self.stage.value,
                "status": self.status.value,
                "score": self.score,
                "message": self.message,
                "metrics": (
                    asdict(self.metrics)
                    if self.metrics is not None
                    else None
                ),
                "confidence_interval": (
                    asdict(self.confidence_interval)
                    if self.confidence_interval
                    is not None
                    else None
                ),
                "statistical_tests": [
                    asdict(result)
                    for result
                    in self.statistical_tests
                ],
                "diagnostics": self.diagnostics,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Regime validation
# ============================================================


@dataclass(frozen=True)
class RegimeValidationResult:
    regime: RegimeType
    observations: int
    metrics: PerformanceMetrics
    stable: bool
    score: float
    diagnostics: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_non_negative_integer(
            self.observations,
            "observations",
        )

        _validate_finite_number(
            self.score,
            "score",
        )

        if not 0.0 <= self.score <= 1.0:
            raise ValidationConfigurationError(
                "score must be between 0 and 1."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "regime": self.regime.value,
                "observations": self.observations,
                "metrics": asdict(self.metrics),
                "stable": self.stable,
                "score": self.score,
                "diagnostics": self.diagnostics,
            }
        )


# ============================================================
# Sensitivity result
# ============================================================


@dataclass(frozen=True)
class SensitivityResult:
    parameter_name: str
    baseline_value: float
    tested_values: List[float]
    performance_values: List[float]
    sensitivity_score: float
    robust: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.parameter_name.strip():
            raise ValidationConfigurationError(
                "parameter_name cannot be empty."
            )

        _validate_finite_number(
            self.baseline_value,
            "baseline_value",
        )

        if not self.tested_values:
            raise ValidationConfigurationError(
                "tested_values cannot be empty."
            )

        if (
            len(self.tested_values)
            != len(self.performance_values)
        ):
            raise ValidationConfigurationError(
                "tested_values and performance_values "
                "must have equal length."
            )

        for value in self.tested_values:
            _validate_finite_number(
                value,
                "tested_value",
            )

        for value in self.performance_values:
            _validate_finite_number(
                value,
                "performance_value",
            )

        _validate_finite_number(
            self.sensitivity_score,
            "sensitivity_score",
        )

        if not 0.0 <= self.sensitivity_score <= 1.0:
            raise ValidationConfigurationError(
                "sensitivity_score must be between 0 and 1."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            asdict(self)
        )


# ============================================================
# Overfitting diagnostics
# ============================================================


@dataclass(frozen=True)
class OverfittingDiagnostics:
    in_sample_score: float
    out_of_sample_score: float

    degradation_ratio: float

    parameter_count: int
    observation_count: int

    multiple_testing_count: int = 1

    probability_of_backtest_overfitting: Optional[
        float
    ] = None

    deflated_sharpe_ratio: Optional[
        float
    ] = None

    risk: OverfittingRisk = OverfittingRisk.MODERATE

    warnings: List[str] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_finite_number(
            self.in_sample_score,
            "in_sample_score",
        )

        _validate_finite_number(
            self.out_of_sample_score,
            "out_of_sample_score",
        )

        _validate_finite_number(
            self.degradation_ratio,
            "degradation_ratio",
        )

        _validate_non_negative_integer(
            self.parameter_count,
            "parameter_count",
        )

        _validate_non_negative_integer(
            self.observation_count,
            "observation_count",
        )

        _validate_positive_integer(
            self.multiple_testing_count,
            "multiple_testing_count",
        )

        if (
            self.probability_of_backtest_overfitting
            is not None
        ):
            _validate_probability(
                self.probability_of_backtest_overfitting,
                "probability_of_backtest_overfitting",
                allow_zero=True,
                allow_one=True,
            )

        if self.deflated_sharpe_ratio is not None:
            _validate_finite_number(
                self.deflated_sharpe_ratio,
                "deflated_sharpe_ratio",
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                **asdict(self),
                "risk": self.risk.value,
            }
        )


# ============================================================
# Complete benchmark report
# ============================================================


@dataclass
class BenchmarkReport:
    research_id: str
    strategy_name: str

    validation_results: List[
        ValidationResult
    ] = field(default_factory=list)

    regime_results: List[
        RegimeValidationResult
    ] = field(default_factory=list)

    sensitivity_results: List[
        SensitivityResult
    ] = field(default_factory=list)

    overfitting_diagnostics: Optional[
        OverfittingDiagnostics
    ] = None

    verdict: AlphaVerdict = (
        AlphaVerdict.INCONCLUSIVE
    )

    overall_score: float = 0.0

    created_at: datetime = field(
        default_factory=utc_now
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    research_only: bool = True
    real_money_execution: bool = False

    def __post_init__(self) -> None:
        if not self.research_id.strip():
            raise ValidationConfigurationError(
                "research_id cannot be empty."
            )

        if not self.strategy_name.strip():
            raise ValidationConfigurationError(
                "strategy_name cannot be empty."
            )

        _validate_finite_number(
            self.overall_score,
            "overall_score",
        )

        if not 0.0 <= self.overall_score <= 1.0:
            raise ValidationConfigurationError(
                "overall_score must be between 0 and 1."
            )

        if self.real_money_execution:
            raise ValidationConfigurationError(
                "BenchmarkReport must remain research-only."
            )

    @property
    def passed_stage_count(self) -> int:
        return sum(
            1
            for result in self.validation_results
            if result.status
            == ValidationStatus.PASSED
        )

    @property
    def failed_stage_count(self) -> int:
        return sum(
            1
            for result in self.validation_results
            if result.status
            == ValidationStatus.FAILED
        )

    @property
    def inconclusive_stage_count(self) -> int:
        return sum(
            1
            for result in self.validation_results
            if result.status
            == ValidationStatus.INCONCLUSIVE
        )

    @property
    def all_required_stages_passed(self) -> bool:
        required = {
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
        }

        passed = {
            result.stage
            for result in self.validation_results
            if result.status
            == ValidationStatus.PASSED
        }

        return required.issubset(passed)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "research_id": self.research_id,
                "strategy_name": self.strategy_name,
                "validation_results": [
                    result.fingerprint
                    for result
                    in self.validation_results
                ],
                "regime_results": [
                    result.fingerprint
                    for result
                    in self.regime_results
                ],
                "sensitivity_results": [
                    result.fingerprint
                    for result
                    in self.sensitivity_results
                ],
                "overfitting_diagnostics": (
                    self.overfitting_diagnostics.fingerprint
                    if self.overfitting_diagnostics
                    is not None
                    else None
                ),
                "verdict": self.verdict.value,
                "overall_score": self.overall_score,
                "metadata": self.metadata,
                "research_only": self.research_only,
                "real_money_execution": (
                    self.real_money_execution
                ),
            }
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "research_id": self.research_id,
            "strategy_name": self.strategy_name,
            "verdict": self.verdict.value,
            "overall_score": self.overall_score,
            "validation_stage_count": len(
                self.validation_results
            ),
            "passed_stage_count": (
                self.passed_stage_count
            ),
            "failed_stage_count": (
                self.failed_stage_count
            ),
            "inconclusive_stage_count": (
                self.inconclusive_stage_count
            ),
            "regime_count": len(
                self.regime_results
            ),
            "sensitivity_test_count": len(
                self.sensitivity_results
            ),
            "all_required_stages_passed": (
                self.all_required_stages_passed
            ),
            "overfitting_risk": (
                self.overfitting_diagnostics.risk.value
                if self.overfitting_diagnostics
                is not None
                else None
            ),
            "research_only": self.research_only,
            "real_money_execution": (
                self.real_money_execution
            ),
            "fingerprint": self.fingerprint,
        }