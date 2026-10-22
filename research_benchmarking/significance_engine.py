from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from research_benchmarking.models import (
    ConfidenceInterval,
    SignificanceLevel,
    StatisticalTestResult,
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


class SignificanceValidationError(Exception):
    """Raised when significance validation cannot be completed."""


# ============================================================
# Helpers
# ============================================================


def _validate_numeric_sequence(
    values: Sequence[float],
) -> List[float]:
    if not values:
        raise SignificanceValidationError(
            "Statistical input cannot be empty."
        )

    result: List[float] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValidationConfigurationError(
                "Observations must be numeric."
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValidationConfigurationError(
                "Observations must be finite."
            )

        result.append(numeric)

    return result


def _sample_standard_deviation(
    values: Sequence[float],
) -> float:
    clean_values = _validate_numeric_sequence(values)

    if len(clean_values) < 2:
        return 0.0

    average = mean(clean_values)

    variance = sum(
        (value - average) ** 2
        for value in clean_values
    ) / (len(clean_values) - 1)

    return math.sqrt(variance)


def _normal_cdf(value: float) -> float:
    return 0.5 * (
        1.0
        + math.erf(
            value / math.sqrt(2.0)
        )
    )


def _normal_two_sided_p_value(
    statistic: float,
) -> float:
    probability = 2.0 * (
        1.0 - _normal_cdf(abs(statistic))
    )

    return max(
        0.0,
        min(1.0, probability),
    )


def _normal_critical_value(
    confidence_level: float,
) -> float:
    """
    Approximate normal critical values without requiring SciPy.

    Common research confidence levels are handled explicitly.
    Other values use a binary search over the normal CDF.
    """

    if (
        isinstance(confidence_level, bool)
        or not isinstance(
            confidence_level,
            (int, float),
        )
        or not math.isfinite(
            float(confidence_level)
        )
        or not 0.0 < confidence_level < 1.0
    ):
        raise ValidationConfigurationError(
            "confidence_level must be between 0 and 1."
        )

    common_values = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }

    for level, critical in common_values.items():
        if math.isclose(
            confidence_level,
            level,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return critical

    target = (
        1.0 + confidence_level
    ) / 2.0

    lower = 0.0
    upper = 10.0

    for _ in range(100):
        midpoint = (
            lower + upper
        ) / 2.0

        if _normal_cdf(midpoint) < target:
            lower = midpoint
        else:
            upper = midpoint

    return (
        lower + upper
    ) / 2.0


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class SignificanceDiagnostics:
    sample_size: int
    sample_mean: float
    null_mean: float
    standard_deviation: float
    standard_error: float
    test_statistic: float
    p_value: float
    effect_size: float
    confidence_interval: ConfidenceInterval
    significance_level: SignificanceLevel
    statistically_significant: bool
    direction_positive: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_size, bool)
            or not isinstance(self.sample_size, int)
            or self.sample_size <= 0
        ):
            raise ValidationConfigurationError(
                "sample_size must be a positive integer."
            )

        for name, value in {
            "sample_mean": self.sample_mean,
            "null_mean": self.null_mean,
            "standard_deviation": (
                self.standard_deviation
            ),
            "standard_error": self.standard_error,
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationConfigurationError(
                    f"{name} must be finite numeric."
                )

        if self.standard_deviation < 0.0:
            raise ValidationConfigurationError(
                "standard_deviation cannot be negative."
            )

        if self.standard_error < 0.0:
            raise ValidationConfigurationError(
                "standard_error cannot be negative."
            )

        if not 0.0 <= self.p_value <= 1.0:
            raise ValidationConfigurationError(
                "p_value must be between 0 and 1."
            )

        if not isinstance(
            self.significance_level,
            SignificanceLevel,
        ):
            raise ValidationConfigurationError(
                "significance_level must be "
                "a SignificanceLevel."
            )

    @property
    def mean_difference(self) -> float:
        return (
            self.sample_mean
            - self.null_mean
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "sample_size": self.sample_size,
                "sample_mean": self.sample_mean,
                "null_mean": self.null_mean,
                "standard_deviation": (
                    self.standard_deviation
                ),
                "standard_error": (
                    self.standard_error
                ),
                "test_statistic": (
                    self.test_statistic
                ),
                "p_value": self.p_value,
                "effect_size": self.effect_size,
                "confidence_interval": (
                    self.confidence_interval.fingerprint
                ),
                "significance_level": (
                    self.significance_level.value
                ),
                "statistically_significant": (
                    self.statistically_significant
                ),
                "direction_positive": (
                    self.direction_positive
                ),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class SignificanceReport:
    diagnostics: SignificanceDiagnostics
    statistical_test: StatisticalTestResult
    minimum_effect_size: float
    effect_size_sufficient: bool
    passed: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_effect_size, bool)
            or not isinstance(
                self.minimum_effect_size,
                (int, float),
            )
            or not math.isfinite(
                float(self.minimum_effect_size)
            )
            or self.minimum_effect_size < 0.0
        ):
            raise ValidationConfigurationError(
                "minimum_effect_size must be "
                "finite and non-negative."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "diagnostics": (
                    self.diagnostics.fingerprint
                ),
                "statistical_test": (
                    self.statistical_test.fingerprint
                ),
                "minimum_effect_size": (
                    self.minimum_effect_size
                ),
                "effect_size_sufficient": (
                    self.effect_size_sufficient
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Significance Engine
# ============================================================


class SignificanceEngine:
    """
    Statistical significance diagnostics for research returns
    or other numeric alpha observations.

    Provides:
    - mean-vs-null significance testing
    - two-sided p-values
    - standardized effect size
    - confidence intervals
    - significance classification
    - ValidationResult integration

    The implementation intentionally avoids a hard SciPy
    dependency so the benchmarking layer remains lightweight.
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[
            ValidationConfig
        ] = None,
        *,
        null_mean: float = 0.0,
        minimum_effect_size: float = 0.20,
    ) -> None:
        self.config = (
            config or ValidationConfig()
        )

        if (
            isinstance(null_mean, bool)
            or not isinstance(
                null_mean,
                (int, float),
            )
            or not math.isfinite(
                float(null_mean)
            )
        ):
            raise ValidationConfigurationError(
                "null_mean must be finite numeric."
            )

        if (
            isinstance(minimum_effect_size, bool)
            or not isinstance(
                minimum_effect_size,
                (int, float),
            )
            or not math.isfinite(
                float(minimum_effect_size)
            )
            or minimum_effect_size < 0.0
        ):
            raise ValidationConfigurationError(
                "minimum_effect_size must be "
                "finite and non-negative."
            )

        self.null_mean = float(null_mean)
        self.minimum_effect_size = float(
            minimum_effect_size
        )

    # ========================================================
    # Significance classification
    # ========================================================

    @staticmethod
    def classify_significance(
        p_value: float,
    ) -> SignificanceLevel:
        if (
            isinstance(p_value, bool)
            or not isinstance(
                p_value,
                (int, float),
            )
            or not math.isfinite(
                float(p_value)
            )
            or not 0.0 <= p_value <= 1.0
        ):
            raise ValidationConfigurationError(
                "p_value must be between 0 and 1."
            )

        if p_value <= 0.01:
            return SignificanceLevel.STRONG

        if p_value <= 0.05:
            return SignificanceLevel.MODERATE

        if p_value <= 0.10:
            return SignificanceLevel.WEAK

        return SignificanceLevel.NOT_SIGNIFICANT

    # ========================================================
    # Core diagnostics
    # ========================================================

    def analyze(
        self,
        values: Sequence[float],
    ) -> SignificanceDiagnostics:
        clean_values = _validate_numeric_sequence(
            values
        )

        if (
            len(clean_values)
            < self.config.minimum_observations
        ):
            raise SignificanceValidationError(
                "Insufficient observations for "
                "statistical significance analysis."
            )

        sample_size = len(clean_values)

        sample_mean = mean(clean_values)

        standard_deviation = (
            _sample_standard_deviation(
                clean_values
            )
        )

        mean_difference = (
            sample_mean
            - self.null_mean
        )

        if standard_deviation <= 1e-12:
            standard_error = 0.0

            if mean_difference > 0.0:
                test_statistic = 1e12
                p_value = 0.0
                effect_size = 1e12

            elif mean_difference < 0.0:
                test_statistic = -1e12
                p_value = 0.0
                effect_size = -1e12

            else:
                test_statistic = 0.0
                p_value = 1.0
                effect_size = 0.0

        else:
            standard_error = (
                standard_deviation
                / math.sqrt(sample_size)
            )

            test_statistic = (
                mean_difference
                / standard_error
            )

            p_value = (
                _normal_two_sided_p_value(
                    test_statistic
                )
            )

            effect_size = (
                mean_difference
                / standard_deviation
            )

        critical_value = (
            _normal_critical_value(
                self.config.confidence_level
            )
        )

        margin = (
            critical_value
            * standard_error
        )

        confidence_interval = (
            ConfidenceInterval(
                lower=sample_mean - margin,
                upper=sample_mean + margin,
                confidence_level=(
                    self.config.confidence_level
                ),
                estimate=sample_mean,
                method=(
                    "normal_mean_interval"
                ),
            )
        )

        significance_level = (
            self.classify_significance(
                p_value
            )
        )

        statistically_significant = (
            p_value
            <= self.config.significance_threshold
        )

        direction_positive = (
            mean_difference > 0.0
        )

        return SignificanceDiagnostics(
            sample_size=sample_size,
            sample_mean=sample_mean,
            null_mean=self.null_mean,
            standard_deviation=(
                standard_deviation
            ),
            standard_error=standard_error,
            test_statistic=test_statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(
                confidence_interval
            ),
            significance_level=(
                significance_level
            ),
            statistically_significant=(
                statistically_significant
            ),
            direction_positive=(
                direction_positive
            ),
            metadata={
                "version": VERSION,
                "test": (
                    "normal_approximation_mean_test"
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # StatisticalTestResult adapter
    # ========================================================

    def to_statistical_test(
        self,
        diagnostics: SignificanceDiagnostics,
    ) -> StatisticalTestResult:
        return StatisticalTestResult(
            test_name=(
                "Mean vs Null Significance Test"
            ),
            statistic=(
                diagnostics.test_statistic
            ),
            p_value=diagnostics.p_value,
            significance_threshold=(
                self.config.significance_threshold
            ),
            effect_size=(
                diagnostics.effect_size
            ),
            null_hypothesis=(
                f"population mean = "
                f"{self.null_mean}"
            ),
            alternative_hypothesis=(
                f"population mean != "
                f"{self.null_mean}"
            ),
            metadata={
                "version": VERSION,
                "sample_size": (
                    diagnostics.sample_size
                ),
                "significance_level": (
                    diagnostics
                    .significance_level
                    .value
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        diagnostics: SignificanceDiagnostics,
    ) -> SignificanceReport:
        statistical_test = (
            self.to_statistical_test(
                diagnostics
            )
        )

        effect_size_sufficient = (
            abs(diagnostics.effect_size)
            >= self.minimum_effect_size
        )

        passed = (
            diagnostics.statistically_significant
            and diagnostics.direction_positive
            and effect_size_sufficient
        )

        return SignificanceReport(
            diagnostics=diagnostics,
            statistical_test=statistical_test,
            minimum_effect_size=(
                self.minimum_effect_size
            ),
            effect_size_sufficient=(
                effect_size_sufficient
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "significance_threshold": (
                    self.config
                    .significance_threshold
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: SignificanceReport,
    ) -> float:
        diagnostics = report.diagnostics

        threshold = (
            self.config.significance_threshold
        )

        significance_component = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    diagnostics.p_value
                    / threshold
                ),
            ),
        )

        if self.minimum_effect_size <= 0.0:
            effect_component = (
                1.0
                if diagnostics.effect_size >= 0.0
                else 0.0
            )
        else:
            effect_component = max(
                0.0,
                min(
                    1.0,
                    diagnostics.effect_size
                    / self.minimum_effect_size,
                ),
            )

        direction_component = (
            1.0
            if diagnostics.direction_positive
            else 0.0
        )

        interval_component = (
            1.0
            if (
                diagnostics
                .confidence_interval
                .lower
                > self.null_mean
            )
            else 0.0
        )

        score = (
            0.40 * significance_component
            + 0.30 * effect_component
            + 0.15 * direction_component
            + 0.15 * interval_component
        )

        return max(
            0.0,
            min(1.0, float(score)),
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: SignificanceReport,
    ) -> ValidationResult:
        diagnostics = report.diagnostics

        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Statistical significance validation passed."
            if report.passed
            else (
                "Statistical significance validation "
                "found insufficient positive evidence."
            )
        )

        return ValidationResult(
            stage=(
                ValidationStage
                .STATISTICAL_SIGNIFICANCE
            ),
            status=status,
            score=self._calculate_score(
                report
            ),
            message=message,
            confidence_interval=(
                diagnostics.confidence_interval
            ),
            statistical_tests=[
                report.statistical_test
            ],
            diagnostics={
                "sample_size": (
                    diagnostics.sample_size
                ),
                "sample_mean": (
                    diagnostics.sample_mean
                ),
                "null_mean": (
                    diagnostics.null_mean
                ),
                "mean_difference": (
                    diagnostics.mean_difference
                ),
                "standard_deviation": (
                    diagnostics.standard_deviation
                ),
                "standard_error": (
                    diagnostics.standard_error
                ),
                "test_statistic": (
                    diagnostics.test_statistic
                ),
                "p_value": (
                    diagnostics.p_value
                ),
                "effect_size": (
                    diagnostics.effect_size
                ),
                "significance_level": (
                    diagnostics
                    .significance_level
                    .value
                ),
                "statistically_significant": (
                    diagnostics
                    .statistically_significant
                ),
                "effect_size_sufficient": (
                    report
                    .effect_size_sufficient
                ),
                "diagnostics_fingerprint": (
                    diagnostics.fingerprint
                ),
                "report_fingerprint": (
                    report.fingerprint
                ),
            },
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Convenience API
    # ========================================================

    def validate(
        self,
        values: Sequence[float],
    ) -> SignificanceReport:
        diagnostics = self.analyze(
            values
        )

        return self.evaluate(
            diagnostics
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
            "null_mean": self.null_mean,
            "minimum_effect_size": (
                self.minimum_effect_size
            ),
            "confidence_level": (
                self.config.confidence_level
            ),
            "significance_threshold": (
                self.config
                .significance_threshold
            ),
            "minimum_observations": (
                self.config
                .minimum_observations
            ),
            "test_method": (
                "normal_approximation_mean_test"
            ),
            "research_only": True,
            "real_money_execution": False,
        }