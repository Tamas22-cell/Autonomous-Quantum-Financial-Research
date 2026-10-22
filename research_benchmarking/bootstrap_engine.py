from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Sequence

from research_benchmarking.models import (
    ConfidenceInterval,
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


class BootstrapValidationError(Exception):
    """Raised when bootstrap validation cannot be completed."""


# ============================================================
# Helpers
# ============================================================


def _validate_numeric_sequence(
    values: Sequence[float],
) -> List[float]:
    if not values:
        raise BootstrapValidationError(
            "Bootstrap input cannot be empty."
        )

    result: List[float] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValidationConfigurationError(
                "Bootstrap observations must be numeric."
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValidationConfigurationError(
                "Bootstrap observations must be finite."
            )

        result.append(numeric)

    return result


def _percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    if not values:
        raise BootstrapValidationError(
            "Cannot calculate percentile of empty data."
        )

    if not 0.0 <= probability <= 1.0:
        raise ValidationConfigurationError(
            "probability must be between 0 and 1."
        )

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = probability * (len(ordered) - 1)

    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return float(ordered[lower_index])

    weight = position - lower_index

    return float(
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * weight
    )


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class BootstrapDistribution:
    observed_estimate: float
    bootstrap_estimates: List[float]
    standard_error: float
    bias: float
    confidence_interval: ConfidenceInterval
    method: str
    sample_size: int
    bootstrap_samples: int
    block_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "observed_estimate": self.observed_estimate,
            "standard_error": self.standard_error,
            "bias": self.bias,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationConfigurationError(
                    f"{name} must be finite numeric."
                )

        if not self.bootstrap_estimates:
            raise ValidationConfigurationError(
                "bootstrap_estimates cannot be empty."
            )

        for value in self.bootstrap_estimates:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationConfigurationError(
                    "bootstrap_estimates must contain "
                    "finite numeric values."
                )

        for name, value in {
            "sample_size": self.sample_size,
            "bootstrap_samples": self.bootstrap_samples,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be a positive integer."
                )

        if self.block_size is not None:
            if (
                isinstance(self.block_size, bool)
                or not isinstance(self.block_size, int)
                or self.block_size <= 0
            ):
                raise ValidationConfigurationError(
                    "block_size must be a positive integer."
                )

        if not self.method.strip():
            raise ValidationConfigurationError(
                "method cannot be empty."
            )

    @property
    def contains_zero(self) -> bool:
        return self.confidence_interval.contains_zero

    @property
    def positive_interval(self) -> bool:
        return self.confidence_interval.lower > 0.0

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "observed_estimate": self.observed_estimate,
                "bootstrap_estimates": self.bootstrap_estimates,
                "standard_error": self.standard_error,
                "bias": self.bias,
                "confidence_interval": (
                    self.confidence_interval.fingerprint
                ),
                "method": self.method,
                "sample_size": self.sample_size,
                "bootstrap_samples": self.bootstrap_samples,
                "block_size": self.block_size,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class BootstrapReport:
    distribution: BootstrapDistribution
    probability_positive: float
    relative_standard_error: float
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.probability_positive, bool)
            or not isinstance(
                self.probability_positive,
                (int, float),
            )
            or not math.isfinite(
                float(self.probability_positive)
            )
            or not 0.0
            <= self.probability_positive
            <= 1.0
        ):
            raise ValidationConfigurationError(
                "probability_positive must be "
                "between 0 and 1."
            )

        if (
            isinstance(self.relative_standard_error, bool)
            or not isinstance(
                self.relative_standard_error,
                (int, float),
            )
            or not math.isfinite(
                float(self.relative_standard_error)
            )
            or self.relative_standard_error < 0.0
        ):
            raise ValidationConfigurationError(
                "relative_standard_error must be "
                "finite and non-negative."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "distribution": (
                    self.distribution.fingerprint
                ),
                "probability_positive": (
                    self.probability_positive
                ),
                "relative_standard_error": (
                    self.relative_standard_error
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Bootstrap engine
# ============================================================


class BootstrapEngine:
    """
    Bootstrap statistical validation engine.

    Supports:
    - IID bootstrap
    - moving-block bootstrap for time-series data
    - percentile confidence intervals
    - deterministic seeded execution
    - uncertainty diagnostics
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        block_size: Optional[int] = None,
        minimum_probability_positive: float = 0.95,
        maximum_relative_standard_error: float = 1.0,
    ) -> None:
        self.config = config or ValidationConfig()

        if block_size is not None:
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, int)
                or block_size <= 0
            ):
                raise ValidationConfigurationError(
                    "block_size must be a positive integer."
                )

        if (
            isinstance(minimum_probability_positive, bool)
            or not isinstance(
                minimum_probability_positive,
                (int, float),
            )
            or not math.isfinite(
                float(minimum_probability_positive)
            )
            or not 0.0
            <= minimum_probability_positive
            <= 1.0
        ):
            raise ValidationConfigurationError(
                "minimum_probability_positive must be "
                "between 0 and 1."
            )

        if (
            isinstance(maximum_relative_standard_error, bool)
            or not isinstance(
                maximum_relative_standard_error,
                (int, float),
            )
            or not math.isfinite(
                float(maximum_relative_standard_error)
            )
            or maximum_relative_standard_error < 0.0
        ):
            raise ValidationConfigurationError(
                "maximum_relative_standard_error must be "
                "finite and non-negative."
            )

        self.block_size = block_size

        self.minimum_probability_positive = float(
            minimum_probability_positive
        )

        self.maximum_relative_standard_error = float(
            maximum_relative_standard_error
        )

    # ========================================================
    # Resampling
    # ========================================================

    @staticmethod
    def iid_resample(
        values: Sequence[float],
        rng: random.Random,
    ) -> List[float]:
        clean_values = _validate_numeric_sequence(
            values
        )

        size = len(clean_values)

        return [
            clean_values[
                rng.randrange(size)
            ]
            for _ in range(size)
        ]

    @staticmethod
    def moving_block_resample(
        values: Sequence[float],
        block_size: int,
        rng: random.Random,
    ) -> List[float]:
        clean_values = _validate_numeric_sequence(
            values
        )

        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise ValidationConfigurationError(
                "block_size must be a positive integer."
            )

        sample_size = len(clean_values)

        if block_size > sample_size:
            raise BootstrapValidationError(
                "block_size cannot exceed sample size."
            )

        maximum_start = (
            sample_size - block_size
        )

        resampled: List[float] = []

        while len(resampled) < sample_size:
            start = rng.randint(
                0,
                maximum_start,
            )

            block = clean_values[
                start:
                start + block_size
            ]

            resampled.extend(block)

        return resampled[:sample_size]

    # ========================================================
    # Bootstrap execution
    # ========================================================

    def run(
        self,
        values: Sequence[float],
        *,
        statistic: Optional[
            Callable[[Sequence[float]], float]
        ] = None,
        block_size: Optional[int] = None,
    ) -> BootstrapDistribution:
        clean_values = _validate_numeric_sequence(
            values
        )

        if (
            len(clean_values)
            < self.config.minimum_observations
        ):
            raise BootstrapValidationError(
                "Insufficient observations for "
                "bootstrap validation."
            )

        statistic_function = (
            statistic
            if statistic is not None
            else mean
        )

        selected_block_size = (
            block_size
            if block_size is not None
            else self.block_size
        )

        if selected_block_size is not None:
            if (
                isinstance(selected_block_size, bool)
                or not isinstance(
                    selected_block_size,
                    int,
                )
                or selected_block_size <= 0
            ):
                raise ValidationConfigurationError(
                    "block_size must be a positive integer."
                )

            if selected_block_size > len(
                clean_values
            ):
                raise BootstrapValidationError(
                    "block_size cannot exceed sample size."
                )

        observed_estimate = float(
            statistic_function(clean_values)
        )

        if not math.isfinite(observed_estimate):
            raise BootstrapValidationError(
                "Observed statistic must be finite."
            )

        rng = random.Random(
            self.config.random_seed
        )

        bootstrap_estimates: List[
            float
        ] = []

        for _ in range(
            self.config.bootstrap_samples
        ):
            if selected_block_size is None:
                sample = self.iid_resample(
                    clean_values,
                    rng,
                )

                method = "iid_percentile"

            else:
                sample = (
                    self.moving_block_resample(
                        clean_values,
                        selected_block_size,
                        rng,
                    )
                )

                method = (
                    "moving_block_percentile"
                )

            estimate = float(
                statistic_function(sample)
            )

            if not math.isfinite(estimate):
                raise BootstrapValidationError(
                    "Bootstrap statistic produced "
                    "a non-finite result."
                )

            bootstrap_estimates.append(
                estimate
            )

        bootstrap_mean = mean(
            bootstrap_estimates
        )

        if len(bootstrap_estimates) > 1:
            squared_deviations = [
                (
                    estimate
                    - bootstrap_mean
                )
                ** 2
                for estimate
                in bootstrap_estimates
            ]

            standard_error = math.sqrt(
                sum(squared_deviations)
                / (
                    len(
                        bootstrap_estimates
                    )
                    - 1
                )
            )

        else:
            standard_error = 0.0

        bias = (
            bootstrap_mean
            - observed_estimate
        )

        alpha = (
            1.0
            - self.config.confidence_level
        )

        lower = _percentile(
            bootstrap_estimates,
            alpha / 2.0,
        )

        upper = _percentile(
            bootstrap_estimates,
            1.0 - alpha / 2.0,
        )

        confidence_interval = ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=(
                self.config.confidence_level
            ),
            estimate=observed_estimate,
            method=method,
        )

        return BootstrapDistribution(
            observed_estimate=observed_estimate,
            bootstrap_estimates=(
                bootstrap_estimates
            ),
            standard_error=standard_error,
            bias=bias,
            confidence_interval=(
                confidence_interval
            ),
            method=method,
            sample_size=len(clean_values),
            bootstrap_samples=(
                self.config.bootstrap_samples
            ),
            block_size=selected_block_size,
            metadata={
                "version": VERSION,
                "random_seed": (
                    self.config.random_seed
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Report
    # ========================================================

    def evaluate(
        self,
        distribution: BootstrapDistribution,
    ) -> BootstrapReport:
        positive_count = sum(
            1
            for estimate
            in distribution.bootstrap_estimates
            if estimate > 0.0
        )

        probability_positive = (
            positive_count
            / len(
                distribution.bootstrap_estimates
            )
        )

        denominator = abs(
            distribution.observed_estimate
        )

        if denominator <= 1e-12:
            relative_standard_error = (
                0.0
                if distribution.standard_error
                <= 1e-12
                else 1e12
            )
        else:
            relative_standard_error = (
                distribution.standard_error
                / denominator
            )

        passed = (
            distribution.positive_interval
            and probability_positive
            >= self.minimum_probability_positive
            and relative_standard_error
            <= self.maximum_relative_standard_error
        )

        return BootstrapReport(
            distribution=distribution,
            probability_positive=(
                probability_positive
            ),
            relative_standard_error=(
                relative_standard_error
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "minimum_probability_positive": (
                    self.minimum_probability_positive
                ),
                "maximum_relative_standard_error": (
                    self.maximum_relative_standard_error
                ),
            },
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def _calculate_score(
        self,
        report: BootstrapReport,
    ) -> float:
        probability_component = (
            report.probability_positive
        )

        if (
            self.maximum_relative_standard_error
            <= 0.0
        ):
            uncertainty_component = (
                1.0
                if report.relative_standard_error
                <= 0.0
                else 0.0
            )

        else:
            uncertainty_component = max(
                0.0,
                1.0
                - (
                    report.relative_standard_error
                    / self.maximum_relative_standard_error
                ),
            )

        interval_component = (
            1.0
            if report.distribution.positive_interval
            else 0.0
        )

        score = (
            0.45 * probability_component
            + 0.35 * interval_component
            + 0.20 * uncertainty_component
        )

        return min(
            1.0,
            max(
                0.0,
                float(score),
            ),
        )

    def to_validation_result(
        self,
        report: BootstrapReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Bootstrap validation passed."
            if report.passed
            else (
                "Bootstrap validation detected "
                "insufficient statistical robustness."
            )
        )

        distribution = (
            report.distribution
        )

        return ValidationResult(
            stage=ValidationStage.BOOTSTRAP,
            status=status,
            score=self._calculate_score(
                report
            ),
            message=message,
            confidence_interval=(
                distribution.confidence_interval
            ),
            diagnostics={
                "observed_estimate": (
                    distribution.observed_estimate
                ),
                "standard_error": (
                    distribution.standard_error
                ),
                "bias": distribution.bias,
                "probability_positive": (
                    report.probability_positive
                ),
                "relative_standard_error": (
                    report.relative_standard_error
                ),
                "sample_size": (
                    distribution.sample_size
                ),
                "bootstrap_samples": (
                    distribution.bootstrap_samples
                ),
                "method": (
                    distribution.method
                ),
                "block_size": (
                    distribution.block_size
                ),
                "distribution_fingerprint": (
                    distribution.fingerprint
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
        *,
        statistic: Optional[
            Callable[[Sequence[float]], float]
        ] = None,
        block_size: Optional[int] = None,
    ) -> BootstrapReport:
        distribution = self.run(
            values,
            statistic=statistic,
            block_size=block_size,
        )

        return self.evaluate(
            distribution
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
            "bootstrap_samples": (
                self.config.bootstrap_samples
            ),
            "confidence_level": (
                self.config.confidence_level
            ),
            "random_seed": (
                self.config.random_seed
            ),
            "block_size": self.block_size,
            "minimum_probability_positive": (
                self.minimum_probability_positive
            ),
            "maximum_relative_standard_error": (
                self.maximum_relative_standard_error
            ),
            "research_only": True,
            "real_money_execution": False,
        }