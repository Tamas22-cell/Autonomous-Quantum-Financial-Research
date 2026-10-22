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


class MonteCarloValidationError(Exception):
    """Raised when Monte Carlo validation cannot be completed."""


# ============================================================
# Helpers
# ============================================================


def _validate_numeric_sequence(
    values: Sequence[float],
) -> List[float]:
    if not values:
        raise MonteCarloValidationError(
            "Monte Carlo input cannot be empty."
        )

    result: List[float] = []

    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValidationConfigurationError(
                "Monte Carlo observations must be numeric."
            )

        numeric = float(value)

        if not math.isfinite(numeric):
            raise ValidationConfigurationError(
                "Monte Carlo observations must be finite."
            )

        result.append(numeric)

    return result


def _percentile(
    values: Sequence[float],
    probability: float,
) -> float:
    if not values:
        raise MonteCarloValidationError(
            "Cannot calculate percentile of empty data."
        )

    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 <= probability <= 1.0
    ):
        raise ValidationConfigurationError(
            "probability must be between 0 and 1."
        )

    ordered = sorted(float(value) for value in values)

    if len(ordered) == 1:
        return ordered[0]

    position = float(probability) * (
        len(ordered) - 1
    )

    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))

    if lower_index == upper_index:
        return ordered[lower_index]

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
class MonteCarloDistribution:
    observed_estimate: float
    simulated_estimates: List[float]

    null_mean: float
    null_standard_deviation: float

    empirical_p_value: float
    probability_exceeding_observed: float

    confidence_interval: ConfidenceInterval

    method: str
    sample_size: int
    simulations: int
    random_seed: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "observed_estimate": self.observed_estimate,
            "null_mean": self.null_mean,
            "null_standard_deviation": (
                self.null_standard_deviation
            ),
            "empirical_p_value": self.empirical_p_value,
            "probability_exceeding_observed": (
                self.probability_exceeding_observed
            ),
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationConfigurationError(
                    f"{name} must be finite numeric."
                )

        if self.null_standard_deviation < 0.0:
            raise ValidationConfigurationError(
                "null_standard_deviation cannot be negative."
            )

        for name, value in {
            "empirical_p_value": self.empirical_p_value,
            "probability_exceeding_observed": (
                self.probability_exceeding_observed
            ),
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValidationConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        if not self.simulated_estimates:
            raise ValidationConfigurationError(
                "simulated_estimates cannot be empty."
            )

        for value in self.simulated_estimates:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationConfigurationError(
                    "simulated_estimates must contain "
                    "finite numeric values."
                )

        for name, value in {
            "sample_size": self.sample_size,
            "simulations": self.simulations,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be a positive integer."
                )

        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValidationConfigurationError(
                "random_seed must be an integer."
            )

        if not self.method.strip():
            raise ValidationConfigurationError(
                "method cannot be empty."
            )

    @property
    def significant(self) -> bool:
        return self.empirical_p_value < 0.05

    @property
    def effect_above_null(self) -> float:
        return (
            self.observed_estimate
            - self.null_mean
        )

    @property
    def z_score(self) -> float:
        if self.null_standard_deviation <= 1e-12:
            if self.effect_above_null > 0.0:
                return 1e12

            if self.effect_above_null < 0.0:
                return -1e12

            return 0.0

        return (
            self.effect_above_null
            / self.null_standard_deviation
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "observed_estimate": self.observed_estimate,
                "simulated_estimates": self.simulated_estimates,
                "null_mean": self.null_mean,
                "null_standard_deviation": (
                    self.null_standard_deviation
                ),
                "empirical_p_value": self.empirical_p_value,
                "probability_exceeding_observed": (
                    self.probability_exceeding_observed
                ),
                "confidence_interval": (
                    self.confidence_interval.fingerprint
                ),
                "method": self.method,
                "sample_size": self.sample_size,
                "simulations": self.simulations,
                "random_seed": self.random_seed,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class MonteCarloReport:
    distribution: MonteCarloDistribution

    significance_threshold: float
    minimum_z_score: float

    statistically_significant: bool
    effect_direction_positive: bool
    passed: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.significance_threshold, bool)
            or not isinstance(
                self.significance_threshold,
                (int, float),
            )
            or not math.isfinite(
                float(self.significance_threshold)
            )
            or not 0.0
            < self.significance_threshold
            < 1.0
        ):
            raise ValidationConfigurationError(
                "significance_threshold must be "
                "between 0 and 1."
            )

        if (
            isinstance(self.minimum_z_score, bool)
            or not isinstance(
                self.minimum_z_score,
                (int, float),
            )
            or not math.isfinite(
                float(self.minimum_z_score)
            )
            or self.minimum_z_score < 0.0
        ):
            raise ValidationConfigurationError(
                "minimum_z_score must be "
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
                "significance_threshold": (
                    self.significance_threshold
                ),
                "minimum_z_score": (
                    self.minimum_z_score
                ),
                "statistically_significant": (
                    self.statistically_significant
                ),
                "effect_direction_positive": (
                    self.effect_direction_positive
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Monte Carlo Engine
# ============================================================


class MonteCarloEngine:
    """
    Monte Carlo / permutation-based null validation.

    The engine asks whether the observed research statistic
    is unusually strong relative to randomized null outcomes.

    Supported null models:
    - sign-flip randomization
    - permutation randomization
    - centered bootstrap null
    """

    VERSION = VERSION

    SUPPORTED_METHODS = {
        "sign_flip",
        "permutation",
        "centered_bootstrap",
    }

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        method: str = "sign_flip",
        minimum_z_score: float = 1.0,
    ) -> None:
        self.config = config or ValidationConfig()

        if method not in self.SUPPORTED_METHODS:
            raise ValidationConfigurationError(
                "Unsupported Monte Carlo method."
            )

        if (
            isinstance(minimum_z_score, bool)
            or not isinstance(
                minimum_z_score,
                (int, float),
            )
            or not math.isfinite(
                float(minimum_z_score)
            )
            or minimum_z_score < 0.0
        ):
            raise ValidationConfigurationError(
                "minimum_z_score must be finite "
                "and non-negative."
            )

        self.method = method
        self.minimum_z_score = float(
            minimum_z_score
        )

    # ========================================================
    # Null simulation methods
    # ========================================================

    @staticmethod
    def sign_flip_sample(
        values: Sequence[float],
        rng: random.Random,
    ) -> List[float]:
        clean_values = _validate_numeric_sequence(
            values
        )

        return [
            value
            if rng.random() >= 0.5
            else -value
            for value in clean_values
        ]

    @staticmethod
    def permutation_sample(
        values: Sequence[float],
        rng: random.Random,
    ) -> List[float]:
        clean_values = _validate_numeric_sequence(
            values
        )

        sample = list(clean_values)
        rng.shuffle(sample)

        return sample

    @staticmethod
    def centered_bootstrap_sample(
        values: Sequence[float],
        rng: random.Random,
    ) -> List[float]:
        clean_values = _validate_numeric_sequence(
            values
        )

        sample_mean = mean(clean_values)

        centered = [
            value - sample_mean
            for value in clean_values
        ]

        size = len(centered)

        return [
            centered[
                rng.randrange(size)
            ]
            for _ in range(size)
        ]

    # ========================================================
    # Null sample dispatch
    # ========================================================

    def _generate_null_sample(
        self,
        values: Sequence[float],
        rng: random.Random,
    ) -> List[float]:
        if self.method == "sign_flip":
            return self.sign_flip_sample(
                values,
                rng,
            )

        if self.method == "permutation":
            return self.permutation_sample(
                values,
                rng,
            )

        if self.method == "centered_bootstrap":
            return (
                self.centered_bootstrap_sample(
                    values,
                    rng,
                )
            )

        raise MonteCarloValidationError(
            "Unsupported Monte Carlo method."
        )

    # ========================================================
    # Simulation
    # ========================================================

    def run(
        self,
        values: Sequence[float],
        *,
        statistic: Optional[
            Callable[[Sequence[float]], float]
        ] = None,
    ) -> MonteCarloDistribution:
        clean_values = _validate_numeric_sequence(
            values
        )

        if (
            len(clean_values)
            < self.config.minimum_observations
        ):
            raise MonteCarloValidationError(
                "Insufficient observations for "
                "Monte Carlo validation."
            )

        statistic_function = (
            statistic
            if statistic is not None
            else mean
        )

        observed_estimate = float(
            statistic_function(clean_values)
        )

        if not math.isfinite(observed_estimate):
            raise MonteCarloValidationError(
                "Observed statistic must be finite."
            )

        rng = random.Random(
            self.config.random_seed
        )

        simulated_estimates: List[
            float
        ] = []

        for _ in range(
            self.config.monte_carlo_simulations
        ):
            null_sample = (
                self._generate_null_sample(
                    clean_values,
                    rng,
                )
            )

            estimate = float(
                statistic_function(
                    null_sample
                )
            )

            if not math.isfinite(estimate):
                raise MonteCarloValidationError(
                    "Monte Carlo statistic produced "
                    "a non-finite result."
                )

            simulated_estimates.append(
                estimate
            )

        null_mean = mean(
            simulated_estimates
        )

        if len(simulated_estimates) > 1:
            squared_deviations = [
                (
                    estimate
                    - null_mean
                )
                ** 2
                for estimate
                in simulated_estimates
            ]

            null_standard_deviation = math.sqrt(
                sum(squared_deviations)
                / (
                    len(
                        simulated_estimates
                    )
                    - 1
                )
            )
        else:
            null_standard_deviation = 0.0

        exceed_count = sum(
            1
            for estimate
            in simulated_estimates
            if estimate >= observed_estimate
        )

        # +1 correction prevents an impossible p-value of 0.
        empirical_p_value = (
            exceed_count + 1
        ) / (
            len(simulated_estimates) + 1
        )

        probability_exceeding_observed = (
            exceed_count
            / len(simulated_estimates)
        )

        alpha = (
            1.0
            - self.config.confidence_level
        )

        lower = _percentile(
            simulated_estimates,
            alpha / 2.0,
        )

        upper = _percentile(
            simulated_estimates,
            1.0 - alpha / 2.0,
        )

        confidence_interval = ConfidenceInterval(
            lower=lower,
            upper=upper,
            confidence_level=(
                self.config.confidence_level
            ),
            estimate=null_mean,
            method=(
                f"monte_carlo_{self.method}"
            ),
        )

        return MonteCarloDistribution(
            observed_estimate=(
                observed_estimate
            ),
            simulated_estimates=(
                simulated_estimates
            ),
            null_mean=null_mean,
            null_standard_deviation=(
                null_standard_deviation
            ),
            empirical_p_value=(
                empirical_p_value
            ),
            probability_exceeding_observed=(
                probability_exceeding_observed
            ),
            confidence_interval=(
                confidence_interval
            ),
            method=self.method,
            sample_size=len(clean_values),
            simulations=(
                self.config.monte_carlo_simulations
            ),
            random_seed=(
                self.config.random_seed
            ),
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        distribution: MonteCarloDistribution,
    ) -> MonteCarloReport:
        statistically_significant = (
            distribution.empirical_p_value
            <= self.config.significance_threshold
        )

        effect_direction_positive = (
            distribution.effect_above_null
            > 0.0
        )

        passed = (
            statistically_significant
            and effect_direction_positive
            and distribution.z_score
            >= self.minimum_z_score
        )

        return MonteCarloReport(
            distribution=distribution,
            significance_threshold=(
                self.config.significance_threshold
            ),
            minimum_z_score=(
                self.minimum_z_score
            ),
            statistically_significant=(
                statistically_significant
            ),
            effect_direction_positive=(
                effect_direction_positive
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: MonteCarloReport,
    ) -> float:
        distribution = (
            report.distribution
        )

        threshold = (
            report.significance_threshold
        )

        if threshold <= 0.0:
            significance_component = 0.0
        else:
            significance_component = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        distribution.empirical_p_value
                        / threshold
                    ),
                ),
            )

        if self.minimum_z_score <= 0.0:
            z_component = (
                1.0
                if distribution.z_score >= 0.0
                else 0.0
            )
        else:
            z_component = max(
                0.0,
                min(
                    1.0,
                    distribution.z_score
                    / self.minimum_z_score,
                ),
            )

        direction_component = (
            1.0
            if report.effect_direction_positive
            else 0.0
        )

        score = (
            0.50 * significance_component
            + 0.30 * z_component
            + 0.20 * direction_component
        )

        return min(
            1.0,
            max(
                0.0,
                float(score),
            ),
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: MonteCarloReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Monte Carlo validation passed."
            if report.passed
            else (
                "Monte Carlo validation found "
                "insufficient evidence against "
                "the randomized null model."
            )
        )

        distribution = (
            report.distribution
        )

        return ValidationResult(
            stage=ValidationStage.MONTE_CARLO,
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
                "null_mean": (
                    distribution.null_mean
                ),
                "null_standard_deviation": (
                    distribution.null_standard_deviation
                ),
                "effect_above_null": (
                    distribution.effect_above_null
                ),
                "z_score": (
                    distribution.z_score
                ),
                "empirical_p_value": (
                    distribution.empirical_p_value
                ),
                "probability_exceeding_observed": (
                    distribution
                    .probability_exceeding_observed
                ),
                "statistically_significant": (
                    report.statistically_significant
                ),
                "method": (
                    distribution.method
                ),
                "sample_size": (
                    distribution.sample_size
                ),
                "simulations": (
                    distribution.simulations
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
    ) -> MonteCarloReport:
        distribution = self.run(
            values,
            statistic=statistic,
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
            "method": self.method,
            "monte_carlo_simulations": (
                self.config.monte_carlo_simulations
            ),
            "confidence_level": (
                self.config.confidence_level
            ),
            "significance_threshold": (
                self.config.significance_threshold
            ),
            "minimum_z_score": (
                self.minimum_z_score
            ),
            "random_seed": (
                self.config.random_seed
            ),
            "research_only": True,
            "real_money_execution": False,
        }