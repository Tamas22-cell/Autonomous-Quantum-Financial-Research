from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from research_benchmarking.models import (
    RegimeType,
    RegimeValidationResult,
    ValidationConfig,
    ValidationConfigurationError,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    stable_hash,
)


VERSION = "v6.0.2"


class RegimeStabilityError(Exception):
    """Raised when regime-stability validation cannot be performed."""


# ============================================================
# Validation helpers
# ============================================================


def _validate_numeric(
    value: float,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationConfigurationError(
            f"{name} must be numeric."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValidationConfigurationError(
            f"{name} must be finite."
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
# Regime observation
# ============================================================


@dataclass(frozen=True)
class RegimeObservation:
    regime: RegimeType
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.regime, RegimeType):
            raise ValidationConfigurationError(
                "regime must be a RegimeType."
            )

        _validate_numeric(
            self.score,
            "score",
        )

    @property
    def positive(self) -> bool:
        return self.score > 0.0

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "regime": self.regime.value,
                "score": self.score,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Per-regime statistics
# ============================================================


@dataclass(frozen=True)
class RegimeStatistics:
    regime: RegimeType
    observation_count: int
    average_score: float
    minimum_score: float
    maximum_score: float
    volatility: float
    positive_ratio: float
    stability_score: float
    sufficient_sample: bool
    stable: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.regime, RegimeType):
            raise ValidationConfigurationError(
                "regime must be a RegimeType."
            )

        _validate_positive_integer(
            self.observation_count,
            "observation_count",
        )

        for name, value in {
            "average_score": self.average_score,
            "minimum_score": self.minimum_score,
            "maximum_score": self.maximum_score,
            "volatility": self.volatility,
        }.items():
            _validate_numeric(value, name)

        if self.volatility < 0.0:
            raise ValidationConfigurationError(
                "volatility must be non-negative."
            )

        _validate_ratio(
            self.positive_ratio,
            "positive_ratio",
        )

        _validate_ratio(
            self.stability_score,
            "stability_score",
        )

        if not isinstance(self.sufficient_sample, bool):
            raise ValidationConfigurationError(
                "sufficient_sample must be bool."
            )

        if not isinstance(self.stable, bool):
            raise ValidationConfigurationError(
                "stable must be bool."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "regime": self.regime.value,
                "observation_count": self.observation_count,
                "average_score": self.average_score,
                "minimum_score": self.minimum_score,
                "maximum_score": self.maximum_score,
                "volatility": self.volatility,
                "positive_ratio": self.positive_ratio,
                "stability_score": self.stability_score,
                "sufficient_sample": self.sufficient_sample,
                "stable": self.stable,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Aggregate report
# ============================================================


@dataclass(frozen=True)
class RegimeStabilityReport:
    regime_results: List[RegimeStatistics]
    evaluated_regime_count: int
    sufficient_regime_count: int
    stable_regime_count: int
    positive_regime_ratio: float
    stable_regime_ratio: float
    average_stability_score: float
    cross_regime_dispersion: float
    weakest_regime: RegimeType
    strongest_regime: RegimeType
    passed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regime_results:
            raise ValidationConfigurationError(
                "regime_results cannot be empty."
            )

        for result in self.regime_results:
            if not isinstance(result, RegimeStatistics):
                raise ValidationConfigurationError(
                    "regime_results must contain "
                    "RegimeStatistics objects."
                )

        for name, value in {
            "evaluated_regime_count": self.evaluated_regime_count,
            "sufficient_regime_count": self.sufficient_regime_count,
            "stable_regime_count": self.stable_regime_count,
        }.items():
            _validate_positive_integer(value, name)

        if self.sufficient_regime_count > self.evaluated_regime_count:
            raise ValidationConfigurationError(
                "sufficient_regime_count cannot exceed "
                "evaluated_regime_count."
            )

        if self.stable_regime_count > self.evaluated_regime_count:
            raise ValidationConfigurationError(
                "stable_regime_count cannot exceed "
                "evaluated_regime_count."
            )

        _validate_ratio(
            self.positive_regime_ratio,
            "positive_regime_ratio",
        )

        _validate_ratio(
            self.stable_regime_ratio,
            "stable_regime_ratio",
        )

        _validate_ratio(
            self.average_stability_score,
            "average_stability_score",
        )

        _validate_numeric(
            self.cross_regime_dispersion,
            "cross_regime_dispersion",
        )

        if self.cross_regime_dispersion < 0.0:
            raise ValidationConfigurationError(
                "cross_regime_dispersion must be non-negative."
            )

        if not isinstance(self.weakest_regime, RegimeType):
            raise ValidationConfigurationError(
                "weakest_regime must be a RegimeType."
            )

        if not isinstance(self.strongest_regime, RegimeType):
            raise ValidationConfigurationError(
                "strongest_regime must be a RegimeType."
            )

        if not isinstance(self.passed, bool):
            raise ValidationConfigurationError(
                "passed must be bool."
            )

    @property
    def unstable_regimes(self) -> List[RegimeType]:
        return [
            result.regime
            for result in self.regime_results
            if not result.stable
        ]

    @property
    def insufficient_regimes(self) -> List[RegimeType]:
        return [
            result.regime
            for result in self.regime_results
            if not result.sufficient_sample
        ]

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "regime_results": [
                    result.fingerprint
                    for result in self.regime_results
                ],
                "evaluated_regime_count": self.evaluated_regime_count,
                "sufficient_regime_count": self.sufficient_regime_count,
                "stable_regime_count": self.stable_regime_count,
                "positive_regime_ratio": self.positive_regime_ratio,
                "stable_regime_ratio": self.stable_regime_ratio,
                "average_stability_score": self.average_stability_score,
                "cross_regime_dispersion": self.cross_regime_dispersion,
                "weakest_regime": self.weakest_regime.value,
                "strongest_regime": self.strongest_regime.value,
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Engine
# ============================================================


class RegimeStabilityEngine:
    """
    Tests whether research alpha survives across market regimes.

    The engine explicitly checks:
    - positive performance by regime
    - minimum observations per regime
    - within-regime volatility
    - cross-regime dispersion
    - stability across bull/bear/volatility/risk regimes
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        minimum_regime_observations: int = 5,
        minimum_positive_ratio: float = 0.60,
        minimum_regime_stability_score: float = 0.60,
        minimum_stable_regime_ratio: float = 0.60,
        maximum_cross_regime_dispersion: float = 1.00,
    ) -> None:
        self.config = config or ValidationConfig()

        self.minimum_regime_observations = _validate_positive_integer(
            minimum_regime_observations,
            "minimum_regime_observations",
        )

        self.minimum_positive_ratio = _validate_ratio(
            minimum_positive_ratio,
            "minimum_positive_ratio",
        )

        self.minimum_regime_stability_score = _validate_ratio(
            minimum_regime_stability_score,
            "minimum_regime_stability_score",
        )

        self.minimum_stable_regime_ratio = _validate_ratio(
            minimum_stable_regime_ratio,
            "minimum_stable_regime_ratio",
        )

        self.maximum_cross_regime_dispersion = _validate_numeric(
            maximum_cross_regime_dispersion,
            "maximum_cross_regime_dispersion",
        )

        if self.maximum_cross_regime_dispersion < 0.0:
            raise ValidationConfigurationError(
                "maximum_cross_regime_dispersion "
                "must be non-negative."
            )

    # ========================================================
    # Statistics
    # ========================================================

    @staticmethod
    def _sample_std(values: Sequence[float]) -> float:
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)

        variance = (
            sum(
                (value - mean) ** 2
                for value in values
            )
            / (len(values) - 1)
        )

        return math.sqrt(variance)

    # ========================================================
    # Single regime
    # ========================================================

    def analyze_regime(
        self,
        regime: RegimeType,
        scores: Sequence[float],
    ) -> RegimeStatistics:
        if not isinstance(regime, RegimeType):
            raise ValidationConfigurationError(
                "regime must be a RegimeType."
            )

        if not scores:
            raise RegimeStabilityError(
                "scores cannot be empty."
            )

        values = [
            _validate_numeric(
                score,
                f"scores[{index}]",
            )
            for index, score in enumerate(scores)
        ]

        count = len(values)
        average_score = sum(values) / count
        minimum_score = min(values)
        maximum_score = max(values)
        volatility = self._sample_std(values)

        positive_ratio = (
            sum(
                1
                for value in values
                if value > 0.0
            )
            / count
        )

        sufficient_sample = (
            count >= self.minimum_regime_observations
        )

        scale = max(
            abs(average_score),
            1e-12,
        )

        relative_volatility = volatility / scale

        consistency_component = max(
            0.0,
            min(
                1.0,
                1.0 - relative_volatility,
            ),
        )

        direction_component = positive_ratio

        average_component = (
            1.0
            if average_score > 0.0
            else 0.0
        )

        sample_component = (
            1.0
            if sufficient_sample
            else min(
                1.0,
                count
                / self.minimum_regime_observations,
            )
        )

        stability_score = (
            0.40 * direction_component
            + 0.30 * consistency_component
            + 0.20 * average_component
            + 0.10 * sample_component
        )

        stability_score = max(
            0.0,
            min(
                1.0,
                float(stability_score),
            ),
        )

        stable = (
            sufficient_sample
            and average_score > 0.0
            and positive_ratio >= self.minimum_positive_ratio
            and stability_score
            >= self.minimum_regime_stability_score
        )

        return RegimeStatistics(
            regime=regime,
            observation_count=count,
            average_score=average_score,
            minimum_score=minimum_score,
            maximum_score=maximum_score,
            volatility=volatility,
            positive_ratio=positive_ratio,
            stability_score=stability_score,
            sufficient_sample=sufficient_sample,
            stable=stable,
            metadata={
                "version": VERSION,
                "relative_volatility": relative_volatility,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Aggregate
    # ========================================================

    def aggregate(
        self,
        results: Sequence[RegimeStatistics],
    ) -> RegimeStabilityReport:
        if not results:
            raise RegimeStabilityError(
                "results cannot be empty."
            )

        validated: List[RegimeStatistics] = []

        for result in results:
            if not isinstance(result, RegimeStatistics):
                raise ValidationConfigurationError(
                    "results must contain "
                    "RegimeStatistics objects."
                )

            validated.append(result)

        evaluated_count = len(validated)

        sufficient_count = sum(
            1
            for result in validated
            if result.sufficient_sample
        )

        stable_count = sum(
            1
            for result in validated
            if result.stable
        )

        positive_count = sum(
            1
            for result in validated
            if result.average_score > 0.0
        )

        positive_regime_ratio = (
            positive_count / evaluated_count
        )

        stable_regime_ratio = (
            stable_count / evaluated_count
        )

        average_stability_score = (
            sum(
                result.stability_score
                for result in validated
            )
            / evaluated_count
        )

        regime_means = [
            result.average_score
            for result in validated
        ]

        cross_regime_dispersion = self._sample_std(
            regime_means
        )

        weakest = min(
            validated,
            key=lambda result: result.average_score,
        )

        strongest = max(
            validated,
            key=lambda result: result.average_score,
        )

        passed = (
            sufficient_count == evaluated_count
            and positive_regime_ratio
            >= self.minimum_positive_ratio
            and stable_regime_ratio
            >= self.minimum_stable_regime_ratio
            and average_stability_score
            >= self.minimum_regime_stability_score
            and cross_regime_dispersion
            <= self.maximum_cross_regime_dispersion
        )

        return RegimeStabilityReport(
            regime_results=validated,
            evaluated_regime_count=evaluated_count,
            sufficient_regime_count=sufficient_count,
            stable_regime_count=stable_count,
            positive_regime_ratio=positive_regime_ratio,
            stable_regime_ratio=stable_regime_ratio,
            average_stability_score=average_stability_score,
            cross_regime_dispersion=cross_regime_dispersion,
            weakest_regime=weakest.regime,
            strongest_regime=strongest.regime,
            passed=passed,
            metadata={
                "version": VERSION,
                "minimum_regime_observations": (
                    self.minimum_regime_observations
                ),
                "minimum_positive_ratio": (
                    self.minimum_positive_ratio
                ),
                "minimum_regime_stability_score": (
                    self.minimum_regime_stability_score
                ),
                "minimum_stable_regime_ratio": (
                    self.minimum_stable_regime_ratio
                ),
                "maximum_cross_regime_dispersion": (
                    self.maximum_cross_regime_dispersion
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Mapping API
    # ========================================================

    def validate_regimes(
        self,
        regime_scores: Mapping[
            RegimeType,
            Sequence[float],
        ],
    ) -> RegimeStabilityReport:
        if not regime_scores:
            raise RegimeStabilityError(
                "regime_scores cannot be empty."
            )

        results = [
            self.analyze_regime(
                regime,
                scores,
            )
            for regime, scores
            in regime_scores.items()
        ]

        return self.aggregate(results)

    # ========================================================
    # Observation API
    # ========================================================

    def validate_observations(
        self,
        observations: Sequence[
            RegimeObservation
        ],
    ) -> RegimeStabilityReport:
        if not observations:
            raise RegimeStabilityError(
                "observations cannot be empty."
            )

        grouped: Dict[
            RegimeType,
            List[float],
        ] = {}

        for observation in observations:
            if not isinstance(
                observation,
                RegimeObservation,
            ):
                raise ValidationConfigurationError(
                    "observations must contain "
                    "RegimeObservation objects."
                )

            grouped.setdefault(
                observation.regime,
                [],
            ).append(
                observation.score
            )

        return self.validate_regimes(
            grouped
        )

    # ========================================================
    # Shared model adapter
    # ========================================================

    def to_regime_validation_results(
        self,
        report: RegimeStabilityReport,
    ) -> List[RegimeValidationResult]:
        """
        Best-effort adapter to the shared v6.0.2
        RegimeValidationResult model.

        The calculation layer does not depend on the shared
        model constructor remaining unchanged.
        """

        converted: List[
            RegimeValidationResult
        ] = []

        for result in report.regime_results:
            try:
                converted.append(
                    RegimeValidationResult(
                        regime=result.regime,
                        observations=(
                            result.observation_count
                        ),
                        score=result.average_score,
                        stability_score=(
                            result.stability_score
                        ),
                        passed=result.stable,
                        metadata={
                            "version": VERSION,
                            "positive_ratio": (
                                result.positive_ratio
                            ),
                            "research_only": True,
                            "real_money_execution": False,
                        },
                    )
                )
            except TypeError:
                break

        return converted

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: RegimeStabilityReport,
    ) -> float:
        dispersion_component = max(
            0.0,
            min(
                1.0,
                1.0
                - (
                    report.cross_regime_dispersion
                    / max(
                        self.maximum_cross_regime_dispersion,
                        1e-12,
                    )
                ),
            ),
        )

        sufficient_component = (
            report.sufficient_regime_count
            / report.evaluated_regime_count
        )

        score = (
            0.30 * report.stable_regime_ratio
            + 0.25 * report.positive_regime_ratio
            + 0.25 * report.average_stability_score
            + 0.10 * dispersion_component
            + 0.10 * sufficient_component
        )

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: RegimeStabilityReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Research alpha remained stable across "
            "evaluated market regimes."
            if report.passed
            else (
                "Research alpha showed material "
                "regime dependence or insufficient "
                "regime evidence."
            )
        )

        return ValidationResult(
            stage=ValidationStage.REGIME_STABILITY,
            status=status,
            score=self._calculate_score(report),
            message=message,
            diagnostics={
                "evaluated_regime_count": (
                    report.evaluated_regime_count
                ),
                "sufficient_regime_count": (
                    report.sufficient_regime_count
                ),
                "stable_regime_count": (
                    report.stable_regime_count
                ),
                "positive_regime_ratio": (
                    report.positive_regime_ratio
                ),
                "stable_regime_ratio": (
                    report.stable_regime_ratio
                ),
                "average_stability_score": (
                    report.average_stability_score
                ),
                "cross_regime_dispersion": (
                    report.cross_regime_dispersion
                ),
                "weakest_regime": (
                    report.weakest_regime.value
                ),
                "strongest_regime": (
                    report.strongest_regime.value
                ),
                "unstable_regimes": [
                    regime.value
                    for regime
                    in report.unstable_regimes
                ],
                "insufficient_regimes": [
                    regime.value
                    for regime
                    in report.insufficient_regimes
                ],
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
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "minimum_regime_observations": (
                self.minimum_regime_observations
            ),
            "minimum_positive_ratio": (
                self.minimum_positive_ratio
            ),
            "minimum_regime_stability_score": (
                self.minimum_regime_stability_score
            ),
            "minimum_stable_regime_ratio": (
                self.minimum_stable_regime_ratio
            ),
            "maximum_cross_regime_dispersion": (
                self.maximum_cross_regime_dispersion
            ),
            "supported_regimes": [
                regime.value
                for regime in RegimeType
            ],
            "research_only": True,
            "real_money_execution": False,
        }