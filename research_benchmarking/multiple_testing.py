from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from research_benchmarking.models import (
    MultipleTestingMethod,
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


class MultipleTestingError(Exception):
    """Raised when multiple-testing correction cannot run."""


# ============================================================
# Helpers
# ============================================================


def _validate_p_values(
    p_values: Sequence[float],
) -> List[float]:
    if not p_values:
        raise MultipleTestingError(
            "p_values cannot be empty."
        )

    result: List[float] = []

    for value in p_values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValidationConfigurationError(
                "p-values must be numeric."
            )

        numeric = float(value)

        if (
            not math.isfinite(numeric)
            or not 0.0 <= numeric <= 1.0
        ):
            raise ValidationConfigurationError(
                "p-values must be finite and "
                "between 0 and 1."
            )

        result.append(numeric)

    return result


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class CorrectedHypothesis:
    index: int
    original_p_value: float
    adjusted_p_value: float
    rejected: bool
    method: MultipleTestingMethod
    significance_threshold: float
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValidationConfigurationError(
                "index must be a non-negative integer."
            )

        for name, value in {
            "original_p_value": self.original_p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "significance_threshold": (
                self.significance_threshold
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

        if not 0.0 <= self.original_p_value <= 1.0:
            raise ValidationConfigurationError(
                "original_p_value must be between 0 and 1."
            )

        if not 0.0 <= self.adjusted_p_value <= 1.0:
            raise ValidationConfigurationError(
                "adjusted_p_value must be between 0 and 1."
            )

        if not 0.0 < self.significance_threshold < 1.0:
            raise ValidationConfigurationError(
                "significance_threshold must be "
                "between 0 and 1."
            )

        if not isinstance(
            self.method,
            MultipleTestingMethod,
        ):
            raise ValidationConfigurationError(
                "method must be a MultipleTestingMethod."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "index": self.index,
                "original_p_value": (
                    self.original_p_value
                ),
                "adjusted_p_value": (
                    self.adjusted_p_value
                ),
                "rejected": self.rejected,
                "method": self.method.value,
                "significance_threshold": (
                    self.significance_threshold
                ),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class MultipleTestingReport:
    method: MultipleTestingMethod
    significance_threshold: float
    hypotheses: List[CorrectedHypothesis]
    rejected_count: int
    retained_count: int
    discovery_rate: float
    passed: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.method,
            MultipleTestingMethod,
        ):
            raise ValidationConfigurationError(
                "method must be a MultipleTestingMethod."
            )

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

        if not self.hypotheses:
            raise ValidationConfigurationError(
                "hypotheses cannot be empty."
            )

        for name, value in {
            "rejected_count": self.rejected_count,
            "retained_count": self.retained_count,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValidationConfigurationError(
                    f"{name} must be a non-negative integer."
                )

        if (
            self.rejected_count
            + self.retained_count
            != len(self.hypotheses)
        ):
            raise ValidationConfigurationError(
                "Hypothesis counts are inconsistent."
            )

        if (
            isinstance(self.discovery_rate, bool)
            or not isinstance(
                self.discovery_rate,
                (int, float),
            )
            or not math.isfinite(
                float(self.discovery_rate)
            )
            or not 0.0 <= self.discovery_rate <= 1.0
        ):
            raise ValidationConfigurationError(
                "discovery_rate must be between 0 and 1."
            )

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)

    @property
    def rejected_indices(self) -> List[int]:
        return [
            item.index
            for item in self.hypotheses
            if item.rejected
        ]

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "method": self.method.value,
                "significance_threshold": (
                    self.significance_threshold
                ),
                "hypotheses": [
                    item.fingerprint
                    for item in self.hypotheses
                ],
                "rejected_count": (
                    self.rejected_count
                ),
                "retained_count": (
                    self.retained_count
                ),
                "discovery_rate": (
                    self.discovery_rate
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Engine
# ============================================================


class MultipleTestingEngine:
    """
    Multiple-hypothesis correction engine.

    Supported methods:
    - NONE
    - BONFERRONI
    - HOLM
    - BENJAMINI_HOCHBERG

    This layer is important for alpha research because testing
    many strategies, agents, parameters or hypotheses increases
    the probability of false discoveries.
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        method: Optional[
            MultipleTestingMethod
        ] = None,
        minimum_discoveries: int = 1,
    ) -> None:
        self.config = config or ValidationConfig()

        self.method = (
            method
            if method is not None
            else self.config.multiple_testing_method
        )

        if not isinstance(
            self.method,
            MultipleTestingMethod,
        ):
            raise ValidationConfigurationError(
                "method must be a MultipleTestingMethod."
            )

        if (
            isinstance(minimum_discoveries, bool)
            or not isinstance(
                minimum_discoveries,
                int,
            )
            or minimum_discoveries < 0
        ):
            raise ValidationConfigurationError(
                "minimum_discoveries must be a "
                "non-negative integer."
            )

        self.minimum_discoveries = (
            minimum_discoveries
        )

    # ========================================================
    # Correction algorithms
    # ========================================================

    @staticmethod
    def bonferroni(
        p_values: Sequence[float],
    ) -> List[float]:
        values = _validate_p_values(
            p_values
        )

        count = len(values)

        return [
            min(
                1.0,
                value * count,
            )
            for value in values
        ]

    @staticmethod
    def holm(
        p_values: Sequence[float],
    ) -> List[float]:
        values = _validate_p_values(
            p_values
        )

        count = len(values)

        ordered = sorted(
            enumerate(values),
            key=lambda item: item[1],
        )

        adjusted_sorted: List[float] = [
            0.0
        ] * count

        running_max = 0.0

        for rank, (_, value) in enumerate(
            ordered
        ):
            multiplier = count - rank

            adjusted = min(
                1.0,
                value * multiplier,
            )

            running_max = max(
                running_max,
                adjusted,
            )

            adjusted_sorted[rank] = (
                running_max
            )

        adjusted_original = [
            0.0
        ] * count

        for rank, (
            original_index,
            _,
        ) in enumerate(ordered):
            adjusted_original[
                original_index
            ] = adjusted_sorted[rank]

        return adjusted_original

    @staticmethod
    def benjamini_hochberg(
        p_values: Sequence[float],
    ) -> List[float]:
        values = _validate_p_values(
            p_values
        )

        count = len(values)

        ordered = sorted(
            enumerate(values),
            key=lambda item: item[1],
        )

        adjusted_sorted = [
            0.0
        ] * count

        running_min = 1.0

        for reverse_rank in range(
            count - 1,
            -1,
            -1,
        ):
            _, value = ordered[
                reverse_rank
            ]

            rank = reverse_rank + 1

            adjusted = min(
                1.0,
                value * count / rank,
            )

            running_min = min(
                running_min,
                adjusted,
            )

            adjusted_sorted[
                reverse_rank
            ] = running_min

        adjusted_original = [
            0.0
        ] * count

        for rank, (
            original_index,
            _,
        ) in enumerate(ordered):
            adjusted_original[
                original_index
            ] = adjusted_sorted[rank]

        return adjusted_original

    # ========================================================
    # Dispatch
    # ========================================================

    def adjust(
        self,
        p_values: Sequence[float],
    ) -> List[float]:
        values = _validate_p_values(
            p_values
        )

        if (
            self.method
            == MultipleTestingMethod.NONE
        ):
            return list(values)

        if (
            self.method
            == MultipleTestingMethod.BONFERRONI
        ):
            return self.bonferroni(
                values
            )

        if (
            self.method
            == MultipleTestingMethod.HOLM
        ):
            return self.holm(
                values
            )

        if (
            self.method
            == MultipleTestingMethod.BENJAMINI_HOCHBERG
        ):
            return (
                self.benjamini_hochberg(
                    values
                )
            )

        raise MultipleTestingError(
            "Unsupported multiple-testing method."
        )

    # ========================================================
    # Evaluation
    # ========================================================

    def evaluate(
        self,
        p_values: Sequence[float],
    ) -> MultipleTestingReport:
        original = _validate_p_values(
            p_values
        )

        adjusted = self.adjust(
            original
        )

        threshold = (
            self.config.significance_threshold
        )

        hypotheses: List[
            CorrectedHypothesis
        ] = []

        for index, (
            original_p,
            adjusted_p,
        ) in enumerate(
            zip(
                original,
                adjusted,
            )
        ):
            rejected = (
                adjusted_p <= threshold
            )

            hypotheses.append(
                CorrectedHypothesis(
                    index=index,
                    original_p_value=(
                        original_p
                    ),
                    adjusted_p_value=(
                        adjusted_p
                    ),
                    rejected=rejected,
                    method=self.method,
                    significance_threshold=(
                        threshold
                    ),
                    metadata={
                        "version": VERSION,
                    },
                )
            )

        rejected_count = sum(
            1
            for item in hypotheses
            if item.rejected
        )

        retained_count = (
            len(hypotheses)
            - rejected_count
        )

        discovery_rate = (
            rejected_count
            / len(hypotheses)
        )

        passed = (
            rejected_count
            >= self.minimum_discoveries
        )

        return MultipleTestingReport(
            method=self.method,
            significance_threshold=(
                threshold
            ),
            hypotheses=hypotheses,
            rejected_count=(
                rejected_count
            ),
            retained_count=(
                retained_count
            ),
            discovery_rate=(
                discovery_rate
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "minimum_discoveries": (
                    self.minimum_discoveries
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # StatisticalTestResult integration
    # ========================================================

    def correct_statistical_tests(
        self,
        tests: Sequence[
            StatisticalTestResult
        ],
    ) -> List[StatisticalTestResult]:
        if not tests:
            raise MultipleTestingError(
                "tests cannot be empty."
            )

        for test in tests:
            if not isinstance(
                test,
                StatisticalTestResult,
            ):
                raise ValidationConfigurationError(
                    "tests must contain "
                    "StatisticalTestResult objects."
                )

        p_values = [
            test.p_value
            for test in tests
        ]

        adjusted = self.adjust(
            p_values
        )

        corrected: List[
            StatisticalTestResult
        ] = []

        for test, adjusted_p in zip(
            tests,
            adjusted,
        ):
            corrected.append(
                StatisticalTestResult(
                    test_name=test.test_name,
                    statistic=test.statistic,
                    p_value=test.p_value,
                    significance_threshold=(
                        test.significance_threshold
                    ),
                    adjusted_p_value=(
                        adjusted_p
                    ),
                    effect_size=(
                        test.effect_size
                    ),
                    null_hypothesis=(
                        test.null_hypothesis
                    ),
                    alternative_hypothesis=(
                        test.alternative_hypothesis
                    ),
                    metadata={
                        **test.metadata,
                        "multiple_testing_method": (
                            self.method.value
                        ),
                        "version": VERSION,
                    },
                )
            )

        return corrected

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: MultipleTestingReport,
    ) -> float:
        if report.hypothesis_count <= 0:
            return 0.0

        discovery_component = (
            report.discovery_rate
        )

        if self.minimum_discoveries <= 0:
            minimum_component = 1.0
        else:
            minimum_component = min(
                1.0,
                report.rejected_count
                / self.minimum_discoveries,
            )

        correction_component = {
            MultipleTestingMethod.NONE: 0.25,
            MultipleTestingMethod.BONFERRONI: 1.0,
            MultipleTestingMethod.HOLM: 1.0,
            MultipleTestingMethod.BENJAMINI_HOCHBERG: 0.90,
        }[report.method]

        score = (
            0.45 * minimum_component
            + 0.35 * discovery_component
            + 0.20 * correction_component
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
        report: MultipleTestingReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Multiple-testing correction retained "
            "statistically significant discoveries."
            if report.passed
            else (
                "Multiple-testing correction removed "
                "the required statistical discoveries."
            )
        )

        return ValidationResult(
            stage=(
                ValidationStage
                .MULTIPLE_TESTING
            ),
            status=status,
            score=self._calculate_score(
                report
            ),
            message=message,
            diagnostics={
                "method": (
                    report.method.value
                ),
                "hypothesis_count": (
                    report.hypothesis_count
                ),
                "rejected_count": (
                    report.rejected_count
                ),
                "retained_count": (
                    report.retained_count
                ),
                "rejected_indices": (
                    report.rejected_indices
                ),
                "discovery_rate": (
                    report.discovery_rate
                ),
                "minimum_discoveries": (
                    self.minimum_discoveries
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
        p_values: Sequence[float],
    ) -> MultipleTestingReport:
        return self.evaluate(
            p_values
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
            "method": self.method.value,
            "significance_threshold": (
                self.config
                .significance_threshold
            ),
            "minimum_discoveries": (
                self.minimum_discoveries
            ),
            "supported_methods": [
                method.value
                for method
                in MultipleTestingMethod
            ],
            "research_only": True,
            "real_money_execution": False,
        }