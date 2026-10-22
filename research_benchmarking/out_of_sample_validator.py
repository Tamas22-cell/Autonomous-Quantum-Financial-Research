from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from research_benchmarking.models import (
    PerformanceMetrics,
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


class OutOfSampleValidationError(Exception):
    """Raised when out-of-sample validation cannot be completed."""


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class OutOfSampleSegment:
    segment_id: int
    start: int
    end: int
    label: str = "OOS"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValidationConfigurationError(
                    f"{name} must be an integer."
                )

        if self.segment_id < 0:
            raise ValidationConfigurationError(
                "segment_id must be non-negative."
            )

        if self.start < 0:
            raise ValidationConfigurationError(
                "start must be non-negative."
            )

        if self.end <= self.start:
            raise ValidationConfigurationError(
                "end must be greater than start."
            )

        if not self.label.strip():
            raise ValidationConfigurationError(
                "label cannot be empty."
            )

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "segment_id": self.segment_id,
                "start": self.start,
                "end": self.end,
                "label": self.label,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class OutOfSampleSegmentResult:
    segment: OutOfSampleSegment

    in_sample_score: float
    out_of_sample_score: float

    benchmark_score: Optional[float] = None

    in_sample_metrics: Optional[PerformanceMetrics] = None
    out_of_sample_metrics: Optional[PerformanceMetrics] = None
    benchmark_metrics: Optional[PerformanceMetrics] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {
            "in_sample_score": self.in_sample_score,
            "out_of_sample_score": self.out_of_sample_score,
        }

        if self.benchmark_score is not None:
            values["benchmark_score"] = self.benchmark_score

        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValidationConfigurationError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValidationConfigurationError(
                    f"{name} must be finite."
                )

    @property
    def degradation_ratio(self) -> float:
        if math.isclose(
            self.in_sample_score,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return (
            self.in_sample_score
            - self.out_of_sample_score
        ) / abs(self.in_sample_score)

    @property
    def benchmark_excess(self) -> Optional[float]:
        if self.benchmark_score is None:
            return None

        return (
            self.out_of_sample_score
            - self.benchmark_score
        )

    @property
    def positive_oos(self) -> bool:
        return self.out_of_sample_score > 0.0

    @property
    def beats_benchmark(self) -> Optional[bool]:
        if self.benchmark_score is None:
            return None

        return (
            self.out_of_sample_score
            > self.benchmark_score
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "segment": self.segment.fingerprint,
                "in_sample_score": self.in_sample_score,
                "out_of_sample_score": self.out_of_sample_score,
                "benchmark_score": self.benchmark_score,
                "in_sample_metrics": (
                    self.in_sample_metrics.fingerprint
                    if self.in_sample_metrics is not None
                    else None
                ),
                "out_of_sample_metrics": (
                    self.out_of_sample_metrics.fingerprint
                    if self.out_of_sample_metrics is not None
                    else None
                ),
                "benchmark_metrics": (
                    self.benchmark_metrics.fingerprint
                    if self.benchmark_metrics is not None
                    else None
                ),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class OutOfSampleReport:
    segments: List[OutOfSampleSegmentResult]

    average_in_sample_score: float
    average_out_of_sample_score: float
    average_degradation_ratio: float

    positive_oos_ratio: float
    stable_segment_ratio: float

    benchmark_comparison_count: int
    benchmark_win_ratio: Optional[float]

    passed: bool

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValidationConfigurationError(
                "OutOfSampleReport requires at least one segment."
            )

        for name, value in {
            "average_in_sample_score": self.average_in_sample_score,
            "average_out_of_sample_score": self.average_out_of_sample_score,
            "average_degradation_ratio": self.average_degradation_ratio,
            "positive_oos_ratio": self.positive_oos_ratio,
            "stable_segment_ratio": self.stable_segment_ratio,
        }.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValidationConfigurationError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValidationConfigurationError(
                    f"{name} must be finite."
                )

        if not 0.0 <= self.positive_oos_ratio <= 1.0:
            raise ValidationConfigurationError(
                "positive_oos_ratio must be between 0 and 1."
            )

        if not 0.0 <= self.stable_segment_ratio <= 1.0:
            raise ValidationConfigurationError(
                "stable_segment_ratio must be between 0 and 1."
            )

        if (
            isinstance(self.benchmark_comparison_count, bool)
            or not isinstance(self.benchmark_comparison_count, int)
            or self.benchmark_comparison_count < 0
        ):
            raise ValidationConfigurationError(
                "benchmark_comparison_count must be "
                "a non-negative integer."
            )

        if self.benchmark_win_ratio is not None:
            if (
                isinstance(self.benchmark_win_ratio, bool)
                or not isinstance(
                    self.benchmark_win_ratio,
                    (int, float),
                )
                or not math.isfinite(
                    float(self.benchmark_win_ratio)
                )
                or not 0.0 <= self.benchmark_win_ratio <= 1.0
            ):
                raise ValidationConfigurationError(
                    "benchmark_win_ratio must be between 0 and 1."
                )

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "segments": [
                    segment.fingerprint
                    for segment in self.segments
                ],
                "average_in_sample_score": (
                    self.average_in_sample_score
                ),
                "average_out_of_sample_score": (
                    self.average_out_of_sample_score
                ),
                "average_degradation_ratio": (
                    self.average_degradation_ratio
                ),
                "positive_oos_ratio": self.positive_oos_ratio,
                "stable_segment_ratio": self.stable_segment_ratio,
                "benchmark_comparison_count": (
                    self.benchmark_comparison_count
                ),
                "benchmark_win_ratio": self.benchmark_win_ratio,
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Out-of-sample validator
# ============================================================


class OutOfSampleValidator:
    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        maximum_degradation: float = 0.50,
        minimum_positive_oos_ratio: float = 0.60,
        minimum_stable_segment_ratio: float = 0.60,
        minimum_benchmark_win_ratio: float = 0.50,
        require_benchmark_outperformance: bool = False,
    ) -> None:
        self.config = config or ValidationConfig()

        for name, value in {
            "maximum_degradation": maximum_degradation,
            "minimum_positive_oos_ratio": minimum_positive_oos_ratio,
            "minimum_stable_segment_ratio": minimum_stable_segment_ratio,
            "minimum_benchmark_win_ratio": minimum_benchmark_win_ratio,
        }.items():
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValidationConfigurationError(
                    f"{name} must be numeric."
                )

            if not math.isfinite(float(value)):
                raise ValidationConfigurationError(
                    f"{name} must be finite."
                )

        if maximum_degradation < 0.0:
            raise ValidationConfigurationError(
                "maximum_degradation must be non-negative."
            )

        for name, value in {
            "minimum_positive_oos_ratio": minimum_positive_oos_ratio,
            "minimum_stable_segment_ratio": minimum_stable_segment_ratio,
            "minimum_benchmark_win_ratio": minimum_benchmark_win_ratio,
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValidationConfigurationError(
                    f"{name} must be between 0 and 1."
                )

        self.maximum_degradation = float(
            maximum_degradation
        )

        self.minimum_positive_oos_ratio = float(
            minimum_positive_oos_ratio
        )

        self.minimum_stable_segment_ratio = float(
            minimum_stable_segment_ratio
        )

        self.minimum_benchmark_win_ratio = float(
            minimum_benchmark_win_ratio
        )

        self.require_benchmark_outperformance = bool(
            require_benchmark_outperformance
        )

    # ========================================================
    # Segment construction
    # ========================================================

    def build_holdout_segment(
        self,
        observation_count: int,
    ) -> OutOfSampleSegment:
        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
        ):
            raise ValidationConfigurationError(
                "observation_count must be an integer."
            )

        if (
            observation_count
            < self.config.minimum_observations
        ):
            raise OutOfSampleValidationError(
                "Insufficient observations for "
                "out-of-sample validation."
            )

        test_size = max(
            1,
            int(
                observation_count
                * self.config.test_fraction
            ),
        )

        start = observation_count - test_size

        if start <= 0:
            raise OutOfSampleValidationError(
                "Unable to construct an out-of-sample holdout."
            )

        return OutOfSampleSegment(
            segment_id=0,
            start=start,
            end=observation_count,
            label="FINAL_HOLDOUT",
            metadata={
                "test_fraction": self.config.test_fraction,
            },
        )

    def build_segments(
        self,
        observation_count: int,
        segment_count: int,
    ) -> List[OutOfSampleSegment]:
        if (
            isinstance(segment_count, bool)
            or not isinstance(segment_count, int)
            or segment_count <= 0
        ):
            raise ValidationConfigurationError(
                "segment_count must be a positive integer."
            )

        if (
            isinstance(observation_count, bool)
            or not isinstance(observation_count, int)
        ):
            raise ValidationConfigurationError(
                "observation_count must be an integer."
            )

        if (
            observation_count
            < self.config.minimum_observations
        ):
            raise OutOfSampleValidationError(
                "Insufficient observations for "
                "out-of-sample validation."
            )

        total_oos_size = max(
            segment_count,
            int(
                observation_count
                * self.config.test_fraction
            ),
        )

        if total_oos_size >= observation_count:
            raise OutOfSampleValidationError(
                "Out-of-sample region consumes "
                "the complete dataset."
            )

        if total_oos_size < segment_count:
            raise OutOfSampleValidationError(
                "Not enough out-of-sample observations "
                "for the requested segment count."
            )

        oos_start = (
            observation_count
            - total_oos_size
        )

        base_size = (
            total_oos_size
            // segment_count
        )

        remainder = (
            total_oos_size
            % segment_count
        )

        segments: List[
            OutOfSampleSegment
        ] = []

        cursor = oos_start

        for segment_id in range(segment_count):
            extra = (
                1
                if segment_id < remainder
                else 0
            )

            size = base_size + extra
            end = cursor + size

            if segment_id == segment_count - 1:
                end = observation_count

            segments.append(
                OutOfSampleSegment(
                    segment_id=segment_id,
                    start=cursor,
                    end=end,
                    label=f"OOS_{segment_id + 1}",
                )
            )

            cursor = end

        return segments

    # ========================================================
    # Sequence extraction
    # ========================================================

    @staticmethod
    def extract_segment(
        values: Sequence[Any],
        segment: OutOfSampleSegment,
    ) -> Sequence[Any]:
        if len(values) < segment.end:
            raise OutOfSampleValidationError(
                "Input sequence is shorter than "
                "the out-of-sample segment."
            )

        return values[
            segment.start:
            segment.end
        ]

    # ========================================================
    # Result construction
    # ========================================================

    def create_segment_result(
        self,
        *,
        segment: OutOfSampleSegment,
        in_sample_score: float,
        out_of_sample_score: float,
        benchmark_score: Optional[float] = None,
        in_sample_metrics: Optional[
            PerformanceMetrics
        ] = None,
        out_of_sample_metrics: Optional[
            PerformanceMetrics
        ] = None,
        benchmark_metrics: Optional[
            PerformanceMetrics
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> OutOfSampleSegmentResult:
        return OutOfSampleSegmentResult(
            segment=segment,
            in_sample_score=in_sample_score,
            out_of_sample_score=out_of_sample_score,
            benchmark_score=benchmark_score,
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            benchmark_metrics=benchmark_metrics,
            metadata=dict(metadata or {}),
        )

    # ========================================================
    # Aggregation
    # ========================================================

    def aggregate(
        self,
        results: Sequence[
            OutOfSampleSegmentResult
        ],
    ) -> OutOfSampleReport:
        if not results:
            raise OutOfSampleValidationError(
                "At least one out-of-sample result "
                "is required."
            )

        result_list = list(results)
        count = len(result_list)

        average_in_sample_score = (
            sum(
                result.in_sample_score
                for result in result_list
            )
            / count
        )

        average_out_of_sample_score = (
            sum(
                result.out_of_sample_score
                for result in result_list
            )
            / count
        )

        average_degradation_ratio = (
            sum(
                result.degradation_ratio
                for result in result_list
            )
            / count
        )

        positive_count = sum(
            1
            for result in result_list
            if result.positive_oos
        )

        positive_oos_ratio = (
            positive_count / count
        )

        stable_count = sum(
            1
            for result in result_list
            if result.degradation_ratio
            <= self.maximum_degradation
        )

        stable_segment_ratio = (
            stable_count / count
        )

        benchmark_results = [
            result
            for result in result_list
            if result.benchmark_score is not None
        ]

        benchmark_comparison_count = len(
            benchmark_results
        )

        if benchmark_results:
            benchmark_wins = sum(
                1
                for result in benchmark_results
                if result.beats_benchmark is True
            )

            benchmark_win_ratio: Optional[
                float
            ] = (
                benchmark_wins
                / benchmark_comparison_count
            )
        else:
            benchmark_win_ratio = None

        benchmark_requirement_passed = True

        if self.require_benchmark_outperformance:
            benchmark_requirement_passed = (
                benchmark_win_ratio is not None
                and benchmark_win_ratio
                >= self.minimum_benchmark_win_ratio
            )

        passed = (
            average_out_of_sample_score > 0.0
            and positive_oos_ratio
            >= self.minimum_positive_oos_ratio
            and stable_segment_ratio
            >= self.minimum_stable_segment_ratio
            and average_degradation_ratio
            <= self.maximum_degradation
            and benchmark_requirement_passed
        )

        return OutOfSampleReport(
            segments=result_list,
            average_in_sample_score=(
                average_in_sample_score
            ),
            average_out_of_sample_score=(
                average_out_of_sample_score
            ),
            average_degradation_ratio=(
                average_degradation_ratio
            ),
            positive_oos_ratio=(
                positive_oos_ratio
            ),
            stable_segment_ratio=(
                stable_segment_ratio
            ),
            benchmark_comparison_count=(
                benchmark_comparison_count
            ),
            benchmark_win_ratio=(
                benchmark_win_ratio
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "maximum_degradation": (
                    self.maximum_degradation
                ),
                "minimum_positive_oos_ratio": (
                    self.minimum_positive_oos_ratio
                ),
                "minimum_stable_segment_ratio": (
                    self.minimum_stable_segment_ratio
                ),
                "minimum_benchmark_win_ratio": (
                    self.minimum_benchmark_win_ratio
                ),
                "require_benchmark_outperformance": (
                    self.require_benchmark_outperformance
                ),
            },
        )

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: OutOfSampleReport,
    ) -> float:
        positive_component = (
            report.positive_oos_ratio
        )

        stability_component = (
            report.stable_segment_ratio
        )

        if (
            report.average_degradation_ratio
            <= 0.0
        ):
            degradation_component = 1.0

        elif self.maximum_degradation <= 0.0:
            degradation_component = 0.0

        else:
            degradation_component = max(
                0.0,
                1.0
                - (
                    report.average_degradation_ratio
                    / self.maximum_degradation
                ),
            )

        if report.benchmark_win_ratio is None:
            benchmark_component = 1.0
        else:
            benchmark_component = (
                report.benchmark_win_ratio
            )

        score = (
            0.35 * positive_component
            + 0.35 * stability_component
            + 0.20 * degradation_component
            + 0.10 * benchmark_component
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
        report: OutOfSampleReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        if report.passed:
            message = (
                "Out-of-sample validation passed."
            )
        else:
            message = (
                "Out-of-sample validation detected "
                "insufficient generalization."
            )

        return ValidationResult(
            stage=ValidationStage.OUT_OF_SAMPLE,
            status=status,
            score=self._calculate_score(report),
            message=message,
            diagnostics={
                "segment_count": (
                    report.segment_count
                ),
                "average_in_sample_score": (
                    report.average_in_sample_score
                ),
                "average_out_of_sample_score": (
                    report.average_out_of_sample_score
                ),
                "average_degradation_ratio": (
                    report.average_degradation_ratio
                ),
                "positive_oos_ratio": (
                    report.positive_oos_ratio
                ),
                "stable_segment_ratio": (
                    report.stable_segment_ratio
                ),
                "benchmark_comparison_count": (
                    report.benchmark_comparison_count
                ),
                "benchmark_win_ratio": (
                    report.benchmark_win_ratio
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
    # Convenience validation
    # ========================================================

    def validate_scores(
        self,
        *,
        observation_count: int,
        in_sample_score: float,
        out_of_sample_scores: Sequence[float],
        benchmark_scores: Optional[
            Sequence[float]
        ] = None,
    ) -> OutOfSampleReport:
        if not out_of_sample_scores:
            raise OutOfSampleValidationError(
                "At least one out-of-sample score "
                "is required."
            )

        segments = self.build_segments(
            observation_count,
            len(out_of_sample_scores),
        )

        if benchmark_scores is not None:
            if (
                len(benchmark_scores)
                != len(out_of_sample_scores)
            ):
                raise OutOfSampleValidationError(
                    "benchmark_scores must have the same "
                    "length as out_of_sample_scores."
                )

        results: List[
            OutOfSampleSegmentResult
        ] = []

        for index, (
            segment,
            oos_score,
        ) in enumerate(
            zip(
                segments,
                out_of_sample_scores,
            )
        ):
            benchmark_score = (
                benchmark_scores[index]
                if benchmark_scores is not None
                else None
            )

            results.append(
                self.create_segment_result(
                    segment=segment,
                    in_sample_score=(
                        in_sample_score
                    ),
                    out_of_sample_score=(
                        oos_score
                    ),
                    benchmark_score=(
                        benchmark_score
                    ),
                )
            )

        return self.aggregate(results)

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "maximum_degradation": (
                self.maximum_degradation
            ),
            "minimum_positive_oos_ratio": (
                self.minimum_positive_oos_ratio
            ),
            "minimum_stable_segment_ratio": (
                self.minimum_stable_segment_ratio
            ),
            "minimum_benchmark_win_ratio": (
                self.minimum_benchmark_win_ratio
            ),
            "require_benchmark_outperformance": (
                self.require_benchmark_outperformance
            ),
            "research_only": True,
            "real_money_execution": False,
        }