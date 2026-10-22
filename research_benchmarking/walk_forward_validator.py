from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


class WalkForwardValidationError(Exception):
    """Raised when walk-forward validation cannot be completed."""


# ============================================================
# Models
# ============================================================


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int

    train_start: int
    train_end: int

    validation_start: int
    validation_end: int

    test_start: int
    test_end: int

    embargo_size: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        integer_fields = {
            "window_id": self.window_id,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "embargo_size": self.embargo_size,
        }

        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise ValidationConfigurationError(
                    f"{name} must be an integer."
                )

        if self.window_id < 0:
            raise ValidationConfigurationError(
                "window_id must be non-negative."
            )

        if self.embargo_size < 0:
            raise ValidationConfigurationError(
                "embargo_size must be non-negative."
            )

        if self.train_start < 0:
            raise ValidationConfigurationError(
                "train_start must be non-negative."
            )

        if self.train_end <= self.train_start:
            raise ValidationConfigurationError(
                "train_end must be greater than train_start."
            )

        if self.validation_start < self.train_end:
            raise ValidationConfigurationError(
                "validation_start cannot overlap training data."
            )

        if self.validation_end <= self.validation_start:
            raise ValidationConfigurationError(
                "validation_end must be greater than "
                "validation_start."
            )

        if self.test_start < self.validation_end:
            raise ValidationConfigurationError(
                "test_start cannot overlap validation data."
            )

        if self.test_end <= self.test_start:
            raise ValidationConfigurationError(
                "test_end must be greater than test_start."
            )

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def validation_size(self) -> int:
        return (
            self.validation_end
            - self.validation_start
        )

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    @property
    def total_size(self) -> int:
        return (
            self.train_size
            + self.validation_size
            + self.test_size
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "window_id": self.window_id,
                "train_start": self.train_start,
                "train_end": self.train_end,
                "validation_start": (
                    self.validation_start
                ),
                "validation_end": (
                    self.validation_end
                ),
                "test_start": self.test_start,
                "test_end": self.test_end,
                "embargo_size": self.embargo_size,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class WalkForwardWindowResult:
    window: WalkForwardWindow

    train_score: float
    validation_score: float
    test_score: float

    train_metrics: Optional[
        PerformanceMetrics
    ] = None

    validation_metrics: Optional[
        PerformanceMetrics
    ] = None

    test_metrics: Optional[
        PerformanceMetrics
    ] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name, value in {
            "train_score": self.train_score,
            "validation_score": self.validation_score,
            "test_score": self.test_score,
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

    @property
    def validation_degradation(self) -> float:
        if math.isclose(
            self.train_score,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return (
            self.train_score
            - self.validation_score
        ) / abs(self.train_score)

    @property
    def test_degradation(self) -> float:
        if math.isclose(
            self.train_score,
            0.0,
            abs_tol=1e-12,
        ):
            return 0.0

        return (
            self.train_score
            - self.test_score
        ) / abs(self.train_score)

    @property
    def positive_oos(self) -> bool:
        return self.test_score > 0.0

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "window": self.window.fingerprint,
                "train_score": self.train_score,
                "validation_score": (
                    self.validation_score
                ),
                "test_score": self.test_score,
                "train_metrics": (
                    self.train_metrics.fingerprint
                    if self.train_metrics
                    is not None
                    else None
                ),
                "validation_metrics": (
                    self.validation_metrics.fingerprint
                    if self.validation_metrics
                    is not None
                    else None
                ),
                "test_metrics": (
                    self.test_metrics.fingerprint
                    if self.test_metrics
                    is not None
                    else None
                ),
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True)
class WalkForwardReport:
    windows: List[WalkForwardWindowResult]

    average_train_score: float
    average_validation_score: float
    average_test_score: float

    average_test_degradation: float

    positive_oos_ratio: float
    stable_window_ratio: float

    passed: bool

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValidationConfigurationError(
                "WalkForwardReport requires at least one window."
            )

        numeric_fields = {
            "average_train_score": (
                self.average_train_score
            ),
            "average_validation_score": (
                self.average_validation_score
            ),
            "average_test_score": (
                self.average_test_score
            ),
            "average_test_degradation": (
                self.average_test_degradation
            ),
            "positive_oos_ratio": (
                self.positive_oos_ratio
            ),
            "stable_window_ratio": (
                self.stable_window_ratio
            ),
        }

        for name, value in numeric_fields.items():
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

        if not 0.0 <= self.stable_window_ratio <= 1.0:
            raise ValidationConfigurationError(
                "stable_window_ratio must be between 0 and 1."
            )

    @property
    def window_count(self) -> int:
        return len(self.windows)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "windows": [
                    result.fingerprint
                    for result in self.windows
                ],
                "average_train_score": (
                    self.average_train_score
                ),
                "average_validation_score": (
                    self.average_validation_score
                ),
                "average_test_score": (
                    self.average_test_score
                ),
                "average_test_degradation": (
                    self.average_test_degradation
                ),
                "positive_oos_ratio": (
                    self.positive_oos_ratio
                ),
                "stable_window_ratio": (
                    self.stable_window_ratio
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Walk-forward validator
# ============================================================


class WalkForwardValidator:
    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        embargo_size: int = 0,
        expanding_window: bool = True,
        maximum_degradation: float = 0.50,
        minimum_positive_oos_ratio: float = 0.60,
        minimum_stable_window_ratio: float = 0.60,
    ) -> None:
        self.config = config or ValidationConfig()

        if (
            isinstance(embargo_size, bool)
            or not isinstance(embargo_size, int)
            or embargo_size < 0
        ):
            raise ValidationConfigurationError(
                "embargo_size must be a non-negative integer."
            )

        for name, value in {
            "maximum_degradation": maximum_degradation,
            "minimum_positive_oos_ratio": (
                minimum_positive_oos_ratio
            ),
            "minimum_stable_window_ratio": (
                minimum_stable_window_ratio
            ),
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

        if not (
            0.0
            <= minimum_positive_oos_ratio
            <= 1.0
        ):
            raise ValidationConfigurationError(
                "minimum_positive_oos_ratio must be "
                "between 0 and 1."
            )

        if not (
            0.0
            <= minimum_stable_window_ratio
            <= 1.0
        ):
            raise ValidationConfigurationError(
                "minimum_stable_window_ratio must be "
                "between 0 and 1."
            )

        self.embargo_size = embargo_size
        self.expanding_window = bool(
            expanding_window
        )

        self.maximum_degradation = float(
            maximum_degradation
        )

        self.minimum_positive_oos_ratio = float(
            minimum_positive_oos_ratio
        )

        self.minimum_stable_window_ratio = float(
            minimum_stable_window_ratio
        )

    # ========================================================
    # Window construction
    # ========================================================

    def build_windows(
        self,
        observation_count: int,
    ) -> List[WalkForwardWindow]:
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
            raise WalkForwardValidationError(
                "Insufficient observations for "
                "walk-forward validation."
            )

        window_count = (
            self.config.walk_forward_windows
        )

        minimum_segment_size = 1

        usable_count = (
            observation_count
            - (
                self.embargo_size
                * 2
                * window_count
            )
        )

        if usable_count <= 0:
            raise WalkForwardValidationError(
                "Embargo configuration leaves no usable data."
            )

        base_test_size = max(
            minimum_segment_size,
            int(
                usable_count
                * self.config.test_fraction
                / window_count
            ),
        )

        validation_size = max(
            minimum_segment_size,
            int(
                usable_count
                * self.config.validation_fraction
                / window_count
            ),
        )

        initial_train_size = (
            observation_count
            - (
                window_count
                * (
                    validation_size
                    + base_test_size
                    + (2 * self.embargo_size)
                )
            )
        )

        if initial_train_size <= 0:
            raise WalkForwardValidationError(
                "Not enough observations to construct "
                "the requested walk-forward windows."
            )

        windows: List[
            WalkForwardWindow
        ] = []

        cursor = initial_train_size

        for window_id in range(window_count):
            if self.expanding_window:
                train_start = 0
                train_end = cursor
            else:
                train_start = (
                    window_id
                    * base_test_size
                )

                train_end = cursor

            validation_start = (
                train_end
                + self.embargo_size
            )

            validation_end = (
                validation_start
                + validation_size
            )

            test_start = (
                validation_end
                + self.embargo_size
            )

            if window_id == window_count - 1:
                test_end = observation_count
            else:
                test_end = min(
                    observation_count,
                    test_start
                    + base_test_size,
                )

            if (
                validation_end
                > observation_count
                or test_start
                >= observation_count
                or test_end
                > observation_count
            ):
                raise WalkForwardValidationError(
                    "Walk-forward window exceeds "
                    "available observations."
                )

            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                validation_start=(
                    validation_start
                ),
                validation_end=(
                    validation_end
                ),
                test_start=test_start,
                test_end=test_end,
                embargo_size=(
                    self.embargo_size
                ),
            )

            windows.append(window)

            cursor = test_end

        return windows

    # ========================================================
    # Sequence splitting
    # ========================================================

    @staticmethod
    def split_sequence(
        values: Sequence[Any],
        window: WalkForwardWindow,
    ) -> Tuple[
        Sequence[Any],
        Sequence[Any],
        Sequence[Any],
    ]:
        if len(values) < window.test_end:
            raise WalkForwardValidationError(
                "Input sequence is shorter than "
                "the walk-forward window."
            )

        train = values[
            window.train_start:
            window.train_end
        ]

        validation = values[
            window.validation_start:
            window.validation_end
        ]

        test = values[
            window.test_start:
            window.test_end
        ]

        return (
            train,
            validation,
            test,
        )

    # ========================================================
    # Result construction
    # ========================================================

    def create_window_result(
        self,
        *,
        window: WalkForwardWindow,
        train_score: float,
        validation_score: float,
        test_score: float,
        train_metrics: Optional[
            PerformanceMetrics
        ] = None,
        validation_metrics: Optional[
            PerformanceMetrics
        ] = None,
        test_metrics: Optional[
            PerformanceMetrics
        ] = None,
        metadata: Optional[
            Dict[str, Any]
        ] = None,
    ) -> WalkForwardWindowResult:
        return WalkForwardWindowResult(
            window=window,
            train_score=train_score,
            validation_score=(
                validation_score
            ),
            test_score=test_score,
            train_metrics=train_metrics,
            validation_metrics=(
                validation_metrics
            ),
            test_metrics=test_metrics,
            metadata=dict(
                metadata or {}
            ),
        )

    # ========================================================
    # Aggregation
    # ========================================================

    def aggregate(
        self,
        results: Sequence[
            WalkForwardWindowResult
        ],
    ) -> WalkForwardReport:
        if not results:
            raise WalkForwardValidationError(
                "At least one walk-forward result "
                "is required."
            )

        result_list = list(results)

        average_train_score = (
            sum(
                result.train_score
                for result in result_list
            )
            / len(result_list)
        )

        average_validation_score = (
            sum(
                result.validation_score
                for result in result_list
            )
            / len(result_list)
        )

        average_test_score = (
            sum(
                result.test_score
                for result in result_list
            )
            / len(result_list)
        )

        degradations = [
            result.test_degradation
            for result in result_list
        ]

        average_test_degradation = (
            sum(degradations)
            / len(degradations)
        )

        positive_oos_count = sum(
            1
            for result in result_list
            if result.positive_oos
        )

        positive_oos_ratio = (
            positive_oos_count
            / len(result_list)
        )

        stable_count = sum(
            1
            for result in result_list
            if (
                result.test_degradation
                <= self.maximum_degradation
            )
        )

        stable_window_ratio = (
            stable_count
            / len(result_list)
        )

        passed = (
            average_test_score > 0.0
            and positive_oos_ratio
            >= self.minimum_positive_oos_ratio
            and stable_window_ratio
            >= self.minimum_stable_window_ratio
            and average_test_degradation
            <= self.maximum_degradation
        )

        return WalkForwardReport(
            windows=result_list,
            average_train_score=(
                average_train_score
            ),
            average_validation_score=(
                average_validation_score
            ),
            average_test_score=(
                average_test_score
            ),
            average_test_degradation=(
                average_test_degradation
            ),
            positive_oos_ratio=(
                positive_oos_ratio
            ),
            stable_window_ratio=(
                stable_window_ratio
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "expanding_window": (
                    self.expanding_window
                ),
                "embargo_size": (
                    self.embargo_size
                ),
                "maximum_degradation": (
                    self.maximum_degradation
                ),
                "minimum_positive_oos_ratio": (
                    self.minimum_positive_oos_ratio
                ),
                "minimum_stable_window_ratio": (
                    self.minimum_stable_window_ratio
                ),
            },
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: WalkForwardReport,
    ) -> ValidationResult:
        if report.passed:
            status = ValidationStatus.PASSED
            message = (
                "Walk-forward validation passed."
            )
        else:
            status = ValidationStatus.FAILED
            message = (
                "Walk-forward validation detected "
                "insufficient out-of-sample stability."
            )

        score = self._calculate_score(
            report
        )

        return ValidationResult(
            stage=ValidationStage.WALK_FORWARD,
            status=status,
            score=score,
            message=message,
            diagnostics={
                "window_count": (
                    report.window_count
                ),
                "average_train_score": (
                    report.average_train_score
                ),
                "average_validation_score": (
                    report.average_validation_score
                ),
                "average_test_score": (
                    report.average_test_score
                ),
                "average_test_degradation": (
                    report.average_test_degradation
                ),
                "positive_oos_ratio": (
                    report.positive_oos_ratio
                ),
                "stable_window_ratio": (
                    report.stable_window_ratio
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
    # Scoring
    # ========================================================

    def _calculate_score(
        self,
        report: WalkForwardReport,
    ) -> float:
        positive_component = (
            report.positive_oos_ratio
        )

        stability_component = (
            report.stable_window_ratio
        )

        if (
            report.average_test_degradation
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
                    report.average_test_degradation
                    / self.maximum_degradation
                ),
            )

        score = (
            0.40 * positive_component
            + 0.40 * stability_component
            + 0.20 * degradation_component
        )

        return min(
            1.0,
            max(
                0.0,
                float(score),
            ),
        )

    # ========================================================
    # Convenience validation
    # ========================================================

    def validate_scores(
        self,
        *,
        observation_count: int,
        scores: Sequence[
            Tuple[float, float, float]
        ],
    ) -> WalkForwardReport:
        windows = self.build_windows(
            observation_count
        )

        if len(scores) != len(windows):
            raise WalkForwardValidationError(
                "The number of score tuples must equal "
                "the number of walk-forward windows."
            )

        results = []

        for window, score_tuple in zip(
            windows,
            scores,
        ):
            if len(score_tuple) != 3:
                raise WalkForwardValidationError(
                    "Each score tuple must contain "
                    "train, validation and test scores."
                )

            train_score = score_tuple[0]
            validation_score = score_tuple[1]
            test_score = score_tuple[2]

            results.append(
                self.create_window_result(
                    window=window,
                    train_score=train_score,
                    validation_score=(
                        validation_score
                    ),
                    test_score=test_score,
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
            "walk_forward_windows": (
                self.config.walk_forward_windows
            ),
            "embargo_size": self.embargo_size,
            "expanding_window": (
                self.expanding_window
            ),
            "maximum_degradation": (
                self.maximum_degradation
            ),
            "minimum_positive_oos_ratio": (
                self.minimum_positive_oos_ratio
            ),
            "minimum_stable_window_ratio": (
                self.minimum_stable_window_ratio
            ),
            "research_only": True,
            "real_money_execution": False,
        }