from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from research_benchmarking.models import (
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


class SensitivityValidationError(Exception):
    """Raised when sensitivity validation cannot be performed."""


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
    numeric = _validate_numeric(
        value,
        name,
    )

    if numeric < 0.0:
        raise ValidationConfigurationError(
            f"{name} must be non-negative."
        )

    return numeric


def _validate_ratio(
    value: float,
    name: str,
) -> float:
    numeric = _validate_numeric(
        value,
        name,
    )

    if not 0.0 <= numeric <= 1.0:
        raise ValidationConfigurationError(
            f"{name} must be between 0 and 1."
        )

    return numeric


# ============================================================
# Parameter perturbation
# ============================================================


@dataclass(frozen=True)
class ParameterPerturbation:
    parameter_name: str
    baseline_value: float
    perturbed_value: float
    baseline_score: float
    perturbed_score: float
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter_name, str)
            or not self.parameter_name.strip()
        ):
            raise ValidationConfigurationError(
                "parameter_name must be a non-empty string."
            )

        for name, value in {
            "baseline_value": self.baseline_value,
            "perturbed_value": self.perturbed_value,
            "baseline_score": self.baseline_score,
            "perturbed_score": self.perturbed_score,
        }.items():
            _validate_numeric(
                value,
                name,
            )

    @property
    def parameter_change(self) -> float:
        return (
            self.perturbed_value
            - self.baseline_value
        )

    @property
    def absolute_parameter_change(self) -> float:
        return abs(
            self.parameter_change
        )

    @property
    def relative_parameter_change(self) -> float:
        if math.isclose(
            self.baseline_value,
            0.0,
            abs_tol=1e-15,
        ):
            if math.isclose(
                self.perturbed_value,
                0.0,
                abs_tol=1e-15,
            ):
                return 0.0

            return 1.0

        return (
            self.parameter_change
            / abs(self.baseline_value)
        )

    @property
    def score_change(self) -> float:
        return (
            self.perturbed_score
            - self.baseline_score
        )

    @property
    def degradation(self) -> float:
        return (
            self.baseline_score
            - self.perturbed_score
        )

    @property
    def degradation_ratio(self) -> float:
        if math.isclose(
            self.baseline_score,
            0.0,
            abs_tol=1e-15,
        ):
            if math.isclose(
                self.perturbed_score,
                0.0,
                abs_tol=1e-15,
            ):
                return 0.0

            return (
                1.0
                if self.perturbed_score < 0.0
                else -1.0
            )

        return (
            self.degradation
            / abs(self.baseline_score)
        )

    @property
    def positive_after_perturbation(self) -> bool:
        return self.perturbed_score > 0.0

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "parameter_name": self.parameter_name,
                "baseline_value": self.baseline_value,
                "perturbed_value": self.perturbed_value,
                "baseline_score": self.baseline_score,
                "perturbed_score": self.perturbed_score,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Parameter sensitivity summary
# ============================================================


@dataclass(frozen=True)
class ParameterSensitivity:
    parameter_name: str
    perturbations: List[ParameterPerturbation]
    average_absolute_degradation_ratio: float
    worst_degradation_ratio: float
    positive_ratio: float
    stable_ratio: float
    local_stability_score: float
    stable: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parameter_name, str)
            or not self.parameter_name.strip()
        ):
            raise ValidationConfigurationError(
                "parameter_name must be a non-empty string."
            )

        if not self.perturbations:
            raise ValidationConfigurationError(
                "perturbations cannot be empty."
            )

        for perturbation in self.perturbations:
            if not isinstance(
                perturbation,
                ParameterPerturbation,
            ):
                raise ValidationConfigurationError(
                    "perturbations must contain "
                    "ParameterPerturbation objects."
                )

            if (
                perturbation.parameter_name
                != self.parameter_name
            ):
                raise ValidationConfigurationError(
                    "All perturbations must belong to "
                    "the same parameter."
                )

        _validate_non_negative(
            self.average_absolute_degradation_ratio,
            "average_absolute_degradation_ratio",
        )

        _validate_numeric(
            self.worst_degradation_ratio,
            "worst_degradation_ratio",
        )

        _validate_ratio(
            self.positive_ratio,
            "positive_ratio",
        )

        _validate_ratio(
            self.stable_ratio,
            "stable_ratio",
        )

        _validate_ratio(
            self.local_stability_score,
            "local_stability_score",
        )

        if not isinstance(
            self.stable,
            bool,
        ):
            raise ValidationConfigurationError(
                "stable must be bool."
            )

    @property
    def perturbation_count(self) -> int:
        return len(
            self.perturbations
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "parameter_name": self.parameter_name,
                "perturbations": [
                    item.fingerprint
                    for item in self.perturbations
                ],
                "average_absolute_degradation_ratio": (
                    self.average_absolute_degradation_ratio
                ),
                "worst_degradation_ratio": (
                    self.worst_degradation_ratio
                ),
                "positive_ratio": self.positive_ratio,
                "stable_ratio": self.stable_ratio,
                "local_stability_score": (
                    self.local_stability_score
                ),
                "stable": self.stable,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Report
# ============================================================


@dataclass(frozen=True)
class SensitivityReport:
    parameter_results: List[ParameterSensitivity]
    average_stability_score: float
    stable_parameter_ratio: float
    minimum_stability_score: float
    most_sensitive_parameter: str
    least_sensitive_parameter: str
    passed: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.parameter_results:
            raise ValidationConfigurationError(
                "parameter_results cannot be empty."
            )

        for result in self.parameter_results:
            if not isinstance(
                result,
                ParameterSensitivity,
            ):
                raise ValidationConfigurationError(
                    "parameter_results must contain "
                    "ParameterSensitivity objects."
                )

        _validate_ratio(
            self.average_stability_score,
            "average_stability_score",
        )

        _validate_ratio(
            self.stable_parameter_ratio,
            "stable_parameter_ratio",
        )

        _validate_ratio(
            self.minimum_stability_score,
            "minimum_stability_score",
        )

        if (
            not isinstance(
                self.most_sensitive_parameter,
                str,
            )
            or not self.most_sensitive_parameter
        ):
            raise ValidationConfigurationError(
                "most_sensitive_parameter must "
                "be a non-empty string."
            )

        if (
            not isinstance(
                self.least_sensitive_parameter,
                str,
            )
            or not self.least_sensitive_parameter
        ):
            raise ValidationConfigurationError(
                "least_sensitive_parameter must "
                "be a non-empty string."
            )

        if not isinstance(
            self.passed,
            bool,
        ):
            raise ValidationConfigurationError(
                "passed must be bool."
            )

    @property
    def parameter_count(self) -> int:
        return len(
            self.parameter_results
        )

    @property
    def stable_parameter_count(self) -> int:
        return sum(
            1
            for result
            in self.parameter_results
            if result.stable
        )

    @property
    def unstable_parameters(self) -> List[str]:
        return [
            result.parameter_name
            for result
            in self.parameter_results
            if not result.stable
        ]

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "parameter_results": [
                    result.fingerprint
                    for result
                    in self.parameter_results
                ],
                "average_stability_score": (
                    self.average_stability_score
                ),
                "stable_parameter_ratio": (
                    self.stable_parameter_ratio
                ),
                "minimum_stability_score": (
                    self.minimum_stability_score
                ),
                "most_sensitive_parameter": (
                    self.most_sensitive_parameter
                ),
                "least_sensitive_parameter": (
                    self.least_sensitive_parameter
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Engine
# ============================================================


class SensitivityEngine:
    """
    Parameter-neighborhood sensitivity analysis.

    The engine measures whether a strategy or research signal
    remains stable when its parameters are moved away from the
    selected optimum.

    A robust strategy should have a reasonably broad stable
    neighborhood rather than a single narrow parameter peak.
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        maximum_degradation_ratio: float = 0.50,
        minimum_positive_ratio: float = 0.70,
        minimum_local_stability_score: float = 0.65,
        minimum_stable_parameter_ratio: float = 0.70,
    ) -> None:
        self.config = (
            config
            or ValidationConfig()
        )

        self.maximum_degradation_ratio = (
            _validate_non_negative(
                maximum_degradation_ratio,
                "maximum_degradation_ratio",
            )
        )

        self.minimum_positive_ratio = (
            _validate_ratio(
                minimum_positive_ratio,
                "minimum_positive_ratio",
            )
        )

        self.minimum_local_stability_score = (
            _validate_ratio(
                minimum_local_stability_score,
                "minimum_local_stability_score",
            )
        )

        self.minimum_stable_parameter_ratio = (
            _validate_ratio(
                minimum_stable_parameter_ratio,
                "minimum_stable_parameter_ratio",
            )
        )

    # ========================================================
    # Perturbation grid
    # ========================================================

    def build_perturbation_grid(
        self,
        *,
        parameter_name: str,
        baseline_value: float,
        shifts: Sequence[float] = (
            -0.20,
            -0.10,
            -0.05,
            0.05,
            0.10,
            0.20,
        ),
    ) -> List[float]:
        if (
            not isinstance(parameter_name, str)
            or not parameter_name.strip()
        ):
            raise ValidationConfigurationError(
                "parameter_name must be a non-empty string."
            )

        baseline = _validate_numeric(
            baseline_value,
            "baseline_value",
        )

        if not shifts:
            raise SensitivityValidationError(
                "shifts cannot be empty."
            )

        validated_shifts = [
            _validate_numeric(
                shift,
                f"shifts[{index}]",
            )
            for index, shift
            in enumerate(shifts)
        ]

        return [
            baseline * (1.0 + shift)
            if not math.isclose(
                baseline,
                0.0,
                abs_tol=1e-15,
            )
            else shift
            for shift in validated_shifts
        ]

    # ========================================================
    # Perturbation construction
    # ========================================================

    def create_perturbation(
        self,
        *,
        parameter_name: str,
        baseline_value: float,
        perturbed_value: float,
        baseline_score: float,
        perturbed_score: float,
    ) -> ParameterPerturbation:
        return ParameterPerturbation(
            parameter_name=parameter_name,
            baseline_value=baseline_value,
            perturbed_value=perturbed_value,
            baseline_score=baseline_score,
            perturbed_score=perturbed_score,
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Single parameter analysis
    # ========================================================

    def analyze_parameter(
        self,
        *,
        parameter_name: str,
        baseline_value: float,
        baseline_score: float,
        perturbed_values: Sequence[float],
        perturbed_scores: Sequence[float],
    ) -> ParameterSensitivity:
        if not perturbed_values:
            raise SensitivityValidationError(
                "perturbed_values cannot be empty."
            )

        if not perturbed_scores:
            raise SensitivityValidationError(
                "perturbed_scores cannot be empty."
            )

        if len(
            perturbed_values
        ) != len(
            perturbed_scores
        ):
            raise SensitivityValidationError(
                "perturbed_values and perturbed_scores "
                "must have equal length."
            )

        perturbations = [
            self.create_perturbation(
                parameter_name=parameter_name,
                baseline_value=baseline_value,
                perturbed_value=value,
                baseline_score=baseline_score,
                perturbed_score=score,
            )
            for value, score
            in zip(
                perturbed_values,
                perturbed_scores,
            )
        ]

        count = len(
            perturbations
        )

        absolute_degradation_ratios = [
            max(
                0.0,
                item.degradation_ratio,
            )
            for item in perturbations
        ]

        average_absolute_degradation_ratio = (
            sum(
                absolute_degradation_ratios
            )
            / count
        )

        worst_degradation_ratio = max(
            item.degradation_ratio
            for item in perturbations
        )

        positive_count = sum(
            1
            for item in perturbations
            if item.positive_after_perturbation
        )

        positive_ratio = (
            positive_count
            / count
        )

        stable_count = sum(
            1
            for item in perturbations
            if (
                item.positive_after_perturbation
                and item.degradation_ratio
                <= self.maximum_degradation_ratio
            )
        )

        stable_ratio = (
            stable_count
            / count
        )

        degradation_component = max(
            0.0,
            min(
                1.0,
                1.0
                - average_absolute_degradation_ratio,
            ),
        )

        worst_case_component = max(
            0.0,
            min(
                1.0,
                1.0
                - max(
                    0.0,
                    worst_degradation_ratio,
                ),
            ),
        )

        local_stability_score = (
            0.35 * positive_ratio
            + 0.35 * stable_ratio
            + 0.20 * degradation_component
            + 0.10 * worst_case_component
        )

        local_stability_score = max(
            0.0,
            min(
                1.0,
                float(local_stability_score),
            ),
        )

        stable = (
            positive_ratio
            >= self.minimum_positive_ratio
            and local_stability_score
            >= self.minimum_local_stability_score
            and worst_degradation_ratio
            <= self.maximum_degradation_ratio
        )

        return ParameterSensitivity(
            parameter_name=parameter_name,
            perturbations=perturbations,
            average_absolute_degradation_ratio=(
                average_absolute_degradation_ratio
            ),
            worst_degradation_ratio=(
                worst_degradation_ratio
            ),
            positive_ratio=positive_ratio,
            stable_ratio=stable_ratio,
            local_stability_score=(
                local_stability_score
            ),
            stable=stable,
            metadata={
                "version": VERSION,
                "maximum_degradation_ratio": (
                    self.maximum_degradation_ratio
                ),
                "minimum_positive_ratio": (
                    self.minimum_positive_ratio
                ),
                "minimum_local_stability_score": (
                    self.minimum_local_stability_score
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Aggregate
    # ========================================================

    def aggregate(
        self,
        results: Sequence[
            ParameterSensitivity
        ],
    ) -> SensitivityReport:
        if not results:
            raise SensitivityValidationError(
                "results cannot be empty."
            )

        validated: List[
            ParameterSensitivity
        ] = []

        for result in results:
            if not isinstance(
                result,
                ParameterSensitivity,
            ):
                raise ValidationConfigurationError(
                    "results must contain "
                    "ParameterSensitivity objects."
                )

            validated.append(result)

        count = len(
            validated
        )

        average_stability_score = (
            sum(
                result.local_stability_score
                for result in validated
            )
            / count
        )

        stable_parameter_count = sum(
            1
            for result in validated
            if result.stable
        )

        stable_parameter_ratio = (
            stable_parameter_count
            / count
        )

        minimum_stability_score = min(
            result.local_stability_score
            for result in validated
        )

        most_sensitive = min(
            validated,
            key=lambda result: (
                result.local_stability_score
            ),
        )

        least_sensitive = max(
            validated,
            key=lambda result: (
                result.local_stability_score
            ),
        )

        passed = (
            stable_parameter_ratio
            >= self.minimum_stable_parameter_ratio
            and average_stability_score
            >= self.minimum_local_stability_score
        )

        return SensitivityReport(
            parameter_results=validated,
            average_stability_score=(
                average_stability_score
            ),
            stable_parameter_ratio=(
                stable_parameter_ratio
            ),
            minimum_stability_score=(
                minimum_stability_score
            ),
            most_sensitive_parameter=(
                most_sensitive.parameter_name
            ),
            least_sensitive_parameter=(
                least_sensitive.parameter_name
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "minimum_stable_parameter_ratio": (
                    self.minimum_stable_parameter_ratio
                ),
                "minimum_local_stability_score": (
                    self.minimum_local_stability_score
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Validation from mappings
    # ========================================================

    def validate_parameters(
        self,
        *,
        baseline_score: float,
        parameter_data: Mapping[
            str,
            Mapping[str, Any],
        ],
    ) -> SensitivityReport:
        _validate_numeric(
            baseline_score,
            "baseline_score",
        )

        if not parameter_data:
            raise SensitivityValidationError(
                "parameter_data cannot be empty."
            )

        results: List[
            ParameterSensitivity
        ] = []

        for parameter_name, data in (
            parameter_data.items()
        ):
            if not isinstance(
                data,
                Mapping,
            ):
                raise ValidationConfigurationError(
                    "Each parameter entry must "
                    "be a mapping."
                )

            if "baseline_value" not in data:
                raise SensitivityValidationError(
                    f"{parameter_name} is missing "
                    "baseline_value."
                )

            if "perturbed_values" not in data:
                raise SensitivityValidationError(
                    f"{parameter_name} is missing "
                    "perturbed_values."
                )

            if "perturbed_scores" not in data:
                raise SensitivityValidationError(
                    f"{parameter_name} is missing "
                    "perturbed_scores."
                )

            results.append(
                self.analyze_parameter(
                    parameter_name=parameter_name,
                    baseline_value=data[
                        "baseline_value"
                    ],
                    baseline_score=baseline_score,
                    perturbed_values=data[
                        "perturbed_values"
                    ],
                    perturbed_scores=data[
                        "perturbed_scores"
                    ],
                )
            )

        return self.aggregate(
            results
        )

    # ========================================================
    # Core model adapter
    # ========================================================

    def to_sensitivity_results(
        self,
        report: SensitivityReport,
    ) -> List[SensitivityResult]:
        """
        Converts internal sensitivity summaries to the shared
        v6.0.2 SensitivityResult model when its constructor
        matches the expected research-benchmarking contract.

        This adapter intentionally stays isolated from the core
        calculation layer.
        """

        converted: List[
            SensitivityResult
        ] = []

        for result in report.parameter_results:
            try:
                converted.append(
                    SensitivityResult(
                        parameter_name=(
                            result.parameter_name
                        ),
                        baseline_value=(
                            result.perturbations[
                                0
                            ].baseline_value
                        ),
                        tested_values=[
                            item.perturbed_value
                            for item
                            in result.perturbations
                        ],
                        scores=[
                            item.perturbed_score
                            for item
                            in result.perturbations
                        ],
                        stability_score=(
                            result.local_stability_score
                        ),
                        metadata={
                            "version": VERSION,
                            "stable": result.stable,
                            "research_only": True,
                            "real_money_execution": False,
                        },
                    )
                )
            except TypeError:
                # The shared model may evolve independently.
                # The core sensitivity engine remains usable.
                break

        return converted

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: SensitivityReport,
    ) -> float:
        score = (
            0.50
            * report.average_stability_score
            + 0.35
            * report.stable_parameter_ratio
            + 0.15
            * report.minimum_stability_score
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
        report: SensitivityReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Parameter neighborhood remained stable."
            if report.passed
            else (
                "Parameter sensitivity indicates a "
                "potentially narrow or unstable optimum."
            )
        )

        return ValidationResult(
            stage=ValidationStage.SENSITIVITY,
            status=status,
            score=self._calculate_score(
                report
            ),
            message=message,
            diagnostics={
                "parameter_count": (
                    report.parameter_count
                ),
                "stable_parameter_count": (
                    report.stable_parameter_count
                ),
                "stable_parameter_ratio": (
                    report.stable_parameter_ratio
                ),
                "average_stability_score": (
                    report.average_stability_score
                ),
                "minimum_stability_score": (
                    report.minimum_stability_score
                ),
                "most_sensitive_parameter": (
                    report.most_sensitive_parameter
                ),
                "least_sensitive_parameter": (
                    report.least_sensitive_parameter
                ),
                "unstable_parameters": (
                    report.unstable_parameters
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
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "maximum_degradation_ratio": (
                self.maximum_degradation_ratio
            ),
            "minimum_positive_ratio": (
                self.minimum_positive_ratio
            ),
            "minimum_local_stability_score": (
                self.minimum_local_stability_score
            ),
            "minimum_stable_parameter_ratio": (
                self.minimum_stable_parameter_ratio
            ),
            "default_parameter_shifts": [
                -0.20,
                -0.10,
                -0.05,
                0.05,
                0.10,
                0.20,
            ],
            "research_only": True,
            "real_money_execution": False,
        }