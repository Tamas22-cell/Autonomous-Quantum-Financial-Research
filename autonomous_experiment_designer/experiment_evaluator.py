from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Mapping, Sequence

from autonomous_experiment_designer.experiment_executor import (
    ExecutionResult,
    ExecutionStatus,
)


class EvaluationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    INCONCLUSIVE = "INCONCLUSIVE"
    CONTRADICTED = "CONTRADICTED"
    FAILED = "FAILED"


class Recommendation(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REPEAT = "REPEAT"
    INVESTIGATE = "INVESTIGATE"


@dataclass(frozen=True)
class EvaluationMetric:
    name: str
    baseline: float
    observed: float
    improvement: float
    relative_change: float


@dataclass(frozen=True)
class ExperimentEvaluation:
    execution_id: str
    experiment_id: str
    status: EvaluationStatus
    recommendation: Recommendation
    metrics: tuple[EvaluationMetric, ...]
    confidence: float
    validation_score: float
    evidence: tuple[str, ...]
    rationale: str


class ExperimentEvaluator:
    """
    Evaluates completed experiment results against a baseline.

    The evaluator never executes experiments and never changes the
    original execution result.
    """

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.70,
        minimum_validation_score: float = 0.70,
        minimum_improvement: float = 0.0,
        relative_change_epsilon: float = 1e-12,
    ) -> None:
        self._validate_probability(
            minimum_confidence,
            "minimum_confidence",
        )
        self._validate_probability(
            minimum_validation_score,
            "minimum_validation_score",
        )
        self._validate_nonnegative(
            minimum_improvement,
            "minimum_improvement",
        )
        self._validate_positive(
            relative_change_epsilon,
            "relative_change_epsilon",
        )

        self.minimum_confidence = minimum_confidence
        self.minimum_validation_score = (
            minimum_validation_score
        )
        self.minimum_improvement = minimum_improvement
        self.relative_change_epsilon = (
            relative_change_epsilon
        )

    @staticmethod
    def _validate_probability(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

        value = float(value)

        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1."
            )

    @staticmethod
    def _validate_nonnegative(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be non-negative."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be non-negative."
            )

        value = float(value)

        if not isfinite(value) or value < 0.0:
            raise ValueError(
                f"{name} must be non-negative."
            )

    @staticmethod
    def _validate_positive(
        value: float,
        name: str,
    ) -> None:
        if isinstance(value, bool):
            raise ValueError(
                f"{name} must be positive."
            )

        if not isinstance(value, (int, float)):
            raise ValueError(
                f"{name} must be positive."
            )

        value = float(value)

        if not isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be positive."
            )

    @staticmethod
    def _validate_execution(
        execution: ExecutionResult,
    ) -> None:
        if not isinstance(execution, ExecutionResult):
            raise TypeError(
                "execution must be an ExecutionResult."
            )

    @staticmethod
    def _validate_metric_mapping(
        metrics: Mapping[str, object],
        field_name: str,
    ) -> dict[str, float]:
        if not isinstance(metrics, Mapping):
            raise TypeError(
                f"{field_name} must be a mapping."
            )

        result: dict[str, float] = {}

        for name, value in metrics.items():
            if (
                not isinstance(name, str)
                or not name.strip()
            ):
                raise ValueError(
                    f"{field_name} contains an invalid metric name."
                )

            if isinstance(value, bool):
                raise ValueError(
                    f"{field_name}[{name!r}] must be finite."
                )

            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{field_name}[{name!r}] must be finite."
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"{field_name}[{name!r}] must be finite."
                )

            result[name] = value

        return result

    def _build_metrics(
        self,
        baseline: Mapping[str, object],
        observed: Mapping[str, object],
    ) -> tuple[EvaluationMetric, ...]:
        baseline_values = self._validate_metric_mapping(
            baseline,
            "baseline",
        )

        observed_values = self._validate_metric_mapping(
            observed,
            "observed",
        )

        if not baseline_values:
            raise ValueError(
                "baseline cannot be empty."
            )

        if not observed_values:
            raise ValueError(
                "observed cannot be empty."
            )

        common_names = sorted(
            set(baseline_values)
            & set(observed_values)
        )

        if not common_names:
            raise ValueError(
                "baseline and observed must share at least one metric."
            )

        metrics: list[EvaluationMetric] = []

        for name in common_names:
            baseline_value = baseline_values[name]
            observed_value = observed_values[name]

            improvement = (
                observed_value - baseline_value
            )

            if (
                abs(baseline_value)
                <= self.relative_change_epsilon
            ):
                relative_change = (
                    improvement
                    if abs(observed_value)
                    > self.relative_change_epsilon
                    else 0.0
                )
            else:
                relative_change = (
                    improvement / abs(baseline_value)
                )

            metrics.append(
                EvaluationMetric(
                    name=name,
                    baseline=baseline_value,
                    observed=observed_value,
                    improvement=improvement,
                    relative_change=relative_change,
                )
            )

        return tuple(metrics)

    @staticmethod
    def _average_improvement(
        metrics: Sequence[EvaluationMetric],
    ) -> float:
        if not metrics:
            return 0.0

        return sum(
            metric.improvement
            for metric in metrics
        ) / len(metrics)

    def evaluate(
        self,
        execution: ExecutionResult,
        *,
        baseline: Mapping[str, object],
        confidence: float,
        validation_score: float,
        evidence: Sequence[str] = (),
    ) -> ExperimentEvaluation:
        self._validate_execution(execution)

        self._validate_probability(
            confidence,
            "confidence",
        )
        self._validate_probability(
            validation_score,
            "validation_score",
        )

        if isinstance(evidence, (str, bytes)):
            raise ValueError(
                "evidence must be a sequence of strings."
            )

        evidence_tuple = tuple(evidence)

        if any(
            not isinstance(item, str) or not item.strip()
            for item in evidence_tuple
        ):
            raise ValueError(
                "evidence must contain non-empty strings."
            )

        if (
            execution.metadata.status
            != ExecutionStatus.COMPLETED
        ):
            return ExperimentEvaluation(
                execution_id=execution.metadata.execution_id,
                experiment_id=execution.metadata.experiment_id,
                status=EvaluationStatus.FAILED,
                recommendation=Recommendation.REPEAT,
                metrics=(),
                confidence=confidence,
                validation_score=validation_score,
                evidence=evidence_tuple,
                rationale=(
                    "The experiment did not complete successfully "
                    "and therefore cannot be validated."
                ),
            )

        observed = execution.output

        metrics = self._build_metrics(
            baseline=baseline,
            observed=observed,
        )

        average_improvement = self._average_improvement(
            metrics
        )

        validation_passed = (
            validation_score
            >= self.minimum_validation_score
        )

        confidence_passed = (
            confidence
            >= self.minimum_confidence
        )

        # A zero default threshold means that an experiment must
        # demonstrate actual positive improvement. For an explicitly
        # positive threshold, the threshold itself is inclusive.
        if self.minimum_improvement == 0.0:
            improvement_passed = (
                average_improvement > 0.0
            )
        else:
            improvement_passed = (
                average_improvement
                >= self.minimum_improvement
            )

        if (
            validation_passed
            and confidence_passed
            and improvement_passed
        ):
            status = EvaluationStatus.VALIDATED
            recommendation = Recommendation.ACCEPT
            rationale = (
                "The experiment achieved the required validation "
                "score and confidence while meeting the minimum "
                "performance improvement threshold."
            )

        elif not validation_passed:
            status = EvaluationStatus.INCONCLUSIVE
            recommendation = Recommendation.REPEAT
            rationale = (
                "Validation quality is below the required threshold; "
                "the experiment should be repeated with stronger "
                "validation."
            )

        elif not confidence_passed:
            status = EvaluationStatus.INCONCLUSIVE
            recommendation = Recommendation.INVESTIGATE
            rationale = (
                "The experiment produced insufficient confidence "
                "to support a reliable conclusion."
            )

        else:
            status = EvaluationStatus.CONTRADICTED
            recommendation = Recommendation.REJECT
            rationale = (
                "The observed result does not meet the required "
                "performance improvement threshold."
            )

        return ExperimentEvaluation(
            execution_id=execution.metadata.execution_id,
            experiment_id=execution.metadata.experiment_id,
            status=status,
            recommendation=recommendation,
            metrics=metrics,
            confidence=float(confidence),
            validation_score=float(validation_score),
            evidence=evidence_tuple,
            rationale=rationale,
        )


__all__ = [
    "EvaluationMetric",
    "EvaluationStatus",
    "ExperimentEvaluation",
    "ExperimentEvaluator",
    "Recommendation",
]