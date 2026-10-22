from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


from autonomous_experiment_designer.experiment_evaluator import (
    EvaluationStatus,
    ExperimentEvaluation,
    Recommendation,
)


class ResearchTaskType(str, Enum):
    REPEAT_EXPERIMENT = "REPEAT_EXPERIMENT"
    INVESTIGATE_CONFIDENCE = "INVESTIGATE_CONFIDENCE"
    IMPROVE_VALIDATION = "IMPROVE_VALIDATION"
    INVESTIGATE_PERFORMANCE = "INVESTIGATE_PERFORMANCE"
    FOLLOW_UP_RESEARCH = "FOLLOW_UP_RESEARCH"


class ResearchTaskPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    experiment_id: str
    task_type: ResearchTaskType
    priority: ResearchTaskPriority
    objective: str
    rationale: str
    required_approval: bool
    source_evidence: tuple[str, ...]


class ResearchTaskGenerator:
    """
    Converts experiment evaluations into the next research tasks.

    The generator does not execute experiments and does not bypass
    human approval gates.
    """

    def __init__(
        self,
        *,
        default_evidence: Sequence[str] = (),
    ) -> None:
        self._default_evidence = self._validate_evidence(
            default_evidence
        )

    @staticmethod
    def _validate_evidence(
        evidence: object,
    ) -> tuple[str, ...]:
        if evidence is None:
            raise ValueError(
                "evidence must be a sequence of strings."
            )

        if isinstance(evidence, (str, bytes)):
            raise ValueError(
                "evidence must be a sequence of strings."
            )

        if not isinstance(evidence, Sequence):
            raise ValueError(
                "evidence must be a sequence of strings."
            )

        result = tuple(evidence)

        if any(
            not isinstance(item, str)
            or not item.strip()
            for item in result
        ):
            raise ValueError(
                "evidence must contain non-empty strings."
            )

        if len(set(result)) != len(result):
            raise ValueError(
                "evidence cannot contain duplicates."
            )

        return result

    @staticmethod
    def _validate_evaluation(
        evaluation: ExperimentEvaluation,
    ) -> None:
        if not isinstance(
            evaluation,
            ExperimentEvaluation,
        ):
            raise TypeError(
                "evaluation must be an ExperimentEvaluation."
            )

    @staticmethod
    def _validate_task_id(
        task_id: object,
    ) -> str:
        if (
            not isinstance(task_id, str)
            or not task_id.strip()
        ):
            raise ValueError(
                "task_id must be a non-empty string."
            )

        return task_id

    @staticmethod
    def _priority_for_evaluation(
        evaluation: ExperimentEvaluation,
    ) -> ResearchTaskPriority:
        if evaluation.status == EvaluationStatus.FAILED:
            return ResearchTaskPriority.CRITICAL

        if (
            evaluation.recommendation
            == Recommendation.REPEAT
        ):
            return ResearchTaskPriority.HIGH

        if (
            evaluation.recommendation
            == Recommendation.INVESTIGATE
        ):
            return ResearchTaskPriority.HIGH

        if (
            evaluation.recommendation
            == Recommendation.REJECT
        ):
            return ResearchTaskPriority.MEDIUM

        return ResearchTaskPriority.LOW

    @staticmethod
    def _task_type_for_evaluation(
        evaluation: ExperimentEvaluation,
    ) -> ResearchTaskType:
        if evaluation.status == EvaluationStatus.FAILED:
            return ResearchTaskType.REPEAT_EXPERIMENT

        if (
            evaluation.recommendation
            == Recommendation.REPEAT
        ):
            if evaluation.validation_score < 0.70:
                return (
                    ResearchTaskType.IMPROVE_VALIDATION
                )

            return ResearchTaskType.REPEAT_EXPERIMENT

        if (
            evaluation.recommendation
            == Recommendation.INVESTIGATE
        ):
            return (
                ResearchTaskType.INVESTIGATE_CONFIDENCE
            )

        if (
            evaluation.recommendation
            == Recommendation.REJECT
        ):
            return (
                ResearchTaskType.INVESTIGATE_PERFORMANCE
            )

        return ResearchTaskType.FOLLOW_UP_RESEARCH

    @staticmethod
    def _objective_for_task_type(
        task_type: ResearchTaskType,
    ) -> str:
        objectives = {
            ResearchTaskType.REPEAT_EXPERIMENT: (
                "Repeat the experiment with controlled "
                "conditions and verify whether the result "
                "is reproducible."
            ),
            ResearchTaskType.INVESTIGATE_CONFIDENCE: (
                "Investigate the sources of uncertainty and "
                "collect additional evidence required to "
                "increase confidence."
            ),
            ResearchTaskType.IMPROVE_VALIDATION: (
                "Strengthen the validation methodology using "
                "additional out-of-sample, walk-forward or "
                "stress-testing evidence."
            ),
            ResearchTaskType.INVESTIGATE_PERFORMANCE: (
                "Investigate why the experiment failed to "
                "deliver the required performance improvement "
                "relative to the baseline."
            ),
            ResearchTaskType.FOLLOW_UP_RESEARCH: (
                "Conduct follow-up research to determine whether "
                "the validated finding generalizes to additional "
                "datasets, regimes or assets."
            ),
        }

        return objectives[task_type]

    def generate(
        self,
        evaluation: ExperimentEvaluation,
        *,
        task_id: str,
    ) -> ResearchTask:
        self._validate_evaluation(evaluation)

        resolved_task_id = self._validate_task_id(
            task_id
        )

        task_type = self._task_type_for_evaluation(
            evaluation
        )

        priority = self._priority_for_evaluation(
            evaluation
        )

        objective = self._objective_for_task_type(
            task_type
        )

        evidence = tuple(
            dict.fromkeys(
                (
                    *self._default_evidence,
                    *evaluation.evidence,
                )
            )
        )

        rationale = evaluation.rationale

        return ResearchTask(
            task_id=resolved_task_id,
            experiment_id=evaluation.experiment_id,
            task_type=task_type,
            priority=priority,
            objective=objective,
            rationale=rationale,
            required_approval=True,
            source_evidence=evidence,
        )

    def generate_many(
        self,
        evaluation: ExperimentEvaluation,
        *,
        task_ids: Sequence[str],
    ) -> tuple[ResearchTask, ...]:
        self._validate_evaluation(evaluation)

        if isinstance(task_ids, (str, bytes)):
            raise ValueError(
                "task_ids must be a sequence of strings."
            )

        if task_ids is None:
            raise ValueError(
                "task_ids must be a sequence of strings."
            )

        if not isinstance(task_ids, Sequence):
            raise ValueError(
                "task_ids must be a sequence of strings."
            )

        ids = tuple(task_ids)

        if not ids:
            raise ValueError(
                "task_ids cannot be empty."
            )

        validated_ids = tuple(
            self._validate_task_id(task_id)
            for task_id in ids
        )

        if len(set(validated_ids)) != len(
            validated_ids
        ):
            raise ValueError(
                "task_ids cannot contain duplicates."
            )

        return tuple(
            self.generate(
                evaluation,
                task_id=task_id,
            )
            for task_id in validated_ids
        )


__all__ = [
    "ResearchTask",
    "ResearchTaskGenerator",
    "ResearchTaskPriority",
    "ResearchTaskType",
]