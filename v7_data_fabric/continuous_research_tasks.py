from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple


class ContinuousResearchTaskError(ValueError):
    """Continuous research task validation error."""


class TaskPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class TaskStatus(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    title: str
    objective: str
    priority: TaskPriority
    source_component: str
    status: TaskStatus = TaskStatus.PROPOSED
    evidence: Mapping[str, object] = field(
        default_factory=dict
    )


@dataclass
class ContinuousResearchTaskGenerator:
    """
    v7.0.12 research-task generation service.

    The generator proposes research work from observed signals.
    It does not execute experiments, trade assets, or approve alpha.
    """

    VERSION = "v7.0.12"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    AUTOMATIC_TRADING = False
    AUTOMATIC_ALPHA_APPROVAL = False

    _tasks: Dict[str, ResearchTask] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @staticmethod
    def _validate_text(
        value: str,
        name: str,
    ) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ContinuousResearchTaskError(
                f"{name} must be a non-empty string."
            )

    @staticmethod
    def _validate_priority(
        priority: TaskPriority,
    ) -> None:
        if not isinstance(priority, TaskPriority):
            raise ContinuousResearchTaskError(
                "priority must be a TaskPriority value."
            )

    def generate_task(
        self,
        task_id: str,
        title: str,
        objective: str,
        *,
        priority: TaskPriority = TaskPriority.MEDIUM,
        source_component: str = "research_control_center",
        evidence: Optional[Mapping[str, object]] = None,
    ) -> ResearchTask:
        self._validate_text(task_id, "task_id")
        self._validate_text(title, "title")
        self._validate_text(objective, "objective")
        self._validate_text(
            source_component,
            "source_component",
        )
        self._validate_priority(priority)

        if task_id in self._tasks:
            raise ContinuousResearchTaskError(
                f"Duplicate task ID: {task_id}"
            )

        task = ResearchTask(
            task_id=task_id,
            title=title,
            objective=objective,
            priority=priority,
            source_component=source_component,
            evidence=dict(evidence or {}),
        )

        self._tasks[task_id] = task
        return task

    def generate_from_signal(
        self,
        signal_name: str,
        signal_value: float,
        *,
        source_component: str,
        task_id: Optional[str] = None,
    ) -> ResearchTask:
        self._validate_text(signal_name, "signal_name")
        self._validate_text(
            source_component,
            "source_component",
        )

        if not isinstance(signal_value, (int, float)):
            raise ContinuousResearchTaskError(
                "signal_value must be numeric."
            )

        generated_id = task_id or (
            f"research-{len(self._tasks) + 1:04d}"
        )

        priority = (
            TaskPriority.HIGH
            if abs(signal_value) >= 0.75
            else TaskPriority.MEDIUM
        )

        return self.generate_task(
            generated_id,
            f"Investigate {signal_name}",
            (
                "Evaluate the robustness, causality, and "
                f"out-of-sample behavior of {signal_name}."
            ),
            priority=priority,
            source_component=source_component,
            evidence={
                "signal_name": signal_name,
                "signal_value": signal_value,
            },
        )

    def approve_task(
        self,
        task_id: str,
    ) -> ResearchTask:
        task = self.get_task(task_id)

        if task.status != TaskStatus.PROPOSED:
            raise ContinuousResearchTaskError(
                "Only proposed tasks can be approved."
            )

        approved = ResearchTask(
            task_id=task.task_id,
            title=task.title,
            objective=task.objective,
            priority=task.priority,
            source_component=task.source_component,
            status=TaskStatus.APPROVED,
            evidence=task.evidence,
        )

        self._tasks[task_id] = approved
        return approved

    def complete_task(
        self,
        task_id: str,
        *,
        result: Optional[Mapping[str, object]] = None,
    ) -> ResearchTask:
        task = self.get_task(task_id)

        if task.status != TaskStatus.APPROVED:
            raise ContinuousResearchTaskError(
                "Only approved tasks can be completed."
            )

        evidence = dict(task.evidence)
        if result:
            evidence["result"] = dict(result)

        completed = ResearchTask(
            task_id=task.task_id,
            title=task.title,
            objective=task.objective,
            priority=task.priority,
            source_component=task.source_component,
            status=TaskStatus.COMPLETED,
            evidence=evidence,
        )

        self._tasks[task_id] = completed
        return completed

    def reject_task(
        self,
        task_id: str,
    ) -> ResearchTask:
        task = self.get_task(task_id)

        if task.status in (
            TaskStatus.COMPLETED,
            TaskStatus.REJECTED,
        ):
            raise ContinuousResearchTaskError(
                "Completed or rejected tasks cannot change state."
            )

        rejected = ResearchTask(
            task_id=task.task_id,
            title=task.title,
            objective=task.objective,
            priority=task.priority,
            source_component=task.source_component,
            status=TaskStatus.REJECTED,
            evidence=task.evidence,
        )

        self._tasks[task_id] = rejected
        return rejected

    def get_task(self, task_id: str) -> ResearchTask:
        self._validate_text(task_id, "task_id")

        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ContinuousResearchTaskError(
                f"Unknown task ID: {task_id}"
            ) from exc

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
    ) -> Tuple[ResearchTask, ...]:
        if (
            status is not None
            and not isinstance(status, TaskStatus)
        ):
            raise ContinuousResearchTaskError(
                "status must be a TaskStatus value."
            )

        tasks = tuple(self._tasks.values())

        if status is None:
            return tasks

        return tuple(
            task
            for task in tasks
            if task.status == status
        )

    def summary(self) -> Dict[str, object]:
        return {
            "component": "continuous_research_tasks",
            "version": self.VERSION,
            "task_count": len(self._tasks),
            "proposed_count": len(
                self.list_tasks(TaskStatus.PROPOSED)
            ),
            "approved_count": len(
                self.list_tasks(TaskStatus.APPROVED)
            ),
            "completed_count": len(
                self.list_tasks(TaskStatus.COMPLETED)
            ),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "automatic_trading": self.AUTOMATIC_TRADING,
            "automatic_alpha_approval": (
                self.AUTOMATIC_ALPHA_APPROVAL
            ),
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "task_count": len(self._tasks),
            "task_ids": list(self._tasks),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
        }