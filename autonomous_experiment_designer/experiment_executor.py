from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Mapping, Sequence
from uuid import uuid4

from autonomous_experiment_designer.experiment_designer import (
    ExperimentPlan,
)


_MISSING = object()


class ExecutionStatus(str, Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class ExecutionMetadata:
    execution_id: str
    experiment_id: str
    started_at: datetime
    completed_at: datetime | None
    status: ExecutionStatus
    datasets: tuple[str, ...]
    agents: tuple[str, ...]
    models: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionResult:
    metadata: ExecutionMetadata
    success: bool
    output: Mapping[str, object]
    error: str | None


class ExperimentExecutor:
    """
    Controlled execution layer for autonomous experiments.

    The executor enforces the human approval gate before execution.
    It does not select experiments itself and does not modify the
    supplied ExperimentPlan.
    """

    def __init__(
        self,
        *,
        dataset_registry: Sequence[str] = (),
        agent_registry: Sequence[str] = (),
        model_registry: Sequence[str] = (),
    ) -> None:
        self._datasets = self._validate_registry(
            dataset_registry,
            "dataset_registry",
        )
        self._agents = self._validate_registry(
            agent_registry,
            "agent_registry",
        )
        self._models = self._validate_registry(
            model_registry,
            "model_registry",
        )

    @staticmethod
    def _validate_registry(
        values: Sequence[str],
        field_name: str,
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"{field_name} must be a sequence of names."
            )

        result = tuple(values)

        if any(
            not isinstance(value, str) or not value.strip()
            for value in result
        ):
            raise ValueError(
                f"{field_name} must contain non-empty strings."
            )

        if len(set(result)) != len(result):
            raise ValueError(
                f"{field_name} cannot contain duplicates."
            )

        return result

    @staticmethod
    def _validate_plan(
        plan: ExperimentPlan,
    ) -> None:
        if not isinstance(plan, ExperimentPlan):
            raise TypeError(
                "plan must be an ExperimentPlan."
            )

    @staticmethod
    def _generate_execution_id() -> str:
        return f"EXEC-{uuid4().hex}"

    @staticmethod
    def _validate_required_execution_id(
        execution_id: object,
    ) -> str:
        if (
            not isinstance(execution_id, str)
            or not execution_id.strip()
        ):
            raise ValueError(
                "execution_id must be a non-empty string."
            )

        return execution_id

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _validate_resources(
        self,
        plan: ExperimentPlan,
    ) -> None:
        if self._datasets:
            missing = set(plan.datasets) - set(self._datasets)

            if missing:
                raise ValueError(
                    "Plan references unavailable datasets: "
                    + ", ".join(sorted(missing))
                )

        if self._agents:
            missing = set(plan.agents) - set(self._agents)

            if missing:
                raise ValueError(
                    "Plan references unavailable agents: "
                    + ", ".join(sorted(missing))
                )

        if self._models:
            missing = set(plan.models) - set(self._models)

            if missing:
                raise ValueError(
                    "Plan references unavailable models: "
                    + ", ".join(sorted(missing))
                )

    def prepare(
        self,
        plan: ExperimentPlan,
        *,
        execution_id: object = _MISSING,
    ) -> ExecutionResult:
        self._validate_plan(plan)

        if execution_id is _MISSING:
            resolved_execution_id = self._generate_execution_id()
        else:
            resolved_execution_id = (
                self._validate_required_execution_id(
                    execution_id
                )
            )

        self._validate_resources(plan)

        now = self._utc_now()

        metadata = ExecutionMetadata(
            execution_id=resolved_execution_id,
            experiment_id=plan.experiment_id,
            started_at=now,
            completed_at=None,
            status=ExecutionStatus.PENDING_APPROVAL,
            datasets=plan.datasets,
            agents=plan.agents,
            models=plan.models,
        )

        return ExecutionResult(
            metadata=metadata,
            success=False,
            output={},
            error=None,
        )

    def execute(
        self,
        plan: ExperimentPlan,
        *,
        approved: bool,
        runner: Callable[
            [ExperimentPlan],
            Mapping[str, object],
        ],
        execution_id: object,
    ) -> ExecutionResult:
        self._validate_plan(plan)

        if not isinstance(approved, bool):
            raise TypeError(
                "approved must be a boolean."
            )

        if not callable(runner):
            raise TypeError(
                "runner must be callable."
            )

        resolved_execution_id = (
            self._validate_required_execution_id(
                execution_id
            )
        )

        self._validate_resources(plan)

        started_at = self._utc_now()

        if not approved:
            metadata = ExecutionMetadata(
                execution_id=resolved_execution_id,
                experiment_id=plan.experiment_id,
                started_at=started_at,
                completed_at=started_at,
                status=ExecutionStatus.REJECTED,
                datasets=plan.datasets,
                agents=plan.agents,
                models=plan.models,
            )

            return ExecutionResult(
                metadata=metadata,
                success=False,
                output={},
                error="Human approval was not granted.",
            )

        try:
            output = runner(plan)

            if not isinstance(output, Mapping):
                raise TypeError(
                    "runner must return a mapping."
                )

        except Exception as exc:
            completed_at = self._utc_now()

            failed_metadata = ExecutionMetadata(
                execution_id=resolved_execution_id,
                experiment_id=plan.experiment_id,
                started_at=started_at,
                completed_at=completed_at,
                status=ExecutionStatus.FAILED,
                datasets=plan.datasets,
                agents=plan.agents,
                models=plan.models,
            )

            return ExecutionResult(
                metadata=failed_metadata,
                success=False,
                output={},
                error=str(exc),
            )

        completed_at = self._utc_now()

        completed_metadata = ExecutionMetadata(
            execution_id=resolved_execution_id,
            experiment_id=plan.experiment_id,
            started_at=started_at,
            completed_at=completed_at,
            status=ExecutionStatus.COMPLETED,
            datasets=plan.datasets,
            agents=plan.agents,
            models=plan.models,
        )

        return ExecutionResult(
            metadata=completed_metadata,
            success=True,
            output=dict(output),
            error=None,
        )


__all__ = [
    "ExecutionMetadata",
    "ExecutionResult",
    "ExecutionStatus",
    "ExperimentExecutor",
]