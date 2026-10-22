from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from production_research_infrastructure.event_bus import (
    EventBus,
    EventPriority,
    EventType,
    ResearchEvent,
)
from production_research_infrastructure.parallel_executor import (
    ExecutionStatus,
    ExecutionTask,
    ParallelExecutionReport,
    ParallelExecutor,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ResearchExecutionError(RuntimeError):
    """Base exception for research execution failures."""


class ResearchExecutionValidationError(ResearchExecutionError):
    """Raised when a research execution request is invalid."""


class ResearchRunStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


@dataclass
class ResearchExecutionRequest:
    name: str
    tasks: List[ExecutionTask]
    parallel: bool = True
    correlation_id: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    request_id: str = field(
        default_factory=lambda: generate_id("request")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
        ):
            raise ResearchExecutionValidationError(
                "name must be a non-empty string"
            )

        self.name = self.name.strip()

        if not isinstance(self.tasks, list):
            raise TypeError("tasks must be a list")

        for task in self.tasks:
            if not isinstance(task, ExecutionTask):
                raise TypeError(
                    "every task must be an ExecutionTask"
                )

        if not isinstance(self.parallel, bool):
            raise TypeError("parallel must be a boolean")

        if not isinstance(self.priority, EventPriority):
            raise TypeError(
                "priority must be an EventPriority"
            )

        if self.correlation_id is not None:
            if (
                not isinstance(self.correlation_id, str)
                or not self.correlation_id.strip()
            ):
                raise ResearchExecutionValidationError(
                    "correlation_id must be a non-empty string"
                )

            self.correlation_id = (
                self.correlation_id.strip()
            )

        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )


@dataclass
class ResearchRun:
    run_id: str
    request_id: str
    name: str
    status: ResearchRunStatus
    execution_report: Optional[
        ParallelExecutionReport
    ] = None
    correlation_id: Optional[str] = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault(
            "real_money_execution",
            False,
        )

    @property
    def task_count(self) -> int:
        if self.execution_report is None:
            return 0

        return self.execution_report.task_count

    @property
    def success_count(self) -> int:
        if self.execution_report is None:
            return 0

        return self.execution_report.success_count

    @property
    def failure_count(self) -> int:
        if self.execution_report is None:
            return 0

        return self.execution_report.failure_count

    @property
    def timeout_count(self) -> int:
        if self.execution_report is None:
            return 0

        return self.execution_report.timeout_count

    @property
    def duration_seconds(self) -> float:
        end = self.completed_at or utc_now()

        return max(
            0.0,
            (end - self.started_at).total_seconds(),
        )

    @property
    def successful(self) -> bool:
        return self.status == ResearchRunStatus.COMPLETED


class ResearchExecutionEngine:
    """
    Event-driven research execution engine.

    Connects:
    - EventBus
    - ParallelExecutor
    - typed research execution requests
    - research run lifecycle
    - execution history
    - correlation and run IDs
    - failure/timeout propagation
    - audit history

    This engine coordinates research workloads only.
    It cannot place broker orders or authorize real-money execution.
    """

    VERSION = "v6.0.1"

    def __init__(
        self,
        *,
        event_bus: Optional[EventBus] = None,
        parallel_executor: Optional[
            ParallelExecutor
        ] = None,
        max_concurrency: int = 9,
    ) -> None:
        if event_bus is not None and not isinstance(
            event_bus,
            EventBus,
        ):
            raise TypeError(
                "event_bus must be an EventBus"
            )

        if (
            parallel_executor is not None
            and not isinstance(
                parallel_executor,
                ParallelExecutor,
            )
        ):
            raise TypeError(
                "parallel_executor must be a ParallelExecutor"
            )

        self.event_bus = event_bus or EventBus()

        self.parallel_executor = (
            parallel_executor
            or ParallelExecutor(
                max_concurrency=max_concurrency
            )
        )

        self._runs: Dict[str, ResearchRun] = {}
        self._run_order: List[str] = []
        self._audit_events: List[
            Dict[str, Any]
        ] = []

    # ========================================================
    # Audit
    # ========================================================

    def _audit(
        self,
        event: str,
        **details: Any,
    ) -> None:
        self._audit_events.append(
            {
                "event": event,
                "created_at": utc_now(),
                "details": details,
            }
        )

    # ========================================================
    # Status mapping
    # ========================================================

    @staticmethod
    def _map_execution_status(
        status: ExecutionStatus,
    ) -> ResearchRunStatus:
        mapping = {
            ExecutionStatus.COMPLETED: (
                ResearchRunStatus.COMPLETED
            ),
            ExecutionStatus.PARTIAL: (
                ResearchRunStatus.PARTIAL
            ),
            ExecutionStatus.FAILED: (
                ResearchRunStatus.FAILED
            ),
            ExecutionStatus.TIMED_OUT: (
                ResearchRunStatus.TIMED_OUT
            ),
            ExecutionStatus.PENDING: (
                ResearchRunStatus.CREATED
            ),
            ExecutionStatus.RUNNING: (
                ResearchRunStatus.RUNNING
            ),
        }

        return mapping[status]

    # ========================================================
    # Event publishing
    # ========================================================

    async def _publish(
        self,
        event_type: EventType,
        *,
        run: ResearchRun,
        payload: Optional[Dict[str, Any]] = None,
        priority: EventPriority = EventPriority.NORMAL,
        causation_id: Optional[str] = None,
    ) -> ResearchEvent:
        event = ResearchEvent(
            event_type=event_type,
            payload=payload or {},
            source="research_execution_engine",
            run_id=run.run_id,
            correlation_id=run.correlation_id,
            causation_id=causation_id,
            priority=priority,
            metadata={
                "request_id": run.request_id,
                "engine_version": self.VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

        await self.event_bus.publish_async(
            event,
            parallel=True,
        )

        return event

    # ========================================================
    # Execution
    # ========================================================

    async def execute_async(
        self,
        request: ResearchExecutionRequest,
    ) -> ResearchRun:
        if not isinstance(
            request,
            ResearchExecutionRequest,
        ):
            raise TypeError(
                "request must be a ResearchExecutionRequest"
            )

        run_id = generate_id("run")

        correlation_id = (
            request.correlation_id
            or run_id
        )

        run = ResearchRun(
            run_id=run_id,
            request_id=request.request_id,
            name=request.name,
            status=ResearchRunStatus.CREATED,
            correlation_id=correlation_id,
            metadata={
                "parallel": request.parallel,
                "priority": request.priority.value,
                "engine_version": self.VERSION,
            },
        )

        self._runs[run_id] = run
        self._run_order.append(run_id)

        self._audit(
            "RUN_CREATED",
            run_id=run.run_id,
            request_id=run.request_id,
            name=run.name,
            task_count=len(request.tasks),
            correlation_id=correlation_id,
        )

        requested_event = await self._publish(
            EventType.RESEARCH_REQUESTED,
            run=run,
            payload={
                "name": request.name,
                "task_count": len(request.tasks),
                "parallel": request.parallel,
            },
            priority=request.priority,
        )

        run.status = ResearchRunStatus.RUNNING

        self._audit(
            "RUN_STARTED",
            run_id=run.run_id,
            request_id=run.request_id,
        )

        await self._publish(
            EventType.RESEARCH_STARTED,
            run=run,
            payload={
                "name": run.name,
                "task_count": len(request.tasks),
            },
            priority=request.priority,
            causation_id=requested_event.event_id,
        )

        try:
            report = (
                await self.parallel_executor.execute_async(
                    request.tasks,
                    parallel=request.parallel,
                )
            )

            run.execution_report = report
            run.status = self._map_execution_status(
                report.status
            )
            run.completed_at = utc_now()

            if run.status == ResearchRunStatus.COMPLETED:
                final_event_type = (
                    EventType.RESEARCH_COMPLETED
                )
            else:
                final_event_type = (
                    EventType.RESEARCH_FAILED
                )

            await self._publish(
                final_event_type,
                run=run,
                payload={
                    "name": run.name,
                    "status": run.status.value,
                    "execution_id": (
                        report.execution_id
                    ),
                    "task_count": (
                        report.task_count
                    ),
                    "success_count": (
                        report.success_count
                    ),
                    "failure_count": (
                        report.failure_count
                    ),
                    "timeout_count": (
                        report.timeout_count
                    ),
                },
                priority=request.priority,
            )

            self._audit(
                "RUN_COMPLETED",
                run_id=run.run_id,
                request_id=run.request_id,
                status=run.status.value,
                execution_id=report.execution_id,
                task_count=report.task_count,
                success_count=report.success_count,
                failure_count=report.failure_count,
                timeout_count=report.timeout_count,
            )

            return run

        except Exception as exc:
            run.status = ResearchRunStatus.FAILED
            run.completed_at = utc_now()

            run.metadata["execution_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            await self._publish(
                EventType.RESEARCH_FAILED,
                run=run,
                payload={
                    "name": run.name,
                    "status": run.status.value,
                    "error": (
                        run.metadata[
                            "execution_error"
                        ]
                    ),
                },
                priority=EventPriority.CRITICAL,
            )

            self._audit(
                "RUN_FAILED",
                run_id=run.run_id,
                request_id=run.request_id,
                error=(
                    run.metadata[
                        "execution_error"
                    ]
                ),
            )

            return run

    def execute(
        self,
        request: ResearchExecutionRequest,
    ) -> ResearchRun:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.execute_async(request)
            )

        raise ResearchExecutionError(
            "execute() cannot be called from an active "
            "asyncio event loop; use "
            "await execute_async() instead"
        )

    # ========================================================
    # Run retrieval
    # ========================================================

    def get_run(
        self,
        run_id: str,
    ) -> Optional[ResearchRun]:
        if (
            not isinstance(run_id, str)
            or not run_id.strip()
        ):
            raise ResearchExecutionValidationError(
                "run_id must be a non-empty string"
            )

        return self._runs.get(
            run_id.strip()
        )

    def runs(self) -> List[ResearchRun]:
        return [
            self._runs[run_id]
            for run_id in self._run_order
        ]

    def runs_by_status(
        self,
        status: ResearchRunStatus,
    ) -> List[ResearchRun]:
        if not isinstance(
            status,
            ResearchRunStatus,
        ):
            raise TypeError(
                "status must be a ResearchRunStatus"
            )

        return [
            run
            for run in self.runs()
            if run.status == status
        ]

    def runs_for_correlation(
        self,
        correlation_id: str,
    ) -> List[ResearchRun]:
        if (
            not isinstance(
                correlation_id,
                str,
            )
            or not correlation_id.strip()
        ):
            raise ResearchExecutionValidationError(
                "correlation_id must be a "
                "non-empty string"
            )

        correlation_id = correlation_id.strip()

        return [
            run
            for run in self.runs()
            if (
                run.correlation_id
                == correlation_id
            )
        ]

    # ========================================================
    # Reporting
    # ========================================================

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item[
                    "created_at"
                ],
                "details": dict(
                    item["details"]
                ),
            }
            for item in self._audit_events
        ]

    def summary(
        self,
    ) -> Dict[str, Any]:
        status_counts = {
            status.value: 0
            for status in ResearchRunStatus
        }

        total_tasks = 0
        successful_tasks = 0
        failed_tasks = 0
        timed_out_tasks = 0

        for run in self.runs():
            status_counts[
                run.status.value
            ] += 1

            total_tasks += run.task_count
            successful_tasks += (
                run.success_count
            )
            failed_tasks += (
                run.failure_count
            )
            timed_out_tasks += (
                run.timeout_count
            )

        return {
            "component": (
                "research_execution_engine"
            ),
            "version": self.VERSION,
            "runs": len(self._runs),
            "status_counts": status_counts,
            "tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "timed_out_tasks": timed_out_tasks,
            "event_bus_events": len(
                self.event_bus.event_history()
            ),
            "parallel_executions": len(
                self.parallel_executor
                .execution_history()
            ),
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def run_report(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        run = self.get_run(run_id)

        if run is None:
            raise ResearchExecutionValidationError(
                f"Research run not found: {run_id}"
            )

        execution_id = None

        if run.execution_report is not None:
            execution_id = (
                run.execution_report.execution_id
            )

        return {
            "run_id": run.run_id,
            "request_id": run.request_id,
            "name": run.name,
            "status": run.status.value,
            "correlation_id": (
                run.correlation_id
            ),
            "execution_id": execution_id,
            "task_count": run.task_count,
            "success_count": run.success_count,
            "failure_count": run.failure_count,
            "timeout_count": run.timeout_count,
            "duration_seconds": (
                run.duration_seconds
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def clear_history(self) -> None:
        self._runs.clear()
        self._run_order.clear()

        self._audit(
            "RUN_HISTORY_CLEARED"
        )