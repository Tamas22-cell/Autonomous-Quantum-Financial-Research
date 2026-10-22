from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class ParallelExecutorError(RuntimeError):
    """Base exception for parallel execution failures."""


class ExecutionValidationError(ParallelExecutorError):
    """Raised when an execution request is invalid."""


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


TaskCallable = Callable[
    [],
    Union[Any, Awaitable[Any]],
]


@dataclass
class ExecutionTask:
    name: str
    callable: TaskCallable
    task_id: str = field(
        default_factory=lambda: generate_id("task")
    )
    agent_name: Optional[str] = None
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ExecutionValidationError(
                "name must be a non-empty string"
            )

        self.name = self.name.strip()

        if not callable(self.callable):
            raise TypeError("callable must be callable")

        if self.agent_name is not None:
            if (
                not isinstance(self.agent_name, str)
                or not self.agent_name.strip()
            ):
                raise ExecutionValidationError(
                    "agent_name must be a non-empty string"
                )

            self.agent_name = self.agent_name.strip()

        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds,
                (int, float),
            ):
                raise TypeError(
                    "timeout_seconds must be numeric"
                )

            if self.timeout_seconds <= 0:
                raise ExecutionValidationError(
                    "timeout_seconds must be greater than zero"
                )

            self.timeout_seconds = float(
                self.timeout_seconds
            )

        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)


@dataclass
class TaskExecutionResult:
    task_id: str
    task_name: str
    success: bool
    status: ExecutionStatus
    result: Any = None
    error: Optional[str] = None
    agent_name: Optional[str] = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (
                self.completed_at
                - self.started_at
            ).total_seconds(),
        )


@dataclass
class ParallelExecutionReport:
    execution_id: str
    status: ExecutionStatus
    task_results: List[TaskExecutionResult]
    started_at: datetime
    completed_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)

    @property
    def task_count(self) -> int:
        return len(self.task_results)

    @property
    def success_count(self) -> int:
        return sum(
            1
            for item in self.task_results
            if item.success
        )

    @property
    def failure_count(self) -> int:
        return sum(
            1
            for item in self.task_results
            if not item.success
        )

    @property
    def timeout_count(self) -> int:
        return sum(
            1
            for item in self.task_results
            if item.status == ExecutionStatus.TIMED_OUT
        )

    @property
    def duration_seconds(self) -> float:
        return max(
            0.0,
            (
                self.completed_at
                - self.started_at
            ).total_seconds(),
        )

    @property
    def successful(self) -> bool:
        return self.status == ExecutionStatus.COMPLETED


class ParallelExecutor:
    """
    Parallel execution engine for research agents and tasks.

    Supports:
    - synchronous and asynchronous callables,
    - parallel and sequential execution,
    - per-task timeouts,
    - failure isolation,
    - deterministic result ordering,
    - execution history,
    - audit trail,
    - concurrency limits.

    This component executes research workloads only.
    It does not authorize broker orders or real-money trading.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 9,
    ) -> None:
        if (
            isinstance(max_concurrency, bool)
            or not isinstance(max_concurrency, int)
        ):
            raise TypeError(
                "max_concurrency must be an integer"
            )

        if max_concurrency < 1:
            raise ExecutionValidationError(
                "max_concurrency must be at least 1"
            )

        self.max_concurrency = max_concurrency

        self._execution_history: List[
            ParallelExecutionReport
        ] = []

        self._audit_events: List[
            Dict[str, Any]
        ] = []

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

    @staticmethod
    def _report_status(
        results: List[TaskExecutionResult],
    ) -> ExecutionStatus:
        if not results:
            return ExecutionStatus.COMPLETED

        success_count = sum(
            1
            for item in results
            if item.success
        )

        timeout_count = sum(
            1
            for item in results
            if item.status == ExecutionStatus.TIMED_OUT
        )

        if success_count == len(results):
            return ExecutionStatus.COMPLETED

        if timeout_count == len(results):
            return ExecutionStatus.TIMED_OUT

        if success_count == 0:
            return ExecutionStatus.FAILED

        return ExecutionStatus.PARTIAL

    async def _call_task(
        self,
        task: ExecutionTask,
    ) -> Any:
        if inspect.iscoroutinefunction(task.callable):
            return await task.callable()

        result = await asyncio.to_thread(
            task.callable
        )

        if inspect.isawaitable(result):
            return await result

        return result

    async def _execute_task(
        self,
        task: ExecutionTask,
        semaphore: asyncio.Semaphore,
    ) -> TaskExecutionResult:
        async with semaphore:
            started_at = utc_now()

            self._audit(
                "TASK_STARTED",
                task_id=task.task_id,
                task_name=task.name,
                agent_name=task.agent_name,
            )

            try:
                if task.timeout_seconds is None:
                    result = await self._call_task(task)
                else:
                    result = await asyncio.wait_for(
                        self._call_task(task),
                        timeout=task.timeout_seconds,
                    )

                execution_result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                    success=True,
                    status=ExecutionStatus.COMPLETED,
                    result=result,
                    started_at=started_at,
                    completed_at=utc_now(),
                )

                self._audit(
                    "TASK_COMPLETED",
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                )

                return execution_result

            except asyncio.TimeoutError:
                execution_result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                    success=False,
                    status=ExecutionStatus.TIMED_OUT,
                    error=(
                        "Task exceeded timeout of "
                        f"{task.timeout_seconds} seconds"
                    ),
                    started_at=started_at,
                    completed_at=utc_now(),
                )

                self._audit(
                    "TASK_TIMED_OUT",
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                    timeout_seconds=task.timeout_seconds,
                )

                return execution_result

            except Exception as exc:
                execution_result = TaskExecutionResult(
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                    success=False,
                    status=ExecutionStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    started_at=started_at,
                    completed_at=utc_now(),
                )

                self._audit(
                    "TASK_FAILED",
                    task_id=task.task_id,
                    task_name=task.name,
                    agent_name=task.agent_name,
                    error=execution_result.error,
                )

                return execution_result

    async def execute_async(
        self,
        tasks: List[ExecutionTask],
        *,
        parallel: bool = True,
    ) -> ParallelExecutionReport:
        if not isinstance(tasks, list):
            raise TypeError(
                "tasks must be a list"
            )

        for task in tasks:
            if not isinstance(task, ExecutionTask):
                raise TypeError(
                    "every task must be an ExecutionTask"
                )

        execution_id = generate_id("execution")
        started_at = utc_now()

        self._audit(
            "EXECUTION_STARTED",
            execution_id=execution_id,
            task_count=len(tasks),
            parallel=parallel,
            max_concurrency=self.max_concurrency,
        )

        semaphore = asyncio.Semaphore(
            self.max_concurrency
        )

        if parallel:
            results = list(
                await asyncio.gather(
                    *[
                        self._execute_task(
                            task,
                            semaphore,
                        )
                        for task in tasks
                    ]
                )
            )
        else:
            results = []

            for task in tasks:
                result = await self._execute_task(
                    task,
                    semaphore,
                )
                results.append(result)

        report = ParallelExecutionReport(
            execution_id=execution_id,
            status=self._report_status(results),
            task_results=results,
            started_at=started_at,
            completed_at=utc_now(),
            metadata={
                "parallel": parallel,
                "max_concurrency": self.max_concurrency,
            },
        )

        self._execution_history.append(report)

        self._audit(
            "EXECUTION_COMPLETED",
            execution_id=execution_id,
            status=report.status.value,
            task_count=report.task_count,
            success_count=report.success_count,
            failure_count=report.failure_count,
            timeout_count=report.timeout_count,
        )

        return report

    def execute(
        self,
        tasks: List[ExecutionTask],
        *,
        parallel: bool = True,
    ) -> ParallelExecutionReport:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.execute_async(
                    tasks,
                    parallel=parallel,
                )
            )

        raise ParallelExecutorError(
            "execute() cannot be called from an active "
            "asyncio event loop; use await execute_async() instead"
        )

    def execution_history(
        self,
    ) -> List[ParallelExecutionReport]:
        return list(
            self._execution_history
        )

    def get_execution(
        self,
        execution_id: str,
    ) -> Optional[ParallelExecutionReport]:
        if (
            not isinstance(execution_id, str)
            or not execution_id.strip()
        ):
            raise ExecutionValidationError(
                "execution_id must be a non-empty string"
            )

        for report in self._execution_history:
            if report.execution_id == execution_id:
                return report

        return None

    def audit_log(
        self,
    ) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item["created_at"],
                "details": dict(item["details"]),
            }
            for item in self._audit_events
        ]

    def summary(
        self,
    ) -> Dict[str, Any]:
        completed = sum(
            1
            for report in self._execution_history
            if report.status == ExecutionStatus.COMPLETED
        )

        partial = sum(
            1
            for report in self._execution_history
            if report.status == ExecutionStatus.PARTIAL
        )

        failed = sum(
            1
            for report in self._execution_history
            if report.status == ExecutionStatus.FAILED
        )

        timed_out = sum(
            1
            for report in self._execution_history
            if report.status == ExecutionStatus.TIMED_OUT
        )

        total_tasks = sum(
            report.task_count
            for report in self._execution_history
        )

        return {
            "component": "parallel_executor",
            "max_concurrency": self.max_concurrency,
            "executions": len(
                self._execution_history
            ),
            "tasks": total_tasks,
            "completed": completed,
            "partial": partial,
            "failed": failed,
            "timed_out": timed_out,
            "audit_events": len(
                self._audit_events
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def clear_history(self) -> None:
        self._execution_history.clear()

        self._audit(
            "HISTORY_CLEARED"
        )