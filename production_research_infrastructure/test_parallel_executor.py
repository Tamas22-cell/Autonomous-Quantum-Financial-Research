from __future__ import annotations

import asyncio
import time

import pytest

from production_research_infrastructure.parallel_executor import (
    ExecutionStatus,
    ExecutionTask,
    ExecutionValidationError,
    ParallelExecutor,
    ParallelExecutorError,
    TaskExecutionResult,
)


def make_task(
    name: str = "macro_analysis",
    *,
    result: object = "ok",
    agent_name: str = "macro_agent",
) -> ExecutionTask:
    def worker() -> object:
        return result

    return ExecutionTask(
        name=name,
        callable=worker,
        agent_name=agent_name,
    )


# ============================================================
# ExecutionTask
# ============================================================


def test_task_creation() -> None:
    task = make_task()

    assert task.name == "macro_analysis"
    assert task.task_id.startswith("task_")
    assert task.agent_name == "macro_agent"


def test_task_research_safety() -> None:
    task = make_task()

    assert task.metadata["research_only"] is True
    assert task.metadata["real_money_execution"] is False


def test_task_rejects_empty_name() -> None:
    with pytest.raises(ExecutionValidationError):
        ExecutionTask(
            name=" ",
            callable=lambda: None,
        )


def test_task_trims_name() -> None:
    task = ExecutionTask(
        name="  research_task  ",
        callable=lambda: None,
    )

    assert task.name == "research_task"


def test_task_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        ExecutionTask(
            name="task",
            callable="invalid",  # type: ignore[arg-type]
        )


def test_task_trims_agent_name() -> None:
    task = ExecutionTask(
        name="task",
        callable=lambda: None,
        agent_name="  stock_agent  ",
    )

    assert task.agent_name == "stock_agent"


def test_task_rejects_empty_agent_name() -> None:
    with pytest.raises(ExecutionValidationError):
        ExecutionTask(
            name="task",
            callable=lambda: None,
            agent_name=" ",
        )


def test_task_accepts_timeout() -> None:
    task = ExecutionTask(
        name="task",
        callable=lambda: None,
        timeout_seconds=2,
    )

    assert task.timeout_seconds == 2.0


def test_task_rejects_zero_timeout() -> None:
    with pytest.raises(ExecutionValidationError):
        ExecutionTask(
            name="task",
            callable=lambda: None,
            timeout_seconds=0,
        )


def test_task_rejects_negative_timeout() -> None:
    with pytest.raises(ExecutionValidationError):
        ExecutionTask(
            name="task",
            callable=lambda: None,
            timeout_seconds=-1,
        )


def test_task_rejects_boolean_timeout() -> None:
    with pytest.raises(TypeError):
        ExecutionTask(
            name="task",
            callable=lambda: None,
            timeout_seconds=True,
        )


def test_task_rejects_string_timeout() -> None:
    with pytest.raises(TypeError):
        ExecutionTask(
            name="task",
            callable=lambda: None,
            timeout_seconds="1",  # type: ignore[arg-type]
        )


# ============================================================
# ParallelExecutor construction
# ============================================================


def test_executor_default_concurrency() -> None:
    executor = ParallelExecutor()

    assert executor.max_concurrency == 9


def test_executor_custom_concurrency() -> None:
    executor = ParallelExecutor(
        max_concurrency=3
    )

    assert executor.max_concurrency == 3


def test_executor_rejects_zero_concurrency() -> None:
    with pytest.raises(ExecutionValidationError):
        ParallelExecutor(
            max_concurrency=0
        )


def test_executor_rejects_negative_concurrency() -> None:
    with pytest.raises(ExecutionValidationError):
        ParallelExecutor(
            max_concurrency=-1
        )


def test_executor_rejects_boolean_concurrency() -> None:
    with pytest.raises(TypeError):
        ParallelExecutor(
            max_concurrency=True
        )


def test_executor_rejects_float_concurrency() -> None:
    with pytest.raises(TypeError):
        ParallelExecutor(
            max_concurrency=1.5  # type: ignore[arg-type]
        )


def test_executor_starts_with_empty_history() -> None:
    executor = ParallelExecutor()

    assert executor.execution_history() == []
    assert executor.audit_log() == []


# ============================================================
# Basic synchronous execution
# ============================================================


def test_execute_single_sync_task() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert report.status == ExecutionStatus.COMPLETED
    assert report.task_count == 1
    assert report.success_count == 1
    assert report.failure_count == 0


def test_sync_task_result_preserved() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [
            make_task(
                result={
                    "signal": "HOLD",
                    "confidence": 0.75,
                }
            )
        ]
    )

    assert report.task_results[0].result == {
        "signal": "HOLD",
        "confidence": 0.75,
    }


def test_task_result_contains_agent_name() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [
            make_task(
                agent_name="risk_agent"
            )
        ]
    )

    assert (
        report.task_results[0].agent_name
        == "risk_agent"
    )


def test_successful_task_status_completed() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    result = report.task_results[0]

    assert result.success is True
    assert result.status == ExecutionStatus.COMPLETED
    assert result.error is None


def test_execution_id_created() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert report.execution_id.startswith(
        "execution_"
    )


def test_execution_report_research_safety() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert report.metadata["research_only"] is True
    assert report.metadata["real_money_execution"] is False


def test_task_result_research_safety() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    result = report.task_results[0]

    assert result.metadata["research_only"] is True
    assert result.metadata["real_money_execution"] is False


# ============================================================
# Empty execution
# ============================================================


def test_empty_task_list_completes() -> None:
    executor = ParallelExecutor()

    report = executor.execute([])

    assert report.status == ExecutionStatus.COMPLETED
    assert report.task_count == 0
    assert report.success_count == 0
    assert report.failure_count == 0


# ============================================================
# Validation
# ============================================================


def test_execute_requires_list() -> None:
    executor = ParallelExecutor()

    with pytest.raises(TypeError):
        executor.execute(
            make_task()  # type: ignore[arg-type]
        )


def test_execute_rejects_invalid_task_item() -> None:
    executor = ParallelExecutor()

    with pytest.raises(TypeError):
        executor.execute(
            ["invalid"]  # type: ignore[list-item]
        )


# ============================================================
# Failure isolation
# ============================================================


def test_failed_task_is_captured() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError(
            "agent failed"
        )

    task = ExecutionTask(
        name="broken_agent",
        callable=broken,
    )

    report = executor.execute([task])

    assert report.status == ExecutionStatus.FAILED
    assert report.failure_count == 1
    assert report.task_results[0].success is False
    assert "RuntimeError" in (
        report.task_results[0].error or ""
    )


def test_failure_does_not_stop_other_tasks() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError("failure")

    bad_task = ExecutionTask(
        name="bad",
        callable=broken,
    )

    good_task = make_task(
        name="good",
        result="healthy",
    )

    report = executor.execute(
        [bad_task, good_task]
    )

    assert report.status == ExecutionStatus.PARTIAL
    assert report.success_count == 1
    assert report.failure_count == 1


def test_all_failures_mark_report_failed() -> None:
    executor = ParallelExecutor()

    def broken_one() -> None:
        raise RuntimeError("one")

    def broken_two() -> None:
        raise ValueError("two")

    report = executor.execute(
        [
            ExecutionTask(
                name="one",
                callable=broken_one,
            ),
            ExecutionTask(
                name="two",
                callable=broken_two,
            ),
        ]
    )

    assert report.status == ExecutionStatus.FAILED
    assert report.success_count == 0
    assert report.failure_count == 2


# ============================================================
# Async execution
# ============================================================


def test_execute_async_callable() -> None:
    executor = ParallelExecutor()

    async def worker() -> str:
        await asyncio.sleep(0)
        return "async-result"

    task = ExecutionTask(
        name="async_task",
        callable=worker,
    )

    report = asyncio.run(
        executor.execute_async([task])
    )

    assert report.status == ExecutionStatus.COMPLETED
    assert report.task_results[0].result == "async-result"


def test_sync_callable_returning_awaitable() -> None:
    executor = ParallelExecutor()

    async def inner() -> str:
        await asyncio.sleep(0)
        return "awaited"

    def worker():
        return inner()

    task = ExecutionTask(
        name="awaitable_result",
        callable=worker,
    )

    report = executor.execute([task])

    assert report.task_results[0].result == "awaited"


def test_execute_rejected_inside_active_loop() -> None:
    executor = ParallelExecutor()

    async def scenario() -> None:
        with pytest.raises(
            ParallelExecutorError
        ):
            executor.execute(
                [make_task()]
            )

    asyncio.run(scenario())


# ============================================================
# Parallel execution
# ============================================================


def test_multiple_parallel_tasks() -> None:
    executor = ParallelExecutor(
        max_concurrency=3
    )

    tasks = [
        make_task(
            name="macro",
            result="macro-result",
            agent_name="macro_agent",
        ),
        make_task(
            name="stock",
            result="stock-result",
            agent_name="stock_agent",
        ),
        make_task(
            name="crypto",
            result="crypto-result",
            agent_name="crypto_agent",
        ),
    ]

    report = executor.execute(
        tasks,
        parallel=True,
    )

    assert report.status == ExecutionStatus.COMPLETED
    assert report.task_count == 3
    assert report.success_count == 3


def test_parallel_result_order_matches_task_order() -> None:
    executor = ParallelExecutor()

    tasks = [
        make_task(
            name="one",
            result=1,
        ),
        make_task(
            name="two",
            result=2,
        ),
        make_task(
            name="three",
            result=3,
        ),
    ]

    report = executor.execute(
        tasks,
        parallel=True,
    )

    assert [
        item.result
        for item in report.task_results
    ] == [1, 2, 3]


def test_parallel_async_tasks_run_concurrently() -> None:
    executor = ParallelExecutor(
        max_concurrency=3
    )

    async def worker() -> str:
        await asyncio.sleep(0.05)
        return "done"

    tasks = [
        ExecutionTask(
            name=f"task_{index}",
            callable=worker,
        )
        for index in range(3)
    ]

    started = time.perf_counter()

    report = executor.execute(
        tasks,
        parallel=True,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    assert report.success_count == 3
    assert elapsed < 0.13


def test_sequential_execution_preserves_order() -> None:
    executor = ParallelExecutor()
    order = []

    def first() -> None:
        order.append("first")

    def second() -> None:
        order.append("second")

    report = executor.execute(
        [
            ExecutionTask(
                name="first",
                callable=first,
            ),
            ExecutionTask(
                name="second",
                callable=second,
            ),
        ],
        parallel=False,
    )

    assert order == [
        "first",
        "second",
    ]
    assert report.success_count == 2


def test_concurrency_limit_is_respected() -> None:
    executor = ParallelExecutor(
        max_concurrency=2
    )

    active = 0
    maximum_active = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal active
        nonlocal maximum_active

        async with lock:
            active += 1
            maximum_active = max(
                maximum_active,
                active,
            )

        await asyncio.sleep(0.02)

        async with lock:
            active -= 1

    tasks = [
        ExecutionTask(
            name=f"task_{index}",
            callable=worker,
        )
        for index in range(6)
    ]

    report = asyncio.run(
        executor.execute_async(
            tasks,
            parallel=True,
        )
    )

    assert report.success_count == 6
    assert maximum_active <= 2


# ============================================================
# Timeout handling
# ============================================================


def test_async_task_timeout() -> None:
    executor = ParallelExecutor()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    task = ExecutionTask(
        name="slow",
        callable=slow,
        timeout_seconds=0.01,
    )

    report = executor.execute([task])

    assert report.status == ExecutionStatus.TIMED_OUT
    assert report.timeout_count == 1
    assert report.failure_count == 1


def test_timeout_error_message() -> None:
    executor = ParallelExecutor()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    task = ExecutionTask(
        name="slow",
        callable=slow,
        timeout_seconds=0.01,
    )

    report = executor.execute([task])

    assert "timeout" in (
        report.task_results[0].error or ""
    ).lower()


def test_timeout_and_success_produce_partial() -> None:
    executor = ParallelExecutor()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    slow_task = ExecutionTask(
        name="slow",
        callable=slow,
        timeout_seconds=0.01,
    )

    good_task = make_task(
        name="good",
        result="ok",
    )

    report = executor.execute(
        [slow_task, good_task]
    )

    assert report.status == ExecutionStatus.PARTIAL
    assert report.success_count == 1
    assert report.timeout_count == 1


# ============================================================
# History
# ============================================================


def test_execution_saved_to_history() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    history = executor.execution_history()

    assert len(history) == 1
    assert history[0] is report


def test_get_execution() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    result = executor.get_execution(
        report.execution_id
    )

    assert result is report


def test_get_missing_execution_returns_none() -> None:
    executor = ParallelExecutor()

    assert (
        executor.get_execution(
            "execution_missing"
        )
        is None
    )


def test_get_execution_rejects_empty_id() -> None:
    executor = ParallelExecutor()

    with pytest.raises(
        ExecutionValidationError
    ):
        executor.get_execution("")


def test_clear_history() -> None:
    executor = ParallelExecutor()

    executor.execute(
        [make_task()]
    )

    executor.clear_history()

    assert executor.execution_history() == []


# ============================================================
# Audit
# ============================================================


def test_execution_creates_audit_events() -> None:
    executor = ParallelExecutor()

    executor.execute(
        [make_task()]
    )

    events = [
        item["event"]
        for item in executor.audit_log()
    ]

    assert "EXECUTION_STARTED" in events
    assert "TASK_STARTED" in events
    assert "TASK_COMPLETED" in events
    assert "EXECUTION_COMPLETED" in events


def test_failed_task_creates_audit_event() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError("failed")

    executor.execute(
        [
            ExecutionTask(
                name="broken",
                callable=broken,
            )
        ]
    )

    assert any(
        item["event"] == "TASK_FAILED"
        for item in executor.audit_log()
    )


def test_timeout_creates_audit_event() -> None:
    executor = ParallelExecutor()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    executor.execute(
        [
            ExecutionTask(
                name="slow",
                callable=slow,
                timeout_seconds=0.01,
            )
        ]
    )

    assert any(
        item["event"] == "TASK_TIMED_OUT"
        for item in executor.audit_log()
    )


def test_clear_history_creates_audit_event() -> None:
    executor = ParallelExecutor()

    executor.clear_history()

    assert any(
        item["event"] == "HISTORY_CLEARED"
        for item in executor.audit_log()
    )


# ============================================================
# Summary
# ============================================================


def test_empty_summary() -> None:
    executor = ParallelExecutor()

    summary = executor.summary()

    assert summary["executions"] == 0
    assert summary["tasks"] == 0
    assert summary["max_concurrency"] == 9
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False


def test_summary_counts_completed_execution() -> None:
    executor = ParallelExecutor()

    executor.execute(
        [make_task()]
    )

    summary = executor.summary()

    assert summary["executions"] == 1
    assert summary["tasks"] == 1
    assert summary["completed"] == 1
    assert summary["failed"] == 0


def test_summary_counts_partial_execution() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError("failure")

    executor.execute(
        [
            make_task(
                name="healthy"
            ),
            ExecutionTask(
                name="broken",
                callable=broken,
            ),
        ]
    )

    summary = executor.summary()

    assert summary["partial"] == 1


def test_summary_counts_failed_execution() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError("failure")

    executor.execute(
        [
            ExecutionTask(
                name="broken",
                callable=broken,
            )
        ]
    )

    summary = executor.summary()

    assert summary["failed"] == 1


def test_summary_counts_timed_out_execution() -> None:
    executor = ParallelExecutor()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    executor.execute(
        [
            ExecutionTask(
                name="slow",
                callable=slow,
                timeout_seconds=0.01,
            )
        ]
    )

    summary = executor.summary()

    assert summary["timed_out"] == 1


# ============================================================
# Result/report properties
# ============================================================


def test_task_duration_non_negative() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert (
        report.task_results[0]
        .duration_seconds
        >= 0.0
    )


def test_report_duration_non_negative() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert report.duration_seconds >= 0.0


def test_completed_report_successful_property() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert report.successful is True


def test_failed_report_successful_property_false() -> None:
    executor = ParallelExecutor()

    def broken() -> None:
        raise RuntimeError("failure")

    report = executor.execute(
        [
            ExecutionTask(
                name="broken",
                callable=broken,
            )
        ]
    )

    assert report.successful is False


def test_task_execution_result_type() -> None:
    executor = ParallelExecutor()

    report = executor.execute(
        [make_task()]
    )

    assert isinstance(
        report.task_results[0],
        TaskExecutionResult,
    )