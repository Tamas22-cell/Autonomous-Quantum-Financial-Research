from __future__ import annotations

import asyncio

import pytest

from production_research_infrastructure.event_bus import (
    EventBus,
    EventPriority,
    EventType,
)
from production_research_infrastructure.execution_engine import (
    ResearchExecutionEngine,
    ResearchExecutionError,
    ResearchExecutionRequest,
    ResearchExecutionValidationError,
    ResearchRunStatus,
)
from production_research_infrastructure.parallel_executor import (
    ExecutionTask,
    ParallelExecutor,
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


def make_request(
    *,
    name: str = "market_research",
    tasks=None,
    parallel: bool = True,
    correlation_id: str | None = None,
) -> ResearchExecutionRequest:
    if tasks is None:
        tasks = [make_task()]

    return ResearchExecutionRequest(
        name=name,
        tasks=tasks,
        parallel=parallel,
        correlation_id=correlation_id,
    )


# ============================================================
# ResearchExecutionRequest
# ============================================================


def test_request_creation() -> None:
    request = make_request()

    assert request.name == "market_research"
    assert request.request_id.startswith("request_")
    assert len(request.tasks) == 1


def test_request_default_parallel() -> None:
    request = make_request()

    assert request.parallel is True


def test_request_default_priority() -> None:
    request = make_request()

    assert request.priority == EventPriority.NORMAL


def test_request_custom_priority() -> None:
    request = ResearchExecutionRequest(
        name="risk_research",
        tasks=[],
        priority=EventPriority.CRITICAL,
    )

    assert request.priority == EventPriority.CRITICAL


def test_request_research_safety() -> None:
    request = make_request()

    assert request.metadata["research_only"] is True
    assert (
        request.metadata["real_money_execution"]
        is False
    )


def test_request_rejects_empty_name() -> None:
    with pytest.raises(
        ResearchExecutionValidationError
    ):
        ResearchExecutionRequest(
            name=" ",
            tasks=[],
        )


def test_request_trims_name() -> None:
    request = ResearchExecutionRequest(
        name="  research_run  ",
        tasks=[],
    )

    assert request.name == "research_run"


def test_request_requires_task_list() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionRequest(
            name="research",
            tasks="invalid",  # type: ignore[arg-type]
        )


def test_request_rejects_invalid_task() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionRequest(
            name="research",
            tasks=["invalid"],  # type: ignore[list-item]
        )


def test_request_requires_boolean_parallel() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionRequest(
            name="research",
            tasks=[],
            parallel=1,  # type: ignore[arg-type]
        )


def test_request_requires_priority_enum() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionRequest(
            name="research",
            tasks=[],
            priority="HIGH",  # type: ignore[arg-type]
        )


def test_request_rejects_empty_correlation_id() -> None:
    with pytest.raises(
        ResearchExecutionValidationError
    ):
        ResearchExecutionRequest(
            name="research",
            tasks=[],
            correlation_id=" ",
        )


def test_request_trims_correlation_id() -> None:
    request = ResearchExecutionRequest(
        name="research",
        tasks=[],
        correlation_id="  corr_001  ",
    )

    assert request.correlation_id == "corr_001"


# ============================================================
# Engine construction
# ============================================================


def test_engine_creation() -> None:
    engine = ResearchExecutionEngine()

    assert isinstance(engine.event_bus, EventBus)
    assert isinstance(
        engine.parallel_executor,
        ParallelExecutor,
    )


def test_engine_version() -> None:
    engine = ResearchExecutionEngine()

    assert engine.VERSION == "v6.0.1"


def test_engine_custom_event_bus() -> None:
    bus = EventBus()

    engine = ResearchExecutionEngine(
        event_bus=bus
    )

    assert engine.event_bus is bus


def test_engine_custom_executor() -> None:
    executor = ParallelExecutor(
        max_concurrency=3
    )

    engine = ResearchExecutionEngine(
        parallel_executor=executor
    )

    assert engine.parallel_executor is executor


def test_engine_custom_max_concurrency() -> None:
    engine = ResearchExecutionEngine(
        max_concurrency=4
    )

    assert (
        engine.parallel_executor.max_concurrency
        == 4
    )


def test_engine_rejects_invalid_event_bus() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionEngine(
            event_bus="invalid"  # type: ignore[arg-type]
        )


def test_engine_rejects_invalid_executor() -> None:
    with pytest.raises(TypeError):
        ResearchExecutionEngine(
            parallel_executor="invalid"  # type: ignore[arg-type]
        )


def test_engine_starts_without_runs() -> None:
    engine = ResearchExecutionEngine()

    assert engine.runs() == []
    assert engine.audit_log() == []


# ============================================================
# Basic execution
# ============================================================


def test_execute_single_task() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.status == ResearchRunStatus.COMPLETED
    assert run.task_count == 1
    assert run.success_count == 1
    assert run.failure_count == 0


def test_run_id_created() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.run_id.startswith("run_")


def test_run_preserves_request_id() -> None:
    engine = ResearchExecutionEngine()
    request = make_request()

    run = engine.execute(request)

    assert run.request_id == request.request_id


def test_run_preserves_name() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(
            name="btc_research"
        )
    )

    assert run.name == "btc_research"


def test_run_research_safety() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.metadata["research_only"] is True
    assert (
        run.metadata["real_money_execution"]
        is False
    )


def test_run_contains_execution_report() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.execution_report is not None
    assert (
        run.execution_report.success_count
        == 1
    )


def test_completed_run_successful_property() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.successful is True


def test_run_duration_non_negative() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.duration_seconds >= 0.0


# ============================================================
# Empty execution
# ============================================================


def test_empty_request_completes() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(tasks=[])
    )

    assert run.status == ResearchRunStatus.COMPLETED
    assert run.task_count == 0


# ============================================================
# Multiple / parallel tasks
# ============================================================


def test_execute_multiple_tasks() -> None:
    engine = ResearchExecutionEngine()

    tasks = [
        make_task(
            name="macro",
            agent_name="macro_agent",
        ),
        make_task(
            name="stock",
            agent_name="stock_agent",
        ),
        make_task(
            name="crypto",
            agent_name="crypto_agent",
        ),
    ]

    run = engine.execute(
        make_request(tasks=tasks)
    )

    assert run.status == ResearchRunStatus.COMPLETED
    assert run.task_count == 3
    assert run.success_count == 3


def test_parallel_flag_passed_to_executor() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(
            tasks=[
                make_task("one"),
                make_task("two"),
            ],
            parallel=False,
        )
    )

    assert run.execution_report is not None
    assert (
        run.execution_report.metadata["parallel"]
        is False
    )


def test_async_execution() -> None:
    engine = ResearchExecutionEngine()

    async def worker() -> str:
        await asyncio.sleep(0)
        return "async-result"

    request = make_request(
        tasks=[
            ExecutionTask(
                name="async_agent",
                callable=worker,
                agent_name="async_agent",
            )
        ]
    )

    run = asyncio.run(
        engine.execute_async(request)
    )

    assert run.status == ResearchRunStatus.COMPLETED
    assert run.execution_report is not None
    assert (
        run.execution_report
        .task_results[0]
        .result
        == "async-result"
    )


def test_execute_rejected_inside_active_loop() -> None:
    engine = ResearchExecutionEngine()

    async def scenario() -> None:
        with pytest.raises(
            ResearchExecutionError
        ):
            engine.execute(
                make_request()
            )

    asyncio.run(scenario())


def test_execute_requires_request() -> None:
    engine = ResearchExecutionEngine()

    with pytest.raises(TypeError):
        engine.execute(
            "invalid"  # type: ignore[arg-type]
        )


# ============================================================
# Failure propagation
# ============================================================


def test_failed_task_marks_run_failed() -> None:
    engine = ResearchExecutionEngine()

    def broken() -> None:
        raise RuntimeError(
            "research agent failed"
        )

    request = make_request(
        tasks=[
            ExecutionTask(
                name="broken",
                callable=broken,
                agent_name="broken_agent",
            )
        ]
    )

    run = engine.execute(request)

    assert run.status == ResearchRunStatus.FAILED
    assert run.failure_count == 1
    assert run.successful is False


def test_partial_execution_marks_run_partial() -> None:
    engine = ResearchExecutionEngine()

    def broken() -> None:
        raise RuntimeError("failure")

    request = make_request(
        tasks=[
            make_task(
                name="healthy"
            ),
            ExecutionTask(
                name="broken",
                callable=broken,
            ),
        ]
    )

    run = engine.execute(request)

    assert run.status == ResearchRunStatus.PARTIAL
    assert run.success_count == 1
    assert run.failure_count == 1


def test_timeout_marks_run_timed_out() -> None:
    engine = ResearchExecutionEngine()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    request = make_request(
        tasks=[
            ExecutionTask(
                name="slow",
                callable=slow,
                timeout_seconds=0.01,
            )
        ]
    )

    run = engine.execute(request)

    assert run.status == ResearchRunStatus.TIMED_OUT
    assert run.timeout_count == 1


# ============================================================
# Correlation IDs
# ============================================================


def test_custom_correlation_id_preserved() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(
            correlation_id="corr_123"
        )
    )

    assert run.correlation_id == "corr_123"


def test_generated_run_id_used_as_default_correlation() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert run.correlation_id == run.run_id


def test_events_use_run_id() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    events = engine.event_bus.events_for_run(
        run.run_id
    )

    assert len(events) == 3

    assert all(
        event.run_id == run.run_id
        for event in events
    )


def test_events_use_correlation_id() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(
            correlation_id="corr_market"
        )
    )

    events = (
        engine.event_bus
        .events_for_correlation(
            "corr_market"
        )
    )

    assert len(events) == 3
    assert run.correlation_id == "corr_market"


# ============================================================
# Event lifecycle
# ============================================================


def test_successful_run_publishes_expected_events() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    event_types = [
        event.event_type
        for event
        in engine.event_bus.events_for_run(
            run.run_id
        )
    ]

    assert event_types == [
        EventType.RESEARCH_REQUESTED,
        EventType.RESEARCH_STARTED,
        EventType.RESEARCH_COMPLETED,
    ]


def test_failed_run_publishes_failed_event() -> None:
    engine = ResearchExecutionEngine()

    def broken() -> None:
        raise RuntimeError("failed")

    run = engine.execute(
        make_request(
            tasks=[
                ExecutionTask(
                    name="broken",
                    callable=broken,
                )
            ]
        )
    )

    event_types = [
        event.event_type
        for event
        in engine.event_bus.events_for_run(
            run.run_id
        )
    ]

    assert event_types[-1] == (
        EventType.RESEARCH_FAILED
    )


def test_partial_run_publishes_failed_event() -> None:
    engine = ResearchExecutionEngine()

    def broken() -> None:
        raise RuntimeError("failed")

    run = engine.execute(
        make_request(
            tasks=[
                make_task("healthy"),
                ExecutionTask(
                    name="broken",
                    callable=broken,
                ),
            ]
        )
    )

    events = engine.event_bus.events_for_run(
        run.run_id
    )

    assert (
        events[-1].event_type
        == EventType.RESEARCH_FAILED
    )


def test_requested_event_contains_task_count() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request(
            tasks=[
                make_task("one"),
                make_task("two"),
            ]
        )
    )

    events = engine.event_bus.events_for_run(
        run.run_id
    )

    assert events[0].payload["task_count"] == 2


def test_completed_event_contains_execution_id() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    events = engine.event_bus.events_for_run(
        run.run_id
    )

    assert (
        events[-1].payload["execution_id"]
        == run.execution_report.execution_id
    )


# ============================================================
# Event subscribers integration
# ============================================================


def test_event_subscriber_receives_started_event() -> None:
    bus = EventBus()
    received = []

    def handler(event) -> None:
        received.append(event.event_type)

    bus.subscribe(
        EventType.RESEARCH_STARTED,
        handler,
    )

    engine = ResearchExecutionEngine(
        event_bus=bus
    )

    engine.execute(
        make_request()
    )

    assert received == [
        EventType.RESEARCH_STARTED
    ]


def test_subscriber_failure_does_not_break_run() -> None:
    bus = EventBus()

    def broken_handler(event) -> None:
        raise RuntimeError(
            "event consumer failed"
        )

    bus.subscribe(
        EventType.RESEARCH_STARTED,
        broken_handler,
    )

    engine = ResearchExecutionEngine(
        event_bus=bus
    )

    run = engine.execute(
        make_request()
    )

    assert run.status == ResearchRunStatus.COMPLETED


# ============================================================
# Run retrieval
# ============================================================


def test_run_saved_to_history() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert engine.runs() == [run]


def test_get_run() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert engine.get_run(
        run.run_id
    ) is run


def test_get_missing_run_returns_none() -> None:
    engine = ResearchExecutionEngine()

    assert engine.get_run(
        "run_missing"
    ) is None


def test_get_run_rejects_empty_id() -> None:
    engine = ResearchExecutionEngine()

    with pytest.raises(
        ResearchExecutionValidationError
    ):
        engine.get_run("")


def test_runs_by_status() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request(
            name="run_one"
        )
    )
    engine.execute(
        make_request(
            name="run_two"
        )
    )

    completed = engine.runs_by_status(
        ResearchRunStatus.COMPLETED
    )

    assert len(completed) == 2


def test_runs_by_status_requires_enum() -> None:
    engine = ResearchExecutionEngine()

    with pytest.raises(TypeError):
        engine.runs_by_status(
            "COMPLETED"  # type: ignore[arg-type]
        )


def test_runs_for_correlation() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request(
            correlation_id="corr_shared"
        )
    )
    engine.execute(
        make_request(
            correlation_id="corr_shared"
        )
    )

    runs = engine.runs_for_correlation(
        "corr_shared"
    )

    assert len(runs) == 2


def test_runs_for_correlation_rejects_empty_id() -> None:
    engine = ResearchExecutionEngine()

    with pytest.raises(
        ResearchExecutionValidationError
    ):
        engine.runs_for_correlation("")


# ============================================================
# Audit
# ============================================================


def test_execution_creates_run_audit_events() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request()
    )

    events = [
        item["event"]
        for item in engine.audit_log()
    ]

    assert "RUN_CREATED" in events
    assert "RUN_STARTED" in events
    assert "RUN_COMPLETED" in events


def test_audit_contains_run_id() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    assert any(
        item["details"].get("run_id")
        == run.run_id
        for item in engine.audit_log()
    )


# ============================================================
# Reporting
# ============================================================


def test_run_report() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    report = engine.run_report(
        run.run_id
    )

    assert report["run_id"] == run.run_id
    assert report["status"] == "COMPLETED"
    assert report["task_count"] == 1
    assert report["success_count"] == 1


def test_run_report_research_safety() -> None:
    engine = ResearchExecutionEngine()

    run = engine.execute(
        make_request()
    )

    report = engine.run_report(
        run.run_id
    )

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


def test_missing_run_report_raises() -> None:
    engine = ResearchExecutionEngine()

    with pytest.raises(
        ResearchExecutionValidationError
    ):
        engine.run_report(
            "run_missing"
        )


def test_empty_summary() -> None:
    engine = ResearchExecutionEngine()

    summary = engine.summary()

    assert summary["version"] == "v6.0.1"
    assert summary["runs"] == 0
    assert summary["tasks"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_after_completed_run() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request()
    )

    summary = engine.summary()

    assert summary["runs"] == 1
    assert summary["tasks"] == 1
    assert summary["successful_tasks"] == 1
    assert (
        summary["status_counts"]["COMPLETED"]
        == 1
    )


def test_summary_counts_failed_tasks() -> None:
    engine = ResearchExecutionEngine()

    def broken() -> None:
        raise RuntimeError("failure")

    engine.execute(
        make_request(
            tasks=[
                ExecutionTask(
                    name="broken",
                    callable=broken,
                )
            ]
        )
    )

    summary = engine.summary()

    assert summary["failed_tasks"] == 1
    assert (
        summary["status_counts"]["FAILED"]
        == 1
    )


def test_summary_counts_timeout_tasks() -> None:
    engine = ResearchExecutionEngine()

    async def slow() -> None:
        await asyncio.sleep(0.1)

    engine.execute(
        make_request(
            tasks=[
                ExecutionTask(
                    name="slow",
                    callable=slow,
                    timeout_seconds=0.01,
                )
            ]
        )
    )

    summary = engine.summary()

    assert summary["timed_out_tasks"] == 1
    assert (
        summary["status_counts"]["TIMED_OUT"]
        == 1
    )


# ============================================================
# Clear history
# ============================================================


def test_clear_history() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request()
    )

    engine.clear_history()

    assert engine.runs() == []


def test_clear_history_creates_audit_event() -> None:
    engine = ResearchExecutionEngine()

    engine.clear_history()

    assert any(
        item["event"]
        == "RUN_HISTORY_CLEARED"
        for item in engine.audit_log()
    )


def test_clear_engine_history_does_not_clear_event_bus() -> None:
    engine = ResearchExecutionEngine()

    engine.execute(
        make_request()
    )

    assert len(
        engine.event_bus.event_history()
    ) == 3

    engine.clear_history()

    assert len(
        engine.event_bus.event_history()
    ) == 3