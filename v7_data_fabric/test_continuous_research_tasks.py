import pytest

from .continuous_research_tasks import (
    ContinuousResearchTaskError,
    ContinuousResearchTaskGenerator,
    TaskPriority,
    TaskStatus,
)


def test_version_and_safety_flags() -> None:
    generator = ContinuousResearchTaskGenerator()

    assert generator.VERSION == "v7.0.12"
    assert generator.RESEARCH_ONLY is True
    assert generator.REAL_MONEY_EXECUTION is False
    assert generator.AUTOMATIC_TRADING is False
    assert generator.AUTOMATIC_ALPHA_APPROVAL is False


def test_task_generation() -> None:
    generator = ContinuousResearchTaskGenerator()

    task = generator.generate_task(
        "task-001",
        "Test causal signal",
        "Evaluate causal robustness.",
        priority=TaskPriority.HIGH,
        source_component="causal_alpha_discovery",
    )

    assert task.task_id == "task-001"
    assert task.priority == TaskPriority.HIGH
    assert task.status == TaskStatus.PROPOSED
    assert len(generator.list_tasks()) == 1


def test_signal_generates_research_task() -> None:
    generator = ContinuousResearchTaskGenerator()

    task = generator.generate_from_signal(
        "regime_break",
        0.90,
        source_component="market_regime_intelligence",
    )

    assert task.priority == TaskPriority.HIGH
    assert task.status == TaskStatus.PROPOSED
    assert task.evidence["signal_name"] == "regime_break"


def test_moderate_signal_gets_medium_priority() -> None:
    generator = ContinuousResearchTaskGenerator()

    task = generator.generate_from_signal(
        "weak_signal",
        0.20,
        source_component="benchmark_engine",
    )

    assert task.priority == TaskPriority.MEDIUM


def test_task_lifecycle() -> None:
    generator = ContinuousResearchTaskGenerator()

    generator.generate_task(
        "task-002",
        "Validate hypothesis",
        "Run independent validation.",
    )

    approved = generator.approve_task("task-002")
    assert approved.status == TaskStatus.APPROVED

    completed = generator.complete_task(
        "task-002",
        result={"score": 0.82},
    )

    assert completed.status == TaskStatus.COMPLETED
    assert completed.evidence["result"]["score"] == 0.82


def test_rejected_task() -> None:
    generator = ContinuousResearchTaskGenerator()

    generator.generate_task(
        "task-003",
        "Reject this task",
        "Not sufficiently supported.",
    )

    rejected = generator.reject_task("task-003")

    assert rejected.status == TaskStatus.REJECTED
    assert len(
        generator.list_tasks(TaskStatus.REJECTED)
    ) == 1


def test_invalid_transitions_are_rejected() -> None:
    generator = ContinuousResearchTaskGenerator()

    generator.generate_task(
        "task-004",
        "Invalid completion",
        "Approval is required first.",
    )

    with pytest.raises(ContinuousResearchTaskError):
        generator.complete_task("task-004")


def test_duplicate_and_unknown_tasks_are_rejected() -> None:
    generator = ContinuousResearchTaskGenerator()

    generator.generate_task(
        "task-005",
        "Unique task",
        "Only once.",
    )

    with pytest.raises(ContinuousResearchTaskError):
        generator.generate_task(
            "task-005",
            "Duplicate task",
            "Must fail.",
        )

    with pytest.raises(ContinuousResearchTaskError):
        generator.get_task("missing-task")


def test_summary_and_snapshot() -> None:
    generator = ContinuousResearchTaskGenerator()

    generator.generate_task(
        "task-006",
        "Summary task",
        "Check operational state.",
    )

    summary = generator.summary()
    snapshot = generator.snapshot()

    assert summary["version"] == "v7.0.12"
    assert summary["task_count"] == 1
    assert summary["proposed_count"] == 1
    assert summary["research_only"] is True
    assert summary["automatic_alpha_approval"] is False

    assert snapshot["version"] == "v7.0.12"
    assert snapshot["task_count"] == 1
    assert snapshot["research_only"] is True