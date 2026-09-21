import pytest

from v7_data_fabric import DataFabric
from v7_data_fabric.experiment_engine import (
    AutonomousExperimentEngine,
    ExperimentApprovalRequiredError,
    ExperimentStatus,
)


def build_engine() -> AutonomousExperimentEngine:
    return AutonomousExperimentEngine(
        fabric=DataFabric()
    )


def test_engine_version_and_safety_flags() -> None:
    engine = build_engine()

    assert engine.VERSION == "v7.0.5"
    assert engine.RESEARCH_ONLY is True
    assert engine.REAL_MONEY_EXECUTION is False
    assert engine.SELF_MODIFYING_CODE is False


def test_experiment_requires_approval() -> None:
    engine = build_engine()

    engine.create_experiment(
        experiment_id="exp-001",
        name="BTC signal test",
        hypothesis=(
            "A momentum signal may predict "
            "short-term BTC returns."
        ),
    )

    with pytest.raises(
        ExperimentApprovalRequiredError
    ):
        engine.run(
            "exp-001",
            lambda spec: {
                "metric": 0.12
            },
        )


def test_approved_experiment_completes() -> None:
    engine = build_engine()

    spec = engine.create_experiment(
        experiment_id="exp-002",
        name="Liquidity experiment",
        hypothesis=(
            "Liquidity conditions may affect "
            "market behavior."
        ),
        dataset_ids=("dataset-001",),
        parameters={
            "window": 30,
        },
    )

    engine.approve("exp-002")

    result = engine.run(
        "exp-002",
        lambda received_spec: {
            "experiment_id": (
                received_spec.experiment_id
            ),
            "score": 0.75,
        },
    )

    assert spec.experiment_id == "exp-002"
    assert result.status == (
        ExperimentStatus.COMPLETED
    )
    assert result.output["score"] == 0.75
    assert engine.get("exp-002").status == (
        ExperimentStatus.COMPLETED
    )


def test_failed_experiment_is_recorded() -> None:
    engine = build_engine()

    engine.create_experiment(
        experiment_id="exp-003",
        name="Failing experiment",
        hypothesis="This experiment should fail.",
    )

    engine.approve("exp-003")

    def failing_executor(spec):
        raise ValueError("controlled failure")

    result = engine.run(
        "exp-003",
        failing_executor,
    )

    assert result.status == ExperimentStatus.FAILED
    assert result.error is not None
    assert "controlled failure" in result.error
    assert engine.get("exp-003").status == (
        ExperimentStatus.FAILED
    )


def test_experiment_events_are_published() -> None:
    engine = build_engine()
    fabric = engine.fabric

    assert fabric is not None

    engine.create_experiment(
        experiment_id="exp-004",
        name="Event experiment",
        hypothesis="Test event lifecycle.",
    )

    engine.approve("exp-004")

    engine.run(
        "exp-004",
        lambda spec: {
            "success": True
        },
    )

    events = fabric.event_bus.events_for_run(
        "exp-004"
    )

    assert len(events) == 2
    assert events[0].event_type.value == (
        "EXPERIMENT_STARTED"
    )
    assert events[1].event_type.value == (
        "EXPERIMENT_COMPLETED"
    )


def test_engine_summary() -> None:
    engine = build_engine()

    engine.create_experiment(
        experiment_id="exp-005",
        name="Summary experiment",
        hypothesis="Test summary.",
    )

    summary = engine.summary()

    assert summary["component"] == (
        "autonomous_experiment_engine"
    )
    assert summary["version"] == "v7.0.5"
    assert summary["experiment_count"] == 1
    assert summary["status_counts"]["DRAFT"] == 1
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["self_modifying_code"] is False