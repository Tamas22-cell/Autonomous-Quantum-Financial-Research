import pytest

from production_research_infrastructure.event_bus import (
    EventType,
    ResearchEvent,
)

from v7_data_fabric import DataFabric
from v7_data_fabric.lineage_service import (
    LineageServiceError,
    ResearchLineageService,
)


def build_service() -> ResearchLineageService:
    fabric = DataFabric()

    fabric.event_bus.publish(
        ResearchEvent(
            event_type=EventType.RESEARCH_STARTED,
            source="test",
            run_id="run-001",
            correlation_id="corr-001",
            payload={
                "stage": "research",
            },
        )
    )

    fabric.event_bus.publish(
        ResearchEvent(
            event_type=EventType.DATASET_REGISTERED,
            source="test",
            run_id="run-001",
            correlation_id="corr-001",
            causation_id="event-parent",
            payload={
                "dataset_id": "dataset-001",
                "dataset_name": "btc_market",
                "dataset_version": "1.0.0",
            },
        )
    )

    return ResearchLineageService(fabric)


def test_lineage_service_version_and_safety_flags() -> None:
    service = build_service()

    assert service.VERSION == "v7.0.4"
    assert service.RESEARCH_ONLY is True
    assert service.REAL_MONEY_EXECUTION is False


def test_trace_run() -> None:
    service = build_service()

    trace = service.trace_run("run-001")

    assert trace["component"] == (
        "research_lineage_service"
    )
    assert trace["version"] == "v7.0.4"
    assert trace["run_id"] == "run-001"
    assert trace["event_count"] == 2
    assert trace["dataset_ids"] == (
        "dataset-001",
    )
    assert "RESEARCH_STARTED" in (
        trace["event_types"]
    )
    assert "DATASET_REGISTERED" in (
        trace["event_types"]
    )
    assert "event-parent" in (
        trace["causation_ids"]
    )


def test_trace_correlation() -> None:
    service = build_service()

    trace = service.trace_correlation(
        "corr-001"
    )

    assert trace["correlation_id"] == (
        "corr-001"
    )
    assert trace["event_count"] == 2
    assert trace["dataset_ids"] == (
        "dataset-001",
    )


def test_invalid_trace_id_is_rejected() -> None:
    service = build_service()

    with pytest.raises(LineageServiceError):
        service.trace_run("")

    with pytest.raises(LineageServiceError):
        service.trace_correlation("")


def test_lineage_summary() -> None:
    service = build_service()

    summary = service.summary()

    assert summary["component"] == (
        "research_lineage_service"
    )
    assert summary["version"] == "v7.0.4"
    assert summary["event_count"] == 2
    assert summary["dataset_count"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"] is False
    )