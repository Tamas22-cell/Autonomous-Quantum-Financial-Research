from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from production_research_infrastructure.event_bus import (
    EventType,
    ResearchEvent,
)

from v7_data_fabric.facade import DataFabric


class LineageServiceError(RuntimeError):
    """Base error for research lineage operations."""


class ResearchLineageService:
    """
    Provides a unified read-only trace over Data Fabric events.

    This service reconstructs research activity from:
    - run IDs;
    - correlation IDs;
    - causation IDs;
    - event history;
    - dataset registration metadata.
    """

    VERSION = "v7.0.4"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False

    def __init__(self, fabric: DataFabric) -> None:
        if not isinstance(fabric, DataFabric):
            raise TypeError(
                "fabric must be a DataFabric"
            )

        self.fabric = fabric

    @staticmethod
    def _require_text(
        value: str,
        name: str,
    ) -> str:
        if (
            not isinstance(value, str)
            or not value.strip()
        ):
            raise LineageServiceError(
                f"{name} must be a non-empty string"
            )

        return value.strip()

    def trace_run(
        self,
        run_id: str,
    ) -> Dict[str, Any]:
        run_id = self._require_text(
            run_id,
            "run_id",
        )

        events = tuple(
            self.fabric.event_bus.events_for_run(
                run_id
            )
        )

        return self._build_trace(
            run_id=run_id,
            events=events,
        )

    def trace_correlation(
        self,
        correlation_id: str,
    ) -> Dict[str, Any]:
        correlation_id = self._require_text(
            correlation_id,
            "correlation_id",
        )

        events = tuple(
            self.fabric.event_bus.events_for_correlation(
                correlation_id
            )
        )

        return self._build_trace(
            correlation_id=correlation_id,
            events=events,
        )

    def _build_trace(
        self,
        *,
        events: Tuple[ResearchEvent, ...],
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        dataset_ids = []
        event_ids = []
        event_types = []
        causation_ids = []
        correlation_ids = []

        for event in events:
            event_ids.append(event.event_id)
            event_types.append(event.event_type.value)

            if event.causation_id is not None:
                causation_ids.append(
                    event.causation_id
                )

            if event.correlation_id is not None:
                correlation_ids.append(
                    event.correlation_id
                )

            if (
                event.event_type
                == EventType.DATASET_REGISTERED
            ):
                dataset_id = event.payload.get(
                    "dataset_id"
                )

                if (
                    isinstance(dataset_id, str)
                    and dataset_id not in dataset_ids
                ):
                    dataset_ids.append(dataset_id)

        return {
            "component": "research_lineage_service",
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "run_id": run_id,
            "correlation_id": correlation_id,
            "event_count": len(events),
            "event_ids": tuple(event_ids),
            "event_types": tuple(event_types),
            "dataset_ids": tuple(dataset_ids),
            "causation_ids": tuple(causation_ids),
            "correlation_ids": tuple(correlation_ids),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "component": "research_lineage_service",
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "event_count": len(
                self.fabric.event_bus.event_history()
            ),
            "dataset_count": (
                self.fabric.dataset_count()
            ),
        }