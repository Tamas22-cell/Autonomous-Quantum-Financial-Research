from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from production_research_infrastructure.data_contracts import (
    DataContract,
    DatasetReference,
)
from production_research_infrastructure.dataset_registry import (
    DatasetRegistry,
    DatasetVersionRecord,
)
from production_research_infrastructure.event_bus import (
    EventBus,
    EventDeliveryReport,
    EventPriority,
    EventType,
    ResearchEvent,
    Subscription,
)


class DataFabricError(RuntimeError):
    """Base error for v7 Data Fabric operations."""


class DataFabric:
    """
    v7.0.1 integration facade.

    Combines the existing v6 dataset registry and event bus
    into one research-only data-ingestion boundary.
    """

    VERSION = "v7.0.1"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False

    def __init__(
        self,
        *,
        registry: Optional[DatasetRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        self.registry = registry or DatasetRegistry()
        self.event_bus = event_bus or EventBus()

    def register_contract(
        self,
        contract: DataContract,
        *,
        replace: bool = False,
    ) -> DataContract:
        return self.registry.register_contract(
            contract,
            replace=replace,
        )

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[..., Any],
        *,
        subscriber_name: Optional[str] = None,
    ) -> Subscription:
        return self.event_bus.subscribe(
            event_type,
            handler,
            subscriber_name=subscriber_name,
        )

    def register_dataset(
        self,
        dataset: DatasetReference,
        *,
        payload: Optional[Dict[str, Any]] = None,
        parent_dataset_id: Optional[str] = None,
        validate_payload: bool = True,
        emit_event: bool = True,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> DatasetVersionRecord:
        record = self.registry.register_dataset(
            dataset,
            payload=payload,
            parent_dataset_id=parent_dataset_id,
            validate_payload=validate_payload,
        )

        if emit_event:
            event = ResearchEvent(
                event_type=EventType.DATASET_REGISTERED,
                source="v7_data_fabric",
                payload={
                    "dataset_id": record.dataset.dataset_id,
                    "dataset_name": record.dataset.name,
                    "dataset_version": record.dataset.version,
                    "dataset_type": record.dataset.dataset_type.value,
                    "contract_name": record.dataset.contract_name,
                    "contract_version": record.dataset.contract_version,
                    "fingerprint": record.dataset.fingerprint,
                    "payload_hash": record.payload_hash,
                    "parent_dataset_id": record.parent_dataset_id,
                },
                priority=priority,
                metadata={
                    "component": "v7_data_fabric",
                    "fabric_version": self.VERSION,
                    "research_only": self.RESEARCH_ONLY,
                    "real_money_execution": self.REAL_MONEY_EXECUTION,
                },
            )

            self.event_bus.publish(event)

        return record

    def publish(
        self,
        event: ResearchEvent,
        *,
        parallel: bool = True,
    ) -> EventDeliveryReport:
        return self.event_bus.publish(
            event,
            parallel=parallel,
        )

    def dataset_count(self) -> int:
        return self.registry.count()

    def contract_count(self) -> int:
        return self.registry.contract_count()

    def event_count(self) -> int:
        return len(self.event_bus.event_history())

    def summary(self) -> Dict[str, Any]:
        return {
            "component": "v7_data_fabric",
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": self.REAL_MONEY_EXECUTION,
            "contract_count": self.contract_count(),
            "dataset_count": self.dataset_count(),
            "event_count": self.event_count(),
        }