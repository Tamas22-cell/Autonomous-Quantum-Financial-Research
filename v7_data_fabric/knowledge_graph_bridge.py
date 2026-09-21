from __future__ import annotations

from typing import Any, Dict, Optional

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
)
from financial_knowledge_graph.models import (
    Evidence,
    EvidenceType,
    KnowledgeNode,
    NodeType,
)
from production_research_infrastructure.event_bus import (
    EventType,
    ResearchEvent,
    Subscription,
)
from v7_data_fabric.facade import DataFabric


class KnowledgeGraphBridgeError(RuntimeError):
    """Base error for Data Fabric to Knowledge Graph integration."""


class KnowledgeGraphBridge:
    """
    Connects v7 Data Fabric dataset events to the Financial Knowledge Graph.

    Dataset registration creates:
    - one KnowledgeNode;
    - one Evidence record.

    No causal relationship is created automatically.
    """

    VERSION = "v7.0.2"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False

    def __init__(
        self,
        *,
        store: Optional[FinancialKnowledgeGraphStore] = None,
    ) -> None:
        self.store = store or FinancialKnowledgeGraphStore()
        self._subscription: Optional[Subscription] = None

    def attach(self, fabric: DataFabric) -> Subscription:
        if not isinstance(fabric, DataFabric):
            raise TypeError("fabric must be a DataFabric")

        self._subscription = fabric.subscribe(
            EventType.DATASET_REGISTERED,
            self.handle_dataset_registered,
            subscriber_name="v7.0.2_knowledge_graph_bridge",
        )

        return self._subscription

    @staticmethod
    def _node_type(dataset_type: str) -> NodeType:
        mapping = {
            "MACRO": NodeType.ECONOMIC_MECHANISM,
            "NEWS": NodeType.MACRO_EVENT,
            "GEOPOLITICAL": NodeType.MACRO_EVENT,
            "MARKET": NodeType.MARKET_VARIABLE,
            "TECHNICAL": NodeType.MARKET_VARIABLE,
            "DERIVATIVES": NodeType.MARKET_VARIABLE,
            "ONCHAIN": NodeType.MARKET_VARIABLE,
        }

        return mapping.get(
            dataset_type,
            NodeType.ASSET,
        )

    def handle_dataset_registered(
        self,
        event: ResearchEvent,
    ) -> Dict[str, Any]:
        if event.event_type != EventType.DATASET_REGISTERED:
            raise KnowledgeGraphBridgeError(
                "Bridge received an unsupported event type"
            )

        payload = event.payload

        required_fields = (
            "dataset_id",
            "dataset_name",
            "dataset_version",
            "dataset_type",
        )

        missing = [
            field_name
            for field_name in required_fields
            if field_name not in payload
        ]

        if missing:
            raise KnowledgeGraphBridgeError(
                f"Dataset event is missing fields: {missing}"
            )

        dataset_id = str(payload["dataset_id"])
        node_id = f"dataset:{dataset_id}"
        evidence_id = f"evidence:{event.event_id}"

        node = KnowledgeNode(
            node_id=node_id,
            node_type=self._node_type(
                str(payload["dataset_type"])
            ),
            label=str(payload["dataset_name"]),
            description=(
                "Dataset registered through the v7.0.1 Data Fabric."
            ),
            attributes={
                "dataset_id": dataset_id,
                "dataset_version": str(
                    payload["dataset_version"]
                ),
                "dataset_type": str(
                    payload["dataset_type"]
                ),
                "contract_name": payload.get(
                    "contract_name"
                ),
                "contract_version": payload.get(
                    "contract_version"
                ),
                "fingerprint": payload.get(
                    "fingerprint"
                ),
                "payload_hash": payload.get(
                    "payload_hash"
                ),
                "fabric_version": "v7.0.1",
            },
        )

        evidence = Evidence(
            evidence_id=evidence_id,
            evidence_type=EvidenceType.MARKET_DATA,
            source=event.source or "v7_data_fabric",
            summary=(
                f"Dataset {payload['dataset_name']} "
                f"version {payload['dataset_version']} "
                "was registered."
            ),
            observed_at=event.created_at,
            reference=event.event_id,
        )

        self.store.add_node(node)
        self.store.add_evidence(evidence)

        return {
            "node_id": node_id,
            "evidence_id": evidence_id,
            "dataset_id": dataset_id,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "component": "knowledge_graph_bridge",
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": self.REAL_MONEY_EXECUTION,
            "node_count": self.store.node_count,
            "evidence_count": self.store.evidence_count,
            "relationship_count": self.store.relationship_count,
            "outcome_count": self.store.outcome_count,
        }