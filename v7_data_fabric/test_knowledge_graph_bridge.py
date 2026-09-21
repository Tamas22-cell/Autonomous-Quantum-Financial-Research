from v7_data_fabric import DataFabric
from v7_data_fabric.knowledge_graph_bridge import (
    KnowledgeGraphBridge,
)

from production_research_infrastructure.data_contracts import (
    DataContract,
    DataField,
    DatasetReference,
    DatasetType,
)


def build_fabric() -> DataFabric:
    fabric = DataFabric()

    fabric.register_contract(
        DataContract(
            name="market_tick",
            version="1.0.0",
            fields=[
                DataField(
                    name="symbol",
                    data_type="str",
                ),
                DataField(
                    name="price",
                    data_type="float",
                ),
            ],
        )
    )

    return fabric


def build_dataset() -> DatasetReference:
    return DatasetReference(
        name="btc_market_tick",
        version="1.0.0",
        dataset_type=DatasetType.MARKET,
        contract_name="market_tick",
        contract_version="1.0.0",
        source="test_source",
    )


def test_bridge_version_and_safety_flags() -> None:
    bridge = KnowledgeGraphBridge()

    assert bridge.VERSION == "v7.0.2"
    assert bridge.RESEARCH_ONLY is True
    assert bridge.REAL_MONEY_EXECUTION is False


def test_dataset_event_creates_graph_node_and_evidence() -> None:
    fabric = build_fabric()
    bridge = KnowledgeGraphBridge()

    bridge.attach(fabric)

    fabric.register_dataset(
        build_dataset(),
        payload={
            "symbol": "BTC",
            "price": 100000.0,
        },
    )

    assert bridge.store.node_count == 1
    assert bridge.store.evidence_count == 1
    assert bridge.store.relationship_count == 0
    assert bridge.store.outcome_count == 0

    node = bridge.store.list_nodes()[0]
    evidence = bridge.store.list_evidence()[0]

    assert node.label == "btc_market_tick"
    assert node.attributes["dataset_version"] == "1.0.0"
    assert node.attributes["fabric_version"] == "v7.0.1"

    assert evidence.source == "v7_data_fabric"
    assert evidence.reference is not None


def test_bridge_summary() -> None:
    fabric = build_fabric()
    bridge = KnowledgeGraphBridge()

    bridge.attach(fabric)

    fabric.register_dataset(
        build_dataset(),
        payload={
            "symbol": "BTC",
            "price": 100000.0,
        },
    )

    summary = bridge.summary()

    assert summary["component"] == "knowledge_graph_bridge"
    assert summary["version"] == "v7.0.2"
    assert summary["node_count"] == 1
    assert summary["evidence_count"] == 1
    assert summary["relationship_count"] == 0
    assert summary["outcome_count"] == 0