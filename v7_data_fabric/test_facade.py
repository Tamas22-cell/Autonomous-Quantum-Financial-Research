from v7_data_fabric import DataFabric

from production_research_infrastructure.data_contracts import (
    DataContract,
    DataField,
    DatasetReference,
    DatasetType,
)
from production_research_infrastructure.event_bus import EventType


def build_fabric() -> DataFabric:
    fabric = DataFabric()

    contract = DataContract(
        name="market_tick",
        version="1.0.0",
        fields=[
            DataField(
                name="symbol",
                data_type="str",
                required=True,
            ),
            DataField(
                name="price",
                data_type="float",
                required=True,
            ),
        ],
    )

    fabric.register_contract(contract)
    return fabric


def build_dataset() -> DatasetReference:
    return DatasetReference(
        name="btc_market_tick",
        version="1.0.0",
        dataset_type=DatasetType.MARKET,
        contract_name="market_tick",
        contract_version="1.0.0",
        source="test",
    )


def test_fabric_version_and_safety_flags() -> None:
    fabric = DataFabric()

    assert fabric.VERSION == "v7.0.1"
    assert fabric.RESEARCH_ONLY is True
    assert fabric.REAL_MONEY_EXECUTION is False


def test_contract_registration() -> None:
    fabric = build_fabric()

    assert fabric.contract_count() == 1
    assert fabric.registry.has_contract(
        "market_tick",
        "1.0.0",
    )


def test_dataset_registration_emits_event() -> None:
    fabric = build_fabric()
    received_events = []

    fabric.subscribe(
        EventType.DATASET_REGISTERED,
        lambda event: received_events.append(event),
        subscriber_name="test_subscriber",
    )

    record = fabric.register_dataset(
        build_dataset(),
        payload={
            "symbol": "BTC",
            "price": 100000.0,
        },
    )

    assert record.dataset.name == "btc_market_tick"
    assert fabric.dataset_count() == 1
    assert fabric.event_count() == 1
    assert len(received_events) == 1

    event = received_events[0]

    assert event.event_type == EventType.DATASET_REGISTERED
    assert event.metadata["research_only"] is True
    assert event.metadata["real_money_execution"] is False
    assert event.payload["dataset_name"] == "btc_market_tick"


def test_summary() -> None:
    fabric = build_fabric()

    summary = fabric.summary()

    assert summary["component"] == "v7_data_fabric"
    assert summary["version"] == "v7.0.1"
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["contract_count"] == 1
    assert summary["dataset_count"] == 0
    assert summary["event_count"] == 0