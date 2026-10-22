from __future__ import annotations

from datetime import datetime, timezone

import pytest

from production_research_infrastructure.data_contracts import (
    DataContract,
    DataEnvelope,
    DataField,
    DataQualityStatus,
    DatasetReference,
    DatasetType,
)
from production_research_infrastructure.dataset_registry import (
    DatasetAlreadyExistsError,
    DatasetContractNotFoundError,
    DatasetNotFoundError,
    DatasetRegistry,
    DatasetRegistryError,
    DatasetValidationError,
    DatasetVersionRecord,
)


def make_contract(
    version: str = "1.0.0",
) -> DataContract:
    return DataContract(
        name="market_snapshot",
        version=version,
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
            DataField(
                name="volume",
                data_type="int",
                required=False,
                nullable=True,
            ),
        ],
    )


def make_dataset(
    *,
    name: str = "btc_market_data",
    version: str = "1",
    contract_version: str = "1.0.0",
    dataset_type: DatasetType = DatasetType.MARKET,
    quality_status: DataQualityStatus = DataQualityStatus.UNKNOWN,
) -> DatasetReference:
    return DatasetReference(
        name=name,
        version=version,
        dataset_type=dataset_type,
        contract_name="market_snapshot",
        contract_version=contract_version,
        quality_status=quality_status,
    )


def make_registry() -> DatasetRegistry:
    registry = DatasetRegistry()
    registry.register_contract(make_contract())
    return registry


def valid_payload(
    symbol: str = "BTC",
    price: float = 100_000.0,
) -> dict:
    return {
        "symbol": symbol,
        "price": price,
        "volume": 1000,
    }


# ============================================================
# Construction
# ============================================================


def test_registry_starts_empty() -> None:
    registry = DatasetRegistry()

    assert registry.count() == 0
    assert registry.contract_count() == 0


def test_registry_list_all_starts_empty() -> None:
    registry = DatasetRegistry()

    assert registry.list_all() == []


def test_registry_audit_log_starts_empty() -> None:
    registry = DatasetRegistry()

    assert registry.audit_log() == []


# ============================================================
# Contract registry
# ============================================================


def test_register_contract() -> None:
    registry = DatasetRegistry()
    contract = make_contract()

    result = registry.register_contract(contract)

    assert result is contract
    assert registry.contract_count() == 1


def test_get_contract() -> None:
    registry = DatasetRegistry()
    contract = make_contract()

    registry.register_contract(contract)

    result = registry.get_contract(
        "market_snapshot",
        "1.0.0",
    )

    assert result is contract


def test_has_contract_true() -> None:
    registry = make_registry()

    assert registry.has_contract(
        "market_snapshot",
        "1.0.0",
    ) is True


def test_has_contract_false() -> None:
    registry = make_registry()

    assert registry.has_contract(
        "market_snapshot",
        "2.0.0",
    ) is False


def test_register_multiple_contract_versions() -> None:
    registry = DatasetRegistry()

    registry.register_contract(
        make_contract("1.0.0")
    )
    registry.register_contract(
        make_contract("1.1.0")
    )

    assert registry.contract_count() == 2
    assert len(
        registry.list_contracts("market_snapshot")
    ) == 2


def test_duplicate_contract_rejected() -> None:
    registry = DatasetRegistry()

    registry.register_contract(
        make_contract("1.0.0")
    )

    with pytest.raises(
        DatasetAlreadyExistsError
    ):
        registry.register_contract(
            make_contract("1.0.0")
        )


def test_contract_can_be_replaced_explicitly() -> None:
    registry = DatasetRegistry()

    first = make_contract("1.0.0")
    second = make_contract("1.0.0")

    registry.register_contract(first)
    registry.register_contract(
        second,
        replace=True,
    )

    assert (
        registry.get_contract(
            "market_snapshot",
            "1.0.0",
        )
        is second
    )


def test_get_missing_contract_raises() -> None:
    registry = DatasetRegistry()

    with pytest.raises(
        DatasetContractNotFoundError
    ):
        registry.get_contract(
            "missing",
            "1.0.0",
        )


def test_list_all_contracts() -> None:
    registry = DatasetRegistry()

    registry.register_contract(
        make_contract("1.0.0")
    )
    registry.register_contract(
        make_contract("1.1.0")
    )

    assert len(registry.list_contracts()) == 2


def test_register_contract_requires_contract() -> None:
    registry = DatasetRegistry()

    with pytest.raises(TypeError):
        registry.register_contract("invalid")  # type: ignore[arg-type]


# ============================================================
# Dataset registration
# ============================================================


def test_register_dataset() -> None:
    registry = make_registry()
    dataset = make_dataset()

    record = registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert isinstance(
        record,
        DatasetVersionRecord,
    )
    assert record.dataset is dataset
    assert registry.count() == 1


def test_registered_dataset_get_by_id() -> None:
    registry = make_registry()
    dataset = make_dataset()

    record = registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert registry.get(
        dataset.dataset_id
    ) is record


def test_registered_dataset_get_by_version() -> None:
    registry = make_registry()
    dataset = make_dataset()

    record = registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert (
        registry.get_version(
            dataset.name,
            dataset.version,
        )
        is record
    )


def test_dataset_exists_true() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert registry.exists(
        dataset.name,
        dataset.version,
    ) is True


def test_dataset_exists_false() -> None:
    registry = make_registry()

    assert registry.exists(
        "btc_market_data",
        "999",
    ) is False


def test_dataset_payload_hash_created() -> None:
    registry = make_registry()

    record = registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    assert record.payload_hash is not None
    assert len(record.payload_hash) == 64


def test_dataset_fingerprint_created() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert dataset.fingerprint is not None
    assert len(dataset.fingerprint) == 64


def test_contract_fingerprint_recorded() -> None:
    registry = make_registry()

    record = registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    assert record.contract_fingerprint is not None
    assert len(
        record.contract_fingerprint
    ) == 64


def test_validated_dataset_becomes_valid() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert (
        dataset.quality_status
        == DataQualityStatus.VALID
    )


def test_dataset_without_payload_can_register() -> None:
    registry = make_registry()

    record = registry.register_dataset(
        make_dataset()
    )

    assert record.payload_hash is None
    assert registry.count() == 1


def test_dataset_without_payload_remains_unknown() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(dataset)

    assert (
        dataset.quality_status
        == DataQualityStatus.UNKNOWN
    )


def test_duplicate_dataset_version_rejected() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(version="1"),
        payload=valid_payload(),
    )

    with pytest.raises(
        DatasetAlreadyExistsError
    ):
        registry.register_dataset(
            make_dataset(version="1"),
            payload=valid_payload(),
        )


def test_multiple_dataset_versions_allowed() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(version="1"),
        payload=valid_payload(
            price=100_000.0
        ),
    )

    registry.register_dataset(
        make_dataset(version="2"),
        payload=valid_payload(
            price=101_000.0
        ),
    )

    assert registry.count() == 2
    assert len(
        registry.list_versions(
            "btc_market_data"
        )
    ) == 2


def test_missing_contract_rejects_dataset() -> None:
    registry = DatasetRegistry()

    with pytest.raises(
        DatasetContractNotFoundError
    ):
        registry.register_dataset(
            make_dataset(),
            payload=valid_payload(),
        )


def test_invalid_payload_rejected() -> None:
    registry = make_registry()

    with pytest.raises(
        DatasetValidationError
    ):
        registry.register_dataset(
            make_dataset(),
            payload={
                "symbol": "BTC",
            },
        )


def test_wrong_payload_type_rejected() -> None:
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.register_dataset(
            make_dataset(),
            payload=["BTC"],  # type: ignore[arg-type]
        )


def test_validation_can_be_disabled() -> None:
    registry = make_registry()
    dataset = make_dataset()

    record = registry.register_dataset(
        dataset,
        payload={
            "anything": "research-data",
        },
        validate_payload=False,
    )

    assert record.payload_hash is not None
    assert (
        record.metadata[
            "contract_validated"
        ]
        is False
    )


# ============================================================
# Envelope registration
# ============================================================


def test_register_valid_envelope() -> None:
    registry = make_registry()
    dataset = make_dataset()

    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        dataset_id=dataset.dataset_id,
        producer="market_data_fabric",
        payload=valid_payload(),
    )

    record = registry.register_envelope(
        dataset,
        envelope,
    )

    assert record.dataset is dataset
    assert record.payload_hash is not None


def test_envelope_wrong_contract_rejected() -> None:
    registry = make_registry()
    dataset = make_dataset()

    envelope = DataEnvelope(
        contract_name="wrong_contract",
        contract_version="1.0.0",
        dataset_id=dataset.dataset_id,
        payload=valid_payload(),
    )

    with pytest.raises(
        DatasetValidationError
    ):
        registry.register_envelope(
            dataset,
            envelope,
        )


def test_envelope_wrong_dataset_id_rejected() -> None:
    registry = make_registry()
    dataset = make_dataset()

    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        dataset_id="dataset_wrong",
        payload=valid_payload(),
    )

    with pytest.raises(
        DatasetValidationError
    ):
        registry.register_envelope(
            dataset,
            envelope,
        )


def test_envelope_without_dataset_id_allowed() -> None:
    registry = make_registry()
    dataset = make_dataset()

    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        payload=valid_payload(),
    )

    record = registry.register_envelope(
        dataset,
        envelope,
    )

    assert record.dataset.dataset_id == (
        dataset.dataset_id
    )


# ============================================================
# Dataset lineage
# ============================================================


def test_parent_dataset_lineage() -> None:
    registry = make_registry()

    parent = make_dataset(
        version="1"
    )
    child = make_dataset(
        version="2"
    )

    registry.register_dataset(
        parent,
        payload=valid_payload(
            price=100_000.0
        ),
    )

    registry.register_dataset(
        child,
        payload=valid_payload(
            price=101_000.0
        ),
        parent_dataset_id=parent.dataset_id,
    )

    result = registry.parent_of(
        child.dataset_id
    )

    assert result is not None
    assert (
        result.dataset.dataset_id
        == parent.dataset_id
    )


def test_root_dataset_has_no_parent() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    assert registry.parent_of(
        dataset.dataset_id
    ) is None


def test_children_of_dataset() -> None:
    registry = make_registry()

    parent = make_dataset(
        version="1"
    )
    child = make_dataset(
        version="2"
    )

    registry.register_dataset(
        parent,
        payload=valid_payload(),
    )

    registry.register_dataset(
        child,
        payload=valid_payload(
            price=101_000.0
        ),
        parent_dataset_id=parent.dataset_id,
    )

    children = registry.children_of(
        parent.dataset_id
    )

    assert len(children) == 1
    assert (
        children[0].dataset.dataset_id
        == child.dataset_id
    )


def test_missing_parent_rejected() -> None:
    registry = make_registry()

    with pytest.raises(
        DatasetNotFoundError
    ):
        registry.register_dataset(
            make_dataset(),
            payload=valid_payload(),
            parent_dataset_id=(
                "dataset_missing"
            ),
        )


def test_dataset_ancestry() -> None:
    registry = make_registry()

    version_1 = make_dataset(
        version="1"
    )
    version_2 = make_dataset(
        version="2"
    )
    version_3 = make_dataset(
        version="3"
    )

    registry.register_dataset(
        version_1,
        payload=valid_payload(
            price=100.0
        ),
    )

    registry.register_dataset(
        version_2,
        payload=valid_payload(
            price=101.0
        ),
        parent_dataset_id=(
            version_1.dataset_id
        ),
    )

    registry.register_dataset(
        version_3,
        payload=valid_payload(
            price=102.0
        ),
        parent_dataset_id=(
            version_2.dataset_id
        ),
    )

    ancestry = registry.ancestry(
        version_3.dataset_id
    )

    assert [
        item.dataset.version
        for item in ancestry
    ] == ["1", "2"]


# ============================================================
# Filtering
# ============================================================


def test_filter_by_dataset_type() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(
            name="btc",
            dataset_type=DatasetType.MARKET,
        ),
        payload=valid_payload(),
    )

    registry.register_dataset(
        make_dataset(
            name="research",
            dataset_type=DatasetType.RESEARCH,
        ),
        payload=valid_payload(),
    )

    result = registry.filter(
        dataset_type=DatasetType.MARKET
    )

    assert len(result) == 1
    assert (
        result[0].dataset.dataset_type
        == DatasetType.MARKET
    )


def test_filter_by_contract_name() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    result = registry.filter(
        contract_name="market_snapshot"
    )

    assert len(result) == 1


def test_filter_by_quality_status() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(
            quality_status=(
                DataQualityStatus.WARNING
            )
        )
    )

    result = registry.filter(
        quality_status=(
            DataQualityStatus.WARNING
        )
    )

    assert len(result) == 1


# ============================================================
# Reporting / snapshot / audit
# ============================================================


def test_dataset_summary() -> None:
    registry = make_registry()
    dataset = make_dataset()

    registry.register_dataset(
        dataset,
        payload=valid_payload(),
    )

    summary = registry.dataset_summary(
        dataset.dataset_id
    )

    assert (
        summary["dataset_id"]
        == dataset.dataset_id
    )
    assert (
        summary["quality_status"]
        == "VALID"
    )
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_registry_summary() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    summary = registry.summary()

    assert summary["contracts"] == 1
    assert summary["datasets"] == 1
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_counts_dataset_type() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    summary = registry.summary()

    assert (
        summary["dataset_type_counts"][
            "MARKET"
        ]
        == 1
    )


def test_summary_counts_quality() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    summary = registry.summary()

    assert (
        summary["quality_counts"]["VALID"]
        == 1
    )


def test_audit_records_contract_registration() -> None:
    registry = DatasetRegistry()

    registry.register_contract(
        make_contract()
    )

    events = registry.audit_log()

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "CONTRACT_REGISTERED"
    )


def test_audit_records_dataset_registration() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    events = registry.audit_log()

    assert any(
        event["event"]
        == "DATASET_REGISTERED"
        for event in events
    )


def test_snapshot_contains_contracts_and_datasets() -> None:
    registry = make_registry()

    registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    snapshot = registry.snapshot()

    assert len(snapshot.contracts) == 1
    assert len(snapshot.datasets) == 1


def test_snapshot_is_research_only() -> None:
    registry = make_registry()

    snapshot = registry.snapshot()

    assert (
        snapshot.metadata["research_only"]
        is True
    )
    assert (
        snapshot.metadata[
            "real_money_execution"
        ]
        is False
    )


def test_dataset_record_is_research_only() -> None:
    registry = make_registry()

    record = registry.register_dataset(
        make_dataset(),
        payload=valid_payload(),
    )

    assert (
        record.metadata["research_only"]
        is True
    )
    assert (
        record.metadata[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# Error handling
# ============================================================


def test_get_missing_dataset_raises() -> None:
    registry = make_registry()

    with pytest.raises(
        DatasetNotFoundError
    ):
        registry.get(
            "dataset_missing"
        )


def test_get_missing_version_raises() -> None:
    registry = make_registry()

    with pytest.raises(
        DatasetNotFoundError
    ):
        registry.get_version(
            "btc",
            "999",
        )


def test_empty_dataset_id_rejected() -> None:
    registry = make_registry()

    with pytest.raises(
        DatasetRegistryError
    ):
        registry.get("")


def test_filter_rejects_invalid_dataset_type() -> None:
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.filter(
            dataset_type="MARKET"  # type: ignore[arg-type]
        )


def test_filter_rejects_invalid_quality_status() -> None:
    registry = make_registry()

    with pytest.raises(TypeError):
        registry.filter(
            quality_status="VALID"  # type: ignore[arg-type]
        )