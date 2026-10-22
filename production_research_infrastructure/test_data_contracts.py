from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from production_research_infrastructure.data_contracts import (
    SCHEMA_VERSION,
    ContractCompatibility,
    ContractVersion,
    ContractVersionError,
    DataContract,
    DataContractError,
    DataContractValidator,
    DataEnvelope,
    DataField,
    DataQualityStatus,
    DatasetReference,
    DatasetType,
    canonical_json,
    stable_hash,
)


def make_market_contract(
    version: str = "1.0.0",
) -> DataContract:
    return DataContract(
        name="market_snapshot",
        version=version,
        description="Unified market-data contract.",
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
            DataField(
                name="metadata",
                data_type="dict",
                required=False,
                nullable=True,
            ),
        ],
    )


# ============================================================
# Canonical JSON / hashing
# ============================================================


def test_canonical_json_is_deterministic() -> None:
    left = {
        "b": 2,
        "a": 1,
    }
    right = {
        "a": 1,
        "b": 2,
    }

    assert canonical_json(left) == canonical_json(right)


def test_stable_hash_is_deterministic() -> None:
    left = {
        "symbol": "BTC",
        "price": 100_000.0,
    }
    right = {
        "price": 100_000.0,
        "symbol": "BTC",
    }

    assert stable_hash(left) == stable_hash(right)


def test_stable_hash_changes_when_payload_changes() -> None:
    first = stable_hash({"price": 100.0})
    second = stable_hash({"price": 101.0})

    assert first != second


def test_stable_hash_returns_sha256_hex() -> None:
    result = stable_hash({"test": True})

    assert isinstance(result, str)
    assert len(result) == 64

    int(result, 16)


def test_canonical_json_supports_datetime() -> None:
    timestamp = datetime(
        2026,
        9,
        18,
        12,
        0,
        tzinfo=timezone.utc,
    )

    result = canonical_json({"time": timestamp})

    assert "2026-09-18T12:00:00+00:00" in result


def test_canonical_json_supports_sets() -> None:
    first = canonical_json({"values": {"B", "A", "C"}})
    second = canonical_json({"values": {"C", "B", "A"}})

    assert first == second


# ============================================================
# ContractVersion
# ============================================================


def test_contract_version_full_parse() -> None:
    version = ContractVersion.parse("1.2.3")

    assert version.major == 1
    assert version.minor == 2
    assert version.patch == 3


def test_contract_version_short_parse() -> None:
    version = ContractVersion.parse("2")

    assert version == ContractVersion(
        major=2,
        minor=0,
        patch=0,
    )


def test_contract_version_two_part_parse() -> None:
    version = ContractVersion.parse("2.5")

    assert version == ContractVersion(
        major=2,
        minor=5,
        patch=0,
    )


def test_contract_version_accepts_v_prefix() -> None:
    version = ContractVersion.parse("v3.4.5")

    assert str(version) == "3.4.5"


def test_contract_version_rejects_invalid_string() -> None:
    with pytest.raises(ContractVersionError):
        ContractVersion.parse("1.2.3.4")


def test_contract_version_rejects_non_numeric() -> None:
    with pytest.raises(ContractVersionError):
        ContractVersion.parse("one.two.three")


def test_contract_version_rejects_negative_component() -> None:
    with pytest.raises(ContractVersionError):
        ContractVersion(
            major=-1,
            minor=0,
            patch=0,
        )


def test_contract_version_rejects_bool() -> None:
    with pytest.raises(TypeError):
        ContractVersion(
            major=True,
            minor=0,
            patch=0,
        )


def test_contract_version_same_major_is_compatible() -> None:
    first = ContractVersion.parse("1.0.0")
    second = ContractVersion.parse("1.9.9")

    assert first.is_compatible_with(second) is True


def test_contract_version_different_major_is_incompatible() -> None:
    first = ContractVersion.parse("1.0.0")
    second = ContractVersion.parse("2.0.0")

    assert first.is_compatible_with(second) is False


# ============================================================
# DataField
# ============================================================


def test_data_field_creation() -> None:
    data_field = DataField(
        name="price",
        data_type="float",
    )

    assert data_field.name == "price"
    assert data_field.data_type == "float"
    assert data_field.required is True


def test_data_field_trims_name_and_type() -> None:
    data_field = DataField(
        name="  price  ",
        data_type="  float  ",
    )

    assert data_field.name == "price"
    assert data_field.data_type == "float"


def test_data_field_rejects_empty_name() -> None:
    with pytest.raises(DataContractError):
        DataField(
            name="",
            data_type="float",
        )


def test_data_field_rejects_empty_type() -> None:
    with pytest.raises(DataContractError):
        DataField(
            name="price",
            data_type="",
        )


# ============================================================
# DataContract
# ============================================================


def test_data_contract_creation() -> None:
    contract = make_market_contract()

    assert contract.name == "market_snapshot"
    assert contract.version == "1.0.0"
    assert len(contract.fields) == 4


def test_data_contract_has_contract_id() -> None:
    contract = make_market_contract()

    assert contract.contract_id.startswith("contract_")


def test_data_contract_has_schema_metadata() -> None:
    contract = make_market_contract()

    assert contract.metadata["schema_version"] == SCHEMA_VERSION


def test_data_contract_is_research_only() -> None:
    contract = make_market_contract()

    assert contract.metadata["research_only"] is True
    assert contract.metadata["real_money_execution"] is False


def test_data_contract_rejects_duplicate_fields() -> None:
    with pytest.raises(DataContractError):
        DataContract(
            name="duplicate_contract",
            version="1.0.0",
            fields=[
                DataField(
                    name="price",
                    data_type="float",
                ),
                DataField(
                    name="price",
                    data_type="float",
                ),
            ],
        )


def test_data_contract_field_names() -> None:
    contract = make_market_contract()

    assert contract.field_names() == [
        "symbol",
        "price",
        "volume",
        "metadata",
    ]


def test_data_contract_required_fields() -> None:
    contract = make_market_contract()

    assert contract.required_fields() == [
        "symbol",
        "price",
    ]


def test_data_contract_get_field() -> None:
    contract = make_market_contract()

    data_field = contract.get_field("price")

    assert data_field is not None
    assert data_field.data_type == "float"


def test_data_contract_get_unknown_field_returns_none() -> None:
    contract = make_market_contract()

    assert contract.get_field("unknown") is None


def test_data_contract_fingerprint_is_deterministic() -> None:
    first = make_market_contract()
    second = make_market_contract()

    assert first.fingerprint == second.fingerprint


def test_contract_same_major_compatible() -> None:
    first = make_market_contract("1.0.0")
    second = make_market_contract("1.5.0")

    result = first.compatibility_with(second)

    assert result == ContractCompatibility.COMPATIBLE


def test_contract_different_major_incompatible() -> None:
    first = make_market_contract("1.0.0")
    second = make_market_contract("2.0.0")

    result = first.compatibility_with(second)

    assert result == ContractCompatibility.INCOMPATIBLE


def test_contract_different_name_incompatible() -> None:
    first = make_market_contract()

    second = DataContract(
        name="other_contract",
        version="1.0.0",
        fields=first.fields,
    )

    assert (
        first.compatibility_with(second)
        == ContractCompatibility.INCOMPATIBLE
    )


def test_contract_missing_required_field_incompatible() -> None:
    first = make_market_contract()

    second = DataContract(
        name="market_snapshot",
        version="1.1.0",
        fields=[
            DataField(
                name="symbol",
                data_type="str",
            )
        ],
    )

    assert (
        first.compatibility_with(second)
        == ContractCompatibility.INCOMPATIBLE
    )


def test_contract_required_type_change_incompatible() -> None:
    first = make_market_contract()

    second = DataContract(
        name="market_snapshot",
        version="1.1.0",
        fields=[
            DataField(
                name="symbol",
                data_type="str",
            ),
            DataField(
                name="price",
                data_type="str",
            ),
        ],
    )

    assert (
        first.compatibility_with(second)
        == ContractCompatibility.INCOMPATIBLE
    )


# ============================================================
# DatasetReference
# ============================================================


def test_dataset_reference_creation() -> None:
    dataset = DatasetReference(
        name="btc_market_data",
        version="2026.09.18",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
    )

    assert dataset.name == "btc_market_data"
    assert dataset.dataset_type == DatasetType.MARKET


def test_dataset_reference_has_id() -> None:
    dataset = DatasetReference(
        name="btc_market_data",
        version="1",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
    )

    assert dataset.dataset_id.startswith("dataset_")


def test_dataset_reference_research_safety() -> None:
    dataset = DatasetReference(
        name="btc_market_data",
        version="1",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
    )

    assert dataset.metadata["research_only"] is True
    assert dataset.metadata["real_money_execution"] is False


def test_dataset_reference_rejects_negative_rows() -> None:
    with pytest.raises(DataContractError):
        DatasetReference(
            name="dataset",
            version="1",
            dataset_type=DatasetType.MARKET,
            contract_name="market_snapshot",
            contract_version="1.0.0",
            row_count=-1,
        )


def test_dataset_reference_rejects_invalid_time_range() -> None:
    start = datetime.now(timezone.utc)
    end = start - timedelta(hours=1)

    with pytest.raises(DataContractError):
        DatasetReference(
            name="dataset",
            version="1",
            dataset_type=DatasetType.MARKET,
            contract_name="market_snapshot",
            contract_version="1.0.0",
            start_time=start,
            end_time=end,
        )


def test_dataset_fingerprint_is_deterministic() -> None:
    first = DatasetReference(
        name="btc",
        version="1",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
    )

    second = DatasetReference(
        name="btc",
        version="1",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
    )

    payload = [
        {
            "price": 100.0,
            "volume": 10,
        }
    ]

    assert (
        first.calculate_fingerprint(payload)
        == second.calculate_fingerprint(payload)
    )


def test_dataset_quality_status() -> None:
    dataset = DatasetReference(
        name="btc",
        version="1",
        dataset_type=DatasetType.MARKET,
        contract_name="market_snapshot",
        contract_version="1.0.0",
        quality_status=DataQualityStatus.VALID,
    )

    assert dataset.quality_status == DataQualityStatus.VALID


# ============================================================
# DataEnvelope
# ============================================================


def test_data_envelope_creation() -> None:
    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        payload={
            "symbol": "BTC",
            "price": 100_000.0,
        },
    )

    assert envelope.contract_name == "market_snapshot"
    assert envelope.envelope_id.startswith("envelope_")


def test_data_envelope_payload_hash_deterministic() -> None:
    first = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        payload={
            "symbol": "BTC",
            "price": 100_000.0,
        },
    )

    second = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        payload={
            "price": 100_000.0,
            "symbol": "BTC",
        },
    )

    assert first.payload_hash == second.payload_hash


def test_data_envelope_fingerprint_changes_with_context() -> None:
    first = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        producer="macro_agent",
        payload={"price": 100.0},
    )

    second = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        producer="technical_agent",
        payload={"price": 100.0},
    )

    assert first.fingerprint != second.fingerprint


def test_data_envelope_research_safety() -> None:
    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.0.0",
        payload={},
    )

    assert envelope.metadata["research_only"] is True
    assert envelope.metadata["real_money_execution"] is False


# ============================================================
# Validator
# ============================================================


def test_validator_accepts_valid_payload() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": "BTC",
            "price": 100_000.0,
            "volume": 500,
        },
    )

    assert result.valid is True
    assert result.error_count == 0


def test_validator_reports_missing_required_field() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": "BTC",
        },
    )

    assert result.valid is False
    assert result.error_count == 1
    assert result.issues[0].code == "REQUIRED_FIELD_MISSING"


def test_validator_reports_type_mismatch() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": "BTC",
            "price": "not-a-number",
        },
    )

    assert result.valid is False
    assert result.issues[0].code == "TYPE_MISMATCH"


def test_validator_accepts_integer_for_float() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": "BTC",
            "price": 100_000,
        },
    )

    assert result.valid is True


def test_validator_rejects_bool_for_integer() -> None:
    contract = DataContract(
        name="integer_contract",
        version="1.0.0",
        fields=[
            DataField(
                name="value",
                data_type="int",
            )
        ],
    )

    result = DataContractValidator.validate(
        contract,
        {"value": True},
    )

    assert result.valid is False
    assert result.issues[0].code == "TYPE_MISMATCH"


def test_validator_rejects_non_nullable_none() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": None,
            "price": 100.0,
        },
    )

    assert result.valid is False
    assert result.issues[0].code == "NULL_NOT_ALLOWED"


def test_validator_accepts_nullable_none() -> None:
    contract = make_market_contract()

    result = DataContractValidator.validate(
        contract,
        {
            "symbol": "BTC",
            "price": 100.0,
            "volume": None,
        },
    )

    assert result.valid is True


def test_validator_unknown_data_type() -> None:
    contract = DataContract(
        name="unknown_type",
        version="1.0.0",
        fields=[
            DataField(
                name="value",
                data_type="quantum_number",
            )
        ],
    )

    result = DataContractValidator.validate(
        contract,
        {"value": 10},
    )

    assert result.valid is False
    assert result.issues[0].code == "UNKNOWN_DATA_TYPE"


def test_validate_envelope_accepts_matching_contract() -> None:
    contract = make_market_contract()

    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="1.2.0",
        payload={
            "symbol": "BTC",
            "price": 100_000.0,
        },
    )

    result = DataContractValidator.validate_envelope(
        contract,
        envelope,
    )

    assert result.valid is True


def test_validate_envelope_rejects_wrong_contract_name() -> None:
    contract = make_market_contract()

    envelope = DataEnvelope(
        contract_name="wrong_contract",
        contract_version="1.0.0",
        payload={
            "symbol": "BTC",
            "price": 100_000.0,
        },
    )

    result = DataContractValidator.validate_envelope(
        contract,
        envelope,
    )

    assert result.valid is False
    assert result.issues[0].code == "CONTRACT_NAME_MISMATCH"


def test_validate_envelope_rejects_incompatible_major_version() -> None:
    contract = make_market_contract("1.0.0")

    envelope = DataEnvelope(
        contract_name="market_snapshot",
        contract_version="2.0.0",
        payload={
            "symbol": "BTC",
            "price": 100_000.0,
        },
    )

    result = DataContractValidator.validate_envelope(
        contract,
        envelope,
    )

    assert result.valid is False
    assert result.issues[0].code == "CONTRACT_VERSION_MISMATCH"