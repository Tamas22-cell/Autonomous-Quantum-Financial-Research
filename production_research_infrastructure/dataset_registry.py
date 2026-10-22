from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from production_research_infrastructure.data_contracts import (
    DataContract,
    DataContractError,
    DataContractValidator,
    DataEnvelope,
    DataQualityStatus,
    DatasetReference,
    DatasetType,
    stable_hash,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DatasetRegistryError(ValueError):
    """Base exception for dataset registry failures."""


class DatasetAlreadyExistsError(DatasetRegistryError):
    """Raised when a dataset version is already registered."""


class DatasetNotFoundError(DatasetRegistryError):
    """Raised when a requested dataset cannot be found."""


class DatasetContractNotFoundError(DatasetRegistryError):
    """Raised when a required data contract is unavailable."""


class DatasetValidationError(DatasetRegistryError):
    """Raised when dataset payload validation fails."""


@dataclass
class DatasetVersionRecord:
    dataset: DatasetReference
    payload_hash: Optional[str] = None
    contract_fingerprint: Optional[str] = None
    parent_dataset_id: Optional[str] = None
    registered_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, DatasetReference):
            raise TypeError(
                "dataset must be a DatasetReference"
            )

        self.metadata.setdefault("versioned", True)
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)


@dataclass
class DatasetRegistrySnapshot:
    contracts: List[DataContract]
    datasets: List[DatasetVersionRecord]
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.metadata.setdefault(
            "component",
            "dataset_registry",
        )
        self.metadata.setdefault("versioned", True)
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)


class DatasetRegistry:
    """
    Registry for versioned research datasets and their contracts.

    Responsibilities:
    - register and retrieve data contracts,
    - register immutable dataset versions,
    - validate payloads against contracts,
    - generate deterministic fingerprints,
    - preserve parent/child dataset lineage,
    - expose reproducible snapshots and audit history.

    This component stores metadata and optional payload hashes only.
    It does not execute trades or authorize real-money activity.
    """

    def __init__(self) -> None:
        self._contracts: Dict[str, Dict[str, DataContract]] = {}
        self._datasets_by_id: Dict[str, DatasetVersionRecord] = {}
        self._dataset_versions: Dict[
            str,
            Dict[str, DatasetVersionRecord],
        ] = {}
        self._audit_events: List[Dict[str, Any]] = []

    # ========================================================
    # Internal helpers
    # ========================================================

    def _audit(
        self,
        event: str,
        **details: Any,
    ) -> None:
        self._audit_events.append(
            {
                "event": event,
                "created_at": utc_now(),
                "details": details,
            }
        )

    @staticmethod
    def _validate_non_empty_string(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")

        cleaned = value.strip()

        if not cleaned:
            raise DatasetRegistryError(
                f"{name} must be a non-empty string"
            )

        return cleaned

    # ========================================================
    # Contract registry
    # ========================================================

    def register_contract(
        self,
        contract: DataContract,
        *,
        replace: bool = False,
    ) -> DataContract:
        if not isinstance(contract, DataContract):
            raise TypeError(
                "contract must be a DataContract"
            )

        versions = self._contracts.setdefault(
            contract.name,
            {},
        )

        if contract.version in versions and not replace:
            raise DatasetAlreadyExistsError(
                "Contract version already registered: "
                f"{contract.name}:{contract.version}"
            )

        versions[contract.version] = contract

        self._audit(
            "CONTRACT_REGISTERED",
            contract_id=contract.contract_id,
            contract_name=contract.name,
            contract_version=contract.version,
            fingerprint=contract.fingerprint,
            replaced=replace,
        )

        return contract

    def get_contract(
        self,
        name: str,
        version: str,
    ) -> DataContract:
        name = self._validate_non_empty_string(
            name,
            "name",
        )
        version = self._validate_non_empty_string(
            version,
            "version",
        )

        try:
            return self._contracts[name][version]
        except KeyError as exc:
            raise DatasetContractNotFoundError(
                f"Contract not found: {name}:{version}"
            ) from exc

    def has_contract(
        self,
        name: str,
        version: str,
    ) -> bool:
        try:
            self.get_contract(name, version)
        except DatasetContractNotFoundError:
            return False

        return True

    def list_contracts(
        self,
        name: Optional[str] = None,
    ) -> List[DataContract]:
        if name is not None:
            name = self._validate_non_empty_string(
                name,
                "name",
            )

            versions = self._contracts.get(
                name,
                {},
            )

            return list(versions.values())

        contracts: List[DataContract] = []

        for versions in self._contracts.values():
            contracts.extend(versions.values())

        return contracts

    def contract_count(self) -> int:
        return sum(
            len(versions)
            for versions in self._contracts.values()
        )

    # ========================================================
    # Dataset registration
    # ========================================================

    def register_dataset(
        self,
        dataset: DatasetReference,
        *,
        payload: Optional[Dict[str, Any]] = None,
        parent_dataset_id: Optional[str] = None,
        validate_payload: bool = True,
    ) -> DatasetVersionRecord:
        if not isinstance(dataset, DatasetReference):
            raise TypeError(
                "dataset must be a DatasetReference"
            )

        contract = self.get_contract(
            dataset.contract_name,
            dataset.contract_version,
        )

        existing_versions = self._dataset_versions.setdefault(
            dataset.name,
            {},
        )

        if dataset.version in existing_versions:
            raise DatasetAlreadyExistsError(
                "Dataset version already registered: "
                f"{dataset.name}:{dataset.version}"
            )

        if dataset.dataset_id in self._datasets_by_id:
            raise DatasetAlreadyExistsError(
                "Dataset ID already registered: "
                f"{dataset.dataset_id}"
            )

        if parent_dataset_id is not None:
            if parent_dataset_id not in self._datasets_by_id:
                raise DatasetNotFoundError(
                    "Parent dataset not found: "
                    f"{parent_dataset_id}"
                )

        payload_hash: Optional[str] = None

        if payload is not None:
            if not isinstance(payload, dict):
                raise TypeError(
                    "payload must be a dictionary"
                )

            if validate_payload:
                result = DataContractValidator.validate(
                    contract,
                    payload,
                )

                if not result.valid:
                    issue_text = "; ".join(
                        f"{issue.field_name}: {issue.message}"
                        for issue in result.issues
                    )

                    raise DatasetValidationError(
                        "Dataset payload failed contract "
                        f"validation: {issue_text}"
                    )

            payload_hash = stable_hash(payload)

            dataset.calculate_fingerprint(payload)

            if dataset.quality_status == DataQualityStatus.UNKNOWN:
                dataset.quality_status = DataQualityStatus.VALID

        record = DatasetVersionRecord(
            dataset=dataset,
            payload_hash=payload_hash,
            contract_fingerprint=contract.fingerprint,
            parent_dataset_id=parent_dataset_id,
            metadata={
                "contract_validated": (
                    payload is not None
                    and validate_payload
                ),
                "payload_available_at_registration": (
                    payload is not None
                ),
            },
        )

        existing_versions[dataset.version] = record
        self._datasets_by_id[dataset.dataset_id] = record

        self._audit(
            "DATASET_REGISTERED",
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            dataset_type=dataset.dataset_type.value,
            contract_name=dataset.contract_name,
            contract_version=dataset.contract_version,
            parent_dataset_id=parent_dataset_id,
            payload_hash=payload_hash,
            fingerprint=dataset.fingerprint,
        )

        return record

    def register_envelope(
        self,
        dataset: DatasetReference,
        envelope: DataEnvelope,
        *,
        parent_dataset_id: Optional[str] = None,
    ) -> DatasetVersionRecord:
        if not isinstance(envelope, DataEnvelope):
            raise TypeError(
                "envelope must be a DataEnvelope"
            )

        contract = self.get_contract(
            dataset.contract_name,
            dataset.contract_version,
        )

        validation = DataContractValidator.validate_envelope(
            contract,
            envelope,
        )

        if not validation.valid:
            issue_text = "; ".join(
                f"{issue.field_name}: {issue.message}"
                for issue in validation.issues
            )

            raise DatasetValidationError(
                "Envelope failed contract validation: "
                f"{issue_text}"
            )

        if (
            envelope.dataset_id is not None
            and envelope.dataset_id != dataset.dataset_id
        ):
            raise DatasetValidationError(
                "Envelope dataset_id does not match "
                "DatasetReference.dataset_id"
            )

        return self.register_dataset(
            dataset,
            payload=envelope.payload,
            parent_dataset_id=parent_dataset_id,
            validate_payload=True,
        )

    # ========================================================
    # Dataset retrieval
    # ========================================================

    def get(
        self,
        dataset_id: str,
    ) -> DatasetVersionRecord:
        dataset_id = self._validate_non_empty_string(
            dataset_id,
            "dataset_id",
        )

        try:
            return self._datasets_by_id[dataset_id]
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"Dataset not found: {dataset_id}"
            ) from exc

    def get_version(
        self,
        name: str,
        version: str,
    ) -> DatasetVersionRecord:
        name = self._validate_non_empty_string(
            name,
            "name",
        )
        version = self._validate_non_empty_string(
            version,
            "version",
        )

        try:
            return self._dataset_versions[name][version]
        except KeyError as exc:
            raise DatasetNotFoundError(
                f"Dataset version not found: "
                f"{name}:{version}"
            ) from exc

    def exists(
        self,
        name: str,
        version: str,
    ) -> bool:
        try:
            self.get_version(
                name,
                version,
            )
        except DatasetNotFoundError:
            return False

        return True

    def list_all(self) -> List[DatasetVersionRecord]:
        return list(self._datasets_by_id.values())

    def list_versions(
        self,
        name: str,
    ) -> List[DatasetVersionRecord]:
        name = self._validate_non_empty_string(
            name,
            "name",
        )

        return list(
            self._dataset_versions.get(
                name,
                {},
            ).values()
        )

    def count(self) -> int:
        return len(self._datasets_by_id)

    # ========================================================
    # Dataset lineage
    # ========================================================

    def parent_of(
        self,
        dataset_id: str,
    ) -> Optional[DatasetVersionRecord]:
        record = self.get(dataset_id)

        if record.parent_dataset_id is None:
            return None

        return self.get(record.parent_dataset_id)

    def children_of(
        self,
        dataset_id: str,
    ) -> List[DatasetVersionRecord]:
        self.get(dataset_id)

        return [
            record
            for record in self._datasets_by_id.values()
            if record.parent_dataset_id == dataset_id
        ]

    def ancestry(
        self,
        dataset_id: str,
    ) -> List[DatasetVersionRecord]:
        current = self.get(dataset_id)
        result: List[DatasetVersionRecord] = []

        visited = set()

        while current.parent_dataset_id is not None:
            parent_id = current.parent_dataset_id

            if parent_id in visited:
                raise DatasetRegistryError(
                    "Dataset lineage cycle detected"
                )

            visited.add(parent_id)

            current = self.get(parent_id)
            result.append(current)

        result.reverse()

        return result

    # ========================================================
    # Filtering / reporting
    # ========================================================

    def filter(
        self,
        *,
        dataset_type: Optional[DatasetType] = None,
        contract_name: Optional[str] = None,
        quality_status: Optional[DataQualityStatus] = None,
    ) -> List[DatasetVersionRecord]:
        records = self.list_all()

        if dataset_type is not None:
            if not isinstance(dataset_type, DatasetType):
                raise TypeError(
                    "dataset_type must be a DatasetType"
                )

            records = [
                record
                for record in records
                if record.dataset.dataset_type == dataset_type
            ]

        if contract_name is not None:
            contract_name = self._validate_non_empty_string(
                contract_name,
                "contract_name",
            )

            records = [
                record
                for record in records
                if record.dataset.contract_name
                == contract_name
            ]

        if quality_status is not None:
            if not isinstance(
                quality_status,
                DataQualityStatus,
            ):
                raise TypeError(
                    "quality_status must be a DataQualityStatus"
                )

            records = [
                record
                for record in records
                if record.dataset.quality_status
                == quality_status
            ]

        return records

    def dataset_summary(
        self,
        dataset_id: str,
    ) -> Dict[str, Any]:
        record = self.get(dataset_id)
        dataset = record.dataset

        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "version": dataset.version,
            "dataset_type": dataset.dataset_type.value,
            "contract_name": dataset.contract_name,
            "contract_version": dataset.contract_version,
            "quality_status": dataset.quality_status.value,
            "fingerprint": dataset.fingerprint,
            "payload_hash": record.payload_hash,
            "contract_fingerprint": (
                record.contract_fingerprint
            ),
            "parent_dataset_id": (
                record.parent_dataset_id
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    def summary(self) -> Dict[str, Any]:
        quality_counts: Dict[str, int] = {
            status.value: 0
            for status in DataQualityStatus
        }

        type_counts: Dict[str, int] = {
            dataset_type.value: 0
            for dataset_type in DatasetType
        }

        for record in self._datasets_by_id.values():
            quality_counts[
                record.dataset.quality_status.value
            ] += 1

            type_counts[
                record.dataset.dataset_type.value
            ] += 1

        return {
            "component": "dataset_registry",
            "contracts": self.contract_count(),
            "datasets": self.count(),
            "quality_counts": quality_counts,
            "dataset_type_counts": type_counts,
            "audit_events": len(self._audit_events),
            "research_only": True,
            "real_money_execution": False,
        }

    def audit_log(self) -> List[Dict[str, Any]]:
        return [
            {
                "event": item["event"],
                "created_at": item["created_at"],
                "details": dict(item["details"]),
            }
            for item in self._audit_events
        ]

    def snapshot(self) -> DatasetRegistrySnapshot:
        return DatasetRegistrySnapshot(
            contracts=self.list_contracts(),
            datasets=self.list_all(),
            metadata={
                "contract_count": self.contract_count(),
                "dataset_count": self.count(),
                "audit_event_count": len(
                    self._audit_events
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )