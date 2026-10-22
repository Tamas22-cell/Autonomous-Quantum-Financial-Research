from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar
from uuid import uuid4


SCHEMA_VERSION = "1.0.0"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _normalize(value: Any) -> Any:
    """
    Convert supported Python values into a deterministic,
    JSON-serializable representation.
    """
    if isinstance(value, Enum):
        return value.value

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if is_dataclass(value) and not isinstance(value, type):
        return _normalize(asdict(value))

    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]

    if isinstance(value, set):
        normalized = [_normalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )

    return value


def canonical_json(value: Any) -> str:
    """
    Create deterministic JSON suitable for hashing,
    caching and reproducibility checks.
    """
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def stable_hash(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DataContractError(ValueError):
    """Base exception for data-contract validation failures."""


class ContractVersionError(DataContractError):
    """Raised when a contract version is invalid or incompatible."""


class DatasetType(str, Enum):
    MARKET = "MARKET"
    MACRO = "MACRO"
    FUNDAMENTAL = "FUNDAMENTAL"
    NEWS = "NEWS"
    ONCHAIN = "ONCHAIN"
    DERIVATIVES = "DERIVATIVES"
    TECHNICAL = "TECHNICAL"
    GEOPOLITICAL = "GEOPOLITICAL"
    ALTERNATIVE = "ALTERNATIVE"
    RESEARCH = "RESEARCH"
    OTHER = "OTHER"


class DataQualityStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    VALID = "VALID"
    WARNING = "WARNING"
    INVALID = "INVALID"


class ContractCompatibility(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"


@dataclass(frozen=True)
class ContractVersion:
    major: int
    minor: int = 0
    patch: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("major", self.major),
            ("minor", self.minor),
            ("patch", self.patch),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")

            if value < 0:
                raise ContractVersionError(
                    f"{name} must be greater than or equal to zero"
                )

    @classmethod
    def parse(cls, version: str) -> "ContractVersion":
        if not isinstance(version, str):
            raise TypeError("version must be a string")

        cleaned = version.strip()

        if cleaned.startswith("v"):
            cleaned = cleaned[1:]

        parts = cleaned.split(".")

        if len(parts) not in (1, 2, 3):
            raise ContractVersionError(
                f"Invalid contract version: {version!r}"
            )

        try:
            numbers = [int(part) for part in parts]
        except ValueError as exc:
            raise ContractVersionError(
                f"Invalid contract version: {version!r}"
            ) from exc

        while len(numbers) < 3:
            numbers.append(0)

        return cls(*numbers)

    def is_compatible_with(
        self,
        other: "ContractVersion",
    ) -> bool:
        if not isinstance(other, ContractVersion):
            raise TypeError("other must be a ContractVersion")

        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class DataField:
    name: str
    data_type: str
    required: bool = True
    nullable: bool = False
    description: Optional[str] = None
    unit: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataContractError(
                "DataField.name must be a non-empty string"
            )

        if not isinstance(self.data_type, str) or not self.data_type.strip():
            raise DataContractError(
                "DataField.data_type must be a non-empty string"
            )

        self.name = self.name.strip()
        self.data_type = self.data_type.strip()


@dataclass
class DataContract:
    name: str
    version: str
    fields: List[DataField]
    description: Optional[str] = None
    contract_id: str = field(
        default_factory=lambda: generate_id("contract")
    )
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DataContractError(
                "DataContract.name must be a non-empty string"
            )

        self.name = self.name.strip()

        ContractVersion.parse(self.version)

        if not isinstance(self.fields, list):
            raise TypeError("fields must be a list")

        for item in self.fields:
            if not isinstance(item, DataField):
                raise TypeError(
                    "every fields item must be a DataField"
                )

        field_names = [item.name for item in self.fields]

        if len(field_names) != len(set(field_names)):
            raise DataContractError(
                "DataContract field names must be unique"
            )

        self.metadata.setdefault("schema_version", SCHEMA_VERSION)
        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)

    @property
    def parsed_version(self) -> ContractVersion:
        return ContractVersion.parse(self.version)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "name": self.name,
                "version": self.version,
                "fields": self.fields,
            }
        )

    def field_names(self) -> List[str]:
        return [item.name for item in self.fields]

    def required_fields(self) -> List[str]:
        return [
            item.name
            for item in self.fields
            if item.required
        ]

    def get_field(self, name: str) -> Optional[DataField]:
        for item in self.fields:
            if item.name == name:
                return item
        return None

    def compatibility_with(
        self,
        other: "DataContract",
    ) -> ContractCompatibility:
        if not isinstance(other, DataContract):
            raise TypeError("other must be a DataContract")

        if self.name != other.name:
            return ContractCompatibility.INCOMPATIBLE

        if not self.parsed_version.is_compatible_with(
            other.parsed_version
        ):
            return ContractCompatibility.INCOMPATIBLE

        current_fields = {
            item.name: item
            for item in self.fields
        }

        other_fields = {
            item.name: item
            for item in other.fields
        }

        for name, current in current_fields.items():
            if not current.required:
                continue

            candidate = other_fields.get(name)

            if candidate is None:
                return ContractCompatibility.INCOMPATIBLE

            if candidate.data_type != current.data_type:
                return ContractCompatibility.INCOMPATIBLE

        return ContractCompatibility.COMPATIBLE


@dataclass
class DatasetReference:
    name: str
    version: str
    dataset_type: DatasetType
    contract_name: str
    contract_version: str
    source: Optional[str] = None
    uri: Optional[str] = None
    row_count: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    quality_status: DataQualityStatus = DataQualityStatus.UNKNOWN
    fingerprint: Optional[str] = None
    dataset_id: str = field(
        default_factory=lambda: generate_id("dataset")
    )
    created_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "version",
            "contract_name",
            "contract_version",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise DataContractError(
                    f"{field_name} must be a non-empty string"
                )

            setattr(self, field_name, value.strip())

        if not isinstance(self.dataset_type, DatasetType):
            raise TypeError(
                "dataset_type must be a DatasetType"
            )

        ContractVersion.parse(self.contract_version)

        if self.row_count is not None:
            if (
                isinstance(self.row_count, bool)
                or not isinstance(self.row_count, int)
            ):
                raise TypeError("row_count must be an integer")

            if self.row_count < 0:
                raise DataContractError(
                    "row_count must be greater than or equal to zero"
                )

        if (
            self.start_time is not None
            and self.end_time is not None
            and self.end_time < self.start_time
        ):
            raise DataContractError(
                "end_time cannot be earlier than start_time"
            )

        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)

    def calculate_fingerprint(
        self,
        payload: Any,
    ) -> str:
        self.fingerprint = stable_hash(
            {
                "dataset": self.name,
                "version": self.version,
                "contract_name": self.contract_name,
                "contract_version": self.contract_version,
                "payload": payload,
            }
        )
        return self.fingerprint


@dataclass
class DataEnvelope:
    contract_name: str
    contract_version: str
    payload: Dict[str, Any]
    dataset_id: Optional[str] = None
    producer: Optional[str] = None
    correlation_id: Optional[str] = None
    event_time: datetime = field(default_factory=utc_now)
    envelope_id: str = field(
        default_factory=lambda: generate_id("envelope")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contract_name, str)
            or not self.contract_name.strip()
        ):
            raise DataContractError(
                "contract_name must be a non-empty string"
            )

        self.contract_name = self.contract_name.strip()

        ContractVersion.parse(self.contract_version)

        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dictionary")

        self.metadata.setdefault("research_only", True)
        self.metadata.setdefault("real_money_execution", False)

    @property
    def payload_hash(self) -> str:
        return stable_hash(self.payload)

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "contract_name": self.contract_name,
                "contract_version": self.contract_version,
                "dataset_id": self.dataset_id,
                "producer": self.producer,
                "correlation_id": self.correlation_id,
                "payload": self.payload,
            }
        )


@dataclass
class ValidationIssue:
    field_name: str
    message: str
    code: str
    severity: str = "ERROR"


@dataclass
class ContractValidationResult:
    valid: bool
    contract_name: str
    contract_version: str
    issues: List[ValidationIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=utc_now)

    @property
    def error_count(self) -> int:
        return sum(
            1
            for issue in self.issues
            if issue.severity.upper() == "ERROR"
        )


T = TypeVar("T")


class DataContractValidator:
    """
    Lightweight schema validator for research data envelopes.

    This intentionally validates the platform's contract boundary
    without introducing an external schema dependency.
    """

    PYTHON_TYPE_MAP: Dict[str, Type[Any]] = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": (int, float),  # type: ignore[assignment]
        "bool": bool,
        "boolean": bool,
        "dict": dict,
        "object": dict,
        "list": list,
        "array": list,
    }

    @classmethod
    def validate(
        cls,
        contract: DataContract,
        payload: Dict[str, Any],
    ) -> ContractValidationResult:
        if not isinstance(contract, DataContract):
            raise TypeError("contract must be a DataContract")

        if not isinstance(payload, dict):
            raise TypeError("payload must be a dictionary")

        issues: List[ValidationIssue] = []

        for data_field in contract.fields:
            exists = data_field.name in payload

            if data_field.required and not exists:
                issues.append(
                    ValidationIssue(
                        field_name=data_field.name,
                        message="Required field is missing",
                        code="REQUIRED_FIELD_MISSING",
                    )
                )
                continue

            if not exists:
                continue

            value = payload[data_field.name]

            if value is None:
                if not data_field.nullable:
                    issues.append(
                        ValidationIssue(
                            field_name=data_field.name,
                            message="Field cannot be null",
                            code="NULL_NOT_ALLOWED",
                        )
                    )
                continue

            expected_type = cls.PYTHON_TYPE_MAP.get(
                data_field.data_type.lower()
            )

            if expected_type is None:
                issues.append(
                    ValidationIssue(
                        field_name=data_field.name,
                        message=(
                            f"Unknown contract data type: "
                            f"{data_field.data_type}"
                        ),
                        code="UNKNOWN_DATA_TYPE",
                    )
                )
                continue

            if data_field.data_type.lower() in (
                "int",
                "integer",
            ):
                type_valid = (
                    isinstance(value, int)
                    and not isinstance(value, bool)
                )
            elif data_field.data_type.lower() in (
                "float",
                "number",
            ):
                type_valid = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                )
            else:
                type_valid = isinstance(value, expected_type)

            if not type_valid:
                issues.append(
                    ValidationIssue(
                        field_name=data_field.name,
                        message=(
                            f"Expected {data_field.data_type}, "
                            f"received {type(value).__name__}"
                        ),
                        code="TYPE_MISMATCH",
                    )
                )

        return ContractValidationResult(
            valid=not any(
                issue.severity.upper() == "ERROR"
                for issue in issues
            ),
            contract_name=contract.name,
            contract_version=contract.version,
            issues=issues,
        )

    @classmethod
    def validate_envelope(
        cls,
        contract: DataContract,
        envelope: DataEnvelope,
    ) -> ContractValidationResult:
        if not isinstance(envelope, DataEnvelope):
            raise TypeError("envelope must be a DataEnvelope")

        if envelope.contract_name != contract.name:
            return ContractValidationResult(
                valid=False,
                contract_name=contract.name,
                contract_version=contract.version,
                issues=[
                    ValidationIssue(
                        field_name="contract_name",
                        message=(
                            "Envelope contract name does not match "
                            "the expected contract"
                        ),
                        code="CONTRACT_NAME_MISMATCH",
                    )
                ],
            )

        expected_version = contract.parsed_version
        envelope_version = ContractVersion.parse(
            envelope.contract_version
        )

        if not expected_version.is_compatible_with(
            envelope_version
        ):
            return ContractValidationResult(
                valid=False,
                contract_name=contract.name,
                contract_version=contract.version,
                issues=[
                    ValidationIssue(
                        field_name="contract_version",
                        message=(
                            "Envelope contract version is not "
                            "compatible with the expected contract"
                        ),
                        code="CONTRACT_VERSION_MISMATCH",
                    )
                ],
            )

        return cls.validate(
            contract=contract,
            payload=envelope.payload,
        )