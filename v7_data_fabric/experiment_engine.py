from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from production_research_infrastructure.event_bus import (
    EventType,
    ResearchEvent,
)

from v7_data_fabric.facade import DataFabric


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentEngineError(RuntimeError):
    """Base error for autonomous experiment operations."""


class ExperimentApprovalRequiredError(
    ExperimentEngineError
):
    """Raised when an experiment is not approved."""


class ExperimentStatus(str, Enum):
    DRAFT = "DRAFT"
    APPROVED = "APPROVED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    name: str
    hypothesis: str
    dataset_ids: Tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(
        default_factory=dict
    )
    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "name",
            "hypothesis",
        ):
            value = getattr(self, field_name)

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ExperimentEngineError(
                    f"{field_name} must be a non-empty string"
                )

        if not isinstance(self.dataset_ids, tuple):
            raise ExperimentEngineError(
                "dataset_ids must be a tuple"
            )

        for dataset_id in self.dataset_ids:
            if (
                not isinstance(dataset_id, str)
                or not dataset_id.strip()
            ):
                raise ExperimentEngineError(
                    "dataset IDs must be non-empty strings"
                )

        if not isinstance(self.parameters, Mapping):
            raise ExperimentEngineError(
                "parameters must be a mapping"
            )

        if (
            self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ExperimentEngineError(
                "created_at must be timezone-aware"
            )


@dataclass
class ExperimentRecord:
    spec: ExperimentSpec
    status: ExperimentStatus = ExperimentStatus.DRAFT
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    output: Any = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    output: Any = None
    error: Optional[str] = None
    completed_at: datetime = field(
        default_factory=utc_now
    )


ExperimentExecutor = Callable[
    [ExperimentSpec],
    Any,
]


class AutonomousExperimentEngine:
    """
    Research-only experiment engine with approval gates.

    The engine does not:
    - modify its own source code;
    - execute real-money trades;
    - bypass explicit experiment approval.
    """

    VERSION = "v7.0.5"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    SELF_MODIFYING_CODE = False

    def __init__(
        self,
        *,
        fabric: Optional[DataFabric] = None,
    ) -> None:
        self.fabric = fabric
        self._records: Dict[
            str,
            ExperimentRecord,
        ] = {}

    def create_experiment(
        self,
        *,
        experiment_id: str,
        name: str,
        hypothesis: str,
        dataset_ids: Tuple[str, ...] = (),
        parameters: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> ExperimentSpec:
        if experiment_id in self._records:
            raise ExperimentEngineError(
                "Experiment ID already exists: "
                f"{experiment_id}"
            )

        spec = ExperimentSpec(
            experiment_id=experiment_id,
            name=name,
            hypothesis=hypothesis,
            dataset_ids=tuple(dataset_ids),
            parameters=dict(parameters or {}),
        )

        self._records[
            experiment_id
        ] = ExperimentRecord(
            spec=spec
        )

        return spec

    def approve(
        self,
        experiment_id: str,
    ) -> ExperimentSpec:
        record = self._get_record(
            experiment_id
        )

        if record.status != ExperimentStatus.DRAFT:
            raise ExperimentEngineError(
                "Only draft experiments can be approved"
            )

        record.status = ExperimentStatus.APPROVED
        record.approved_at = utc_now()

        return record.spec

    def run(
        self,
        experiment_id: str,
        executor: ExperimentExecutor,
    ) -> ExperimentResult:
        record = self._get_record(
            experiment_id
        )

        if record.status != ExperimentStatus.APPROVED:
            raise ExperimentApprovalRequiredError(
                "Experiment must be explicitly approved "
                "before execution"
            )

        if not callable(executor):
            raise TypeError(
                "executor must be callable"
            )

        record.status = ExperimentStatus.RUNNING
        record.started_at = utc_now()

        self._publish(
            EventType.EXPERIMENT_STARTED,
            record,
        )

        try:
            output = executor(record.spec)

            record.output = output
            record.status = ExperimentStatus.COMPLETED
            record.completed_at = utc_now()

            self._publish(
                EventType.EXPERIMENT_COMPLETED,
                record,
                payload={
                    "success": True,
                },
            )

            return ExperimentResult(
                experiment_id=(
                    record.spec.experiment_id
                ),
                status=record.status,
                output=output,
                completed_at=(
                    record.completed_at
                    or utc_now()
                ),
            )

        except Exception as exc:
            record.status = ExperimentStatus.FAILED
            record.error = (
                f"{type(exc).__name__}: {exc}"
            )
            record.completed_at = utc_now()

            self._publish(
                EventType.EXPERIMENT_COMPLETED,
                record,
                payload={
                    "success": False,
                    "error": record.error,
                },
            )

            return ExperimentResult(
                experiment_id=(
                    record.spec.experiment_id
                ),
                status=record.status,
                error=record.error,
                completed_at=(
                    record.completed_at
                    or utc_now()
                ),
            )

    def get(
        self,
        experiment_id: str,
    ) -> ExperimentRecord:
        return self._get_record(
            experiment_id
        )

    def count(self) -> int:
        return len(self._records)

    def summary(self) -> Dict[str, Any]:
        status_counts = {
            status.value: 0
            for status in ExperimentStatus
        }

        for record in self._records.values():
            status_counts[
                record.status.value
            ] += 1

        return {
            "component": (
                "autonomous_experiment_engine"
            ),
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "self_modifying_code": (
                self.SELF_MODIFYING_CODE
            ),
            "experiment_count": self.count(),
            "status_counts": status_counts,
        }

    def _get_record(
        self,
        experiment_id: str,
    ) -> ExperimentRecord:
        if (
            not isinstance(experiment_id, str)
            or not experiment_id.strip()
        ):
            raise ExperimentEngineError(
                "experiment_id must be a non-empty string"
            )

        try:
            return self._records[
                experiment_id
            ]
        except KeyError as exc:
            raise ExperimentEngineError(
                f"Unknown experiment: {experiment_id}"
            ) from exc

    def _publish(
        self,
        event_type: EventType,
        record: ExperimentRecord,
        *,
        payload: Optional[
            Dict[str, Any]
        ] = None,
    ) -> None:
        if self.fabric is None:
            return

        event_payload = {
            "experiment_id": (
                record.spec.experiment_id
            ),
            "experiment_name": (
                record.spec.name
            ),
            "status": record.status.value,
            **(payload or {}),
        }

        self.fabric.event_bus.publish(
            ResearchEvent(
                event_type=event_type,
                source=(
                    "v7.0.5_autonomous_experiment_engine"
                ),
                run_id=(
                    record.spec.experiment_id
                ),
                payload=event_payload,
                metadata={
                    "component": (
                        "autonomous_experiment_engine"
                    ),
                    "version": self.VERSION,
                    "research_only": (
                        self.RESEARCH_ONLY
                    ),
                    "real_money_execution": (
                        self.REAL_MONEY_EXECUTION
                    ),
                    "self_modifying_code": (
                        self.SELF_MODIFYING_CODE
                    ),
                },
            )
        )