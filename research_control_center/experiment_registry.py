from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional

from research_control_center.models import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentType,
    utc_now,
)


class ExperimentRegistry:
    """
    Central experiment registry for Research Control Center v5.0.10.

    Responsibilities:
    - register experiments
    - preserve reproducibility metadata
    - track lifecycle/status
    - store parameters, metrics and artifacts
    - filter experiments by asset/regime/horizon/type/status
    - compare experiment configurations
    - expose reproducibility information
    - maintain an auditable experiment history

    This component is research-only.
    It does not execute trades or modify production systems.
    """

    def __init__(self) -> None:
        self._experiments: Dict[str, ExperimentRecord] = {}
        self._audit_log: List[dict] = []

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    @staticmethod
    def _validate_record(record: ExperimentRecord) -> None:
        if not isinstance(record, ExperimentRecord):
            raise TypeError(
                "record must be an ExperimentRecord"
            )

        if not record.name.strip():
            raise ValueError(
                "experiment name cannot be empty"
            )

        if not isinstance(
            record.experiment_type,
            ExperimentType,
        ):
            raise TypeError(
                "experiment_type must be ExperimentType"
            )

        if not isinstance(
            record.status,
            ExperimentStatus,
        ):
            raise TypeError(
                "status must be ExperimentStatus"
            )

    def _audit(
        self,
        event: str,
        experiment_id: str,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "experiment_id": experiment_id,
                "details": deepcopy(details or {}),
            }
        )

    def _get_internal(
        self,
        experiment_id: str,
    ) -> ExperimentRecord:
        if not isinstance(experiment_id, str):
            raise TypeError(
                "experiment_id must be a string"
            )

        if not experiment_id.strip():
            raise ValueError(
                "experiment_id cannot be empty"
            )

        try:
            return self._experiments[experiment_id]
        except KeyError as exc:
            raise KeyError(
                f"unknown experiment_id: {experiment_id}"
            ) from exc

    # =========================================================
    # REGISTRATION
    # =========================================================

    def register(
        self,
        record: ExperimentRecord,
    ) -> ExperimentRecord:
        self._validate_record(record)

        if record.experiment_id in self._experiments:
            raise ValueError(
                f"experiment already registered: "
                f"{record.experiment_id}"
            )

        stored = deepcopy(record)

        stored.metadata = {
            **stored.metadata,
            "registry": "v5.0.10-experiment-registry",
            "research_only": True,
            "real_money_execution": False,
        }

        self._experiments[
            stored.experiment_id
        ] = stored

        self._audit(
            event="EXPERIMENT_REGISTERED",
            experiment_id=stored.experiment_id,
            details={
                "name": stored.name,
                "experiment_type": (
                    stored.experiment_type.value
                ),
                "status": stored.status.value,
            },
        )

        return deepcopy(stored)

    def create(
        self,
        name: str,
        experiment_type: ExperimentType,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        parameters: Optional[dict] = None,
        random_seed: Optional[int] = None,
        code_version: Optional[str] = None,
        data_version: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ExperimentRecord:
        if not isinstance(name, str):
            raise TypeError(
                "name must be a string"
            )

        if not name.strip():
            raise ValueError(
                "name cannot be empty"
            )

        if not isinstance(
            experiment_type,
            ExperimentType,
        ):
            raise TypeError(
                "experiment_type must be ExperimentType"
            )

        if random_seed is not None:
            if isinstance(random_seed, bool):
                raise TypeError(
                    "random_seed must be an integer"
                )

            if not isinstance(random_seed, int):
                raise TypeError(
                    "random_seed must be an integer"
                )

        record = ExperimentRecord(
            name=name.strip(),
            experiment_type=experiment_type,
            asset=asset,
            regime=regime,
            horizon=horizon,
            parameters=deepcopy(parameters or {}),
            random_seed=random_seed,
            code_version=code_version,
            data_version=data_version,
            metadata=deepcopy(metadata or {}),
        )

        return self.register(record)

    # =========================================================
    # READ
    # =========================================================

    def get(
        self,
        experiment_id: str,
    ) -> ExperimentRecord:
        return deepcopy(
            self._get_internal(experiment_id)
        )

    def list_all(self) -> List[ExperimentRecord]:
        return [
            deepcopy(record)
            for record in self._experiments.values()
        ]

    def count(self) -> int:
        return len(self._experiments)

    # =========================================================
    # STATUS MANAGEMENT
    # =========================================================

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
    ) -> ExperimentRecord:
        if not isinstance(status, ExperimentStatus):
            raise TypeError(
                "status must be ExperimentStatus"
            )

        record = self._get_internal(experiment_id)

        old_status = record.status
        record.status = status

        if status == ExperimentStatus.COMPLETED:
            record.completed_at = utc_now()

        elif status in {
            ExperimentStatus.CREATED,
            ExperimentStatus.RUNNING,
        }:
            record.completed_at = None

        self._audit(
            event="EXPERIMENT_STATUS_UPDATED",
            experiment_id=experiment_id,
            details={
                "old_status": old_status.value,
                "new_status": status.value,
            },
        )

        return deepcopy(record)

    # =========================================================
    # METRICS
    # =========================================================

    def update_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
    ) -> ExperimentRecord:
        if not isinstance(metrics, dict):
            raise TypeError(
                "metrics must be a dictionary"
            )

        normalized: Dict[str, float] = {}

        for key, value in metrics.items():
            if not isinstance(key, str):
                raise TypeError(
                    "metric names must be strings"
                )

            if isinstance(value, bool):
                raise TypeError(
                    "metric values must be numeric"
                )

            if not isinstance(value, (int, float)):
                raise TypeError(
                    "metric values must be numeric"
                )

            normalized[key] = float(value)

        record = self._get_internal(experiment_id)

        record.metrics.update(normalized)

        self._audit(
            event="EXPERIMENT_METRICS_UPDATED",
            experiment_id=experiment_id,
            details={
                "metrics": deepcopy(normalized),
            },
        )

        return deepcopy(record)

    # =========================================================
    # ARTIFACTS
    # =========================================================

    def add_artifact(
        self,
        experiment_id: str,
        artifact: str,
    ) -> ExperimentRecord:
        if not isinstance(artifact, str):
            raise TypeError(
                "artifact must be a string"
            )

        artifact = artifact.strip()

        if not artifact:
            raise ValueError(
                "artifact cannot be empty"
            )

        record = self._get_internal(experiment_id)

        if artifact not in record.artifacts:
            record.artifacts.append(artifact)

        self._audit(
            event="EXPERIMENT_ARTIFACT_ADDED",
            experiment_id=experiment_id,
            details={
                "artifact": artifact,
            },
        )

        return deepcopy(record)

    # =========================================================
    # REPRODUCIBILITY
    # =========================================================

    def reproducibility_view(
        self,
        experiment_id: str,
    ) -> dict:
        record = self._get_internal(experiment_id)

        missing = []

        if record.random_seed is None:
            missing.append("random_seed")

        if not record.code_version:
            missing.append("code_version")

        if not record.data_version:
            missing.append("data_version")

        reproducible = len(missing) == 0

        return {
            "experiment_id": record.experiment_id,
            "name": record.name,
            "experiment_type": (
                record.experiment_type.value
            ),
            "asset": record.asset,
            "regime": record.regime,
            "horizon": record.horizon,
            "parameters": deepcopy(
                record.parameters
            ),
            "random_seed": record.random_seed,
            "code_version": record.code_version,
            "data_version": record.data_version,
            "reproducible": reproducible,
            "missing_requirements": missing,
            "research_only": True,
            "real_money_execution": False,
        }

    def is_reproducible(
        self,
        experiment_id: str,
    ) -> bool:
        return bool(
            self.reproducibility_view(
                experiment_id
            )["reproducible"]
        )

    # =========================================================
    # FILTERING
    # =========================================================

    def filter(
        self,
        *,
        asset: Optional[str] = None,
        regime: Optional[str] = None,
        horizon: Optional[str] = None,
        experiment_type: Optional[
            ExperimentType
        ] = None,
        status: Optional[
            ExperimentStatus
        ] = None,
    ) -> List[ExperimentRecord]:
        if (
            experiment_type is not None
            and not isinstance(
                experiment_type,
                ExperimentType,
            )
        ):
            raise TypeError(
                "experiment_type must be "
                "ExperimentType"
            )

        if (
            status is not None
            and not isinstance(
                status,
                ExperimentStatus,
            )
        ):
            raise TypeError(
                "status must be ExperimentStatus"
            )

        results: List[ExperimentRecord] = []

        for record in self._experiments.values():
            if (
                asset is not None
                and record.asset != asset
            ):
                continue

            if (
                regime is not None
                and record.regime != regime
            ):
                continue

            if (
                horizon is not None
                and record.horizon != horizon
            ):
                continue

            if (
                experiment_type is not None
                and record.experiment_type
                != experiment_type
            ):
                continue

            if (
                status is not None
                and record.status != status
            ):
                continue

            results.append(
                deepcopy(record)
            )

        return results

    # =========================================================
    # COMPARISON
    # =========================================================

    def compare(
        self,
        experiment_id_a: str,
        experiment_id_b: str,
    ) -> dict:
        a = self._get_internal(
            experiment_id_a
        )

        b = self._get_internal(
            experiment_id_b
        )

        parameter_keys = sorted(
            set(a.parameters)
            | set(b.parameters)
        )

        metric_keys = sorted(
            set(a.metrics)
            | set(b.metrics)
        )

        parameter_comparison = {}

        for key in parameter_keys:
            value_a = a.parameters.get(key)
            value_b = b.parameters.get(key)

            parameter_comparison[key] = {
                "a": deepcopy(value_a),
                "b": deepcopy(value_b),
                "equal": value_a == value_b,
            }

        metric_comparison = {}

        for key in metric_keys:
            value_a = a.metrics.get(key)
            value_b = b.metrics.get(key)

            delta = None

            if (
                value_a is not None
                and value_b is not None
            ):
                delta = float(
                    value_b - value_a
                )

            metric_comparison[key] = {
                "a": value_a,
                "b": value_b,
                "delta_b_minus_a": delta,
            }

        return {
            "experiment_a": a.experiment_id,
            "experiment_b": b.experiment_id,
            "same_asset": (
                a.asset == b.asset
            ),
            "same_regime": (
                a.regime == b.regime
            ),
            "same_horizon": (
                a.horizon == b.horizon
            ),
            "parameter_comparison": (
                parameter_comparison
            ),
            "metric_comparison": (
                metric_comparison
            ),
        }

    # =========================================================
    # AUDIT
    # =========================================================

    def audit_log(
        self,
        experiment_id: Optional[str] = None,
    ) -> List[dict]:
        if experiment_id is None:
            return deepcopy(
                self._audit_log
            )

        return [
            deepcopy(event)
            for event in self._audit_log
            if event["experiment_id"]
            == experiment_id
        ]

    # =========================================================
    # SNAPSHOT
    # =========================================================

    def snapshot(self) -> dict:
        experiments = self.list_all()

        reproducible_count = sum(
            1
            for record in experiments
            if self.is_reproducible(
                record.experiment_id
            )
        )

        return {
            "total_experiments": len(
                experiments
            ),
            "reproducible_experiments": (
                reproducible_count
            ),
            "non_reproducible_experiments": (
                len(experiments)
                - reproducible_count
            ),
            "experiments": experiments,
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }