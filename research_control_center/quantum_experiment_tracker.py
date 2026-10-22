from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Dict, List, Optional

from research_control_center.models import (
    QuantumClassicalExperiment,
    utc_now,
)


class QuantumExperimentTrackerConfig:
    """
    Configuration for the v5.0.10
    Quantum-vs-Classical Experiment Tracker.
    """

    def __init__(
        self,
        minimum_approximation_ratio: float = 0.80,
        maximum_runtime_ratio: float = 10.0,
        minimum_score_improvement: float = 0.0,
    ) -> None:
        numeric_values = {
            "minimum_approximation_ratio": (
                minimum_approximation_ratio
            ),
            "maximum_runtime_ratio": (
                maximum_runtime_ratio
            ),
            "minimum_score_improvement": (
                minimum_score_improvement
            ),
        }

        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(
                value, (int, float)
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

        if (
            minimum_approximation_ratio < 0.0
            or minimum_approximation_ratio > 1.0
        ):
            raise ValueError(
                "minimum_approximation_ratio must be "
                "between 0 and 1"
            )

        if maximum_runtime_ratio <= 0.0:
            raise ValueError(
                "maximum_runtime_ratio must be positive"
            )

        self.minimum_approximation_ratio = float(
            minimum_approximation_ratio
        )

        self.maximum_runtime_ratio = float(
            maximum_runtime_ratio
        )

        self.minimum_score_improvement = float(
            minimum_score_improvement
        )


class QuantumExperimentTracker:
    """
    Research Control Center v5.0.10
    Quantum-vs-Classical Experiment Tracker.

    Responsibilities:
    - register classical-vs-quantum experiments
    - track optimization scores
    - track objective values
    - track runtimes
    - calculate runtime ratios
    - calculate score improvements
    - calculate approximation ratios
    - track qubits and circuit depth
    - evaluate quantum competitiveness
    - compare experiments
    - rank experiments
    - preserve audit history

    Research-only component.
    No real-money execution is permitted.
    """

    def __init__(
        self,
        config: Optional[
            QuantumExperimentTrackerConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or QuantumExperimentTrackerConfig()
        )

        self._experiments: Dict[
            str,
            QuantumClassicalExperiment,
        ] = {}

        self._audit_log: List[dict] = []

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_string(
        value: str,
        name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be a string"
            )

        value = value.strip()

        if not value:
            raise ValueError(
                f"{name} cannot be empty"
            )

        return value

    @staticmethod
    def _validate_optional_number(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise TypeError(
                f"{name} must be numeric"
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                f"{name} must be finite"
            )

        return value

    @staticmethod
    def _validate_non_negative_number(
        value: Optional[float],
        name: str,
    ) -> Optional[float]:
        value = (
            QuantumExperimentTracker
            ._validate_optional_number(
                value,
                name,
            )
        )

        if value is None:
            return None

        if value < 0.0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    @staticmethod
    def _validate_optional_integer(
        value: Optional[int],
        name: str,
    ) -> Optional[int]:
        if value is None:
            return None

        if isinstance(value, bool):
            raise TypeError(
                f"{name} must be an integer"
            )

        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{name} cannot be negative"
            )

        return value

    @staticmethod
    def _validate_repetitions(
        repetitions: int,
    ) -> int:
        if isinstance(repetitions, bool):
            raise TypeError(
                "repetitions must be an integer"
            )

        if not isinstance(repetitions, int):
            raise TypeError(
                "repetitions must be an integer"
            )

        if repetitions <= 0:
            raise ValueError(
                "repetitions must be positive"
            )

        return repetitions

    @staticmethod
    def _validate_asset_universe(
        asset_universe: List[str],
    ) -> List[str]:
        if not isinstance(asset_universe, list):
            raise TypeError(
                "asset_universe must be a list"
            )

        if not asset_universe:
            raise ValueError(
                "asset_universe cannot be empty"
            )

        result: List[str] = []

        for asset in asset_universe:
            if not isinstance(asset, str):
                raise TypeError(
                    "asset_universe values must be strings"
                )

            asset = asset.strip()

            if not asset:
                raise ValueError(
                    "asset names cannot be empty"
                )

            if asset not in result:
                result.append(asset)

        return result

    def _get_internal(
        self,
        experiment_id: str,
    ) -> QuantumClassicalExperiment:
        experiment_id = self._validate_string(
            experiment_id,
            "experiment_id",
        )

        try:
            return self._experiments[
                experiment_id
            ]
        except KeyError as exc:
            raise KeyError(
                f"unknown experiment_id: "
                f"{experiment_id}"
            ) from exc

    def _audit(
        self,
        event: str,
        experiment: QuantumClassicalExperiment,
        details: Optional[dict] = None,
    ) -> None:
        self._audit_log.append(
            {
                "timestamp": utc_now(),
                "event": event,
                "experiment_id": (
                    experiment.experiment_id
                ),
                "problem_name": (
                    experiment.problem_name
                ),
                "classical_method": (
                    experiment.classical_method
                ),
                "quantum_method": (
                    experiment.quantum_method
                ),
                "details": deepcopy(
                    details or {}
                ),
                "research_only": True,
                "real_money_execution": False,
            }
        )

    # =========================================================
    # REGISTRATION
    # =========================================================

    def register(
        self,
        experiment: QuantumClassicalExperiment,
    ) -> QuantumClassicalExperiment:
        if not isinstance(
            experiment,
            QuantumClassicalExperiment,
        ):
            raise TypeError(
                "experiment must be a "
                "QuantumClassicalExperiment"
            )

        experiment.problem_name = (
            self._validate_string(
                experiment.problem_name,
                "problem_name",
            )
        )

        experiment.classical_method = (
            self._validate_string(
                experiment.classical_method,
                "classical_method",
            )
        )

        experiment.quantum_method = (
            self._validate_string(
                experiment.quantum_method,
                "quantum_method",
            )
        )

        experiment.asset_universe = (
            self._validate_asset_universe(
                experiment.asset_universe
            )
        )

        experiment.classical_score = (
            self._validate_optional_number(
                experiment.classical_score,
                "classical_score",
            )
        )

        experiment.quantum_score = (
            self._validate_optional_number(
                experiment.quantum_score,
                "quantum_score",
            )
        )

        experiment.classical_runtime_seconds = (
            self._validate_non_negative_number(
                experiment.classical_runtime_seconds,
                "classical_runtime_seconds",
            )
        )

        experiment.quantum_runtime_seconds = (
            self._validate_non_negative_number(
                experiment.quantum_runtime_seconds,
                "quantum_runtime_seconds",
            )
        )

        experiment.classical_objective = (
            self._validate_optional_number(
                experiment.classical_objective,
                "classical_objective",
            )
        )

        experiment.quantum_objective = (
            self._validate_optional_number(
                experiment.quantum_objective,
                "quantum_objective",
            )
        )

        experiment.approximation_ratio = (
            self._validate_non_negative_number(
                experiment.approximation_ratio,
                "approximation_ratio",
            )
        )

        if (
            experiment.approximation_ratio is not None
            and experiment.approximation_ratio > 1.0
        ):
            raise ValueError(
                "approximation_ratio cannot exceed 1.0"
            )

        experiment.qubits = (
            self._validate_optional_integer(
                experiment.qubits,
                "qubits",
            )
        )

        experiment.circuit_depth = (
            self._validate_optional_integer(
                experiment.circuit_depth,
                "circuit_depth",
            )
        )

        experiment.repetitions = (
            self._validate_repetitions(
                experiment.repetitions
            )
        )

        if (
            experiment.experiment_id
            in self._experiments
        ):
            raise ValueError(
                "experiment already registered: "
                f"{experiment.experiment_id}"
            )

        stored = deepcopy(experiment)

        stored.metadata = {
            **stored.metadata,
            "tracker": (
                "v5.0.10-quantum-experiment-tracker"
            ),
            "research_only": True,
            "real_money_execution": False,
        }

        self._experiments[
            stored.experiment_id
        ] = stored

        self._audit(
            "QUANTUM_EXPERIMENT_REGISTERED",
            stored,
        )

        return deepcopy(stored)

    def create(
        self,
        problem_name: str,
        classical_method: str,
        quantum_method: str,
        asset_universe: List[str],
        *,
        classical_score: Optional[
            float
        ] = None,
        quantum_score: Optional[
            float
        ] = None,
        classical_runtime_seconds: Optional[
            float
        ] = None,
        quantum_runtime_seconds: Optional[
            float
        ] = None,
        classical_objective: Optional[
            float
        ] = None,
        quantum_objective: Optional[
            float
        ] = None,
        approximation_ratio: Optional[
            float
        ] = None,
        qubits: Optional[int] = None,
        circuit_depth: Optional[int] = None,
        repetitions: int = 1,
        metadata: Optional[dict] = None,
    ) -> QuantumClassicalExperiment:
        experiment = QuantumClassicalExperiment(
            problem_name=problem_name,
            classical_method=classical_method,
            quantum_method=quantum_method,
            asset_universe=deepcopy(
                asset_universe
            ),
            classical_score=classical_score,
            quantum_score=quantum_score,
            classical_runtime_seconds=(
                classical_runtime_seconds
            ),
            quantum_runtime_seconds=(
                quantum_runtime_seconds
            ),
            classical_objective=(
                classical_objective
            ),
            quantum_objective=(
                quantum_objective
            ),
            approximation_ratio=(
                approximation_ratio
            ),
            qubits=qubits,
            circuit_depth=circuit_depth,
            repetitions=repetitions,
            metadata=deepcopy(
                metadata or {}
            ),
        )

        return self.register(experiment)

    # =========================================================
    # READ
    # =========================================================

    def get(
        self,
        experiment_id: str,
    ) -> QuantumClassicalExperiment:
        return deepcopy(
            self._get_internal(
                experiment_id
            )
        )

    def list_all(
        self,
    ) -> List[QuantumClassicalExperiment]:
        return [
            deepcopy(experiment)
            for experiment
            in self._experiments.values()
        ]

    def count(self) -> int:
        return len(self._experiments)

    # =========================================================
    # METRICS
    # =========================================================

    @staticmethod
    def score_improvement(
        experiment: QuantumClassicalExperiment,
    ) -> Optional[float]:
        if (
            experiment.classical_score is None
            or experiment.quantum_score is None
        ):
            return None

        return (
            experiment.quantum_score
            - experiment.classical_score
        )

    @staticmethod
    def runtime_ratio(
        experiment: QuantumClassicalExperiment,
    ) -> Optional[float]:
        if (
            experiment.classical_runtime_seconds
            is None
            or experiment.quantum_runtime_seconds
            is None
        ):
            return None

        if (
            experiment.classical_runtime_seconds
            == 0.0
        ):
            if (
                experiment.quantum_runtime_seconds
                == 0.0
            ):
                return 1.0

            return float("inf")

        return (
            experiment.quantum_runtime_seconds
            / experiment.classical_runtime_seconds
        )

    @staticmethod
    def objective_gap(
        experiment: QuantumClassicalExperiment,
    ) -> Optional[float]:
        if (
            experiment.classical_objective is None
            or experiment.quantum_objective is None
        ):
            return None

        return (
            experiment.quantum_objective
            - experiment.classical_objective
        )

    # =========================================================
    # APPROXIMATION RATIO
    # =========================================================

    def calculate_approximation_ratio(
        self,
        experiment_id: str,
    ) -> Optional[float]:
        experiment = self._get_internal(
            experiment_id
        )

        if (
            experiment.classical_objective is None
            or experiment.quantum_objective is None
        ):
            return None

        classical = abs(
            experiment.classical_objective
        )

        quantum = abs(
            experiment.quantum_objective
        )

        if classical == 0.0:
            if quantum == 0.0:
                ratio = 1.0
            else:
                ratio = 0.0
        else:
            ratio = min(
                1.0,
                quantum / classical,
            )

        experiment.approximation_ratio = ratio

        self._audit(
            "APPROXIMATION_RATIO_CALCULATED",
            experiment,
            {
                "approximation_ratio": ratio,
            },
        )

        return ratio

    # =========================================================
    # COMPETITIVENESS
    # =========================================================

    def competitiveness_report(
        self,
        experiment_id: str,
    ) -> dict:
        experiment = self._get_internal(
            experiment_id
        )

        improvement = self.score_improvement(
            experiment
        )

        runtime = self.runtime_ratio(
            experiment
        )

        approximation = (
            experiment.approximation_ratio
        )

        if approximation is None:
            approximation = (
                self.calculate_approximation_ratio(
                    experiment_id
                )
            )

        score_requirement = (
            improvement is not None
            and improvement
            >= self.config.minimum_score_improvement
        )

        approximation_requirement = (
            approximation is not None
            and approximation
            >= self.config.minimum_approximation_ratio
        )

        runtime_requirement = (
            runtime is not None
            and runtime
            <= self.config.maximum_runtime_ratio
        )

        competitive = (
            score_requirement
            and approximation_requirement
            and runtime_requirement
        )

        return {
            "experiment_id": (
                experiment.experiment_id
            ),
            "problem_name": (
                experiment.problem_name
            ),
            "classical_method": (
                experiment.classical_method
            ),
            "quantum_method": (
                experiment.quantum_method
            ),
            "classical_score": (
                experiment.classical_score
            ),
            "quantum_score": (
                experiment.quantum_score
            ),
            "score_improvement": improvement,
            "classical_runtime_seconds": (
                experiment.classical_runtime_seconds
            ),
            "quantum_runtime_seconds": (
                experiment.quantum_runtime_seconds
            ),
            "runtime_ratio": runtime,
            "classical_objective": (
                experiment.classical_objective
            ),
            "quantum_objective": (
                experiment.quantum_objective
            ),
            "objective_gap": (
                self.objective_gap(
                    experiment
                )
            ),
            "approximation_ratio": (
                approximation
            ),
            "qubits": experiment.qubits,
            "circuit_depth": (
                experiment.circuit_depth
            ),
            "repetitions": (
                experiment.repetitions
            ),
            "score_requirement": (
                score_requirement
            ),
            "approximation_requirement": (
                approximation_requirement
            ),
            "runtime_requirement": (
                runtime_requirement
            ),
            "competitive": competitive,
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # FILTERING
    # =========================================================

    def filter(
        self,
        *,
        problem_name: Optional[str] = None,
        classical_method: Optional[
            str
        ] = None,
        quantum_method: Optional[
            str
        ] = None,
        asset: Optional[str] = None,
    ) -> List[QuantumClassicalExperiment]:
        results = []

        for experiment in self._experiments.values():
            if (
                problem_name is not None
                and experiment.problem_name
                != problem_name
            ):
                continue

            if (
                classical_method is not None
                and experiment.classical_method
                != classical_method
            ):
                continue

            if (
                quantum_method is not None
                and experiment.quantum_method
                != quantum_method
            ):
                continue

            if (
                asset is not None
                and asset
                not in experiment.asset_universe
            ):
                continue

            results.append(
                deepcopy(experiment)
            )

        return results

    # =========================================================
    # RANKING
    # =========================================================

    def experiment_quality_score(
        self,
        experiment: QuantumClassicalExperiment,
    ) -> float:
        components: List[float] = []

        approximation = (
            experiment.approximation_ratio
        )

        if approximation is not None:
            components.append(
                max(
                    0.0,
                    min(
                        1.0,
                        approximation,
                    ),
                )
            )

        improvement = self.score_improvement(
            experiment
        )

        if improvement is not None:
            improvement_component = (
                0.5 + improvement
            )

            components.append(
                max(
                    0.0,
                    min(
                        1.0,
                        improvement_component,
                    ),
                )
            )

        runtime = self.runtime_ratio(
            experiment
        )

        if runtime is not None:
            if runtime == float("inf"):
                runtime_component = 0.0
            else:
                runtime_component = (
                    1.0
                    / (1.0 + runtime)
                )

            components.append(
                runtime_component
            )

        if not components:
            return 0.0

        return (
            sum(components)
            / len(components)
        )

    def rank(
        self,
    ) -> List[QuantumClassicalExperiment]:
        experiments = self.list_all()

        experiments.sort(
            key=self.experiment_quality_score,
            reverse=True,
        )

        return experiments

    def top(
        self,
        limit: int = 10,
    ) -> List[QuantumClassicalExperiment]:
        if isinstance(limit, bool):
            raise TypeError(
                "limit must be an integer"
            )

        if not isinstance(limit, int):
            raise TypeError(
                "limit must be an integer"
            )

        if limit <= 0:
            raise ValueError(
                "limit must be positive"
            )

        return self.rank()[:limit]

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

        quality_a = (
            self.experiment_quality_score(a)
        )

        quality_b = (
            self.experiment_quality_score(b)
        )

        return {
            "experiment_a": (
                a.experiment_id
            ),
            "experiment_b": (
                b.experiment_id
            ),
            "quality_score_a": quality_a,
            "quality_score_b": quality_b,
            "quality_delta_b_minus_a": (
                quality_b - quality_a
            ),
            "score_improvement_a": (
                self.score_improvement(a)
            ),
            "score_improvement_b": (
                self.score_improvement(b)
            ),
            "runtime_ratio_a": (
                self.runtime_ratio(a)
            ),
            "runtime_ratio_b": (
                self.runtime_ratio(b)
            ),
            "approximation_ratio_a": (
                a.approximation_ratio
            ),
            "approximation_ratio_b": (
                b.approximation_ratio
            ),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def summary(self) -> dict:
        experiments = self.list_all()

        competitive_count = 0

        for experiment in experiments:
            report = (
                self.competitiveness_report(
                    experiment.experiment_id
                )
            )

            if report["competitive"]:
                competitive_count += 1

        if experiments:
            average_quality = (
                sum(
                    self.experiment_quality_score(
                        experiment
                    )
                    for experiment
                    in experiments
                )
                / len(experiments)
            )
        else:
            average_quality = 0.0

        return {
            "total_experiments": (
                len(experiments)
            ),
            "competitive_experiments": (
                competitive_count
            ),
            "non_competitive_experiments": (
                len(experiments)
                - competitive_count
            ),
            "average_quality_score": (
                average_quality
            ),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }

    # =========================================================
    # AUDIT
    # =========================================================

    def audit_log(
        self,
        experiment_id: Optional[
            str
        ] = None,
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
        return {
            "experiments": self.list_all(),
            "ranking": self.rank(),
            "summary": self.summary(),
            "generated_at": utc_now(),
            "research_only": True,
            "real_money_execution": False,
        }