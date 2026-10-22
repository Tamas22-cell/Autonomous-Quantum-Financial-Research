from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class QuantumExperimentType(str, Enum):
    QAOA = "QAOA"
    VQE = "VQE"
    QUANTUM_KERNEL = "QUANTUM_KERNEL"
    QUANTUM_ML = "QUANTUM_ML"
    QUANTUM_MONTE_CARLO = "QUANTUM_MONTE_CARLO"
    HYBRID = "HYBRID"
    CLASSICAL_QUANTUM_BENCHMARK = "CLASSICAL_QUANTUM_BENCHMARK"


class QuantumPriority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class QuantumResearchPlan:
    experiment_id: str
    experiment_type: QuantumExperimentType
    objective: str
    problem_definition: str
    classical_baseline_required: bool
    quantum_method_required: bool
    hybrid_method_allowed: bool
    datasets: tuple[str, ...]
    algorithms: tuple[str, ...]
    metrics: tuple[str, ...]
    validation_tests: tuple[str, ...]
    priority: QuantumPriority
    requires_human_approval: bool


class QuantumResearchDesigner:
    """
    Designs controlled quantum-computing research experiments.

    The designer creates research plans only. It does not execute
    quantum circuits, optimization routines, or simulations.
    """

    DEFAULT_DATASETS = (
        "synthetic_benchmark",
        "historical_market_data",
    )

    DEFAULT_METRICS = (
        "objective_value",
        "solution_quality",
        "runtime",
        "convergence",
        "resource_usage",
    )

    DEFAULT_VALIDATION_TESTS = (
        "classical_baseline",
        "out_of_sample_validation",
        "repeated_runs",
        "noise_sensitivity",
    )

    DEFAULT_ALGORITHMS = {
        QuantumExperimentType.QAOA: (
            "QAOA",
            "parameter_optimization",
        ),
        QuantumExperimentType.VQE: (
            "VQE",
            "variational_optimizer",
        ),
        QuantumExperimentType.QUANTUM_KERNEL: (
            "quantum_kernel",
            "classical_svm",
        ),
        QuantumExperimentType.QUANTUM_ML: (
            "quantum_feature_map",
            "quantum_neural_network",
        ),
        QuantumExperimentType.QUANTUM_MONTE_CARLO: (
            "quantum_monte_carlo",
            "amplitude_estimation",
        ),
        QuantumExperimentType.HYBRID: (
            "classical_optimizer",
            "quantum_circuit",
        ),
        QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK: (
            "classical_baseline",
            "quantum_method",
        ),
    }

    def __init__(
        self,
        *,
        default_datasets: Sequence[str] | None = None,
        default_metrics: Sequence[str] | None = None,
        default_validation_tests: Sequence[str] | None = None,
    ) -> None:
        self.datasets = self._validate_names(
            self.DEFAULT_DATASETS
            if default_datasets is None
            else default_datasets,
            "default_datasets",
        )

        self.metrics = self._validate_names(
            self.DEFAULT_METRICS
            if default_metrics is None
            else default_metrics,
            "default_metrics",
        )

        self.validation_tests = self._validate_names(
            self.DEFAULT_VALIDATION_TESTS
            if default_validation_tests is None
            else default_validation_tests,
            "default_validation_tests",
        )

    @staticmethod
    def _validate_names(
        values: Sequence[str],
        field_name: str,
    ) -> tuple[str, ...]:
        if values is None:
            raise ValueError(
                f"{field_name} cannot be None."
            )

        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"{field_name} must be a sequence of strings."
            )

        try:
            result = tuple(values)
        except TypeError as exc:
            raise ValueError(
                f"{field_name} must be a sequence of strings."
            ) from exc

        if not result:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        if any(
            not isinstance(value, str)
            or not value.strip()
            for value in result
        ):
            raise ValueError(
                f"{field_name} must contain non-empty strings."
            )

        if len(set(result)) != len(result):
            raise ValueError(
                f"{field_name} cannot contain duplicates."
            )

        return result

    @staticmethod
    def _validate_experiment_id(
        experiment_id: object,
    ) -> str:
        if (
            not isinstance(experiment_id, str)
            or not experiment_id.strip()
        ):
            raise ValueError(
                "experiment_id must be a non-empty string."
            )

        return experiment_id

    @staticmethod
    def _validate_problem_definition(
        problem_definition: object,
    ) -> str:
        if (
            not isinstance(problem_definition, str)
            or not problem_definition.strip()
        ):
            raise ValueError(
                "problem_definition must be a non-empty string."
            )

        return problem_definition

    @staticmethod
    def _validate_experiment_type(
        experiment_type: object,
    ) -> QuantumExperimentType:
        if isinstance(
            experiment_type,
            QuantumExperimentType,
        ):
            return experiment_type

        raise ValueError(
            "experiment_type must be a QuantumExperimentType."
        )

    @staticmethod
    def _priority_for_type(
        experiment_type: QuantumExperimentType,
    ) -> QuantumPriority:
        if experiment_type in (
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK,
            QuantumExperimentType.HYBRID,
        ):
            return QuantumPriority.HIGH

        if experiment_type in (
            QuantumExperimentType.QAOA,
            QuantumExperimentType.VQE,
        ):
            return QuantumPriority.HIGH

        return QuantumPriority.MEDIUM

    @staticmethod
    def _objective_for_type(
        experiment_type: QuantumExperimentType,
    ) -> str:
        objectives = {
            QuantumExperimentType.QAOA: (
                "Optimize QAOA parameters and evaluate convergence "
                "and solution quality against a classical baseline."
            ),
            QuantumExperimentType.VQE: (
                "Evaluate variational quantum optimization and "
                "measure convergence, stability and solution quality."
            ),
            QuantumExperimentType.QUANTUM_KERNEL: (
                "Evaluate quantum kernel methods for feature "
                "representation and classification performance."
            ),
            QuantumExperimentType.QUANTUM_ML: (
                "Evaluate a quantum machine-learning workflow "
                "against an equivalent classical method."
            ),
            QuantumExperimentType.QUANTUM_MONTE_CARLO: (
                "Investigate quantum Monte Carlo methods and "
                "evaluate estimation accuracy and resource usage."
            ),
            QuantumExperimentType.HYBRID: (
                "Evaluate a hybrid classical-quantum workflow "
                "under a controlled problem definition."
            ),
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK: (
                "Compare classical and quantum methods under an "
                "identical problem definition and evaluation protocol."
            ),
        }

        return objectives[experiment_type]

    @staticmethod
    def _problem_requirements(
        experiment_type: QuantumExperimentType,
    ) -> tuple[bool, bool, bool]:
        if experiment_type == (
            QuantumExperimentType.CLASSICAL_QUANTUM_BENCHMARK
        ):
            return True, True, True

        if experiment_type == QuantumExperimentType.HYBRID:
            return True, True, True

        return True, True, False

    def design(
        self,
        *,
        experiment_id: str,
        experiment_type: QuantumExperimentType,
        problem_definition: str,
        datasets: Sequence[str] | None = None,
        algorithms: Sequence[str] | None = None,
        metrics: Sequence[str] | None = None,
        validation_tests: Sequence[str] | None = None,
    ) -> QuantumResearchPlan:
        experiment_id = self._validate_experiment_id(
            experiment_id
        )

        experiment_type = self._validate_experiment_type(
            experiment_type
        )

        problem_definition = (
            self._validate_problem_definition(
                problem_definition
            )
        )

        selected_datasets = (
            self.datasets
            if datasets is None
            else self._validate_names(
                datasets,
                "datasets",
            )
        )

        selected_metrics = (
            self.metrics
            if metrics is None
            else self._validate_names(
                metrics,
                "metrics",
            )
        )

        selected_validation_tests = (
            self.validation_tests
            if validation_tests is None
            else self._validate_names(
                validation_tests,
                "validation_tests",
            )
        )

        selected_algorithms = (
            self.DEFAULT_ALGORITHMS[experiment_type]
            if algorithms is None
            else self._validate_names(
                algorithms,
                "algorithms",
            )
        )

        (
            classical_required,
            quantum_required,
            hybrid_allowed,
        ) = self._problem_requirements(
            experiment_type
        )

        return QuantumResearchPlan(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            objective=self._objective_for_type(
                experiment_type
            ),
            problem_definition=problem_definition,
            classical_baseline_required=classical_required,
            quantum_method_required=quantum_required,
            hybrid_method_allowed=hybrid_allowed,
            datasets=selected_datasets,
            algorithms=selected_algorithms,
            metrics=selected_metrics,
            validation_tests=selected_validation_tests,
            priority=self._priority_for_type(
                experiment_type
            ),
            requires_human_approval=True,
        )


__all__ = [
    "QuantumExperimentType",
    "QuantumPriority",
    "QuantumResearchPlan",
    "QuantumResearchDesigner",
]