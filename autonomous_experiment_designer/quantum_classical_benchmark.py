from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping, Sequence


class BenchmarkStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BenchmarkConclusion(str, Enum):
    QUANTUM_BETTER = "QUANTUM_BETTER"
    CLASSICAL_BETTER = "CLASSICAL_BETTER"
    EQUIVALENT = "EQUIVALENT"


@dataclass(frozen=True)
class QuantumClassicalBenchmarkResult:
    status: BenchmarkStatus
    classical_metric: float | None
    quantum_metric: float | None
    performance_delta: float | None
    relative_improvement: float | None
    conclusion: BenchmarkConclusion | None
    classical_output: object | None
    quantum_output: object | None
    metadata: Mapping[str, object]
    error: str | None = None


class QuantumClassicalBenchmark:
    """
    Reproducible classical-vs-quantum benchmark layer.

    Both methods receive exactly the same problem definition and
    input data. The benchmark itself does not decide which method
    is universally superior; it records the measured comparison.
    """

    def __init__(
        self,
        *,
        metric_name: str = "score",
        tolerance: float = 1e-9,
        require_human_approval: bool = True,
    ) -> None:
        self.metric_name = self._validate_metric_name(
            metric_name
        )
        self.tolerance = self._validate_tolerance(
            tolerance
        )

        if not isinstance(
            require_human_approval,
            bool,
        ):
            raise ValueError(
                "require_human_approval must be a boolean."
            )

        self.require_human_approval = (
            require_human_approval
        )

    @staticmethod
    def _validate_metric_name(
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "metric_name must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                "metric_name must not be empty."
            )

        return value

    @staticmethod
    def _validate_tolerance(
        value: object,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "tolerance must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                "tolerance must be finite."
            )

        if value < 0.0:
            raise ValueError(
                "tolerance must be non-negative."
            )

        return value

    @staticmethod
    def _validate_callable(
        value: object,
        field_name: str,
    ) -> None:
        if not callable(value):
            raise ValueError(
                f"{field_name} must be callable."
            )

    @staticmethod
    def _validate_metric(
        value: object,
        field_name: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"{field_name} must be numeric."
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                f"{field_name} must be finite."
            )

        return value

    @staticmethod
    def _validate_problem_definition(
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "problem_definition must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                "problem_definition must not be empty."
            )

        return value

    @staticmethod
    def _validate_dataset(
        value: object,
    ) -> tuple[object, ...]:
        if value is None:
            raise ValueError(
                "dataset cannot be None."
            )

        if isinstance(
            value,
            (str, bytes),
        ):
            raise ValueError(
                "dataset must be a sequence."
            )

        try:
            return tuple(value)
        except TypeError as exc:
            raise ValueError(
                "dataset must be a sequence."
            ) from exc

    def _failed_result(
        self,
        *,
        error: str,
        metadata: Mapping[str, object] | None = None,
    ) -> QuantumClassicalBenchmarkResult:
        return QuantumClassicalBenchmarkResult(
            status=BenchmarkStatus.FAILED,
            classical_metric=None,
            quantum_metric=None,
            performance_delta=None,
            relative_improvement=None,
            conclusion=None,
            classical_output=None,
            quantum_output=None,
            metadata=dict(metadata or {}),
            error=error,
        )

    def _conclude(
        self,
        classical_metric: float,
        quantum_metric: float,
    ) -> BenchmarkConclusion:
        difference = (
            quantum_metric
            - classical_metric
        )

        if abs(difference) <= self.tolerance:
            return BenchmarkConclusion.EQUIVALENT

        if difference > 0:
            return BenchmarkConclusion.QUANTUM_BETTER

        return BenchmarkConclusion.CLASSICAL_BETTER

    def compare(
        self,
        problem_definition: str,
        dataset: Sequence[object],
        classical_runner: Callable[
            [str, tuple[object, ...]],
            object,
        ],
        quantum_runner: Callable[
            [str, tuple[object, ...]],
            object,
        ],
        metric_function: Callable[[object], float],
    ) -> QuantumClassicalBenchmarkResult:
        """
        Run classical and quantum methods on the same problem.

        Both runners receive the exact same normalized problem
        definition and immutable dataset tuple.
        """

        self._validate_callable(
            classical_runner,
            "classical_runner",
        )
        self._validate_callable(
            quantum_runner,
            "quantum_runner",
        )
        self._validate_callable(
            metric_function,
            "metric_function",
        )

        problem_definition = (
            self._validate_problem_definition(
                problem_definition
            )
        )

        dataset = self._validate_dataset(
            dataset
        )

        try:
            classical_output = classical_runner(
                problem_definition,
                dataset,
            )

            quantum_output = quantum_runner(
                problem_definition,
                dataset,
            )

            classical_metric = (
                self._validate_metric(
                    metric_function(
                        classical_output
                    ),
                    "classical_metric",
                )
            )

            quantum_metric = (
                self._validate_metric(
                    metric_function(
                        quantum_output
                    ),
                    "quantum_metric",
                )
            )

            performance_delta = (
                quantum_metric
                - classical_metric
            )

            relative_improvement = None

            if classical_metric != 0.0:
                relative_improvement = (
                    performance_delta
                    / abs(classical_metric)
                )

            conclusion = self._conclude(
                classical_metric,
                quantum_metric,
            )

            return QuantumClassicalBenchmarkResult(
                status=BenchmarkStatus.COMPLETED,
                classical_metric=classical_metric,
                quantum_metric=quantum_metric,
                performance_delta=performance_delta,
                relative_improvement=(
                    relative_improvement
                ),
                conclusion=conclusion,
                classical_output=classical_output,
                quantum_output=quantum_output,
                metadata={
                    "metric_name": self.metric_name,
                    "tolerance": self.tolerance,
                    "problem_definition": (
                        problem_definition
                    ),
                    "dataset_size": len(dataset),
                    "same_problem_definition": True,
                    "same_dataset": True,
                    "human_approval_required": (
                        self.require_human_approval
                    ),
                },
            )

        except Exception as exc:
            return self._failed_result(
                error=str(exc),
                metadata={
                    "metric_name": self.metric_name,
                    "tolerance": self.tolerance,
                    "problem_definition": (
                        problem_definition
                    ),
                    "dataset_size": len(dataset),
                },
            )


__all__ = [
    "BenchmarkStatus",
    "BenchmarkConclusion",
    "QuantumClassicalBenchmarkResult",
    "QuantumClassicalBenchmark",
]