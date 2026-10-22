from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class QuantumResearchStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class QuantumResearchMethod(str, Enum):
    QAOA = "QAOA"
    VQE = "VQE"
    QUANTUM_KERNEL = "QUANTUM_KERNEL"
    QUANTUM_MONTE_CARLO = "QUANTUM_MONTE_CARLO"


@dataclass(frozen=True)
class QuantumResearchResult:
    status: QuantumResearchStatus
    experiment_id: str
    method: QuantumResearchMethod
    problem_definition: str
    quantum_result: Any | None
    classical_result: Any | None
    benchmark_result: Any | None
    metadata: Mapping[str, object]
    error: str | None = None


class QuantumResearchOrchestrator:
    """
    v6.0.7 Quantum Research orchestration layer.

    Integrates:
        QAOA
        VQE
        Quantum Kernel / QML
        Quantum Monte Carlo
        Classical baseline
        Classical-vs-quantum benchmark

    The orchestrator controls execution flow and human approval,
    while algorithm-specific logic remains inside the dedicated
    research engines.
    """

    def __init__(
        self,
        *,
        require_human_approval: bool = True,
    ) -> None:
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
    def _validate_experiment_id(
        value: object,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "experiment_id must be a string."
            )

        value = value.strip()

        if not value:
            raise ValueError(
                "experiment_id must not be empty."
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
    def _validate_method(
        value: object,
    ) -> QuantumResearchMethod:
        if not isinstance(
            value,
            QuantumResearchMethod,
        ):
            raise ValueError(
                "method must be a QuantumResearchMethod."
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

    @staticmethod
    def _validate_callable(
        value: object,
        name: str,
        *,
        allow_none: bool = True,
    ) -> None:
        if value is None and allow_none:
            return

        if not callable(value):
            raise ValueError(
                f"{name} must be callable or None."
            )

    @staticmethod
    def _validate_approval(
        value: object,
    ) -> bool:
        if not isinstance(value, bool):
            raise ValueError(
                "human_approval must be a boolean."
            )

        return value

    @staticmethod
    def _validate_output(
        value: object,
        name: str,
    ) -> object:
        if value is None:
            raise ValueError(
                f"{name} returned None."
            )

        return value

    @staticmethod
    def _copy_metadata(
        metadata: Mapping[str, object] | None,
    ) -> dict[str, object]:
        if metadata is None:
            return {}

        if not isinstance(
            metadata,
            Mapping,
        ):
            raise ValueError(
                "metadata must be a mapping or None."
            )

        return dict(metadata)

    @staticmethod
    def _failed_result(
        *,
        experiment_id: str,
        method: QuantumResearchMethod,
        problem_definition: str,
        error: str,
        metadata: Mapping[str, object] | None = None,
    ) -> QuantumResearchResult:
        return QuantumResearchResult(
            status=QuantumResearchStatus.FAILED,
            experiment_id=experiment_id,
            method=method,
            problem_definition=problem_definition,
            quantum_result=None,
            classical_result=None,
            benchmark_result=None,
            metadata=dict(metadata or {}),
            error=error,
        )

    @staticmethod
    def _blocked_result(
        *,
        experiment_id: str,
        method: QuantumResearchMethod,
        problem_definition: str,
        dataset_size: int,
        human_approval_required: bool,
    ) -> QuantumResearchResult:
        return QuantumResearchResult(
            status=QuantumResearchStatus.BLOCKED,
            experiment_id=experiment_id,
            method=method,
            problem_definition=problem_definition,
            quantum_result=None,
            classical_result=None,
            benchmark_result=None,
            metadata={
                "human_approval_required": (
                    human_approval_required
                ),
                "human_approved": False,
                "dataset_size": dataset_size,
                "execution_started": False,
                "classical_baseline_executed": False,
                "benchmark_executed": False,
                "same_problem_definition": True,
                "same_dataset": True,
            },
            error=(
                "Execution blocked because human "
                "approval was not granted."
            ),
        )

    def run(
        self,
        *,
        experiment_id: str,
        method: QuantumResearchMethod,
        problem_definition: str,
        dataset: Sequence[object],
        quantum_runner: Callable[
            [str, tuple[object, ...]],
            object,
        ],
        classical_runner: Callable[
            [str, tuple[object, ...]],
            object,
        ] | None = None,
        benchmark_runner: Callable[
            [
                str,
                tuple[object, ...],
                object,
                object,
            ],
            object,
        ] | None = None,
        human_approval: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> QuantumResearchResult:

        experiment_id = self._validate_experiment_id(
            experiment_id
        )

        problem_definition = (
            self._validate_problem_definition(
                problem_definition
            )
        )

        method = self._validate_method(method)

        dataset = self._validate_dataset(dataset)

        self._validate_callable(
            quantum_runner,
            "quantum_runner",
            allow_none=False,
        )

        self._validate_callable(
            classical_runner,
            "classical_runner",
        )

        self._validate_callable(
            benchmark_runner,
            "benchmark_runner",
        )

        human_approval = self._validate_approval(
            human_approval
        )

        user_metadata = self._copy_metadata(
            metadata
        )

        if (
            self.require_human_approval
            and not human_approval
        ):
            return self._blocked_result(
                experiment_id=experiment_id,
                method=method,
                problem_definition=problem_definition,
                dataset_size=len(dataset),
                human_approval_required=(
                    self.require_human_approval
                ),
            )

        base_metadata = {
            **user_metadata,
            "human_approval_required": (
                self.require_human_approval
            ),
            "human_approved": human_approval,
            "dataset_size": len(dataset),
            "execution_started": True,
            "classical_baseline_executed": False,
            "benchmark_executed": False,
            "same_problem_definition": True,
            "same_dataset": True,
        }

        try:
            quantum_result = quantum_runner(
                problem_definition,
                dataset,
            )

            quantum_result = self._validate_output(
                quantum_result,
                "quantum_runner",
            )

            classical_result = None

            if classical_runner is not None:
                classical_result = (
                    classical_runner(
                        problem_definition,
                        dataset,
                    )
                )

                classical_result = (
                    self._validate_output(
                        classical_result,
                        "classical_runner",
                    )
                )

                base_metadata[
                    "classical_baseline_executed"
                ] = True

            benchmark_result = None

            if benchmark_runner is not None:
                if classical_runner is None:
                    raise ValueError(
                        "benchmark_runner requires "
                        "a classical_runner."
                    )

                benchmark_result = (
                    benchmark_runner(
                        problem_definition,
                        dataset,
                        classical_result,
                        quantum_result,
                    )
                )

                benchmark_result = (
                    self._validate_output(
                        benchmark_result,
                        "benchmark_runner",
                    )
                )

                base_metadata[
                    "benchmark_executed"
                ] = True

            return QuantumResearchResult(
                status=QuantumResearchStatus.COMPLETED,
                experiment_id=experiment_id,
                method=method,
                problem_definition=problem_definition,
                quantum_result=quantum_result,
                classical_result=classical_result,
                benchmark_result=benchmark_result,
                metadata=base_metadata,
                error=None,
            )

        except Exception as exc:
            return self._failed_result(
                experiment_id=experiment_id,
                method=method,
                problem_definition=problem_definition,
                error=str(exc),
                metadata=base_metadata,
            )


__all__ = [
    "QuantumResearchStatus",
    "QuantumResearchMethod",
    "QuantumResearchResult",
    "QuantumResearchOrchestrator",
]