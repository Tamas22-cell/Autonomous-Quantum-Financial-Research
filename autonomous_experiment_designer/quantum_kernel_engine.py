from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping, Sequence


class QuantumKernelMethod(str, Enum):
    """Supported quantum-kernel evaluation strategies."""

    EXACT = "EXACT"
    APPROXIMATE = "APPROXIMATE"


class QuantumKernelExecutionStatus(str, Enum):
    """Execution state of a quantum-kernel experiment."""

    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class QuantumKernelResult:
    """Immutable result of a quantum-kernel experiment."""

    status: QuantumKernelExecutionStatus
    method: QuantumKernelMethod
    kernel_matrix: tuple[tuple[float, ...], ...]
    similarity_scores: tuple[float, ...]
    average_similarity: float | None
    evaluations: int
    metadata: Mapping[str, object]
    error: str | None = None


class QuantumKernelEngine:
    """
    Backend-agnostic quantum-kernel experiment engine.

    The engine represents the classical orchestration layer around
    a quantum feature map. A supplied kernel function represents
    the measured quantum-kernel value between two feature vectors.

    The component deliberately does not require Qiskit or a quantum
    backend, making it deterministic and independently testable.

    Kernel values are expected to lie in [0, 1].
    """

    def __init__(
        self,
        *,
        feature_dimension: int = 2,
        method: QuantumKernelMethod = QuantumKernelMethod.EXACT,
        max_evaluations: int = 100,
        diagonal_value: float = 1.0,
    ) -> None:
        self.feature_dimension = (
            self._validate_positive_int(
                feature_dimension,
                "feature_dimension",
            )
        )

        self.method = self._validate_method(method)

        self.max_evaluations = (
            self._validate_positive_int(
                max_evaluations,
                "max_evaluations",
            )
        )

        self.diagonal_value = (
            self._validate_kernel_value(
                diagonal_value,
                "diagonal_value",
            )
        )

    @staticmethod
    def _validate_positive_int(
        value: object,
        field_name: str,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                f"{field_name} must be a positive integer."
            )

        return value

    @staticmethod
    def _validate_method(
        value: object,
    ) -> QuantumKernelMethod:
        if isinstance(value, QuantumKernelMethod):
            return value

        raise ValueError(
            "method must be a QuantumKernelMethod."
        )

    @staticmethod
    def _validate_kernel_value(
        value: object,
        field_name: str = "kernel_value",
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

        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

        return value

    def _validate_vector(
        self,
        vector: object,
    ) -> tuple[float, ...]:
        if vector is None:
            raise ValueError(
                "feature vector cannot be None."
            )

        if isinstance(vector, (str, bytes)):
            raise ValueError(
                "feature vector must be numeric."
            )

        try:
            values = tuple(vector)
        except TypeError as exc:
            raise ValueError(
                "feature vector must be a sequence."
            ) from exc

        if len(values) != self.feature_dimension:
            raise ValueError(
                "feature vector dimension does not match "
                "feature_dimension."
            )

        validated = []

        for index, value in enumerate(values):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"feature {index} must be numeric."
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"feature {index} must be finite."
                )

            validated.append(value)

        return tuple(validated)

    def _validate_dataset(
        self,
        dataset: object,
    ) -> tuple[tuple[float, ...], ...]:
        if dataset is None:
            raise ValueError(
                "dataset cannot be None."
            )

        if isinstance(dataset, (str, bytes)):
            raise ValueError(
                "dataset must be a sequence of feature vectors."
            )

        try:
            rows = tuple(dataset)
        except TypeError as exc:
            raise ValueError(
                "dataset must be a sequence of feature vectors."
            ) from exc

        if not rows:
            raise ValueError(
                "dataset cannot be empty."
            )

        return tuple(
            self._validate_vector(row)
            for row in rows
        )

    @staticmethod
    def _validate_kernel_function(
        kernel_function: object,
    ) -> None:
        if not callable(kernel_function):
            raise ValueError(
                "kernel_function must be callable."
            )

    def _evaluate_kernel(
        self,
        kernel_function: Callable[
            [tuple[float, ...], tuple[float, ...]],
            float,
        ],
        left: tuple[float, ...],
        right: tuple[float, ...],
    ) -> float:
        value = kernel_function(left, right)

        return self._validate_kernel_value(value)

    @staticmethod
    def _is_symmetric(
        matrix: tuple[tuple[float, ...], ...],
        tolerance: float = 1e-12,
    ) -> bool:
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                if abs(
                    value
                    - matrix[column_index][row_index]
                ) > tolerance:
                    return False

        return True

    def _failed_result(
        self,
        *,
        error: str,
        evaluations: int = 0,
    ) -> QuantumKernelResult:
        return QuantumKernelResult(
            status=QuantumKernelExecutionStatus.FAILED,
            method=self.method,
            kernel_matrix=(),
            similarity_scores=(),
            average_similarity=None,
            evaluations=evaluations,
            metadata={
                "feature_dimension": self.feature_dimension,
                "max_evaluations": self.max_evaluations,
                "diagonal_value": self.diagonal_value,
            },
            error=error,
        )

    def compute(
        self,
        dataset: Sequence[Sequence[float]],
        kernel_function: Callable[
            [tuple[float, ...], tuple[float, ...]],
            float,
        ],
    ) -> QuantumKernelResult:
        """
        Compute a kernel matrix for a feature dataset.

        EXACT evaluates all matrix entries.

        APPROXIMATE evaluates only the upper triangle and mirrors
        the values, reducing kernel-function evaluations.
        """

        self._validate_kernel_function(
            kernel_function
        )

        try:
            validated_dataset = self._validate_dataset(
                dataset
            )

            size = len(validated_dataset)

            theoretical_evaluations = (
                size * size
                if self.method
                == QuantumKernelMethod.EXACT
                else size * (size + 1) // 2
            )

            if (
                theoretical_evaluations
                > self.max_evaluations
            ):
                raise ValueError(
                    "dataset requires more kernel evaluations "
                    "than max_evaluations."
                )

            matrix = [
                [0.0 for _ in range(size)]
                for _ in range(size)
            ]

            evaluations = 0

            for i in range(size):
                start_j = (
                    i
                    if self.method
                    == QuantumKernelMethod.APPROXIMATE
                    else 0
                )

                for j in range(start_j, size):
                    if (
                        evaluations
                        >= self.max_evaluations
                    ):
                        raise RuntimeError(
                            "Maximum kernel evaluation limit reached."
                        )

                    if i == j:
                        value = self.diagonal_value
                    else:
                        value = self._evaluate_kernel(
                            kernel_function,
                            validated_dataset[i],
                            validated_dataset[j],
                        )

                    matrix[i][j] = value

                    if (
                        self.method
                        == QuantumKernelMethod.APPROXIMATE
                        and i != j
                    ):
                        matrix[j][i] = value

                    evaluations += 1

            frozen_matrix = tuple(
                tuple(row)
                for row in matrix
            )

            similarity_scores = []

            for i in range(size):
                for j in range(i + 1, size):
                    similarity_scores.append(
                        frozen_matrix[i][j]
                    )

            average_similarity = None

            if similarity_scores:
                average_similarity = (
                    sum(similarity_scores)
                    / len(similarity_scores)
                )

            return QuantumKernelResult(
                status=(
                    QuantumKernelExecutionStatus.COMPLETED
                ),
                method=self.method,
                kernel_matrix=frozen_matrix,
                similarity_scores=tuple(
                    similarity_scores
                ),
                average_similarity=average_similarity,
                evaluations=evaluations,
                metadata={
                    "feature_dimension": (
                        self.feature_dimension
                    ),
                    "dataset_size": size,
                    "max_evaluations": (
                        self.max_evaluations
                    ),
                    "diagonal_value": (
                        self.diagonal_value
                    ),
                    "symmetric": self._is_symmetric(
                        frozen_matrix
                    ),
                },
            )

        except Exception as exc:
            return self._failed_result(
                error=str(exc),
                evaluations=0,
            )


__all__ = [
    "QuantumKernelMethod",
    "QuantumKernelExecutionStatus",
    "QuantumKernelResult",
    "QuantumKernelEngine",
]