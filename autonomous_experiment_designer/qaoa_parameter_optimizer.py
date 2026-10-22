from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping, Sequence


_UNSET = object()


class QAOAOptimizerMethod(str, Enum):
    GRID = "GRID"
    RANDOM = "RANDOM"
    COBYLA = "COBYLA"
    SPSA = "SPSA"


class QAOAOptimizationStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class QAOAOptimizationResult:
    status: QAOAOptimizationStatus
    method: QAOAOptimizerMethod
    best_parameters: tuple[float, ...]
    best_objective: float | None
    evaluations: int
    convergence_history: tuple[float, ...]
    metadata: Mapping[str, object]
    error: str | None = None


class QAOAParameterOptimizer:
    """
    Controlled QAOA parameter-search engine.

    Invalid API configuration raises ValueError.
    Runtime objective failures are captured as FAILED results.
    """

    def __init__(
        self,
        *,
        parameter_count: int = 2,
        method: QAOAOptimizerMethod = QAOAOptimizerMethod.GRID,
        parameter_bounds: Sequence[tuple[float, float]] | None = None,
        grid_points: int = 5,
        max_evaluations: int = 100,
    ) -> None:
        self.parameter_count = self._validate_parameter_count(
            parameter_count
        )
        self.method = self._validate_method(method)
        self.parameter_bounds = self._validate_bounds(
            parameter_bounds,
            self.parameter_count,
        )
        self.grid_points = self._validate_positive_int(
            grid_points,
            "grid_points",
        )
        self.max_evaluations = self._validate_positive_int(
            max_evaluations,
            "max_evaluations",
        )

    @staticmethod
    def _validate_parameter_count(
        value: object,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise ValueError(
                "parameter_count must be a positive integer."
            )
        return value

    @staticmethod
    def _validate_method(
        value: object,
    ) -> QAOAOptimizerMethod:
        if isinstance(value, QAOAOptimizerMethod):
            return value

        raise ValueError(
            "method must be a QAOAOptimizerMethod."
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
    def _validate_bounds(
        bounds: Sequence[tuple[float, float]] | None,
        parameter_count: int,
    ) -> tuple[tuple[float, float], ...]:
        if bounds is None:
            return tuple(
                (0.0, 2.0)
                for _ in range(parameter_count)
            )

        if isinstance(bounds, (str, bytes)):
            raise ValueError(
                "parameter_bounds must be a sequence."
            )

        try:
            result = tuple(bounds)
        except TypeError as exc:
            raise ValueError(
                "parameter_bounds must be a sequence."
            ) from exc

        if len(result) != parameter_count:
            raise ValueError(
                "parameter_bounds length must match "
                "parameter_count."
            )

        validated = []

        for index, bound in enumerate(result):
            if (
                not isinstance(bound, (tuple, list))
                or len(bound) != 2
            ):
                raise ValueError(
                    f"parameter_bounds[{index}] must contain "
                    "exactly two values."
                )

            lower, upper = bound

            if (
                isinstance(lower, bool)
                or isinstance(upper, bool)
                or not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
            ):
                raise ValueError(
                    f"parameter_bounds[{index}] must contain "
                    "numeric values."
                )

            lower = float(lower)
            upper = float(upper)

            if not isfinite(lower) or not isfinite(upper):
                raise ValueError(
                    f"parameter_bounds[{index}] must contain "
                    "finite values."
                )

            if lower >= upper:
                raise ValueError(
                    f"parameter_bounds[{index}] requires "
                    "lower < upper."
                )

            validated.append((lower, upper))

        return tuple(validated)

    @staticmethod
    def _validate_objective(
        objective: object,
    ) -> None:
        if not callable(objective):
            raise ValueError(
                "objective must be callable."
            )

    def _validate_parameters(
        self,
        parameters: object,
    ) -> tuple[float, ...]:
        if parameters is None:
            raise ValueError(
                "initial_parameters cannot be None "
                "when explicitly provided."
            )

        if isinstance(parameters, (str, bytes)):
            raise ValueError(
                "parameters must be numeric."
            )

        try:
            result = tuple(parameters)
        except TypeError as exc:
            raise ValueError(
                "parameters must be a sequence."
            ) from exc

        if len(result) != self.parameter_count:
            raise ValueError(
                "parameter count does not match "
                "parameter_count."
            )

        validated = []

        for index, value in enumerate(result):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise ValueError(
                    f"parameter {index} must be numeric."
                )

            value = float(value)

            if not isfinite(value):
                raise ValueError(
                    f"parameter {index} must be finite."
                )

            lower, upper = self.parameter_bounds[index]

            if value < lower or value > upper:
                raise ValueError(
                    f"parameter {index} is outside "
                    "its bounds."
                )

            validated.append(value)

        return tuple(validated)

    def _evaluate(
        self,
        objective: Callable[[tuple[float, ...]], float],
        parameters: tuple[float, ...],
    ) -> float:
        value = objective(parameters)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "objective must return a numeric value."
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                "objective must return a finite value."
            )

        return value

    def _grid_candidates(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        if self.grid_points == 1:
            return (
                tuple(
                    (lower + upper) / 2.0
                    for lower, upper in self.parameter_bounds
                ),
            )

        axes = []

        for lower, upper in self.parameter_bounds:
            step = (
                upper - lower
            ) / (self.grid_points - 1)

            axes.append(
                tuple(
                    lower + index * step
                    for index in range(self.grid_points)
                )
            )

        candidates = []

        def build(
            dimension: int,
            current: list[float],
        ) -> None:
            if len(candidates) >= self.max_evaluations:
                return

            if dimension == len(axes):
                candidates.append(tuple(current))
                return

            for value in axes[dimension]:
                current.append(value)

                build(
                    dimension + 1,
                    current,
                )

                current.pop()

                if len(candidates) >= self.max_evaluations:
                    break

        build(0, [])

        return tuple(candidates)

    def _random_candidates(
        self,
        *,
        seed: int,
    ) -> tuple[tuple[float, ...], ...]:
        import random

        rng = random.Random(seed)

        return tuple(
            tuple(
                rng.uniform(lower, upper)
                for lower, upper in self.parameter_bounds
            )
            for _ in range(self.max_evaluations)
        )

    def _local_candidates(
        self,
        initial: tuple[float, ...],
    ) -> tuple[tuple[float, ...], ...]:
        candidates = [initial]
        step = 0.25

        for index in range(self.parameter_count):
            for direction in (-1.0, 1.0):
                candidate = list(initial)

                lower, upper = self.parameter_bounds[index]

                candidate[index] = min(
                    upper,
                    max(
                        lower,
                        candidate[index]
                        + direction * step,
                    ),
                )

                candidate_tuple = tuple(candidate)

                if candidate_tuple not in candidates:
                    candidates.append(candidate_tuple)

                if len(candidates) >= self.max_evaluations:
                    return tuple(candidates)

        return tuple(candidates)

    def _failed_result(
        self,
        *,
        error: str,
        seed: object,
    ) -> QAOAOptimizationResult:
        return QAOAOptimizationResult(
            status=QAOAOptimizationStatus.FAILED,
            method=self.method,
            best_parameters=(),
            best_objective=None,
            evaluations=0,
            convergence_history=(),
            metadata={
                "parameter_count": self.parameter_count,
                "max_evaluations": self.max_evaluations,
                "grid_points": self.grid_points,
                "seed": seed,
            },
            error=error,
        )

    def optimize(
        self,
        objective: Callable[[tuple[float, ...]], float],
        *,
        seed: int = 42,
        initial_parameters: object = _UNSET,
    ) -> QAOAOptimizationResult:

        # API validation intentionally happens BEFORE
        # the runtime execution try/except.
        self._validate_objective(objective)

        try:
            if (
                isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                raise ValueError(
                    "seed must be an integer."
                )

            if initial_parameters is _UNSET:
                initial = tuple(
                    (lower + upper) / 2.0
                    for lower, upper in self.parameter_bounds
                )
            else:
                initial = self._validate_parameters(
                    initial_parameters
                )

            if self.method == QAOAOptimizerMethod.GRID:
                candidates = self._grid_candidates()

            elif self.method == QAOAOptimizerMethod.RANDOM:
                candidates = self._random_candidates(
                    seed=seed
                )

            else:
                candidates = self._local_candidates(
                    initial
                )

            best_parameters = None
            best_objective = None
            history = []
            evaluations = 0

            for parameters in candidates:
                if evaluations >= self.max_evaluations:
                    break

                value = self._evaluate(
                    objective,
                    parameters,
                )

                evaluations += 1
                history.append(value)

                if (
                    best_objective is None
                    or value < best_objective
                ):
                    best_objective = value
                    best_parameters = parameters

            if best_parameters is None:
                raise RuntimeError(
                    "No valid parameter evaluation was produced."
                )

            return QAOAOptimizationResult(
                status=QAOAOptimizationStatus.COMPLETED,
                method=self.method,
                best_parameters=best_parameters,
                best_objective=best_objective,
                evaluations=evaluations,
                convergence_history=tuple(history),
                metadata={
                    "parameter_count": self.parameter_count,
                    "max_evaluations": self.max_evaluations,
                    "grid_points": self.grid_points,
                    "seed": seed,
                },
            )

        except Exception as exc:
            return self._failed_result(
                error=str(exc),
                seed=seed,
            )


__all__ = [
    "QAOAOptimizerMethod",
    "QAOAOptimizationStatus",
    "QAOAOptimizationResult",
    "QAOAParameterOptimizer",
]