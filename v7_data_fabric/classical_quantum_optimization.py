from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Dict, Iterable, Mapping, Optional, Tuple


class OptimizationError(ValueError):
    """Optimization validation or execution error."""


class OptimizationMethod(str, Enum):
    CLASSICAL = "CLASSICAL"
    QUANTUM_SIMULATION = "QUANTUM_SIMULATION"


class OptimizationStatus(str, Enum):
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class OptimizationProblem:
    asset_ids: Tuple[str, ...]
    expected_returns: Tuple[float, ...]
    risk_scores: Tuple[float, ...]
    budget: float = 1.0
    max_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.asset_ids:
            raise OptimizationError(
                "asset_ids cannot be empty."
            )

        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise OptimizationError(
                "asset_ids must be unique."
            )

        if len(self.expected_returns) != len(self.asset_ids):
            raise OptimizationError(
                "expected_returns length must match asset_ids."
            )

        if len(self.risk_scores) != len(self.asset_ids):
            raise OptimizationError(
                "risk_scores length must match asset_ids."
            )

        for asset_id in self.asset_ids:
            if not isinstance(asset_id, str) or not asset_id.strip():
                raise OptimizationError(
                    "asset_ids must contain non-empty strings."
                )

        for value in (
            *self.expected_returns,
            *self.risk_scores,
            self.budget,
            self.max_weight,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise OptimizationError(
                    "Problem values must be finite numbers."
                )

        if any(value < 0 for value in self.risk_scores):
            raise OptimizationError(
                "risk_scores cannot be negative."
            )

        if self.budget <= 0:
            raise OptimizationError(
                "budget must be greater than zero."
            )

        if not 0.0 < self.max_weight <= 1.0:
            raise OptimizationError(
                "max_weight must be between 0 and 1."
            )


@dataclass(frozen=True)
class OptimizationResult:
    method: OptimizationMethod
    status: OptimizationStatus
    weights: Mapping[str, float]
    expected_return: float
    risk_score: float
    objective_score: float
    research_only: bool
    real_money_execution: bool
    fingerprint: str

    @property
    def invested_weight(self) -> float:
        return sum(self.weights.values())


@dataclass
class ClassicalQuantumOptimizer:
    """
    v7.0.10 classical and quantum-inspired optimization facade.

    The quantum path is a deterministic research simulation adapter.
    It does not connect to quantum hardware and cannot execute trades.
    """

    VERSION = "v7.0.10"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    BROKER_CONNECTION = False
    QUANTUM_HARDWARE_EXECUTION = False

    _history: list[OptimizationResult] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    @staticmethod
    def _fingerprint(
        method: OptimizationMethod,
        weights: Mapping[str, float],
        expected_return: float,
        risk_score: float,
    ) -> str:
        payload = (
            method.value,
            tuple(sorted(weights.items())),
            round(expected_return, 12),
            round(risk_score, 12),
        )
        return f"opt-{abs(hash(payload)):016x}"

    @staticmethod
    def _validate_method(
        method: OptimizationMethod,
    ) -> None:
        if not isinstance(method, OptimizationMethod):
            raise OptimizationError(
                "method must be an OptimizationMethod value."
            )

    @staticmethod
    def _normalise(
        scores: Mapping[str, float],
        max_weight: float,
    ) -> Dict[str, float]:
        positive = {
            asset_id: max(0.0, score)
            for asset_id, score in scores.items()
        }

        total = sum(positive.values())

        if total <= 0:
            equal_weight = min(
                max_weight,
                1.0 / len(scores),
            )
            return {
                asset_id: equal_weight
                for asset_id in scores
            }

        weights = {
            asset_id: min(
                max_weight,
                score / total,
            )
            for asset_id, score in positive.items()
        }

        remaining = 1.0 - sum(weights.values())

        for asset_id in weights:
            if remaining <= 1e-12:
                break

            capacity = max_weight - weights[asset_id]
            addition = min(capacity, remaining)
            weights[asset_id] += addition
            remaining -= addition

        return weights

    def optimize(
        self,
        problem: OptimizationProblem,
        *,
        method: OptimizationMethod = (
            OptimizationMethod.CLASSICAL
        ),
    ) -> OptimizationResult:
        if not isinstance(problem, OptimizationProblem):
            raise OptimizationError(
                "problem must be an OptimizationProblem."
            )

        self._validate_method(method)

        scores = {
            asset_id: expected_return / (1.0 + risk)
            for asset_id, expected_return, risk in zip(
                problem.asset_ids,
                problem.expected_returns,
                problem.risk_scores,
            )
        }

        weights = self._normalise(
            scores,
            problem.max_weight,
        )

        expected_return = sum(
            weights[asset_id] * expected
            for asset_id, expected in zip(
                problem.asset_ids,
                problem.expected_returns,
            )
        )

        risk_score = sum(
            weights[asset_id] * risk
            for asset_id, risk in zip(
                problem.asset_ids,
                problem.risk_scores,
            )
        )

        objective_score = expected_return - risk_score

        result = OptimizationResult(
            method=method,
            status=OptimizationStatus.COMPLETED,
            weights=weights,
            expected_return=expected_return,
            risk_score=risk_score,
            objective_score=objective_score,
            research_only=self.RESEARCH_ONLY,
            real_money_execution=self.REAL_MONEY_EXECUTION,
            fingerprint=self._fingerprint(
                method,
                weights,
                expected_return,
                risk_score,
            ),
        )

        self._history.append(result)
        return result

    def history(self) -> Tuple[OptimizationResult, ...]:
        return tuple(self._history)

    def summary(self) -> Dict[str, object]:
        return {
            "component": "classical_quantum_optimization",
            "version": self.VERSION,
            "optimization_count": len(self._history),
            "has_last_result": bool(self._history),
            "supported_methods": [
                method.value
                for method in OptimizationMethod
            ],
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "broker_connection": self.BROKER_CONNECTION,
            "quantum_hardware_execution": (
                self.QUANTUM_HARDWARE_EXECUTION
            ),
            "automatic_trade_execution": False,
        }

    def snapshot(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "quantum_hardware_execution": (
                self.QUANTUM_HARDWARE_EXECUTION
            ),
            "history_count": len(self._history),
            "last_fingerprint": (
                self._history[-1].fingerprint
                if self._history
                else None
            ),
        }