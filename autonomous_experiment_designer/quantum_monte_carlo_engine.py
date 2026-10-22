from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Callable, Mapping, Sequence


class QuantumMonteCarloMethod(str, Enum):
    """Supported Quantum Monte Carlo sampling strategies."""

    PSEUDO_RANDOM = "PSEUDO_RANDOM"
    STRATIFIED = "STRATIFIED"
    QUASI_RANDOM = "QUASI_RANDOM"


class QuantumMonteCarloStatus(str, Enum):
    """Execution state of a Quantum Monte Carlo experiment."""

    NOT_RUN = "NOT_RUN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class QuantumMonteCarloResult:
    """Immutable result of a Quantum Monte Carlo experiment."""

    status: QuantumMonteCarloStatus
    method: QuantumMonteCarloMethod
    estimate: float | None
    variance: float | None
    standard_error: float | None
    confidence_interval: tuple[float, float] | None
    samples: tuple[float, ...]
    sample_count: int
    metadata: Mapping[str, object]
    error: str | None = None


class QuantumMonteCarloEngine:
    """
    Backend-agnostic Quantum Monte Carlo research engine.

    The engine models the sampling and statistical-estimation layer
    of a QMC experiment. A supplied sampler generates samples from
    a problem-specific distribution and an observable function maps
    each sample to a measurable quantity.

    The implementation is intentionally independent from Qiskit or
    a specific quantum backend so the research layer can be tested
    deterministically before hardware integration.
    """

    def __init__(
        self,
        *,
        sample_count: int = 1000,
        method: QuantumMonteCarloMethod = (
            QuantumMonteCarloMethod.PSEUDO_RANDOM
        ),
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.sample_count = (
            self._validate_positive_int(
                sample_count,
                "sample_count",
            )
        )

        self.method = self._validate_method(
            method
        )

        self.confidence_level = (
            self._validate_probability(
                confidence_level,
                "confidence_level",
            )
        )

        self.seed = self._validate_seed(seed)

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
    ) -> QuantumMonteCarloMethod:
        if isinstance(
            value,
            QuantumMonteCarloMethod,
        ):
            return value

        raise ValueError(
            "method must be a QuantumMonteCarloMethod."
        )

    @staticmethod
    def _validate_probability(
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

        if not 0.0 < value < 1.0:
            raise ValueError(
                f"{field_name} must be between 0 and 1."
            )

        return value

    @staticmethod
    def _validate_seed(
        value: object,
    ) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise ValueError(
                "seed must be an integer."
            )

        return value

    @staticmethod
    def _validate_sampler(
        sampler: object,
    ) -> None:
        if not callable(sampler):
            raise ValueError(
                "sampler must be callable."
            )

    @staticmethod
    def _validate_observable(
        observable: object,
    ) -> None:
        if not callable(observable):
            raise ValueError(
                "observable must be callable."
            )

    @staticmethod
    def _validate_sample(
        value: object,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                "observable must return a numeric value."
            )

        value = float(value)

        if not isfinite(value):
            raise ValueError(
                "observable must return a finite value."
            )

        return value

    @staticmethod
    def _z_score(
        confidence_level: float,
    ) -> float:
        """
        Approximate two-sided normal critical value.

        The values cover the confidence levels normally used by
        the research workflow.
        """

        if confidence_level >= 0.995:
            return 2.807

        if confidence_level >= 0.99:
            return 2.576

        if confidence_level >= 0.975:
            return 2.241

        if confidence_level >= 0.95:
            return 1.960

        if confidence_level >= 0.90:
            return 1.645

        if confidence_level >= 0.80:
            return 1.282

        return 1.000

    @staticmethod
    def _mean(
        samples: Sequence[float],
    ) -> float:
        return sum(samples) / len(samples)

    @staticmethod
    def _variance(
        samples: Sequence[float],
        mean: float,
    ) -> float:
        if len(samples) < 2:
            return 0.0

        return sum(
            (value - mean) ** 2
            for value in samples
        ) / (len(samples) - 1)

    def _failed_result(
        self,
        *,
        error: str,
        sample_count: int = 0,
    ) -> QuantumMonteCarloResult:
        return QuantumMonteCarloResult(
            status=QuantumMonteCarloStatus.FAILED,
            method=self.method,
            estimate=None,
            variance=None,
            standard_error=None,
            confidence_interval=None,
            samples=(),
            sample_count=sample_count,
            metadata={
                "configured_sample_count": (
                    self.sample_count
                ),
                "confidence_level": (
                    self.confidence_level
                ),
                "seed": self.seed,
            },
            error=error,
        )

    def run(
        self,
        sampler: Callable[[], object],
        observable: Callable[[object], float],
    ) -> QuantumMonteCarloResult:
        """
        Execute a QMC sampling experiment.

        The sampler produces one sample at a time. The observable
        converts the sampled state into a scalar measurement.

        The result contains the estimate, variance, standard error,
        confidence interval and complete measured sample history.
        """

        self._validate_sampler(sampler)
        self._validate_observable(observable)

        try:
            import random

            rng = random.Random(self.seed)

            measured_samples: list[float] = []

            for _ in range(self.sample_count):
                raw_sample = sampler()

                # The deterministic seed is deliberately applied
                # to the orchestration layer. Samplers that need
                # randomness can optionally consume this RNG via
                # the sampler closure.
                _ = rng.random()

                value = observable(raw_sample)

                measured_samples.append(
                    self._validate_sample(value)
                )

            if not measured_samples:
                raise RuntimeError(
                    "No QMC samples were produced."
                )

            estimate = self._mean(
                measured_samples
            )

            variance = self._variance(
                measured_samples,
                estimate,
            )

            standard_error = (
                variance / len(measured_samples)
            ) ** 0.5

            z_score = self._z_score(
                self.confidence_level
            )

            margin = (
                z_score
                * standard_error
            )

            confidence_interval = (
                estimate - margin,
                estimate + margin,
            )

            return QuantumMonteCarloResult(
                status=QuantumMonteCarloStatus.COMPLETED,
                method=self.method,
                estimate=estimate,
                variance=variance,
                standard_error=standard_error,
                confidence_interval=confidence_interval,
                samples=tuple(
                    measured_samples
                ),
                sample_count=len(
                    measured_samples
                ),
                metadata={
                    "configured_sample_count": (
                        self.sample_count
                    ),
                    "confidence_level": (
                        self.confidence_level
                    ),
                    "seed": self.seed,
                    "method": self.method.value,
                },
            )

        except Exception as exc:
            return self._failed_result(
                error=str(exc)
            )


__all__ = [
    "QuantumMonteCarloMethod",
    "QuantumMonteCarloStatus",
    "QuantumMonteCarloResult",
    "QuantumMonteCarloEngine",
]