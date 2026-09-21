from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from research_benchmarking.benchmark_engine import (
    BenchmarkEngine,
    BenchmarkStagePolicy,
)
from research_benchmarking.models import (
    ValidationConfig,
    ValidationResult,
)


class StatisticalValidationError(ValueError):
    """Statistical validation facade error."""


class StatisticalValidationFacade:
    """
    v7.0.6 statistical validation integration facade.

    The facade delegates statistical validation to the existing
    research_benchmarking implementation and adds a stable v7 API.

    This component is research-only:
    - no real-money execution
    - no autonomous capital allocation
    - no automatic alpha approval
    - anti-overfitting controls remain enabled
    """

    VERSION = "v7.0.6"
    RESEARCH_ONLY = True
    REAL_MONEY_EXECUTION = False
    ANTI_OVERFITTING_ENABLED = True

    def __init__(
        self,
        config: Optional[ValidationConfig] = None,
        *,
        policy: Optional[BenchmarkStagePolicy] = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self.policy = policy or BenchmarkStagePolicy()

        self._engine = BenchmarkEngine(
            config=self.config,
            policy=self.policy,
        )

        self._analysis_count = 0
        self._last_analysis: Optional[Dict[str, Any]] = None

    @property
    def version(self) -> str:
        return self.VERSION

    @property
    def analysis_count(self) -> int:
        return self._analysis_count

    @property
    def last_analysis(self) -> Optional[Dict[str, Any]]:
        if self._last_analysis is None:
            return None

        return dict(self._last_analysis)

    def analyze(
        self,
        results: Iterable[ValidationResult],
    ) -> Dict[str, Any]:
        """
        Run the complete statistical validation pipeline.

        The underlying benchmark engine performs:
        - stage aggregation
        - pass-ratio evaluation
        - out-of-sample checks
        - anti-overfitting checks
        - robustness assessment
        - benchmark decision generation
        """
        try:
            analysis = self._engine.analyze(results)
        except Exception as exc:
            if isinstance(exc, StatisticalValidationError):
                raise

            raise StatisticalValidationError(
                "Statistical validation failed."
            ) from exc

        result = dict(analysis)

        result.update(
            {
                "component": "statistical_validation",
                "version": self.VERSION,
                "research_only": self.RESEARCH_ONLY,
                "real_money_execution": (
                    self.REAL_MONEY_EXECUTION
                ),
                "anti_overfitting_enabled": (
                    self.ANTI_OVERFITTING_ENABLED
                ),
                "automatic_alpha_approval": False,
            }
        )

        self._analysis_count += 1
        self._last_analysis = dict(result)

        return result

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a deterministic configuration and safety snapshot.
        """
        try:
            engine_snapshot = self._engine.snapshot()
        except Exception as exc:
            raise StatisticalValidationError(
                "Could not create validation snapshot."
            ) from exc

        result = dict(engine_snapshot)

        result.update(
            {
                "component": "statistical_validation",
                "version": self.VERSION,
                "research_only": self.RESEARCH_ONLY,
                "real_money_execution": (
                    self.REAL_MONEY_EXECUTION
                ),
                "anti_overfitting_enabled": (
                    self.ANTI_OVERFITTING_ENABLED
                ),
                "automatic_alpha_approval": False,
                "analysis_count": self._analysis_count,
            }
        )

        return result

    def summary(self) -> Dict[str, Any]:
        """
        Return a compact operational summary.
        """
        return {
            "component": "statistical_validation",
            "version": self.VERSION,
            "analysis_count": self._analysis_count,
            "has_last_analysis": (
                self._last_analysis is not None
            ),
            "research_only": self.RESEARCH_ONLY,
            "real_money_execution": (
                self.REAL_MONEY_EXECUTION
            ),
            "anti_overfitting_enabled": (
                self.ANTI_OVERFITTING_ENABLED
            ),
            "automatic_alpha_approval": False,
            "required_stages": [
                stage.value
                for stage in self.policy.required_stages
            ],
        }