from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from autonomous_experiment_designer.uncertainty_detector import (
    Severity,
    UncertaintyReport,
    UncertaintyType,
)


class ExperimentType(str, Enum):
    CONFIDENCE_VALIDATION = "CONFIDENCE_VALIDATION"
    REGIME_STABILITY = "REGIME_STABILITY"
    PERFORMANCE_REPLICATION = "PERFORMANCE_REPLICATION"
    EVIDENCE_RECONCILIATION = "EVIDENCE_RECONCILIATION"
    MULTI_FACTOR_VALIDATION = "MULTI_FACTOR_VALIDATION"


class ValidationMethod(str, Enum):
    HOLDOUT = "HOLDOUT"
    WALK_FORWARD = "WALK_FORWARD"
    CROSS_VALIDATION = "CROSS_VALIDATION"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    STRESS_TEST = "STRESS_TEST"


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    experiment_type: ExperimentType
    objective: str
    datasets: tuple[str, ...]
    agents: tuple[str, ...]
    models: tuple[str, ...]
    validation_methods: tuple[ValidationMethod, ...]
    priority: int
    severity: Severity
    requires_human_approval: bool
    source_signals: tuple[UncertaintyType, ...]


class ExperimentDesigner:
    """
    Converts uncertainty signals into deterministic experiment plans.

    This component only designs experiments.
    It never executes an experiment and never bypasses
    human approval.
    """

    def __init__(
        self,
        *,
        default_datasets: Iterable[str] = (
            "historical_market_data",
            "macro_economic_data",
            "portfolio_performance_history",
        ),
        default_agents: Iterable[str] = (
            "macro_agent",
            "technical_agent",
            "risk_agent",
        ),
        default_models: Iterable[str] = (
            "baseline_model",
            "challenger_model",
        ),
    ) -> None:
        self._datasets = self._validate_names(
            default_datasets,
            "default_datasets",
        )
        self._agents = self._validate_names(
            default_agents,
            "default_agents",
        )
        self._models = self._validate_names(
            default_models,
            "default_models",
        )

    @staticmethod
    def _validate_names(
        values: Iterable[str],
        field_name: str,
    ) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError(
                f"{field_name} must be an iterable of names."
            )

        result = tuple(values)

        if not result:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        if any(
            not isinstance(value, str) or not value.strip()
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
    def _validate_report(
        report: UncertaintyReport,
    ) -> None:
        if not isinstance(report, UncertaintyReport):
            raise TypeError(
                "report must be an UncertaintyReport."
            )

    @staticmethod
    def _priority(
        severity: Severity,
    ) -> int:
        return {
            Severity.LOW: 4,
            Severity.MEDIUM: 3,
            Severity.HIGH: 2,
            Severity.CRITICAL: 1,
        }[severity]

    @staticmethod
    def _configuration_for_signal(
        signal_type: UncertaintyType,
    ) -> tuple[
        ExperimentType,
        tuple[ValidationMethod, ...],
        str,
    ]:
        if signal_type == UncertaintyType.CONFIDENCE_GAP:
            return (
                ExperimentType.CONFIDENCE_VALIDATION,
                (
                    ValidationMethod.CROSS_VALIDATION,
                    ValidationMethod.OUT_OF_SAMPLE,
                ),
                (
                    "Validate whether the current model confidence "
                    "is supported by independent evidence and "
                    "out-of-sample performance."
                ),
            )

        if signal_type == UncertaintyType.REGIME_CHANGE:
            return (
                ExperimentType.REGIME_STABILITY,
                (
                    ValidationMethod.WALK_FORWARD,
                    ValidationMethod.STRESS_TEST,
                ),
                (
                    "Test whether the model and research assumptions "
                    "remain valid across the detected regime change."
                ),
            )

        if signal_type == UncertaintyType.PERFORMANCE_GAP:
            return (
                ExperimentType.PERFORMANCE_REPLICATION,
                (
                    ValidationMethod.HOLDOUT,
                    ValidationMethod.OUT_OF_SAMPLE,
                ),
                (
                    "Replicate the observed performance gap using "
                    "independent historical data and a holdout period."
                ),
            )

        if signal_type == UncertaintyType.EVIDENCE_CONFLICT:
            return (
                ExperimentType.EVIDENCE_RECONCILIATION,
                (
                    ValidationMethod.CROSS_VALIDATION,
                    ValidationMethod.STRESS_TEST,
                ),
                (
                    "Reconcile conflicting evidence and determine "
                    "whether the underlying research hypothesis remains "
                    "supported."
                ),
            )

        raise ValueError(
            f"Unsupported uncertainty type: {signal_type}"
        )

    def design(
        self,
        report: UncertaintyReport,
        *,
        experiment_id: str,
    ) -> ExperimentPlan | None:
        self._validate_report(report)

        if (
            not isinstance(experiment_id, str)
            or not experiment_id.strip()
        ):
            raise ValueError(
                "experiment_id must be a non-empty string."
            )

        if not report.signals:
            return None

        primary_signal = report.signals[0]

        (
            experiment_type,
            validation_methods,
            objective,
        ) = self._configuration_for_signal(
            primary_signal.signal_type
        )

        if report.signal_count > 1:
            experiment_type = (
                ExperimentType.MULTI_FACTOR_VALIDATION
            )

            objective = (
                "Resolve multiple interacting uncertainty signals "
                "through a controlled multi-factor validation experiment."
            )

            validation_methods = (
                ValidationMethod.WALK_FORWARD,
                ValidationMethod.OUT_OF_SAMPLE,
                ValidationMethod.STRESS_TEST,
            )

        return ExperimentPlan(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            objective=objective,
            datasets=self._datasets,
            agents=self._agents,
            models=self._models,
            validation_methods=validation_methods,
            priority=self._priority(
                primary_signal.severity
            ),
            severity=primary_signal.severity,
            requires_human_approval=True,
            source_signals=tuple(
                signal.signal_type
                for signal in report.signals
            ),
        )


__all__ = [
    "ExperimentDesigner",
    "ExperimentPlan",
    "ExperimentType",
    "ValidationMethod",
]