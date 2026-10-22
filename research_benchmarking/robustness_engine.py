from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from research_benchmarking.models import (
    ValidationConfig,
    ValidationConfigurationError,
    ValidationResult,
    ValidationStage,
    ValidationStatus,
    stable_hash,
)


VERSION = "v6.0.2"


# ============================================================
# Exceptions
# ============================================================


class RobustnessValidationError(Exception):
    """Raised when robustness validation cannot be performed."""


# ============================================================
# Helpers
# ============================================================


def _validate_numeric(
    value: float,
    name: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValidationConfigurationError(
            f"{name} must be numeric."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValidationConfigurationError(
            f"{name} must be finite."
        )

    return numeric


def _validate_non_negative(
    value: float,
    name: str,
) -> float:
    numeric = _validate_numeric(
        value,
        name,
    )

    if numeric < 0.0:
        raise ValidationConfigurationError(
            f"{name} must be non-negative."
        )

    return numeric


def _validate_ratio(
    value: float,
    name: str,
) -> float:
    numeric = _validate_numeric(
        value,
        name,
    )

    if not 0.0 <= numeric <= 1.0:
        raise ValidationConfigurationError(
            f"{name} must be between 0 and 1."
        )

    return numeric


def _validate_scores(
    scores: Sequence[float],
) -> List[float]:
    if not scores:
        raise RobustnessValidationError(
            "scores cannot be empty."
        )

    return [
        _validate_numeric(
            value,
            f"scores[{index}]",
        )
        for index, value in enumerate(scores)
    ]


# ============================================================
# Scenario model
# ============================================================


@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str
    category: str
    severity: float
    cost_bps: float = 0.0
    slippage_bps: float = 0.0
    execution_delay: int = 0
    parameter_shift: float = 0.0
    data_perturbation: float = 0.0
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scenario_id, str)
            or not self.scenario_id.strip()
        ):
            raise ValidationConfigurationError(
                "scenario_id must be a non-empty string."
            )

        if (
            not isinstance(self.category, str)
            or not self.category.strip()
        ):
            raise ValidationConfigurationError(
                "category must be a non-empty string."
            )

        _validate_ratio(
            self.severity,
            "severity",
        )

        _validate_non_negative(
            self.cost_bps,
            "cost_bps",
        )

        _validate_non_negative(
            self.slippage_bps,
            "slippage_bps",
        )

        if (
            isinstance(self.execution_delay, bool)
            or not isinstance(
                self.execution_delay,
                int,
            )
            or self.execution_delay < 0
        ):
            raise ValidationConfigurationError(
                "execution_delay must be a "
                "non-negative integer."
            )

        _validate_non_negative(
            self.parameter_shift,
            "parameter_shift",
        )

        _validate_non_negative(
            self.data_perturbation,
            "data_perturbation",
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "scenario_id": self.scenario_id,
                "category": self.category,
                "severity": self.severity,
                "cost_bps": self.cost_bps,
                "slippage_bps": self.slippage_bps,
                "execution_delay": (
                    self.execution_delay
                ),
                "parameter_shift": (
                    self.parameter_shift
                ),
                "data_perturbation": (
                    self.data_perturbation
                ),
                "metadata": self.metadata,
            }
        )


# ============================================================
# Scenario result
# ============================================================


@dataclass(frozen=True)
class RobustnessScenarioResult:
    scenario: RobustnessScenario
    baseline_score: float
    stressed_score: float
    degradation: float
    degradation_ratio: float
    positive_after_stress: bool
    stable: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.scenario,
            RobustnessScenario,
        ):
            raise ValidationConfigurationError(
                "scenario must be a RobustnessScenario."
            )

        _validate_numeric(
            self.baseline_score,
            "baseline_score",
        )

        _validate_numeric(
            self.stressed_score,
            "stressed_score",
        )

        _validate_numeric(
            self.degradation,
            "degradation",
        )

        _validate_numeric(
            self.degradation_ratio,
            "degradation_ratio",
        )

        if not isinstance(
            self.positive_after_stress,
            bool,
        ):
            raise ValidationConfigurationError(
                "positive_after_stress must be bool."
            )

        if not isinstance(
            self.stable,
            bool,
        ):
            raise ValidationConfigurationError(
                "stable must be bool."
            )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "scenario": (
                    self.scenario.fingerprint
                ),
                "baseline_score": (
                    self.baseline_score
                ),
                "stressed_score": (
                    self.stressed_score
                ),
                "degradation": self.degradation,
                "degradation_ratio": (
                    self.degradation_ratio
                ),
                "positive_after_stress": (
                    self.positive_after_stress
                ),
                "stable": self.stable,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Robustness report
# ============================================================


@dataclass(frozen=True)
class RobustnessReport:
    baseline_score: float
    scenario_results: List[
        RobustnessScenarioResult
    ]
    average_stressed_score: float
    average_degradation: float
    average_degradation_ratio: float
    positive_survival_ratio: float
    stable_scenario_ratio: float
    worst_stressed_score: float
    worst_degradation_ratio: float
    passed: bool
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_numeric(
            self.baseline_score,
            "baseline_score",
        )

        if not self.scenario_results:
            raise ValidationConfigurationError(
                "scenario_results cannot be empty."
            )

        for result in self.scenario_results:
            if not isinstance(
                result,
                RobustnessScenarioResult,
            ):
                raise ValidationConfigurationError(
                    "scenario_results must contain "
                    "RobustnessScenarioResult objects."
                )

        for name, value in {
            "average_stressed_score": (
                self.average_stressed_score
            ),
            "average_degradation": (
                self.average_degradation
            ),
            "average_degradation_ratio": (
                self.average_degradation_ratio
            ),
            "worst_stressed_score": (
                self.worst_stressed_score
            ),
            "worst_degradation_ratio": (
                self.worst_degradation_ratio
            ),
        }.items():
            _validate_numeric(
                value,
                name,
            )

        _validate_ratio(
            self.positive_survival_ratio,
            "positive_survival_ratio",
        )

        _validate_ratio(
            self.stable_scenario_ratio,
            "stable_scenario_ratio",
        )

        if not isinstance(
            self.passed,
            bool,
        ):
            raise ValidationConfigurationError(
                "passed must be bool."
            )

    @property
    def scenario_count(self) -> int:
        return len(
            self.scenario_results
        )

    @property
    def stable_count(self) -> int:
        return sum(
            1
            for result
            in self.scenario_results
            if result.stable
        )

    @property
    def positive_count(self) -> int:
        return sum(
            1
            for result
            in self.scenario_results
            if result.positive_after_stress
        )

    @property
    def failed_scenarios(
        self,
    ) -> List[str]:
        return [
            result.scenario.scenario_id
            for result
            in self.scenario_results
            if not result.stable
        ]

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "version": VERSION,
                "baseline_score": (
                    self.baseline_score
                ),
                "scenario_results": [
                    result.fingerprint
                    for result
                    in self.scenario_results
                ],
                "average_stressed_score": (
                    self.average_stressed_score
                ),
                "average_degradation": (
                    self.average_degradation
                ),
                "average_degradation_ratio": (
                    self.average_degradation_ratio
                ),
                "positive_survival_ratio": (
                    self.positive_survival_ratio
                ),
                "stable_scenario_ratio": (
                    self.stable_scenario_ratio
                ),
                "worst_stressed_score": (
                    self.worst_stressed_score
                ),
                "worst_degradation_ratio": (
                    self.worst_degradation_ratio
                ),
                "passed": self.passed,
                "metadata": self.metadata,
            }
        )


# ============================================================
# Robustness Engine
# ============================================================


class RobustnessEngine:
    """
    Stress-tests research alpha under adverse assumptions.

    Supported robustness dimensions:
    - transaction costs
    - slippage
    - execution delay
    - parameter perturbation
    - data perturbation

    The engine evaluates whether the observed research effect
    survives realistic degradation instead of relying only on
    the original backtest result.
    """

    VERSION = VERSION

    def __init__(
        self,
        config: Optional[
            ValidationConfig
        ] = None,
        *,
        maximum_degradation_ratio: float = 0.50,
        minimum_positive_survival_ratio: float = 0.70,
        minimum_stable_scenario_ratio: float = 0.70,
    ) -> None:
        self.config = (
            config
            or ValidationConfig()
        )

        self.maximum_degradation_ratio = (
            _validate_non_negative(
                maximum_degradation_ratio,
                "maximum_degradation_ratio",
            )
        )

        self.minimum_positive_survival_ratio = (
            _validate_ratio(
                minimum_positive_survival_ratio,
                "minimum_positive_survival_ratio",
            )
        )

        self.minimum_stable_scenario_ratio = (
            _validate_ratio(
                minimum_stable_scenario_ratio,
                "minimum_stable_scenario_ratio",
            )
        )

    # ========================================================
    # Default stress scenarios
    # ========================================================

    def build_default_scenarios(
        self,
    ) -> List[RobustnessScenario]:
        return [
            RobustnessScenario(
                scenario_id="cost_low",
                category="transaction_cost",
                severity=0.25,
                cost_bps=5.0,
            ),
            RobustnessScenario(
                scenario_id="cost_high",
                category="transaction_cost",
                severity=0.60,
                cost_bps=20.0,
            ),
            RobustnessScenario(
                scenario_id="slippage_low",
                category="slippage",
                severity=0.25,
                slippage_bps=5.0,
            ),
            RobustnessScenario(
                scenario_id="slippage_high",
                category="slippage",
                severity=0.60,
                slippage_bps=20.0,
            ),
            RobustnessScenario(
                scenario_id="delay_1",
                category="execution_delay",
                severity=0.30,
                execution_delay=1,
            ),
            RobustnessScenario(
                scenario_id="delay_3",
                category="execution_delay",
                severity=0.60,
                execution_delay=3,
            ),
            RobustnessScenario(
                scenario_id="parameter_5pct",
                category="parameter_shift",
                severity=0.25,
                parameter_shift=0.05,
            ),
            RobustnessScenario(
                scenario_id="parameter_20pct",
                category="parameter_shift",
                severity=0.65,
                parameter_shift=0.20,
            ),
            RobustnessScenario(
                scenario_id="data_noise_5pct",
                category="data_perturbation",
                severity=0.25,
                data_perturbation=0.05,
            ),
            RobustnessScenario(
                scenario_id="data_noise_20pct",
                category="data_perturbation",
                severity=0.65,
                data_perturbation=0.20,
            ),
        ]

    # ========================================================
    # Result construction
    # ========================================================

    def create_scenario_result(
        self,
        scenario: RobustnessScenario,
        *,
        baseline_score: float,
        stressed_score: float,
    ) -> RobustnessScenarioResult:
        if not isinstance(
            scenario,
            RobustnessScenario,
        ):
            raise ValidationConfigurationError(
                "scenario must be a RobustnessScenario."
            )

        baseline = _validate_numeric(
            baseline_score,
            "baseline_score",
        )

        stressed = _validate_numeric(
            stressed_score,
            "stressed_score",
        )

        degradation = (
            baseline
            - stressed
        )

        if math.isclose(
            baseline,
            0.0,
            abs_tol=1e-15,
        ):
            degradation_ratio = (
                0.0
                if math.isclose(
                    stressed,
                    baseline,
                    abs_tol=1e-15,
                )
                else (
                    1.0
                    if stressed < baseline
                    else -1.0
                )
            )
        else:
            degradation_ratio = (
                degradation
                / abs(baseline)
            )

        positive_after_stress = (
            stressed > 0.0
        )

        stable = (
            positive_after_stress
            and degradation_ratio
            <= self.maximum_degradation_ratio
        )

        return RobustnessScenarioResult(
            scenario=scenario,
            baseline_score=baseline,
            stressed_score=stressed,
            degradation=degradation,
            degradation_ratio=(
                degradation_ratio
            ),
            positive_after_stress=(
                positive_after_stress
            ),
            stable=stable,
            metadata={
                "version": VERSION,
                "maximum_degradation_ratio": (
                    self.maximum_degradation_ratio
                ),
            },
        )

    # ========================================================
    # Aggregate
    # ========================================================

    def aggregate(
        self,
        results: Sequence[
            RobustnessScenarioResult
        ],
    ) -> RobustnessReport:
        if not results:
            raise RobustnessValidationError(
                "results cannot be empty."
            )

        validated: List[
            RobustnessScenarioResult
        ] = []

        for result in results:
            if not isinstance(
                result,
                RobustnessScenarioResult,
            ):
                raise ValidationConfigurationError(
                    "results must contain "
                    "RobustnessScenarioResult objects."
                )

            validated.append(result)

        baseline = (
            validated[0].baseline_score
        )

        for result in validated:
            if not math.isclose(
                result.baseline_score,
                baseline,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise RobustnessValidationError(
                    "All scenario results must use "
                    "the same baseline_score."
                )

        count = len(validated)

        average_stressed_score = (
            sum(
                result.stressed_score
                for result in validated
            )
            / count
        )

        average_degradation = (
            sum(
                result.degradation
                for result in validated
            )
            / count
        )

        average_degradation_ratio = (
            sum(
                result.degradation_ratio
                for result in validated
            )
            / count
        )

        positive_count = sum(
            1
            for result in validated
            if result.positive_after_stress
        )

        stable_count = sum(
            1
            for result in validated
            if result.stable
        )

        positive_survival_ratio = (
            positive_count
            / count
        )

        stable_scenario_ratio = (
            stable_count
            / count
        )

        worst_stressed_score = min(
            result.stressed_score
            for result in validated
        )

        worst_degradation_ratio = max(
            result.degradation_ratio
            for result in validated
        )

        passed = (
            average_stressed_score > 0.0
            and positive_survival_ratio
            >= self.minimum_positive_survival_ratio
            and stable_scenario_ratio
            >= self.minimum_stable_scenario_ratio
            and average_degradation_ratio
            <= self.maximum_degradation_ratio
        )

        return RobustnessReport(
            baseline_score=baseline,
            scenario_results=validated,
            average_stressed_score=(
                average_stressed_score
            ),
            average_degradation=(
                average_degradation
            ),
            average_degradation_ratio=(
                average_degradation_ratio
            ),
            positive_survival_ratio=(
                positive_survival_ratio
            ),
            stable_scenario_ratio=(
                stable_scenario_ratio
            ),
            worst_stressed_score=(
                worst_stressed_score
            ),
            worst_degradation_ratio=(
                worst_degradation_ratio
            ),
            passed=passed,
            metadata={
                "version": VERSION,
                "maximum_degradation_ratio": (
                    self.maximum_degradation_ratio
                ),
                "minimum_positive_survival_ratio": (
                    self.minimum_positive_survival_ratio
                ),
                "minimum_stable_scenario_ratio": (
                    self.minimum_stable_scenario_ratio
                ),
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Score
    # ========================================================

    def _calculate_score(
        self,
        report: RobustnessReport,
    ) -> float:
        positive_component = (
            report.positive_survival_ratio
        )

        stability_component = (
            report.stable_scenario_ratio
        )

        degradation_component = max(
            0.0,
            min(
                1.0,
                1.0
                - max(
                    0.0,
                    report.average_degradation_ratio,
                ),
            ),
        )

        worst_case_component = (
            1.0
            if report.worst_stressed_score > 0.0
            else 0.0
        )

        score = (
            0.30 * positive_component
            + 0.35 * stability_component
            + 0.25 * degradation_component
            + 0.10 * worst_case_component
        )

        return max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

    # ========================================================
    # ValidationResult adapter
    # ========================================================

    def to_validation_result(
        self,
        report: RobustnessReport,
    ) -> ValidationResult:
        status = (
            ValidationStatus.PASSED
            if report.passed
            else ValidationStatus.FAILED
        )

        message = (
            "Research alpha remained robust "
            "under stress scenarios."
            if report.passed
            else (
                "Research alpha showed material "
                "instability under stress scenarios."
            )
        )

        return ValidationResult(
            stage=ValidationStage.ROBUSTNESS,
            status=status,
            score=self._calculate_score(
                report
            ),
            message=message,
            diagnostics={
                "scenario_count": (
                    report.scenario_count
                ),
                "stable_count": (
                    report.stable_count
                ),
                "positive_count": (
                    report.positive_count
                ),
                "positive_survival_ratio": (
                    report.positive_survival_ratio
                ),
                "stable_scenario_ratio": (
                    report.stable_scenario_ratio
                ),
                "average_stressed_score": (
                    report.average_stressed_score
                ),
                "average_degradation": (
                    report.average_degradation
                ),
                "average_degradation_ratio": (
                    report.average_degradation_ratio
                ),
                "worst_stressed_score": (
                    report.worst_stressed_score
                ),
                "worst_degradation_ratio": (
                    report.worst_degradation_ratio
                ),
                "failed_scenarios": (
                    report.failed_scenarios
                ),
                "report_fingerprint": (
                    report.fingerprint
                ),
            },
            metadata={
                "version": VERSION,
                "research_only": True,
                "real_money_execution": False,
            },
        )

    # ========================================================
    # Convenience validation
    # ========================================================

    def validate_scores(
        self,
        *,
        baseline_score: float,
        stressed_scores: Sequence[float],
        scenarios: Optional[
            Sequence[RobustnessScenario]
        ] = None,
    ) -> RobustnessReport:
        baseline = _validate_numeric(
            baseline_score,
            "baseline_score",
        )

        scores = _validate_scores(
            stressed_scores
        )

        if scenarios is None:
            default_scenarios = (
                self.build_default_scenarios()
            )

            if len(scores) > len(
                default_scenarios
            ):
                generated: List[
                    RobustnessScenario
                ] = []

                for index in range(
                    len(scores)
                ):
                    generated.append(
                        RobustnessScenario(
                            scenario_id=(
                                f"stress_{index}"
                            ),
                            category="generic",
                            severity=min(
                                1.0,
                                (
                                    index + 1
                                )
                                / len(scores),
                            ),
                        )
                    )

                scenario_list = generated
            else:
                scenario_list = (
                    default_scenarios[
                        :len(scores)
                    ]
                )
        else:
            scenario_list = list(
                scenarios
            )

            if not scenario_list:
                raise RobustnessValidationError(
                    "scenarios cannot be empty."
                )

            for scenario in scenario_list:
                if not isinstance(
                    scenario,
                    RobustnessScenario,
                ):
                    raise ValidationConfigurationError(
                        "scenarios must contain "
                        "RobustnessScenario objects."
                    )

        if len(
            scenario_list
        ) != len(scores):
            raise RobustnessValidationError(
                "scenarios and stressed_scores "
                "must have equal length."
            )

        results = [
            self.create_scenario_result(
                scenario,
                baseline_score=baseline,
                stressed_score=score,
            )
            for scenario, score in zip(
                scenario_list,
                scores,
            )
        ]

        return self.aggregate(
            results
        )

    # ========================================================
    # Category analysis
    # ========================================================

    def category_summary(
        self,
        report: RobustnessReport,
    ) -> Dict[str, Dict[str, float]]:
        categories: Dict[
            str,
            List[RobustnessScenarioResult],
        ] = {}

        for result in report.scenario_results:
            categories.setdefault(
                result.scenario.category,
                [],
            ).append(result)

        summary: Dict[
            str,
            Dict[str, float],
        ] = {}

        for category, results in categories.items():
            count = len(results)

            summary[category] = {
                "scenario_count": float(
                    count
                ),
                "average_stressed_score": (
                    sum(
                        item.stressed_score
                        for item in results
                    )
                    / count
                ),
                "average_degradation_ratio": (
                    sum(
                        item.degradation_ratio
                        for item in results
                    )
                    / count
                ),
                "positive_survival_ratio": (
                    sum(
                        1
                        for item in results
                        if item.positive_after_stress
                    )
                    / count
                ),
                "stable_ratio": (
                    sum(
                        1
                        for item in results
                        if item.stable
                    )
                    / count
                ),
            }

        return summary

    # ========================================================
    # Scenario mapping API
    # ========================================================

    def validate_mapping(
        self,
        *,
        baseline_score: float,
        stressed_scores: Mapping[
            str,
            float,
        ],
    ) -> RobustnessReport:
        if not stressed_scores:
            raise RobustnessValidationError(
                "stressed_scores cannot be empty."
            )

        scenarios = [
            RobustnessScenario(
                scenario_id=name,
                category="custom",
                severity=min(
                    1.0,
                    (
                        index + 1
                    )
                    / len(stressed_scores),
                ),
            )
            for index, name
            in enumerate(
                stressed_scores.keys()
            )
        ]

        return self.validate_scores(
            baseline_score=baseline_score,
            stressed_scores=list(
                stressed_scores.values()
            ),
            scenarios=scenarios,
        )

    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": VERSION,
            "config_fingerprint": (
                self.config.fingerprint
            ),
            "maximum_degradation_ratio": (
                self.maximum_degradation_ratio
            ),
            "minimum_positive_survival_ratio": (
                self.minimum_positive_survival_ratio
            ),
            "minimum_stable_scenario_ratio": (
                self.minimum_stable_scenario_ratio
            ),
            "default_scenario_count": len(
                self.build_default_scenarios()
            ),
            "stress_dimensions": [
                "transaction_cost",
                "slippage",
                "execution_delay",
                "parameter_shift",
                "data_perturbation",
            ],
            "research_only": True,
            "real_money_execution": False,
        }