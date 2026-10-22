"""
v5.0.9 - Autonomous Research Loop
Strategy Validation Engine.

Evaluates StrategyCandidate objects using externally supplied,
measurable validation metrics.

The engine is deliberately separated from the backtesting and causal
research implementations. Existing platform modules can later provide:

- out-of-sample performance
- robustness measurements
- causal evidence
- Sharpe ratio
- Sortino ratio
- maximum drawdown

The validator converts those measurements into one of:

- REJECT
- REVIEW
- PROMOTE

PROMOTE does NOT authorize real-money execution.
A promoted candidate must still pass the independent Risk Governor
and enter shadow allocation before any higher-trust state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from autonomous_research_loop.models import (
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)


@dataclass(frozen=True)
class ValidationMetrics:
    """
    Normalized input contract for candidate validation.

    Scores use the range 0.0 to 1.0.

    Sharpe and Sortino are raw ratios.
    maximum_drawdown is represented as a positive decimal:
    0.10 means a 10% maximum drawdown.
    """

    out_of_sample_score: float
    robustness_score: float
    causal_score: float

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    maximum_drawdown: float = 0.0

    sample_size: int = 0

    def __post_init__(self) -> None:
        bounded_scores = (
            self.out_of_sample_score,
            self.robustness_score,
            self.causal_score,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in bounded_scores
        ):
            raise ValueError(
                "Validation scores must be between 0.0 and 1.0."
            )

        if self.maximum_drawdown < 0.0:
            raise ValueError(
                "maximum_drawdown cannot be negative."
            )

        if self.sample_size < 0:
            raise ValueError(
                "sample_size cannot be negative."
            )


@dataclass(frozen=True)
class ValidationEngineConfig:
    """
    Conservative research validation thresholds.
    """

    out_of_sample_weight: float = 0.40
    robustness_weight: float = 0.35
    causal_weight: float = 0.25

    promote_score_threshold: float = 0.70
    review_score_threshold: float = 0.50

    minimum_out_of_sample_score: float = 0.55
    minimum_robustness_score: float = 0.50
    minimum_causal_score: float = 0.40

    minimum_sharpe_ratio: float = 0.75
    minimum_sortino_ratio: float = 0.90

    maximum_allowed_drawdown: float = 0.20

    minimum_sample_size: int = 30

    def __post_init__(self) -> None:
        weights = (
            self.out_of_sample_weight,
            self.robustness_weight,
            self.causal_weight,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in weights
        ):
            raise ValueError(
                "Validation weights must be between 0.0 and 1.0."
            )

        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError(
                "Validation weights must sum to 1.0."
            )

        bounded_thresholds = (
            self.promote_score_threshold,
            self.review_score_threshold,
            self.minimum_out_of_sample_score,
            self.minimum_robustness_score,
            self.minimum_causal_score,
        )

        if any(
            value < 0.0 or value > 1.0
            for value in bounded_thresholds
        ):
            raise ValueError(
                "Validation thresholds must be between 0.0 and 1.0."
            )

        if (
            self.review_score_threshold
            >= self.promote_score_threshold
        ):
            raise ValueError(
                "review_score_threshold must be lower than "
                "promote_score_threshold."
            )

        if self.maximum_allowed_drawdown < 0.0:
            raise ValueError(
                "maximum_allowed_drawdown cannot be negative."
            )

        if self.minimum_sample_size < 1:
            raise ValueError(
                "minimum_sample_size must be at least 1."
            )


class StrategyValidationEngine:
    """
    Convert measurable validation evidence into a controlled
    ValidationResult.

    Important:
    this class does not manufacture backtest metrics.

    Metrics must come from an external validation process.
    """

    def __init__(
        self,
        config: Optional[ValidationEngineConfig] = None,
    ) -> None:
        self.config = config or ValidationEngineConfig()

    @staticmethod
    def _clamp(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                numeric,
            ),
        )

    def calculate_score(
        self,
        metrics: ValidationMetrics,
    ) -> float:
        """
        Weighted validation score.

        Raw Sharpe, Sortino and drawdown remain independent gates
        rather than being hidden inside the composite score.
        """

        score = (
            metrics.out_of_sample_score
            * self.config.out_of_sample_weight
            + metrics.robustness_score
            * self.config.robustness_weight
            + metrics.causal_score
            * self.config.causal_weight
        )

        return round(
            self._clamp(score),
            6,
        )

    def collect_gate_failures(
        self,
        metrics: ValidationMetrics,
    ) -> List[str]:
        """
        Return all hard validation failures.
        """

        failures: List[str] = []

        if (
            metrics.sample_size
            < self.config.minimum_sample_size
        ):
            failures.append(
                "Insufficient validation sample size: "
                f"{metrics.sample_size} < "
                f"{self.config.minimum_sample_size}."
            )

        if (
            metrics.out_of_sample_score
            < self.config.minimum_out_of_sample_score
        ):
            failures.append(
                "Out-of-sample score below minimum."
            )

        if (
            metrics.robustness_score
            < self.config.minimum_robustness_score
        ):
            failures.append(
                "Robustness score below minimum."
            )

        if (
            metrics.causal_score
            < self.config.minimum_causal_score
        ):
            failures.append(
                "Causal score below minimum."
            )

        if (
            metrics.sharpe_ratio
            < self.config.minimum_sharpe_ratio
        ):
            failures.append(
                "Sharpe ratio below minimum."
            )

        if (
            metrics.sortino_ratio
            < self.config.minimum_sortino_ratio
        ):
            failures.append(
                "Sortino ratio below minimum."
            )

        if (
            metrics.maximum_drawdown
            > self.config.maximum_allowed_drawdown
        ):
            failures.append(
                "Maximum drawdown exceeds allowed limit."
            )

        return failures

    def determine_decision(
        self,
        score: float,
        failures: List[str],
    ) -> ValidationDecision:
        """
        Decision policy.

        Hard gate failures always prevent PROMOTE.

        Strong composite score + zero failures:
            PROMOTE

        Intermediate evidence:
            REVIEW

        Weak evidence:
            REJECT
        """

        if failures:
            if (
                score
                >= self.config.review_score_threshold
            ):
                return ValidationDecision.REVIEW

            return ValidationDecision.REJECT

        if (
            score
            >= self.config.promote_score_threshold
        ):
            return ValidationDecision.PROMOTE

        if (
            score
            >= self.config.review_score_threshold
        ):
            return ValidationDecision.REVIEW

        return ValidationDecision.REJECT

    def build_reasons(
        self,
        score: float,
        decision: ValidationDecision,
        failures: List[str],
    ) -> List[str]:
        """
        Produce an auditable explanation of the validation result.
        """

        reasons: List[str] = [
            f"Composite validation score: {score:.3f}.",
            f"Validation decision: {decision.value}.",
        ]

        if failures:
            reasons.extend(failures)
        else:
            reasons.append(
                "All configured hard validation gates passed."
            )

        if decision == ValidationDecision.PROMOTE:
            reasons.append(
                "Candidate may proceed to the independent "
                "Risk Governor."
            )

        elif decision == ValidationDecision.REVIEW:
            reasons.append(
                "Candidate requires additional validation before "
                "promotion."
            )

        else:
            reasons.append(
                "Candidate is rejected from the current "
                "research cycle."
            )

        return reasons

    def validate(
        self,
        candidate: StrategyCandidate,
        metrics: ValidationMetrics,
    ) -> ValidationResult:
        """
        Validate one StrategyCandidate.
        """

        if not isinstance(
            candidate,
            StrategyCandidate,
        ):
            raise TypeError(
                "StrategyValidationEngine requires "
                "a StrategyCandidate."
            )

        if not isinstance(
            metrics,
            ValidationMetrics,
        ):
            raise TypeError(
                "metrics must be a ValidationMetrics instance."
            )

        score = self.calculate_score(
            metrics
        )

        failures = self.collect_gate_failures(
            metrics
        )

        decision = self.determine_decision(
            score,
            failures,
        )

        reasons = self.build_reasons(
            score,
            decision,
            failures,
        )

        return ValidationResult(
            strategy_id=candidate.strategy_id,
            decision=decision,
            score=score,
            out_of_sample_score=(
                metrics.out_of_sample_score
            ),
            robustness_score=(
                metrics.robustness_score
            ),
            causal_score=(
                metrics.causal_score
            ),
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.maximum_drawdown,
            reasons=reasons,
            metadata={
                "validator":
                    "v5.0.9-strategy-validation-engine",
                "sample_size":
                    metrics.sample_size,
                "hard_gate_failures":
                    list(failures),
                "requires_risk_governor":
                    decision
                    == ValidationDecision.PROMOTE,
                "real_money_authorized":
                    False,
                "shadow_only_after_risk_gate":
                    True,
            },
        )

    def validate_from_mapping(
        self,
        candidate: StrategyCandidate,
        metrics: Mapping[str, Any],
    ) -> ValidationResult:
        """
        Adapter-friendly interface for existing backtesting systems.

        This makes later integration easier when the institutional
        backtester or causal engine returns dictionaries.
        """

        if not isinstance(
            metrics,
            Mapping,
        ):
            raise TypeError(
                "metrics must be a mapping."
            )

        normalized = ValidationMetrics(
            out_of_sample_score=float(
                metrics.get(
                    "out_of_sample_score",
                    0.0,
                )
            ),
            robustness_score=float(
                metrics.get(
                    "robustness_score",
                    0.0,
                )
            ),
            causal_score=float(
                metrics.get(
                    "causal_score",
                    0.0,
                )
            ),
            sharpe_ratio=float(
                metrics.get(
                    "sharpe_ratio",
                    0.0,
                )
            ),
            sortino_ratio=float(
                metrics.get(
                    "sortino_ratio",
                    0.0,
                )
            ),
            maximum_drawdown=float(
                metrics.get(
                    "maximum_drawdown",
                    metrics.get(
                        "max_drawdown",
                        0.0,
                    ),
                )
            ),
            sample_size=int(
                metrics.get(
                    "sample_size",
                    0,
                )
            ),
        )

        return self.validate(
            candidate,
            normalized,
        )

    def __call__(
        self,
        candidate: StrategyCandidate,
        metrics: ValidationMetrics,
    ) -> ValidationResult:
        """
        Callable validation interface.
        """

        return self.validate(
            candidate,
            metrics,
        )