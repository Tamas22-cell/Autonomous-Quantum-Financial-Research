"""
Promotion / Demotion Engine for the v5.0.6 Agent Tournament System.

The engine converts historical tournament performance into controlled
champion/challenger governance decisions.

Core principles:

- A challenger must not replace a champion because of one lucky result.
- Promotion requires sufficient historical evidence.
- Performance is evaluated inside Asset x Regime x Horizon dimensions.
- Minimum score improvement can be required.
- Prediction quality, calibration and consistency can be used as safeguards.
- Latency and cost regression can be limited.
- Every decision contains an explicit reason.
- Promotion decisions are proposals until explicitly approved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .models import (
    DecisionType,
    PromotionDecision,
    generate_id,
)
from .performance_matrix import (
    AggregatedPerformance,
    PerformanceMatrix,
    PerformanceMatrixError,
)


class PromotionEngineError(Exception):
    """
    Raised when a promotion decision cannot be evaluated safely.
    """


@dataclass(frozen=True)
class PromotionPolicy:
    """
    Governance thresholds for challenger promotion.
    """

    minimum_tournaments: int = 3

    minimum_score_improvement: float = 0.02

    minimum_prediction_improvement: float = 0.0

    minimum_calibration_delta: float = -0.02

    minimum_consistency_delta: float = -0.02

    maximum_latency_regression_ms: float = 1000.0

    maximum_cost_regression_usd: float = 0.01

    minimum_challenger_win_rate: float = 0.60

    def __post_init__(self) -> None:
        if self.minimum_tournaments < 1:
            raise PromotionEngineError(
                "minimum_tournaments must be at least 1."
            )

        if self.minimum_score_improvement < 0:
            raise PromotionEngineError(
                "minimum_score_improvement cannot be negative."
            )

        if self.minimum_prediction_improvement < 0:
            raise PromotionEngineError(
                "minimum_prediction_improvement cannot be negative."
            )

        if self.maximum_latency_regression_ms < 0:
            raise PromotionEngineError(
                "maximum_latency_regression_ms cannot be negative."
            )

        if self.maximum_cost_regression_usd < 0:
            raise PromotionEngineError(
                "maximum_cost_regression_usd cannot be negative."
            )

        if not 0.0 <= self.minimum_challenger_win_rate <= 1.0:
            raise PromotionEngineError(
                "minimum_challenger_win_rate must be between 0 and 1."
            )


@dataclass
class PromotionEvaluation:
    """
    Complete evidence package for one promotion evaluation.
    """

    champion_id: str
    challenger_id: str

    asset: str
    regime: str
    horizon: str

    champion: AggregatedPerformance
    challenger: AggregatedPerformance

    score_delta: float
    prediction_delta: float
    calibration_delta: float
    consistency_delta: float
    latency_delta_ms: float
    cost_delta_usd: float

    sufficient_history: bool

    score_requirement_passed: bool
    prediction_requirement_passed: bool
    calibration_requirement_passed: bool
    consistency_requirement_passed: bool
    latency_requirement_passed: bool
    cost_requirement_passed: bool
    win_rate_requirement_passed: bool

    promotion_eligible: bool

    @property
    def failed_requirements(self) -> List[str]:
        """
        Return all failed governance checks.
        """

        failures: List[str] = []

        if not self.sufficient_history:
            failures.append(
                "insufficient_history"
            )

        if not self.score_requirement_passed:
            failures.append(
                "score_improvement"
            )

        if not self.prediction_requirement_passed:
            failures.append(
                "prediction_improvement"
            )

        if not self.calibration_requirement_passed:
            failures.append(
                "calibration"
            )

        if not self.consistency_requirement_passed:
            failures.append(
                "consistency"
            )

        if not self.latency_requirement_passed:
            failures.append(
                "latency"
            )

        if not self.cost_requirement_passed:
            failures.append(
                "cost"
            )

        if not self.win_rate_requirement_passed:
            failures.append(
                "win_rate"
            )

        return failures


class PromotionEngine:
    """
    Controlled Champion / Challenger promotion engine.

    The engine evaluates historical evidence but does not
    automatically mutate competitor roles.

    Governance flow:

        PerformanceMatrix
            ->
        PromotionEvaluation
            ->
        PromotionDecision
            ->
        Explicit Approval
            ->
        Future deployment / rollback layer
    """

    def __init__(
        self,
        performance_matrix: PerformanceMatrix,
        policy: Optional[PromotionPolicy] = None,
    ) -> None:

        if not isinstance(
            performance_matrix,
            PerformanceMatrix,
        ):
            raise PromotionEngineError(
                "performance_matrix must be a "
                "PerformanceMatrix instance."
            )

        self.performance_matrix = (
            performance_matrix
        )

        self.policy = (
            policy
            or PromotionPolicy()
        )

        self._decisions: Dict[
            str,
            PromotionDecision,
        ] = {}

        self._evaluations: Dict[
            str,
            PromotionEvaluation,
        ] = {}

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    def evaluate(
        self,
        champion_id: str,
        challenger_id: str,
        asset: str,
        regime: str,
        horizon: str,
    ) -> PromotionEvaluation:
        """
        Evaluate whether a challenger satisfies
        promotion requirements.
        """

        if not champion_id:
            raise PromotionEngineError(
                "champion_id cannot be empty."
            )

        if not challenger_id:
            raise PromotionEngineError(
                "challenger_id cannot be empty."
            )

        if champion_id == challenger_id:
            raise PromotionEngineError(
                "Champion and challenger must be "
                "different competitors."
            )

        if not asset:
            raise PromotionEngineError(
                "asset cannot be empty."
            )

        if not regime:
            raise PromotionEngineError(
                "regime cannot be empty."
            )

        if not horizon:
            raise PromotionEngineError(
                "horizon cannot be empty."
            )

        try:
            champion = (
                self.performance_matrix.aggregate(
                    competitor_id=champion_id,
                    asset=asset,
                    regime=regime,
                    horizon=horizon,
                )
            )

            challenger = (
                self.performance_matrix.aggregate(
                    competitor_id=challenger_id,
                    asset=asset,
                    regime=regime,
                    horizon=horizon,
                )
            )

        except PerformanceMatrixError as exc:
            raise PromotionEngineError(
                str(exc)
            ) from exc

        score_delta = (
            challenger.average_composite_score
            - champion.average_composite_score
        )

        prediction_delta = (
            challenger.average_prediction_score
            - champion.average_prediction_score
        )

        calibration_delta = (
            challenger.average_calibration_score
            - champion.average_calibration_score
        )

        consistency_delta = (
            challenger.average_consistency_score
            - champion.average_consistency_score
        )

        latency_delta_ms = (
            challenger.average_latency_ms
            - champion.average_latency_ms
        )

        cost_delta_usd = (
            challenger.average_cost_usd
            - champion.average_cost_usd
        )

        sufficient_history = (
            champion.tournament_count
            >= self.policy.minimum_tournaments
            and
            challenger.tournament_count
            >= self.policy.minimum_tournaments
        )

        score_requirement_passed = (
            score_delta
            >= self.policy.minimum_score_improvement
        )

        prediction_requirement_passed = (
            prediction_delta
            >= self.policy.minimum_prediction_improvement
        )

        calibration_requirement_passed = (
            calibration_delta
            >= self.policy.minimum_calibration_delta
        )

        consistency_requirement_passed = (
            consistency_delta
            >= self.policy.minimum_consistency_delta
        )

        latency_requirement_passed = (
            latency_delta_ms
            <= self.policy.maximum_latency_regression_ms
        )

        cost_requirement_passed = (
            cost_delta_usd
            <= self.policy.maximum_cost_regression_usd
        )

        win_rate_requirement_passed = (
            challenger.win_rate
            >= self.policy.minimum_challenger_win_rate
        )

        promotion_eligible = all(
            [
                sufficient_history,
                score_requirement_passed,
                prediction_requirement_passed,
                calibration_requirement_passed,
                consistency_requirement_passed,
                latency_requirement_passed,
                cost_requirement_passed,
                win_rate_requirement_passed,
            ]
        )

        return PromotionEvaluation(
            champion_id=champion_id,
            challenger_id=challenger_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
            champion=champion,
            challenger=challenger,
            score_delta=score_delta,
            prediction_delta=prediction_delta,
            calibration_delta=calibration_delta,
            consistency_delta=consistency_delta,
            latency_delta_ms=latency_delta_ms,
            cost_delta_usd=cost_delta_usd,
            sufficient_history=sufficient_history,
            score_requirement_passed=(
                score_requirement_passed
            ),
            prediction_requirement_passed=(
                prediction_requirement_passed
            ),
            calibration_requirement_passed=(
                calibration_requirement_passed
            ),
            consistency_requirement_passed=(
                consistency_requirement_passed
            ),
            latency_requirement_passed=(
                latency_requirement_passed
            ),
            cost_requirement_passed=(
                cost_requirement_passed
            ),
            win_rate_requirement_passed=(
                win_rate_requirement_passed
            ),
            promotion_eligible=(
                promotion_eligible
            ),
        )

    # --------------------------------------------------------------
    # Decision proposal
    # --------------------------------------------------------------

    def propose(
        self,
        tournament_id: str,
        champion_id: str,
        challenger_id: str,
        asset: str,
        regime: str,
        horizon: str,
    ) -> PromotionDecision:
        """
        Produce either:

        PROMOTE_CHALLENGER

        or

        KEEP_CHAMPION

        The resulting decision is not automatically approved.
        """

        if not tournament_id:
            raise PromotionEngineError(
                "tournament_id cannot be empty."
            )

        evaluation = self.evaluate(
            champion_id=champion_id,
            challenger_id=challenger_id,
            asset=asset,
            regime=regime,
            horizon=horizon,
        )

        if evaluation.promotion_eligible:

            decision_type = (
                DecisionType.PROMOTE_CHALLENGER
            )

            proposed_champion_id: Optional[str] = (
                challenger_id
            )

            reason = (
                self._promotion_reason(
                    evaluation
                )
            )

        else:

            decision_type = (
                DecisionType.KEEP_CHAMPION
            )

            proposed_champion_id = (
                champion_id
            )

            reason = (
                self._keep_reason(
                    evaluation
                )
            )

        decision = PromotionDecision(
            decision_id=generate_id(
                "promotion_decision"
            ),
            tournament_id=tournament_id,
            decision_type=decision_type,
            current_champion_id=champion_id,
            proposed_champion_id=(
                proposed_champion_id
            ),
            reason=reason,
            champion_score=(
                evaluation
                .champion
                .average_composite_score
            ),
            challenger_score=(
                evaluation
                .challenger
                .average_composite_score
            ),
            approved=False,
            metadata={
                "asset": asset,
                "regime": regime,
                "horizon": horizon,
                "score_delta": (
                    evaluation.score_delta
                ),
                "prediction_delta": (
                    evaluation.prediction_delta
                ),
                "calibration_delta": (
                    evaluation.calibration_delta
                ),
                "consistency_delta": (
                    evaluation.consistency_delta
                ),
                "latency_delta_ms": (
                    evaluation.latency_delta_ms
                ),
                "cost_delta_usd": (
                    evaluation.cost_delta_usd
                ),
                "challenger_win_rate": (
                    evaluation
                    .challenger
                    .win_rate
                ),
                "failed_requirements": (
                    evaluation.failed_requirements
                ),
            },
        )

        self._decisions[
            decision.decision_id
        ] = decision

        self._evaluations[
            decision.decision_id
        ] = evaluation

        return decision

    # --------------------------------------------------------------
    # Approval
    # --------------------------------------------------------------

    def approve(
        self,
        decision_id: str,
    ) -> PromotionDecision:
        """
        Explicitly approve an existing governance decision.
        """

        decision = self.get_decision(
            decision_id
        )

        decision.approved = True

        return decision

    # --------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------

    def get_decision(
        self,
        decision_id: str,
    ) -> PromotionDecision:
        """
        Retrieve a previously created decision.
        """

        if decision_id not in self._decisions:
            raise PromotionEngineError(
                "Unknown promotion decision: "
                f"{decision_id}"
            )

        return self._decisions[
            decision_id
        ]

    def get_evaluation(
        self,
        decision_id: str,
    ) -> PromotionEvaluation:
        """
        Retrieve the evaluation evidence
        associated with a decision.
        """

        if decision_id not in self._evaluations:
            raise PromotionEngineError(
                "No evaluation found for decision: "
                f"{decision_id}"
            )

        return self._evaluations[
            decision_id
        ]

    @property
    def decisions(
        self,
    ) -> List[PromotionDecision]:
        """
        Return all governance decisions.
        """

        return list(
            self._decisions.values()
        )

    # --------------------------------------------------------------
    # Explanation generation
    # --------------------------------------------------------------

    @staticmethod
    def _promotion_reason(
        evaluation: PromotionEvaluation,
    ) -> str:
        """
        Build human-readable promotion explanation.
        """

        return (
            "Challenger satisfied all promotion requirements. "
            f"Composite score improvement: "
            f"{evaluation.score_delta:.6f}. "
            f"Prediction improvement: "
            f"{evaluation.prediction_delta:.6f}. "
            f"Challenger win rate: "
            f"{evaluation.challenger.win_rate:.2%}."
        )

    @staticmethod
    def _keep_reason(
        evaluation: PromotionEvaluation,
    ) -> str:
        """
        Build human-readable champion-retention explanation.
        """

        failures = (
            ", ".join(
                evaluation.failed_requirements
            )
            or "unknown"
        )

        return (
            "Champion retained because challenger did not "
            "satisfy all promotion requirements. "
            f"Failed requirements: {failures}. "
            f"Composite score delta: "
            f"{evaluation.score_delta:.6f}."
        )


__all__ = [
    "PromotionEngine",
    "PromotionEngineError",
    "PromotionEvaluation",
    "PromotionPolicy",
]