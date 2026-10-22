"""
Tests for the v5.0.6 Promotion / Demotion Engine.

Tests:

- Challenger promotion with sufficient evidence
- Champion retention when challenger is weaker
- Insufficient-history protection
- Latency regression protection
- Cost regression protection
- Explicit decision approval
- Decision and evaluation retrieval
- Invalid competitor protection
"""

import pytest

from agent_tournament.models import (
    CompetitorResult,
    DecisionType,
    EvaluationMetrics,
)
from agent_tournament.performance_matrix import PerformanceMatrix
from agent_tournament.promotion_engine import (
    PromotionEngine,
    PromotionEngineError,
    PromotionPolicy,
)


def create_result(
    competitor_id: str,
    asset: str,
    regime: str,
    horizon: str,
    composite_score: float,
    prediction_score: float,
    calibration_score: float,
    consistency_score: float,
    latency_ms: float,
    cost_usd: float,
    rank: int,
) -> CompetitorResult:
    """
    Create one CompetitorResult for the performance matrix.
    """

    metrics = EvaluationMetrics(
        prediction_score=prediction_score,
        calibration_score=calibration_score,
        consistency_score=consistency_score,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        composite_score=composite_score,
        sample_count=1,
    )

    return CompetitorResult(
        competitor_id=competitor_id,
        asset=asset,
        regime=regime,
        horizon=horizon,
        metrics=metrics,
        rank=rank,
    )


def populate_matrix(
    matrix: PerformanceMatrix,
    tournament_count: int,
    champion_score: float,
    challenger_score: float,
    champion_prediction: float,
    challenger_prediction: float,
    champion_calibration: float = 0.80,
    challenger_calibration: float = 0.85,
    champion_consistency: float = 0.80,
    challenger_consistency: float = 0.85,
    champion_latency_ms: float = 1000.0,
    challenger_latency_ms: float = 900.0,
    champion_cost_usd: float = 0.010,
    challenger_cost_usd: float = 0.009,
) -> None:
    """
    Populate a PerformanceMatrix with repeated champion/challenger
    observations for the same Asset x Regime x Horizon dimension.
    """

    for index in range(tournament_count):
        tournament_id = f"tournament_{index + 1}"

        if challenger_score > champion_score:
            champion_rank = 2
            challenger_rank = 1
        else:
            champion_rank = 1
            challenger_rank = 2

        champion_result = create_result(
            competitor_id="champion",
            asset="BTC",
            regime="high_volatility",
            horizon="7D",
            composite_score=champion_score,
            prediction_score=champion_prediction,
            calibration_score=champion_calibration,
            consistency_score=champion_consistency,
            latency_ms=champion_latency_ms,
            cost_usd=champion_cost_usd,
            rank=champion_rank,
        )

        challenger_result = create_result(
            competitor_id="challenger",
            asset="BTC",
            regime="high_volatility",
            horizon="7D",
            composite_score=challenger_score,
            prediction_score=challenger_prediction,
            calibration_score=challenger_calibration,
            consistency_score=challenger_consistency,
            latency_ms=challenger_latency_ms,
            cost_usd=challenger_cost_usd,
            rank=challenger_rank,
        )

        matrix.add_competitor_result(
            tournament_id=tournament_id,
            result=champion_result,
        )

        matrix.add_competitor_result(
            tournament_id=tournament_id,
            result=challenger_result,
        )


def test_challenger_is_eligible_for_promotion() -> None:
    """
    A challenger with sufficient history and clearly superior
    performance must become promotion eligible.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.85,
        champion_prediction=0.72,
        challenger_prediction=0.88,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    evaluation = engine.evaluate(
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert evaluation.sufficient_history is True
    assert evaluation.score_requirement_passed is True
    assert evaluation.prediction_requirement_passed is True
    assert evaluation.calibration_requirement_passed is True
    assert evaluation.consistency_requirement_passed is True
    assert evaluation.latency_requirement_passed is True
    assert evaluation.cost_requirement_passed is True
    assert evaluation.win_rate_requirement_passed is True
    assert evaluation.promotion_eligible is True

    assert evaluation.score_delta == pytest.approx(
        0.15
    )

    assert evaluation.failed_requirements == []


def test_promotion_decision_is_created() -> None:
    """
    Eligible challenger must generate a PROMOTE_CHALLENGER proposal.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.85,
        champion_prediction=0.72,
        challenger_prediction=0.88,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    decision = engine.propose(
        tournament_id="promotion_test",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert (
        decision.decision_type
        == DecisionType.PROMOTE_CHALLENGER
    )

    assert (
        decision.current_champion_id
        == "champion"
    )

    assert (
        decision.proposed_champion_id
        == "challenger"
    )

    assert decision.approved is False

    assert decision.champion_score == pytest.approx(
        0.70
    )

    assert decision.challenger_score == pytest.approx(
        0.85
    )

    assert (
        decision.metadata["failed_requirements"]
        == []
    )


def test_weaker_challenger_keeps_champion() -> None:
    """
    Challenger with worse performance must not replace champion.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.85,
        challenger_score=0.70,
        champion_prediction=0.88,
        challenger_prediction=0.72,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    decision = engine.propose(
        tournament_id="keep_champion_test",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert (
        decision.decision_type
        == DecisionType.KEEP_CHAMPION
    )

    assert (
        decision.proposed_champion_id
        == "champion"
    )

    assert decision.approved is False

    assert (
        "score_improvement"
        in decision.metadata["failed_requirements"]
    )


def test_insufficient_history_blocks_promotion() -> None:
    """
    Challenger must not be promoted from too little evidence.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=1,
        champion_score=0.70,
        challenger_score=0.90,
        champion_prediction=0.72,
        challenger_prediction=0.92,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    evaluation = engine.evaluate(
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert evaluation.sufficient_history is False
    assert evaluation.promotion_eligible is False

    assert (
        "insufficient_history"
        in evaluation.failed_requirements
    )


def test_latency_regression_blocks_promotion() -> None:
    """
    A more accurate challenger may still be rejected if latency
    regression exceeds the governance policy.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.90,
        champion_prediction=0.72,
        challenger_prediction=0.92,
        champion_latency_ms=1000.0,
        challenger_latency_ms=5000.0,
    )

    policy = PromotionPolicy(
        maximum_latency_regression_ms=1000.0
    )

    engine = PromotionEngine(
        performance_matrix=matrix,
        policy=policy,
    )

    evaluation = engine.evaluate(
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert evaluation.latency_requirement_passed is False
    assert evaluation.promotion_eligible is False

    assert (
        "latency"
        in evaluation.failed_requirements
    )


def test_cost_regression_blocks_promotion() -> None:
    """
    Challenger must not be promoted if its cost increase
    exceeds the configured governance limit.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.90,
        champion_prediction=0.72,
        challenger_prediction=0.92,
        champion_cost_usd=0.010,
        challenger_cost_usd=0.100,
    )

    policy = PromotionPolicy(
        maximum_cost_regression_usd=0.01
    )

    engine = PromotionEngine(
        performance_matrix=matrix,
        policy=policy,
    )

    evaluation = engine.evaluate(
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert evaluation.cost_requirement_passed is False
    assert evaluation.promotion_eligible is False

    assert (
        "cost"
        in evaluation.failed_requirements
    )


def test_decision_can_be_explicitly_approved() -> None:
    """
    Promotion proposal must require explicit approval.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.90,
        champion_prediction=0.72,
        challenger_prediction=0.92,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    decision = engine.propose(
        tournament_id="approval_test",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert (
        decision.decision_type
        == DecisionType.PROMOTE_CHALLENGER
    )

    assert decision.approved is False

    approved_decision = engine.approve(
        decision.decision_id
    )

    assert approved_decision.approved is True

    stored_decision = engine.get_decision(
        decision.decision_id
    )

    assert stored_decision.approved is True


def test_decision_evaluation_can_be_retrieved() -> None:
    """
    Evidence used for a governance decision must remain retrievable.
    """

    matrix = PerformanceMatrix()

    populate_matrix(
        matrix=matrix,
        tournament_count=5,
        champion_score=0.70,
        challenger_score=0.90,
        champion_prediction=0.72,
        challenger_prediction=0.92,
    )

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    decision = engine.propose(
        tournament_id="evidence_test",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert (
        decision.decision_type
        == DecisionType.PROMOTE_CHALLENGER
    )

    evaluation = engine.get_evaluation(
        decision.decision_id
    )

    assert evaluation.champion_id == "champion"
    assert evaluation.challenger_id == "challenger"
    assert evaluation.promotion_eligible is True


def test_same_competitor_cannot_compete_against_itself() -> None:
    """
    Champion and challenger IDs must be different.
    """

    matrix = PerformanceMatrix()

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    with pytest.raises(
        PromotionEngineError
    ):
        engine.evaluate(
            champion_id="same_agent",
            challenger_id="same_agent",
            asset="BTC",
            regime="high_volatility",
            horizon="7D",
        )


def test_unknown_decision_is_rejected() -> None:
    """
    Unknown decision IDs must raise an explicit error.
    """

    matrix = PerformanceMatrix()

    engine = PromotionEngine(
        performance_matrix=matrix
    )

    with pytest.raises(
        PromotionEngineError
    ):
        engine.get_decision(
            "unknown_decision"
        )


def test_invalid_policy_is_rejected() -> None:
    """
    Invalid governance thresholds must fail immediately.
    """

    with pytest.raises(
        PromotionEngineError
    ):
        PromotionPolicy(
            minimum_tournaments=0
        )