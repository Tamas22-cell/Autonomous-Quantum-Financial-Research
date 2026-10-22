"""
Tests for the v5.0.6 Agent Tournament Performance Matrix.

Tests:

- TournamentResult ingestion
- Asset x Regime x Horizon dimensions
- Historical aggregation
- Dimension leaderboard
- Champion vs Challenger comparison
- Best / worst dimension detection
- Duplicate tournament protection
"""

import pytest

from agent_tournament.models import (
    CompetitorResult,
    EvaluationMetrics,
    TournamentResult,
    TournamentStatus,
)
from agent_tournament.performance_matrix import (
    PerformanceMatrix,
    PerformanceMatrixError,
)


def create_metrics(
    composite_score: float,
    prediction_score: float,
    calibration_score: float,
    consistency_score: float,
    latency_ms: float,
    cost_usd: float,
    sample_count: int = 1,
) -> EvaluationMetrics:
    """
    Helper for creating EvaluationMetrics.
    """

    return EvaluationMetrics(
        prediction_score=prediction_score,
        calibration_score=calibration_score,
        consistency_score=consistency_score,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        composite_score=composite_score,
        sample_count=sample_count,
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
    Helper for creating CompetitorResult.
    """

    return CompetitorResult(
        competitor_id=competitor_id,
        asset=asset,
        regime=regime,
        horizon=horizon,
        metrics=create_metrics(
            composite_score=composite_score,
            prediction_score=prediction_score,
            calibration_score=calibration_score,
            consistency_score=consistency_score,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
        ),
        rank=rank,
    )


def create_tournament(
    tournament_id: str,
    champion_id: str,
    challenger_id: str,
    asset: str,
    regime: str,
    horizon: str,
    champion_score: float,
    challenger_score: float,
) -> TournamentResult:
    """
    Create a completed two-agent tournament.
    """

    if challenger_score > champion_score:
        champion_rank = 2
        challenger_rank = 1
        winner_id = challenger_id
    else:
        champion_rank = 1
        challenger_rank = 2
        winner_id = champion_id

    champion_result = create_result(
        competitor_id=champion_id,
        asset=asset,
        regime=regime,
        horizon=horizon,
        composite_score=champion_score,
        prediction_score=champion_score,
        calibration_score=champion_score,
        consistency_score=champion_score,
        latency_ms=1000.0,
        cost_usd=0.010,
        rank=champion_rank,
    )

    challenger_result = create_result(
        competitor_id=challenger_id,
        asset=asset,
        regime=regime,
        horizon=horizon,
        composite_score=challenger_score,
        prediction_score=challenger_score,
        calibration_score=challenger_score,
        consistency_score=challenger_score,
        latency_ms=800.0,
        cost_usd=0.008,
        rank=challenger_rank,
    )

    return TournamentResult(
        tournament_id=tournament_id,
        champion_id=champion_id,
        challenger_ids=[challenger_id],
        results=[
            champion_result,
            challenger_result,
        ],
        winner_id=winner_id,
        status=TournamentStatus.COMPLETED,
    )


def test_add_tournament_result() -> None:
    """
    Completed tournament results must be stored
    in the performance matrix.
    """

    matrix = PerformanceMatrix()

    tournament = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.70,
        challenger_score=0.85,
    )

    matrix.add_tournament_result(tournament)

    assert len(matrix) == 2
    assert bool(matrix) is True

    records = matrix.records

    assert len(records) == 2

    competitor_ids = {
        record.competitor_id
        for record in records
    }

    assert competitor_ids == {
        "champion",
        "challenger",
    }


def test_dimension_detection() -> None:
    """
    Matrix must correctly identify unique
    Asset x Regime x Horizon dimensions.
    """

    matrix = PerformanceMatrix()

    tournament_1 = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.70,
        challenger_score=0.85,
    )

    tournament_2 = create_tournament(
        tournament_id="tournament_002",
        champion_id="champion",
        challenger_id="challenger",
        asset="ETH",
        regime="risk_on",
        horizon="30D",
        champion_score=0.80,
        challenger_score=0.75,
    )

    matrix.add_tournament_result(tournament_1)
    matrix.add_tournament_result(tournament_2)

    dimensions = matrix.dimensions()

    assert len(dimensions) == 2

    dimension_keys = {
        dimension.key
        for dimension in dimensions
    }

    assert (
        "BTC",
        "high_volatility",
        "7D",
    ) in dimension_keys

    assert (
        "ETH",
        "risk_on",
        "30D",
    ) in dimension_keys


def test_historical_aggregation() -> None:
    """
    Multiple tournaments in the same dimension
    must be aggregated correctly.
    """

    matrix = PerformanceMatrix()

    tournament_1 = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.60,
        challenger_score=0.80,
    )

    tournament_2 = create_tournament(
        tournament_id="tournament_002",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.70,
        challenger_score=0.90,
    )

    matrix.add_tournament_result(tournament_1)
    matrix.add_tournament_result(tournament_2)

    challenger = matrix.aggregate(
        competitor_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert challenger.tournament_count == 2

    assert challenger.average_composite_score == pytest.approx(
        0.85
    )

    assert challenger.best_composite_score == pytest.approx(
        0.90
    )

    assert challenger.worst_composite_score == pytest.approx(
        0.80
    )

    assert challenger.win_count == 2

    assert challenger.win_rate == pytest.approx(
        1.0
    )


def test_dimension_leaderboard() -> None:
    """
    Challenger with the higher historical composite score
    must rank above the champion.
    """

    matrix = PerformanceMatrix()

    tournament_1 = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.65,
        challenger_score=0.85,
    )

    tournament_2 = create_tournament(
        tournament_id="tournament_002",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.70,
        challenger_score=0.90,
    )

    matrix.add_tournament_result(tournament_1)
    matrix.add_tournament_result(tournament_2)

    leaderboard = matrix.leaderboard(
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert len(leaderboard) == 2

    assert leaderboard[0].rank == 1
    assert leaderboard[1].rank == 2

    assert (
        leaderboard[0].competitor_id
        == "challenger"
    )

    assert (
        leaderboard[1].competitor_id
        == "champion"
    )


def test_champion_challenger_comparison() -> None:
    """
    Matrix must calculate the performance delta
    between champion and challenger.
    """

    matrix = PerformanceMatrix()

    tournament = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.70,
        challenger_score=0.90,
    )

    matrix.add_tournament_result(tournament)

    comparison = matrix.compare(
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
    )

    assert comparison["champion_id"] == "champion"

    assert (
        comparison["challenger_id"]
        == "challenger"
    )

    assert comparison[
        "champion_score"
    ] == pytest.approx(0.70)

    assert comparison[
        "challenger_score"
    ] == pytest.approx(0.90)

    assert comparison[
        "score_delta"
    ] == pytest.approx(0.20)

    assert (
        comparison["preferred_competitor_id"]
        == "challenger"
    )


def test_best_and_worst_dimension() -> None:
    """
    Matrix must identify where an agent performs
    strongest and weakest.
    """

    matrix = PerformanceMatrix()

    btc_tournament = create_tournament(
        tournament_id="tournament_btc",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.60,
        challenger_score=0.90,
    )

    eth_tournament = create_tournament(
        tournament_id="tournament_eth",
        champion_id="champion",
        challenger_id="challenger",
        asset="ETH",
        regime="risk_off",
        horizon="30D",
        champion_score=0.80,
        challenger_score=0.55,
    )

    matrix.add_tournament_result(btc_tournament)
    matrix.add_tournament_result(eth_tournament)

    best = matrix.best_dimension(
        competitor_id="challenger"
    )

    worst = matrix.worst_dimension(
        competitor_id="challenger"
    )

    assert best.asset == "BTC"
    assert best.regime == "high_volatility"
    assert best.horizon == "7D"

    assert worst.asset == "ETH"
    assert worst.regime == "risk_off"
    assert worst.horizon == "30D"


def test_global_average_score() -> None:
    """
    Global average must combine performance
    across all recorded dimensions.
    """

    matrix = PerformanceMatrix()

    tournament_1 = create_tournament(
        tournament_id="tournament_001",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        champion_score=0.60,
        challenger_score=0.80,
    )

    tournament_2 = create_tournament(
        tournament_id="tournament_002",
        champion_id="champion",
        challenger_id="challenger",
        asset="ETH",
        regime="risk_on",
        horizon="30D",
        champion_score=0.70,
        challenger_score=0.90,
    )

    matrix.add_tournament_result(tournament_1)
    matrix.add_tournament_result(tournament_2)

    challenger_global_score = (
        matrix.global_average_score(
            competitor_id="challenger"
        )
    )

    assert challenger_global_score == pytest.approx(
        0.85
    )


def test_duplicate_tournament_is_rejected() -> None:
    """
    The same TournamentResult must not be ingested twice.
    """

    matrix = PerformanceMatrix()

    tournament = create_tournament(
        tournament_id="duplicate_tournament",
        champion_id="champion",
        challenger_id="challenger",
        asset="BTC",
        regime="normal",
        horizon="1D",
        champion_score=0.70,
        challenger_score=0.80,
    )

    matrix.add_tournament_result(tournament)

    with pytest.raises(
        PerformanceMatrixError
    ):
        matrix.add_tournament_result(
            tournament
        )


def test_missing_dimension_returns_empty_leaderboard() -> None:
    """
    Unknown matrix dimensions should return
    an empty leaderboard instead of crashing.
    """

    matrix = PerformanceMatrix()

    leaderboard = matrix.leaderboard(
        asset="SOL",
        regime="unknown",
        horizon="90D",
    )

    assert leaderboard == []