"""
Integration tests for the v5.0.6 Agent Tournament System.

Tests the complete pipeline:

models
    -> blind evaluator
    -> scoring engine
    -> tournament engine
"""

from datetime import datetime, timezone

from agent_tournament.blind_evaluator import BlindEvaluator
from agent_tournament.models import (
    AgentCompetitor,
    CompetitorRole,
    MarketSnapshot,
    TournamentStatus,
    TournamentTask,
)
from agent_tournament.scoring_engine import (
    ScoredObservation,
    ScoringEngine,
)
from agent_tournament.tournament_engine import TournamentEngine


def test_complete_champion_challenger_tournament() -> None:
    """
    Champion and Challenger receive the same market snapshot,
    submit predictions anonymously, are scored blindly,
    and receive a final ranking.
    """

    champion = AgentCompetitor(
        competitor_id="agent_champion",
        name="Macro Agent",
        version="1.0",
        role=CompetitorRole.CHAMPION,
        model_name="champion-model",
        prompt_version="prompt-v1",
        policy_version="policy-v1",
    )

    challenger = AgentCompetitor(
        competitor_id="agent_challenger",
        name="Macro Agent Challenger",
        version="2.0",
        role=CompetitorRole.CHALLENGER,
        model_name="challenger-model",
        prompt_version="prompt-v2",
        policy_version="policy-v2",
    )

    snapshot = MarketSnapshot(
        snapshot_id="snapshot_btc_001",
        timestamp=datetime.now(timezone.utc),
        asset="BTC",
        regime="high_volatility",
        horizon="7D",
        features={
            "price": 100_000.0,
            "volatility": 0.65,
            "momentum": 0.20,
        },
        source="integration_test",
    )

    task = TournamentTask(
        task_id="task_btc_001",
        snapshot_id=snapshot.snapshot_id,
        asset=snapshot.asset,
        regime=snapshot.regime,
        horizon=snapshot.horizon,
        instruction=(
            "Predict BTC directional outcome over the next 7 days."
        ),
    )

    blind_evaluator = BlindEvaluator()
    scoring_engine = ScoringEngine()

    engine = TournamentEngine(
        blind_evaluator=blind_evaluator,
        scoring_engine=scoring_engine,
    )

    engine.register_competitors(
        [
            champion,
            challenger,
        ]
    )

    engine.prepare_snapshot(snapshot)

    blind_competitors = blind_evaluator.register_competitors(
        [
            champion,
            challenger,
        ]
    )

    assert len(blind_competitors) == 2

    champion_blind_id = blind_competitors[0].blind_id
    challenger_blind_id = blind_competitors[1].blind_id

    assert champion_blind_id != challenger_blind_id

    champion_submission = blind_evaluator.submit(
        blind_id=champion_blind_id,
        task=task,
        snapshot=snapshot,
        prediction=0.70,
        confidence=0.80,
        latency_ms=1200.0,
        cost_usd=0.010,
    )

    challenger_submission = blind_evaluator.submit(
        blind_id=challenger_blind_id,
        task=task,
        snapshot=snapshot,
        prediction=0.85,
        confidence=0.90,
        latency_ms=900.0,
        cost_usd=0.008,
    )

    actual_outcome = 0.90

    observations = [
        ScoredObservation(
            submission=champion_submission,
            predicted_outcome=0.70,
            actual_outcome=actual_outcome,
        ),
        ScoredObservation(
            submission=challenger_submission,
            predicted_outcome=0.85,
            actual_outcome=actual_outcome,
        ),
    ]

    result = engine.run_tournament(
        task=task,
        snapshot=snapshot,
        observations=observations,
    )

    assert result.status == TournamentStatus.COMPLETED

    assert result.champion_id == champion.competitor_id

    assert challenger.competitor_id in result.challenger_ids

    assert result.winner_id == challenger.competitor_id

    assert len(result.results) == 2

    leaderboard = engine.leaderboard(result)

    assert leaderboard[0].rank == 1
    assert leaderboard[1].rank == 2

    assert (
        leaderboard[0].metrics.composite_score
        >= leaderboard[1].metrics.composite_score
    )

    assert (
        leaderboard[0].competitor_id
        == challenger.competitor_id
    )


def test_blind_identity_is_stable() -> None:
    """
    Re-registering the same competitor must preserve
    its anonymous tournament identity.
    """

    competitor = AgentCompetitor(
        competitor_id="stable_agent",
        name="Stable Agent",
        version="1.0",
        role=CompetitorRole.CHAMPION,
    )

    evaluator = BlindEvaluator()

    first = evaluator.register_competitors(
        [competitor]
    )[0]

    second = evaluator.register_competitors(
        [competitor]
    )[0]

    assert first.blind_id == second.blind_id


def test_snapshot_integrity_detection() -> None:
    """
    Registered market snapshots must remain immutable
    during blind tournament evaluation.
    """

    evaluator = BlindEvaluator()

    timestamp = datetime.now(timezone.utc)

    original = MarketSnapshot(
        snapshot_id="snapshot_integrity_test",
        timestamp=timestamp,
        asset="BTC",
        regime="normal",
        horizon="1D",
        features={
            "price": 100_000.0,
        },
        source="integration_test",
    )

    evaluator.register_snapshot(original)

    assert evaluator.verify_snapshot(original) is True

    modified = MarketSnapshot(
        snapshot_id="snapshot_integrity_test",
        timestamp=timestamp,
        asset="BTC",
        regime="normal",
        horizon="1D",
        features={
            "price": 110_000.0,
        },
        source="integration_test",
    )

    try:
        evaluator.verify_snapshot(modified)

    except Exception:
        pass

    else:
        raise AssertionError(
            "Modified snapshot should have failed "
            "integrity validation."
        )