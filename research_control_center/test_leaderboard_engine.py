import pytest

from research_control_center.leaderboard_engine import (
    LeaderboardConfig,
    LeaderboardEngine,
)


def make_engine() -> LeaderboardEngine:
    return LeaderboardEngine()


def add_agent(
    engine: LeaderboardEngine,
    name: str = "MacroAgent",
    asset: str = "BTC",
    regime: str = "RISK_ON",
    horizon: str = "1D",
    prediction_quality: float = 0.80,
    calibration_score: float = 0.75,
    consistency_score: float = 0.78,
    latency_ms: float = 200.0,
    cost: float = 0.10,
    observations: int = 100,
):
    return engine.add_entry(
        participant_name=name,
        participant_type="AGENT",
        asset=asset,
        regime=regime,
        horizon=horizon,
        prediction_quality=prediction_quality,
        calibration_score=calibration_score,
        consistency_score=consistency_score,
        latency_ms=latency_ms,
        cost=cost,
        observations=observations,
    )


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = LeaderboardConfig()

    assert config.prediction_quality_weight == 0.45
    assert config.calibration_weight == 0.25
    assert config.consistency_weight == 0.20
    assert config.efficiency_weight == 0.10


def test_custom_config():
    config = LeaderboardConfig(
        prediction_quality_weight=0.40,
        calibration_weight=0.30,
        consistency_weight=0.20,
        efficiency_weight=0.10,
    )

    assert config.prediction_quality_weight == 0.40


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        LeaderboardConfig(
            prediction_quality_weight=0.50,
            calibration_weight=0.50,
            consistency_weight=0.50,
            efficiency_weight=0.10,
        )


def test_negative_weight_rejected():
    with pytest.raises(ValueError):
        LeaderboardConfig(
            prediction_quality_weight=-0.10,
            calibration_weight=0.50,
            consistency_weight=0.40,
            efficiency_weight=0.20,
        )


def test_boolean_weight_rejected():
    with pytest.raises(TypeError):
        LeaderboardConfig(
            prediction_quality_weight=True,
            calibration_weight=0.30,
            consistency_weight=0.30,
            efficiency_weight=0.40,
        )


def test_invalid_latency_reference():
    with pytest.raises(ValueError):
        LeaderboardConfig(
            latency_reference_ms=0.0
        )


def test_invalid_cost_reference():
    with pytest.raises(ValueError):
        LeaderboardConfig(
            cost_reference=0.0
        )


# ============================================================
# EFFICIENCY
# ============================================================


def test_efficiency_score_is_bounded():
    engine = make_engine()

    score = engine.calculate_efficiency_score(
        latency_ms=100.0,
        cost=0.1,
    )

    assert 0.0 <= score <= 1.0


def test_missing_efficiency_data_returns_neutral():
    engine = make_engine()

    assert (
        engine.calculate_efficiency_score(
            None,
            None,
        )
        == 0.5
    )


def test_lower_latency_improves_efficiency():
    engine = make_engine()

    fast = engine.calculate_efficiency_score(
        latency_ms=100.0,
        cost=0.1,
    )

    slow = engine.calculate_efficiency_score(
        latency_ms=5000.0,
        cost=0.1,
    )

    assert fast > slow


def test_lower_cost_improves_efficiency():
    engine = make_engine()

    cheap = engine.calculate_efficiency_score(
        latency_ms=100.0,
        cost=0.01,
    )

    expensive = engine.calculate_efficiency_score(
        latency_ms=100.0,
        cost=10.0,
    )

    assert cheap > expensive


def test_negative_latency_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.calculate_efficiency_score(
            -1.0,
            0.1,
        )


def test_negative_cost_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.calculate_efficiency_score(
            100.0,
            -0.1,
        )


# ============================================================
# SCORE
# ============================================================


def test_composite_score_is_bounded():
    engine = make_engine()

    score = engine.calculate_score(
        prediction_quality=0.9,
        calibration_score=0.8,
        consistency_score=0.7,
        latency_ms=100.0,
        cost=0.1,
    )

    assert 0.0 <= score <= 1.0


def test_higher_quality_produces_higher_score():
    engine = make_engine()

    strong = engine.calculate_score(
        0.95,
        0.90,
        0.90,
        100.0,
        0.1,
    )

    weak = engine.calculate_score(
        0.30,
        0.30,
        0.30,
        100.0,
        0.1,
    )

    assert strong > weak


def test_invalid_prediction_quality_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.calculate_score(
            1.1,
            0.5,
            0.5,
            100.0,
            0.1,
        )


# ============================================================
# ADD ENTRY
# ============================================================


def test_add_agent_entry():
    engine = make_engine()

    result = add_agent(engine)

    assert result.participant_name == "MacroAgent"
    assert result.participant_type == "AGENT"


def test_add_model_entry():
    engine = make_engine()

    result = engine.add_entry(
        participant_name="ResearchLLM",
        participant_type="MODEL",
        asset="BTC",
        regime="RISK_ON",
        horizon="1D",
        prediction_quality=0.85,
    )

    assert result.participant_type == "MODEL"


def test_entry_has_composite_score():
    engine = make_engine()

    result = add_agent(engine)

    assert 0.0 <= result.score <= 1.0


def test_entry_is_research_only():
    engine = make_engine()

    result = add_agent(engine)

    assert result.metadata["research_only"] is True
    assert result.metadata["real_money_execution"] is False


def test_invalid_participant_type_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.add_entry(
            participant_name="Invalid",
            participant_type="HUMAN",
            asset="BTC",
            regime="RISK_ON",
            horizon="1D",
        )


def test_empty_participant_name_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.add_entry(
            participant_name=" ",
            participant_type="AGENT",
            asset="BTC",
            regime="RISK_ON",
            horizon="1D",
        )


def test_empty_asset_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.add_entry(
            participant_name="Agent",
            participant_type="AGENT",
            asset="",
            regime="RISK_ON",
            horizon="1D",
        )


def test_negative_observations_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.add_entry(
            participant_name="Agent",
            participant_type="AGENT",
            asset="BTC",
            regime="RISK_ON",
            horizon="1D",
            observations=-1,
        )


def test_boolean_observations_rejected():
    engine = make_engine()

    with pytest.raises(TypeError):
        engine.add_entry(
            participant_name="Agent",
            participant_type="AGENT",
            asset="BTC",
            regime="RISK_ON",
            horizon="1D",
            observations=True,
        )


# ============================================================
# READ / FILTER
# ============================================================


def test_get_entry():
    engine = make_engine()

    created = add_agent(engine)

    result = engine.get(
        created.entry_id
    )

    assert result.entry_id == created.entry_id


def test_unknown_entry_rejected():
    engine = make_engine()

    with pytest.raises(KeyError):
        engine.get("missing")


def test_list_entries():
    engine = make_engine()

    add_agent(engine, name="A")
    add_agent(engine, name="B")

    assert len(engine.list_entries()) == 2


def test_filter_by_asset():
    engine = make_engine()

    add_agent(
        engine,
        name="BTC Agent",
        asset="BTC",
    )

    add_agent(
        engine,
        name="SPY Agent",
        asset="SPY",
    )

    results = engine.filter_entries(
        asset="BTC"
    )

    assert len(results) == 1
    assert results[0].asset == "BTC"


def test_filter_by_regime():
    engine = make_engine()

    add_agent(
        engine,
        name="Risk On",
        regime="RISK_ON",
    )

    add_agent(
        engine,
        name="Risk Off",
        regime="RISK_OFF",
    )

    results = engine.filter_entries(
        regime="RISK_OFF"
    )

    assert len(results) == 1


def test_filter_by_horizon():
    engine = make_engine()

    add_agent(
        engine,
        name="Short",
        horizon="1D",
    )

    add_agent(
        engine,
        name="Long",
        horizon="30D",
    )

    results = engine.filter_entries(
        horizon="30D"
    )

    assert len(results) == 1


def test_filter_by_participant_type():
    engine = make_engine()

    add_agent(engine)

    engine.add_entry(
        participant_name="Model",
        participant_type="MODEL",
        asset="BTC",
        regime="RISK_ON",
        horizon="1D",
    )

    results = engine.filter_entries(
        participant_type="MODEL"
    )

    assert len(results) == 1
    assert results[0].participant_type == "MODEL"


# ============================================================
# RANKING
# ============================================================


def test_rank_orders_highest_score_first():
    engine = make_engine()

    add_agent(
        engine,
        name="Weak",
        prediction_quality=0.30,
        calibration_score=0.30,
        consistency_score=0.30,
    )

    add_agent(
        engine,
        name="Strong",
        prediction_quality=0.95,
        calibration_score=0.90,
        consistency_score=0.90,
    )

    ranking = engine.rank()

    assert ranking[0].participant_name == "Strong"
    assert ranking[0].rank == 1
    assert ranking[1].rank == 2


def test_rank_can_be_context_filtered():
    engine = make_engine()

    add_agent(
        engine,
        name="BTC",
        asset="BTC",
    )

    add_agent(
        engine,
        name="SPY",
        asset="SPY",
    )

    ranking = engine.rank(
        asset="BTC"
    )

    assert len(ranking) == 1
    assert ranking[0].participant_name == "BTC"


def test_top_limits_results():
    engine = make_engine()

    add_agent(engine, name="A")
    add_agent(engine, name="B")
    add_agent(engine, name="C")

    results = engine.top(limit=2)

    assert len(results) == 2


def test_invalid_top_limit_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.top(limit=0)


# ============================================================
# PERFORMANCE MATRIX
# ============================================================


def test_performance_matrix_contains_contexts():
    engine = make_engine()

    add_agent(
        engine,
        name="MacroAgent",
        asset="BTC",
        regime="RISK_ON",
        horizon="1D",
    )

    add_agent(
        engine,
        name="MacroAgent",
        asset="SPY",
        regime="RISK_OFF",
        horizon="7D",
    )

    matrix = engine.performance_matrix(
        "MacroAgent"
    )

    assert (
        "BTC",
        "RISK_ON",
        "1D",
    ) in matrix

    assert (
        "SPY",
        "RISK_OFF",
        "7D",
    ) in matrix


# ============================================================
# PARTICIPANT SUMMARY
# ============================================================


def test_participant_summary():
    engine = make_engine()

    add_agent(
        engine,
        name="MacroAgent",
        asset="BTC",
    )

    add_agent(
        engine,
        name="MacroAgent",
        asset="SPY",
    )

    summary = engine.participant_summary(
        "MacroAgent"
    )

    assert summary["contexts"] == 2
    assert summary["total_observations"] == 200
    assert 0.0 <= summary["average_score"] <= 1.0


def test_unknown_participant_summary_rejected():
    engine = make_engine()

    with pytest.raises(KeyError):
        engine.participant_summary(
            "MissingAgent"
        )


# ============================================================
# SNAPSHOT / AUDIT
# ============================================================


def test_snapshot_counts_agents_and_models():
    engine = make_engine()

    add_agent(engine)

    engine.add_entry(
        participant_name="Model",
        participant_type="MODEL",
        asset="BTC",
        regime="RISK_ON",
        horizon="1D",
    )

    snapshot = engine.snapshot()

    assert snapshot["total_entries"] == 2
    assert snapshot["agent_entries"] == 1
    assert snapshot["model_entries"] == 1


def test_snapshot_is_research_only():
    engine = make_engine()

    snapshot = engine.snapshot()

    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False


def test_add_entry_creates_audit_event():
    engine = make_engine()

    add_agent(engine)

    events = engine.audit_log()

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "LEADERBOARD_ENTRY_ADDED"
    )