import pytest

from research_control_center.backtest_diagnostics import (
    BacktestDiagnosticsConfig,
    BacktestDiagnosticsEngine,
)
from research_control_center.models import BacktestDiagnostic


def make_engine() -> BacktestDiagnosticsEngine:
    return BacktestDiagnosticsEngine()


def make_diagnostic(
    strategy_id: str = "strategy_001",
) -> BacktestDiagnostic:
    return BacktestDiagnostic(
        strategy_id=strategy_id,
        in_sample_return=0.20,
        out_of_sample_return=0.15,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=1.5,
        in_sample_drawdown=-0.10,
        out_of_sample_drawdown=-0.14,
        robustness_score=0.80,
        p_value=0.03,
        confidence_interval_low=0.01,
        confidence_interval_high=0.08,
        observations=100,
    )


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = BacktestDiagnosticsConfig()

    assert config.max_sharpe_degradation == 0.40
    assert config.max_return_degradation == 0.50
    assert config.minimum_robustness_score == 0.60
    assert config.significance_level == 0.05
    assert config.minimum_observations == 30


def test_custom_config():
    config = BacktestDiagnosticsConfig(
        max_sharpe_degradation=0.30,
        max_return_degradation=0.40,
        minimum_robustness_score=0.70,
        significance_level=0.01,
        minimum_observations=50,
    )

    assert config.max_sharpe_degradation == 0.30
    assert config.minimum_observations == 50


def test_invalid_config_score_rejected():
    with pytest.raises(ValueError):
        BacktestDiagnosticsConfig(
            max_sharpe_degradation=1.1
        )


def test_boolean_config_rejected():
    with pytest.raises(TypeError):
        BacktestDiagnosticsConfig(
            max_sharpe_degradation=True
        )


def test_invalid_minimum_observations_rejected():
    with pytest.raises(ValueError):
        BacktestDiagnosticsConfig(
            minimum_observations=0
        )


def test_boolean_minimum_observations_rejected():
    with pytest.raises(TypeError):
        BacktestDiagnosticsConfig(
            minimum_observations=True
        )


# ============================================================
# REGISTRATION
# ============================================================


def test_register_diagnostic():
    engine = make_engine()

    diagnostic = make_diagnostic()

    result = engine.register(diagnostic)

    assert (
        result.diagnostic_id
        == diagnostic.diagnostic_id
    )

    assert engine.count() == 1


def test_registered_diagnostic_is_research_only():
    engine = make_engine()

    result = engine.register(
        make_diagnostic()
    )

    assert result.metadata["research_only"] is True
    assert (
        result.metadata["real_money_execution"]
        is False
    )


def test_duplicate_diagnostic_rejected():
    engine = make_engine()

    diagnostic = make_diagnostic()

    engine.register(diagnostic)

    with pytest.raises(ValueError):
        engine.register(diagnostic)


def test_invalid_registration_type_rejected():
    engine = make_engine()

    with pytest.raises(TypeError):
        engine.register("invalid")


def test_empty_strategy_id_rejected():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.strategy_id = " "

    with pytest.raises(ValueError):
        engine.register(diagnostic)


# ============================================================
# CREATE
# ============================================================


def test_create_diagnostic():
    engine = make_engine()

    result = engine.create(
        strategy_id="strategy_002",
        in_sample_return=0.15,
        out_of_sample_return=0.12,
        in_sample_sharpe=1.8,
        out_of_sample_sharpe=1.5,
        robustness_score=0.75,
        p_value=0.04,
        confidence_interval_low=0.01,
        confidence_interval_high=0.06,
        observations=80,
    )

    assert result.strategy_id == "strategy_002"
    assert result.in_sample_return == 0.15
    assert result.out_of_sample_return == 0.12
    assert engine.count() == 1


def test_invalid_confidence_interval_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.create(
            strategy_id="strategy",
            confidence_interval_low=0.10,
            confidence_interval_high=0.05,
        )


def test_invalid_robustness_score_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.create(
            strategy_id="strategy",
            robustness_score=1.2,
        )


def test_invalid_p_value_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.create(
            strategy_id="strategy",
            p_value=-0.1,
        )


def test_non_finite_number_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.create(
            strategy_id="strategy",
            in_sample_return=float("inf"),
        )


# ============================================================
# READ
# ============================================================


def test_get_diagnostic():
    engine = make_engine()

    created = engine.register(
        make_diagnostic()
    )

    result = engine.get(
        created.diagnostic_id
    )

    assert (
        result.diagnostic_id
        == created.diagnostic_id
    )


def test_unknown_diagnostic_rejected():
    engine = make_engine()

    with pytest.raises(KeyError):
        engine.get("missing")


def test_get_returns_copy():
    engine = make_engine()

    created = engine.register(
        make_diagnostic()
    )

    result = engine.get(
        created.diagnostic_id
    )

    result.strategy_id = "CHANGED"

    original = engine.get(
        created.diagnostic_id
    )

    assert original.strategy_id != "CHANGED"


def test_list_all():
    engine = make_engine()

    engine.register(
        make_diagnostic("strategy_a")
    )

    engine.register(
        make_diagnostic("strategy_b")
    )

    assert len(engine.list_all()) == 2


# ============================================================
# DEGRADATION
# ============================================================


def test_return_degradation():
    result = BacktestDiagnosticsEngine.degradation(
        0.20,
        0.10,
    )

    assert result == pytest.approx(0.50)


def test_no_degradation_when_oos_is_better():
    result = BacktestDiagnosticsEngine.degradation(
        0.10,
        0.15,
    )

    assert result == 0.0


def test_degradation_missing_data():
    result = BacktestDiagnosticsEngine.degradation(
        None,
        0.10,
    )

    assert result is None


def test_zero_zero_degradation():
    result = BacktestDiagnosticsEngine.degradation(
        0.0,
        0.0,
    )

    assert result == 0.0


def test_zero_is_nonzero_oos_degradation():
    result = BacktestDiagnosticsEngine.degradation(
        0.0,
        0.10,
    )

    assert result == 1.0


# ============================================================
# GENERALIZATION
# ============================================================


def test_generalization_score_is_bounded():
    engine = make_engine()

    diagnostic = make_diagnostic()

    score = engine.generalization_score(
        diagnostic
    )

    assert 0.0 <= score <= 1.0


def test_good_oos_has_high_generalization():
    engine = make_engine()

    diagnostic = BacktestDiagnostic(
        strategy_id="good",
        in_sample_return=0.20,
        out_of_sample_return=0.19,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=1.9,
        robustness_score=0.90,
    )

    score = engine.generalization_score(
        diagnostic
    )

    assert score > 0.80


def test_bad_oos_has_lower_generalization():
    engine = make_engine()

    good = BacktestDiagnostic(
        strategy_id="good",
        in_sample_return=0.20,
        out_of_sample_return=0.18,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=1.8,
        robustness_score=0.90,
    )

    bad = BacktestDiagnostic(
        strategy_id="bad",
        in_sample_return=0.20,
        out_of_sample_return=0.02,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=0.2,
        robustness_score=0.30,
    )

    assert (
        engine.generalization_score(good)
        > engine.generalization_score(bad)
    )


# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================


def test_statistically_significant():
    engine = make_engine()

    diagnostic = make_diagnostic()

    assert (
        engine.is_statistically_significant(
            diagnostic
        )
        is True
    )


def test_high_p_value_not_significant():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.p_value = 0.20

    assert (
        engine.is_statistically_significant(
            diagnostic
        )
        is False
    )


def test_small_sample_not_significant():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.observations = 10

    assert (
        engine.is_statistically_significant(
            diagnostic
        )
        is False
    )


def test_missing_p_value_not_significant():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.p_value = None

    assert (
        engine.is_statistically_significant(
            diagnostic
        )
        is False
    )


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================


def test_positive_confidence_interval():
    engine = make_engine()

    diagnostic = make_diagnostic()

    assert (
        engine.has_positive_confidence_interval(
            diagnostic
        )
        is True
    )


def test_confidence_interval_crossing_zero():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.confidence_interval_low = -0.01

    assert (
        engine.has_positive_confidence_interval(
            diagnostic
        )
        is False
    )


# ============================================================
# OVERFITTING
# ============================================================


def test_good_strategy_not_flagged_overfit():
    engine = make_engine()

    diagnostic = make_diagnostic()

    assert (
        engine.potential_overfitting(
            diagnostic
        )
        is False
    )


def test_large_return_degradation_flags_overfit():
    engine = make_engine()

    diagnostic = make_diagnostic()

    diagnostic.in_sample_return = 0.20
    diagnostic.out_of_sample_return = 0.05

    assert (
        engine.potential_overfitting(
            diagnostic
        )
        is True
    )


def test_large_sharpe_degradation_flags_overfit():
    engine = make_engine()

    diagnostic = make_diagnostic()

    diagnostic.in_sample_sharpe = 2.0
    diagnostic.out_of_sample_sharpe = 0.5

    assert (
        engine.potential_overfitting(
            diagnostic
        )
        is True
    )


def test_low_robustness_flags_overfit():
    engine = make_engine()

    diagnostic = make_diagnostic()
    diagnostic.robustness_score = 0.30

    assert (
        engine.potential_overfitting(
            diagnostic
        )
        is True
    )


# ============================================================
# REPORT
# ============================================================


def test_diagnostic_report():
    engine = make_engine()

    diagnostic = engine.register(
        make_diagnostic()
    )

    report = engine.diagnostic_report(
        diagnostic.diagnostic_id
    )

    assert (
        report["strategy_id"]
        == diagnostic.strategy_id
    )

    assert (
        report["statistically_significant"]
        is True
    )

    assert (
        report["potential_overfitting"]
        is False
    )

    assert (
        0.0
        <= report["generalization_score"]
        <= 1.0
    )


def test_report_is_research_only():
    engine = make_engine()

    diagnostic = engine.register(
        make_diagnostic()
    )

    report = engine.diagnostic_report(
        diagnostic.diagnostic_id
    )

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


# ============================================================
# STRATEGY FILTER
# ============================================================


def test_by_strategy():
    engine = make_engine()

    engine.register(
        make_diagnostic("strategy_a")
    )

    engine.register(
        make_diagnostic("strategy_a")
    )

    engine.register(
        make_diagnostic("strategy_b")
    )

    results = engine.by_strategy(
        "strategy_a"
    )

    assert len(results) == 2


# ============================================================
# RANKING
# ============================================================


def test_rank_best_generalization_first():
    engine = make_engine()

    good = BacktestDiagnostic(
        strategy_id="good",
        in_sample_return=0.20,
        out_of_sample_return=0.19,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=1.9,
        robustness_score=0.90,
    )

    bad = BacktestDiagnostic(
        strategy_id="bad",
        in_sample_return=0.20,
        out_of_sample_return=0.02,
        in_sample_sharpe=2.0,
        out_of_sample_sharpe=0.2,
        robustness_score=0.20,
    )

    engine.register(bad)
    engine.register(good)

    ranking = engine.rank()

    assert ranking[0].strategy_id == "good"


def test_top_limits_results():
    engine = make_engine()

    engine.register(
        make_diagnostic("A")
    )

    engine.register(
        make_diagnostic("B")
    )

    engine.register(
        make_diagnostic("C")
    )

    assert len(engine.top(2)) == 2


def test_invalid_top_limit_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.top(0)


# ============================================================
# COMPARISON
# ============================================================


def test_compare_diagnostics():
    engine = make_engine()

    a = engine.register(
        BacktestDiagnostic(
            strategy_id="A",
            in_sample_return=0.20,
            out_of_sample_return=0.10,
            in_sample_sharpe=2.0,
            out_of_sample_sharpe=1.0,
            robustness_score=0.50,
        )
    )

    b = engine.register(
        BacktestDiagnostic(
            strategy_id="B",
            in_sample_return=0.20,
            out_of_sample_return=0.19,
            in_sample_sharpe=2.0,
            out_of_sample_sharpe=1.9,
            robustness_score=0.90,
        )
    )

    comparison = engine.compare(
        a.diagnostic_id,
        b.diagnostic_id,
    )

    assert (
        comparison[
            "generalization_score_b"
        ]
        > comparison[
            "generalization_score_a"
        ]
    )


# ============================================================
# SUMMARY / AUDIT / SNAPSHOT
# ============================================================


def test_summary():
    engine = make_engine()

    engine.register(
        make_diagnostic("good")
    )

    bad = make_diagnostic("bad")
    bad.out_of_sample_return = 0.01
    bad.out_of_sample_sharpe = 0.10
    bad.robustness_score = 0.20
    bad.p_value = 0.50

    engine.register(bad)

    summary = engine.summary()

    assert summary["total_diagnostics"] == 2
    assert summary["potential_overfitting"] == 1
    assert summary["statistically_significant"] == 1
    assert summary["robust_diagnostics"] == 1


def test_registration_creates_audit_event():
    engine = make_engine()

    diagnostic = engine.register(
        make_diagnostic()
    )

    events = engine.audit_log(
        diagnostic.diagnostic_id
    )

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "DIAGNOSTIC_REGISTERED"
    )


def test_snapshot_is_research_only():
    engine = make_engine()

    snapshot = engine.snapshot()

    assert snapshot["research_only"] is True
    assert (
        snapshot["real_money_execution"]
        is False
    )