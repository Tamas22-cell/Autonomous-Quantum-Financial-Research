import pytest

from research_control_center.alpha_library import (
    AlphaLibrary,
    AlphaLibraryConfig,
)
from research_control_center.models import (
    AlphaRecord,
    AlphaStatus,
    RobustnessStatus,
)


def make_library() -> AlphaLibrary:
    return AlphaLibrary()


def make_alpha(
    name: str = "BTC Momentum Alpha",
    asset: str = "BTC",
    regime: str = "RISK_ON",
    horizon: str = "7D",
) -> AlphaRecord:
    return AlphaRecord(
        name=name,
        asset=asset,
        regime=regime,
        horizon=horizon,
        expected_return=0.08,
        source_strategy_id="strategy_001",
        source_hypothesis_id="hypothesis_001",
        source_experiment_id="experiment_001",
    )


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = AlphaLibraryConfig()

    assert config.active_validation_threshold == 0.65
    assert config.active_robustness_threshold == 0.60
    assert config.decay_warning_threshold == 0.50
    assert config.retirement_decay_threshold == 0.80


def test_custom_config():
    config = AlphaLibraryConfig(
        active_validation_threshold=0.70,
        active_robustness_threshold=0.65,
        decay_warning_threshold=0.40,
        retirement_decay_threshold=0.90,
        weak_robustness_threshold=0.30,
        strong_robustness_threshold=0.80,
    )

    assert config.active_validation_threshold == 0.70
    assert config.retirement_decay_threshold == 0.90


def test_invalid_config_score_rejected():
    with pytest.raises(ValueError):
        AlphaLibraryConfig(
            active_validation_threshold=1.1
        )


def test_boolean_config_rejected():
    with pytest.raises(TypeError):
        AlphaLibraryConfig(
            active_validation_threshold=True
        )


def test_invalid_decay_threshold_order_rejected():
    with pytest.raises(ValueError):
        AlphaLibraryConfig(
            decay_warning_threshold=0.90,
            retirement_decay_threshold=0.80,
        )


def test_invalid_robustness_threshold_order_rejected():
    with pytest.raises(ValueError):
        AlphaLibraryConfig(
            weak_robustness_threshold=0.80,
            strong_robustness_threshold=0.70,
        )


# ============================================================
# REGISTRATION
# ============================================================


def test_register_alpha():
    library = make_library()
    alpha = make_alpha()

    result = library.register(alpha)

    assert result.alpha_id == alpha.alpha_id
    assert library.count() == 1


def test_registered_alpha_is_research_only():
    library = make_library()

    result = library.register(
        make_alpha()
    )

    assert result.metadata["research_only"] is True
    assert result.metadata["real_money_execution"] is False


def test_duplicate_alpha_rejected():
    library = make_library()
    alpha = make_alpha()

    library.register(alpha)

    with pytest.raises(ValueError):
        library.register(alpha)


def test_invalid_registration_type_rejected():
    library = make_library()

    with pytest.raises(TypeError):
        library.register("invalid")


def test_empty_alpha_name_rejected():
    library = make_library()

    alpha = make_alpha()
    alpha.name = " "

    with pytest.raises(ValueError):
        library.register(alpha)


# ============================================================
# CREATE
# ============================================================


def test_create_alpha():
    library = make_library()

    result = library.create(
        name="SPY Mean Reversion",
        asset="SPY",
        regime="MIXED",
        horizon="1D",
        expected_return=0.03,
        source_strategy_id="strategy_002",
    )

    assert result.name == "SPY Mean Reversion"
    assert result.asset == "SPY"
    assert result.status == AlphaStatus.DISCOVERY
    assert library.count() == 1


def test_create_empty_asset_rejected():
    library = make_library()

    with pytest.raises(ValueError):
        library.create(
            name="Alpha",
            asset="",
            horizon="1D",
        )


# ============================================================
# READ
# ============================================================


def test_get_alpha():
    library = make_library()

    created = library.register(
        make_alpha()
    )

    result = library.get(
        created.alpha_id
    )

    assert result.alpha_id == created.alpha_id


def test_get_unknown_alpha_rejected():
    library = make_library()

    with pytest.raises(KeyError):
        library.get("missing")


def test_get_returns_copy():
    library = make_library()

    created = library.register(
        make_alpha()
    )

    result = library.get(
        created.alpha_id
    )

    result.name = "CHANGED"

    original = library.get(
        created.alpha_id
    )

    assert original.name != "CHANGED"


def test_list_all():
    library = make_library()

    library.register(
        make_alpha(name="Alpha A")
    )

    library.register(
        make_alpha(name="Alpha B")
    )

    assert len(library.list_all()) == 2


# ============================================================
# ROBUSTNESS
# ============================================================


def test_weak_robustness():
    library = make_library()

    assert (
        library.classify_robustness(0.20)
        == RobustnessStatus.WEAK
    )


def test_moderate_robustness():
    library = make_library()

    assert (
        library.classify_robustness(0.55)
        == RobustnessStatus.MODERATE
    )


def test_strong_robustness():
    library = make_library()

    assert (
        library.classify_robustness(0.85)
        == RobustnessStatus.STRONG
    )


def test_invalid_robustness_score_rejected():
    library = make_library()

    with pytest.raises(ValueError):
        library.classify_robustness(1.5)


# ============================================================
# VALIDATION
# ============================================================


def test_update_validation():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.update_validation(
        alpha.alpha_id,
        validation_score=0.80,
        out_of_sample_score=0.75,
        robustness_score=0.82,
    )

    assert result.validation_score == 0.80
    assert result.out_of_sample_score == 0.75
    assert result.robustness_score == 0.82
    assert result.robustness_status == RobustnessStatus.STRONG
    assert result.status == AlphaStatus.VALIDATING


def test_invalid_validation_score_rejected():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    with pytest.raises(ValueError):
        library.update_validation(
            alpha.alpha_id,
            validation_score=1.2,
        )


# ============================================================
# PERFORMANCE
# ============================================================


def test_update_performance():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.update_performance(
        alpha.alpha_id,
        realized_return=0.12,
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        max_drawdown=-0.08,
    )

    assert result.realized_return == 0.12
    assert result.sharpe_ratio == 1.8
    assert result.sortino_ratio == 2.1
    assert result.max_drawdown == -0.08


# ============================================================
# DECAY
# ============================================================


def test_low_decay_does_not_force_decay_status():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.update_decay(
        alpha.alpha_id,
        0.20,
    )

    assert result.decay_score == 0.20
    assert result.status == AlphaStatus.DISCOVERY


def test_decay_warning_changes_status():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.update_decay(
        alpha.alpha_id,
        0.60,
    )

    assert result.status == AlphaStatus.DECAYING


def test_high_decay_retires_alpha():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.update_decay(
        alpha.alpha_id,
        0.90,
    )

    assert result.status == AlphaStatus.RETIRED


def test_invalid_decay_score_rejected():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    with pytest.raises(ValueError):
        library.update_decay(
            alpha.alpha_id,
            1.2,
        )


# ============================================================
# LIFECYCLE
# ============================================================


def test_validated_robust_alpha_becomes_active():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.85,
        out_of_sample_score=0.80,
        robustness_score=0.80,
    )

    result = library.evaluate_lifecycle(
        alpha.alpha_id
    )

    assert result.status == AlphaStatus.ACTIVE


def test_weak_validation_stays_validating():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.40,
        robustness_score=0.80,
    )

    result = library.evaluate_lifecycle(
        alpha.alpha_id
    )

    assert result.status == AlphaStatus.VALIDATING


def test_decay_overrides_active_status():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.90,
        robustness_score=0.90,
    )

    library.evaluate_lifecycle(
        alpha.alpha_id
    )

    result = library.update_decay(
        alpha.alpha_id,
        0.60,
    )

    assert result.status == AlphaStatus.DECAYING


def test_retired_alpha_stays_retired():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.retire(
        alpha.alpha_id,
        "Alpha exhausted",
    )

    result = library.evaluate_lifecycle(
        alpha.alpha_id
    )

    assert result.status == AlphaStatus.RETIRED


def test_rejected_alpha_stays_rejected():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.reject(
        alpha.alpha_id,
        "Failed validation",
    )

    result = library.evaluate_lifecycle(
        alpha.alpha_id
    )

    assert result.status == AlphaStatus.REJECTED


# ============================================================
# MANUAL LIFECYCLE
# ============================================================


def test_reject_alpha():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.reject(
        alpha.alpha_id,
        "OOS failure",
    )

    assert result.status == AlphaStatus.REJECTED


def test_retire_alpha():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    result = library.retire(
        alpha.alpha_id,
        "Structural decay",
    )

    assert result.status == AlphaStatus.RETIRED


def test_empty_rejection_reason_rejected():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    with pytest.raises(ValueError):
        library.reject(
            alpha.alpha_id,
            " ",
        )


# ============================================================
# FILTERING
# ============================================================


def test_filter_by_asset():
    library = make_library()

    library.register(
        make_alpha(
            name="BTC Alpha",
            asset="BTC",
        )
    )

    library.register(
        make_alpha(
            name="SPY Alpha",
            asset="SPY",
        )
    )

    results = library.filter(
        asset="BTC"
    )

    assert len(results) == 1
    assert results[0].asset == "BTC"


def test_filter_by_regime():
    library = make_library()

    library.register(
        make_alpha(
            name="Risk On",
            regime="RISK_ON",
        )
    )

    library.register(
        make_alpha(
            name="Risk Off",
            regime="RISK_OFF",
        )
    )

    results = library.filter(
        regime="RISK_OFF"
    )

    assert len(results) == 1


def test_filter_by_status():
    library = make_library()

    alpha = library.register(
        make_alpha(name="Active")
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.90,
        robustness_score=0.90,
    )

    library.evaluate_lifecycle(
        alpha.alpha_id
    )

    library.register(
        make_alpha(name="Discovery")
    )

    results = library.filter(
        status=AlphaStatus.ACTIVE
    )

    assert len(results) == 1
    assert results[0].name == "Active"


def test_filter_by_robustness():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.80,
        robustness_score=0.90,
    )

    results = library.filter(
        robustness_status=RobustnessStatus.STRONG
    )

    assert len(results) == 1


# ============================================================
# QUALITY / RANKING
# ============================================================


def test_quality_score_is_bounded():
    library = make_library()

    alpha = make_alpha()
    alpha.validation_score = 0.90
    alpha.out_of_sample_score = 0.80
    alpha.robustness_score = 0.85
    alpha.decay_score = 0.10

    score = library.alpha_quality_score(
        alpha
    )

    assert 0.0 <= score <= 1.0


def test_decay_reduces_quality_score():
    library = make_library()

    strong = make_alpha(name="Strong")
    strong.validation_score = 0.90
    strong.out_of_sample_score = 0.90
    strong.robustness_score = 0.90
    strong.decay_score = 0.10

    decayed = make_alpha(name="Decayed")
    decayed.validation_score = 0.90
    decayed.out_of_sample_score = 0.90
    decayed.robustness_score = 0.90
    decayed.decay_score = 0.70

    assert (
        library.alpha_quality_score(strong)
        > library.alpha_quality_score(decayed)
    )


def test_rank_orders_best_alpha_first():
    library = make_library()

    weak = library.register(
        make_alpha(name="Weak")
    )

    strong = library.register(
        make_alpha(name="Strong")
    )

    library.update_validation(
        weak.alpha_id,
        validation_score=0.40,
        out_of_sample_score=0.40,
        robustness_score=0.40,
    )

    library.update_validation(
        strong.alpha_id,
        validation_score=0.90,
        out_of_sample_score=0.90,
        robustness_score=0.90,
    )

    ranking = library.rank()

    assert ranking[0].name == "Strong"


def test_top_limits_results():
    library = make_library()

    library.register(
        make_alpha(name="A")
    )

    library.register(
        make_alpha(name="B")
    )

    library.register(
        make_alpha(name="C")
    )

    assert len(
        library.top(limit=2)
    ) == 2


def test_invalid_top_limit_rejected():
    library = make_library()

    with pytest.raises(ValueError):
        library.top(limit=0)


# ============================================================
# SUMMARY / AUDIT / SNAPSHOT
# ============================================================


def test_summary_counts_lifecycle_states():
    library = make_library()

    active = library.register(
        make_alpha(name="Active")
    )

    library.update_validation(
        active.alpha_id,
        validation_score=0.90,
        robustness_score=0.90,
    )

    library.evaluate_lifecycle(
        active.alpha_id
    )

    decaying = library.register(
        make_alpha(name="Decaying")
    )

    library.update_decay(
        decaying.alpha_id,
        0.60,
    )

    retired = library.register(
        make_alpha(name="Retired")
    )

    library.update_decay(
        retired.alpha_id,
        0.90,
    )

    summary = library.summary()

    assert summary["total_alphas"] == 3
    assert summary["active_alphas"] == 1
    assert summary["decaying_alphas"] == 1
    assert summary["retired_alphas"] == 1


def test_registration_creates_audit_event():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    events = library.audit_log(
        alpha.alpha_id
    )

    assert len(events) == 1
    assert events[0]["event"] == "ALPHA_REGISTERED"


def test_lifecycle_operations_are_audited():
    library = make_library()

    alpha = library.register(
        make_alpha()
    )

    library.update_validation(
        alpha.alpha_id,
        validation_score=0.80,
        robustness_score=0.80,
    )

    library.evaluate_lifecycle(
        alpha.alpha_id
    )

    library.update_decay(
        alpha.alpha_id,
        0.60,
    )

    events = library.audit_log(
        alpha.alpha_id
    )

    assert len(events) == 4


def test_snapshot_is_research_only():
    library = make_library()

    snapshot = library.snapshot()

    assert snapshot["research_only"] is True
    assert snapshot["real_money_execution"] is False