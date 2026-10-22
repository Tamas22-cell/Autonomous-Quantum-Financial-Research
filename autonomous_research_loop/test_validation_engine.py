"""
Tests for v5.0.9 Strategy Validation Engine.

Covered:
- PROMOTE
- REVIEW
- REJECT
- out-of-sample hard gate
- robustness hard gate
- causal hard gate
- Sharpe / Sortino gates
- drawdown protection
- minimum sample size
- weighted composite score
- mapping adapter
- audit metadata
- invalid inputs
"""

import pytest

from autonomous_research_loop.models import (
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)
from autonomous_research_loop.validation_engine import (
    StrategyValidationEngine,
    ValidationEngineConfig,
    ValidationMetrics,
)


def make_candidate() -> StrategyCandidate:
    return StrategyCandidate(
        hypothesis_id="hypothesis-test-001",
        name="BTC LONG 7D research candidate",
        entry_rule=(
            "Enter LONG research exposure when "
            "validation conditions are satisfied."
        ),
        exit_rule=(
            "Exit after 7D or on hypothesis invalidation."
        ),
        asset="BTC",
        horizon="7D",
        position_size=0.05,
        stop_condition=(
            "Stop on hypothesis invalidation or Risk Governor veto."
        ),
        parameters={
            "direction": "LONG",
            "research_only": True,
            "real_money_execution": False,
            "requires_validation": True,
            "requires_risk_gate": True,
            "shadow_before_trust": True,
        },
    )


def make_strong_metrics() -> ValidationMetrics:
    return ValidationMetrics(
        out_of_sample_score=0.82,
        robustness_score=0.78,
        causal_score=0.72,
        sharpe_ratio=1.40,
        sortino_ratio=1.65,
        maximum_drawdown=0.10,
        sample_size=250,
    )


def test_strong_candidate_is_promoted() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    result = engine.validate(
        candidate,
        make_strong_metrics(),
    )

    assert isinstance(result, ValidationResult)
    assert result.decision == ValidationDecision.PROMOTE
    assert result.strategy_id == candidate.strategy_id
    assert result.score >= 0.70


def test_composite_score_is_weighted_correctly() -> None:
    engine = StrategyValidationEngine()

    metrics = ValidationMetrics(
        out_of_sample_score=0.80,
        robustness_score=0.70,
        causal_score=0.60,
        sharpe_ratio=1.20,
        sortino_ratio=1.30,
        maximum_drawdown=0.10,
        sample_size=100,
    )

    score = engine.calculate_score(metrics)

    expected = (
        0.80 * 0.40
        + 0.70 * 0.35
        + 0.60 * 0.25
    )

    assert score == pytest.approx(
        expected,
        abs=1e-6,
    )


def test_intermediate_candidate_goes_to_review() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.60,
        robustness_score=0.58,
        causal_score=0.50,
        sharpe_ratio=0.90,
        sortino_ratio=1.00,
        maximum_drawdown=0.15,
        sample_size=100,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision == ValidationDecision.REVIEW
    assert 0.50 <= result.score < 0.70


def test_weak_candidate_is_rejected() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.30,
        robustness_score=0.25,
        causal_score=0.20,
        sharpe_ratio=0.20,
        sortino_ratio=0.30,
        maximum_drawdown=0.35,
        sample_size=10,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision == ValidationDecision.REJECT
    assert result.score < 0.50


def test_low_out_of_sample_score_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.50,
        robustness_score=0.95,
        causal_score=0.95,
        sharpe_ratio=1.80,
        sortino_ratio=2.00,
        maximum_drawdown=0.08,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "Out-of-sample" in reason
        for reason in result.reasons
    )


def test_low_robustness_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.95,
        robustness_score=0.40,
        causal_score=0.95,
        sharpe_ratio=1.80,
        sortino_ratio=2.00,
        maximum_drawdown=0.08,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "Robustness" in reason
        for reason in result.reasons
    )


def test_low_causal_score_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.95,
        robustness_score=0.95,
        causal_score=0.30,
        sharpe_ratio=1.80,
        sortino_ratio=2.00,
        maximum_drawdown=0.08,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "Causal" in reason
        for reason in result.reasons
    )


def test_low_sharpe_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.90,
        robustness_score=0.90,
        causal_score=0.90,
        sharpe_ratio=0.50,
        sortino_ratio=1.50,
        maximum_drawdown=0.08,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "Sharpe" in reason
        for reason in result.reasons
    )


def test_low_sortino_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.90,
        robustness_score=0.90,
        causal_score=0.90,
        sharpe_ratio=1.50,
        sortino_ratio=0.60,
        maximum_drawdown=0.08,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "Sortino" in reason
        for reason in result.reasons
    )


def test_excessive_drawdown_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.90,
        robustness_score=0.90,
        causal_score=0.90,
        sharpe_ratio=1.50,
        sortino_ratio=1.70,
        maximum_drawdown=0.30,
        sample_size=300,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "drawdown" in reason.lower()
        for reason in result.reasons
    )


def test_small_sample_blocks_promotion() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.90,
        robustness_score=0.90,
        causal_score=0.90,
        sharpe_ratio=1.50,
        sortino_ratio=1.70,
        maximum_drawdown=0.08,
        sample_size=20,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision != ValidationDecision.PROMOTE
    assert any(
        "sample size" in reason.lower()
        for reason in result.reasons
    )


def test_mapping_adapter_accepts_backtester_style_dictionary() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    metrics = {
        "out_of_sample_score": 0.82,
        "robustness_score": 0.78,
        "causal_score": 0.72,
        "sharpe_ratio": 1.40,
        "sortino_ratio": 1.65,
        "max_drawdown": 0.10,
        "sample_size": 250,
    }

    result = engine.validate_from_mapping(
        candidate,
        metrics,
    )

    assert result.decision == ValidationDecision.PROMOTE
    assert result.max_drawdown == 0.10


def test_promoted_candidate_still_requires_risk_governor() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    result = engine.validate(
        candidate,
        make_strong_metrics(),
    )

    assert result.decision == ValidationDecision.PROMOTE

    assert (
        result.metadata["requires_risk_governor"]
        is True
    )

    assert (
        result.metadata["real_money_authorized"]
        is False
    )

    assert (
        result.metadata["shadow_only_after_risk_gate"]
        is True
    )


def test_validation_result_preserves_metrics() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()
    metrics = make_strong_metrics()

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.out_of_sample_score == 0.82
    assert result.robustness_score == 0.78
    assert result.causal_score == 0.72
    assert result.sharpe_ratio == 1.40
    assert result.sortino_ratio == 1.65
    assert result.max_drawdown == 0.10


def test_invalid_candidate_raises_type_error() -> None:
    engine = StrategyValidationEngine()

    with pytest.raises(TypeError):
        engine.validate(
            "not-a-strategy-candidate",
            make_strong_metrics(),
        )


def test_invalid_metrics_raise_type_error() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    with pytest.raises(TypeError):
        engine.validate(
            candidate,
            {"invalid": "metrics"},
        )


def test_invalid_validation_score_is_rejected() -> None:
    with pytest.raises(ValueError):
        ValidationMetrics(
            out_of_sample_score=1.20,
            robustness_score=0.80,
            causal_score=0.70,
            sharpe_ratio=1.00,
            sortino_ratio=1.20,
            maximum_drawdown=0.10,
            sample_size=100,
        )


def test_invalid_negative_drawdown_is_rejected() -> None:
    with pytest.raises(ValueError):
        ValidationMetrics(
            out_of_sample_score=0.80,
            robustness_score=0.80,
            causal_score=0.70,
            sharpe_ratio=1.00,
            sortino_ratio=1.20,
            maximum_drawdown=-0.10,
            sample_size=100,
        )


def test_custom_configuration_is_respected() -> None:
    config = ValidationEngineConfig(
        promote_score_threshold=0.80,
        review_score_threshold=0.60,
        minimum_out_of_sample_score=0.50,
        minimum_robustness_score=0.50,
        minimum_causal_score=0.40,
        minimum_sharpe_ratio=0.50,
        minimum_sortino_ratio=0.50,
        maximum_allowed_drawdown=0.25,
        minimum_sample_size=20,
    )

    engine = StrategyValidationEngine(
        config
    )

    candidate = make_candidate()

    metrics = ValidationMetrics(
        out_of_sample_score=0.72,
        robustness_score=0.72,
        causal_score=0.72,
        sharpe_ratio=1.00,
        sortino_ratio=1.10,
        maximum_drawdown=0.10,
        sample_size=100,
    )

    result = engine.validate(
        candidate,
        metrics,
    )

    assert result.decision == ValidationDecision.REVIEW


def test_callable_interface_returns_validation_result() -> None:
    engine = StrategyValidationEngine()
    candidate = make_candidate()

    result = engine(
        candidate,
        make_strong_metrics(),
    )

    assert isinstance(result, ValidationResult)
    assert result.decision == ValidationDecision.PROMOTE