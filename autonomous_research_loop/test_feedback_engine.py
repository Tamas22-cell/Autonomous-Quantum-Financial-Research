"""
Tests for v5.0.9 Learning Feedback Engine.
"""

import pytest

from autonomous_research_loop.feedback_engine import (
    FeedbackEngine,
    FeedbackEngineConfig,
)
from autonomous_research_loop.models import (
    LearningFeedback,
    ResearchOutcome,
)


def make_outcome(
    *,
    strategy_id: str = "strategy-test",
    return_pct: float = 0.10,
    alpha: float | None = 0.05,
    hit: bool | None = True,
    max_drawdown: float = 0.05,
    pnl: float = 1000.0,
) -> ResearchOutcome:
    """
    Create a deterministic ResearchOutcome for tests.
    """

    return ResearchOutcome(
        strategy_id=strategy_id,
        pnl=pnl,
        return_pct=return_pct,
        benchmark_return_pct=(
            None
            if alpha is None
            else return_pct - alpha
        ),
        alpha=alpha,
        hit=hit,
        metadata={
            "shadow_allocation_id": "shadow-test",
            "max_drawdown": max_drawdown,
            "success": hit,
            "research_only": True,
            "real_money_execution": False,
        },
    )


def test_default_config():
    config = FeedbackEngineConfig()

    assert config.strong_performance_score == pytest.approx(0.75)
    assert config.weak_performance_score == pytest.approx(0.40)
    assert config.strong_alpha == pytest.approx(0.05)
    assert config.weak_alpha == pytest.approx(0.0)
    assert config.acceptable_drawdown == pytest.approx(0.20)
    assert config.create_task_on_failure is True
    assert config.create_task_on_weak_performance is True


def test_invalid_strong_performance_score_below_zero():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            strong_performance_score=-0.01
        )


def test_invalid_strong_performance_score_above_one():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            strong_performance_score=1.01
        )


def test_invalid_weak_performance_score_below_zero():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            weak_performance_score=-0.01
        )


def test_invalid_weak_performance_score_above_one():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            weak_performance_score=1.01
        )


def test_weak_threshold_cannot_exceed_strong_threshold():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            weak_performance_score=0.80,
            strong_performance_score=0.70,
        )


def test_invalid_drawdown_limit_below_zero():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            acceptable_drawdown=-0.01
        )


def test_invalid_drawdown_limit_above_one():
    with pytest.raises(ValueError):
        FeedbackEngineConfig(
            acceptable_drawdown=1.01
        )


def test_engine_accepts_custom_config():
    config = FeedbackEngineConfig(
        strong_performance_score=0.80,
        weak_performance_score=0.30,
        acceptable_drawdown=0.15,
    )

    engine = FeedbackEngine(config)

    assert engine.config is config


def test_clamp_lower_bound():
    assert FeedbackEngine._clamp(-10.0) == pytest.approx(0.0)


def test_clamp_upper_bound():
    assert FeedbackEngine._clamp(10.0) == pytest.approx(1.0)


def test_clamp_middle_value():
    assert FeedbackEngine._clamp(0.55) == pytest.approx(0.55)


def test_get_drawdown_from_metadata():
    outcome = make_outcome(
        max_drawdown=0.12
    )

    assert FeedbackEngine._get_drawdown(
        outcome
    ) == pytest.approx(0.12)


def test_negative_drawdown_is_normalized():
    outcome = make_outcome(
        max_drawdown=-0.12
    )

    assert FeedbackEngine._get_drawdown(
        outcome
    ) == pytest.approx(0.12)


def test_drawdown_is_capped_at_one():
    outcome = make_outcome(
        max_drawdown=5.0
    )

    assert FeedbackEngine._get_drawdown(
        outcome
    ) == pytest.approx(1.0)


def test_missing_drawdown_defaults_to_zero():
    outcome = make_outcome()
    outcome.metadata.pop("max_drawdown")

    assert FeedbackEngine._get_drawdown(
        outcome
    ) == pytest.approx(0.0)


def test_invalid_drawdown_defaults_to_zero():
    outcome = make_outcome()
    outcome.metadata["max_drawdown"] = "invalid"

    assert FeedbackEngine._get_drawdown(
        outcome
    ) == pytest.approx(0.0)


def test_performance_score_is_between_zero_and_one():
    engine = FeedbackEngine()
    outcome = make_outcome()

    score = engine.calculate_performance_score(
        outcome
    )

    assert 0.0 <= score <= 1.0


def test_strong_outcome_has_high_performance_score():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=0.10,
        alpha=0.06,
        hit=True,
        max_drawdown=0.02,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert score >= engine.config.strong_performance_score


def test_failed_outcome_scores_lower_than_successful_outcome():
    engine = FeedbackEngine()

    successful = make_outcome(
        return_pct=0.08,
        alpha=0.04,
        hit=True,
        max_drawdown=0.05,
    )

    failed = make_outcome(
        return_pct=0.08,
        alpha=0.04,
        hit=False,
        max_drawdown=0.05,
    )

    successful_score = (
        engine.calculate_performance_score(
            successful
        )
    )

    failed_score = (
        engine.calculate_performance_score(
            failed
        )
    )

    assert successful_score > failed_score


def test_negative_return_reduces_performance_score():
    engine = FeedbackEngine()

    positive = make_outcome(
        return_pct=0.10,
        alpha=0.05,
        hit=True,
    )

    negative = make_outcome(
        return_pct=-0.10,
        alpha=0.05,
        hit=True,
    )

    assert (
        engine.calculate_performance_score(
            positive
        )
        >
        engine.calculate_performance_score(
            negative
        )
    )


def test_negative_alpha_reduces_performance_score():
    engine = FeedbackEngine()

    positive = make_outcome(
        alpha=0.05
    )

    negative = make_outcome(
        alpha=-0.05
    )

    assert (
        engine.calculate_performance_score(
            positive
        )
        >
        engine.calculate_performance_score(
            negative
        )
    )


def test_larger_drawdown_reduces_performance_score():
    engine = FeedbackEngine()

    low_drawdown = make_outcome(
        max_drawdown=0.02
    )

    high_drawdown = make_outcome(
        max_drawdown=0.20
    )

    assert (
        engine.calculate_performance_score(
            low_drawdown
        )
        >
        engine.calculate_performance_score(
            high_drawdown
        )
    )


def test_none_alpha_is_supported():
    engine = FeedbackEngine()

    outcome = make_outcome(
        alpha=None
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert 0.0 <= score <= 1.0


def test_invalid_outcome_for_performance_score():
    engine = FeedbackEngine()

    with pytest.raises(TypeError):
        engine.calculate_performance_score(
            "not-an-outcome"
        )


def test_success_lesson_is_generated():
    engine = FeedbackEngine()

    outcome = make_outcome(
        hit=True
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "satisfied" in lesson.lower()
        for lesson in lessons
    )


def test_failure_lesson_is_generated():
    engine = FeedbackEngine()

    outcome = make_outcome(
        hit=False
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "failed" in lesson.lower()
        for lesson in lessons
    )


def test_positive_return_lesson():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=0.05
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "positive forward return" in lesson.lower()
        for lesson in lessons
    )


def test_negative_return_lesson():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=-0.05,
        hit=False,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "negative forward return" in lesson.lower()
        for lesson in lessons
    )


def test_negative_alpha_lesson():
    engine = FeedbackEngine()

    outcome = make_outcome(
        alpha=-0.05,
        hit=False,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "underperformed" in lesson.lower()
        for lesson in lessons
    )


def test_excessive_drawdown_lesson():
    engine = FeedbackEngine()

    outcome = make_outcome(
        max_drawdown=0.30,
        hit=False,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    lessons = engine.build_lessons(
        outcome,
        score,
    )

    assert any(
        "exceeded" in lesson.lower()
        for lesson in lessons
    )


def test_successful_outcome_does_not_create_new_task():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=0.10,
        alpha=0.06,
        hit=True,
        max_drawdown=0.02,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert (
        engine.should_create_new_task(
            outcome,
            score,
        )
        is False
    )


def test_failed_outcome_creates_new_task():
    engine = FeedbackEngine()

    outcome = make_outcome(
        hit=False
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert (
        engine.should_create_new_task(
            outcome,
            score,
        )
        is True
    )


def test_failure_task_creation_can_be_disabled():
    config = FeedbackEngineConfig(
        create_task_on_failure=False,
        create_task_on_weak_performance=False,
    )

    engine = FeedbackEngine(
        config
    )

    outcome = make_outcome(
        hit=False
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert (
        engine.should_create_new_task(
            outcome,
            score,
        )
        is False
    )


def test_negative_alpha_generates_review_action():
    engine = FeedbackEngine()

    outcome = make_outcome(
        alpha=-0.05,
        hit=False,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    actions = engine.build_recommended_actions(
        outcome,
        score,
    )

    assert any(
        "benchmark underperformance" in action.lower()
        for action in actions
    )


def test_excessive_drawdown_generates_risk_action():
    engine = FeedbackEngine()

    outcome = make_outcome(
        max_drawdown=0.30,
        hit=False,
    )

    score = engine.calculate_performance_score(
        outcome
    )

    actions = engine.build_recommended_actions(
        outcome,
        score,
    )

    assert any(
        "risk constraints" in action.lower()
        for action in actions
    )


def test_actions_always_keep_research_only():
    engine = FeedbackEngine()

    outcome = make_outcome()

    score = engine.calculate_performance_score(
        outcome
    )

    actions = engine.build_recommended_actions(
        outcome,
        score,
    )

    assert any(
        "research or shadow mode" in action.lower()
        for action in actions
    )


def test_generate_returns_learning_feedback():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine.generate(
        outcome
    )

    assert isinstance(
        result,
        LearningFeedback,
    )


def test_generate_preserves_strategy_id():
    engine = FeedbackEngine()

    outcome = make_outcome(
        strategy_id="strategy-123"
    )

    result = engine.generate(
        outcome
    )

    assert result.strategy_id == "strategy-123"


def test_generate_preserves_outcome_id():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine.generate(
        outcome
    )

    assert result.outcome_id == outcome.outcome_id


def test_generated_feedback_has_valid_score():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine.generate(
        outcome
    )

    assert 0.0 <= result.performance_score <= 1.0


def test_generated_feedback_contains_lessons():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine.generate(
        outcome
    )

    assert result.lessons
    assert isinstance(
        result.lessons,
        list,
    )


def test_generated_feedback_contains_actions():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine.generate(
        outcome
    )

    assert result.recommended_actions
    assert isinstance(
        result.recommended_actions,
        list,
    )


def test_generated_feedback_metadata():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=0.10,
        alpha=0.06,
        hit=True,
        max_drawdown=0.05,
    )

    result = engine.generate(
        outcome
    )

    assert result.metadata["engine"] == (
        "v5.0.9-feedback-engine"
    )

    assert result.metadata["source"] == (
        "SHADOW_OUTCOME"
    )

    assert result.metadata["research_only"] is True

    assert (
        result.metadata["real_money_execution"]
        is False
    )

    assert (
        result.metadata["automatic_live_promotion"]
        is False
    )

    assert (
        result.metadata["feeds_meta_learning"]
        is True
    )

    assert (
        result.metadata["feeds_alpha_lifecycle"]
        is True
    )

    assert (
        result.metadata["feeds_agent_tournament"]
        is True
    )


def test_generated_feedback_contains_outcome_values():
    engine = FeedbackEngine()

    outcome = make_outcome(
        return_pct=0.08,
        alpha=0.03,
        max_drawdown=0.07,
    )

    result = engine.generate(
        outcome
    )

    assert result.metadata["return_pct"] == pytest.approx(
        0.08
    )

    assert result.metadata["alpha"] == pytest.approx(
        0.03
    )

    assert result.metadata["max_drawdown"] == pytest.approx(
        0.07
    )


def test_generate_rejects_invalid_type():
    engine = FeedbackEngine()

    with pytest.raises(TypeError):
        engine.generate(
            "invalid"
        )


def test_callable_interface():
    engine = FeedbackEngine()

    outcome = make_outcome()

    result = engine(
        outcome
    )

    assert isinstance(
        result,
        LearningFeedback,
    )

    assert result.strategy_id == outcome.strategy_id
    assert result.outcome_id == outcome.outcome_id


def test_zero_acceptable_drawdown_with_zero_drawdown():
    config = FeedbackEngineConfig(
        acceptable_drawdown=0.0
    )

    engine = FeedbackEngine(
        config
    )

    outcome = make_outcome(
        max_drawdown=0.0
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert 0.0 <= score <= 1.0


def test_zero_acceptable_drawdown_with_positive_drawdown():
    config = FeedbackEngineConfig(
        acceptable_drawdown=0.0
    )

    engine = FeedbackEngine(
        config
    )

    outcome = make_outcome(
        max_drawdown=0.10
    )

    score = engine.calculate_performance_score(
        outcome
    )

    assert 0.0 <= score <= 1.0