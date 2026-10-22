"""
Tests for the Evidence Score Engine.
Version: v5.0.5
"""

import pytest

from causal_research_engine.evidence_score_engine import (
    EvidenceScoreEngine,
)
from causal_research_engine.models import (
    CausalTestResult,
    EventStudyResult,
    EvidenceType,
    TestOutcome,
)


def test_calculate_without_evidence():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=0.80,
    )

    assert result.predictive_score == 0.80
    assert result.causal_score == pytest.approx(0.60)
    assert result.confounder_score == 1.0
    assert result.leakage_score == 1.0
    assert result.falsification_score == 1.0
    assert result.event_evidence_score == 0.0


def test_scores_are_bounded():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=2.0,
        confounder_score=-1.0,
        leakage_score=5.0,
        confidence=3.0,
    )

    assert 0.0 <= result.predictive_score <= 1.0
    assert 0.0 <= result.causal_score <= 1.0
    assert 0.0 <= result.confounder_score <= 1.0
    assert 0.0 <= result.leakage_score <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_passed_causal_test():
    engine = EvidenceScoreEngine()

    test = CausalTestResult(
        test_name="placebo_test",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.PASSED,
        confidence=1.0,
    )

    result = engine.calculate(
        predictive_score=0.70,
        causal_tests=[test],
    )

    assert result.falsification_score == 1.0
    assert result.causal_score > 0.0


def test_failed_causal_test():
    engine = EvidenceScoreEngine()

    test = CausalTestResult(
        test_name="placebo_test",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.FAILED,
        confidence=1.0,
    )

    result = engine.calculate(
        predictive_score=0.70,
        causal_tests=[test],
    )

    assert result.falsification_score == 0.0


def test_warning_causal_test():
    engine = EvidenceScoreEngine()

    test = CausalTestResult(
        test_name="falsification_test",
        evidence_type=EvidenceType.FALSIFICATION,
        outcome=TestOutcome.WARNING,
        confidence=0.5,
    )

    result = engine.calculate(
        predictive_score=0.70,
        causal_tests=[test],
    )

    assert result.falsification_score == 0.5


def test_event_study_evidence():
    engine = EvidenceScoreEngine()

    study = EventStudyResult(
        event_name="BTC_EVENT",
        abnormal_effect=0.05,
        cumulative_abnormal_effect=0.10,
        p_value=0.01,
        confidence=0.90,
    )

    result = engine.calculate(
        predictive_score=0.75,
        event_studies=[study],
    )

    assert result.event_evidence_score >= 0.90
    assert result.causal_score > 0.0


def test_non_significant_event_study():
    engine = EvidenceScoreEngine()

    study = EventStudyResult(
        event_name="BTC_EVENT",
        abnormal_effect=0.02,
        cumulative_abnormal_effect=0.03,
        p_value=0.50,
        confidence=0.20,
    )

    result = engine.calculate(
        predictive_score=0.75,
        event_studies=[study],
    )

    assert result.event_evidence_score == 0.5


def test_custom_falsification_score():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=0.80,
        falsification_score=0.25,
    )

    assert result.falsification_score == 0.25


def test_explanation_is_generated():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=0.80,
        confounder_score=0.90,
        leakage_score=0.95,
        falsification_score=0.85,
    )

    assert len(result.explanation) >= 5
    assert any(
        "Predictive score" in item
        for item in result.explanation
    )
    assert any(
        "Causal score" in item
        for item in result.explanation
    )


def test_custom_explanation_is_preserved():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=0.80,
        explanation=["Custom research note."],
    )

    assert "Custom research note." in result.explanation


def test_predictive_vs_causal_comparison():
    engine = EvidenceScoreEngine()

    score = engine.calculate(
        predictive_score=0.90,
        causal_tests=[],
    )

    comparison = engine.compare_predictive_vs_causal(
        score
    )

    assert comparison["predictive_score"] == 0.90
    assert comparison["causal_score"] == score.causal_score
    assert comparison["difference"] == pytest.approx(
        0.90 - score.causal_score
    )


def test_predictive_and_causal_scores_remain_separate():
    engine = EvidenceScoreEngine()

    result = engine.calculate(
        predictive_score=0.95,
        causal_tests=[],
        event_studies=[],
        confounder_score=1.0,
        leakage_score=1.0,
        falsification_score=1.0,
    )

    assert result.predictive_score == 0.95
    assert result.predictive_score != result.causal_score


def test_multiple_falsification_tests():
    engine = EvidenceScoreEngine()

    tests = [
        CausalTestResult(
            test_name="placebo_test",
            evidence_type=EvidenceType.PLACEBO,
            outcome=TestOutcome.PASSED,
        ),
        CausalTestResult(
            test_name="falsification_test",
            evidence_type=EvidenceType.FALSIFICATION,
            outcome=TestOutcome.PASSED,
        ),
        CausalTestResult(
            test_name="negative_control_test",
            evidence_type=EvidenceType.NEGATIVE_CONTROL,
            outcome=TestOutcome.PASSED,
        ),
    ]

    result = engine.calculate(
        predictive_score=0.80,
        causal_tests=tests,
    )

    assert result.falsification_score == 1.0


def test_mixed_falsification_results():
    engine = EvidenceScoreEngine()

    tests = [
        CausalTestResult(
            test_name="placebo_test",
            evidence_type=EvidenceType.PLACEBO,
            outcome=TestOutcome.PASSED,
        ),
        CausalTestResult(
            test_name="falsification_test",
            evidence_type=EvidenceType.FALSIFICATION,
            outcome=TestOutcome.FAILED,
        ),
    ]

    result = engine.calculate(
        predictive_score=0.80,
        causal_tests=tests,
    )

    assert result.falsification_score == 0.5


def test_confounded_result_reduces_causal_score():
    engine = EvidenceScoreEngine()

    clean = engine.calculate(
        predictive_score=0.80,
        confounder_score=1.0,
        leakage_score=1.0,
        falsification_score=1.0,
    )

    confounded = engine.calculate(
        predictive_score=0.80,
        confounder_score=0.20,
        leakage_score=1.0,
        falsification_score=1.0,
    )

    assert confounded.causal_score < clean.causal_score


def test_leakage_reduces_causal_score():
    engine = EvidenceScoreEngine()

    clean = engine.calculate(
        predictive_score=0.80,
        leakage_score=1.0,
    )

    leaked = engine.calculate(
        predictive_score=0.80,
        leakage_score=0.10,
    )

    assert leaked.causal_score < clean.causal_score