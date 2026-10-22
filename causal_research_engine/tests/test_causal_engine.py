"""
Tests for the main Causal Research Engine.
Version: v5.0.5
"""

import pytest

from causal_research_engine.causal_engine import (
    CausalEngine,
    CausalEngineError,
)
from causal_research_engine.models import (
    CausalHypothesis,
    CausalTestResult,
    ConfounderCheck,
    EvidenceType,
    EventStudyResult,
    HypothesisStatus,
    LeakageCheck,
    TestOutcome,
)


def create_hypothesis():
    engine = CausalEngine()

    return engine.create_hypothesis(
        hypothesis_id="H001",
        cause="funding_rate",
        effect="btc_return",
        description="Funding pressure may affect future BTC returns.",
        expected_direction="negative",
        mechanism="High leverage increases liquidation pressure.",
        confounders=["market_regime"],
        controls=["volume"],
    )


def test_engine_initializes_all_components():
    engine = CausalEngine()

    assert engine.hypothesis_engine is not None
    assert engine.diagnostic_engine is not None
    assert engine.falsification_engine is not None
    assert engine.event_study_engine is not None
    assert engine.evidence_score_engine is not None


def test_create_hypothesis():
    engine = CausalEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H002",
        cause="interest_rates",
        effect="equity_returns",
        expected_direction="negative",
        mechanism="Higher discount rates can reduce valuations.",
    )

    assert isinstance(hypothesis, CausalHypothesis)
    assert hypothesis.hypothesis_id == "H002"
    assert hypothesis.cause == "interest_rates"
    assert hypothesis.effect == "equity_returns"
    assert hypothesis.status == HypothesisStatus.PROPOSED


def test_start_testing():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    result = engine.start_testing(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.TESTING


def test_start_testing_invalid_hypothesis():
    engine = CausalEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="INVALID",
        cause="funding_rate",
        effect="btc_return",
    )

    with pytest.raises(CausalEngineError):
        engine.start_testing(hypothesis)


def test_run_research_returns_report():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.80,
    )

    assert report.hypothesis is hypothesis
    assert report.evidence_score is not None
    assert report.evidence_score.predictive_score == 0.80


def test_run_research_with_diagnostics():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    confounder = ConfounderCheck(
        variable="market_regime",
        outcome=TestOutcome.PASSED,
        severity=0.10,
    )

    leakage = LeakageCheck(
        check_name="future_information",
        outcome=TestOutcome.PASSED,
        leakage_detected=False,
        severity=0.0,
    )

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.80,
        confounder_checks=[confounder],
        leakage_checks=[leakage],
    )

    assert len(report.confounder_checks) == 1
    assert len(report.leakage_checks) == 1
    assert report.evidence_score is not None
    assert report.evidence_score.confounder_score == 0.90
    assert report.evidence_score.leakage_score == 1.0


def test_run_research_with_causal_tests():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    causal_test = CausalTestResult(
        test_name="placebo_test",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.PASSED,
        confidence=1.0,
    )

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.75,
        causal_tests=[causal_test],
    )

    assert len(report.causal_tests) == 1
    assert report.causal_tests[0].outcome == TestOutcome.PASSED
    assert report.evidence_score is not None


def test_run_research_with_event_study():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    event = EventStudyResult(
        event_name="BTC_EVENT",
        pre_event_mean=0.01,
        post_event_mean=0.05,
        abnormal_effect=0.04,
        cumulative_abnormal_effect=0.08,
        p_value=0.02,
        confidence=0.90,
    )

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.75,
        event_studies=[event],
    )

    assert len(report.event_studies) == 1
    assert report.event_studies[0].event_name == "BTC_EVENT"
    assert report.evidence_score is not None
    assert report.evidence_score.event_evidence_score >= 0.90


def test_run_from_raw_event():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    report = engine.run_from_raw_event(
        hypothesis=hypothesis,
        predictive_score=0.80,
        event_name="BTC_ETF_EVENT",
        pre_event_values=[1.0, 2.0, 3.0],
        post_event_values=[4.0, 5.0, 6.0],
    )

    assert report.hypothesis is hypothesis
    assert len(report.event_studies) == 1

    event = report.event_studies[0]

    assert event.event_name == "BTC_ETF_EVENT"
    assert event.pre_event_mean == 2.0
    assert event.post_event_mean == 5.0
    assert event.abnormal_effect == 3.0


def test_run_from_raw_event_with_benchmark():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    report = engine.run_from_raw_event(
        hypothesis=hypothesis,
        predictive_score=0.80,
        event_name="MARKET_EVENT",
        pre_event_values=[2.0, 4.0],
        post_event_values=[6.0, 8.0],
        benchmark_effect=1.0,
    )

    event = report.event_studies[0]

    assert event.pre_event_mean == 3.0
    assert event.post_event_mean == 7.0
    assert event.abnormal_effect == 3.0


def test_diagnostic_summary():
    engine = CausalEngine()

    confounder = ConfounderCheck(
        variable="inflation",
        outcome=TestOutcome.WARNING,
        severity=0.30,
    )

    leakage = LeakageCheck(
        check_name="lookahead",
        outcome=TestOutcome.PASSED,
        leakage_detected=False,
        severity=0.0,
    )

    summary = engine.diagnostic_summary(
        confounder_checks=[confounder],
        leakage_checks=[leakage],
    )

    assert summary["confounder_count"] == 1
    assert summary["leakage_check_count"] == 1
    assert summary["leakage_detected_count"] == 0


def test_warnings_are_generated_without_diagnostics():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.80,
    )

    assert "No confounder checks were supplied." in report.warnings
    assert "No leakage checks were supplied." in report.warnings


def test_critical_leakage_generates_warning():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    leakage = LeakageCheck(
        check_name="future_information",
        outcome=TestOutcome.FAILED,
        leakage_detected=True,
        severity=1.0,
    )

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.90,
        leakage_checks=[leakage],
    )

    assert report.evidence_score is not None
    assert report.evidence_score.leakage_score == 0.0
    assert any(
        "leakage" in warning.lower()
        for warning in report.warnings
    )


def test_invalid_hypothesis_is_rejected():
    engine = CausalEngine()

    invalid = "not-a-hypothesis"

    with pytest.raises(CausalEngineError):
        engine.run_research(
            hypothesis=invalid,
            predictive_score=0.80,
        )


def test_predictive_and_causal_scores_are_separate():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.95,
    )

    assert report.evidence_score is not None
    assert report.evidence_score.predictive_score == 0.95
    assert (
        report.evidence_score.predictive_score
        != report.evidence_score.causal_score
    )


def test_complete_causal_pipeline():
    engine = CausalEngine()

    hypothesis = create_hypothesis()

    confounder = ConfounderCheck(
        variable="market_regime",
        outcome=TestOutcome.PASSED,
        severity=0.10,
    )

    leakage = LeakageCheck(
        check_name="temporal_leakage",
        outcome=TestOutcome.PASSED,
        leakage_detected=False,
        severity=0.0,
    )

    placebo = CausalTestResult(
        test_name="placebo_test",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.PASSED,
        confidence=1.0,
    )

    event = EventStudyResult(
        event_name="BTC_EVENT",
        pre_event_mean=0.01,
        post_event_mean=0.04,
        abnormal_effect=0.03,
        cumulative_abnormal_effect=0.06,
        p_value=0.02,
        confidence=0.90,
    )

    report = engine.run_research(
        hypothesis=hypothesis,
        predictive_score=0.85,
        confounder_checks=[confounder],
        leakage_checks=[leakage],
        causal_tests=[placebo],
        event_studies=[event],
        confidence=0.90,
        notes=["Complete causal pipeline test."],
    )

    assert report.hypothesis is hypothesis
    assert len(report.confounder_checks) == 1
    assert len(report.leakage_checks) == 1
    assert len(report.causal_tests) == 1
    assert len(report.event_studies) == 1

    assert report.evidence_score is not None
    assert report.evidence_score.predictive_score == 0.85
    assert report.evidence_score.confounder_score == 0.90
    assert report.evidence_score.leakage_score == 1.0
    assert report.evidence_score.falsification_score == 1.0

    assert "Complete causal pipeline test." in (
        report.evidence_score.explanation
    )