"""
Tests for Causal Research Engine core models.
Version: v5.0.5
"""

from causal_research_engine.models import (
    CausalEvidenceScore,
    CausalHypothesis,
    CausalResearchReport,
    CausalTestResult,
    ConfounderCheck,
    EventStudyResult,
    EvidenceType,
    HypothesisStatus,
    LeakageCheck,
    TestOutcome,
)


def test_hypothesis_defaults():
    hypothesis = CausalHypothesis(
        hypothesis_id="H001",
        cause="funding_rate",
        effect="btc_return",
    )

    assert hypothesis.hypothesis_id == "H001"
    assert hypothesis.cause == "funding_rate"
    assert hypothesis.effect == "btc_return"
    assert hypothesis.status == HypothesisStatus.PROPOSED
    assert hypothesis.confounders == []
    assert hypothesis.controls == []
    assert hypothesis.metadata == {}


def test_hypothesis_full_configuration():
    hypothesis = CausalHypothesis(
        hypothesis_id="H002",
        cause="interest_rates",
        effect="equity_returns",
        description="Rate shock hypothesis",
        expected_direction="negative",
        mechanism="Higher discount rates reduce valuations.",
        confounders=["inflation"],
        controls=["market_regime"],
        status=HypothesisStatus.TESTING,
        metadata={"asset": "SPY"},
    )

    assert hypothesis.expected_direction == "negative"
    assert hypothesis.mechanism is not None
    assert hypothesis.confounders == ["inflation"]
    assert hypothesis.controls == ["market_regime"]
    assert hypothesis.status == HypothesisStatus.TESTING
    assert hypothesis.metadata["asset"] == "SPY"


def test_confounder_severity_is_bounded_upper():
    check = ConfounderCheck(
        variable="inflation",
        outcome=TestOutcome.WARNING,
        severity=2.5,
    )

    assert check.severity == 1.0


def test_confounder_severity_is_bounded_lower():
    check = ConfounderCheck(
        variable="inflation",
        outcome=TestOutcome.PASSED,
        severity=-1.0,
    )

    assert check.severity == 0.0


def test_leakage_severity_is_bounded():
    check = LeakageCheck(
        check_name="temporal",
        outcome=TestOutcome.FAILED,
        leakage_detected=True,
        severity=4.0,
    )

    assert check.severity == 1.0


def test_causal_test_confidence_is_bounded():
    result = CausalTestResult(
        test_name="placebo",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.PASSED,
        confidence=5.0,
    )

    assert result.confidence == 1.0


def test_causal_test_p_value_upper_bound():
    result = CausalTestResult(
        test_name="falsification",
        evidence_type=EvidenceType.FALSIFICATION,
        outcome=TestOutcome.FAILED,
        p_value=3.0,
    )

    assert result.p_value == 1.0


def test_causal_test_p_value_lower_bound():
    result = CausalTestResult(
        test_name="negative_control",
        evidence_type=EvidenceType.NEGATIVE_CONTROL,
        outcome=TestOutcome.PASSED,
        p_value=-2.0,
    )

    assert result.p_value == 0.0


def test_event_study_values_are_bounded():
    result = EventStudyResult(
        event_name="FOMC",
        p_value=2.0,
        confidence=-1.0,
    )

    assert result.p_value == 1.0
    assert result.confidence == 0.0


def test_evidence_score_fields_are_bounded():
    score = CausalEvidenceScore(
        predictive_score=2.0,
        causal_score=-1.0,
        confounder_score=1.5,
        leakage_score=-0.5,
        falsification_score=3.0,
        event_evidence_score=-4.0,
        confidence=7.0,
    )

    assert score.predictive_score == 1.0
    assert score.causal_score == 0.0
    assert score.confounder_score == 1.0
    assert score.leakage_score == 0.0
    assert score.falsification_score == 1.0
    assert score.event_evidence_score == 0.0
    assert score.confidence == 1.0


def test_research_report_defaults():
    hypothesis = CausalHypothesis(
        hypothesis_id="H003",
        cause="oil_price",
        effect="energy_equity_return",
    )

    report = CausalResearchReport(
        hypothesis=hypothesis,
    )

    assert report.hypothesis is hypothesis
    assert report.confounder_checks == []
    assert report.leakage_checks == []
    assert report.causal_tests == []
    assert report.event_studies == []
    assert report.evidence_score is None
    assert report.warnings == []
    assert report.metadata == {}


def test_evidence_type_values():
    assert EvidenceType.OBSERVATIONAL.value == "observational"
    assert EvidenceType.EVENT_STUDY.value == "event_study"
    assert EvidenceType.INTERVENTION.value == "intervention"
    assert EvidenceType.PLACEBO.value == "placebo"
    assert EvidenceType.FALSIFICATION.value == "falsification"
    assert EvidenceType.NEGATIVE_CONTROL.value == "negative_control"


def test_hypothesis_status_values():
    assert HypothesisStatus.PROPOSED.value == "proposed"
    assert HypothesisStatus.TESTING.value == "testing"
    assert HypothesisStatus.SUPPORTED.value == "supported"
    assert HypothesisStatus.WEAK.value == "weak"
    assert HypothesisStatus.FALSIFIED.value == "falsified"
    assert HypothesisStatus.INCONCLUSIVE.value == "inconclusive"


def test_test_outcome_values():
    assert TestOutcome.PASSED.value == "passed"
    assert TestOutcome.FAILED.value == "failed"
    assert TestOutcome.WARNING.value == "warning"
    assert TestOutcome.INCONCLUSIVE.value == "inconclusive"