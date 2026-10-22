"""
Full integration test for Causal Research Engine v5.0.5.
"""

from causal_research_engine.causal_engine import CausalEngine
from causal_research_engine.models import (
    CausalTestResult,
    ConfounderCheck,
    EvidenceType,
    LeakageCheck,
    TestOutcome,
)


def test_complete_causal_research_pipeline():
    engine = CausalEngine()

    # 1. Hypothesis
    hypothesis = engine.create_hypothesis(
        hypothesis_id="BTC-FUNDING-001",
        cause="funding_rate",
        effect="btc_forward_return",
        description=(
            "High funding rates may increase future BTC downside risk."
        ),
        expected_direction="negative",
        mechanism=(
            "Excessive leverage increases liquidation pressure."
        ),
        confounders=[
            "market_regime",
            "volatility",
        ],
        controls=[
            "volume",
        ],
    )

    # 2. Diagnostics
    confounder = ConfounderCheck(
        variable="market_regime",
        outcome=TestOutcome.PASSED,
        association_with_cause=0.20,
        association_with_effect=0.15,
        severity=0.10,
        notes="Low residual confounding risk.",
    )

    leakage = LeakageCheck(
        check_name="temporal_leakage",
        outcome=TestOutcome.PASSED,
        leakage_detected=False,
        severity=0.0,
        notes="No future information detected.",
    )

    # 3. Falsification
    placebo = CausalTestResult(
        test_name="placebo_test",
        evidence_type=EvidenceType.PLACEBO,
        outcome=TestOutcome.PASSED,
        statistic=0.12,
        p_value=0.64,
        effect_size=0.01,
        confidence=0.90,
        notes="Placebo relationship not detected.",
    )

    falsification = CausalTestResult(
        test_name="falsification_test",
        evidence_type=EvidenceType.FALSIFICATION,
        outcome=TestOutcome.PASSED,
        statistic=0.18,
        p_value=0.58,
        effect_size=0.02,
        confidence=0.90,
        notes="Falsification test passed.",
    )

    negative_control = CausalTestResult(
        test_name="negative_control_test",
        evidence_type=EvidenceType.NEGATIVE_CONTROL,
        outcome=TestOutcome.PASSED,
        statistic=0.09,
        p_value=0.72,
        effect_size=0.01,
        confidence=0.90,
        notes="Negative control passed.",
    )

    # 4. Event Study + complete causal report
    report = engine.run_from_raw_event(
        hypothesis=hypothesis,
        predictive_score=0.84,
        event_name="BTC_FUNDING_EVENT",
        pre_event_values=[
            -0.01,
            0.00,
            0.01,
        ],
        post_event_values=[
            -0.04,
            -0.03,
            -0.02,
        ],
        confounder_checks=[confounder],
        leakage_checks=[leakage],
        causal_tests=[
            placebo,
            falsification,
            negative_control,
        ],
        benchmark_effect=0.0,
        p_value=0.02,
        confidence=0.90,
        notes=[
            "Full v5.0.5 causal integration test."
        ],
    )

    # 5. Report validation
    assert report.hypothesis is hypothesis

    assert len(report.confounder_checks) == 1
    assert len(report.leakage_checks) == 1
    assert len(report.causal_tests) == 3
    assert len(report.event_studies) == 1

    # 6. Evidence score
    assert report.evidence_score is not None

    assert report.evidence_score.predictive_score == 0.84

    assert (
        report.evidence_score.confounder_score
        == 0.90
    )

    assert report.evidence_score.leakage_score == 1.0

    assert (
        report.evidence_score.falsification_score
        == 1.0
    )

    assert (
        report.evidence_score.event_evidence_score
        >= 0.90
    )

    assert 0.0 <= report.evidence_score.causal_score <= 1.0

    # 7. Event-study validation
    event = report.event_studies[0]

    assert event.event_name == "BTC_FUNDING_EVENT"
    assert event.pre_event_mean == 0.0
    assert event.post_event_mean == -0.03
    assert event.abnormal_effect == -0.03

    # 8. Final auditability
    assert (
        "Full v5.0.5 causal integration test."
        in report.evidence_score.explanation
    )