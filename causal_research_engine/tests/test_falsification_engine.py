"""
Tests for the Causal Falsification Engine.
Version: v5.0.5
"""

from causal_research_engine.falsification_engine import (
    FalsificationEngine,
)
from causal_research_engine.models import (
    EvidenceType,
    TestOutcome,
)


def test_placebo_without_data_is_inconclusive():
    engine = FalsificationEngine()

    result = engine.run_placebo_test()

    assert result.evidence_type == EvidenceType.PLACEBO
    assert result.outcome == TestOutcome.INCONCLUSIVE


def test_placebo_with_non_significant_result_passes():
    engine = FalsificationEngine()

    result = engine.run_placebo_test(
        p_value=0.80,
        effect_size=0.02,
    )

    assert result.outcome == TestOutcome.PASSED
    assert result.p_value == 0.80
    assert result.effect_size == 0.02


def test_placebo_with_significant_effect_fails():
    engine = FalsificationEngine()

    result = engine.run_placebo_test(
        p_value=0.01,
        effect_size=0.20,
    )

    assert result.outcome == TestOutcome.FAILED


def test_falsification_without_data_is_inconclusive():
    engine = FalsificationEngine()

    result = engine.run_falsification_test()

    assert result.evidence_type == EvidenceType.FALSIFICATION
    assert result.outcome == TestOutcome.INCONCLUSIVE


def test_falsification_passes_when_effect_is_absent():
    engine = FalsificationEngine()

    result = engine.run_falsification_test(
        p_value=0.70,
        effect_size=0.01,
    )

    assert result.outcome == TestOutcome.PASSED


def test_falsification_fails_when_effect_is_present():
    engine = FalsificationEngine()

    result = engine.run_falsification_test(
        p_value=0.01,
        effect_size=0.30,
    )

    assert result.outcome == TestOutcome.FAILED


def test_negative_control_without_data_is_inconclusive():
    engine = FalsificationEngine()

    result = engine.run_negative_control_test()

    assert result.evidence_type == EvidenceType.NEGATIVE_CONTROL
    assert result.outcome == TestOutcome.INCONCLUSIVE


def test_negative_control_passes():
    engine = FalsificationEngine()

    result = engine.run_negative_control_test(
        p_value=0.90,
        effect_size=0.01,
    )

    assert result.outcome == TestOutcome.PASSED


def test_negative_control_fails():
    engine = FalsificationEngine()

    result = engine.run_negative_control_test(
        p_value=0.01,
        effect_size=0.25,
    )

    assert result.outcome == TestOutcome.FAILED


def test_custom_test_metadata_is_preserved():
    engine = FalsificationEngine()

    result = engine.run_placebo_test(
        test_name="BTC_placebo",
        statistic=2.15,
        p_value=0.55,
        effect_size=0.03,
        confidence=0.90,
        notes="Placebo test for BTC funding hypothesis.",
    )

    assert result.test_name == "BTC_placebo"
    assert result.statistic == 2.15
    assert result.p_value == 0.55
    assert result.effect_size == 0.03
    assert result.confidence == 0.90
    assert result.notes == (
        "Placebo test for BTC funding hypothesis."
    )


def test_run_all_creates_three_test_results():
    engine = FalsificationEngine()

    results = engine.run_all()

    assert len(results) == 3

    assert results[0].evidence_type == EvidenceType.PLACEBO
    assert results[1].evidence_type == EvidenceType.FALSIFICATION
    assert results[2].evidence_type == EvidenceType.NEGATIVE_CONTROL


def test_run_all_can_disable_tests():
    engine = FalsificationEngine()

    results = engine.run_all(
        placebo=False,
        falsification=True,
        negative_control=False,
    )

    assert len(results) == 1
    assert results[0].evidence_type == EvidenceType.FALSIFICATION