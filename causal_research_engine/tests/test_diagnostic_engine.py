"""
Tests for the Causal Diagnostic Engine.
Version: v5.0.5
"""

from causal_research_engine.diagnostic_engine import (
    CausalDiagnosticEngine,
)
from causal_research_engine.models import (
    ConfounderCheck,
    LeakageCheck,
    TestOutcome,
)


def test_confounder_evaluation_empty():
    engine = CausalDiagnosticEngine()

    result = engine.evaluate_confounders([])

    assert result == []


def test_confounder_evaluation_preserves_check():
    engine = CausalDiagnosticEngine()

    check = ConfounderCheck(
        variable="inflation",
        outcome=TestOutcome.PASSED,
        association_with_cause=0.20,
        association_with_effect=0.10,
        severity=0.20,
    )

    result = engine.evaluate_confounders([check])

    assert len(result) == 1
    assert result[0].variable == "inflation"
    assert result[0].outcome == TestOutcome.PASSED
    assert result[0].severity == 0.20


def test_high_severity_confounder():
    engine = CausalDiagnosticEngine()

    check = ConfounderCheck(
        variable="inflation",
        outcome=TestOutcome.WARNING,
        severity=0.90,
    )

    result = engine.evaluate_confounders([check])

    assert len(result) == 1
    assert result[0].severity == 0.90
    assert result[0].outcome == TestOutcome.WARNING


def test_leakage_evaluation_empty():
    engine = CausalDiagnosticEngine()

    result = engine.evaluate_leakage([])

    assert result == []


def test_leakage_evaluation_preserves_check():
    engine = CausalDiagnosticEngine()

    check = LeakageCheck(
        check_name="future_information",
        outcome=TestOutcome.PASSED,
        leakage_detected=False,
        severity=0.0,
    )

    result = engine.evaluate_leakage([check])

    assert len(result) == 1
    assert result[0].check_name == "future_information"
    assert result[0].leakage_detected is False
    assert result[0].outcome == TestOutcome.PASSED


def test_detected_leakage():
    engine = CausalDiagnosticEngine()

    check = LeakageCheck(
        check_name="lookahead_bias",
        outcome=TestOutcome.FAILED,
        leakage_detected=True,
        severity=1.0,
    )

    result = engine.evaluate_leakage([check])

    assert len(result) == 1
    assert result[0].leakage_detected is True
    assert result[0].severity == 1.0
    assert result[0].outcome == TestOutcome.FAILED


def test_confounder_score_empty():
    engine = CausalDiagnosticEngine()

    score = engine.confounder_score([])

    assert score == 1.0


def test_leakage_score_empty():
    engine = CausalDiagnosticEngine()

    score = engine.leakage_score([])

    assert score == 1.0


def test_confounder_score_is_bounded():
    engine = CausalDiagnosticEngine()

    checks = [
        ConfounderCheck(
            variable="inflation",
            outcome=TestOutcome.FAILED,
            severity=1.0,
        ),
    ]

    score = engine.confounder_score(checks)

    assert 0.0 <= score <= 1.0


def test_leakage_score_is_bounded():
    engine = CausalDiagnosticEngine()

    checks = [
        LeakageCheck(
            check_name="lookahead_bias",
            outcome=TestOutcome.FAILED,
            leakage_detected=True,
            severity=1.0,
        ),
    ]

    score = engine.leakage_score(checks)

    assert 0.0 <= score <= 1.0