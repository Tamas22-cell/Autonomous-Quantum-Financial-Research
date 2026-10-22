"""
Tests for v5.0.9 Hypothesis Engine.

Covered:
- hypothesis generation
- confidence bounds
- regime-change hypothesis
- risk-event hypothesis
- uncertainty hypothesis
- performance-gap hypothesis
- alpha-decay hypothesis
- agent-signal hypothesis
- evidence and assumptions
- callable dependency interface
"""

from autonomous_research_loop.hypothesis_engine import (
    HypothesisEngine,
    HypothesisEngineConfig,
)
from autonomous_research_loop.models import (
    Hypothesis,
    ResearchPriority,
    ResearchTask,
    ResearchTrigger,
)


def make_task(
    trigger: ResearchTrigger = ResearchTrigger.AGENT_SIGNAL,
) -> ResearchTask:
    return ResearchTask(
        title="BTC autonomous research",
        description="Research task generated from orchestrator.",
        trigger=trigger,
        asset="BTC",
        regime="RISK-ON",
        horizon="7D",
        priority=ResearchPriority.HIGH,
        uncertainty_score=0.25,
        performance_gap_score=0.30,
        metadata={
            "decision": "BUY",
            "market_score": 0.68,
            "average_confidence": 0.82,
            "consensus_strength": 0.76,
            "signal_conflict": 0.12,
            "top_positive_signals": [
                "MACRO: Growth conditions constructive.",
                "ONCHAIN: Accumulation remains positive.",
            ],
            "top_risks": [
                "RISK: Volatility remains elevated.",
            ],
            "market_interpretation": (
                "Constructive market environment."
            ),
        },
    )


def test_generate_returns_hypothesis() -> None:
    engine = HypothesisEngine()
    task = make_task()

    hypothesis = engine.generate(task)

    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.task_id == task.task_id
    assert hypothesis.asset == "BTC"
    assert hypothesis.regime == "RISK-ON"
    assert hypothesis.horizon == "7D"


def test_confidence_is_bounded() -> None:
    engine = HypothesisEngine(
        HypothesisEngineConfig(
            minimum_confidence=0.10,
            maximum_confidence=0.95,
        )
    )

    task = make_task()

    confidence = engine.calculate_confidence(task)

    assert 0.10 <= confidence <= 0.95


def test_regime_change_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.REGIME_CHANGE
    )

    hypothesis = engine.generate(task)

    assert "risk-adjusted performance" in hypothesis.statement
    assert "RISK-ON" in hypothesis.statement
    assert "preceding regime" in hypothesis.statement


def test_risk_event_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.RISK_EVENT
    )

    hypothesis = engine.generate(task)

    assert "downside risk" in hypothesis.statement
    assert "drawdown behavior" in hypothesis.statement


def test_uncertainty_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.UNCERTAINTY
    )

    hypothesis = engine.generate(task)

    assert "uncertainty" in hypothesis.statement
    assert "volatility" in hypothesis.statement


def test_performance_gap_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.PERFORMANCE_GAP
    )

    hypothesis = engine.generate(task)

    assert "disagreement" in hypothesis.statement
    assert "predictive information" in hypothesis.statement


def test_alpha_decay_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.ALPHA_DECAY
    )

    hypothesis = engine.generate(task)

    assert "predictive relationship" in hypothesis.statement
    assert "weakened materially" in hypothesis.statement


def test_agent_signal_statement() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.AGENT_SIGNAL
    )

    hypothesis = engine.generate(task)

    assert "market score" in hypothesis.statement
    assert "positive" in hypothesis.statement
    assert "BTC" in hypothesis.statement


def test_evidence_contains_agent_information() -> None:
    engine = HypothesisEngine()

    task = make_task()

    hypothesis = engine.generate(task)

    evidence_text = " ".join(
        hypothesis.evidence
    )

    assert "MACRO" in evidence_text
    assert "ONCHAIN" in evidence_text
    assert "RISK" in evidence_text

    assert "AGENT_SIGNAL" in evidence_text


def test_assumptions_are_explicit() -> None:
    engine = HypothesisEngine()

    task = make_task()

    hypothesis = engine.generate(task)

    assumptions_text = " ".join(
        hypothesis.assumptions
    ).lower()

    assert "look-ahead" in assumptions_text
    assert "transaction costs" in assumptions_text
    assert "validation" in assumptions_text
    assert "market-regime" in assumptions_text


def test_low_quality_inputs_reduce_confidence() -> None:
    engine = HypothesisEngine()

    strong_task = make_task()
    weak_task = make_task()

    weak_task.uncertainty_score = 0.90
    weak_task.metadata["average_confidence"] = 0.20
    weak_task.metadata["consensus_strength"] = 0.20
    weak_task.metadata["signal_conflict"] = 0.90

    strong_confidence = engine.calculate_confidence(
        strong_task
    )

    weak_confidence = engine.calculate_confidence(
        weak_task
    )

    assert weak_confidence < strong_confidence


def test_metadata_is_auditable() -> None:
    engine = HypothesisEngine()

    task = make_task(
        ResearchTrigger.REGIME_CHANGE
    )

    hypothesis = engine.generate(task)

    assert (
        hypothesis.metadata["generator"]
        == "v5.0.9-hypothesis-engine"
    )

    assert (
        hypothesis.metadata["research_trigger"]
        == "REGIME_CHANGE"
    )

    assert (
        hypothesis.metadata["source_priority"]
        == "HIGH"
    )


def test_engine_is_callable_for_research_loop_dependency() -> None:
    engine = HypothesisEngine()

    task = make_task()

    hypothesis = engine(task)

    assert isinstance(hypothesis, Hypothesis)
    assert hypothesis.task_id == task.task_id