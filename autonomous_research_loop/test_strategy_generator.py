"""
Tests for v5.0.9 Strategy Generator.

Covered:
- StrategyCandidate generation
- LONG / SHORT / NEUTRAL direction
- confidence-based research sizing
- maximum position-size protection
- entry and exit rules
- stop conditions
- research-only safety flags
- metadata/audit trail
- callable dependency interface
"""

from autonomous_research_loop.models import (
    Hypothesis,
    StrategyCandidate,
)
from autonomous_research_loop.strategy_generator import (
    StrategyGenerator,
    StrategyGeneratorConfig,
)


def make_hypothesis(
    confidence: float = 0.70,
    statement: str = (
        "The current multi-agent signal contains measurable "
        "positive predictive information for BTC over the 7D horizon."
    ),
    regime: str = "RISK-ON",
) -> Hypothesis:
    return Hypothesis(
        task_id="task-test-001",
        statement=statement,
        asset="BTC",
        regime=regime,
        horizon="7D",
        confidence=confidence,
        evidence=[
            "Positive signal: MACRO constructive.",
            "Positive signal: ONCHAIN accumulation.",
        ],
        assumptions=[
            "No look-ahead information.",
            "Transaction costs will be included in validation.",
        ],
        metadata={
            "generator": "v5.0.9-hypothesis-engine",
            "research_trigger": "AGENT_SIGNAL",
            "source_priority": "HIGH",
        },
    )


def test_generate_returns_strategy_candidate() -> None:
    generator = StrategyGenerator()
    hypothesis = make_hypothesis()

    candidate = generator.generate(hypothesis)

    assert isinstance(candidate, StrategyCandidate)
    assert candidate.hypothesis_id == hypothesis.hypothesis_id
    assert candidate.asset == "BTC"
    assert candidate.horizon == "7D"


def test_positive_hypothesis_generates_long_direction() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        statement=(
            "The signal contains positive predictive information "
            "for BTC."
        )
    )

    candidate = generator.generate(hypothesis)

    assert candidate.parameters["direction"] == "LONG"
    assert "LONG" in candidate.name
    assert "LONG" in candidate.entry_rule


def test_negative_hypothesis_generates_short_direction() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        statement=(
            "The signal contains negative predictive information "
            "for BTC."
        ),
        regime="RISK-OFF",
    )

    candidate = generator.generate(hypothesis)

    assert candidate.parameters["direction"] == "SHORT"
    assert "SHORT" in candidate.name
    assert "SHORT" in candidate.entry_rule


def test_ambiguous_hypothesis_generates_neutral_direction() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        statement=(
            "BTC requires additional observation before "
            "a directional relationship can be established."
        ),
        regime="MIXED",
    )

    candidate = generator.generate(hypothesis)

    assert candidate.parameters["direction"] == "NEUTRAL"
    assert "NEUTRAL" in candidate.name
    assert "Do not initiate directional exposure" in candidate.entry_rule


def test_low_confidence_uses_small_position_size() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        confidence=0.30
    )

    candidate = generator.generate(hypothesis)

    assert candidate.position_size == 0.02


def test_medium_confidence_uses_default_position_size() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        confidence=0.60
    )

    candidate = generator.generate(hypothesis)

    assert candidate.position_size == 0.05


def test_high_confidence_uses_high_position_size() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        confidence=0.85
    )

    candidate = generator.generate(hypothesis)

    assert candidate.position_size == 0.08


def test_position_size_never_exceeds_configured_maximum() -> None:
    generator = StrategyGenerator(
        StrategyGeneratorConfig(
            default_position_size=0.05,
            low_confidence_position_size=0.02,
            high_confidence_position_size=0.08,
            maximum_position_size=0.08,
        )
    )

    hypothesis = make_hypothesis(
        confidence=0.95
    )

    candidate = generator.generate(hypothesis)

    assert candidate.position_size <= 0.08


def test_exit_rule_contains_risk_controls() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis()

    candidate = generator.generate(hypothesis)

    assert "7D" in candidate.exit_rule
    assert "5.0%" in candidate.exit_rule
    assert "10.0%" in candidate.exit_rule
    assert "hypothesis invalidation" in candidate.exit_rule


def test_stop_condition_preserves_risk_governor_authority() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis()

    candidate = generator.generate(hypothesis)

    assert "Risk Governor" in candidate.stop_condition
    assert "REDUCE" in candidate.stop_condition
    assert "VETO" in candidate.stop_condition
    assert "SHUTDOWN" in candidate.stop_condition


def test_candidate_is_research_only() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis()

    candidate = generator.generate(hypothesis)

    assert candidate.parameters["research_only"] is True
    assert candidate.parameters["real_money_execution"] is False
    assert candidate.parameters["requires_validation"] is True
    assert candidate.parameters["requires_risk_gate"] is True
    assert candidate.parameters["shadow_before_trust"] is True


def test_strategy_parameters_are_auditable() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis(
        confidence=0.72,
        regime="RISK-ON",
    )

    candidate = generator.generate(hypothesis)

    assert candidate.parameters["hypothesis_confidence"] == 0.72
    assert candidate.parameters["regime"] == "RISK-ON"
    assert candidate.parameters["horizon"] == "7D"

    assert candidate.parameters["stop_loss"] == 0.05
    assert candidate.parameters["take_profit"] == 0.10


def test_metadata_preserves_hypothesis_information() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis()

    candidate = generator.generate(hypothesis)

    assert (
        candidate.metadata["generator"]
        == "v5.0.9-strategy-generator"
    )

    assert (
        candidate.metadata["source_hypothesis_confidence"]
        == hypothesis.confidence
    )

    assert (
        candidate.metadata["source_hypothesis_statement"]
        == hypothesis.statement
    )

    assert (
        candidate.metadata["source_hypothesis_metadata"]
        == hypothesis.metadata
    )


def test_invalid_input_raises_type_error() -> None:
    generator = StrategyGenerator()

    try:
        generator.generate("not-a-hypothesis")
    except TypeError as error:
        assert "Hypothesis" in str(error)
    else:
        raise AssertionError(
            "StrategyGenerator must reject invalid input."
        )


def test_engine_is_callable_for_research_loop_dependency() -> None:
    generator = StrategyGenerator()

    hypothesis = make_hypothesis()

    candidate = generator(hypothesis)

    assert isinstance(candidate, StrategyCandidate)
    assert candidate.hypothesis_id == hypothesis.hypothesis_id