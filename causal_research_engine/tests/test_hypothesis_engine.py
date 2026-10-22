"""
Tests for Causal Hypothesis Engine
Version: v5.0.5
"""

import pytest

from causal_research_engine.hypothesis_engine import (
    CausalHypothesisEngine,
    HypothesisValidationError,
)
from causal_research_engine.models import (
    CausalHypothesis,
    HypothesisStatus,
)


def test_create_hypothesis():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H001",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Funding pressure increases liquidation risk.",
    )

    assert isinstance(hypothesis, CausalHypothesis)
    assert hypothesis.hypothesis_id == "H001"
    assert hypothesis.cause == "funding_rate"
    assert hypothesis.effect == "btc_return"
    assert hypothesis.expected_direction == "negative"
    assert hypothesis.mechanism == (
        "Funding pressure increases liquidation risk."
    )
    assert hypothesis.status == HypothesisStatus.PROPOSED


def test_create_hypothesis_with_confounders():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H002",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
        confounders=["market_regime", "volatility"],
        controls=["volume"],
    )

    assert hypothesis.confounders == [
        "market_regime",
        "volatility",
    ]

    assert hypothesis.controls == ["volume"]


def test_create_hypothesis_normalizes_text():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id=" H003 ",
        cause=" funding_rate ",
        effect=" btc_return ",
        expected_direction=" NEGATIVE ",
        mechanism=" Liquidation channel. ",
    )

    assert hypothesis.hypothesis_id == "H003"
    assert hypothesis.cause == "funding_rate"
    assert hypothesis.effect == "btc_return"
    assert hypothesis.expected_direction == "negative"
    assert hypothesis.mechanism == "Liquidation channel."


def test_direction_alias():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H004",
        cause="oil_price",
        effect="energy_returns",
        expected_direction="increase",
        mechanism="Revenue channel.",
    )

    assert hypothesis.expected_direction == "positive"


def test_duplicate_variables_are_removed():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H005",
        cause="oil_price",
        effect="energy_returns",
        expected_direction="positive",
        mechanism="Revenue channel.",
        confounders=[
            "inflation",
            "inflation",
            "market_regime",
        ],
        controls=[
            "volume",
            "volume",
        ],
    )

    assert hypothesis.confounders == [
        "inflation",
        "market_regime",
    ]

    assert hypothesis.controls == ["volume"]


def test_same_cause_and_effect_rejected():
    engine = CausalHypothesisEngine()

    with pytest.raises(HypothesisValidationError):
        engine.create_hypothesis(
            hypothesis_id="H006",
            cause="btc_return",
            effect="btc_return",
            expected_direction="positive",
            mechanism="Invalid self-causation.",
        )


def test_invalid_direction_rejected():
    engine = CausalHypothesisEngine()

    with pytest.raises(HypothesisValidationError):
        engine.create_hypothesis(
            hypothesis_id="H007",
            cause="funding_rate",
            effect="btc_return",
            expected_direction="sideways",
            mechanism="Liquidation channel.",
        )


def test_confounder_control_overlap_rejected():
    engine = CausalHypothesisEngine()

    with pytest.raises(HypothesisValidationError):
        engine.create_hypothesis(
            hypothesis_id="H008",
            cause="interest_rates",
            effect="equity_returns",
            expected_direction="negative",
            mechanism="Discount-rate channel.",
            confounders=["inflation"],
            controls=["inflation"],
        )


def test_validate_valid_hypothesis():
    engine = CausalHypothesisEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="H009",
        cause="interest_rates",
        effect="equity_returns",
        expected_direction="negative",
        mechanism="Higher rates reduce valuations.",
    )

    issues = engine.validate_hypothesis(
        hypothesis,
        require_mechanism=True,
        require_direction=True,
    )

    assert issues == []


def test_validate_missing_direction():
    engine = CausalHypothesisEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="H010",
        cause="interest_rates",
        effect="equity_returns",
        mechanism="Discount-rate channel.",
    )

    issues = engine.validate_hypothesis(
        hypothesis,
        require_direction=True,
    )

    assert issues == [
        "Expected causal direction is missing."
    ]


def test_validate_missing_mechanism():
    engine = CausalHypothesisEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="H011",
        cause="interest_rates",
        effect="equity_returns",
        expected_direction="negative",
    )

    issues = engine.validate_hypothesis(
        hypothesis,
        require_mechanism=True,
    )

    assert issues == [
        "Proposed causal mechanism is missing."
    ]


def test_validate_invalid_object():
    engine = CausalHypothesisEngine()

    issues = engine.validate_hypothesis(
        "invalid"
    )

    assert issues == [
        "Object is not a CausalHypothesis."
    ]


def test_research_ready():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H012",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    assert engine.is_research_ready(hypothesis) is True


def test_not_research_ready_without_mechanism():
    engine = CausalHypothesisEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="H013",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
    )

    assert engine.is_research_ready(hypothesis) is False


def test_start_testing():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H014",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    result = engine.start_testing(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.TESTING


def test_start_testing_rejects_invalid_hypothesis():
    engine = CausalHypothesisEngine()

    hypothesis = CausalHypothesis(
        hypothesis_id="H015",
        cause="funding_rate",
        effect="btc_return",
    )

    with pytest.raises(HypothesisValidationError):
        engine.start_testing(hypothesis)


def test_mark_supported():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H016",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    result = engine.mark_supported(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.SUPPORTED


def test_mark_weak():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H017",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    result = engine.mark_weak(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.WEAK


def test_mark_falsified():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H018",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    result = engine.mark_falsified(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.FALSIFIED


def test_mark_inconclusive():
    engine = CausalHypothesisEngine()

    hypothesis = engine.create_hypothesis(
        hypothesis_id="H019",
        cause="funding_rate",
        effect="btc_return",
        expected_direction="negative",
        mechanism="Liquidation pressure.",
    )

    result = engine.mark_inconclusive(hypothesis)

    assert result is hypothesis
    assert hypothesis.status == HypothesisStatus.INCONCLUSIVE