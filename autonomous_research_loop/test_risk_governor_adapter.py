"""
Tests for v5.0.9 Independent Risk Governor Adapter.

Covered:
- APPROVE
- REDUCE
- VETO
- SHUTDOWN
- bearish veto
- low-confidence reduction
- position-size reduction
- zero exposure after veto/shutdown
- risk-score conversion
- validation gate enforcement
- strategy/validation identity protection
- extreme drawdown
- extreme volatility
- extreme correlation
- existing RiskAgent analyze() contract
- audit/safety metadata
"""

import pytest

from autonomous_research_loop.models import (
    RiskAssessment,
    RiskDecision,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)
from autonomous_research_loop.risk_governor_adapter import (
    RiskGovernorAdapter,
)


def make_candidate() -> StrategyCandidate:
    return StrategyCandidate(
        hypothesis_id="hypothesis-risk-test",
        name="BTC LONG 7D research candidate",
        entry_rule="Enter LONG research exposure.",
        exit_rule="Exit after 7D or invalidation.",
        asset="BTC",
        horizon="7D",
        position_size=0.08,
        stop_condition="Stop on Risk Governor veto.",
        parameters={
            "direction": "LONG",
            "research_only": True,
            "real_money_execution": False,
            "requires_risk_gate": True,
        },
    )


def make_validation(
    candidate: StrategyCandidate,
    decision: ValidationDecision = ValidationDecision.PROMOTE,
) -> ValidationResult:
    return ValidationResult(
        strategy_id=candidate.strategy_id,
        decision=decision,
        score=0.80,
        out_of_sample_score=0.82,
        robustness_score=0.78,
        causal_score=0.75,
        sharpe_ratio=1.40,
        sortino_ratio=1.60,
        max_drawdown=0.10,
        reasons=["Validation completed."],
    )


def make_risk_output(
    score: float = 0.70,
    confidence: float = 0.80,
    signal: str = "bullish",
    evidence=None,
    risks=None,
) -> dict:
    return {
        "agent": "risk",
        "signal": signal,
        "score": score,
        "confidence": confidence,
        "summary": "Portfolio risk assessment.",
        "risks": risks or [],
        "evidence": evidence or {
            "avg_volatility": 0.20,
            "avg_correlation": 0.30,
            "max_drawdown": -0.05,
        },
    }


def test_safe_candidate_is_approved() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    result = adapter.evaluate(
        candidate,
        validation,
        make_risk_output(),
    )

    assert isinstance(result, RiskAssessment)
    assert result.decision == RiskDecision.APPROVE
    assert result.approved_position_size == 0.08


def test_medium_risk_reduces_position() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.50,
        confidence=0.80,
        signal="neutral",
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.REDUCE
    assert result.approved_position_size == 0.04


def test_low_confidence_reduces_position() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.70,
        confidence=0.30,
        signal="neutral",
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.REDUCE
    assert result.approved_position_size == 0.04


def test_low_score_vetoes_candidate() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.15,
        confidence=0.80,
        signal="neutral",
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.VETO
    assert result.approved_position_size == 0.0


def test_bearish_signal_vetoes_candidate() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.50,
        confidence=0.80,
        signal="bearish",
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.VETO
    assert result.approved_position_size == 0.0


def test_extreme_drawdown_triggers_shutdown() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.70,
        evidence={
            "avg_volatility": 0.20,
            "avg_correlation": 0.30,
            "max_drawdown": -0.35,
        },
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.SHUTDOWN
    assert result.approved_position_size == 0.0


def test_extreme_volatility_triggers_shutdown() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.70,
        evidence={
            "avg_volatility": 0.80,
            "avg_correlation": 0.30,
            "max_drawdown": -0.05,
        },
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.SHUTDOWN


def test_extreme_correlation_triggers_shutdown() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk_output = make_risk_output(
        score=0.70,
        evidence={
            "avg_volatility": 0.20,
            "avg_correlation": 0.95,
            "max_drawdown": -0.05,
        },
    )

    result = adapter.evaluate(
        candidate,
        validation,
        risk_output,
    )

    assert result.decision == RiskDecision.SHUTDOWN


def test_risk_score_inverts_existing_risk_agent_score() -> None:
    adapter = RiskGovernorAdapter()

    assert adapter.calculate_risk_score(0.80) == pytest.approx(0.20)
    assert adapter.calculate_risk_score(0.20) == pytest.approx(0.80)


def test_non_promoted_candidate_cannot_enter_risk_governor() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()

    validation = make_validation(
        candidate,
        ValidationDecision.REVIEW,
    )

    with pytest.raises(ValueError):
        adapter.evaluate(
            candidate,
            validation,
            make_risk_output(),
        )


def test_validation_must_belong_to_candidate() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()

    validation = ValidationResult(
        strategy_id="different-strategy-id",
        decision=ValidationDecision.PROMOTE,
        score=0.80,
        out_of_sample_score=0.80,
        robustness_score=0.80,
        causal_score=0.80,
        sharpe_ratio=1.20,
        sortino_ratio=1.40,
        max_drawdown=0.10,
        reasons=[],
    )

    with pytest.raises(ValueError):
        adapter.evaluate(
            candidate,
            validation,
            make_risk_output(),
        )


def test_invalid_candidate_is_rejected() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    with pytest.raises(TypeError):
        adapter.evaluate(
            "invalid-candidate",
            validation,
            make_risk_output(),
        )


def test_invalid_risk_output_is_rejected() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    with pytest.raises(TypeError):
        adapter.evaluate(
            candidate,
            validation,
            "invalid-risk-output",
        )


def test_metadata_preserves_risk_agent_information() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    result = adapter.evaluate(
        candidate,
        validation,
        make_risk_output(
            score=0.70,
            confidence=0.80,
            signal="bullish",
        ),
    )

    assert (
        result.metadata["governor"]
        == "v5.0.9-risk-governor-adapter"
    )
    assert result.metadata["source_agent"] == "risk"
    assert result.metadata["risk_agent_score"] == 0.70
    assert result.metadata["risk_agent_confidence"] == 0.80
    assert result.metadata["risk_agent_signal"] == "bullish"
    assert result.metadata["real_money_authorized"] is False
    assert result.metadata["shadow_only"] is True


def test_veto_is_not_shadow_eligible() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    result = adapter.evaluate(
        candidate,
        validation,
        make_risk_output(
            score=0.15,
            signal="neutral",
        ),
    )

    assert result.decision == RiskDecision.VETO
    assert result.metadata["shadow_only"] is False


def test_existing_risk_agent_contract_can_be_called() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    class FakeRiskAgent:
        def analyze(self):
            return make_risk_output(
                score=0.70,
                confidence=0.80,
                signal="bullish",
            )

    result = adapter.run_risk_agent(
        candidate,
        validation,
        FakeRiskAgent(),
    )

    assert isinstance(result, RiskAssessment)
    assert result.decision == RiskDecision.APPROVE


def test_invalid_risk_agent_contract_is_rejected() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    class InvalidRiskAgent:
        pass

    with pytest.raises(TypeError):
        adapter.run_risk_agent(
            candidate,
            validation,
            InvalidRiskAgent(),
        )


def test_risk_agent_must_return_mapping() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    class InvalidRiskAgent:
        def analyze(self):
            return "invalid-output"

    with pytest.raises(TypeError):
        adapter.run_risk_agent(
            candidate,
            validation,
            InvalidRiskAgent(),
        )


def test_shutdown_reason_is_auditable() -> None:
    adapter = RiskGovernorAdapter()
    candidate = make_candidate()
    validation = make_validation(candidate)

    result = adapter.evaluate(
        candidate,
        validation,
        make_risk_output(
            evidence={
                "avg_volatility": 0.80,
                "avg_correlation": 0.95,
                "max_drawdown": -0.35,
            }
        ),
    )

    reasons = " ".join(result.reasons)

    assert result.decision == RiskDecision.SHUTDOWN
    assert "Extreme volatility" in reasons
    assert "Extreme cross-asset correlation" in reasons
    assert "Severe drawdown" in reasons