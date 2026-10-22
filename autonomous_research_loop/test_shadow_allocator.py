"""
Tests for v5.0.9 Shadow Allocation Engine.

Covered:
- APPROVE creates shadow allocation
- REDUCE uses Risk Governor reduced size
- global maximum shadow weight
- simulated notional calculation
- VETO blocking
- SHUTDOWN blocking
- non-promoted validation blocking
- strategy identity protection
- invalid object protection
- entry-price validation
- required entry price
- minimum shadow weight
- research-only / no-real-money metadata
- callable interface
"""

import pytest

from autonomous_research_loop.models import (
    RiskAssessment,
    RiskDecision,
    StrategyCandidate,
    ValidationDecision,
    ValidationResult,
)
from autonomous_research_loop.shadow_allocator import (
    ShadowAllocator,
    ShadowAllocatorConfig,
)


def make_candidate(
    position_size: float = 0.08,
) -> StrategyCandidate:
    return StrategyCandidate(
        hypothesis_id="hypothesis-shadow-test",
        name="BTC LONG 7D shadow candidate",
        entry_rule="Enter LONG research exposure.",
        exit_rule="Exit after 7D or invalidation.",
        asset="BTC",
        horizon="7D",
        position_size=position_size,
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


def make_risk(
    candidate: StrategyCandidate,
    decision: RiskDecision = RiskDecision.APPROVE,
    approved_position_size: float = 0.08,
) -> RiskAssessment:
    return RiskAssessment(
        strategy_id=candidate.strategy_id,
        decision=decision,
        risk_score=0.20,
        approved_position_size=approved_position_size,
        reasons=["Risk Governor completed."],
        metadata={
            "real_money_authorized": False,
            "shadow_only": decision
            in {
                RiskDecision.APPROVE,
                RiskDecision.REDUCE,
            },
        },
    )


def test_approved_candidate_creates_shadow_allocation() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    result = allocator.allocate(
        candidate,
        validation,
        risk,
        entry_price=60_000.0,
    )

    assert result.strategy_id == candidate.strategy_id
    assert result.asset == "BTC"
    assert result.weight == pytest.approx(0.08)
    assert result.entry_price == pytest.approx(60_000.0)


def test_reduce_uses_risk_governor_position_size() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate(position_size=0.08)
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        decision=RiskDecision.REDUCE,
        approved_position_size=0.04,
    )

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert result.weight == pytest.approx(0.04)


def test_risk_governor_cannot_increase_candidate_size() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate(position_size=0.05)
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        approved_position_size=0.08,
    )

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert result.weight == pytest.approx(0.05)


def test_global_maximum_shadow_weight_is_enforced() -> None:
    allocator = ShadowAllocator(
        ShadowAllocatorConfig(
            maximum_shadow_weight=0.10,
        )
    )

    candidate = make_candidate(position_size=0.20)
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        approved_position_size=0.20,
    )

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert result.weight == pytest.approx(0.10)


def test_shadow_notional_is_calculated() -> None:
    allocator = ShadowAllocator(
        ShadowAllocatorConfig(
            shadow_capital=100_000.0,
        )
    )

    candidate = make_candidate(position_size=0.08)
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert result.notional_value == pytest.approx(8_000.0)


def test_veto_blocks_shadow_allocation() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        decision=RiskDecision.VETO,
        approved_position_size=0.0,
    )

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_shutdown_blocks_shadow_allocation() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        decision=RiskDecision.SHUTDOWN,
        approved_position_size=0.0,
    )

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_non_promoted_validation_is_blocked() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()

    validation = make_validation(
        candidate,
        ValidationDecision.REVIEW,
    )

    risk = make_risk(candidate)

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_validation_strategy_must_match_candidate() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()

    validation = ValidationResult(
        strategy_id="wrong-strategy-id",
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

    risk = make_risk(candidate)

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_risk_strategy_must_match_candidate() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk = RiskAssessment(
        strategy_id="wrong-strategy-id",
        decision=RiskDecision.APPROVE,
        risk_score=0.20,
        approved_position_size=0.08,
        reasons=[],
        metadata={},
    )

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_invalid_candidate_is_rejected() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    with pytest.raises(TypeError):
        allocator.allocate(
            "invalid-candidate",
            validation,
            risk,
        )


def test_invalid_validation_is_rejected() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    risk = make_risk(candidate)

    with pytest.raises(TypeError):
        allocator.allocate(
            candidate,
            "invalid-validation",
            risk,
        )


def test_invalid_risk_assessment_is_rejected() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)

    with pytest.raises(TypeError):
        allocator.allocate(
            candidate,
            validation,
            "invalid-risk",
        )


def test_invalid_entry_price_is_rejected() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
            entry_price=-100.0,
        )


def test_non_numeric_entry_price_is_rejected() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
            entry_price="invalid",
        )


def test_required_entry_price_is_enforced() -> None:
    allocator = ShadowAllocator(
        ShadowAllocatorConfig(
            require_entry_price=True,
        )
    )

    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_minimum_shadow_weight_is_enforced() -> None:
    allocator = ShadowAllocator(
        ShadowAllocatorConfig(
            minimum_shadow_weight=0.01,
        )
    )

    candidate = make_candidate(
        position_size=0.005,
    )
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        approved_position_size=0.005,
    )

    with pytest.raises(ValueError):
        allocator.allocate(
            candidate,
            validation,
            risk,
        )


def test_metadata_marks_allocation_as_shadow_only() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert result.metadata["mode"] == "SHADOW"
    assert result.metadata["research_only"] is True
    assert result.metadata["paper_allocation"] is True
    assert result.metadata["real_money_execution"] is False
    assert result.metadata["broker_order_allowed"] is False
    assert result.metadata["eligible_for_real_execution"] is False
    assert result.metadata["requires_forward_observation"] is True


def test_metadata_preserves_gate_decisions() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)

    risk = make_risk(
        candidate,
        decision=RiskDecision.REDUCE,
        approved_position_size=0.04,
    )

    result = allocator.allocate(
        candidate,
        validation,
        risk,
    )

    assert (
        result.metadata["validation_decision"]
        == ValidationDecision.PROMOTE.value
    )

    assert (
        result.metadata["risk_decision"]
        == RiskDecision.REDUCE.value
    )

    assert (
        result.metadata["risk_approved_position_size"]
        == pytest.approx(0.04)
    )


def test_callable_interface_creates_allocation() -> None:
    allocator = ShadowAllocator()
    candidate = make_candidate()
    validation = make_validation(candidate)
    risk = make_risk(candidate)

    result = allocator(
        candidate,
        validation,
        risk,
        60_000.0,
    )

    assert result.strategy_id == candidate.strategy_id
    assert result.weight == pytest.approx(0.08)
    assert result.notional_value == pytest.approx(8_000.0)


def test_config_rejects_invalid_shadow_capital() -> None:
    with pytest.raises(ValueError):
        ShadowAllocatorConfig(
            shadow_capital=0.0,
        )


def test_config_rejects_invalid_maximum_weight() -> None:
    with pytest.raises(ValueError):
        ShadowAllocatorConfig(
            maximum_shadow_weight=1.50,
        )


def test_config_rejects_minimum_above_maximum() -> None:
    with pytest.raises(ValueError):
        ShadowAllocatorConfig(
            minimum_shadow_weight=0.20,
            maximum_shadow_weight=0.10,
        )