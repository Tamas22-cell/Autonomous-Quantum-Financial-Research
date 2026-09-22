from __future__ import annotations

import pytest

from v7_data_fabric.portfolio_risk_governance import (
    PortfolioResearchSignal,
    PortfolioRiskGovernance,
    PortfolioRiskGovernanceError,
    RecommendationStatus,
    RiskLimits,
    RiskSeverity,
    SignalAction,
)


def _signal(
    signal_id: str,
    asset_id: str,
    confidence: float = 0.8,
) -> PortfolioResearchSignal:
    return PortfolioResearchSignal(
        signal_id=signal_id,
        asset_id=asset_id,
        action=SignalAction.BUY,
        confidence=confidence,
        expected_return=0.10,
        expected_volatility=0.20,
        rationale="Validated research signal.",
        evidence_fingerprint=f"evidence-{signal_id}",
    )


def test_version_and_safety_flags():
    governance = PortfolioRiskGovernance()

    assert governance.VERSION == "v7.0.8"
    assert governance.RESEARCH_ONLY is True
    assert governance.REAL_MONEY_EXECUTION is False
    assert governance.AUTOMATIC_CAPITAL_ALLOCATION is False


def test_signal_registration_and_translation():
    governance = PortfolioRiskGovernance()

    signals = governance.translate_signals(
        (
            _signal("s-001", "BTC"),
            _signal("s-002", "SPY"),
        )
    )

    assert len(signals) == 2
    assert governance.signal_count == 2


def test_risk_assessment_detects_concentration():
    governance = PortfolioRiskGovernance()

    assessment = governance.assess_risk(
        weights={"BTC": 0.40},
        cash_weight=0.60,
        current_drawdown=0.0,
    )

    assert assessment.severity == RiskSeverity.HIGH
    assert assessment.concentration_breach is True
    assert assessment.is_clear is False
    assert assessment.violations


def test_risk_assessment_detects_drawdown_and_cash_breach():
    governance = PortfolioRiskGovernance()

    assessment = governance.assess_risk(
        weights={"BTC": 0.20},
        cash_weight=0.01,
        current_drawdown=0.25,
    )

    assert assessment.drawdown_breach is True
    assert assessment.cash_breach is True
    assert assessment.severity == RiskSeverity.HIGH
    assert len(assessment.violations) >= 2


def test_recommendation_requires_human_approval():
    governance = PortfolioRiskGovernance()

    recommendation = governance.build_recommendation(
        recommendation_id="r-001",
        signals=(
            _signal("s-001", "BTC"),
        ),
    )

    assert recommendation.research_only is True
    assert recommendation.real_money_execution is False
    assert recommendation.human_approval_required is True
    assert recommendation.status == (
        RecommendationStatus.REQUIRES_APPROVAL
    )
    assert abs(
        sum(recommendation.weights.values())
        + recommendation.cash_weight
        - 1.0
    ) < 1e-6


def test_balanced_recommendation_can_be_approved_for_research():
    governance = PortfolioRiskGovernance()

    recommendation = governance.build_recommendation(
        recommendation_id="r-002",
        signals=(
            _signal("s-001", "BTC"),
            _signal("s-002", "SPY"),
            _signal("s-003", "GLD"),
            _signal("s-004", "TLT"),
        ),
    )

    assert recommendation.risk_assessment.is_clear is True

    approved = governance.approve_for_research(
        "r-002",
        approved=True,
    )

    assert approved.status == (
        RecommendationStatus.APPROVED_FOR_RESEARCH
    )
    assert approved.research_only is True
    assert approved.real_money_execution is False


def test_approval_is_blocked_when_risk_violation_exists():
    governance = PortfolioRiskGovernance(
        limits=RiskLimits(
            max_position_weight=0.20,
            max_asset_concentration=0.20,
            minimum_cash_weight=0.05,
            require_human_approval=True,
        )
    )

    recommendation = governance.build_recommendation(
        recommendation_id="r-003",
        signals=(
            _signal("s-001", "BTC"),
        ),
    )

    assert recommendation.risk_assessment.is_clear is False

    with pytest.raises(
        PortfolioRiskGovernanceError
    ):
        governance.approve_for_research(
            "r-003",
            approved=True,
        )


def test_summary_contains_governance_state():
    governance = PortfolioRiskGovernance()

    governance.register_signal(
        _signal("s-001", "BTC")
    )

    summary = governance.summary()

    assert summary["version"] == "v7.0.8"
    assert summary["signal_count"] == 1
    assert summary["recommendation_count"] == 0
    assert summary["research_only"] is True
    assert summary["real_money_execution"] is False
    assert summary["automatic_capital_allocation"] is False
    assert summary["human_approval_required"] is True


def test_duplicate_signal_is_rejected():
    governance = PortfolioRiskGovernance()

    governance.register_signal(
        _signal("s-001", "BTC")
    )

    with pytest.raises(
        PortfolioRiskGovernanceError
    ):
        governance.register_signal(
            _signal("s-001", "BTC")
        )