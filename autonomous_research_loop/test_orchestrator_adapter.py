"""
Tests for v5.0.9 Orchestrator Research Adapter.

Covered:
- normal agent signal
- uncertainty trigger
- performance-gap trigger
- regime-change trigger
- risk-event trigger
- metadata preservation
- direct Orchestrator.run() integration
"""

from autonomous_research_loop.models import (
    ResearchTrigger,
)
from autonomous_research_loop.orchestrator_adapter import (
    OrchestratorResearchAdapter,
)


def make_snapshot() -> dict:
    """
    Create a representative snapshot matching the existing
    Orchestrator.run() output contract.
    """

    return {
        "agents": [
            {
                "agent": "macro",
                "signal": "bullish",
                "score": 0.70,
                "confidence": 0.80,
                "summary": "Macro conditions constructive.",
                "risks": [],
                "evidence": [],
            },
            {
                "agent": "risk",
                "signal": "neutral",
                "score": 0.50,
                "confidence": 0.80,
                "summary": "No major risk override.",
                "risks": [],
                "evidence": [],
            },
        ],
        "market_score": 0.62,
        "decision": "HOLD",
        "market_regime": "MIXED",
        "market_interpretation": (
            "Mixed market environment with positive bias."
        ),
        "bullish_agents": 5,
        "neutral_agents": 3,
        "bearish_agents": 1,
        "bullish_pct": 55.56,
        "neutral_pct": 33.33,
        "bearish_pct": 11.11,
        "consensus_strength": 0.75,
        "average_confidence": 0.80,
        "signal_conflict": 0.10,
        "risk_override": False,
        "strongest_bullish_agent": "macro",
        "strongest_bearish_agent": "risk",
        "top_positive_signals": [
            "MACRO: Macro conditions constructive."
        ],
        "top_risks": [],
    }


def test_adapter_creates_research_task() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    task = adapter.create_research_task(
        snapshot,
        asset="BTC",
        horizon="7D",
    )

    assert task.asset == "BTC"
    assert task.horizon == "7D"
    assert task.regime == "MIXED"

    assert task.metadata["source"] == "orchestrator"
    assert task.metadata["decision"] == "HOLD"
    assert task.metadata["agent_count"] == 2

    assert 0.0 <= task.uncertainty_score <= 1.0
    assert 0.0 <= task.performance_gap_score <= 1.0


def test_high_uncertainty_creates_uncertainty_trigger() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    snapshot["average_confidence"] = 0.20
    snapshot["consensus_strength"] = 0.20
    snapshot["signal_conflict"] = 0.80

    task = adapter.create_research_task(
        snapshot,
        asset="BTC",
    )

    assert task.trigger == ResearchTrigger.UNCERTAINTY


def test_regime_change_has_priority_over_uncertainty() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    snapshot["market_regime"] = "RISK-ON"
    snapshot["average_confidence"] = 0.20
    snapshot["consensus_strength"] = 0.20
    snapshot["signal_conflict"] = 0.80

    task = adapter.create_research_task(
        snapshot,
        previous_regime="RISK-OFF",
    )

    assert task.trigger == ResearchTrigger.REGIME_CHANGE


def test_risk_override_has_highest_trigger_priority() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    snapshot["risk_override"] = True
    snapshot["market_regime"] = "RISK-ON"

    task = adapter.create_research_task(
        snapshot,
        previous_regime="RISK-OFF",
    )

    assert task.trigger == ResearchTrigger.RISK_EVENT


def test_risk_off_with_top_risks_creates_risk_event() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    snapshot["market_regime"] = "RISK-OFF"
    snapshot["bearish_pct"] = 66.67
    snapshot["top_risks"] = [
        "RISK: Elevated systemic risk."
    ]

    task = adapter.create_research_task(
        snapshot,
    )

    assert task.trigger == ResearchTrigger.RISK_EVENT


def test_performance_gap_trigger() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    snapshot["average_confidence"] = 0.95
    snapshot["consensus_strength"] = 0.95
    snapshot["signal_conflict"] = 0.05

    snapshot["bullish_pct"] = 45.0
    snapshot["neutral_pct"] = 10.0
    snapshot["bearish_pct"] = 45.0

    task = adapter.create_research_task(
        snapshot,
    )

    assert task.trigger == ResearchTrigger.PERFORMANCE_GAP


def test_metadata_preserves_orchestrator_information() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    task = adapter.create_research_task(
        snapshot,
        asset="SPY",
        horizon="30D",
    )

    assert task.metadata["market_score"] == 0.62
    assert task.metadata["average_confidence"] == 0.80
    assert task.metadata["consensus_strength"] == 0.75
    assert task.metadata["signal_conflict"] == 0.10

    assert task.metadata["strongest_bullish_agent"] == "macro"
    assert task.metadata["strongest_bearish_agent"] == "risk"

    assert task.metadata["top_positive_signals"] == [
        "MACRO: Macro conditions constructive."
    ]


def test_run_orchestrator_uses_existing_run_contract() -> None:
    adapter = OrchestratorResearchAdapter()

    snapshot = make_snapshot()

    class FakeOrchestrator:
        def run(self) -> dict:
            return snapshot

    orchestrator = FakeOrchestrator()

    task = adapter.run_orchestrator(
        orchestrator,
        asset="BTC",
        horizon="7D",
    )

    assert task.asset == "BTC"
    assert task.horizon == "7D"
    assert task.metadata["source"] == "orchestrator"
    assert task.metadata["market_score"] == 0.62