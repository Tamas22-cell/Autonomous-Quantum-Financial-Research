from research_control_center.control_center import ResearchControlCenter
from research_control_center.models import ResearchControlCenterSnapshot


def build_complete_research_flow():
    """
    Build one complete v5.0.10 research-observability flow.

    This intentionally remains research-only and
    never authorizes real-money execution.
    """

    center = ResearchControlCenter()

    # ========================================================
    # 1. QUANTUM VS CLASSICAL EXPERIMENT
    # ========================================================

    quantum_experiment = center.quantum_tracker.create(
        problem_name="BTC Portfolio Optimization",
        classical_method="RiskParity",
        quantum_method="QAOA",
        asset_universe=[
            "BTC",
            "ETH",
            "SOL",
        ],
        classical_score=0.78,
        quantum_score=0.84,
        classical_runtime_seconds=2.0,
        quantum_runtime_seconds=5.0,
        classical_objective=0.90,
        quantum_objective=0.86,
        approximation_ratio=0.95,
        qubits=6,
        circuit_depth=24,
        repetitions=5,
        metadata={
            "research_cycle": "e2e_001",
        },
    )

    # ========================================================
    # 2. DECISION LINEAGE
    # ========================================================

    lineage = center.decision_lineage.create(
        root_id="research_cycle_001",
        metadata={
            "asset": "BTC",
            "regime": "RISK_ON",
            "horizon": "7D",
        },
    )

    raw_data = center.decision_lineage.add_raw_data(
        lineage.lineage_id,
        "raw_market_snapshot_001",
        summary="BTC market snapshot",
        evidence={
            "asset": "BTC",
            "price": 100000.0,
            "regime": "RISK_ON",
        },
    )

    agent_evidence = (
        center.decision_lineage
        .add_agent_evidence(
            lineage.lineage_id,
            "agent_consensus_001",
            raw_data.entity_id,
            summary=(
                "Multi-agent bullish evidence"
            ),
            evidence={
                "bullish_pct": 70.0,
                "bearish_pct": 15.0,
                "neutral_pct": 15.0,
                "confidence": 0.82,
            },
        )
    )

    hypothesis = (
        center.decision_lineage
        .add_hypothesis(
            lineage.lineage_id,
            "hypothesis_001",
            agent_evidence.entity_id,
            summary=(
                "BTC positive forward-return "
                "hypothesis"
            ),
            evidence={
                "direction": "BULLISH",
                "horizon": "7D",
            },
        )
    )

    strategy = (
        center.decision_lineage
        .add_strategy(
            lineage.lineage_id,
            "strategy_001",
            hypothesis.entity_id,
            summary=(
                "Research-only BTC allocation "
                "strategy"
            ),
            evidence={
                "asset": "BTC",
                "direction": "LONG",
            },
        )
    )

    validation = (
        center.decision_lineage
        .add_validation(
            lineage.lineage_id,
            "validation_001",
            strategy.entity_id,
            summary="Strategy validation",
            evidence={
                "decision": "PROMOTE",
                "score": 0.84,
                "out_of_sample_score": 0.79,
                "robustness_score": 0.82,
            },
        )
    )

    risk = center.decision_lineage.add_risk(
        lineage.lineage_id,
        "risk_001",
        validation.entity_id,
        summary="Risk governor decision",
        evidence={
            "decision": "APPROVE",
            "risk_score": 0.20,
            "real_money_execution": False,
        },
    )

    shadow_allocation = (
        center.decision_lineage
        .add_shadow_allocation(
            lineage.lineage_id,
            "shadow_allocation_001",
            risk.entity_id,
            summary=(
                "Paper-only shadow allocation"
            ),
            evidence={
                "asset": "BTC",
                "weight": 0.10,
                "notional_value": 10000.0,
                "research_only": True,
                "real_money_execution": False,
            },
        )
    )

    outcome = (
        center.decision_lineage
        .add_outcome(
            lineage.lineage_id,
            "outcome_001",
            shadow_allocation.entity_id,
            summary="Forward outcome",
            evidence={
                "return_pct": 0.04,
                "pnl": 400.0,
                "hit": True,
            },
        )
    )

    return {
        "center": center,
        "quantum_experiment": (
            quantum_experiment
        ),
        "lineage": lineage,
        "raw_data": raw_data,
        "agent_evidence": agent_evidence,
        "hypothesis": hypothesis,
        "strategy": strategy,
        "validation": validation,
        "risk": risk,
        "shadow_allocation": (
            shadow_allocation
        ),
        "outcome": outcome,
    }


# ============================================================
# CONTROL CENTER
# ============================================================


def test_e2e_control_center_is_healthy():
    flow = build_complete_research_flow()

    center = flow["center"]

    health = center.health()

    assert health["healthy"] is True
    assert health["healthy_components"] == 7
    assert health["total_components"] == 7


def test_e2e_all_components_available():
    flow = build_complete_research_flow()

    center = flow["center"]

    assert len(center.components()) == 7

    assert (
        center.component_names()
        == (
            "experiment_registry",
            "leaderboard",
            "alpha_library",
            "backtest_diagnostics",
            "shadow_fund",
            "quantum_tracker",
            "decision_lineage",
        )
    )


# ============================================================
# QUANTUM TRACKING
# ============================================================


def test_e2e_quantum_experiment_registered():
    flow = build_complete_research_flow()

    center = flow["center"]

    experiments = (
        center.quantum_tracker.list_all()
    )

    assert len(experiments) == 1

    assert (
        experiments[0].quantum_method
        == "QAOA"
    )


def test_e2e_quantum_competitiveness():
    flow = build_complete_research_flow()

    center = flow["center"]

    experiment = flow[
        "quantum_experiment"
    ]

    report = (
        center.quantum_tracker
        .competitiveness_report(
            experiment.experiment_id
        )
    )

    assert report["competitive"] is True
    assert report["score_requirement"] is True

    assert (
        report["approximation_requirement"]
        is True
    )

    assert (
        report["runtime_requirement"]
        is True
    )


def test_e2e_quantum_score_improvement():
    flow = build_complete_research_flow()

    center = flow["center"]

    experiment = flow[
        "quantum_experiment"
    ]

    improvement = (
        center.quantum_tracker
        .score_improvement(
            experiment
        )
    )

    assert improvement > 0.0


def test_e2e_quantum_safety():
    flow = build_complete_research_flow()

    center = flow["center"]

    experiment = flow[
        "quantum_experiment"
    ]

    report = (
        center.quantum_tracker
        .competitiveness_report(
            experiment.experiment_id
        )
    )

    assert report["research_only"] is True

    assert (
        report["real_money_execution"]
        is False
    )


# ============================================================
# DECISION LINEAGE
# ============================================================


def test_e2e_lineage_has_full_pipeline():
    flow = build_complete_research_flow()

    center = flow["center"]

    lineage = center.decision_lineage.get(
        flow["lineage"].lineage_id
    )

    assert len(lineage.events) == 8

    assert [
        event.stage
        for event in lineage.events
    ] == [
        "RAW_DATA",
        "AGENT_EVIDENCE",
        "HYPOTHESIS",
        "STRATEGY",
        "VALIDATION",
        "RISK",
        "SHADOW_ALLOCATION",
        "OUTCOME",
    ]


def test_e2e_lineage_stage_coverage_complete():
    flow = build_complete_research_flow()

    center = flow["center"]

    coverage = (
        center.decision_lineage
        .stage_coverage(
            flow["lineage"].lineage_id
        )
    )

    assert coverage["complete"] is True
    assert coverage["completed_stages"] == 8
    assert coverage["total_stages"] == 8
    assert coverage["coverage_ratio"] == 1.0


def test_e2e_lineage_integrity_valid():
    flow = build_complete_research_flow()

    center = flow["center"]

    report = (
        center.decision_lineage
        .integrity_report(
            flow["lineage"].lineage_id
        )
    )

    assert report["valid"] is True

    assert (
        report["broken_parent_references"]
        == []
    )

    assert (
        report["duplicate_entities"]
        == []
    )

    assert (
        report["invalid_stage_links"]
        == []
    )


def test_e2e_lineage_path_raw_to_outcome():
    flow = build_complete_research_flow()

    center = flow["center"]

    path = center.decision_lineage.path_to(
        flow["lineage"].lineage_id,
        flow["outcome"].entity_id,
    )

    assert len(path) == 8

    assert path[0].stage == "RAW_DATA"
    assert path[-1].stage == "OUTCOME"


def test_e2e_lineage_ancestry():
    flow = build_complete_research_flow()

    center = flow["center"]

    ancestry = (
        center.decision_lineage
        .ancestry(
            flow["lineage"].lineage_id,
            flow["outcome"].entity_id,
        )
    )

    assert len(ancestry) == 7

    assert (
        ancestry[0].entity_id
        == flow["raw_data"].entity_id
    )

    assert (
        ancestry[-1].entity_id
        == flow[
            "shadow_allocation"
        ].entity_id
    )


def test_e2e_lineage_descendants():
    flow = build_complete_research_flow()

    center = flow["center"]

    descendants = (
        center.decision_lineage
        .descendants(
            flow["lineage"].lineage_id,
            flow["raw_data"].entity_id,
        )
    )

    assert len(descendants) == 7

    assert (
        descendants[-1].entity_id
        == flow["outcome"].entity_id
    )


def test_e2e_lineage_pipeline_report():
    flow = build_complete_research_flow()

    center = flow["center"]

    report = (
        center.decision_lineage
        .pipeline_report(
            flow["lineage"].lineage_id
        )
    )

    assert report["events"] == 8
    assert report["auditable"] is True
    assert report["coverage"]["complete"] is True
    assert report["integrity"]["valid"] is True


# ============================================================
# EVIDENCE PRESERVATION
# ============================================================


def test_e2e_agent_evidence_preserved():
    flow = build_complete_research_flow()

    assert (
        flow["agent_evidence"]
        .evidence["confidence"]
        == 0.82
    )


def test_e2e_validation_evidence_preserved():
    flow = build_complete_research_flow()

    assert (
        flow["validation"]
        .evidence["decision"]
        == "PROMOTE"
    )

    assert (
        flow["validation"]
        .evidence[
            "out_of_sample_score"
        ]
        == 0.79
    )


def test_e2e_risk_evidence_preserved():
    flow = build_complete_research_flow()

    assert (
        flow["risk"]
        .evidence["decision"]
        == "APPROVE"
    )


def test_e2e_outcome_preserved():
    flow = build_complete_research_flow()

    assert (
        flow["outcome"]
        .evidence["return_pct"]
        == 0.04
    )

    assert (
        flow["outcome"]
        .evidence["hit"]
        is True
    )


# ============================================================
# CONTROL CENTER SNAPSHOT
# ============================================================


def test_e2e_model_snapshot():
    flow = build_complete_research_flow()

    center = flow["center"]

    snapshot = center.model_snapshot()

    assert isinstance(
        snapshot,
        ResearchControlCenterSnapshot,
    )

    assert (
        len(
            snapshot
            .quantum_classical_experiments
        )
        == 1
    )

    assert (
        len(snapshot.decision_lineages)
        == 1
    )


def test_e2e_counts():
    flow = build_complete_research_flow()

    center = flow["center"]

    counts = center.counts()

    assert (
        counts["quantum_experiments"]
        == 1
    )

    assert (
        counts["decision_lineages"]
        == 1
    )

    assert (
        counts["shadow_fund_available"]
        is True
    )


def test_e2e_snapshot_safety():
    flow = build_complete_research_flow()

    snapshot = flow[
        "center"
    ].snapshot()

    assert snapshot["auditable"] is True
    assert snapshot["research_only"] is True

    assert (
        snapshot["real_money_execution"]
        is False
    )


# ============================================================
# DASHBOARD
# ============================================================


def test_e2e_dashboard_contains_research_flow():
    flow = build_complete_research_flow()

    dashboard = flow[
        "center"
    ].dashboard()

    assert (
        len(
            dashboard[
                "quantum_classical_experiments"
            ]
        )
        == 1
    )

    assert (
        len(
            dashboard[
                "decision_lineages"
            ]
        )
        == 1
    )


def test_e2e_dashboard_is_auditable():
    flow = build_complete_research_flow()

    dashboard = flow[
        "center"
    ].dashboard()

    assert dashboard["auditable"] is True
    assert dashboard["research_only"] is True

    assert (
        dashboard["real_money_execution"]
        is False
    )


# ============================================================
# AUDIT
# ============================================================


def test_e2e_lineage_audit_complete():
    flow = build_complete_research_flow()

    center = flow["center"]

    audit = (
        center.decision_lineage
        .audit_log(
            flow["lineage"].lineage_id
        )
    )

    # 1 lineage creation + 8 events
    assert len(audit) == 9

    assert (
        audit[0]["event"]
        == "LINEAGE_CREATED"
    )

    assert (
        audit[-1]["event"]
        == "LINEAGE_EVENT_ADDED"
    )


def test_e2e_quantum_audit_exists():
    flow = build_complete_research_flow()

    audit = (
        flow["center"]
        .quantum_tracker
        .audit_log()
    )

    assert len(audit) >= 1

    assert (
        audit[0]["event"]
        == "QUANTUM_EXPERIMENT_REGISTERED"
    )


def test_e2e_control_center_audit():
    flow = build_complete_research_flow()

    center = flow["center"]

    center.dashboard()

    audit = center.audit_log()

    assert len(audit) == 2

    assert (
        audit[0]["event"]
        == "CONTROL_CENTER_INITIALIZED"
    )

    assert (
        audit[-1]["event"]
        == "CONTROL_CENTER_DASHBOARD_GENERATED"
    )


# ============================================================
# SYSTEM REPORT
# ============================================================


def test_e2e_system_report():
    flow = build_complete_research_flow()

    report = (
        flow["center"]
        .system_report()
    )

    assert report["version"] == "v5.0.10"

    assert (
        report["health"]["healthy"]
        is True
    )

    assert (
        report["health"][
            "total_components"
        ]
        == 7
    )


# ============================================================
# SAFETY BOUNDARY
# ============================================================


def test_e2e_control_center_cannot_execute_orders():
    flow = build_complete_research_flow()

    safety = (
        flow["center"]
        .safety_report()
    )

    assert (
        safety[
            "control_center_can_execute_orders"
        ]
        is False
    )

    assert (
        safety["broker_order_allowed"]
        is False
    )


def test_e2e_no_real_money_execution():
    flow = build_complete_research_flow()

    center = flow["center"]

    safety = center.safety_report()

    dashboard = center.dashboard()

    snapshot = center.snapshot()

    assert (
        safety["real_money_execution"]
        is False
    )

    assert (
        dashboard["real_money_execution"]
        is False
    )

    assert (
        snapshot["real_money_execution"]
        is False
    )


def test_e2e_no_risk_or_validation_bypass():
    flow = build_complete_research_flow()

    safety = (
        flow["center"]
        .safety_report()
    )

    assert (
        safety["risk_bypass_allowed"]
        is False
    )

    assert (
        safety[
            "validation_bypass_allowed"
        ]
        is False
    )


def test_e2e_shadow_allocation_is_research_only():
    flow = build_complete_research_flow()

    allocation = flow[
        "shadow_allocation"
    ]

    assert (
        allocation.evidence[
            "research_only"
        ]
        is True
    )

    assert (
        allocation.evidence[
            "real_money_execution"
        ]
        is False
    )


# ============================================================
# FINAL v5.0.10 CONTRACT
# ============================================================


def test_e2e_v5010_contract():
    flow = build_complete_research_flow()

    center = flow["center"]

    health = center.health()
    counts = center.counts()
    lineage_report = (
        center.decision_lineage
        .pipeline_report(
            flow["lineage"].lineage_id
        )
    )
    quantum_report = (
        center.quantum_tracker
        .competitiveness_report(
            flow[
                "quantum_experiment"
            ].experiment_id
        )
    )
    safety = center.safety_report()

    assert center.VERSION == "v5.0.10"

    assert health["healthy"] is True

    assert (
        counts["quantum_experiments"]
        == 1
    )

    assert (
        counts["decision_lineages"]
        == 1
    )

    assert (
        lineage_report["coverage"][
            "complete"
        ]
        is True
    )

    assert (
        lineage_report["integrity"][
            "valid"
        ]
        is True
    )

    assert (
        quantum_report["competitive"]
        is True
    )

    assert safety["research_only"] is True

    assert (
        safety["real_money_execution"]
        is False
    )

    assert (
        safety["broker_order_allowed"]
        is False
    )