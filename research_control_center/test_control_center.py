from research_control_center.control_center import ResearchControlCenter
from research_control_center.models import ResearchControlCenterSnapshot


# ============================================================
# INITIALIZATION
# ============================================================


def test_control_center_initialization():
    center = ResearchControlCenter()

    assert center.VERSION == "v5.0.10"


def test_all_seven_components_exist():
    center = ResearchControlCenter()

    components = center.components()

    assert len(components) == 7

    assert components["experiment_registry"] is not None
    assert components["leaderboard"] is not None
    assert components["alpha_library"] is not None
    assert components["backtest_diagnostics"] is not None
    assert components["shadow_fund"] is not None
    assert components["quantum_tracker"] is not None
    assert components["decision_lineage"] is not None


def test_component_names():
    center = ResearchControlCenter()

    assert center.component_names() == (
        "experiment_registry",
        "leaderboard",
        "alpha_library",
        "backtest_diagnostics",
        "shadow_fund",
        "quantum_tracker",
        "decision_lineage",
    )


# ============================================================
# HEALTH
# ============================================================


def test_health():
    center = ResearchControlCenter()

    health = center.health()

    assert health["version"] == "v5.0.10"
    assert health["healthy"] is True
    assert health["healthy_components"] == 7
    assert health["total_components"] == 7


def test_health_all_components_true():
    center = ResearchControlCenter()

    health = center.health()

    assert all(
        health["components"].values()
    )


def test_health_is_research_only():
    center = ResearchControlCenter()

    health = center.health()

    assert health["research_only"] is True
    assert (
        health["real_money_execution"]
        is False
    )


# ============================================================
# EMPTY MODEL SNAPSHOT
# ============================================================


def test_model_snapshot_type():
    center = ResearchControlCenter()

    snapshot = center.model_snapshot()

    assert isinstance(
        snapshot,
        ResearchControlCenterSnapshot,
    )


def test_model_snapshot_version():
    center = ResearchControlCenter()

    snapshot = center.model_snapshot()

    assert (
        snapshot.metadata["version"]
        == "v5.0.10"
    )


def test_model_snapshot_safety():
    center = ResearchControlCenter()

    snapshot = center.model_snapshot()

    assert (
        snapshot.metadata["research_only"]
        is True
    )

    assert (
        snapshot.metadata[
            "real_money_execution"
        ]
        is False
    )

    assert (
        snapshot.metadata["auditable"]
        is True
    )


def test_empty_model_collections():
    center = ResearchControlCenter()

    snapshot = center.model_snapshot()

    assert snapshot.experiments == []
    assert snapshot.leaderboard == []
    assert snapshot.alpha_library == []
    assert snapshot.backtest_diagnostics == []
    assert (
        snapshot.quantum_classical_experiments
        == []
    )
    assert snapshot.decision_lineages == []


# ============================================================
# COUNTS
# ============================================================


def test_empty_counts():
    center = ResearchControlCenter()

    counts = center.counts()

    assert counts["experiments"] == 0
    assert counts["leaderboard_entries"] == 0
    assert counts["alpha_records"] == 0
    assert counts["backtest_diagnostics"] == 0
    assert counts["quantum_experiments"] == 0
    assert counts["decision_lineages"] == 0


def test_shadow_fund_available():
    center = ResearchControlCenter()

    counts = center.counts()

    assert (
        counts["shadow_fund_available"]
        is True
    )


# ============================================================
# SUMMARY
# ============================================================


def test_summary_status():
    center = ResearchControlCenter()

    summary = center.summary()

    assert summary["version"] == "v5.0.10"
    assert summary["status"] == "HEALTHY"


def test_summary_component_count():
    center = ResearchControlCenter()

    summary = center.summary()

    assert len(
        summary["components"]
    ) == 7


def test_summary_safety():
    center = ResearchControlCenter()

    summary = center.summary()

    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )
    assert summary["auditable"] is True


# ============================================================
# EXPERIMENT REGISTRY INTEGRATION
# ============================================================


def test_experiment_registry_integration():
    center = ResearchControlCenter()

    assert (
        center.experiment_registry
        is center.components()[
            "experiment_registry"
        ]
    )


# ============================================================
# LEADERBOARD INTEGRATION
# ============================================================


def test_leaderboard_integration():
    center = ResearchControlCenter()

    assert (
        center.leaderboard
        is center.components()[
            "leaderboard"
        ]
    )


# ============================================================
# ALPHA LIBRARY INTEGRATION
# ============================================================


def test_alpha_library_integration():
    center = ResearchControlCenter()

    assert (
        center.alpha_library
        is center.components()[
            "alpha_library"
        ]
    )


# ============================================================
# BACKTEST INTEGRATION
# ============================================================


def test_backtest_integration():
    center = ResearchControlCenter()

    assert (
        center.backtest_diagnostics
        is center.components()[
            "backtest_diagnostics"
        ]
    )


# ============================================================
# SHADOW FUND INTEGRATION
# ============================================================


def test_shadow_fund_integration():
    center = ResearchControlCenter()

    assert (
        center.shadow_fund
        is center.components()[
            "shadow_fund"
        ]
    )


# ============================================================
# QUANTUM INTEGRATION
# ============================================================


def test_quantum_tracker_integration():
    center = ResearchControlCenter()

    assert (
        center.quantum_tracker
        is center.components()[
            "quantum_tracker"
        ]
    )


def test_quantum_tracker_data_reaches_snapshot():
    center = ResearchControlCenter()

    center.quantum_tracker.create(
        problem_name="Portfolio Selection",
        classical_method="MeanVariance",
        quantum_method="QAOA",
        asset_universe=[
            "NVDA",
            "MSFT",
        ],
        classical_score=0.80,
        quantum_score=0.85,
        classical_runtime_seconds=1.0,
        quantum_runtime_seconds=2.0,
        classical_objective=0.90,
        quantum_objective=0.85,
        approximation_ratio=0.94,
        qubits=4,
        circuit_depth=12,
        repetitions=3,
    )

    snapshot = center.model_snapshot()

    assert (
        len(
            snapshot
            .quantum_classical_experiments
        )
        == 1
    )

    assert (
        snapshot
        .quantum_classical_experiments[0]
        .quantum_method
        == "QAOA"
    )


def test_quantum_count_updates():
    center = ResearchControlCenter()

    center.quantum_tracker.create(
        problem_name="Portfolio",
        classical_method="Classical",
        quantum_method="QAOA",
        asset_universe=["BTC"],
    )

    assert (
        center.counts()[
            "quantum_experiments"
        ]
        == 1
    )


# ============================================================
# DECISION LINEAGE INTEGRATION
# ============================================================


def test_decision_lineage_integration():
    center = ResearchControlCenter()

    assert (
        center.decision_lineage
        is center.components()[
            "decision_lineage"
        ]
    )


def test_decision_lineage_reaches_snapshot():
    center = ResearchControlCenter()

    lineage = (
        center.decision_lineage.create(
            root_id="research_001"
        )
    )

    center.decision_lineage.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    snapshot = center.model_snapshot()

    assert (
        len(snapshot.decision_lineages)
        == 1
    )

    assert (
        snapshot
        .decision_lineages[0]
        .root_id
        == "research_001"
    )


def test_decision_lineage_count_updates():
    center = ResearchControlCenter()

    center.decision_lineage.create(
        root_id="research_001"
    )

    assert (
        center.counts()[
            "decision_lineages"
        ]
        == 1
    )


# ============================================================
# DASHBOARD
# ============================================================


def test_dashboard():
    center = ResearchControlCenter()

    dashboard = center.dashboard()

    assert dashboard["version"] == "v5.0.10"

    assert "health" in dashboard
    assert "summary" in dashboard
    assert "experiments" in dashboard
    assert "leaderboard" in dashboard
    assert "alpha_library" in dashboard
    assert "backtest_diagnostics" in dashboard
    assert "shadow_fund" in dashboard

    assert (
        "quantum_classical_experiments"
        in dashboard
    )

    assert (
        "decision_lineages"
        in dashboard
    )


def test_dashboard_safety():
    center = ResearchControlCenter()

    dashboard = center.dashboard()

    assert dashboard["auditable"] is True
    assert dashboard["research_only"] is True
    assert (
        dashboard["real_money_execution"]
        is False
    )


def test_dashboard_quantum_data():
    center = ResearchControlCenter()

    center.quantum_tracker.create(
        problem_name="QUBO",
        classical_method="MinimumVariance",
        quantum_method="QAOA",
        asset_universe=["NVDA"],
    )

    dashboard = center.dashboard()

    assert (
        len(
            dashboard[
                "quantum_classical_experiments"
            ]
        )
        == 1
    )


def test_dashboard_lineage_data():
    center = ResearchControlCenter()

    center.decision_lineage.create(
        root_id="research_001"
    )

    dashboard = center.dashboard()

    assert (
        len(
            dashboard[
                "decision_lineages"
            ]
        )
        == 1
    )


# ============================================================
# SAFETY
# ============================================================


def test_safety_report():
    center = ResearchControlCenter()

    safety = center.safety_report()

    assert safety["research_only"] is True
    assert safety["shadow_execution_only"] is True
    assert safety["real_money_execution"] is False
    assert safety["broker_order_allowed"] is False
    assert safety["live_trade_authorization"] is False
    assert safety["risk_bypass_allowed"] is False
    assert (
        safety["validation_bypass_allowed"]
        is False
    )
    assert (
        safety[
            "control_center_can_execute_orders"
        ]
        is False
    )


# ============================================================
# AUDIT
# ============================================================


def test_initialization_audit():
    center = ResearchControlCenter()

    audit = center.audit_log()

    assert len(audit) == 1

    assert (
        audit[0]["event"]
        == "CONTROL_CENTER_INITIALIZED"
    )


def test_dashboard_generates_audit():
    center = ResearchControlCenter()

    center.dashboard()

    audit = center.audit_log()

    assert len(audit) == 2

    assert (
        audit[-1]["event"]
        == "CONTROL_CENTER_DASHBOARD_GENERATED"
    )


def test_audit_safety():
    center = ResearchControlCenter()

    event = center.audit_log()[0]

    assert event["research_only"] is True
    assert (
        event["real_money_execution"]
        is False
    )


def test_audit_returns_copy():
    center = ResearchControlCenter()

    audit = center.audit_log()

    audit[0]["event"] = "CHANGED"

    original = center.audit_log()

    assert (
        original[0]["event"]
        == "CONTROL_CENTER_INITIALIZED"
    )


# ============================================================
# SYSTEM REPORT
# ============================================================


def test_system_report():
    center = ResearchControlCenter()

    report = center.system_report()

    assert report["version"] == "v5.0.10"
    assert report["health"]["healthy"] is True
    assert report["health"]["total_components"] == 7
    assert report["safety"]["research_only"] is True
    assert (
        report["safety"]["real_money_execution"]
        is False
    )


# ============================================================
# SNAPSHOT
# ============================================================


def test_snapshot():
    center = ResearchControlCenter()

    snapshot = center.snapshot()

    assert snapshot["version"] == "v5.0.10"

    assert isinstance(
        snapshot["model"],
        ResearchControlCenterSnapshot,
    )

    assert snapshot["health"]["healthy"] is True


def test_snapshot_safety():
    center = ResearchControlCenter()

    snapshot = center.snapshot()

    assert snapshot["auditable"] is True
    assert snapshot["research_only"] is True
    assert (
        snapshot["real_money_execution"]
        is False
    )

    assert (
        snapshot["safety"][
            "broker_order_allowed"
        ]
        is False
    )


# ============================================================
# CROSS-COMPONENT OBSERVABILITY
# ============================================================


def test_quantum_and_lineage_visible_together():
    center = ResearchControlCenter()

    center.quantum_tracker.create(
        problem_name="Portfolio Optimization",
        classical_method="RiskParity",
        quantum_method="QAOA",
        asset_universe=[
            "BTC",
            "ETH",
        ],
        classical_score=0.70,
        quantum_score=0.75,
    )

    lineage = center.decision_lineage.create(
        root_id="research_quantum_001"
    )

    center.decision_lineage.add_raw_data(
        lineage.lineage_id,
        "raw_quantum_001",
    )

    snapshot = center.snapshot()

    assert (
        snapshot["counts"][
            "quantum_experiments"
        ]
        == 1
    )

    assert (
        snapshot["counts"][
            "decision_lineages"
        ]
        == 1
    )


def test_control_center_remains_non_executional():
    center = ResearchControlCenter()

    dashboard = center.dashboard()
    safety = center.safety_report()

    assert dashboard["research_only"] is True

    assert (
        dashboard["real_money_execution"]
        is False
    )

    assert (
        safety["broker_order_allowed"]
        is False
    )

    assert (
        safety["live_trade_authorization"]
        is False
    )