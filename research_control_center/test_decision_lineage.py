import pytest

from research_control_center.decision_lineage import (
    DecisionLineageConfig,
    DecisionLineageEngine,
)


def make_engine() -> DecisionLineageEngine:
    return DecisionLineageEngine()


def make_lineage(
    engine: DecisionLineageEngine,
):
    return engine.create(
        root_id="research_task_001"
    )


def build_full_pipeline(
    engine: DecisionLineageEngine,
):
    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
        summary="Market snapshot",
        evidence={
            "asset": "BTC",
            "price": 100000.0,
        },
    )

    agent = engine.add_agent_evidence(
        lineage.lineage_id,
        "agent_001",
        raw.entity_id,
        summary="Agent evidence",
        evidence={
            "signal": "BULLISH",
            "confidence": 0.82,
        },
    )

    hypothesis = engine.add_hypothesis(
        lineage.lineage_id,
        "hypothesis_001",
        agent.entity_id,
        summary="Bullish hypothesis",
    )

    strategy = engine.add_strategy(
        lineage.lineage_id,
        "strategy_001",
        hypothesis.entity_id,
        summary="Long strategy",
    )

    validation = engine.add_validation(
        lineage.lineage_id,
        "validation_001",
        strategy.entity_id,
        evidence={
            "score": 0.81,
            "decision": "PROMOTE",
        },
    )

    risk = engine.add_risk(
        lineage.lineage_id,
        "risk_001",
        validation.entity_id,
        evidence={
            "decision": "APPROVE",
        },
    )

    allocation = engine.add_shadow_allocation(
        lineage.lineage_id,
        "allocation_001",
        risk.entity_id,
        evidence={
            "weight": 0.10,
            "research_only": True,
        },
    )

    outcome = engine.add_outcome(
        lineage.lineage_id,
        "outcome_001",
        allocation.entity_id,
        evidence={
            "return_pct": 0.05,
        },
    )

    return {
        "lineage": lineage,
        "raw": raw,
        "agent": agent,
        "hypothesis": hypothesis,
        "strategy": strategy,
        "validation": validation,
        "risk": risk,
        "allocation": allocation,
        "outcome": outcome,
    }


# ============================================================
# CONFIG
# ============================================================


def test_default_config():
    config = DecisionLineageConfig()

    assert config.enforce_stage_order is True
    assert config.require_parent is True
    assert config.allow_branching is True

    assert config.stage_order == (
        "RAW_DATA",
        "AGENT_EVIDENCE",
        "HYPOTHESIS",
        "STRATEGY",
        "VALIDATION",
        "RISK",
        "SHADOW_ALLOCATION",
        "OUTCOME",
    )


def test_custom_config():
    config = DecisionLineageConfig(
        enforce_stage_order=False,
        require_parent=False,
        allow_branching=False,
    )

    assert config.enforce_stage_order is False
    assert config.require_parent is False
    assert config.allow_branching is False


@pytest.mark.parametrize(
    "field",
    [
        "enforce_stage_order",
        "require_parent",
        "allow_branching",
    ],
)
def test_invalid_config_type_rejected(field):
    with pytest.raises(TypeError):
        DecisionLineageConfig(
            **{field: 1}
        )


# ============================================================
# CREATE
# ============================================================


def test_create_lineage():
    engine = make_engine()

    lineage = engine.create(
        root_id="task_001"
    )

    assert lineage.root_id == "task_001"
    assert lineage.events == []
    assert engine.count() == 1


def test_create_lineage_safety_metadata():
    engine = make_engine()

    lineage = engine.create(
        root_id="task_001"
    )

    assert lineage.metadata["auditable"] is True
    assert lineage.metadata["research_only"] is True
    assert (
        lineage.metadata["real_money_execution"]
        is False
    )


def test_create_with_custom_metadata():
    engine = make_engine()

    lineage = engine.create(
        root_id="task_001",
        metadata={
            "asset": "BTC",
            "regime": "RISK_ON",
        },
    )

    assert lineage.metadata["asset"] == "BTC"
    assert lineage.metadata["regime"] == "RISK_ON"


def test_empty_root_id_rejected():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.create(
            root_id=""
        )


def test_invalid_metadata_rejected():
    engine = make_engine()

    with pytest.raises(TypeError):
        engine.create(
            root_id="task_001",
            metadata="invalid",
        )


# ============================================================
# READ
# ============================================================


def test_get_lineage():
    engine = make_engine()

    created = make_lineage(engine)

    result = engine.get(
        created.lineage_id
    )

    assert (
        result.lineage_id
        == created.lineage_id
    )


def test_unknown_lineage_rejected():
    engine = make_engine()

    with pytest.raises(KeyError):
        engine.get(
            "missing"
        )


def test_get_returns_copy():
    engine = make_engine()

    created = make_lineage(engine)

    result = engine.get(
        created.lineage_id
    )

    result.root_id = "changed"

    original = engine.get(
        created.lineage_id
    )

    assert (
        original.root_id
        == "research_task_001"
    )


def test_list_all():
    engine = make_engine()

    engine.create("task_a")
    engine.create("task_b")

    assert len(
        engine.list_all()
    ) == 2


# ============================================================
# ADD EVENTS
# ============================================================


def test_add_raw_data():
    engine = make_engine()

    lineage = make_lineage(engine)

    event = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
        summary="Raw market data",
    )

    assert event.stage == "RAW_DATA"
    assert event.entity_type == "RAW_DATA"
    assert event.entity_id == "raw_001"
    assert event.parent_entity_id is None


def test_add_event_normalizes_stage():
    engine = make_engine()

    lineage = make_lineage(engine)

    event = engine.add_event(
        lineage.lineage_id,
        stage="raw_data",
        entity_type="RAW_DATA",
        entity_id="raw_001",
    )

    assert event.stage == "RAW_DATA"


def test_unknown_stage_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    with pytest.raises(ValueError):
        engine.add_event(
            lineage.lineage_id,
            stage="UNKNOWN",
            entity_type="TEST",
            entity_id="entity_001",
        )


def test_duplicate_entity_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    with pytest.raises(ValueError):
        engine.add_raw_data(
            lineage.lineage_id,
            "raw_001",
        )


def test_missing_parent_rejected_for_non_root():
    engine = make_engine()

    lineage = make_lineage(engine)

    engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    with pytest.raises(ValueError):
        engine.add_event(
            lineage.lineage_id,
            stage="AGENT_EVIDENCE",
            entity_type="AGENT",
            entity_id="agent_001",
        )


def test_unknown_parent_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    with pytest.raises(ValueError):
        engine.add_agent_evidence(
            lineage.lineage_id,
            "agent_001",
            "missing_parent",
        )


def test_same_stage_parent_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    with pytest.raises(ValueError):
        engine.add_event(
            lineage.lineage_id,
            stage="RAW_DATA",
            entity_type="RAW_DATA",
            entity_id="raw_002",
            parent_entity_id=raw.entity_id,
        )


def test_backward_stage_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    agent = engine.add_agent_evidence(
        lineage.lineage_id,
        "agent_001",
        raw.entity_id,
    )

    with pytest.raises(ValueError):
        engine.add_event(
            lineage.lineage_id,
            stage="RAW_DATA",
            entity_type="RAW_DATA",
            entity_id="raw_002",
            parent_entity_id=agent.entity_id,
        )


def test_stage_skipping_is_allowed():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    strategy = engine.add_strategy(
        lineage.lineage_id,
        "strategy_001",
        raw.entity_id,
    )

    assert strategy.stage == "STRATEGY"


def test_event_safety_metadata():
    engine = make_engine()

    lineage = make_lineage(engine)

    event = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    assert event.metadata["research_only"] is True
    assert (
        event.metadata["real_money_execution"]
        is False
    )


# ============================================================
# BRANCHING
# ============================================================


def test_branching_allowed():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    first = engine.add_agent_evidence(
        lineage.lineage_id,
        "macro_agent",
        raw.entity_id,
    )

    second = engine.add_agent_evidence(
        lineage.lineage_id,
        "technical_agent",
        raw.entity_id,
    )

    assert first.parent_entity_id == "raw_001"
    assert second.parent_entity_id == "raw_001"


def test_branching_disabled():
    engine = DecisionLineageEngine(
        DecisionLineageConfig(
            allow_branching=False
        )
    )

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    engine.add_agent_evidence(
        lineage.lineage_id,
        "agent_001",
        raw.entity_id,
    )

    with pytest.raises(ValueError):
        engine.add_agent_evidence(
            lineage.lineage_id,
            "agent_002",
            raw.entity_id,
        )


# ============================================================
# PIPELINE HELPERS
# ============================================================


def test_full_pipeline():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    lineage = engine.get(
        pipeline["lineage"].lineage_id
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


def test_full_pipeline_preserves_evidence():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    validation = engine.get_event_by_entity(
        pipeline["lineage"].lineage_id,
        "validation_001",
    )

    assert (
        validation.evidence["score"]
        == 0.81
    )
    assert (
        validation.evidence["decision"]
        == "PROMOTE"
    )


# ============================================================
# EVENT LOOKUP
# ============================================================


def test_get_event():
    engine = make_engine()

    lineage = make_lineage(engine)

    created = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    result = engine.get_event(
        lineage.lineage_id,
        created.event_id,
    )

    assert (
        result.event_id
        == created.event_id
    )


def test_get_event_by_entity():
    engine = make_engine()

    lineage = make_lineage(engine)

    engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    result = engine.get_event_by_entity(
        lineage.lineage_id,
        "raw_001",
    )

    assert result.entity_id == "raw_001"


def test_unknown_event_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    with pytest.raises(KeyError):
        engine.get_event(
            lineage.lineage_id,
            "missing",
        )


def test_unknown_entity_rejected():
    engine = make_engine()

    lineage = make_lineage(engine)

    with pytest.raises(KeyError):
        engine.get_event_by_entity(
            lineage.lineage_id,
            "missing",
        )


# ============================================================
# CHILDREN
# ============================================================


def test_children():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    engine.add_agent_evidence(
        lineage.lineage_id,
        "macro_agent",
        raw.entity_id,
    )

    engine.add_agent_evidence(
        lineage.lineage_id,
        "technical_agent",
        raw.entity_id,
    )

    children = engine.children(
        lineage.lineage_id,
        raw.entity_id,
    )

    assert len(children) == 2

    assert {
        child.entity_id
        for child in children
    } == {
        "macro_agent",
        "technical_agent",
    }


# ============================================================
# ANCESTRY
# ============================================================


def test_ancestry():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    ancestors = engine.ancestry(
        pipeline["lineage"].lineage_id,
        "outcome_001",
    )

    assert [
        event.entity_id
        for event in ancestors
    ] == [
        "raw_001",
        "agent_001",
        "hypothesis_001",
        "strategy_001",
        "validation_001",
        "risk_001",
        "allocation_001",
    ]


def test_root_ancestry_is_empty():
    engine = make_engine()

    lineage = make_lineage(engine)

    engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    assert (
        engine.ancestry(
            lineage.lineage_id,
            "raw_001",
        )
        == []
    )


# ============================================================
# DESCENDANTS
# ============================================================


def test_descendants():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    descendants = engine.descendants(
        pipeline["lineage"].lineage_id,
        "raw_001",
    )

    assert len(descendants) == 7

    assert (
        descendants[-1].entity_id
        == "outcome_001"
    )


def test_leaf_has_no_descendants():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    descendants = engine.descendants(
        pipeline["lineage"].lineage_id,
        "outcome_001",
    )

    assert descendants == []


# ============================================================
# PATH
# ============================================================


def test_path_to_outcome():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    path = engine.path_to(
        pipeline["lineage"].lineage_id,
        "outcome_001",
    )

    assert len(path) == 8

    assert path[0].stage == "RAW_DATA"
    assert path[-1].stage == "OUTCOME"


# ============================================================
# STAGE COVERAGE
# ============================================================


def test_partial_stage_coverage():
    engine = make_engine()

    lineage = make_lineage(engine)

    raw = engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    engine.add_agent_evidence(
        lineage.lineage_id,
        "agent_001",
        raw.entity_id,
    )

    coverage = engine.stage_coverage(
        lineage.lineage_id
    )

    assert coverage["completed_stages"] == 2
    assert coverage["total_stages"] == 8
    assert coverage["coverage_ratio"] == 0.25
    assert coverage["complete"] is False


def test_full_stage_coverage():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    coverage = engine.stage_coverage(
        pipeline["lineage"].lineage_id
    )

    assert coverage["completed_stages"] == 8
    assert coverage["coverage_ratio"] == 1.0
    assert coverage["complete"] is True


# ============================================================
# INTEGRITY
# ============================================================


def test_valid_integrity_report():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    report = engine.integrity_report(
        pipeline["lineage"].lineage_id
    )

    assert report["valid"] is True
    assert report["event_count"] == 8
    assert report["root_event_count"] == 1
    assert report["duplicate_entities"] == []
    assert report["broken_parent_references"] == []
    assert report["invalid_stage_links"] == []


def test_integrity_report_is_research_only():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    report = engine.integrity_report(
        pipeline["lineage"].lineage_id
    )

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


# ============================================================
# PIPELINE REPORT
# ============================================================


def test_pipeline_report():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    report = engine.pipeline_report(
        pipeline["lineage"].lineage_id
    )

    assert report["events"] == 8
    assert report["coverage"]["complete"] is True
    assert report["integrity"]["valid"] is True
    assert len(
        report["by_stage"]["RAW_DATA"]
    ) == 1
    assert len(
        report["by_stage"]["OUTCOME"]
    ) == 1


def test_pipeline_report_is_auditable():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    report = engine.pipeline_report(
        pipeline["lineage"].lineage_id
    )

    assert report["auditable"] is True
    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


# ============================================================
# SUMMARY
# ============================================================


def test_summary_empty():
    engine = make_engine()

    summary = engine.summary()

    assert summary["total_lineages"] == 0
    assert summary["total_events"] == 0
    assert summary["complete_lineages"] == 0
    assert summary["valid_lineages"] == 0


def test_summary_full_pipeline():
    engine = make_engine()

    build_full_pipeline(
        engine
    )

    summary = engine.summary()

    assert summary["total_lineages"] == 1
    assert summary["total_events"] == 8
    assert summary["complete_lineages"] == 1
    assert summary["valid_lineages"] == 1
    assert summary["auditable"] is True


# ============================================================
# AUDIT
# ============================================================


def test_create_generates_audit_event():
    engine = make_engine()

    lineage = make_lineage(engine)

    events = engine.audit_log(
        lineage.lineage_id
    )

    assert len(events) == 1
    assert (
        events[0]["event"]
        == "LINEAGE_CREATED"
    )


def test_add_event_generates_audit_event():
    engine = make_engine()

    lineage = make_lineage(engine)

    engine.add_raw_data(
        lineage.lineage_id,
        "raw_001",
    )

    events = engine.audit_log(
        lineage.lineage_id
    )

    assert len(events) == 2
    assert (
        events[1]["event"]
        == "LINEAGE_EVENT_ADDED"
    )


def test_full_pipeline_audit_count():
    engine = make_engine()

    pipeline = build_full_pipeline(
        engine
    )

    events = engine.audit_log(
        pipeline["lineage"].lineage_id
    )

    assert len(events) == 9


def test_audit_is_research_only():
    engine = make_engine()

    lineage = make_lineage(engine)

    event = engine.audit_log(
        lineage.lineage_id
    )[0]

    assert event["research_only"] is True
    assert (
        event["real_money_execution"]
        is False
    )


def test_audit_log_returns_copy():
    engine = make_engine()

    lineage = make_lineage(engine)

    events = engine.audit_log(
        lineage.lineage_id
    )

    events[0]["event"] = "CHANGED"

    original = engine.audit_log(
        lineage.lineage_id
    )

    assert (
        original[0]["event"]
        == "LINEAGE_CREATED"
    )


# ============================================================
# SNAPSHOT
# ============================================================


def test_snapshot():
    engine = make_engine()

    build_full_pipeline(
        engine
    )

    snapshot = engine.snapshot()

    assert len(snapshot["lineages"]) == 1
    assert (
        snapshot["summary"]["total_lineages"]
        == 1
    )
    assert (
        snapshot["summary"]["total_events"]
        == 8
    )


def test_snapshot_is_research_only():
    engine = make_engine()

    snapshot = engine.snapshot()

    assert snapshot["auditable"] is True
    assert snapshot["research_only"] is True
    assert (
        snapshot["real_money_execution"]
        is False
    )