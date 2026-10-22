from __future__ import annotations

import pytest

from production_research_infrastructure.lineage_engine import (
    LineageCycleError,
    LineageEdge,
    LineageEdgeAlreadyExistsError,
    LineageEngine,
    LineageEngineError,
    LineageNode,
    LineageNodeNotFoundError,
    LineageNodeType,
    LineagePath,
    LineageRelation,
    LineageValidationError,
)


def make_node(
    engine: LineageEngine,
    node_id: str,
    node_type: LineageNodeType = LineageNodeType.OTHER,
):
    return engine.create_node(
        node_id=node_id,
        node_type=node_type,
        label=node_id,
    )


def build_chain() -> LineageEngine:
    engine = LineageEngine()

    make_node(
        engine,
        "dataset_001",
        LineageNodeType.DATASET,
    )
    make_node(
        engine,
        "model_v1",
        LineageNodeType.MODEL_VERSION,
    )
    make_node(
        engine,
        "run_001",
        LineageNodeType.RUN,
    )
    make_node(
        engine,
        "artifact_001",
        LineageNodeType.ARTIFACT,
    )

    engine.connect(
        "dataset_001",
        "run_001",
        LineageRelation.USES,
    )
    engine.connect(
        "model_v1",
        "run_001",
        LineageRelation.USES,
    )
    engine.connect(
        "run_001",
        "artifact_001",
        LineageRelation.PRODUCES,
    )

    return engine


# ============================================================
# LineageNode
# ============================================================


def test_node_creation() -> None:
    node = LineageNode(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
    )

    assert node.node_id == "run_001"
    assert node.node_type == LineageNodeType.RUN


def test_node_id_trimmed() -> None:
    node = LineageNode(
        node_id=" run_001 ",
        node_type=LineageNodeType.RUN,
    )

    assert node.node_id == "run_001"


def test_node_rejects_empty_id() -> None:
    with pytest.raises(LineageValidationError):
        LineageNode(
            node_id=" ",
            node_type=LineageNodeType.RUN,
        )


def test_node_requires_node_type_enum() -> None:
    with pytest.raises(TypeError):
        LineageNode(
            node_id="run_001",
            node_type="RUN",
        )


def test_node_label_trimmed() -> None:
    node = LineageNode(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
        label=" Research Run ",
    )

    assert node.label == "Research Run"


def test_node_empty_label_rejected() -> None:
    with pytest.raises(LineageValidationError):
        LineageNode(
            node_id="run_001",
            node_type=LineageNodeType.RUN,
            label=" ",
        )


def test_node_fingerprint_trimmed() -> None:
    node = LineageNode(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
        fingerprint=" abc123 ",
    )

    assert node.fingerprint == "abc123"


def test_node_empty_fingerprint_rejected() -> None:
    with pytest.raises(LineageValidationError):
        LineageNode(
            node_id="run_001",
            node_type=LineageNodeType.RUN,
            fingerprint=" ",
        )


def test_node_safety_metadata() -> None:
    node = LineageNode(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
    )

    assert node.metadata["research_only"] is True
    assert (
        node.metadata["real_money_execution"]
        is False
    )


@pytest.mark.parametrize(
    "node_type",
    list(LineageNodeType),
)
def test_all_node_types_supported(
    node_type,
) -> None:
    node = LineageNode(
        node_id=f"node_{node_type.value}",
        node_type=node_type,
    )

    assert node.node_type == node_type


# ============================================================
# LineageEdge
# ============================================================


def test_edge_creation() -> None:
    edge = LineageEdge(
        source_id="dataset_001",
        target_id="run_001",
        relation=LineageRelation.USES,
    )

    assert edge.source_id == "dataset_001"
    assert edge.target_id == "run_001"
    assert edge.edge_id.startswith("lineage_")


def test_edge_ids_trimmed() -> None:
    edge = LineageEdge(
        source_id=" dataset_001 ",
        target_id=" run_001 ",
        relation=LineageRelation.USES,
    )

    assert edge.source_id == "dataset_001"
    assert edge.target_id == "run_001"


@pytest.mark.parametrize(
    "source,target",
    [
        (" ", "target"),
        ("source", " "),
    ],
)
def test_edge_rejects_empty_node_ids(
    source,
    target,
) -> None:
    with pytest.raises(LineageValidationError):
        LineageEdge(
            source_id=source,
            target_id=target,
            relation=LineageRelation.USES,
        )


def test_edge_rejects_self_reference() -> None:
    with pytest.raises(LineageValidationError):
        LineageEdge(
            source_id="run_001",
            target_id="run_001",
            relation=LineageRelation.RELATED_TO,
        )


def test_edge_requires_relation_enum() -> None:
    with pytest.raises(TypeError):
        LineageEdge(
            source_id="a",
            target_id="b",
            relation="USES",
        )


def test_edge_id_is_deterministic() -> None:
    first = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
    )

    second = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
    )

    assert first.edge_id == second.edge_id


def test_different_relation_has_different_edge_id() -> None:
    first = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
    )

    second = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.DEPENDS_ON,
    )

    assert first.edge_id != second.edge_id


def test_custom_edge_id() -> None:
    edge = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
        edge_id="edge_custom",
    )

    assert edge.edge_id == "edge_custom"


def test_empty_custom_edge_id_rejected() -> None:
    with pytest.raises(LineageValidationError):
        LineageEdge(
            source_id="a",
            target_id="b",
            relation=LineageRelation.USES,
            edge_id=" ",
        )


def test_edge_safety_metadata() -> None:
    edge = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
    )

    assert edge.metadata["research_only"] is True
    assert (
        edge.metadata["real_money_execution"]
        is False
    )


@pytest.mark.parametrize(
    "relation",
    list(LineageRelation),
)
def test_all_relations_supported(
    relation,
) -> None:
    edge = LineageEdge(
        source_id="source",
        target_id="target",
        relation=relation,
    )

    assert edge.relation == relation


# ============================================================
# LineagePath
# ============================================================


def test_empty_path_rejected() -> None:
    with pytest.raises(LineageValidationError):
        LineagePath(
            node_ids=[],
            edge_ids=[],
            relations=[],
        )


def test_single_node_path() -> None:
    path = LineagePath(
        node_ids=["run_001"],
        edge_ids=[],
        relations=[],
    )

    assert path.length == 0


def test_path_length() -> None:
    path = LineagePath(
        node_ids=["a", "b", "c"],
        edge_ids=["e1", "e2"],
        relations=[
            LineageRelation.USES,
            LineageRelation.PRODUCES,
        ],
    )

    assert path.length == 2


def test_path_rejects_wrong_edge_count() -> None:
    with pytest.raises(LineageValidationError):
        LineagePath(
            node_ids=["a", "b", "c"],
            edge_ids=["e1"],
            relations=[
                LineageRelation.USES
            ],
        )


def test_path_rejects_wrong_relation_count() -> None:
    with pytest.raises(LineageValidationError):
        LineagePath(
            node_ids=["a", "b"],
            edge_ids=["e1"],
            relations=[],
        )


def test_path_lists_required() -> None:
    with pytest.raises(TypeError):
        LineagePath(
            node_ids="a",
            edge_ids=[],
            relations=[],
        )


def test_path_safety_metadata() -> None:
    path = LineagePath(
        node_ids=["a"],
        edge_ids=[],
        relations=[],
    )

    assert path.metadata["research_only"] is True
    assert (
        path.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Engine basics
# ============================================================


def test_engine_version() -> None:
    assert LineageEngine.VERSION == "v6.0.1"


def test_engine_starts_empty() -> None:
    engine = LineageEngine()

    assert engine.node_count() == 0
    assert engine.edge_count() == 0
    assert engine.nodes() == []
    assert engine.edges() == []
    assert engine.audit_log() == []


def test_add_node() -> None:
    engine = LineageEngine()

    node = LineageNode(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
    )

    result = engine.add_node(node)

    assert result is node
    assert engine.node_count() == 1


def test_add_node_requires_node() -> None:
    engine = LineageEngine()

    with pytest.raises(TypeError):
        engine.add_node("invalid")


def test_duplicate_node_returns_existing() -> None:
    engine = LineageEngine()

    first = make_node(
        engine,
        "run_001",
        LineageNodeType.RUN,
    )

    second = engine.create_node(
        node_id="run_001",
        node_type=LineageNodeType.RUN,
    )

    assert second is first
    assert engine.node_count() == 1


def test_duplicate_node_different_type_rejected() -> None:
    engine = LineageEngine()

    make_node(
        engine,
        "shared_id",
        LineageNodeType.RUN,
    )

    with pytest.raises(LineageValidationError):
        engine.create_node(
            node_id="shared_id",
            node_type=LineageNodeType.DATASET,
        )


def test_get_node() -> None:
    engine = LineageEngine()

    node = make_node(
        engine,
        "run_001",
        LineageNodeType.RUN,
    )

    assert engine.get_node("run_001") is node


def test_get_missing_node() -> None:
    engine = LineageEngine()

    with pytest.raises(
        LineageNodeNotFoundError
    ):
        engine.get_node("missing")


def test_get_node_empty_id_rejected() -> None:
    engine = LineageEngine()

    with pytest.raises(LineageValidationError):
        engine.get_node(" ")


def test_has_node_true() -> None:
    engine = LineageEngine()

    make_node(engine, "node_001")

    assert engine.has_node("node_001")


def test_has_node_false() -> None:
    engine = LineageEngine()

    assert not engine.has_node("missing")


def test_nodes_preserve_order() -> None:
    engine = LineageEngine()

    first = make_node(engine, "one")
    second = make_node(engine, "two")

    assert engine.nodes() == [
        first,
        second,
    ]


def test_nodes_by_type() -> None:
    engine = LineageEngine()

    run = make_node(
        engine,
        "run",
        LineageNodeType.RUN,
    )

    make_node(
        engine,
        "dataset",
        LineageNodeType.DATASET,
    )

    assert engine.nodes_by_type(
        LineageNodeType.RUN
    ) == [run]


def test_nodes_by_type_requires_enum() -> None:
    engine = LineageEngine()

    with pytest.raises(TypeError):
        engine.nodes_by_type("RUN")


# ============================================================
# Edge management
# ============================================================


def test_add_edge() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    edge = LineageEdge(
        source_id="a",
        target_id="b",
        relation=LineageRelation.USES,
    )

    result = engine.add_edge(edge)

    assert result is edge
    assert engine.edge_count() == 1


def test_add_edge_requires_edge() -> None:
    engine = LineageEngine()

    with pytest.raises(TypeError):
        engine.add_edge("invalid")


def test_edge_requires_source_node() -> None:
    engine = LineageEngine()

    make_node(engine, "b")

    with pytest.raises(
        LineageNodeNotFoundError
    ):
        engine.connect(
            "missing",
            "b",
            LineageRelation.USES,
        )


def test_edge_requires_target_node() -> None:
    engine = LineageEngine()

    make_node(engine, "a")

    with pytest.raises(
        LineageNodeNotFoundError
    ):
        engine.connect(
            "a",
            "missing",
            LineageRelation.USES,
        )


def test_connect() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    edge = engine.connect(
        "a",
        "b",
        LineageRelation.DERIVED_FROM,
    )

    assert edge.source_id == "a"
    assert edge.target_id == "b"
    assert (
        edge.relation
        == LineageRelation.DERIVED_FROM
    )


def test_duplicate_edge_rejected() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    with pytest.raises(
        LineageEdgeAlreadyExistsError
    ):
        engine.connect(
            "a",
            "b",
            LineageRelation.USES,
        )


def test_same_nodes_different_relation_allowed() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    engine.connect(
        "a",
        "b",
        LineageRelation.DEPENDS_ON,
    )

    assert engine.edge_count() == 2


def test_get_edge() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    edge = engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    assert engine.get_edge(
        edge.edge_id
    ) is edge


def test_get_missing_edge() -> None:
    engine = LineageEngine()

    with pytest.raises(LineageEngineError):
        engine.get_edge("missing")


def test_get_edge_empty_id_rejected() -> None:
    engine = LineageEngine()

    with pytest.raises(LineageValidationError):
        engine.get_edge(" ")


def test_edges_preserve_order() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")
    make_node(engine, "c")

    first = engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    second = engine.connect(
        "b",
        "c",
        LineageRelation.PRODUCES,
    )

    assert engine.edges() == [
        first,
        second,
    ]


def test_edges_by_relation() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")
    make_node(engine, "c")

    uses = engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    engine.connect(
        "b",
        "c",
        LineageRelation.PRODUCES,
    )

    assert engine.edges_by_relation(
        LineageRelation.USES
    ) == [uses]


def test_edges_by_relation_requires_enum() -> None:
    engine = LineageEngine()

    with pytest.raises(TypeError):
        engine.edges_by_relation("USES")


# ============================================================
# Cycle protection
# ============================================================


def test_direct_cycle_rejected() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    engine.connect(
        "a",
        "b",
        LineageRelation.DEPENDS_ON,
    )

    with pytest.raises(LineageCycleError):
        engine.connect(
            "b",
            "a",
            LineageRelation.DEPENDS_ON,
        )


def test_indirect_cycle_rejected() -> None:
    engine = LineageEngine()

    for node_id in ("a", "b", "c"):
        make_node(engine, node_id)

    engine.connect(
        "a",
        "b",
        LineageRelation.DEPENDS_ON,
    )
    engine.connect(
        "b",
        "c",
        LineageRelation.DEPENDS_ON,
    )

    with pytest.raises(LineageCycleError):
        engine.connect(
            "c",
            "a",
            LineageRelation.DEPENDS_ON,
        )


def test_cycle_can_be_explicitly_allowed() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    engine.connect(
        "a",
        "b",
        LineageRelation.RELATED_TO,
    )

    engine.connect(
        "b",
        "a",
        LineageRelation.RELATED_TO,
        allow_cycle=True,
    )

    assert engine.edge_count() == 2


# ============================================================
# Navigation
# ============================================================


def test_outgoing_edges() -> None:
    engine = build_chain()

    edges = engine.outgoing_edges(
        "run_001"
    )

    assert len(edges) == 1
    assert (
        edges[0].target_id
        == "artifact_001"
    )


def test_incoming_edges() -> None:
    engine = build_chain()

    edges = engine.incoming_edges(
        "run_001"
    )

    assert len(edges) == 2


def test_children() -> None:
    engine = build_chain()

    children = engine.children(
        "run_001"
    )

    assert [
        node.node_id
        for node in children
    ] == ["artifact_001"]


def test_parents() -> None:
    engine = build_chain()

    parents = engine.parents(
        "run_001"
    )

    assert {
        node.node_id
        for node in parents
    } == {
        "dataset_001",
        "model_v1",
    }


def test_descendants() -> None:
    engine = LineageEngine()

    for node_id in (
        "dataset",
        "run",
        "artifact",
        "report",
    ):
        make_node(engine, node_id)

    engine.connect(
        "dataset",
        "run",
        LineageRelation.USES,
    )
    engine.connect(
        "run",
        "artifact",
        LineageRelation.PRODUCES,
    )
    engine.connect(
        "artifact",
        "report",
        LineageRelation.DERIVED_FROM,
    )

    descendants = engine.descendants(
        "dataset"
    )

    assert [
        node.node_id
        for node in descendants
    ] == [
        "run",
        "artifact",
        "report",
    ]


def test_ancestors() -> None:
    engine = LineageEngine()

    for node_id in (
        "dataset",
        "run",
        "artifact",
    ):
        make_node(engine, node_id)

    engine.connect(
        "dataset",
        "run",
        LineageRelation.USES,
    )
    engine.connect(
        "run",
        "artifact",
        LineageRelation.PRODUCES,
    )

    ancestors = engine.ancestors(
        "artifact"
    )

    assert [
        node.node_id
        for node in ancestors
    ] == [
        "run",
        "dataset",
    ]


# ============================================================
# Paths
# ============================================================


def test_find_path() -> None:
    engine = LineageEngine()

    for node_id in (
        "dataset",
        "run",
        "artifact",
    ):
        make_node(engine, node_id)

    engine.connect(
        "dataset",
        "run",
        LineageRelation.USES,
    )
    engine.connect(
        "run",
        "artifact",
        LineageRelation.PRODUCES,
    )

    path = engine.find_path(
        "dataset",
        "artifact",
    )

    assert path is not None
    assert path.node_ids == [
        "dataset",
        "run",
        "artifact",
    ]
    assert path.length == 2


def test_find_path_same_node() -> None:
    engine = LineageEngine()

    make_node(engine, "run")

    path = engine.find_path(
        "run",
        "run",
    )

    assert path is not None
    assert path.node_ids == ["run"]
    assert path.length == 0


def test_find_path_none() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    assert engine.find_path(
        "a",
        "b",
    ) is None


def test_find_path_uses_shortest_path() -> None:
    engine = LineageEngine()

    for node_id in (
        "a",
        "b",
        "c",
        "d",
    ):
        make_node(engine, node_id)

    engine.connect(
        "a",
        "b",
        LineageRelation.RELATED_TO,
    )
    engine.connect(
        "b",
        "c",
        LineageRelation.RELATED_TO,
    )
    engine.connect(
        "c",
        "d",
        LineageRelation.RELATED_TO,
    )
    engine.connect(
        "a",
        "d",
        LineageRelation.DERIVED_FROM,
    )

    path = engine.find_path(
        "a",
        "d",
    )

    assert path is not None
    assert path.node_ids == ["a", "d"]
    assert path.length == 1


# ============================================================
# Roots / leaves
# ============================================================


def test_roots() -> None:
    engine = build_chain()

    assert {
        node.node_id
        for node in engine.roots()
    } == {
        "dataset_001",
        "model_v1",
    }


def test_leaves() -> None:
    engine = build_chain()

    assert [
        node.node_id
        for node in engine.leaves()
    ] == ["artifact_001"]


def test_isolated_node_is_root_and_leaf() -> None:
    engine = LineageEngine()

    node = make_node(
        engine,
        "isolated",
    )

    assert engine.roots() == [node]
    assert engine.leaves() == [node]


# ============================================================
# Fingerprints
# ============================================================


def test_lineage_fingerprint_deterministic() -> None:
    first = build_chain()
    second = build_chain()

    assert (
        first.lineage_fingerprint(
            "artifact_001"
        )
        == second.lineage_fingerprint(
            "artifact_001"
        )
    )


def test_lineage_fingerprint_changes_with_parent() -> None:
    engine = build_chain()

    before = engine.lineage_fingerprint(
        "artifact_001"
    )

    make_node(
        engine,
        "policy_v1",
        LineageNodeType.POLICY_VERSION,
    )

    engine.connect(
        "policy_v1",
        "artifact_001",
        LineageRelation.DEPENDS_ON,
    )

    after = engine.lineage_fingerprint(
        "artifact_001"
    )

    assert before != after


def test_graph_fingerprint_deterministic() -> None:
    first = build_chain()
    second = build_chain()

    assert (
        first.full_graph_fingerprint()
        == second.full_graph_fingerprint()
    )


def test_graph_fingerprint_changes() -> None:
    engine = build_chain()

    before = (
        engine.full_graph_fingerprint()
    )

    make_node(
        engine,
        "experiment_001",
        LineageNodeType.EXPERIMENT,
    )

    after = (
        engine.full_graph_fingerprint()
    )

    assert before != after


# ============================================================
# Audit
# ============================================================


def test_node_creation_audited() -> None:
    engine = LineageEngine()

    make_node(engine, "run")

    assert any(
        event["event"]
        == "LINEAGE_NODE_ADDED"
        for event in engine.audit_log()
    )


def test_edge_creation_audited() -> None:
    engine = LineageEngine()

    make_node(engine, "a")
    make_node(engine, "b")

    engine.connect(
        "a",
        "b",
        LineageRelation.USES,
    )

    assert any(
        event["event"]
        == "LINEAGE_EDGE_ADDED"
        for event in engine.audit_log()
    )


def test_duplicate_node_not_audited_twice() -> None:
    engine = LineageEngine()

    make_node(
        engine,
        "run",
        LineageNodeType.RUN,
    )

    engine.create_node(
        node_id="run",
        node_type=LineageNodeType.RUN,
    )

    events = [
        event
        for event in engine.audit_log()
        if (
            event["event"]
            == "LINEAGE_NODE_ADDED"
        )
    ]

    assert len(events) == 1


# ============================================================
# Snapshot
# ============================================================


def test_empty_snapshot() -> None:
    engine = LineageEngine()

    snapshot = engine.snapshot()

    assert snapshot.total_nodes == 0
    assert snapshot.total_edges == 0
    assert snapshot.root_nodes == 0
    assert snapshot.leaf_nodes == 0


def test_snapshot_counts() -> None:
    engine = build_chain()

    snapshot = engine.snapshot()

    assert snapshot.total_nodes == 4
    assert snapshot.total_edges == 3
    assert snapshot.root_nodes == 2
    assert snapshot.leaf_nodes == 1


def test_snapshot_node_type_counts() -> None:
    engine = build_chain()

    snapshot = engine.snapshot()

    assert (
        snapshot.node_type_counts[
            "DATASET"
        ]
        == 1
    )
    assert (
        snapshot.node_type_counts[
            "MODEL_VERSION"
        ]
        == 1
    )
    assert (
        snapshot.node_type_counts[
            "RUN"
        ]
        == 1
    )
    assert (
        snapshot.node_type_counts[
            "ARTIFACT"
        ]
        == 1
    )


def test_snapshot_relation_counts() -> None:
    engine = build_chain()

    snapshot = engine.snapshot()

    assert (
        snapshot.relation_counts["USES"]
        == 2
    )
    assert (
        snapshot.relation_counts[
            "PRODUCES"
        ]
        == 1
    )


def test_snapshot_contains_graph_fingerprint() -> None:
    engine = build_chain()

    snapshot = engine.snapshot()

    assert len(
        snapshot.metadata[
            "graph_fingerprint"
        ]
    ) == 64


def test_snapshot_safety_metadata() -> None:
    engine = LineageEngine()

    snapshot = engine.snapshot()

    assert snapshot.metadata["research_only"] is True
    assert (
        snapshot.metadata["real_money_execution"]
        is False
    )


# ============================================================
# Reports
# ============================================================


def test_node_report() -> None:
    engine = build_chain()

    report = engine.node_report(
        "run_001"
    )

    assert report["node_id"] == "run_001"
    assert report["node_type"] == "RUN"

    assert set(
        report["parent_ids"]
    ) == {
        "dataset_001",
        "model_v1",
    }

    assert report["child_ids"] == [
        "artifact_001"
    ]

    assert len(
        report["lineage_fingerprint"]
    ) == 64

    assert report["research_only"] is True
    assert (
        report["real_money_execution"]
        is False
    )


def test_summary_empty() -> None:
    engine = LineageEngine()

    summary = engine.summary()

    assert (
        summary["component"]
        == "lineage_engine"
    )
    assert summary["version"] == "v6.0.1"
    assert summary["total_nodes"] == 0
    assert summary["total_edges"] == 0
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"]
        is False
    )


def test_summary_chain() -> None:
    engine = build_chain()

    summary = engine.summary()

    assert summary["total_nodes"] == 4
    assert summary["total_edges"] == 3
    assert summary["root_nodes"] == 2
    assert summary["leaf_nodes"] == 1
    assert len(
        summary["graph_fingerprint"]
    ) == 64


# ============================================================
# Full institutional lineage
# ============================================================


def test_full_experiment_artifact_lineage() -> None:
    engine = LineageEngine()

    engine.create_node(
        node_id="dataset_prices_v3",
        node_type=LineageNodeType.DATASET,
        fingerprint="prices_fingerprint",
    )

    engine.create_node(
        node_id="dataset_macro_v2",
        node_type=LineageNodeType.DATASET,
        fingerprint="macro_fingerprint",
    )

    engine.create_node(
        node_id="model_crypto_v4",
        node_type=(
            LineageNodeType.MODEL_VERSION
        ),
    )

    engine.create_node(
        node_id="prompt_research_v7",
        node_type=(
            LineageNodeType.PROMPT_VERSION
        ),
    )

    engine.create_node(
        node_id="policy_risk_v2",
        node_type=(
            LineageNodeType.POLICY_VERSION
        ),
    )

    engine.create_node(
        node_id="run_btc_001",
        node_type=LineageNodeType.RUN,
    )

    engine.create_node(
        node_id="experiment_btc_001",
        node_type=LineageNodeType.EXPERIMENT,
    )

    engine.create_node(
        node_id="artifact_backtest_001",
        node_type=LineageNodeType.ARTIFACT,
    )

    engine.create_node(
        node_id="artifact_report_001",
        node_type=LineageNodeType.ARTIFACT,
    )

    engine.connect(
        "dataset_prices_v3",
        "run_btc_001",
        LineageRelation.USES,
    )

    engine.connect(
        "dataset_macro_v2",
        "run_btc_001",
        LineageRelation.USES,
    )

    engine.connect(
        "model_crypto_v4",
        "run_btc_001",
        LineageRelation.USES,
    )

    engine.connect(
        "prompt_research_v7",
        "run_btc_001",
        LineageRelation.USES,
    )

    engine.connect(
        "policy_risk_v2",
        "run_btc_001",
        LineageRelation.USES,
    )

    engine.connect(
        "run_btc_001",
        "experiment_btc_001",
        LineageRelation.PRODUCES,
    )

    engine.connect(
        "experiment_btc_001",
        "artifact_backtest_001",
        LineageRelation.PRODUCES,
    )

    engine.connect(
        "artifact_backtest_001",
        "artifact_report_001",
        LineageRelation.DERIVED_FROM,
    )

    path = engine.find_path(
        "dataset_prices_v3",
        "artifact_report_001",
    )

    assert path is not None

    assert path.node_ids == [
        "dataset_prices_v3",
        "run_btc_001",
        "experiment_btc_001",
        "artifact_backtest_001",
        "artifact_report_001",
    ]

    assert path.length == 4

    ancestors = {
        node.node_id
        for node in engine.ancestors(
            "artifact_report_001"
        )
    }

    assert ancestors == {
        "dataset_prices_v3",
        "dataset_macro_v2",
        "model_crypto_v4",
        "prompt_research_v7",
        "policy_risk_v2",
        "run_btc_001",
        "experiment_btc_001",
        "artifact_backtest_001",
    }

    snapshot = engine.snapshot()

    assert snapshot.total_nodes == 9
    assert snapshot.total_edges == 8
    assert snapshot.root_nodes == 5
    assert snapshot.leaf_nodes == 1

    report = engine.node_report(
        "artifact_report_001"
    )

    assert len(
        report["lineage_fingerprint"]
    ) == 64