
from datetime import datetime, timedelta, timezone

import pytest

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
    GraphStoreError,
)
from financial_knowledge_graph.models import (
    Evidence,
    EvidenceType,
    HypothesisStatus,
    KnowledgeGraphValidationError,
    KnowledgeNode,
    KnowledgeRelationship,
    NodeType,
    RelationshipType,
    ValidityPeriod,
)
from financial_knowledge_graph.path_finder import (
    FinancialGraphPathFinder,
    GraphPath,
)


NOW = datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def graph():
    store = FinancialKnowledgeGraphStore()

    nodes = (
        ("macro", NodeType.MACRO_EVENT),
        ("mechanism", NodeType.ECONOMIC_MECHANISM),
        ("market", NodeType.MARKET_VARIABLE),
        ("asset", NodeType.ASSET),
        ("portfolio", NodeType.PORTFOLIO),
        ("outcome", NodeType.OUTCOME),
        ("alternative", NodeType.MARKET_VARIABLE),
        ("isolated", NodeType.ASSET),
    )

    for node_id, node_type in nodes:
        store.add_node(
            KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                label=node_id,
            )
        )

    for evidence_id in ("e1", "e2", "e3", "e4", "e5"):
        store.add_evidence(
            Evidence(
                evidence_id=evidence_id,
                evidence_type=EvidenceType.AGENT_RESEARCH,
                source="research-agent",
                summary=f"Research evidence {evidence_id}",
                observed_at=NOW,
            )
        )

    return store


def add_edge(
    graph,
    relationship_id,
    source,
    target,
    *,
    evidence_ids=(),
    status=HypothesisStatus.PROPOSED,
    valid_from=NOW,
    valid_until=None,
):
    relationship = KnowledgeRelationship(
        relationship_id=relationship_id,
        source_node_id=source,
        target_node_id=target,
        relationship_type=RelationshipType.INFLUENCES,
        confidence=0.7,
        validity=ValidityPeriod(
            valid_from=valid_from,
            valid_until=valid_until,
        ),
        evidence_ids=evidence_ids,
        hypothesis_status=status,
    )

    graph.add_relationship(relationship)
    return relationship


def test_path_finder_requires_graph():
    with pytest.raises(GraphStoreError):
        FinancialGraphPathFinder("not-a-graph")


def test_empty_graph_has_no_path(graph):
    finder = FinancialGraphPathFinder(graph)

    assert finder.find_paths("macro", "outcome") == ()


def test_find_direct_path(graph):
    add_edge(
        graph,
        "r1",
        "macro",
        "mechanism",
        evidence_ids=("e1",),
    )

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "mechanism",
    )

    assert len(paths) == 1
    assert paths[0].node_ids == ("macro", "mechanism")
    assert paths[0].relationship_ids == ("r1",)
    assert paths[0].evidence_ids == ("e1",)
    assert paths[0].length == 1


def test_find_full_financial_chain(graph):
    edges = (
        ("r1", "macro", "mechanism"),
        ("r2", "mechanism", "market"),
        ("r3", "market", "asset"),
        ("r4", "asset", "portfolio"),
        ("r5", "portfolio", "outcome"),
    )

    for relationship_id, source, target in edges:
        add_edge(
            graph,
            relationship_id,
            source,
            target,
        )

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert len(paths) == 1
    assert paths[0].node_ids == (
        "macro",
        "mechanism",
        "market",
        "asset",
        "portfolio",
        "outcome",
    )
    assert paths[0].relationship_ids == (
        "r1",
        "r2",
        "r3",
        "r4",
        "r5",
    )
    assert paths[0].length == 5


def test_direction_is_respected(graph):
    add_edge(graph, "r1", "macro", "mechanism")

    finder = FinancialGraphPathFinder(graph)

    assert finder.find_paths("macro", "mechanism")
    assert finder.find_paths("mechanism", "macro") == ()


def test_no_path_to_isolated_node(graph):
    add_edge(graph, "r1", "macro", "mechanism")

    assert FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "isolated",
    ) == ()


def test_same_source_and_target_returns_no_path(graph):
    assert FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "macro",
    ) == ()


def test_unknown_source_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown node ID",
    ):
        FinancialGraphPathFinder(graph).find_paths(
            "missing",
            "outcome",
        )


def test_unknown_target_raises_error(graph):
    with pytest.raises(
        GraphStoreError,
        match="Unknown node ID",
    ):
        FinancialGraphPathFinder(graph).find_paths(
            "macro",
            "missing",
        )


def test_max_depth_excludes_longer_path(graph):
    add_edge(graph, "r1", "macro", "mechanism")
    add_edge(graph, "r2", "mechanism", "market")

    finder = FinancialGraphPathFinder(graph)

    assert finder.find_paths(
        "macro",
        "market",
        max_depth=1,
    ) == ()

    assert len(
        finder.find_paths(
            "macro",
            "market",
            max_depth=2,
        )
    ) == 1


@pytest.mark.parametrize(
    "max_depth",
    [0, -1, 1.5, True, "3"],
)
def test_invalid_max_depth_raises_error(
    graph,
    max_depth,
):
    with pytest.raises(GraphStoreError):
        FinancialGraphPathFinder(graph).find_paths(
            "macro",
            "outcome",
            max_depth=max_depth,
        )


@pytest.mark.parametrize(
    "max_paths",
    [0, -1, 1.5, True, "3"],
)
def test_invalid_max_paths_raises_error(
    graph,
    max_paths,
):
    with pytest.raises(GraphStoreError):
        FinancialGraphPathFinder(graph).find_paths(
            "macro",
            "outcome",
            max_paths=max_paths,
        )


def test_valid_at_filters_out_future_relationship(graph):
    add_edge(
        graph,
        "r1",
        "macro",
        "mechanism",
        valid_from=NOW + timedelta(days=1),
    )

    finder = FinancialGraphPathFinder(graph)

    assert finder.find_paths(
        "macro",
        "mechanism",
        valid_at=NOW,
    ) == ()

    assert len(
        finder.find_paths(
            "macro",
            "mechanism",
            valid_at=NOW + timedelta(days=1),
        )
    ) == 1


def test_valid_at_filters_out_expired_relationship(graph):
    add_edge(
        graph,
        "r1",
        "macro",
        "mechanism",
        valid_from=NOW - timedelta(days=2),
        valid_until=NOW - timedelta(days=1),
    )

    finder = FinancialGraphPathFinder(graph)

    assert finder.find_paths(
        "macro",
        "mechanism",
        valid_at=NOW,
    ) == ()

    assert len(
        finder.find_paths(
            "macro",
            "mechanism",
            valid_at=NOW - timedelta(days=2),
        )
    ) == 1


def test_valid_at_rejects_naive_datetime(graph):
    with pytest.raises(KnowledgeGraphValidationError):
        FinancialGraphPathFinder(graph).find_paths(
            "macro",
            "outcome",
            valid_at=datetime(2026, 9, 20),
        )


def test_path_does_not_revisit_nodes(graph):
    add_edge(graph, "r1", "macro", "mechanism")
    add_edge(graph, "r2", "mechanism", "market")
    add_edge(graph, "r3", "market", "macro")
    add_edge(graph, "r4", "market", "outcome")

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert len(paths) == 1
    assert paths[0].node_ids == (
        "macro",
        "mechanism",
        "market",
        "outcome",
    )
    assert len(paths[0].node_ids) == len(
        set(paths[0].node_ids)
    )


def test_multiple_paths_are_sorted_by_length(graph):
    add_edge(graph, "r1", "macro", "mechanism")
    add_edge(graph, "r2", "mechanism", "outcome")
    add_edge(graph, "r3", "macro", "outcome")

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert len(paths) == 2
    assert paths[0].relationship_ids == ("r3",)
    assert paths[1].relationship_ids == ("r1", "r2")


def test_equal_length_paths_are_sorted_by_relationship_ids(
    graph,
):
    add_edge(graph, "z1", "macro", "mechanism")
    add_edge(graph, "z2", "mechanism", "outcome")
    add_edge(graph, "a1", "macro", "market")
    add_edge(graph, "a2", "market", "outcome")

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert tuple(
        path.relationship_ids for path in paths
    ) == (
        ("a1", "a2"),
        ("z1", "z2"),
    )


def test_max_paths_limits_results(graph):
    add_edge(graph, "r1", "macro", "outcome")
    add_edge(graph, "r2", "macro", "mechanism")
    add_edge(graph, "r3", "mechanism", "outcome")

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
        max_paths=1,
    )

    assert len(paths) == 1
    assert paths[0].relationship_ids == ("r1",)


def test_evidence_ids_are_deduplicated_in_path(graph):
    add_edge(
        graph,
        "r1",
        "macro",
        "mechanism",
        evidence_ids=("e1", "e2"),
    )
    add_edge(
        graph,
        "r2",
        "mechanism",
        "outcome",
        evidence_ids=("e2", "e3"),
    )

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert paths[0].evidence_ids == (
        "e1",
        "e2",
        "e3",
    )


def test_path_preserves_hypothesis_statuses(graph):
    add_edge(
        graph,
        "r1",
        "macro",
        "mechanism",
        status=HypothesisStatus.PROPOSED,
    )
    add_edge(
        graph,
        "r2",
        "mechanism",
        "outcome",
        status=HypothesisStatus.CONTRADICTED,
    )

    paths = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert paths[0].hypothesis_statuses == (
        HypothesisStatus.PROPOSED,
        HypothesisStatus.CONTRADICTED,
    )


def test_graph_path_length_property():
    path = GraphPath(
        node_ids=("macro", "mechanism", "outcome"),
        relationship_ids=("r1", "r2"),
        evidence_ids=("e1",),
        hypothesis_statuses=(
            HypothesisStatus.PROPOSED,
            HypothesisStatus.SUPPORTED,
        ),
    )

    assert path.length == 2


def test_find_paths_returns_tuple(graph):
    add_edge(graph, "r1", "macro", "outcome")

    result = FinancialGraphPathFinder(graph).find_paths(
        "macro",
        "outcome",
    )

    assert isinstance(result, tuple)
    assert isinstance(result[0], GraphPath)