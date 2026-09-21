from datetime import datetime, timezone

import pytest

from financial_knowledge_graph.graph_store import (
    FinancialKnowledgeGraphStore,
)
from financial_knowledge_graph.models import (
    KnowledgeNode,
    NodeType,
    RelationshipType,
)

from v7_data_fabric.relationship_manager import (
    RelationshipApprovalRequiredError,
    ResearchRelationshipManager,
)


def build_manager() -> ResearchRelationshipManager:
    store = FinancialKnowledgeGraphStore()

    store.add_node(
        KnowledgeNode(
            node_id="asset:btc",
            node_type=NodeType.ASSET,
            label="Bitcoin",
        )
    )

    store.add_node(
        KnowledgeNode(
            node_id="variable:liquidity",
            node_type=NodeType.MARKET_VARIABLE,
            label="Global Liquidity",
        )
    )

    return ResearchRelationshipManager(
        store=store
    )


def test_manager_version_and_safety_flags() -> None:
    manager = build_manager()

    assert manager.VERSION == "v7.0.3"
    assert manager.RESEARCH_ONLY is True
    assert manager.REAL_MONEY_EXECUTION is False


def test_relationship_requires_explicit_approval() -> None:
    manager = build_manager()

    with pytest.raises(
        RelationshipApprovalRequiredError
    ):
        manager.add_relationship(
            relationship_id="rel:liquidity-btc",
            source_node_id="variable:liquidity",
            target_node_id="asset:btc",
            relationship_type=(
                RelationshipType.INFLUENCES
            ),
            confidence=0.85,
            valid_from=datetime.now(
                timezone.utc
            ),
            rationale=(
                "Liquidity conditions may influence "
                "Bitcoin market behavior."
            ),
            approved=False,
        )


def test_approved_relationship_is_stored() -> None:
    manager = build_manager()

    relationship = manager.add_relationship(
        relationship_id="rel:liquidity-btc",
        source_node_id="variable:liquidity",
        target_node_id="asset:btc",
        relationship_type=RelationshipType.INFLUENCES,
        confidence=0.85,
        valid_from=datetime.now(
            timezone.utc
        ),
        rationale=(
            "Liquidity conditions may influence "
            "Bitcoin market behavior."
        ),
        approved=True,
    )

    assert (
        relationship.relationship_id
        == "rel:liquidity-btc"
    )
    assert (
        relationship.source_node_id
        == "variable:liquidity"
    )
    assert (
        relationship.target_node_id
        == "asset:btc"
    )
    assert manager.relationship_count() == 1
    assert (
        manager.get_relationship(
            "rel:liquidity-btc"
        )
        == relationship
    )


def test_relationship_summary() -> None:
    manager = build_manager()

    manager.add_relationship(
        relationship_id="rel:liquidity-btc",
        source_node_id="variable:liquidity",
        target_node_id="asset:btc",
        relationship_type=RelationshipType.INFLUENCES,
        confidence=0.85,
        valid_from=datetime.now(
            timezone.utc
        ),
        approved=True,
    )

    summary = manager.summary()

    assert (
        summary["component"]
        == "research_relationship_manager"
    )
    assert summary["version"] == "v7.0.3"
    assert summary["research_only"] is True
    assert (
        summary["real_money_execution"] is False
    )
    assert summary["relationship_count"] == 1