"""
v7.0.1 Data Fabric, v7.0.2 Knowledge Graph,
and v7.0.3 Research Relationship integration.

Research-only architecture.
No real-money execution is supported.
"""

from .facade import (
    DataFabric,
    DataFabricError,
)

from .knowledge_graph_bridge import (
    KnowledgeGraphBridge,
    KnowledgeGraphBridgeError,
)

from .relationship_manager import (
    RelationshipApprovalRequiredError,
    RelationshipManagerError,
    ResearchRelationshipManager,
)

__version__ = "v7.0.3"

__all__ = [
    "DataFabric",
    "DataFabricError",
    "KnowledgeGraphBridge",
    "KnowledgeGraphBridgeError",
    "RelationshipApprovalRequiredError",
    "RelationshipManagerError",
    "ResearchRelationshipManager",
    "__version__",
]