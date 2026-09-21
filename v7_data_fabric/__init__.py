"""
v7.0.1 Data Fabric, v7.0.2 Knowledge Graph,
v7.0.3 Research Relationships, and
v7.0.4 Research Event Lineage.

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

from .lineage_service import (
    LineageServiceError,
    ResearchLineageService,
)

__version__ = "v7.0.4"

__all__ = [
    "DataFabric",
    "DataFabricError",
    "KnowledgeGraphBridge",
    "KnowledgeGraphBridgeError",
    "RelationshipApprovalRequiredError",
    "RelationshipManagerError",
    "ResearchRelationshipManager",
    "LineageServiceError",
    "ResearchLineageService",
    "__version__",
]