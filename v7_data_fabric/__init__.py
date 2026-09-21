"""
v7.0.1 Data Fabric and v7.0.2 Knowledge Graph integration.

Research-only architecture.
No real-money execution is supported.
"""

from .facade import DataFabric, DataFabricError
from .knowledge_graph_bridge import (
    KnowledgeGraphBridge,
    KnowledgeGraphBridgeError,
)

__version__ = "v7.0.2"

__all__ = [
    "DataFabric",
    "DataFabricError",
    "KnowledgeGraphBridge",
    "KnowledgeGraphBridgeError",
    "__version__",
]