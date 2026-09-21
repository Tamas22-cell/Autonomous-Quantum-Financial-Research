"""
v7.0.1 Data Fabric integration layer.

Research-only architecture.
No real-money execution is supported.
"""

from .facade import DataFabric, DataFabricError

__version__ = "v7.0.1"

__all__ = [
    "DataFabric",
    "DataFabricError",
    "__version__",
]