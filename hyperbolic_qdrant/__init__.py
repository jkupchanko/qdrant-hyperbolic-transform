"""Index deep hyperbolic (Poincare-ball) embeddings in Qdrant with ordinary HNSW.

The transform is a preprocessing layer. Qdrant is unmodified and needs no
knowledge of hyperbolic geometry: it stores unit-norm float vectors and searches
them with Distance.DOT.
"""
from .transform import (HyperbolicTransform, transform_documents, transform_query,
                        poincare_distance, stored_dim)
from .depth import measure_depth, DepthReport
from .index import build, create_collection, upload
from .search import search
from .benchmark import benchmark, BenchmarkResult, ArmResult

__version__ = "0.1.0"
__all__ = ["HyperbolicTransform", "transform_documents", "transform_query",
           "poincare_distance", "stored_dim", "measure_depth", "DepthReport",
           "build", "create_collection", "upload", "search",
           "benchmark", "BenchmarkResult", "ArmResult"]
