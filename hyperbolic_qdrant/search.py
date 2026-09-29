"""Search a transformed collection. This is an ordinary Qdrant query."""

from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models

from .transform import HyperbolicTransform

__all__ = ["search"]


def search(client: QdrantClient, name: str, query: np.ndarray,
           transform: HyperbolicTransform, *, limit: int = 10,
           hnsw_ef: int | None = None,
           query_filter: models.Filter | None = None) -> list[Any]:
    """Transform the query, then hand it to Qdrant unchanged.

    Nothing hyperbolic happens on the server. The returned order is the
    hyperbolic nearest-neighbour order, up to HNSW's usual approximation.
    """
    q = transform.transform_query(np.asarray(query, dtype=np.float64))
    params = models.SearchParams(hnsw_ef=hnsw_ef) if hnsw_ef else None
    return client.query_points(
        collection_name=name,
        query=[float(x) for x in np.ravel(q)],
        limit=limit,
        query_filter=query_filter,
        search_params=params,
    ).points
