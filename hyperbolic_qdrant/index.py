"""Create a Qdrant collection for transformed hyperbolic vectors and upload.

Qdrant is not modified and does not need to know anything about hyperbolic
geometry. The transform is a preprocessing layer; what lands in the collection
is an ordinary unit-norm float vector searched with Distance.DOT.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from .transform import HyperbolicTransform, stored_dim

__all__ = ["create_collection", "upload", "build"]


def create_collection(client: QdrantClient, name: str, ball_dim: int, *,
                      m: int = 16, ef_construct: int = 200,
                      recreate: bool = False) -> None:
    """Create the collection. Distance.DOT is required, not a preference.

    The transform is derived from an inner product. Cosine would renormalise the
    padded vectors and break the identity; Euclid would reintroduce the depth
    term the transform exists to remove.
    """
    if client.collection_exists(name):
        if not recreate:
            raise ValueError(
                "collection %r already exists. Pass recreate=True to replace it."
                % name)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(
            size=stored_dim(ball_dim), distance=models.Distance.DOT
        ),
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
    )


def upload(client: QdrantClient, name: str, transformed: np.ndarray, *,
           payloads: Sequence[dict[str, Any]] | None = None,
           ids: Iterable[int] | None = None, batch_size: int = 1000) -> None:
    """Upload already-transformed vectors."""
    v = np.asarray(transformed, dtype=np.float64)
    if payloads is not None and len(payloads) != len(v):
        raise ValueError("payloads length %d does not match vectors %d"
                         % (len(payloads), len(v)))
    idx = list(ids) if ids is not None else list(range(len(v)))
    for s in range(0, len(v), batch_size):
        e = min(s + batch_size, len(v))
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=idx[i],
                    vector=[float(x) for x in v[i]],
                    payload=(payloads[i] if payloads is not None else None),
                )
                for i in range(s, e)
            ],
            wait=True,
        )


def build(client: QdrantClient, name: str, documents: np.ndarray, *,
          payloads: Sequence[dict[str, Any]] | None = None,
          m: int = 16, ef_construct: int = 200,
          recreate: bool = False) -> HyperbolicTransform:
    """Fit, create, upload. Returns the transform -- you need it for queries.

    The transform's ``scale`` is fitted to these documents. Keep it with the
    collection; a query transformed under a different scale will rank wrongly.
    """
    docs = np.asarray(documents, dtype=np.float64)
    transform = HyperbolicTransform().fit(docs)
    create_collection(client, name, ball_dim=docs.shape[1],
                      m=m, ef_construct=ef_construct, recreate=recreate)
    upload(client, name, transform.transform_documents(docs), payloads=payloads)
    return transform
