"""The transform itself.

A Poincare-ball point cannot be indexed directly by HNSW when the embedding is
deep: every point sits at norm ~0.9999, so Euclidean distances between stored
points are dominated by a depth-difference term rather than by the hyperbolic
distance you care about, and the graph links the wrong neighbours.

This module maps documents and queries into a space where an ordinary dot
product reproduces the hyperbolic nearest-neighbour ordering exactly.

The identity
------------
Poincare distance:

    d(u, v) = arcosh(1 + 2 * ||u - v||^2 / ((1 - ||u||^2) * (1 - ||v||^2)))

Let w_v = 1 / (1 - ||v||^2). Using w_v * ||v||^2 = w_v - 1, minimising d over v
for a fixed query u is equivalent to minimising

    w_v * (1 + ||u||^2) - 2 * (w_v * v) . u

which is the inner product of

    phi(v) = [w_v, w_v * v]           (document)
    psi(u) = [1 + ||u||^2, -2u]       (query)

So the hyperbolic nearest neighbour is the document MINIMISING <phi(v), psi(u)>.
Qdrant's Dot distance returns the largest score first, so the query is negated.
Getting that sign wrong yields recall 0.0000, not a degraded result.

Constant-norm padding
---------------------
phi has an unbounded norm as points approach the boundary. phi is first divided by a high
quantile of the document row-norms seen at fit time, then one extra coordinate,
sqrt(1 - ||phi_scaled||^2), pads every document to unit norm. This does not
change the ordering induced by the dot product with a query whose pad slot is
zero, and it keeps the stored vectors well conditioned for graph construction.
"""

from __future__ import annotations

import numpy as np

__all__ = ["HyperbolicTransform", "transform_documents", "transform_query"]

_EPS = 1e-12


def _as_2d(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def _check_ball(v: np.ndarray) -> np.ndarray:
    """Norms must be < 1. Raise rather than silently clipping."""
    n = np.linalg.norm(v, axis=1)
    if not np.all(np.isfinite(v)):
        raise ValueError("vectors contain NaN or inf")
    bad = n >= 1.0
    if bad.any():
        raise ValueError(
            "%d of %d vectors have norm >= 1 and are not inside the Poincare ball "
            "(max %.6f). Rescale before transforming." % (bad.sum(), len(n), n.max())
        )
    return n


class HyperbolicTransform:
    """Fit on your documents, then transform documents and queries.

    The scale factor is learned from the documents so the padded representation
    stays inside the unit ball. Queries reuse it, so you must keep the fitted
    object (or its ``scale``) alongside the collection.
    """

    def __init__(self, scale: float | None = None, quantile: float = 0.99) -> None:
        self.scale = scale
        self.quantile = quantile

    def fit(self, documents: np.ndarray) -> "HyperbolicTransform":
        v = _as_2d(documents)
        n = _check_ball(v)
        w = 1.0 / np.maximum(1.0 - n ** 2, _EPS)
        phi = np.concatenate([w[:, None], w[:, None] * v], axis=1)
        # A QUANTILE of the row norms, not the max. One anomalously deep point
        # otherwise sets the scale for the whole corpus and crushes every other
        # document toward the pad axis, where HNSW cannot tell them apart.
        # Measured: scaling by the max gave recall 0.0025 on one corpus and
        # 0.9988 at quantile 0.99. It improved every corpus tested, not just the
        # pathological ones. The cost is that documents above the quantile are
        # clipped and are no longer ranked exactly -- see transform_documents.
        self.scale = float(np.quantile(np.linalg.norm(phi, axis=1), self.quantile))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("degenerate scale; check the input vectors")
        return self

    def transform_documents(self, documents: np.ndarray) -> np.ndarray:
        """Poincare points -> unit-norm vectors to store in Qdrant."""
        if self.scale is None:
            raise RuntimeError("call fit() first, or pass scale= to the constructor")
        v = _as_2d(documents)
        n = _check_ball(v)
        w = 1.0 / np.maximum(1.0 - n ** 2, _EPS)
        phi = np.concatenate([w[:, None], w[:, None] * v], axis=1) / self.scale
        # Documents past the fitted quantile overflow the unit ball. They are
        # projected back onto it, which makes their ranking approximate rather
        # than exact. That is the deliberate trade: scaling by the max keeps
        # every document exact but produces a graph nobody can search.
        over = np.linalg.norm(phi, axis=1) > 1.0
        pad = np.sqrt(np.clip(1.0 - (phi ** 2).sum(axis=1), 0.0, None))
        out = np.concatenate([phi, pad[:, None]], axis=1)
        if over.any():
            out[over] /= np.linalg.norm(out[over], axis=1, keepdims=True)
        return out

    def clipped_fraction(self, documents: np.ndarray) -> float:
        """Share of documents ranked approximately rather than exactly."""
        v = _as_2d(documents); n = _check_ball(v)
        w = 1.0 / np.maximum(1.0 - n ** 2, _EPS)
        phi = np.concatenate([w[:, None], w[:, None] * v], axis=1) / self.scale
        return float((np.linalg.norm(phi, axis=1) > 1.0).mean())

    def transform_query(self, query: np.ndarray) -> np.ndarray:
        """Poincare point -> the vector to send to Qdrant with Distance.DOT.

        Negated, because the maths minimises the inner product and Dot maximises.
        """
        u = _as_2d(query)
        nu = _check_ball(u)
        psi = np.concatenate(
            [-(1.0 + nu ** 2)[:, None], 2.0 * u, np.zeros((len(u), 1))], axis=1
        )
        return psi[0] if np.asarray(query).ndim == 1 else psi


def stored_dim(ball_dim: int) -> int:
    """A d-dimensional Poincare ball becomes d + 2 stored coordinates."""
    return ball_dim + 2


def transform_documents(documents: np.ndarray) -> tuple[np.ndarray, HyperbolicTransform]:
    """Convenience: fit and transform in one call. Returns (vectors, transform)."""
    t = HyperbolicTransform().fit(documents)
    return t.transform_documents(documents), t


def transform_query(query: np.ndarray, transform: HyperbolicTransform) -> np.ndarray:
    """Convenience wrapper so the query path reads symmetrically."""
    return transform.transform_query(query)


def poincare_distance(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Exact Poincare geodesic distance, for ground truth and verification."""
    u = _as_2d(u); v = _as_2d(v)
    nu = np.linalg.norm(u, axis=1) ** 2
    nv = np.linalg.norm(v, axis=1) ** 2
    sq = ((u[:, None, :] - v[None, :, :]) ** 2).sum(-1)
    den = np.maximum((1 - nu)[:, None] * (1 - nv)[None, :], _EPS)
    return np.arccosh(1.0 + np.clip(2.0 * sq / den, 0.0, None))
