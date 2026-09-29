"""Tests that pin the things that have actually gone wrong.

The sign of the query transform is the one that matters. Getting it backwards
gives recall 0.0000, not a degraded result, and it looks like a plumbing bug
rather than a maths bug. That mistake was made once already.
"""

from __future__ import annotations

import numpy as np
import pytest

from hyperbolic_qdrant import (HyperbolicTransform, measure_depth,
                               poincare_distance, stored_dim)


def deep_ball(n: int = 400, dim: int = 5, seed: int = 0) -> np.ndarray:
    """A rim-crowded Poincare corpus, like a trained deep taxonomy."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, dim))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    a = rng.uniform(6.0, 10.0, size=n)           # hyperbolic radius
    r = np.sqrt(1.0 - np.exp(-a))
    return u * r[:, None]


def test_query_sign_is_not_reversed():
    """THE regression test. Ranking by the transform must match the geodesic."""
    X = deep_ball()
    t = HyperbolicTransform().fit(X)
    P, Q = t.transform_documents(X), t.transform_query(X)
    scores = Q @ P.T                              # what Qdrant's Dot computes
    D = poincare_distance(X, X)
    for i in range(len(X)):
        by_dot = np.argsort(-scores[i])[:10]      # Qdrant returns highest first
        by_geo = np.argsort(D[i])[:10]
        overlap = len(set(by_dot.tolist()) & set(by_geo.tolist())) / 10.0
        assert overlap == 1.0, (
            "query %d: dot ordering does not match the geodesic (overlap %.2f). "
            "If this is ~0.0 the query sign is reversed." % (i, overlap))


def test_transform_is_exact_not_approximate():
    """Top-1 must be the true nearest neighbour for every query, exactly."""
    X = deep_ball(200)
    t = HyperbolicTransform().fit(X)
    scores = t.transform_query(X) @ t.transform_documents(X).T
    D = poincare_distance(X, X)
    assert np.array_equal(np.argmax(scores, axis=1), np.argmin(D, axis=1))


def test_documents_are_unit_norm():
    X = deep_ball()
    P = HyperbolicTransform().fit(X).transform_documents(X)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-9)


def test_stored_dimension_is_d_plus_2():
    X = deep_ball(50, dim=7)
    P = HyperbolicTransform().fit(X).transform_documents(X)
    assert P.shape[1] == stored_dim(7) == 9


def test_rejects_points_outside_the_ball():
    bad = np.array([[0.9, 0.9, 0.9]])
    with pytest.raises(ValueError, match="norm >= 1"):
        HyperbolicTransform().fit(bad)


def test_rejects_unfitted_use():
    with pytest.raises(RuntimeError, match="fit"):
        HyperbolicTransform().transform_documents(deep_ball(10))


def test_documents_past_the_quantile_are_clipped_not_rejected():
    """Above the fitted quantile a document is projected back, not refused.

    That is the deliberate trade behind quantile scaling: scaling by the max
    keeps every document exact and produces a graph nobody can search.
    """
    X = deep_ball(500)
    t = HyperbolicTransform(quantile=0.99).fit(X)
    P = t.transform_documents(X)
    np.testing.assert_allclose(np.linalg.norm(P, axis=1), 1.0, atol=1e-9)
    frac = t.clipped_fraction(X)
    assert 0.0 < frac < 0.05, "expected about 1%% clipped, got %.3f" % frac


def test_one_deep_outlier_does_not_wreck_the_rest():
    """The bug that made recall 0.0025: a single deepest point set the scale.

    With quantile scaling the other documents must keep a usable share of the
    representation instead of collapsing onto the pad axis.
    """
    X = deep_ball(400)
    outlier = X[0] / np.linalg.norm(X[0]) * 0.99999999      # far deeper than any
    X = np.vstack([X, outlier])

    by_max = HyperbolicTransform(quantile=1.0).fit(X).transform_documents(X)
    by_q99 = HyperbolicTransform(quantile=0.99).fit(X).transform_documents(X)

    # the pad is the last coordinate, so 1 - pad is the share of each vector
    # that carries actual signal. If the pad is ~1 for everything the documents
    # are near-identical and HNSW has nothing to navigate by.
    signal_max = float(np.median(1.0 - by_max[:, -1]))
    signal_q99 = float(np.median(1.0 - by_q99[:, -1]))
    assert signal_q99 > 3 * signal_max, (
        "quantile scaling recovered too little signal: %.2e vs %.2e under max"
        % (signal_q99, signal_max))


def test_exactness_holds_for_unclipped_documents():
    """Ranking stays exact for everything below the quantile."""
    X = deep_ball(300)
    t = HyperbolicTransform(quantile=0.99).fit(X)
    P, Q = t.transform_documents(X), t.transform_query(X)
    D = poincare_distance(X, X)
    scores = Q @ P.T
    agree = [np.argmax(scores[i]) == np.argmin(D[i]) for i in range(len(X))]
    assert np.mean(agree) > 0.98, "top-1 agreement fell to %.3f" % np.mean(agree)


def test_depth_report_flags_the_three_regimes():
    rng = np.random.default_rng(0)

    def at_depth(a_target: float, n: int = 300) -> np.ndarray:
        u = rng.normal(size=(n, 5))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        return u * np.sqrt(1.0 - np.exp(-a_target))

    assert measure_depth(at_depth(0.5)).expectation == "LITTLE TO GAIN"
    assert measure_depth(at_depth(2.0)).expectation == "MODEST GAIN EXPECTED"
    assert measure_depth(at_depth(8.0)).expectation == "LARGE GAIN EXPECTED"


def test_depth_report_flags_outliers():
    """a_max/a_med is what used to break the transform; it must be surfaced."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(300, 5)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    X = u * np.sqrt(1.0 - np.exp(-3.0))
    X = np.vstack([X, u[0] * 0.99999999])          # one far deeper point
    rep = measure_depth(X)
    assert rep.a_ratio > 3.0
    assert "a_max/a_med" in rep.detail


def test_depth_survives_a_point_at_the_origin():
    """A taxonomy root sits at exactly 0. norm_spread must not divide by it."""
    X = np.vstack([np.zeros((1, 5)), deep_ball(50)])
    assert np.isfinite(measure_depth(X).norm_spread)
