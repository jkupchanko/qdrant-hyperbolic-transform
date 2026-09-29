"""How deep is your embedding, and what should you expect?

The governing quantity is hyperbolic radius:

    a = -log(1 - ||x||^2)

It grows without bound as points approach the boundary of the ball. How far
plain Euclidean indexing degrades tracks the median of a across the corpus.

What was measured
-----------------
One generated tree corpus (6,000 nodes, 5d) rescaled radially so that depth was
the only variable. Direction and tree structure held fixed. Prefetch width 10,
exact geodesic ground truth, 80 queries, three rebuilds per arm against a live
Qdrant server, all arms in the same run:

    a_med   euclid   cosine   transform    gain    noise floor
     2.00   0.7825   0.7762      0.9987   +0.216         0.000
     3.00   0.6413   0.6388      1.0000   +0.359         0.000
     3.50   0.5688   0.5688      0.9842   +0.415         0.024
     5.00   0.3837   0.3837      0.9721   +0.588         0.046
     6.50   0.2450   0.2450      0.9371   +0.692         0.083
     8.00   0.1425   0.1425      0.9550   +0.813         0.035

Every gain is far outside the measured build-to-build noise. There is no
crossover at this width: the transform wins everywhere, and the margin grows
with depth because that is where plain Euclid falls apart.

The advantage is at NARROW prefetch
-----------------------------------
That table is width 10. Widen the prefetch and plain Euclid catches up -- on the
product-taxonomy demo at width 100 the difference was inside the noise floor.
So the honest claim is not "better recall", it is "the same recall from a much
cheaper query".

Why this module still will not decide for you
---------------------------------------------
The table above is one generated corpus. Real embeddings differ in ways that are
not captured by a_med alone, and this project has already retired four different
thresholds that were read off intervals nobody sampled. Run ``benchmark()`` on
your own data; it takes a minute and it reports a noise floor so you can tell a
real difference from a rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DepthReport", "measure_depth"]


@dataclass
class DepthReport:
    n: int
    dim: int
    a_median: float
    a_max: float
    a_ratio: float       # a_max / a_median -- depth outliers
    norm_median: float
    norm_max: float
    norm_spread: float   # max/min over non-zero norms; a root at the origin is skipped
    expectation: str
    detail: str

    def __str__(self) -> str:
        return (
            "%d points, %dd Poincare ball\n"
            "  a_median      %.3f\n"
            "  a_max         %.3f\n"
            "  a_max/a_med   %.2f\n"
            "  norm_median   %.6f\n"
            "  norm_max      %.6f\n"
            "  norm_spread   %.1f\n"
            "  expectation   %s\n"
            "  %s" % (self.n, self.dim, self.a_median, self.a_max, self.a_ratio,
                      self.norm_median, self.norm_max, self.norm_spread,
                      self.expectation, self.detail)
        )


def measure_depth(vectors: np.ndarray) -> DepthReport:
    """Report hyperbolic depth and what to expect. Decides nothing."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError("expected a 2-D array of Poincare-ball points")
    n = np.linalg.norm(v, axis=1)
    if n.max() >= 1.0:
        raise ValueError(
            "max norm %.6f >= 1: these are not Poincare-ball points" % n.max())
    a = -np.log(np.maximum(1.0 - n ** 2, 1e-300))
    a_med = float(np.median(a))
    a_max = float(a.max())
    ratio = a_max / max(a_med, 1e-12)

    if a_med < 1.0:
        expectation = "LITTLE TO GAIN"
        detail = ("Barely hyperbolic. Plain Distance.EUCLID on the raw coordinates "
                  "will do fine. Off-the-shelf hyperbolic CLIP models live here "
                  "(a_med around 0.01).")
    elif a_med < 3.0:
        expectation = "MODEST GAIN EXPECTED"
        detail = ("At a_med 2.0 the transform scored 0.9987 against 0.7825 for "
                  "Euclid at prefetch 10. Worth benchmarking, especially if you "
                  "want to keep the prefetch narrow.")
    else:
        expectation = "LARGE GAIN EXPECTED"
        detail = ("Plain Euclid degrades steeply from here (0.64 at a_med 3.0 down "
                  "to 0.14 at 8.0, prefetch 10) while the transform held 0.94-1.00. "
                  "The deeper you are, the more this is worth.")

    if ratio > 3.0:
        detail += (" NOTE: a_max/a_med is %.1f, so a few points are far deeper than "
                   "the rest. Scaling by the maximum used to collapse corpora like "
                   "this to recall 0.0025; the fitted quantile handles it, but check "
                   "clipped_fraction()." % ratio)

    return DepthReport(
        n=len(v), dim=v.shape[1], a_median=a_med, a_max=a_max, a_ratio=ratio,
        norm_median=float(np.median(n)), norm_max=float(n.max()),
        norm_spread=float(n.max() / n[n > 0].min()) if (n > 0).any() else 1.0,
        expectation=expectation, detail=detail,
    )
