"""Euclid vs Cosine vs transformed Dot, on your data, same queries, same truth.

Ground truth is the exact Poincare geodesic in float64 over the whole index.
Every arm is scored against that same truth with the same query set, so the
three numbers are directly comparable. Queries are held out of the index.

THREE THINGS THIS HARNESS REFUSES TO LET YOU GET WRONG
-------------------------------------------------------

1. **Controls.** A ground-truth-as-candidate control must score 1.0 and a random
   control must score ~0. If either fails the run raises instead of reporting
   numbers. A control that cannot fail is not a control.

2. **Build noise.** Rebuilding one *identical* configuration moved recall@10 by
   **0.1175** on the product-taxonomy corpus. A single build cannot distinguish
   a real difference from that. Every arm is therefore built ``builds`` times
   and the spread is reported as a noise floor. Any gap between arms smaller
   than the noise floor is not a result.

3. **Whether HNSW ran at all.** ``QdrantClient(":memory:")`` does not build a
   graph: ``indexed_vectors_count`` stays 0. A server under its default
   ``indexing_threshold`` will not build one for a small collection either.
   Under brute force the transform is exact by construction and scores 1.0000
   everywhere, which *flatters* it relative to a real graph. The harness labels
   every run and refuses to print a table when no graph was built.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qdrant_client import QdrantClient, models

from .depth import measure_depth
from .index import upload
from .transform import HyperbolicTransform, poincare_distance

__all__ = ["ArmResult", "BenchmarkResult", "benchmark"]


@dataclass
class ArmResult:
    arm: str
    distance: str
    dim: int
    mode: str                       # "hnsw" or "brute_force"
    indexed_vectors_count: int
    recall_at_10: dict[int, float] = field(default_factory=dict)   # mean over builds
    spread_at_10: dict[int, float] = field(default_factory=dict)   # max - min
    per_build: dict[int, list[float]] = field(default_factory=dict)
    ms_per_query: dict[int, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    n_index: int
    n_queries: int
    a_median: float
    arms: list[ArmResult]
    control_truth: float
    control_random: float
    builds: int = 1

    @property
    def mode(self) -> str:
        modes = {a.mode for a in self.arms}
        return modes.pop() if len(modes) == 1 else "mixed"

    @property
    def noise_floor(self) -> float:
        """Largest spread across rebuilds of one identical configuration."""
        return max((s for a in self.arms for s in a.spread_at_10.values()),
                   default=0.0)

    def beats(self, arm: str, other: str, width: int) -> str:
        """Compare two arms honestly against the measured noise floor."""
        a = next(x for x in self.arms if x.arm == arm)
        b = next(x for x in self.arms if x.arm == other)
        gap = a.recall_at_10[width] - b.recall_at_10[width]
        if abs(gap) <= self.noise_floor:
            return "no difference detected (%+0.4f, inside the %.4f noise floor)" % (
                gap, self.noise_floor)
        return "%s %s %s by %+0.4f" % (arm, "beats" if gap > 0 else "loses to",
                                       other, gap)

    def table(self, allow_brute: bool = False) -> str:
        if self.mode != "hnsw" and not allow_brute:
            return "\n".join([
                "NO RESULTS: no HNSW graph was built, so there is nothing here worth",
                "quoting. Under brute force the transform is exact by construction",
                "and scores 1.0000 at every depth. That is the maths, not a",
                "benchmark, and it flatters the transform against a real graph.",
                "",
                "  docker compose up -d",
                "  ... --url http://localhost:6333",
                "",
                "Or call .table(allow_brute=True) for representation-only numbers.",
                "",
                "n_index %d, n_queries %d, a_median %.3f" % (
                    self.n_index, self.n_queries, self.a_median),
                "controls: truth-as-candidate %.4f, random %.4f" % (
                    self.control_truth, self.control_random),
            ])

        widths = sorted(self.arms[0].recall_at_10)
        head = "%-12s %5s %6s  " % ("arm", "dim", "mode") + "".join(
            "%-22s" % ("ef=%d" % w) for w in widths)
        lines = [head, "-" * len(head)]
        for a in self.arms:
            cells = "".join("%.4f +-%.3f %5.1fms  " % (
                a.recall_at_10[w], a.spread_at_10[w] / 2.0,
                a.ms_per_query.get(w, float("nan"))) for w in widths)
            lines.append("%-12s %5d %6s  " % (
                a.arm, a.dim, "hnsw" if a.mode == "hnsw" else "brute") + cells)

        lines.append("")
        lines.append("n_index %d, n_queries %d, a_median %.3f, %d build(s) per arm"
                     % (self.n_index, self.n_queries, self.a_median, self.builds))
        lines.append("controls: truth-as-candidate %.4f (want 1.0), random %.4f "
                     "(want ~0)" % (self.control_truth, self.control_random))
        if self.builds > 1:
            lines.append("NOISE FLOOR %.4f -- rebuilding one identical configuration"
                         % self.noise_floor)
            lines.append("moves recall by this much. A gap smaller than it is not a "
                         "result.")
            names = [a.arm for a in self.arms]
            for w in widths:
                for other in names:
                    if other == "transform":
                        continue
                    lines.append("  ef=%-5d transform vs %-8s %s"
                                 % (w, other, self.beats("transform", other, w)))
            unstable = [a.arm for a in self.arms
                        if max(a.spread_at_10.values()) > 0.05]
            if unstable:
                lines.append("UNSTABLE ACROSS REBUILDS: %s. The graph this arm builds"
                             % ", ".join(unstable))
                lines.append("varies run to run, which is a cost in its own right.")
        else:
            lines.append("SINGLE BUILD: no noise floor measured. Use builds=3 before "
                         "believing any gap.")
        base = min(a.dim for a in self.arms)
        over = [a for a in self.arms if a.dim > base]
        if over:
            lines.append("DIMENSION COST: %s stores %dd against %dd, so every distance"
                         % (", ".join(a.arm for a in over), over[0].dim, base))
            lines.append("computation and every stored vector is %.0f%% larger. At low d"
                         % (100.0 * (over[0].dim - base) / base))
            lines.append("that is a real overhead; on a 512d embedding it is noise.")
        lines.append("ms/query is CLIENT-SIDE and network-dominated against a "
                     "remote cluster.")
        lines.append("Compare arms at the same ef, not by the timings.")
        if self.mode != "hnsw":
            lines += ["", "WARNING: brute force, no graph. The transform is exact by",
                      "construction here, so these are the maths and not a benchmark",
                      "result. Do not quote them."]
        return "\n".join(lines)


def _recall(found, truth: set[int], k: int) -> float:
    return len(set(found[:k]) & truth) / float(k)


def _graph_mode(indexed: int, n: int) -> str:
    """Did Qdrant actually build an HNSW graph over these vectors?

    ``indexed_vectors_count`` is the definitive signal. An earlier version
    compared hnsw_ef=4 against hnsw_ef=512 and called a match brute force. That
    is wrong twice over: Qdrant uses ``effective ef = max(hnsw_ef, limit)`` so
    an ef below your limit is silently ignored, and on easy data a good graph
    returns the same top-k at any ef. Both make a real graph look like a scan.
    """
    return "hnsw" if indexed >= n > 0 else "brute_force"


def benchmark(documents: np.ndarray, *, n_queries: int = 100,
              widths: tuple[int, ...] = (16, 64, 256), k: int = 10,
              seed: int = 0, client: QdrantClient | None = None,
              m: int = 16, ef_construct: int = 200,
              force_index: bool = True, builds: int = 3,
              index_timeout_s: int = 600) -> BenchmarkResult:
    """Run all three arms, ``builds`` times each, and report the noise floor.

    ``client`` defaults to in-memory Qdrant, which cannot build a graph. Pass a
    real server client to measure HNSW. ``force_index`` sets
    ``indexing_threshold=1`` so a server builds a graph even for a small
    collection; it has no effect in-memory. ``index_timeout_s`` bounds the wait
    for that build, which is asynchronous.

    ``widths`` is swept as **hnsw_ef**, with ``limit`` fixed at ``k``. ef is the
    knob that costs work, so this compares arms at matched cost. Comparing at a
    fixed ef and varying ``limit`` instead measures nothing about cost.

    ``builds`` rebuilds every arm that many times. Measured here: rebuilding one
    identical configuration moved recall@10 by 0.1175. A single build cannot
    tell a real difference from that, and this project has already shipped
    conclusions that were nothing but build noise.
    """
    docs = np.asarray(documents, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_queries = min(n_queries, max(1, len(docs) // 10))
    qi = rng.choice(len(docs), n_queries, replace=False)
    keep = np.ones(len(docs), bool); keep[qi] = False
    index, queries = docs[keep], docs[qi]

    D = poincare_distance(queries, index)
    truth = [set(np.argsort(D[i])[:k].tolist()) for i in range(len(queries))]

    control_truth = float(np.mean([
        _recall(list(np.argsort(D[i])[:k]), truth[i], k) for i in range(len(queries))]))
    control_random = float(np.mean([
        _recall(list(rng.choice(len(index), k, replace=False)), truth[i], k)
        for i in range(len(queries))]))
    if control_truth < 0.999:
        raise RuntimeError("truth-as-candidate control scored %.4f, expected 1.0; "
                           "the harness is broken" % control_truth)
    if control_random > 0.05:
        raise RuntimeError("random control scored %.4f, expected ~0; the truth set "
                           "is degenerate" % control_random)

    own = client is None
    client = client or QdrantClient(":memory:")
    transform = HyperbolicTransform().fit(index)

    arms = [
        ("euclid",    index, queries, models.Distance.EUCLID),
        ("cosine",    index, queries, models.Distance.COSINE),
        ("transform", transform.transform_documents(index),
                      np.array([transform.transform_query(q) for q in queries]),
                      models.Distance.DOT),
    ]
    out: list[ArmResult] = []
    try:
        for arm, V, Q, dist in arms:
            res = ArmResult(arm=arm, distance=dist.name, dim=int(V.shape[1]),
                            mode="brute_force", indexed_vectors_count=0)
            runs: dict[int, list[float]] = {w: [] for w in widths}
            lat: dict[int, list[float]] = {w: [] for w in widths}
            for b in range(builds):
                name = "bench_%s_%d" % (arm, b)
                if client.collection_exists(name):
                    client.delete_collection(name)
                client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=int(V.shape[1]),
                                                       distance=dist),
                    hnsw_config=models.HnswConfigDiff(
                        m=m, ef_construct=ef_construct,
                        # benchmark-only: forces the graph to be used even on a
                        # small collection so the arms are comparable. Do NOT
                        # copy this into production; the default is right there.
                        full_scan_threshold=10 if force_index else None),
                    optimizers_config=(models.OptimizersConfigDiff(
                        indexing_threshold=1) if force_index else None),
                )
                upload(client, name, V)
                # HNSW is built asynchronously; without this wait the check
                # below sees 0 on a real server and wrongly reports brute force.
                indexed = 0
                for _ in range(max(1, index_timeout_s // 2)):
                    info = client.get_collection(name)
                    indexed = info.indexed_vectors_count or 0
                    if str(info.status).endswith("green") and indexed >= len(V):
                        break
                    time.sleep(2)
                res.mode = _graph_mode(indexed, len(V))
                res.indexed_vectors_count = int(indexed)
                for w in widths:
                    # w is hnsw_ef, the knob that actually costs work. limit is
                    # held at k so every arm does the same amount of graph
                    # traversal and returns the same number of points. An
                    # earlier version used hnsw_ef=max(w,128) with limit=w,
                    # which held the real cost CONSTANT while appearing to vary
                    # it, and produced a "10x cheaper query" claim that nothing
                    # supported.
                    tot = 0.0
                    t0 = time.perf_counter()
                    for j in range(len(Q)):
                        hits = client.query_points(
                            collection_name=name,
                            query=[float(x) for x in np.ravel(Q[j])],
                            limit=k,
                            search_params=models.SearchParams(hnsw_ef=w),
                        ).points
                        tot += _recall([h.id for h in hits], truth[j], k)
                    runs[w].append(tot / len(Q))
                    lat[w].append((time.perf_counter() - t0) / len(Q) * 1000.0)
                client.delete_collection(name)
            for w in widths:
                res.per_build[w] = runs[w]
                res.recall_at_10[w] = float(np.mean(runs[w]))
                res.spread_at_10[w] = float(max(runs[w]) - min(runs[w]))
                res.ms_per_query[w] = float(np.median(lat[w]))
            out.append(res)
    finally:
        if own:
            client.close()

    return BenchmarkResult(
        n_index=len(index), n_queries=len(queries),
        a_median=measure_depth(index).a_median,
        arms=out, control_truth=control_truth, control_random=control_random,
        builds=builds,
    )
