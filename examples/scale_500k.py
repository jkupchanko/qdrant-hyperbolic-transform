"""Does the transform still hold at half a million points?

Every other number in this repository is measured at about 5,000 points,
which is small enough that a reader can fairly ask whether any of it survives
scale. This runs the same comparison at 500,000.

It generates its own corpus, so there is nothing to download. The hierarchy
comes from examples/depth_ladder.py's generator and is then rescaled radially
to a target median depth, which is the variable that decides whether the
transform matters at all: near the origin hyperbolic space is almost flat and
plain Euclid is fine, and the crowding that breaks it only appears deep.

Two depths are run. The first is deep but ordinary. The second is genuinely
rim-crowded, and is there because a result that only holds in the easy regime
is not worth reporting.

    docker compose up -d
    python examples/scale_500k.py --url http://localhost:6333

Expect this to take a while: it builds four collections of 500k points.
Use --n to try it smaller first.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.depth_ladder import at_depth, hierarchical_corpus  # noqa: E402
from hyperbolic_qdrant.transform import (  # noqa: E402
    HyperbolicTransform,
    poincare_distance,
)

DEPTHS = (3.5, 7.75)
EFS = (64, 128, 256, 512, 1024, 2048)


def truth_top10(index: np.ndarray, queries: np.ndarray, k: int = 10):
    """Exact geodesic neighbours, in blocks so 500k x 150 fits in memory."""
    out = []
    for q in queries:
        d = poincare_distance(q[None, :], index)[0]
        o = np.argpartition(d, k)[:k]
        out.append(set(o[np.argsort(d[o])].tolist()))
    return out


def upload(client, coll, vectors, batch=2000):
    for i in range(0, len(vectors), batch):
        chunk = vectors[i:i + batch]
        client.upsert(coll, wait=True, points=[
            models.PointStruct(id=int(i + j), vector=[float(x) for x in v])
            for j, v in enumerate(chunk)])


def wait_indexed(client, coll, n, timeout_s=7200):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = client.get_collection(coll)
        if (str(info.status).endswith("green")
                and (info.indexed_vectors_count or 0) >= n):
            return info
        time.sleep(10)
    raise TimeoutError("%s never finished indexing" % coll)


def run_depth(client, X, a_med, nq, builds):
    rng = np.random.default_rng(7)
    qi = rng.choice(len(X), nq, replace=False)
    mask = np.ones(len(X), bool)
    mask[qi] = False
    index, queries = X[mask], X[qi]

    print("\n  depth a_med %.2f, %d indexed, %d queries" % (a_med, len(index), nq))
    t0 = time.time()
    truth = truth_top10(index, queries)
    print("    exact truth in %.0fs" % (time.time() - t0))

    transform = HyperbolicTransform().fit(index)
    arms = [
        ("raw poincare / Euclid", index, queries,
         models.Distance.EUCLID, index.shape[1] * 4),
        ("transform / Dot", transform.transform_documents(index),
         np.array([transform.transform_query(q) for q in queries]),
         models.Distance.DOT, (index.shape[1] + 2) * 4),
    ]

    rows = {}
    for name, V, Q, dist, bpp in arms:
        per_build = {ef: [] for ef in EFS}
        for b in range(builds):
            coll = "scale500k_b%d_%s" % (b, name.split()[0])
            if client.collection_exists(coll):
                client.delete_collection(coll)
            client.create_collection(
                coll,
                vectors_config=models.VectorParams(size=V.shape[1], distance=dist),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100,
                                                  full_scan_threshold=10,
                                                  payload_m=0),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
            )
            t0 = time.time()
            upload(client, coll, V)
            wait_indexed(client, coll, len(V))
            print("    %-22s build %d uploaded in %.0fs" % (name, b, time.time() - t0))

            # the graph has to be walked, or every number below is a scan
            ex = client.query_points(coll, query=Q[0].tolist(), limit=10,
                                     search_params=models.SearchParams(exact=True)).points
            ap = client.query_points(coll, query=Q[0].tolist(), limit=10,
                                     search_params=models.SearchParams(hnsw_ef=64)).points
            overlap = len({p.id for p in ex} & {p.id for p in ap}) / 10.0
            if overlap == 1.0:
                print("      WARNING gate: exact and ef=64 agree exactly. The graph may "
                      "not be in use, or it may be degenerate. Treat this arm with care.")

            for ef in EFS:
                hit = 0.0
                for j, q in enumerate(Q):
                    got = client.query_points(
                        coll, query=q.tolist(), limit=10,
                        search_params=models.SearchParams(hnsw_ef=ef)).points
                    hit += len({p.id for p in got} & truth[j]) / 10.0
                per_build[ef].append(hit / len(Q))
            client.delete_collection(coll)
        rows[name] = (bpp, per_build)

    spread = max((max(v) - min(v)) for _, pb in rows.values()
                 for v in pb.values() if len(v) > 1) if builds > 1 else 0.0
    print("\n    %-22s %6s  %s" % ("arm", "B/pt",
                                   " ".join("ef=%-6d" % e for e in EFS)))
    for name, (bpp, pb) in rows.items():
        print("    %-22s %6d  %s" % (name, bpp, " ".join(
            "%.4f " % (sum(pb[e]) / len(pb[e])) for e in EFS)))
    if builds > 1:
        print("    build-to-build spread %.4f; a smaller gap than this is noise"
              % spread)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--n", type=int, default=500000)
    ap.add_argument("--queries", type=int, default=150)
    ap.add_argument("--builds", type=int, default=1)
    a = ap.parse_args()

    client = QdrantClient(url=a.url, timeout=7200)
    base, frac = hierarchical_corpus(a.n)
    for a_med in DEPTHS:
        run_depth(client, at_depth(base, frac, a_med), a_med, a.queries, a.builds)


if __name__ == "__main__":
    main()
