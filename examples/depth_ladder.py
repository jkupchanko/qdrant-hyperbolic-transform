"""Where is the crossover? Reproduce the central claim from scratch.

This is the experiment the README's crossover table comes from, and it needs no
external data: it generates one hierarchical corpus, then rescales it radially
to a ladder of depths. Direction and tree structure are held fixed, so a_med is
the only thing that varies between rows. That is what makes the comparison a
ladder rather than eight unrelated corpora.

    python examples/depth_ladder.py                          # brute force
    python examples/depth_ladder.py --url http://localhost:6333   # real HNSW

Expect it to take a few minutes: eight depths times three arms.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient  # noqa: E402

from hyperbolic_qdrant import benchmark  # noqa: E402

DIM = 5
N = 6000
DEPTHS = (2.0, 3.0, 3.5, 5.0, 6.5, 8.0)
EFS = (16, 64, 256)   # hnsw_ef, the knob that costs work


def hierarchical_corpus(n: int = N, dim: int = DIM, branch: int = 4,
                        seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """An actual tree: every child inherits its parent's direction.

    An earlier version of this picked a random existing direction and a random
    existing depth INDEPENDENTLY, which is not a tree at all -- direction and
    depth were uncorrelated. The transform exploits the fact that a deep node
    points the same way as its parent, so on that corpus it had nothing to work
    with and collapsed to recall 0.13. The bug was here, not in the transform.
    """
    rng = np.random.default_rng(seed)
    d0 = rng.normal(size=dim)
    dirs = [d0 / np.linalg.norm(d0)]
    lev = [0]
    frontier = [0]
    while len(dirs) < n and frontier:
        nxt = []
        for p in frontier:
            for _ in range(branch):
                if len(dirs) >= n:
                    break
                d = dirs[p] + rng.normal(size=dim) * (0.6 / (1 + lev[p]))
                dirs.append(d / np.linalg.norm(d))
                lev.append(lev[p] + 1)
                nxt.append(len(dirs) - 1)
        frontier = nxt
    lv = np.array(lev, dtype=np.float64)
    return np.array(dirs), lv / max(lv.max(), 1.0)


def at_depth(u: np.ndarray, frac: np.ndarray, a_med_target: float) -> np.ndarray:
    """Rescale so the median hyperbolic radius hits the target."""
    a = 0.3 + frac * 2.0                     # relative depth profile, shape fixed
    a = a * (a_med_target / np.median(a))
    r = np.sqrt(np.clip(1.0 - np.exp(-a), 0.0, 1.0 - 1e-12))
    return u * r[:, None]


def _client(args) -> QdrantClient:
    return QdrantClient(url=args.url,
                        api_key=args.api_key or os.environ.get("QDRANT_API_KEY"),
                        timeout=600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Qdrant server URL for real HNSW")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--builds", type=int, default=3,
                    help="rebuilds per arm; the spread becomes the noise floor")
    ap.add_argument("--ef", type=int, default=256, choices=EFS,
                    help="which swept hnsw_ef to report in the table. ef is the "
                         "knob that costs work, so arms are compared at matched ef.")
    ap.add_argument("--api-key", default=None,
                    help="API key for a managed Qdrant. Also read from QDRANT_API_KEY.")
    ap.add_argument("--allow-brute", action="store_true",
                    help="print representation-only numbers with no graph. They "
                         "flatter the transform; see the README.")
    args = ap.parse_args()

    W = args.ef
    u, frac = hierarchical_corpus()
    client = _client(args) if args.url else None
    mode = None

    print("all arms at hnsw_ef=%d, limit=10, %d builds each\n" % (W, args.builds))
    print("%-8s %10s %10s %12s %10s %9s" % (
        "a_med", "euclid", "cosine", "transform", "gain", "noise"))
    print("-" * 64)
    for target in DEPTHS:
        X = at_depth(u, frac, target)
        res = benchmark(X, n_queries=args.queries, widths=EFS, seed=0,
                        client=client, builds=args.builds)
        mode = res.mode
        if mode != "hnsw" and not args.allow_brute:
            print(res.table())
            print("\nStopping: a ladder measured under brute force shows the "
                  "transform at 1.0000\non every row, which is the maths, not "
                  "the crossover.")
            return
        got = {a.arm: a.recall_at_10[W] for a in res.arms}
        gain = got["transform"] - got["euclid"]
        flag = "" if abs(gain) > res.noise_floor else "  (inside noise)"
        print("%-8.2f %10.4f %10.4f %12.4f %+10.4f %9.4f%s" % (
            res.a_median, got["euclid"], got["cosine"], got["transform"],
            gain, res.noise_floor, flag))
    if client:
        client.close()

    print()
    if mode == "hnsw":
        print("HNSW graph confirmed on every arm.")
    else:
        print("NO HNSW graph was built -- these rows compare representations under")
        print("brute force, not graph behaviour. Pass --url for the real measurement.")


if __name__ == "__main__":
    main()
