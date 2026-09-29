"""A product hierarchy, indexed three ways, so you can see the difference.

    Products
      Footwear
        Running Shoes
          Trail Running Shoes
            Waterproof Trail Running Shoes
              Waterproof Trail Running Shoes, Winter

The tree is embedded in a 5-dimensional Poincare ball by the standard
construction: root at the origin, each level pushed further toward the boundary,
siblings separated by angle. That is what a trained hyperbolic embedding of a
taxonomy looks like, and it is why deep nodes end up jammed against the rim.

The question the demo answers: given a leaf, does the index return its true
hyperbolic neighbours -- its siblings and its parent chain?

Run:
    python examples/product_taxonomy.py
    python examples/product_taxonomy.py --url http://localhost:6333   # real HNSW
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient  # noqa: E402

from hyperbolic_qdrant import (benchmark, build, measure_depth,  # noqa: E402
                               poincare_distance, search)

DIM = 5

# A real-looking catalogue. Each level refines the one above it.
LEVELS = [
    ["Footwear", "Outerwear", "Packs", "Shelter"],
    {"Footwear": ["Running Shoes", "Hiking Boots", "Sandals", "Casual Shoes"],
     "Outerwear": ["Jackets", "Vests", "Fleece", "Rainwear"],
     "Packs": ["Daypacks", "Backpacking Packs", "Hydration Packs", "Duffels"],
     "Shelter": ["Tents", "Tarps", "Bivy Sacks", "Hammocks"]},
    ["Trail", "Road", "Ultralight", "Expedition"],
    ["Waterproof", "Breathable", "Insulated", "Reinforced"],
    ["Winter", "Three-Season", "Alpine", "Desert"],
    ["Mens", "Womens", "Kids", "Unisex"],
]


def build_tree() -> tuple[np.ndarray, list[str], list[int], list[int]]:
    """Return (poincare points, full names, parent index, depth)."""
    rng = np.random.default_rng(0)
    names = ["Products"]; parent = [-1]; depth = [0]
    direction = [np.zeros(DIM)]; radius = [0.0]

    def child_dir(pd: np.ndarray, level: int) -> np.ndarray:
        # siblings keep most of the parent's bearing and diverge a little.
        # that shrinking divergence is what makes a branch a branch.
        d = (pd + rng.normal(size=DIM) * (0.6 / (1 + level))
             if np.any(pd) else rng.normal(size=DIM))
        return d / np.linalg.norm(d)

    frontier = [0]
    for level in range(1, len(LEVELS) + 1):
        spec = LEVELS[level - 1]
        nxt = []
        for p in frontier:
            head = names[p].split(" > ")[-1]
            kids = spec[head] if isinstance(spec, dict) else spec
            for label in kids:
                # read naturally: qualifiers prepend, top levels stand alone
                full = label if level <= 2 else "%s %s" % (label, head)
                names.append(full if level == 1 else "%s > %s" % (names[p], full))
                parent.append(p); depth.append(level)
                direction.append(child_dir(direction[p], level))
                # radius grows with depth, with jitter so nodes at one level are
                # not degenerate. this is what pushes a deep tree to the rim.
                base = 1.0 - 0.5 ** level
                radius.append(float(np.clip(base + rng.normal() * 0.012 * (1 - base),
                                            0.02, 0.999999)))
                nxt.append(len(names) - 1)
        frontier = nxt

    X = np.array([d * r for d, r in zip(direction, radius)], dtype=np.float64)
    X[0] = 0.0
    return X, names, parent, depth


def lineage(i: int, parent: list[int]) -> list[int]:
    out, cur = [], i
    while cur != -1:
        out.append(cur); cur = parent[cur]
    return out


def _client(args) -> QdrantClient:
    return QdrantClient(url=args.url,
                        api_key=args.api_key or os.environ.get("QDRANT_API_KEY"),
                        timeout=600)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None,
                    help="Qdrant server URL. Without it, in-memory is used and "
                         "NO HNSW graph is built.")
    ap.add_argument("--api-key", default=None,
                    help="API key for a managed Qdrant. Also read from QDRANT_API_KEY.")
    ap.add_argument("--allow-brute", action="store_true",
                    help="print representation-only numbers with no graph. They "
                         "flatter the transform; see the README.")
    args = ap.parse_args()

    X, names, parent, depth = build_tree()
    print("Product tree: %d nodes, %d levels, %dd Poincare ball" % (
        len(X), max(depth), DIM))
    print("deepest example: %s\n" % names[int(np.argmax(depth))])
    print(measure_depth(X), "\n")

    print("=" * 74)
    print("Three ways to index the same tree")
    print("=" * 74)
    client = _client(args) if args.url else None
    res = benchmark(X, n_queries=80, widths=(16, 64, 256), seed=0, client=client)
    print(res.table(allow_brute=args.allow_brute))

    print("\n" + "=" * 74)
    print("What a single query returns")
    print("=" * 74)
    c = _client(args) if args.url else QdrantClient(":memory:")
    transform = build(c, "products", X, recreate=True,
                      payloads=[{"name": n, "depth": d} for n, d in zip(names, depth)])
    qi = int(np.argmax(np.array(depth)))
    print("query: %s\n" % names[qi])

    truth = set(np.argsort(poincare_distance(X[qi][None, :], X)[0])[:6].tolist())
    fam = set(lineage(qi, parent))
    for h in search(c, "products", X[qi], transform, limit=6):
        tag = "ancestor" if (h.id in fam and h.id != qi) else (
            "self" if h.id == qi else "")
        print("%-6s %-9s %s" % ("hit" if h.id in truth else "MISS", tag,
                                h.payload["name"]))
    c.delete_collection("products")      # leave no collection behind
    c.close()

    if res.mode != "hnsw":
        print("\nNOTE: in-memory Qdrant does not build an HNSW graph, so the table")
        print("above compares representations under brute force. Pass --url to a")
        print("real server to measure the graph, which is what the transform fixes.")


if __name__ == "__main__":
    main()
