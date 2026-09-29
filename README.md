# qdrant-hyperbolic-transform

[![tests and benchmark](https://github.com/jkupchanko/qdrant-hyperbolic-transform/actions/workflows/benchmark.yml/badge.svg)](https://github.com/jkupchanko/qdrant-hyperbolic-transform/actions/workflows/benchmark.yml)

Index deep hyperbolic (Poincaré-ball) embeddings in Qdrant using ordinary HNSW.

This is a preprocessing layer, not a Qdrant change. You convert your vectors, create
a normal collection with `Distance.DOT`, and search it normally. Qdrant does not need
to know anything about hyperbolic geometry.

**The claim in one line:** on a deep hierarchy, plain Euclidean indexing hits a
recall ceiling that more search cannot fix. The transform does not.

---

## The problem

Train a Poincaré embedding on a deep hierarchy and almost every point ends up at norm
0.9999 — crushed into a shell one ten-thousandth thick against the boundary of the
ball. Store those coordinates directly and HNSW's construction distance is

```
||x_a - x_b||²  =  (depth difference)²  +  (the hyperbolic distance you want)
```

The depth term dominates, so the graph links same-depth siblings instead of
parent→child, and search has no edges to descend.

## The transform

For a Poincaré point `v`, let `w = 1/(1 - ||v||²)`. Minimising the geodesic distance
to a query `u` is equivalent to minimising

```
document:  phi(v) = [w, w·v]           scaled by a quantile, padded to unit norm
query:     psi(u) = [-(1 + ||u||²), 2u, 0]
distance:  Dot
```

`w` blows the crushed rim wide open, so near-boundary points become far apart in the
index and HNSW can navigate.

**The stored dimension is `d + 2`, and that is not free.** On the 5-dimensional
corpora here that is 7d against 5d — **40% more** per distance computation and per
stored vector. At 512 dimensions the same two coordinates are 0.4% and irrelevant.
The benchmark prints this overhead alongside the recall so you can net it yourself;
none of the recall numbers below are adjusted for it.

**Two things that are not optional:**

- **The minus sign.** The maths minimises the inner product and Qdrant's `Dot`
  maximises, so the query is negated. Getting it wrong gives recall **0.0000**, not a
  degraded result. Pinned by a test.
- **Quantile scaling, not max.** Scaling by the largest document norm lets a single
  anomalously deep point set the scale for the whole corpus, crushing everything else
  onto the pad axis where HNSW cannot tell documents apart. Measured: **0.0025** under
  max-scaling, **0.9988** at quantile 0.99, on the same corpus. Check
  `transform.clipped_fraction(docs)` — documents above the quantile are ranked
  approximately rather than exactly.

## Usage

```python
from qdrant_client import QdrantClient
from hyperbolic_qdrant import measure_depth, build, search, benchmark

print(measure_depth(vectors))          # what to expect
print(benchmark(vectors).table())      # euclid vs cosine vs transform, your data

client = QdrantClient(url="http://localhost:6333")
transform = build(client, "products", vectors)     # fit + create + upload
hits = search(client, "products", query_vector, transform, limit=10)
```

Keep the returned `transform` with the collection. Its `scale` is fitted to your
documents, and a query transformed under a different scale will rank wrongly.

---

## Results

### You need a real server

```bash
docker compose up -d
python examples/depth_ladder.py     --url http://localhost:6333
python examples/product_taxonomy.py --url http://localhost:6333
```

`QdrantClient(":memory:")` builds no graph at all — `indexed_vectors_count` stays 0.
**Without `--url` the scripts refuse to print a results table.** Under brute force the
transform is exact by construction and scores 1.0000 everywhere, which *flatters* it.
A number nobody should quote is worse than no number. `--allow-brute` overrides.

### Depth ladder, prefetch 10

One generated tree corpus (6,000 nodes, 5d) rescaled radially so depth is the only
variable. Exact geodesic truth, 80 queries, **three rebuilds per arm**, live server:

| `a_med` | Euclid | Cosine | transform | gain | noise floor |
|---|---|---|---|---|---|
| 2.00 | 0.7825 | 0.7762 | **0.9987** | +0.216 | 0.000 |
| 3.00 | 0.6413 | 0.6388 | **1.0000** | +0.359 | 0.000 |
| 3.50 | 0.5688 | 0.5688 | **0.9842** | +0.415 | 0.024 |
| 5.00 | 0.3837 | 0.3837 | **0.9721** | +0.588 | 0.046 |
| 6.50 | 0.2450 | 0.2450 | **0.9371** | +0.692 | 0.083 |
| 8.00 | 0.1425 | 0.1425 | **0.9550** | +0.813 | 0.035 |

Every gain is far outside the measured noise. Note this table was taken at
`limit=10` with `hnsw_ef` clamped to 128; the matched-cost sweep below is the one to
read for a cost comparison. The margin grows with depth because
that is where plain Euclid falls apart, not because the transform improves.

Reproduce with `examples/depth_ladder.py`, which generates its own corpus.

### Compared at matched cost

`hnsw_ef` is the knob that costs work, so the arms are swept across it with `limit`
fixed. Product-taxonomy demo, three builds per arm, live server:

| arm | ef=16 | ef=64 | ef=256 |
|---|---|---|---|
| Euclid | 0.4650 ±0.000 | 0.4650 ±0.000 | **0.4650 ±0.000** |
| Cosine | 0.4650 ±0.000 | 0.4650 ±0.000 | **0.4650 ±0.000** |
| **transform** | 0.5646 ±0.101 | 0.6725 ±0.132 | **0.8550 ±0.116** |

**Euclid is flat.** Spending 16× more search changes nothing, because the neighbours
it needs are not in its graph at any ef. The transform climbs 0.56 → 0.86 with the
same budget.

```
ef=16   no difference detected (+0.0996, inside the 0.2637 noise floor)
ef=64   no difference detected (+0.2075, inside the 0.2637 noise floor)
ef=256  transform beats euclid by +0.3900
```

Only ef=256 clears the noise floor. Below that, nothing is proven.

An earlier version of this README claimed "the same recall from a 10× cheaper query".
That was wrong: the benchmark used `hnsw_ef=max(limit, 128)`, which held ef constant
at 128 while appearing to vary the work done. No cost was ever measured. Client-side
latency is reported now but is network-dominated against a remote cluster (~80 ms for
every arm) and should not be used to compare anything.

### Build noise is large, and the harness measures it

Rebuilding one *identical* configuration moved recall@10 by up to **0.1175**. Every
arm is built three times by default and the spread is reported as a noise floor; any
gap smaller than it is printed as "no difference detected".

The transform's graph is also **less stable** than Euclid's across rebuilds (±0.031
against ±0.000). That is a real cost and the harness flags it.

---

## When **not** to use this

**Hyperbolic CLIP models.** Off-the-shelf models like Hyper3-CLIP, MERU and HyCoCLIP
sit at `a_med` ≈ 0.009 — their points are near the origin where hyperbolic space is
effectively flat, and the exact geodesic, Euclid and cosine produce **bit-identical**
rankings on them. Use `Distance.EUCLID` on the raw coordinates.

**At low ef.** Below ef≈256 on the demo corpus the gap sits inside the noise floor.
If you run a tight ef budget, measure before assuming a win.

**Without benchmarking.** Every number above is one generated corpus plus one demo.
This project has already retired four separate thresholds that were read off intervals
nobody sampled, and `measure_depth()` deliberately gates nothing. `benchmark()` takes
a minute and reports a noise floor.

## Honest limits

- **The d+2 overhead is not netted out.** The transform wins on recall while storing
  40% more per vector at d=5. Whether that trade is worth it at your dimension is
  yours to judge; the benchmark reports both.
- **Untested above 271k points.** Depth, not size, was the failure mode in everything
  measured, but scale beyond that is unverified.
- **The deepest ~1% are approximate.** Quantile scaling clips them; `clipped_fraction()`
  tells you how many.
- **Poincaré ball only.** Lorentz-model embeddings must be converted first.
- **The noise floor itself moves.** Three builds gave 0.0625 on one run and 0.2150 on
  another of the same corpus. Use more builds if a gap is close to the floor.
- The results on trained WordNet and Google Product Taxonomy embeddings predate the
  quantile fix and are not reproduced here.

## Don't take these numbers on trust

Every table below is regenerated on each push by
[CI](https://github.com/jkupchanko/qdrant-hyperbolic-transform/actions/workflows/benchmark.yml),
against a real Qdrant service container on infrastructure the author does not
control. The run fails if no HNSW graph was built or if no noise floor was
reported, so a green run with a table in it is itself the evidence. Open the latest
run and read the job summary.

## Install

```bash
pip install -e ".[dev]"
pytest                      # 12 tests, ~2s
docker compose up -d        # a real Qdrant, needed for every number above
```

## Layout

| path | what |
|---|---|
| `hyperbolic_qdrant/transform.py` | the document and query transforms, and the derivation |
| `hyperbolic_qdrant/depth.py` | `measure_depth()` — reports `a_med` and what to expect |
| `hyperbolic_qdrant/index.py` | create the Dot collection and upload |
| `hyperbolic_qdrant/search.py` | transform a query, call Qdrant normally |
| `hyperbolic_qdrant/benchmark.py` | three arms, replicate builds, controls, noise floor |
| `examples/product_taxonomy.py` | the demo |
| `examples/depth_ladder.py` | reproduces the depth table from a generated corpus |
| `tests/` | pins the query sign, the quantile fix, exactness, depth reporting |

## Status

Reference implementation and a reproducible result. Not a Qdrant feature, not
affiliated with Qdrant.
