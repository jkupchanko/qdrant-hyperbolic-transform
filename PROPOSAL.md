# Proposal: does anyone need this?

This repository shows that Qdrant **can** serve a deep hyperbolic embedding
well, with no change to the engine. It does not show that anyone **should have**
one. Those are different claims and only the second is worth product attention.

This is a plan to settle the second one, written so the result is useful either
way. If it comes out negative, that is a quarter nobody spends.

---

## Where this stands today

**The ranking half already landed.** `acosh` in Formula Query was merged on
17 August 2026 ([qdrant#10231](https://github.com/qdrant/qdrant/pull/10231)).
It is in `dev`, not in v1.19.1, so it ships in the next minor. Nothing below
asks for engine work.

**The indexing half is this repository.** A preprocessing layer: convert the
vectors, create an ordinary `Distance.DOT` collection, search it normally.
Qdrant does not learn anything about hyperbolic geometry.

**The two are not alternatives, they are halves of one pipeline.** Measured on
the 5,595-point Google Product Taxonomy, recall@10:

| pipeline | w=10 | w=50 | w=1000 |
| --- | ---: | ---: | ---: |
| Euclidean prefetch + `acosh` rescore | 0.257 | 0.472 | 0.920 |
| transform + `acosh` rescore | **0.717** | **0.957** | **1.000** |
| transform, no rescore | 0.723 | 0.723 | 0.723 |

The third row is the point. The transform alone is **flat in width**: more
candidates buy nothing, because taking the top ten straight from the prefetch
cannot depend on how many were fetched. It is a good candidate generator and a
poor ranker. `acosh` does the ranking and is not optional.

Together: 0.95 at **50** candidates rather than ~1,500, and 0.669 ms rather
than 3.943 ms.

---

## The gap, stated so it cannot be argued around

Three objections, and only two have answers.

**"Compared to what?"** The headline `0.905 MAP` against a flat `0.658` is
reconstruction of relationships the embedding saw in training. Asked for a
category's direct parent instead, the same embedding scores `0.539`. That is a
reconstruction result, not a retrieval result, and it should never be quoted as
one.

**"What does it cost Qdrant?"** Nothing. No metric, no index type, no
dependency. This objection has an answer.

**"Who actually has a deep hyperbolic embedding?"** This one has no answer yet,
and it is the one that decides everything. Measured on real weights, pretrained
hyperbolic CLIP models sit in a thin shell close to the origin:

| model | median hyperbolic radius |
| --- | ---: |
| HyCoCLIP-ViT-S | 0.627 |
| MERU-ViT-S | 0.872 |

That is the near-flat region. At that radius plain cosine recovers `0.9997` of
the exact geodesic ranking, so none of this machinery is needed. Every corpus
where the transform helps is one we trained ourselves.

**So the question is not whether the transform works. It is whether a
hierarchy-bearing embedding is worth training in the first place.**

---

## The experiment

One question: **on structure the embedding never saw, does hyperbolic plus this
pipeline beat a flat model at production dimension?**

### Arms

| arm | dims | index | rescore | what it isolates |
| --- | ---: | --- | --- | --- |
| A | 384 | cosine | none | the flat baseline anyone would reach for |
| B | 5 | Euclid on raw Poincaré | `acosh` | what the published article does |
| C | 5 | transform, `Distance.DOT` | `acosh` | the full pipeline |
| D | 5 | transform, `Distance.DOT` | none | proves the rescore is load-bearing |
| E | 5 | exact search | exact geodesic | the ceiling any index is chasing |

A is the arm that matters. If A wins, the rest is a curiosity.

### Task

**Held-out edges, not reconstruction.** Remove 20% of the taxonomy's
parent-child edges before training. Train on what remains. Then ask every arm
to retrieve the true parent of a node whose edge was removed.

No arm has seen the answer. This is the test the current `0.539` number fails,
and it is the only one that supports a product claim.

### Corpus

It must be in the regime where the transform does anything: **median hyperbolic
radius above 2**. Report `a_med` before any recall number, so nobody argues
afterwards about whether the corpus was favourable.

A corpus below that threshold is a valid result too, and it says use cosine.

### Metrics

- recall@10 and MRR of the true parent, held-out edges only
- candidates examined to reach recall 0.95, per arm
- server-side latency at matched recall, not at matched width
- build-to-build spread over at least 3 builds, reported as a noise floor

A gap smaller than the noise floor is not a result.

### Pre-registered, so the goalposts cannot move

Written down before the run:

1. **A wins if it is within the noise floor of C.** Parity counts as a win for
   the flat model, because it is simpler and already in production.
2. `a_med` is reported first, whatever it is.
3. The flat baseline is a real production-dimension model, not a crippled one.
4. If C beats A only on the corpus we trained, that is reported as the
   limitation it is.

### What a negative result buys

A definite answer that hierarchy-bearing embeddings do not pay at retrieval
time, costing one experiment rather than a roadmap slot. That is worth running
for on its own.

---

## What this asks for

Nothing from the engine. `acosh` is already merged, and the transform is a
preprocessing layer that lives outside Qdrant.

The ask is a decision on whether the question is worth answering, and if so,
agreement on the arms and the pre-registered conditions **before** the numbers
exist.
