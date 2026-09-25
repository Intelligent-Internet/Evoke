# High-DF low-discrimination semantic atom sidecar

Date: 2026-08-13

Status: local read-only experiment; not a product default

## Question

Can an Evoke query omit semantic atoms whose posting lists cover most of the
corpus because those atoms contribute little ranking discrimination?

The useful version of the hypothesis is not a global high-DF blacklist. It is
a query-aware admission rule for semantic atoms that combine:

- high physical traversal cost;
- high document frequency;
- low bounded score contribution for the current query; and
- a declared approximation budget or exact fallback.

## Safety boundary

The experiment used the installed local PostgreSQL 18 instance and existing
v3 Evoke indexes. It did not rebuild or mutate an index, change a catalog, alter
the Shadow rebuild, or modify the public query contract. All A/B controls were
session-local hidden test GUCs.

Raw output is under
`docs/performance/data/raw/high-df-low-discrimination-2026-08-13/`.

The deployed-code boundary was frozen independently as commit `bb6a46ea`.
It keeps retained-impact builder policy 2 with a 64-document seed cap. The
larger-cap, retained-union, centered-exception, and Seismic runs below are
research controls; none remains as an additional runtime authority.

## Surfaces

| Surface | Documents | Queries | Index |
| --- | ---: | ---: | --- |
| FIQA official | 57,638 | 648 | `evoke_beir15.docs_fiqa_bm25_idx` |
| NFCorpus official | 3,633 | 323 | `evoke_beir15.docs_nfcorpus_bm25_idx` |

SciFact was intentionally not rebuilt for this side experiment. Its local
index still used obsolete metapage v2 and the current runtime correctly
rejected it with a v3 REINDEX requirement.

## FIQA canary work telemetry

The 50-query canary ran both public native search and the page-native work
probe. DF limit means that semantic-only terms above the threshold were
omitted. Term budget admits omitted terms in ascending absolute contribution
bound until the query-level budget is consumed.

| Policy | Setting | Mean pruned postings | O@100 | Recall@100 delta | NDCG@10 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| DF limit | 75% | 39,863 | 0.9760 | 0.0000 | +0.00111 |
| DF limit | 50% | 85,676 | 0.9450 | 0.0000 | +0.00702 |
| DF limit | 35% | 140,428 | 0.8730 | -0.00250 | +0.00556 |
| Term budget | 0.5% | 40,523 | 0.9920 | 0.0000 | -0.00022 |
| Term budget | 1.0% | 61,653 | 0.9858 | 0.0000 | -0.00040 |
| Term budget | 2.0% | 86,525 | 0.9716 | 0.0000 | +0.00598 |

At approximately 40,000 omitted postings, the query-aware bound preserved
substantially more exact top-100 membership than DF alone (`0.992` versus
`0.976`). High DF therefore identifies expensive terms, but it does not by
itself predict the quality cost of omitting them.

## Full official results

### FIQA

| Policy | Setting | O@100 | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | - | 1.0000 | 0.33418 | 0.27918 | 0.61878 | 0.41774 | 83-87 |
| DF limit | 75% | 0.9657 | 0.33557 | 0.28043 | 0.61933 | 0.42027 | 86.3 |
| DF limit | 50% | 0.9315 | 0.33926 | 0.28272 | 0.62464 | 0.42446 | 85.8 |
| Term budget | 0.5% | 0.9907 | 0.33470 | 0.27962 | 0.61789 | 0.41821 | 90.8 |
| Term budget | 2.0% | 0.9725 | 0.33614 | 0.28011 | 0.62509 | 0.42173 | 104.6 |

The macro metrics can improve even when individual queries are badly harmed.
At DF 75%, five queries lost Recall@100 and the worst query lost all of its
relevant top-100 results. At a 0.5% term budget, three queries lost recall.
This does not reject an approximate product mode. It means the product must
publish the observed tail-risk distribution instead of claiming an exact or
row-safe contract.

### NFCorpus

| Policy | Setting | O@100 | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | - | 1.0000 | 0.34649 | 0.16663 | 0.29263 | 0.55910 | 58-61 |
| DF limit | 75% | 0.9852 | 0.34656 | 0.16683 | 0.29445 | 0.55933 | 59.2 |
| DF limit | 50% | 0.9251 | 0.34506 | 0.16649 | 0.29408 | 0.55439 | 63.1 |
| Term budget | 0.5% | 0.9942 | 0.34678 | 0.16668 | 0.29287 | 0.55944 | 55.5 |
| Term budget | 2.0% | 0.9794 | 0.34592 | 0.16674 | 0.29430 | 0.55772 | 59.2 |

The cross-corpus result has the same shape. A small query-aware budget is more
conservative than a comparable global DF rule, while aggressive omission
trades macro recall against MRR/NDCG and harms individual queries.

## Scale closure

The follow-up separated four possible explanations: insufficient retained
impact, physical document locality, semantic geometry, and a rank-invariant
per-term component. Each was evaluated before adding product state.

### Retained impact and global threshold

A fixed 64-posting cap reduced direct scoring work to `0.069-0.222x` over ten
surfaces, but mean O@100 ranged from `0.505` to `0.897`; minimum O@100 was at
most `0.18` and commonly near zero. Candidate coverage appeared complete only
after adding the exact residual union. Global impact thresholds did not repair
the Touche or Hotpot tails. Increasing the native ArXiv cap from 64 to 1,000
raised mean O@100 only from about `0.9920` to `0.9927`, improved minimum O@100
from `0.72` to `0.75`, and slowed p50 from about `140-143` to `162.9 ms`.

The result rejects cap expansion and global-threshold sweeps. The exact residual
union, not the retained surface, was preserving quality.

### Physical blocks and semantic geometry

Source-order document blocks do not provide portable semantic locality. At a
6,400-document budget, Touche reached mean/minimum O@100 of `0.9937/0.92` at
`0.389x` work, while Hotpot reached only `0.4947/0.13` at `0.553x` work.

The Seismic geometry search itself was cheap: about `4-7 ms` on Touche and
`11 ms` on the 5.23-million-document Hotpot surface. It was not a sufficient
candidate source:

| Surface | Geometry mass | Geometry O@100 | Geometry Recall@100 | Exact Recall@100 | Residual-union candidates | End-to-end p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Touche | 30% | 0.5373 | 0.2556 | 0.4715 | 83,204 | 116.3 ms |
| Touche | 45% | 0.7694 | 0.3755 | 0.4715 | 43,662 | 61.4 ms warm replay |
| Hotpot | 30% | 0.3023 | 0.4250 | 0.8750 | 662,400 | 1,285.1 ms |

The Touche 45% point improves coverage, but still loses substantial retrieval
quality. On Hotpot, exact residual completion is near-corpus and slower than the
about `898 ms` exact route. Geometry can reduce candidate generation work; it
cannot eliminate the linear residual evidence required by the current model.

### Centered exceptions

The rank-invariant decomposition was tested explicitly:

```text
posting_weight(t, d) = fixed_baseline(t) + signed_exception(t, d)
```

On Touche, a 2% tolerance retained `99.73%` of selected postings as exceptions;
a 5% tolerance retained `98.46%`. On FIQA the corresponding ratios were
`98.85%` and `90.76%`, with the 5% point reducing mean/minimum O@100 to
`0.9562/0.89`. The posting impacts are not concentrated around a reusable
constant. Centered signed exceptions therefore do not remove enough DF to
justify a new product format.

### Seeded BMP

The guarded seeded-BMP route remained exact only because admission rejected the
experimental stream and fell back to packed scoring. It was slower
(`303.3 ms` p50 versus `216.3 ms` exact), while a forced 20-query canary exposed
an invalid serialized format. It is not a promotion candidate.

## Query-time error-budget curve

The query-aware budget is applied after immutable term plans expose DF and BMP
impact bounds but before omitted posting streams are opened. The final native
ArXiv run used 150 fixed cross-domain queries:

| Budget | Decoded posting ratio | p50 ms | p95 ms | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 1.0000 | 217.9 | 279.6 | 1.0000 | 1.00 |
| 0.10% | 0.9781 | 216.5 | 273.6 | 0.9985 | 0.98 |
| 0.25% | 0.9542 | 214.5 | 276.4 | 0.9950 | 0.97 |
| 0.50% | 0.9362 | 216.2 | 276.6 | 0.9890 | 0.94 |

That first sweep used the original conservative `0.95` minimum-overlap floor.
A broader follow-up removed that assumption and measured the complete
work-versus-overlap curve. The decision remains query-specific: immutable run
metadata supplies DF and impact bounds, while the encoded query weights decide
which semantic streams fit within the requested bound. Lexical streams are
never omitted.

| Budget | Mean decoded reduction | Mean O@100 | P05 O@100 | P10 O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.10% | 2.2% | 0.9985 | 0.99 | 0.99 | 0.98 |
| 0.25% | 4.6% | 0.9950 | 0.98 | 0.99 | 0.97 |
| 0.50% | 6.4% | 0.9890 | 0.97 | 0.97 | 0.94 |
| 0.75% | 8.3% | 0.9839 | 0.96 | 0.97 | 0.90 |
| 1.00% | 10.4% | 0.9770 | 0.94 | 0.95 | 0.87 |
| 2.00% | 14.0% | 0.9630 | 0.90 | 0.91 | 0.87 |
| 5.00% | 18.3% | 0.9203 | 0.80 | 0.84 | 0.66 |
| 10.00% | 21.4% | 0.8619 | 0.64 | 0.76 | 0.44 |
| 20.00% | 31.6% | 0.7519 | 0.42 | 0.52 | 0.08 |
| 50.00% | 36.1% | 0.5199 | 0.13 | 0.22 | 0.02 |
| 100.00% | 38.3% | 0.1991 | 0.00 | 0.01 | 0.00 |

These are observed decoded-posting reductions, not merely the sum of streams
eligible for omission. The initial `0.75-10%` sweep still allowed the existing
semantic-BMP admission heuristic. Its occasional failed attempts consumed part
of the expected saving, especially at `5%` and `10%`. The repeated candidate
timing and the clean `20-100%` endpoint run disable BMP, which is the required
contract for this route.

The curve has two knees. Between `0.5%` and `2%`, decoded work continues to
fall while mean overlap declines gradually. Above `2%`, quality loss
accelerates. Above `20%`, the work curve itself saturates: moving from `20%`
to `100%` removes only another `6.7%` of decoded postings because lexical
streams remain exact and a very small retained term set can switch the query
to a less efficient completion path.

The first endpoint run also exposed a routing interaction. Aggressive
error-budget pruning made the semantic-BMP admission heuristic attempt a path
that subsequently fell back to TAAT, duplicating work. The clean endpoint run
therefore disables semantic BMP whenever the error-budget route is selected.
This restores monotonic decoded work and prevents two approximate engines from
being composed implicitly.

The practical `1%`, `2%`, and `5%` points were then repeated three times over
all 150 queries with full telemetry disabled:

| Budget | p50 ms | p95 ms | p99 ms | p50 gain | p95 gain | Mean O@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact | 219.2 | 280.1 | 313.8 | - | - | 1.0000 |
| 1.00% | 212.4 | 275.8 | 315.2 | 3.1% | 1.5% | 0.9770 |
| 2.00% | 209.1 | 272.5 | 312.9 | 4.6% | 2.7% | 0.9630 |
| 5.00% | 207.2 | 268.7 | 308.7 | 5.5% | 4.1% | 0.9203 |

The speedup is real but sublinear: fixed term planning, score-array, heap, and
other query costs absorb most of the decoded-posting reduction. `1-2%` is the
only plausible product range on this surface. `5%` and larger budgets lose too
much rank membership for their additional latency gain.

## Propagating the budget below the term layer

The standalone budget does not remove the corpus-sized TAAT score array or its
final document scan. A direct attempt to route a 1% budget through the existing
ordered-block scorer proved that the mathematical pruning was real but the
physical route was unusable. For one ArXiv query it reduced decoded postings
from `12.83M` to `2.09M`, documents examined from `3.12M` to `538,704`, and
scored blocks from `24,427` to `4,213`. Latency nevertheless increased from
`242.5 ms` to `10,480.2 ms` because the current BMP compatibility path
reconstructed `178,732` block metadata records from packed refs during the
query. Query-time metadata synthesis is therefore rejected; block bounds must
be a publication-time part of the sole-authority stream before this exact
route can be reconsidered.

The productive route applies the same error contract inside the existing
semantic accelerator. Terms already covered by accelerator clusters remain
untouched, preserving the accelerator candidate source. Only pure-semantic
residual terms may be omitted before their posting streams are opened. This
removes residual posting work without introducing another artifact or
lifecycle.

On the fixed 150-query cross-domain workload:

| Route | p50 ms | p95 ms | p99 ms | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact packed | 214.4 | 273.3 | 306.7 | 1.0000 | 1.00 |
| Accelerator m32 | 137.2 | 167.4 | 172.6 | 0.9918 | 0.72 |
| Accelerator m32 + 0.5% residual budget | 132.7 | 165.1 | 171.6 | 0.9836 | 0.71 |
| Accelerator m32 + 1.0% residual budget | 129.2 | 158.4 | 169.4 | 0.9765 | 0.71 |

The 0.5% point reduced mean residual postings from `3.602M` to `3.135M`
(`13.0%`) while leaving the accelerator candidate set unchanged. It improved
accelerator p50 by `3.3%`; unlike standalone TAAT pruning, the saved streams no
longer pay for a corpus score-array pass.

Document-level propagation was tested independently by lowering the residual
candidate multiplier from 32 to 24. This reduced residual candidates from
`3,200` to `2,400`. In the 20-query TREC canary it reduced documents scored and
forward rows from `3,744.6` to `2,944.6`, and forward bytes from `12.07 MB` to
`9.36 MB`. Combining m24 with a 0.5% residual budget reached `116.3 ms` p50,
versus `131.5 ms` for m32, at mean/minimum O@100 `0.992/0.93` versus
`0.999/0.98`.

The broader workload rejects that fixed document budget as a default:

| Route | p50 ms | p95 ms | p99 ms | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accelerator m24 | 129.4 | 157.3 | 166.5 | 0.9861 | 0.61 |
| Accelerator m24 + 0.5% residual budget | 125.0 | 153.0 | 165.3 | 0.9787 | 0.61 |

Across the 20 instrumented cross-domain queries, m24 reduced documents scored,
forward rows, and forward bytes by about `18.6%`, `18.6%`, and `19.9%`
relative to m32. The latency gain over m32 was only `5.7%`, while the minimum
overlap loss was material. The important result is not m24 itself: posting
savings now propagate into bounded residual candidates and forward-row work.
A future document budget must be selected from a score-bound/convergence
condition, not a smaller fixed multiplier.

Raising the existing accelerator heap factor from `0.70` to `0.75` and `0.80`
was also rejected. All three settings opened the same eight clusters, examined
the same 512 document refs, and scored the same 3,708 mean documents on the
TREC canary. The small timing differences were noise, not reduced work. The
current cluster scores do not expose the document-level convergence condition
needed here.

## Compact forward publication

The remaining document-level cost was forward completion. The legacy forward
object stored separate fixed-width term, offset, and contribution arrays and
read about `19.50 MB` per query on the 150-query ArXiv workload. A direct-row
oracle compared delta-varint term IDs plus one per-row scale and signed
quantized contributions:

| Row encoding | Projected bytes/query | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: |
| Legacy fixed-width | 19.50 MB | 0.998867 | 0.92 |
| Delta-varint + int16 | 5.17 MB | 0.998867 | 0.92 |
| Delta-varint + int8 | 3.71 MB | 0.982667 | 0.92 |

The int8 mean loss is accepted for this bounded approximate path because it is
stable on the fixed cross-domain surface and reduces projected forward bytes
by about `81%`. The old oracle timings are not product timings: that diagnostic
read the legacy row and then computed a compact projection, so it paid both
costs.

The selected v4 format changes the actual derived accelerator publication.
Each forward object now contains a byte-offset table followed by compact rows;
each non-empty row stores one float32 scale, delta-varint term IDs, and signed
int8 contributions. V4 has one readable and writable encoding: int8. Query
scoring reads only the requested row and scores the serialized bytes directly.
There is no int16 compatibility reader, quantization sidecar, document-sketch
oracle, or exact-rerank copy.

Exactness remains explicit: exact queries bypass the lossy accelerator and use
the packed root. The bounded accelerator uses compact v4 rows. Local product
qualification passed all 36 convergent lifecycle gates, including the default
accelerator route, CRUD,
asynchronous semantic completion, ordinary maintenance without repeated
inference, REINDEX, VACUUM, crash/restart recovery, derived accelerator
invalidation, and rebuild.

The Shadow ArXiv accelerator was then rebuilt in place from the immutable
49 GB root. Publication completed in 2 minutes 21 seconds without model
inference or a root rebuild. The native A/B results are:

| Workload and route | p50 ms | p95 ms | p99 ms | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-domain 150, exact | 213.78 | 275.78 | 310.71 | 1.0000 | 1.00 |
| Cross-domain 150, v4 int8 m64 | 139.17 | 161.28 | 172.05 | 0.9826 | 0.92 |
| TREC 50, exact | 220.76 | 263.60 | 290.93 | 1.0000 | 1.00 |
| TREC 50, v4 int8 m64 | 136.51 | 153.34 | 160.17 | 0.9834 | 0.92 |

After promoting the route and restarting Shadow with the installed product
binary, the same fixed workloads were repeated through `evoke_query` with no
accelerator tuning GUCs. This checks the actual session defaults rather than a
manually reconstructed preset:

| Workload and installed route | p50 ms | p95 ms | p99 ms | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-domain 150, exact override | 215.22 | 272.56 | 309.07 | 1.0000 | 1.00 |
| Cross-domain 150, product default | 138.83 | 163.71 | 172.31 | 0.9826 | 0.92 |
| TREC 50, exact override | 218.88 | 272.28 | 290.60 | 1.0000 | 1.00 |
| TREC 50, product default | 134.50 | 153.27 | 162.63 | 0.9834 | 0.92 |

The installed default is `1.55x/1.66x` faster at cross-domain p50/p95 and
`1.63x/1.78x` faster at TREC p50/p95 than exact packed scoring. Its latency and
overlap reproduce the forced m64 qualification, so promotion did not depend on
an unshipped session override. The existing policy-3 artifact remained current
after binary deployment; no root rebuild or model inference was required.

On the cross-domain workload, v4 int8 reduced p50 by `34.9%` and p95 by
`41.5%` relative to exact packed scoring. It also improved p50 by `13.4%`
relative to the old legacy-forward m64 route. The 20 instrumented
cross-domain queries read a mean `4.11 MB` of compact forward rows, versus
`22.73 MB` for the same first 20 queries through the old fixed-width forward
object. No whole forward chunk was loaded.

The native cross-domain mean/minimum overlap (`0.9826/0.92`) reproduces the
direct-row oracle (`0.982667/0.92`). The independent TREC result
(`0.9834/0.92`) confirms that the accepted loss is stable across the two fixed
surfaces. The v4 int8 publication format therefore passes its artifact gate.

## Product interpretation

Do not add a separately maintained SQL table, a second posting authority, or a
new accelerator artifact. The v4 compact forward stream replaces the legacy
forward representation inside the existing derived accelerator lifecycle.
The existing packed-run metadata provides live DF and conservative
minimum/maximum impact bounds. Policy 3 v4 int8 with `heap_factor=0.7`, bounded
cross-term residual accumulation, and candidate multiplier 64 is the current
default derived query route. Missing, stale, L0-active, unsupported, or
memory-blocked artifacts fail closed to exact packed scoring. Exact scoring is
also retained as a hidden A/B reference; it is not a second authority.

The residual error-budget and smaller candidate-multiplier profiles remain
diagnostics. Quantization and support omission compound, so they are not part
of the default. Public promotion beyond the current fixed profile still
requires relevance-metric and concurrent-throughput evidence rather than a
new threshold grid.

The reusable result is the executable error contract. For query atom `t`, the
bound is based on `abs(query_weight[t]) * max_abs_impact[t]`; the sum over
omitted atoms bounds every document's score error by the triangle inequality.
Keep this machinery hidden as a qualification oracle. It may evaluate a future
DF-controlled model, but it is not worth another product lifecycle or routing
surface for the current posting distribution.

## Decision

The broad hypothesis is real: expensive query atoms contain removable scoring
work. The engine-only geometry, cap, threshold, source-order block, centered
exception, and seeded-BMP shapes do not safely identify enough of it. Do not
run more cap or threshold grids and do not add another independently maintained
sidecar.

The native ArXiv A/B establishes a bounded quality/work trade and qualifies
the fixed v4 int8 m64 route as the default derived executor. It does not
justify more threshold grids or a second index lifecycle. Reject `5%+`
standalone budgets and fixed m24 document budgets for product use. The next
gate is native qrels and concurrent-throughput validation of the selected
m64 route; any further approximation must beat that baseline under the same
quality floor. A fixed smaller candidate multiplier is not an acceptable
substitute for a score-bound convergence condition.

The next model iteration must control corpus DF during training rather than ask
the runtime to recover a distribution the representation never learned. This
handoff is supported by [DF-FLOPS](https://arxiv.org/abs/2505.15070), which
targets observed corpus document frequency rather than treating average sparse
activation as a sufficient proxy, and by the independent
[IDF-aware FLOPS](https://arxiv.org/abs/2411.04403) result that term frequency
must enter learned-sparse efficiency control explicitly.
