# Semantic Accelerator Bounded Execution

> **Historical policy-4 development record.** The phase decisions, formats,
> measurements, and gate counts below describe the August 2026 experiments,
> not the Beta 1 serving implementation. Later lifecycle annotations do not
> requalify those measurements. See the current query-semantics and convergent
> design guides linked from the documentation map for policy 7 and baseline
> reuse. Preserve this record as experimental evidence, not as a runbook.
>
> Current authorities: [documentation map](../../README.md),
> [query semantics](../../query-semantics.md),
> [convergent design](../../convergent-segmented-index.md).

Date: 2026-08-12; bounded builder updated 2026-08-14

## Goal

Turn the root-derived semantic accelerator into a useful high-DF query route
without changing the exact packed authority or adding another mutation
lifecycle. Index bytes and one-time build duration are capacity telemetry.
Query p95/p99, bounded RSS, predictable convergence, and measurable quality
loss are the promotion gates.

The current product route uses the root-derived policy 4 v4 int8 accelerator
when it is current and eligible. Exact packed scoring remains authoritative
and is selected automatically when the derived route is unavailable or stale.
Representative Hotpot and ArXiv work that led to this final route is retained
below.

## Execution Status At This Milestone

| Phase | Status | Decision |
| --- | --- | --- |
| A residual-free canary | closed | Work fell sharply, but candidate coverage collapsed. |
| A.1 per-term residual quota | closed | Cross-term evidence was lost and quality failed. |
| C direct-row completion | retained | Exact completion no longer requires whole forward chunks. |
| A.1 exact cross-term oracle | retained diagnostic | Cross-term accumulation preserves official FIQA metrics but uses a corpus-sized score array. |
| A.1 weighted residual summary | retained hidden A/B | A 2x summary is the balanced point; final scoring remains exact. |
| A.2 30% versus 45% geometry | closed | The 45% surface increased term objects and lowered overlap; retain 30%. |
| A.3 cohesive block metadata | retained diagnostic | Term-local grouping is lifecycle-safe, but scale coverage does not pass promotion. |
| B query-aware error budget | closed | Conservative admission saves too little decoded work before quality-tail loss. |
| D integrated neighbors | closed | Weak scale anchors and residual-dominated work leave no lower-work recovery case. |
| E large-corpus routing | closed | Geometry plus exact residual completion does not beat exact packed scoring at scale. |
| F compact v4 forward route | promoted | Int8 direct-row scoring plus bounded cross-term residuals beats exact on ArXiv under the accepted overlap floor. |

The current staged binary passes all 37 mutable SAE lifecycle gates after the
compact writer and default route were connected to the existing
derived-accelerator worker. These include default-route telemetry, exact
fallback, CRUD invalidation, worker rebuild, REINDEX, VACUUM, and restart.

The current format is bounded-forward-int8 builder policy `4`. It keeps at
most 64 seed documents per selected term, uses delta-varint term IDs and one
signed int8 contribution per row entry, and records the cap in directory wire
version 3. The builder reads one source term at a time and spills its document
transpose to PostgreSQL temporary files before bounded publication, so total
corpus postings are not resident. The cap deliberately bounds the derived
candidate surface; it is not intended to reproduce each source posting list.
Policies 1 through 3 are historical evidence, not readable runtime fallbacks.
Older directories are reported as `stale_policy`, rebuilt by normal
maintenance, and cannot enter the default accelerator route.

## Invariants

- One packed semantic authority, one manifest root, and one mutation lifecycle.
- Accelerator term, forward, and optional neighbor data share one source-root
  checksum and one directory publication.
- Foreground CRUD never rewrites accelerator metadata. A compatible published
  directory remains the serving generation across manifest-only successors,
  including an active L0 tail, until debt-based convergence replaces it.
- A compatible stale baseline has no age deadline. It runs once with bounded
  overfetch and current COW/TID/predicate revalidation; low-volume
  post-baseline additions may remain absent until periodic low-debt maintenance
  publishes a replacement. Missing, incompatible, corrupt, or memory-blocked
  derived data falls back to exact packed scoring.
- Every approximate result reports policy, candidate work, forward reads,
  omitted support, score-error bound, and fallback state.
- No qrels, dataset identity, or per-dataset threshold enters routing.

## Phase A: Residual-Free Candidate Canary

The current bounded path prunes accelerated clusters but then expands every
unaccelerated query term into a residual document union. On FIQA this creates
about 31,000 candidates and opens all forward chunks, dominating latency.

1. Add an A/B-only policy that lets accelerated high-DF terms generate the
   candidate set without residual posting expansion.
2. Continue exact full-query scoring for every generated candidate through the
   unified forward stream. Terms omitted from candidate generation still
   affect final scores.
3. Preserve `h0` plus residual union as the exact accelerator diagnostic.
4. Record whether residual candidate generation was omitted and keep existing
   cluster, candidate, chunk, and overlap telemetry.

Gate A1: at least 4x fewer candidate documents or forward-chunk bytes than the
current bounded path. Stop if the candidate set remains corpus-wide.

Gate A2: mean O@100 at least 0.995 on FIQA/TREC-COVID/Quora, minimum O@100 at
least 0.95, macro Recall@100 loss no worse than 0.001, and no material NDCG@10
or MRR@20 loss. A failed row cannot be hidden by macro averaging.

### Phase A result

The native FIQA canary failed Gate A2 and closed the residual-free branch.
Across 32 fixed queries, omitting residual candidate generation changed p50
from about 350 ms to 60.6 ms and reduced exact-scored documents from about
29,799 to 4,058. It also reduced forward reads from all 15 chunks to one.
However, mean O@100 fell from 0.999375 to 0.062813 and minimum O@100 fell to
0.02. The exact score authority was unchanged; candidate coverage collapsed.

This gives a precise next boundary. Selected high-DF geometry is an effective
work reducer but is not a sufficient candidate source. SeismicWave neighbors
cannot be used to hide that missing source because they only expand existing
anchors. Before durable graph metadata is added, the residual stream needs a
bounded, score-aware candidate summary.

## Phase A.1: Residual Heavy-Hitter Candidate Summary

Replace the all-document residual union with a bounded positive-contribution
summary while still decoding residual posting streams in their existing exact
order.

1. Feed residual posting contributions into a weighted heavy-hitter sketch or
   another bounded accumulator with a stated additive error bound.
2. Merge its document candidates with geometric candidates, deduplicate them,
   and compute the final score from the complete unified forward stream.
3. Report residual mass, retained candidate count, sketch error, and whether
   the kth boundary is unresolved. An unresolved `certified` query continues
   exact; only explicit `bounded` mode may return under its declared contract.
4. Compare the sketch with a simple per-term top-impact baseline. Retain the
   simpler design if quality and work are equivalent.

Gate A.1: reduce residual forward candidates by at least 4x while preserving
mean O@100 at 0.995, minimum O@100 at 0.95, and the row-level metric floors.
If no bounded residual summary passes, stop this candidate source rather than
adding thresholds.

### Phase A.1 initial result

The simple per-term top-contribution baseline reduced mean residual candidates
from 31,121.5 to 1,421.3 and all exact-scored documents from 32,942.9 to
5,315.3. It still read 15 of 15 forward chunks for every query because the
retained document IDs were distributed across the corpus. FIQA p50 changed
only from 342.7 ms to 328.8 ms, while mean O@100 was 0.41375 and minimum O@100
was 0.23. This baseline fails both quality and useful-work gates.

Do not sweep its multiplier. A weighted heavy-hitter summary remains a
possible quality improvement because it can retain cross-term accumulation,
but it could not be evaluated fairly while completion retained the measured
forward-read floor. Phase C direct-row completion has now removed that floor,
so the next canary is an exact cross-term residual accumulator with the same
fixed candidate budget. This is an oracle for the summary structure, not a
product memory model. Only if it passes the quality gate will a bounded
SpaceSaving-style implementation be justified.

### Phase A.1 cross-term result

The exact cross-term accumulator confirms that per-term quota partitioning was
a major source of candidate loss. The first 16-query canary moved from mean
O@100 0.889375 at 1,600 residual candidates to 0.98875 at 6,400. The complete
648-query FIQA run at 7,200 candidates preserved exact NDCG@10, Recall@100, and
MRR@20; MAP@100 differed by less than 0.000001. Mean O@100 was 0.998827, but
three queries remained below 0.95 and the minimum was 0.80.

The oracle proves cross-term candidate accumulation but is not a product memory
model. Its float array scales with every corpus document and therefore exceeds
the 64 MB query working-set at about 16.8 million document slots. It remains a
hidden reference and fails closed to exact packed scoring above that limit.

### Phase A.1 weighted summary result

A positive weighted Space-Saving summary now approximates the oracle without a
corpus-sized score array. It preserves the standard estimate/error invariant:
for every retained item, true weight is between `estimate - error` and
`estimate`, while maximum replacement error is exposed in query telemetry.
Final ranking does not use the estimate. The summary selects candidates, then
the complete unified forward stream computes every final score.

This follows the bounded-counter contract in
[Efficient Computation of Frequent and Top-k Elements in Data Streams](https://www.cs.ucsb.edu/sites/default/files/documents/2005-23.pdf),
adapted to positive weighted posting contributions. It is not used as an exact
score approximation.

On the first 64 official FIQA queries at a 7,200 residual-candidate output:

| Summary capacity | Mean O@100 | Minimum O@100 | p50 | Official metrics |
| --- | ---: | ---: | ---: | --- |
| 7,200 (1x) | 0.994531 | 0.91 | 112.3 ms | equal to exact |
| 14,400 (2x) | 0.998594 | 0.95 | 112.2 ms | equal to exact |
| 28,800 (4x) | 0.998750 | 0.97 | 107.1 ms | equal to the oracle |
| 57,600 (8x) | 0.998750 | 0.97 | 108.0 ms | equal to the oracle |

The complete 648-query 2x run preserved exact NDCG@10, Recall@100, and MRR@20;
MAP@100 changed from 0.274312502 to 0.274313169. Relative to the corpus-sized
oracle, mean O@100 was 0.999491 and minimum O@100 was 0.91. Relative to exact,
mean O@100 was 0.998688 and minimum was 0.81. The summary therefore adds little
loss; the remaining hard rows are caused primarily by the 7,200-candidate
budget.

The fixed canary points are:

- `balanced`: 7,200 output candidates and a 14,400-counter summary. It uses
  about 1.05 MiB for counters, heap, and hash slots, independent of corpus
  document count.
- `quality`: 14,400 output candidates and a 28,800-counter summary. It uses
  about 2.10 MiB for the same structures. On all nine FIQA boundary outliers,
  minimum O@100 reached 0.95 and the summary exactly matched the oracle.

Increasing the quality output to 16,000 raised the outlier minimum only to
0.96 and did not change qrel metrics. Do not widen further on FIQA. Its exact
packed p50 is about 61 ms, versus about 111 ms for balanced bounded execution,
so the product router must choose exact on this corpus. The bounded points are
for scale qualification, not FIQA promotion.

The summarized route now iterates its unique candidate vector directly. It no
longer writes candidates into a corpus bitmap and scans every document slot to
recover them. The unbounded residual-union route retains its bitmap because it
has no bounded candidate vector. This preserves scores while preventing a
bounded PubMed/ArXiv route from reintroducing an O(corpus) completion pass.

## Phase A.2: Geometry Coverage Audit

The current native builder accelerates the smallest high-DF term set covering
30% of corpus posting mass. The representation oracle also validated a 45%
mass point. Rebuild only the disposable accelerator at 30% and 45%, then rerun
the same residual-summary candidate policy.

Gate A.2: promote 45% only when its reduced residual work pays for the added
term objects and does not lower overlap. This is a corpus-mass policy, not a
dataset-specific setting. Do not use graph expansion to compensate for a
coverage setting that already fails without it.

### Phase A.2 result

The 45% builder expanded the FIQA accelerator from 178 to 486 term objects but
reduced cross-term candidate quality. Across the fixed 16-query canary, its
mean/minimum O@100 was 0.794375/0.51 at 1,600 candidates and 0.964375/0.83 at
6,400. The retained 30% surface reached 0.98875/0.94 at the comparable 6,400
point. Heap-factor changes did not recover the loss. The 45% geometry is
rejected; 30% remains the only retained coverage policy.

## Phase A.3: Cohesive Block Metadata

The block metadata experiment tested whether a second summary level can avoid
opening high-DF posting blocks. The hierarchy is motivated by
[Dynamic Superblock Pruning](https://arxiv.org/abs/2504.17045), but the useful
condition is not hierarchy alone: documents assigned to one block must also be
geometrically cohesive, otherwise the summary bound approaches the union of
every active query coordinate.

For a nonnegative query and document postings, the coordinate bound and the
centroid-ball bound tested by the oracle are:

```text
U_max(B, q)  = sum_j q_j * max_{d in B}(x_dj)
U_ball(B, q) = q dot c_B + norm(q, 2) * max_{d in B} norm(x_d - c_B, 2)
```

Both are conservative upper bounds. Their failure therefore says that the
blocks are not tight enough for exact pruning; it is not a threshold-tuning
failure. The retained summary score is used only for block order and bounded
candidate generation. It is never presented as an exact certificate.

Three qrels-free negative controls closed the unsafe or ineffective shapes:

- A 13-level source-order hierarchy on the ArXiv surface had an oracle median
  sparse-metadata ratio of 0.5316, but the hard `machine learning` query still
  required 0.9544 of the metadata. Fixed levels regressed important queries.
- Exact coordinate-maximum summaries examined 99.996% of FIQA postings, while
  exact centroid-radius bounds examined 100%. These bounds are safe but too
  loose for learned sparse vectors.
- At an equal candidate budget, global SimHash clustering reduced summary hits
  by 9.69% but increased posting work by 5.83%, with no quality gain. Hash
  proximity is not a sufficient surrogate for query-weighted score geometry.

The positive control uses the
[Seismic](https://arxiv.org/abs/2404.18812) shape: construct
centroid-cohesive blocks inside each selected term posting list, rank block
summaries for the query, generate a bounded candidate set, and compute final
scores from the existing unified forward stream. This is approximate candidate
generation followed by exact candidate scoring, not an exact block bound and
not a second authority.

Current-source side-oracle results are:

| Surface | Candidate + exact completion p50 | Exact p50 | Mean O@100 | Metric result |
| --- | ---: | ---: | ---: | --- |
| FIQA | 23.01 ms | 23.92 ms | 0.999844 | NDCG/MAP/Recall/MRR equal; exact remains the cheaper-tail route |
| TREC-COVID | 44.62 ms | 55.39 ms | 0.997800 | NDCG/MRR equal; MAP and Recall do not regress |

This validates the representation, not product promotion. The standalone
sidecar bytes are not a product storage target. The product implementation now
uses the scalable term-local centroid builder from the existing root-derived
accelerator worker, publishes through the same directory CAS, and invalidates
with the same root mutation. It adds no metapage, worker, encoder pass, score
authority, or retirement path. All 34 mutable SAE lifecycle gates pass,
including mutation invalidation and rebuild after convergence without model
inference.

Gate A.3 did not pass native scale qualification. The grouping remains useful
as a bounded diagnostic surface under the existing derived-data lifecycle, but
it is not a promoted query route. Cheap or uncertain queries remain on exact
packed scoring.

## Phase B: Bounded Support And Error Contract

Residual-free generation was only a canary. The bounded admission diagnostic
constrains which semantic-only query atoms may be omitted from approximate
scoring while retaining an explicit per-document score-error bound.

1. Reuse the conservative per-term absolute score bound already implemented
   by the semantic error-budget diagnostic.
2. Order eligible terms by logical postings saved per unit of absolute error.
3. Apply omission after term-plan metadata is available and before the posting
   stream is opened. Lexical terms, mutable L0 projections, and missing bounds
   fail closed to exact execution.
4. Measure decoded postings and top-k movement; do not infer speed from logical
   DF alone.
5. Add support floors or exact certification only if the unrestricted oracle
   first produces a material speed margin.

Gate B: each preset must have a monotone quality/work curve and preserve its
declared per-query bound. Otherwise retain only exact routing.

### Phase B result

The complete native ArXiv A/B used 150 fixed cross-domain queries. Relative to
the exact route, error budgets of 0.10%, 0.25%, and 0.50% reduced mean decoded
postings by `2.2%`, `4.6%`, and `6.4%`. Their p50 latency ratios were `0.994`,
`0.984`, and `0.992`; p95 ratios were `0.979`, `0.989`, and `0.989`.

Quality movement arrives before a useful speed margin. Mean/minimum O@100 was
`0.9985/0.98` at 0.10%, `0.9950/0.97` at 0.25%, and `0.9890/0.94` at 0.50%.
The best p50 improvement was only `1.6%`, and the best p95 improvement was
`2.1%`. This fails the material-speed gate. A semantic-support floor can only
retain more terms and reduce these already-small savings, so no follow-up sweep
is justified. Keep the controls hidden and exact packed scoring as the only
product route.

## Phase C: Fine-Grained Forward Completion

The current directory records the forward document shift, so completion
granularity can be changed without creating another authority.

1. Compare 256, 512, 1,024, and 4,096-document forward objects using the same
   candidate sets. Also evaluate independently framed rows or small row groups
   when distributed candidates still open nearly every object.
2. Measure bytes loaded, object/page reads, decompression, p50/p95/p99, builder
   RSS, and final bytes. Storage growth and build time are acceptable when
   bounded and when query tail latency improves.
3. Select one corpus-independent granularity or an object-size target derived
   from posting bytes. If fixed chunks remain span-bound, add a checksummed
   row-offset table and independently decodable row groups inside the same
   forward object. Do not create a second score authority or tune by dataset
   name.
4. Keep directory and object validation strict; cancellation may leave only
   unreachable derived objects.

Gate C: materially lower forward bytes and p95 on sparse candidate completion
without more than 5% regression on dense candidate completion.

### Phase C direct-row result

The existing forward v3 object already contained authenticated row offsets, so
the first implementation added no persistent format and no second authority.
It range-reads one retained row from the same forward object and computes the
same complete unified score. Unit parity and the 32-gate mutable SAE lifecycle
both pass.

On the retained 57,600-document FIQA root, the bounded residual route stopped
opening 15 complete forward chunks and instead loaded a mean 5,315.25 rows and
10.90 MB. Its p50 changed from 328.8 ms to 59.2 ms at `h0.7` and 58.9 ms at
`h1`, versus 61.0 ms for the exact packed route. Mean and minimum O@100 stayed
exactly at the previous candidate-source values, 0.41375 and 0.23. Therefore
fine-grained completion passes its implementation/parity gate and exposes the
remaining problem cleanly: candidate generation, not final scoring I/O.

Do not promote per-document range reads as the only product completion policy.
At larger candidate density, use the same row offsets to coalesce nearby rows
or load a complete chunk when that is cheaper. The routing decision must be
based on touched-page density, not dataset identity.

## Phase D: Integrated SeismicWave Neighbor Metadata

Neighbor expansion is a quality recovery mechanism, not a parallel index.
The adjacent evidence is
[SeismicWave](https://arxiv.org/abs/2408.04443), which combines ordered block
traversal with graph expansion. In Evoke, neighbor IDs may only repair a
measured candidate-coverage loss; they cannot become score authority.

The graph cannot be represented by status counters alone: query-time neighbor
expansion needs durable adjacency bytes. Those bytes are nevertheless ordinary
children of the existing accelerator directory, in the same sense as current
term and forward objects. They do not introduce another index relation,
metapage, manifest root, worker class, mutation stream, or score authority.

- Extend accelerator directory v2 in place with an optional neighbor-chunk
  section. Each entry covers a contiguous document range and owns one bounded
  adjacency object. The manifest continues to contain only the existing
  `semantic_accelerator_directory` reference; there is no second manifest root
  or separately visible publication.
- Reuse the forward document shift initially so the same candidate-density
  planner can choose row, coalesced-range, or complete-chunk reads for scores
  and neighbors. A later independent shift requires native evidence.
- Bind adjacency to the same source authority checksum and publish it in the
  same final directory CAS. Query never observes a partially published graph.
- Neighbor IDs only expand candidates. All final scores still come from the
  unified forward stream; adjacency is never score authority.
- A root mutation does not mutate or immediately invalidate the published
  accelerator. It remains an immutable baseline while query-time candidate
  validation rejects changed rows; post-baseline additions may wait for the
  next debt-driven refresh. The worker may reuse
  an unchanged term or adjacency object only when a term-local source checksum
  proves identity; otherwise it rebuilds that object from completed postings.
  This is incremental derived publication, not incremental graph mutation.
- Reclamation, restart, replication, cancellation, and memory admission reuse
  existing accelerator object rules.
- Build adjacency only after the same worker has a complete source-root
  snapshot. Store document-neighbor IDs in bounded chunks referenced by the
  accelerator directory, then publish the directory once. There is no graph
  metapage, graph root, foreground graph mutation, or graph-specific worker.
- The worker writes term, forward, and neighbor children first, validates all
  refs and non-overlap, writes one directory, then publishes one successor
  manifest. Cancellation leaves only unreachable derived pages, reclaimed by
  the existing retirement path.
- Compaction and fold do not rewrite neighbor data independently. Changed source
  authority makes a replacement due but leaves the prior baseline readable
  until the complete successor is atomically published. Reuse is an optional
  worker optimization after correctness, never an additional lifecycle state.
- Status and telemetry add neighbor bytes/chunks, anchors expanded, unique
  candidates admitted, and completion rows. These counters belong under the
  existing `semantic_accelerator` object.
- Prefer term-local neighbor reuse only when its term posting checksum and
  geometry options are unchanged. Otherwise rebuilding the complete derived
  directory is the safe initial implementation.

Gate D: enable neighbors only if they recover a failing bounded row by at
least 50% of its O@100 loss at lower total work than widening the base
candidate heap. Otherwise omit graph metadata.

Phase D closes without a product implementation. The scale audit found weak
base-anchor coverage and residual evidence that approaches a corpus scan on
Hotpot. Neighbor expansion can only add candidates reachable from those weak
anchors; it does not eliminate the missing residual score mass. There is no
measured lower-work recovery case that meets Gate D.

## Phase E: Routing And Scale Qualification

1. Calibrate a monotone host-level work estimate from postings, clusters,
   candidate documents, forward bytes, and exact packed work.
2. Route cheap queries to exact packed scoring before accelerator allocation.
3. Test exact, residual-union bounded, residual-free bounded, fine-forward, and
   optional neighbor recovery on isolated PubMed/ArXiv roots.
4. Run cold/hot concurrency 1/8/32, cancellation, restart, CRUD invalidation,
   convergence, VACUUM/reclamation, root races, and physical-replica replay.

Promotion requires high-DF p50 improvement of at least 3x and p95 improvement
of at least 2x, bounded backend/runtime RSS, less than 5% BM25-only p99
regression, and predictable worker completion without busy loops. Exact remains
available for every query and remains the default until this gate passes.

### Phase E result

The scale run separates fast geometry from total retrieval cost. On the
5.23-million-document Hotpot surface, Seismic search took about `11 ms`, but
its 800 geometry candidates covered only `0.3023` of exact top-100 membership.
Their qrels Recall@100 was `0.4250`, versus `0.8750` for exact. Restoring exact
coverage required a mean residual union of 662,400 documents and about
`1,285 ms` p50 end to end, slower than the approximately `898 ms` exact route.

Touche has a friendlier distribution. Increasing selected geometry from 30%
to 45% raised geometry-only O@100 from `0.5373` to `0.7694` and reduced the
residual union from 83,204 to 43,662 documents. Geometry-only Recall@100 still
reached only `0.3755`, versus `0.4715` exact. This cannot qualify a global
policy, and the warm `61.4 ms` replay is not comparable to a differently cached
exact run.

The native ArXiv cap control reached the same boundary. Increasing retained
documents per term from 64 to 1,000 changed mean/minimum O@100 only from about
`0.9920/0.72` to `0.9927/0.75`, while p50 worsened from about `140-143` to
`162.9 ms`. Cap expansion is rejected. A seeded-BMP canary was exact only after
its admission guard fell back to packed scoring and was slower than exact; a
forced path exposed an invalid stream format and remains rejected.

Phase E closed policy 2 geometry as a promoted product route. The later compact
v4 forward result below supersedes that executor decision without changing the
single-authority or lifecycle conclusion. Graph metadata remains unjustified:
it would expand anchors whose cross-corpus coverage is already too weak,
without removing the residual evidence that dominates total work.

### Phase F result

Policy 3 replaces the legacy fixed-width forward object with direct compact
rows: one float32 scale, delta-varint term IDs, and signed int8 contributions.
On Shadow ArXiv, cross-term64 reduced exact packed p50/p95 from
`213.78/275.78 ms` to `139.17/161.28 ms`; mean/minimum O@100 were
`0.9826/0.92`. The independent TREC-50 surface measured `136.51/153.34 ms`
with `0.9834/0.92` overlap. The artifact rebuilt from the immutable root in
2 minutes 21 seconds without document inference.

This qualified policy 3 cross-term64 as the first default derived executor; the
current product format has since advanced to policy 7. The packed root remains
the sole score authority and exact fallback. Compatible baseline staleness and
active L0 debt may use the declared approximate route with current-row
revalidation. Unsupported formats, contract mismatch, corruption, or failed
memory admission must not.

Deployment qualification repeated both fixed surfaces through the installed
Shadow binary without route overrides. Cross-domain 150 measured
`138.83/163.71 ms` p50/p95 with `0.9826/0.92` mean/minimum O@100, versus
`215.22/272.56 ms` exact. TREC 50 measured `134.50/153.27 ms` with
`0.9834/0.92`, versus `218.88/272.28 ms` exact. The current ArXiv policy-3
artifact was accepted after deployment, proving that promotion is a query-code
upgrade when the artifact is already current rather than a mandatory corpus or
encoder rebuild.

### Phase G bounded-builder closure

Policy 4 preserves the compact v4 query representation and cross-term64 route,
but replaces the corpus-resident publication builder with a term-streaming,
temporary-file-backed transpose. Workspace admission depends on document
metadata, vocabulary state, the largest decoded term, bounded tuplesort memory,
and fixed publication scratch rather than total corpus postings. Lexical-neutral
residual rows now use the product BM25 TFC, including document-length
normalization, rather than treating term frequency as an impact.

The staged product passes 37 mutable SAE lifecycle gates and the complete
isolated runtime-service smoke, including cancellation during a partial remote
HTTP response. On Shadow, policy 4 is published and query-ready for PubMed and
Washington chunks. The fixed 20-query PubMed closure measured default
`333.84/359.86 ms` p50/p95 versus exact `362.16/426.50 ms`, with mean/minimum
O@100 `0.9785/0.95`. Washington chunks measured about `103-107 ms` warm versus
about `156 ms` exact. These are deployment evidence, not a claim that every
corpus benefits; exact remains the automatic fallback.

## Execution Order And Stop Rules

Phase A closed the pure residual-free route, Phase C removed its completion-I/O
blocker, A.1 retained bounded summary machinery, A.2 rejected the 45% surface,
and A.3 retained term-local grouping only as a lifecycle-safe diagnostic.
Phase E closed the legacy-forward route; Phase F promoted compact v4
cross-term64; Phase G retained that query shape while bounding publication
memory and correcting lexical residual scoring. Do not build graph metadata,
expand caps, add support-floor grids, or run another threshold sweep to replace
these structural gates. Further engine work must compare against the promoted
route, while model training must control observed corpus DF directly. Exact
packed scoring remains the automatic fallback and qualification reference.
