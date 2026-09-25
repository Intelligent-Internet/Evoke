# Page-Native Learned-Sparse Query Cost

Date: 2026-08-10, implementation status updated 2026-08-11

## Decision

The PubMed latency problem is not primarily a cache problem. The hot query
still performs exact term-at-a-time accumulation over about 25 million
postings and writes scores for almost six million documents. Keeping pages hot
removes page faults, but does not remove the decoding, multiplication, branch,
or random score-array write work.

The packed v7 posting stream and its BMP directories remain the sole exact
authority. They close duplicate storage and make selective exact pruning
possible, but they cannot make a non-selective high-DF query cheap. Fixed DF
pruning, fixed query-term cuts, and a corpus-global sparse IVF were evaluated
and rejected: they removed logical postings without reducing the number of
documents touched enough to improve latency, or they lost most of the exact
top-100 candidates.

The first route that materially changes this cost is a **per-term geometric
candidate accelerator** derived from the exact packed authority. It groups a
posting list into geometrically cohesive document clusters, ranks sparse
cluster summaries for the complete query, and exact-scores only the selected
documents from the packed stream. A fixed per-list cap initially failed at
larger corpus sizes. Replacing it with a qrels-free 45% retained-posting-mass
target, a minimum cap of 1,000, and eight-way exact candidate completion
closes that scale defect.

The shared graph-free quality candidate (`query_cut=16`, `heap_factor=0.7`)
has 0.99837 mean top-100 overlap on seven BEIR surfaces. Macro NDCG@10 is
unchanged, MAP@100 changes by +0.000050, Recall@100 by +0.000007, and MRR@20
by less than +0.000001. Its geometric-mean speedup is 7.0x on the four
surfaces with at least 25,000 documents. NFCorpus and SciFact are too cheap to
benefit and must remain on the exact route. This is still an explicit bounded
mode, not a replacement for the exact default, and the timings are from the
external representation oracle rather than PostgreSQL.

The exact path remains a block-forward learned-sparse executor:

```text
query lexical terms
    -> exact lexical accumulation
query semantic terms
    -> b256 conservative superblock bounds
    -> b16 conservative fine bounds
    -> exact forward scoring for competitive b16 blocks only
    -> exact top-k
non-selective bounds
    -> abandon BMP attempt
    -> existing exact TAAT path
```

This is based on BMP rather than a conventional BM25 WAND assumption. Learned
sparse indexes have longer queries, smaller vocabularies, higher-DF posting
lists, and different score distributions; the BMP paper identifies precisely
this mismatch and proposes document-range block filtering for rank-safe search.
See [Faster Learned Sparse Retrieval with Block-Max Pruning](https://arxiv.org/abs/2405.01117).

The implementation publishes one exact compact semantic authority in segment
payload version 7 and neutral fold version 4. Local correctness and lifecycle
gates, plus an isolated 100,000-row PubMed sidecar, pass on that format. The
geometric accelerator is now integrated into the PostgreSQL page-native path
as disposable, root-scoped derived data. The packed stream remains the sole
score authority. Missing, corrupt, stale, L0-active, memory-blocked, or
unsupported accelerator state fails closed to the exact packed executor.
Native FIQA and PubMed/ArXiv latency gates remain open; integration is not a
scale qualification by itself.

## Production Evidence

The read-only Shadow PubMed audit used:

- index size: about 182 GB;
- document slots: 8,214,026;
- query: `machine learning`, `k=100`;
- query terms: 104;
- exact query postings: 25,030,081;
- documents receiving a score: 5,953,678;
- semantic postings: 24,457,468, or 97.7% of query posting work;
- hot exact TAAT latency before this structural prototype: about 0.35-0.38 s;
- cold latency: about 1.15 s after the earlier page-reader repair.

The same exact score surface was audited at several fixed document block
sizes. A block is competitive only when its conservative maximum can still
reach the exact final kth score.

| Block | Competitive postings | Competitive documents | Bound bytes |
| ---: | ---: | ---: | ---: |
| 8 | 0.443% | 0.153% | 148.9 MB |
| 16 | 2.334% | 1.193% | 119.1 MB |
| 32 | 12.050% | 8.296% | 87.7 MB |
| 64 | 43.286% | 36.439% | 59.0 MB |
| 128 | 85.887% | 81.760% | 36.5 MB |

The b16 result is the important signal: only 584,289 of 25,030,081 postings
belong to competitive blocks, covering 98,016 document slots. The existing
b128 ordered path cannot exploit this because its bounds remain competitive
for most of the corpus. A coarse block is therefore useful only as a directory
level; it cannot be the final pruning unit.

The raw read-only result is retained outside the repository at:

```text
/tmp/evoke-shadow-pubmed-machine-learning-block-cost-20260810.json
```

## Implemented Exact Structure

The new `evoke_semantic_bmp` module builds a signed, exact block-forward index:

- b16 document blocks;
- b256 superblocks containing 16 b16 blocks;
- a sorted term directory;
- outward-quantized per-term block minima and maxima;
- outward-quantized superblock minima and maxima;
- block-major forward records with 16-bit document masks;
- exact float impacts;
- format checksum and strict structural validation.

Signed query weights and signed document impacts are supported. Bounds select
the correct minimum or maximum impact according to the sign of the query
weight and are rounded outward. Pruning uses strict `< kth`, not `<= kth`, so
equal-score ties remain exact.

The structure is stored in:

- segment payload version 7;
- neutral fold version 4;
- one packed impact/delta stream used by query, exact fallback, merge,
  compaction, and fold.

Semantic impacts are no longer copied into generic legacy posting runs. Version
6 payloads and version 3 folds are rejected as rebuild-required rather than
carried as a second runtime authority.

The page-native scorer uses the BMP path only when all of the following hold:

- no active or pending L0 overlay;
- every document slot is visible;
- every semantic run has compatible BMP metadata;
- the query working set remains within the existing 64 MB backend limit.

Lexical postings are accumulated exactly first. Semantic superblocks are then
ranked by safe bounds. One seed batch of eight b256 superblocks establishes an
initial exact kth score before the executor scans the compact b16 bounds once.
If the seed blocks plus the remaining competitive blocks exceed 25% of all
b16 blocks, it records a BMP fallback and runs exact compressed TAAT over the
same packed authority. This is a cost guard, not an approximation or a switch
to a legacy representation. Active L0 surfaces use the existing exact
streaming/projected route until maintenance seals them into the compact
authority.

Telemetry now preserves failed-attempt cost instead of hiding it:

```text
semantic_bmp_attempted
semantic_bmp_fallback
semantic_bmp_query_path
semantic_bmp_super_ref_reads
semantic_bmp_ref_reads
semantic_bmp_record_reads
semantic_bmp_postings_examined
```

## Synthetic Exact Benchmark

The focused C benchmark uses one million documents, 64 query terms, and 3.2
million postings. Results are medians of five runs on the local Mac.

| Distribution | TAAT | BMP | Posting fraction | Exact |
| --- | ---: | ---: | ---: | ---: |
| clustered high-score head | 12.73 ms | 6.75 ms | 1.0% | yes |
| uniform adverse case | 12.76 ms | 361.43 ms | 100.0% | yes |

The clustered case gives a 1.89x speedup while preserving byte-identical top-k
scores. The adverse case proves that BMP must not be mandatory. The in-memory
benchmark intentionally runs BMP to completion so it exposes the pathological
cost; the PostgreSQL executor has the 25% route-abandon gate and falls back to
TAAT instead.

This distinction matters. A cache would make both rows warmer but would not
change the 1% versus 100% posting work. The structural gate changes work.

The first benchmark deliberately made every indexed term a query term. A
second benchmark separates corpus and query sparsity: 100,000 documents,
1,024 indexed terms, about 128 active terms per document, and only 64 query
terms. It contains 12.8 million corpus postings and 800,000 query postings.

| Distribution | TAAT | Full BMP | Query postings scored | Forward records read | Record/query ratio | Guard |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| clustered high-score head | 2.29 ms | 1.61 ms | 1.008% | 64,512 | 0.0806 | keep BMP |
| uniform adverse case | 2.31 ms | 118.92 ms | 100.0% | 6,400,000 | 8.0 | fall back |

The full in-memory benchmark intentionally does not abort. This exposes the
cost curve and verifies exactness. The PostgreSQL scorer uses the new record
budget and stops the adverse route before reading all 6.4 million records.
The clustered case remains useful even after accounting for the non-query
atoms: it is exact and 1.43x faster on the local scalar path.

## Correctness and Lifecycle Evidence

The staged PostgreSQL lifecycle suite passes all 26 gates. They cover:

- compact scorer versus streaming fallback versus full snapshot oracle;
- signed exact scoring;
- insert lexical visibility before semantic completion;
- update and delete MVCC behavior;
- semantic completion and stale-version rejection;
- crash recovery;
- ordinary maintenance reuse of completed semantic postings;
- VACUUM retirement convergence;
- cold restart;
- REINDEX;
- fused and field-aware multicolumn modes.

The suite forces the compact scorer after immediate WAL recovery and compares
it with both compressed fallback and the complete snapshot oracle. For cold
restart, it captures the physical exact route before shutdown and requires an
equivalent compact, ordered-block, TAAT, or materialized authority afterward.
It also mutates a version 4 fold to version 3 and requires a rebuild-required
error, proving the runtime does not silently retain the experimental authority.

The C unit suite also includes randomized signed exactness, serialization
round trips, checksum corruption, invalid reference, and invalid superblock
metadata tests.

The current C unit binary passes with AddressSanitizer and
UndefinedBehaviorSanitizer enabled. LeakSanitizer is unavailable on this macOS
runtime, so leak detection is not claimed from that run. Direct C unit,
isolated PostgreSQL regression, and all lifecycle gates pass separately.

## Native FIQA Closure

The structural prototype was also rebuilt through the staged PostgreSQL 18
extension against the real FIQA corpus, not a generated unit fixture:

- 57,638 documents;
- 854.04 seconds for the full SAE build;
- about 44-46 MB build-backend RSS during the observed inference phase;
- 387,301,376 index bytes, or 369 MB;
- zero L0 records after publication;
- valid, query-ready, semantically converged status after restart.

The size is intentionally disqualifying evidence for the dual format: this
medium corpus already demonstrates that appending a full forward BMP copy to
the legacy exact stream is not a product storage answer.

Six public `evoke_query(...)` queries were compared with BMP routing enabled
and with the legacy exact scorer forced. The guarded and legacy top-100 rows
were identical for every query. All six queries contained only 0.3-0.6 million
semantic postings, so the production route now rejects BMP before allocating
or reading its bounds. Guarded/legacy p50 ratios ranged from 0.967 to 1.033,
which is measurement noise around the same route.

For diagnostic purposes the route was then forced for the first query. The
forward prototype read 113,133 records and 16,486 exact semantic postings,
then correctly abandoned the route. Before the earlier route fix, the same
query read 473,143 forward records and made public p50 1.69x slower than the
legacy scorer. A smaller seed batch, a total-block rather than remaining-block
25% gate, and the minimum-work floor remove that product regression while
preserving forced-path coverage in integration tests.

The raw evidence is retained outside the repository:

```text
/tmp/evoke-fiqa-semantic-bmp-ab-20260810.json
/tmp/evoke-fiqa-public-search-ab-route-v2-20260810.json
/tmp/evoke-fiqa-public-search-ab-min-work-v3-20260810.json
```

### Query-local coarse-directory reuse

The page-native scorer now decodes each selected term run's b256 references
once per query. Superblock admission, b16 bound expansion, and exact record
resolution share that query-local directory instead of rereading the same
packed b256 metadata in each phase. The cache is bounded by the existing
64 MB query-working-set gate. A query whose complete directory would exceed
the gate retains the prior bounded streaming reader instead of abandoning BMP
or allocating index-sized backend state. The durable format, publication path,
and lifecycle are unchanged.

An isolated v7 NFCorpus index with 2,063 real documents was used for a direct
old/new binary comparison. Four corpus queries produced byte-identical top-100
document and score hashes under forced BMP and exact fallback. Three queries
used BMP; one crossed the existing 25% work guard and fell back exactly.

| Query | Old BMP median | Cached BMP median | Change |
| --- | ---: | ---: | ---: |
| PLAIN-1008 | 13.202 ms | 13.023 ms | -1.4% |
| PLAIN-1018 | 10.806 ms | 10.109 ms | -6.5% |
| PLAIN-102 | 16.007 ms | 14.480 ms | -9.5% |
| PLAIN-1028, guarded fallback | 21.817 ms | 19.338 ms | -11.4% |

This removes one verified source of repeated metadata work, but it does not
make latency corpus-size invariant. For the earlier FIQA trace, the exact
scorer still examined 14,631 semantic postings and 4,641 exact records,
compared with 380 postings and 281 records for NFCorpus. Remaining ratios must
therefore be attributed primarily to useful posting and record work, not to
the eliminated b256 rereads. A batch-local fine-reference cache was also
tested and rejected: page validation already makes those reads cheap, while
the extra lookup and memory traffic made the FIQA scorer slower.

The complete FIQA product index was also checked with the same ten-query,
three-repeat workload after one warm-up query and a PostgreSQL restart for each
binary. The prior streaming binary measured p50 219.699 ms and p95 478.505 ms;
the bounded cache measured p50 221.444 ms and p95 459.700 ms. The p50 result is
noise-level and does not justify a broad latency claim, while the 3.9% p95
reduction is directionally consistent with the NFCorpus result. All ten
queries separately matched the complete snapshot oracle at top 100. The cache
is therefore retained as a bounded exact metadata optimization, not presented
as the solution to high-DF posting work.

## Single-Authority Packed Representation

Segment v7 and neutral fold v4 replace the experimental dual format. Semantic
values exist once in a packed term-major impact/delta stream; b256 and b16
metadata are skip directories over that stream. All durable publication and
read paths now share this authority.

The representation study rejected an initially attractive block-major-only
layout. It stored exact impacts by document block and let term references point
into those blocks. Selective BMP was fast, but adverse exact fallback had to
chase impact locations through millions of small records. The sparse fallback
took about 118 ms versus 12-13 ms for the direct term-major baseline. A cache
could not repair that access pattern, so the layout was rejected.

The surviving representation is a block-skipping posting stream, not a second
index. Its single score authority consists of:

- one term-major exact float impact stream;
- one per-term compressed document-delta stream using a fixed 1, 2, or 4-byte
  width selected from that term's maximum delta;
- one sorted term directory containing the first document, posting count and
  delta width;
- fixed 16-byte b256 references containing the first fine reference and exact
  impact offset;
- fixed 5-byte b16 references containing a local block id, 16-document mask,
  and outward-quantized minimum and maximum impacts.

The exact impacts exist once. The compressed document deltas are the
sequential traversal authority; the masks and bounds are skip metadata over
that same stream. Selective queries use b256/b16 bounds and score only
competitive records. Non-selective queries abandon pruning and decode the
term-major deltas sequentially. This is the same structural principle as a
document-ordered posting list plus block-max skip data, adapted to learned
sparse b16 forward scoring.

The format now has an actual checksummed little-endian serializer and strict
deserializer, not only a byte-count estimate. Unit coverage includes signed
top-k parity after serialization, randomized exactness, and one-byte corruption
rejection.

### Storage and Query Results

The two synthetic shapes expose different metadata regimes:

| Corpus shape | v2 forward BMP | Rejected expanded hybrid | Packed single-impact authority |
| --- | ---: | ---: | ---: |
| 3.2M postings, 64 indexed/query terms | 25.485 B/posting | 17.251 B/posting | **11.251 B/posting** |
| 12.8M postings, 1,024 indexed/64 query terms | 14.509 B/posting | 12.503 B/posting | **8.004 B/posting** |

The serialized packed files are 36,003,656 and 102,450,312 bytes respectively.
The second result is approximately the old uncompressed 8-byte
`(docid, impact)` baseline while also carrying the two-level pruning directory.
The first shape is intentionally adverse for metadata because every posting
creates a separate term/block reference. It still uses 56% of the transitional
duplicated format.

| Corpus shape | Direct TAAT | Expanded BMP | Packed adaptive | Packed fallback | Route |
| --- | ---: | ---: | ---: | ---: | --- |
| 3.2M sparse/clustered | 13.070 ms | 6.604 ms | **11.342 ms** | 18.988 ms | BMP |
| 3.2M sparse/uniform | 13.298 ms | 340.024 ms | **34.841 ms** | 19.358 ms | fallback |
| 12.8M dense/clustered | 2.243 ms | 1.514 ms | **1.320 ms** | 3.437 ms | BMP |
| 12.8M dense/uniform | 2.423 ms | 115.029 ms | **13.223 ms** | 3.560 ms | fallback |

All rows return identical top-k document IDs and scores. In the dense clustered
shape, the packed path reads 4,032 query-term references and scores 8,064 of
800,000 query postings. In both uniform shapes, the executor observes 64
consecutive b256 ranges that cannot establish a pruning boundary, abandons BMP,
and runs exact compressed TAAT over the same authority. This bounds the
pathological route at 13-35 ms instead of 115-436 ms.

The packed fallback remains about 1.45x slower than the benchmark's raw
32-bit-doc-id TAAT because it decodes compressed deltas. That is a measured
space/time trade rather than hidden duplicated state. The adaptive total also
includes the failed bound preflight, so it is higher than the fallback alone.
Before product promotion, a real query trace must determine whether the 64
superblock abandon point minimizes p95; it must not be tuned to one dataset.

The compact codec now has a direct runs-to-packed writer. It uses a bounded
term-local sizing pass to choose each term's fixed delta width, allocates the
final arrays once, and writes the packed authority without first constructing
the expanded BMP. A strict unit gate requires its serialized bytes to be
identical to the original expanded-to-packed oracle, including checksums and
outward-quantized bounds. A caller-owned `serialize_into` path also lets the
publication layer write into its final immutable buffer without another
serialized copy.

The page publication caller invokes `evoke_segment_payload_serialize()`
directly; it does not call `evoke_segment_payload_serialized_size()` first.
Consequently the standalone size API cannot create a second publication
prepass. Delta-width discovery remains bounded to the current term's runs
inside the one publication operation rather than rebuilding or re-inferring
the corpus.

The direct writer removes the original 1.6-1.8x representation-oracle build
penalty. On the current local benchmark it builds 3.2 million postings in
238.715 ms, versus 663.822 ms for the expanded authority builder. At 12.8
million postings it takes 743.055 ms, versus 1,485.388 ms. These figures
measure the codec layer, not PostgreSQL publication or model inference, but
they establish that the compact format does not require a second corpus-sized
build.

The result agrees with
[Efficiency Optimizations for Superblock-based Sparse Retrieval](https://arxiv.org/abs/2602.02883):
superblock computation and uncompressed metadata can dominate, so compactness
must be designed around the actual traversal. It also follows the BMP/SP cache
lesson by scoring one document block at a time rather than returning to a
corpus-sized random-write score array.

### Product Boundary

The sole-authority cutover is implemented. New segment and fold publication,
query-time BMP pruning, exact fallback, merge, compaction, and fold all consume
the same packed stream. Generic semantic posting runs are no longer written,
and the runtime rejects version 6 payloads and version 3 folds instead of
translating them during queries.

Local C unit tests, isolated `pg_regress`, and the 26-gate SAE lifecycle suite
pass. The lifecycle coverage includes CRUD, asynchronous semantic completion,
VACUUM, crash recovery, cold restart, REINDEX, multicolumn modes, and compact
versus streaming exact parity.

An isolated PostgreSQL sidecar on Shadow rebuilt 100,000 real PubMed rows into
a title-only v7 SAE index:

- build time: 937.737 seconds;
- index size: 170,909,696 bytes;
- public warm-query wall time, including a new `psql` process: 0.42-0.56
  seconds across three probes;
- insert was lexically visible before semantic inference and converged in
  about 661 ms;
- update, delete, maintenance, exact A/B probes, and cold restart passed.

The sidecar exposed one important integration defect: outward-quantized block
bounds were reused as exact materialized-block bounds, violating the generic
layout validator after an update created multiple semantic runs. The reader
now recomputes exact materialized minima and maxima from decoded impacts while
retaining conservative quantized metadata for pruning. The repaired candidate
passed the same update/delete/restart sequence.

The compact route adaptively fell back for the tested 100,000-row title-only
queries because more than 25% of their small, high-density b16 surface remained
competitive. The fallback decoded the same packed authority and remained
exact; it did not read a legacy stream. The full 8.2-million-row PubMed oracle
predicts only 2.334% of query postings and 1.193% of document slots are
competitive for `machine learning`, but only an out-of-place full rebuild can
confirm that selectivity, index size, publication memory, and tail latency in
the deployed format.

The product boundary is therefore narrower now: representation authority,
direct publication, correctness, and medium-smoke lifecycle are complete. A
large PubMed rebuild and workload remain mandatory before declaring the
deployment-cost problem solved.

#### Sole-authority FIQA rebuild

The installed PostgreSQL 18 extension subsequently rebuilt the complete FIQA
corpus directly into the product format, without first publishing or retaining
a legacy posting authority:

- 57,638 source documents;
- metapage storage version 3, segment payload version 7, and fold version 4;
- 500.67 seconds to build with one validated remote CUDA/fp16 accelerator and
  two local CoreML workers;
- 221,872,128 bytes immediately after publication, compared with 387,301,376
  bytes for the earlier dual-format experiment, a 42.7% reduction;
- 27,084 reachable blocks, no delta records, and no trailing unpublished
  blocks immediately after publication;
- valid, query-ready, runtime-signature-matched, and semantically converged
  status before and after a PostgreSQL restart.

The build-time improvement over the earlier 854.04-second run is not attributed
to the format alone because the runtime topology differed. The size result is
directly comparable and closes the duplicate-authority storage objection for
this corpus.

The full index also passed an explicit mutable-lifecycle probe. An inserted
document was visible through lexical atoms before semantic completion. A
same-transaction update to another lexically representable document was
visible at rank 15 before commit. A semantic-only update became visible after
the asynchronous worker completed, the old version disappeared, and deletion
was immediately hidden by the MVCC overlay. VACUUM reclaimed retired pages,
semantic pending work returned to zero, and the source table returned to
57,638 rows. Short churn windows can still raise the stable document-slot high
watermark until compaction removes all posting and event residency; the fixed
live-set lifecycle gate separately requires and observes a plateau after
repeated convergence cycles.

One exact page-native diagnostic used 53 query atoms. The compact executor
examined 6,892 semantic postings before the 25% work guard selected exact
packed TAAT. The fallback decoded 514,978 postings and scored 56,769 document
slots. Its top-k matched the complete snapshot oracle. Both routes consumed the
same v7 packed stream.

Ten representative FIQA-style queries were each run three times through the
public `evoke_query(...)` entrypoint after restart. The warm end-to-end
distribution was:

| Samples | p50 | p95 | Minimum | Maximum |
| ---: | ---: | ---: | ---: | ---: |
| 30 | 220.744 ms | 484.820 ms | 48.191 ms | 492.260 ms |

The first post-restart query took 968.243 ms while the local query compiler was
loaded. Per-query warm medians ranged from 48.457 ms to 491.031 ms. The range
tracks useful work rather than model inference alone: fast queries scored
746-6,337 documents and decoded 66,932-207,239 postings, while the slowest
queries scored 52,698-56,854 documents and decoded 379,087-691,470 postings.
For the 53-atom diagnostic, warm query encoding took 34-38 ms and the native
scorer took 327-330 ms. The remaining tail-latency problem is therefore
high-document-frequency exact accumulation, not a compatibility reader or a
second posting authority.

#### Exact packed-stream follow-up

The exact fallback originally reconstructed every 16-document semantic block
through separate BMP metadata and record range loads. That retained exactness,
but turned a high-DF fallback into millions of small page-range operations.
The v7 reader now keeps a query-pinned cursor over the canonical packed term
stream and decodes document deltas and contiguous impacts in bounded windows.
It does not materialize a second posting authority and does not change the
segment, fold, scoring, or lifecycle contract.

On the same local FIQA v3 index, ten real FIQA queries with two warm samples
each produced the following end-to-end distribution after this change:

| Samples | p50 | p95 | Minimum | Maximum |
| ---: | ---: | ---: | ---: | ---: |
| 20 | 125.984 ms | 151.139 ms | 76.113 ms | 158.926 ms |

All repeated top-100 result hashes were stable. A preceding run of the same
stream reader measured 116.845 ms p50 and 149.176 ms p95, so the useful claim
is the approximately 120-150 ms warm band rather than a difference between
those two noisy local samples. The prior v7 reader reported 220.744 ms p50 and
484.820 ms p95. Build, C unit tests, the 83-gate page-native parity suite, and
the current 26-gate SAE CRUD/restart/crash lifecycle suite all pass.

The BMP admission gate also no longer requires a complete query-local
document-length vector. If a resident shared vector exists, it is reused. If a
query-local vector still fits inside the fixed 64 MiB exact-query working-set
limit, the existing contiguous path remains selected. Only when that vector
would push BMP over the limit does the lexical prepass read document lengths
from query-pinned document blocks in posting order. This preserves exact BM25
normalization while allowing semantic BMP pruning to remain eligible under
large-corpus or shared-arena pressure instead of immediately falling back to
full TAAT accumulation.

The coarse two-level pass also no longer reserves the flat-fallback upper-bound
and ranking arrays before it knows that fallback is necessary. If fallback is
entered, the upper-bound array is allocated under the same 64 MiB gate, the
nonzero remaining blocks are counted, and the ranking array is sized to that
count rather than every corpus block. Queries completed by normal superblock
pruning therefore avoid approximately `0.75 * document_count` bytes of unused
scratch space. If the actual fallback surface still exceeds the cap, the route
fails closed to the exact streaming scorer.

This is not yet a PubMed or arXiv scale closure. The extant Elm PubMed index is
an old transitional metapage-v2/neutral-fold generation and is correctly
rejected by the current sole-authority v3 binary. It was restored unchanged
after a compatibility probe. Representative latency, RSS, posting counts, and
BMP fallback rates require an out-of-place v3 rebuild; translating the old
index or weakening the runtime contract would reintroduce the compatibility
debt removed by the product cutover.

#### Bounded query-adaptive semantic-term pruning

The shipped P2.1 checkpoint does not currently provide a useful negative-
posting optimization. Its query weights and published semantic document
impacts are nonnegative, and zero-impact atoms are not stored. The compact
codec and exact executor retain signed support for future checkpoints and
correctness tests, but removing nonpositive values from the current product
index would remove no work.

A static DF or contribution threshold is also the wrong product contract. It
does not quantify the accumulated score error, and the same threshold can
remove a very different fraction of a short and a long query. The replacement
diagnostic uses an explicit per-query error budget. For each semantic-only
term `i` and immutable run `r`, it computes the conservative contribution
bound:

```text
b_i = sum_r max(abs(q_i * min_impact_i,r),
                abs(q_i * max_impact_i,r))
T(q) = sum_i b_i
E(q) = sum_{i omitted} b_i <= epsilon * T(q)
```

The triangle inequality then gives the executable contract for every
document `d`:

```text
abs(score_exact(q, d) - score_approx(q, d)) <= E(q)
```

Eligible terms are ordered by logical postings saved per unit of absolute
error bound. Lexical terms are never omitted. Missing or incompatible BMP
metadata fails closed, and any mutable L0 projection disables approximation.
The hidden `evoke.test_query_semantic_error_budget_ratio` A/B control defaults
to zero, so the public product path remains exact.

Three fixed budgets were evaluated, without a threshold search. Full qrels
results on all 648 FIQA queries were:

| Error budget | NDCG@10 | MAP@100 | Recall@100 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| exact | 0.334180 | 0.279181 | 0.618778 | 0.417735 |
| 0.5% | 0.334696 | 0.279619 | 0.617888 | 0.418207 |
| 2% | 0.336143 | 0.280106 | 0.625094 | 0.421730 |
| 5% | 0.337618 | 0.280321 | 0.624966 | 0.424250 |

The corresponding 323-query NFCorpus results were:

| Error budget | NDCG@10 | MAP@100 | Recall@100 | MRR@20 |
| --- | ---: | ---: | ---: | ---: |
| exact | 0.346492 | 0.175056 | 0.292635 | 0.559095 |
| 0.5% | 0.346777 | 0.175088 | 0.292867 | 0.559441 |
| 2% | 0.345918 | 0.175194 | 0.294300 | 0.557722 |
| 5% | 0.346419 | 0.175690 | 0.296691 | 0.558988 |

Macro quality is not enough to promote the route. At 0.5%, FIQA had 17 NDCG
gains and 14 losses; three queries lost Recall and none gained it. NFCorpus had
six NDCG gains and eight losses, with four Recall gains and six losses. The
larger budgets improve macro Recall on both surfaces but increase query-level
movement and retain material worst-case harm.

An alternating 64-query sample measured end-to-end latency and exact-result
overlap under the same process and cache state:

| Surface | Budget | Median latency ratio | Mean O@10 | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FIQA | 0.5% | 0.990 | 0.994 | 0.991 | 0.970 |
| FIQA | 2% | 0.978 | 0.966 | 0.972 | 0.930 |
| FIQA | 5% | 0.968 | 0.931 | 0.947 | 0.890 |
| NFCorpus | 0.5% | 1.055 | 0.994 | 0.995 | 0.980 |
| NFCorpus | 2% | 0.960 | 0.978 | 0.980 | 0.920 |
| NFCorpus | 5% | 0.967 | 0.948 | 0.959 | 0.850 |

The low-single-digit speed signal is real on FIQA but not yet robust on the
small NFCorpus surface, where planning overhead can exceed saved traversal.
On a 20-query FIQA telemetry sample, 0.5% omitted 3.65 atoms and 34,545 logical
postings on average, but reduced actually decoded BMP postings by only about
11%. Existing BMP block pruning had already avoided most logical postings.

The strongest observed harm pattern was support loss, not absolute score
budget. At 0.5%, the FIQA NDCG-harm sample omitted 7.3% of query atoms versus
4.3% for the benefit sample. NFCorpus separated more strongly at 14.6% versus
6.2%. Harm queries were also shorter on both surfaces. The omitted-bound ratio
was effectively identical between groups, so `epsilon` alone cannot express a
safe query policy.

The next product-shaped policy must therefore solve a constrained work
problem rather than spend the complete error budget on every query:

1. Keep the absolute score-error contract above.
2. Add a semantic-support retention floor so short queries cannot lose a
   disproportionate number of atoms.
3. Add an estimated posting-work target and remain exact when a query is
   already below it.
4. For queries that cross the work target, oversample approximate candidates,
   rerank them with all query atoms, and use `E(q)` plus the candidate boundary
   margin to certify exact top-k membership when possible.

This is a bounded approximation foundation, not a promoted product default.
It still requires large-index PubMed/ArXiv cost validation and broader qrels
validation. Exact remains the default; a query that cannot satisfy both the
quality constraints and the work target must report that fact rather than
silently opening the error budget further.

## High-Concurrency Large-Index Direction

### Diagnosis

The remaining product problem is not the absence of a pruning primitive. The
current executor already has exact b256/b16 bounds, an adaptive TAAT fallback,
a corpus-independent semantic score-error bound, posting-heat telemetry, and
root-scoped hot folds. The missing capability is a query-global control plane
that combines those primitives into one budgeted execution decision.

| Existing capability | Remaining gap |
| --- | --- |
| Exact BMP block bounds | No strong exact initial top-k threshold for difficult queries |
| Term-level absolute error budget | No support floor, candidate completion, or top-k certificate |
| Per-query 64 MiB memory guard | No cross-backend CPU/posting/memory admission |
| BMP uses a `float scores[document_count]` scratch array | Memory is capped per backend but remains O(corpus x concurrency) |
| Posting heat and lexical-impact hot folds | Single-term terminal path only; no semantic impact-prefix seed |
| Path and posting telemetry | No calibrated pre-execution work estimate or query-class routing |

The 0.5% pruning experiment reinforces this diagnosis. It removed 34,545
logical FIQA postings per query but only around 11% of the postings that BMP
would actually decode. Spending a term-level error budget after block pruning
has already removed most of the same work cannot produce a large speedup. It
also harms short queries by removing too much support. Further tuning of one
global epsilon is therefore not the next step.

This behavior is expected for learned sparse indexes. The weight distribution
reduces normal DaaT skipping opportunities and makes score-at-a-time access
more competitive, especially for predictable tail latency
([Wacky Weights](https://arxiv.org/abs/2110.11540)). BMP instead evaluates
small document ranges in decreasing upper-bound order and can stop safely when
the heap threshold exceeds the next range bound
([BMP](https://arxiv.org/abs/2405.01117)). Superblock Pruning and its newer
lightweight variant show that a coarse first level can reduce child-bound
work, but they also show that fixed approximate parameters can be fragile
under index/model drift
([SP](https://arxiv.org/abs/2504.17045),
[LSP](https://arxiv.org/abs/2602.02883)).

Two adjacent results identify the missing bridge. Top-k threshold estimation
has been demonstrated directly on learned sparse indexes
([Beyond Quantile Methods](https://arxiv.org/abs/2412.10701)). SPRAWL obtains
a strong candidate set by reading a bounded impact-ordered prefix and then
completing candidate scores with exact lookups; pair prefixes improve the
moderate-budget regime, while full score completion is important for quality
([SPRAWL](https://www.pinecone.io/research/SIGIR25c.pdf)). Evoke can test the
same principle without introducing a second authority. The current hot-fold
payload is lexical-impact-only, so it cannot directly supply P2 semantic
candidates. Its root-versioned shared-cache lifecycle and posting-heat policy
can be reused for a semantic prefix derived from the compact v7 authority.

### Rejected local controls

The first experiments tested whether the current physical order could be kept
and the work controlled only by thresholds.

- A fixed DF policy removed 31.5% of logical postings, but reduced documents
  touched by only 2.7%. Serial latency did not improve, and concurrency
  throughput did not scale.
- Keeping only the strongest 16 semantic query terms reduced logical posting
  work by 40.5%, but still examined 97.5% of the documents and changed p50 from
  94.1 ms to 107.4 ms. Direct top-100 overlap was only 0.8573.
- Oversampling 400 candidates after the 16-term cut recovered 0.9990 of the
  exact top-100 candidates. This proves that exact candidate completion is
  viable, but it did not make candidate generation cheap because the retained
  posting lists still covered almost the complete corpus.
- The measured FIQA concurrency ceiling was about 42.5 QPS at both 8 and 32
  workers. Increasing concurrency from 8 to 32 raised p95 from about 203 ms to
  719 ms without increasing throughput. Admission can bound this queueing, but
  only less work per query can raise the ceiling.

A corpus-global sparse IVF was then tested to determine whether ordinary
vector clustering was enough. At about 400 candidates it covered only 0.2918
of the exact top 100; at about 4,000 candidates it covered 0.7897. Making the
sparse summaries exact restored top-100 parity, but required opening a median
46,703 of 57,638 FIQA documents and took about 200 ms. The failure is
structural: one global partition does not preserve the query-term-local
geometry needed by learned sparse vectors.

### Per-term geometric candidate oracle

The next oracle used the organization introduced by
[Seismic](https://arxiv.org/abs/2404.18812): prune each inverted list to a
bounded candidate surface, partition that list into geometrically cohesive
document clusters, attach sparse maximum summaries, rank clusters for the full
query, and complete candidate scores through a forward lookup. Ordering the
clusters by their summary score follows the improvement later evaluated by
[SeismicWave](https://arxiv.org/abs/2408.04443).

The fixed Space-C build used `n_postings=1000`, `centroid_fraction=0.02`,
`summary_energy=0.10`, `doc_cut=15`, and a 400-document candidate heap. These
are shared settings, not dataset-specific tuning. Inputs are saved M1914 CSR
surfaces; they represent the product checkpoint but are not byte-for-byte
timings of the installed PostgreSQL v7 reader.

The broader run exposed one scale error in that nominally shared setting.
`n_postings` is a per-list absolute cap, so it retains radically different
fractions of each corpus posting surface:

| Surface | Docs | Retained at 1,000 | Retained at 2,000 | Retained at 3,500 |
| --- | ---: | ---: | ---: | ---: |
| FIQA | 57,638 | 45.45% | 60.31% | 72.34% |
| NFCorpus | 3,633 | 97.15% | 99.63% | 100.00% |
| SciFact | 5,183 | 97.38% | 99.63% | 100.00% |
| Arguana | 8,674 | 83.49% | 93.84% | 98.05% |
| SciDocs | 25,657 | 70.77% | 84.82% | 93.24% |
| TREC-COVID | 171,331 | 30.28% | 44.14% | 56.78% |
| Quora | 522,931 | 23.18% | 33.10% | 42.76% |

This explains why increasing query cut improves Quora/TREC overlap only
partially: documents removed at build time cannot be recovered at query time.
The product builder must select pruning from a corpus-level retained-posting
mass and derived-byte budget, not a fixed list length. This is still a global,
qrels-free rule; it does not tune a dataset's ranking metrics. The 45% retained-
mass region is a validated representation-oracle point, not yet a promoted
product constant.

The quality preset uses `query_cut=16` and `heap_factor=0.7`:

| Surface | Docs | Exact p50 | Geometric p50 | Speed | O@100 | Delta NDCG@10 | Delta MAP@100 | Delta Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIQA | 57,638 | 23.111 ms | 3.304 ms | 7.0x | 0.996605 | +0.000080 | +0.000082 | -0.000600 |
| NFCorpus | 3,633 | 1.217 ms | 0.573 ms | 2.1x | 0.997492 | +0.000021 | +0.000040 | +0.000086 |
| SciFact | 5,183 | 1.640 ms | 0.816 ms | 2.0x | 0.999433 | +0.000000 | -0.000003 | +0.000000 |
| Arguana | 8,674 | 3.280 ms | 1.093 ms | 3.0x | 0.999500 | -0.000022 | -0.000024 | +0.000000 |
| SciDocs | 25,657 | 6.074 ms | 1.274 ms | 4.8x | 0.999510 | +0.000038 | +0.000053 | +0.000000 |
| TREC-COVID | 171,331 | 41.991 ms | 1.998 ms | 21.0x | 0.971200 | -0.002075 | +0.000195 | +0.000498 |
| Quora | 522,931 | 63.930 ms | 0.823 ms | 77.7x | 0.962371 | -0.000226 | -0.000254 | -0.000981 |

The seven-surface macro comparison separates the speed/quality policies:

| Preset | Query cut | Heap factor | Geometric mean speed | Mean O@100 | Delta NDCG@10 | Delta MAP@100 | Delta Recall@100 | Delta MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fast | 10 | 1.0 | 11.35x | 0.971663 | -0.000455 | -0.000461 | -0.001548 | -0.000453 |
| balanced-fast | 10 | 0.7 | 8.89x | 0.983671 | -0.000375 | -0.000194 | -0.000850 | -0.000063 |
| balanced | 12 | 0.7 | 8.04x | 0.985908 | -0.000361 | -0.000118 | -0.000489 | -0.000055 |
| quality | 16 | 0.7 | 6.84x | 0.989445 | -0.000312 | +0.000013 | -0.000143 | -0.000027 |

TREC-COVID and Quora are the important stop signals. TREC Recall and MAP are
preserved by the quality preset, but NDCG@10 and exact-result overlap remain
visibly worse. Quora quality metrics move by less than 0.001, but its exact
membership overlap is only 0.96237. No preset may therefore be described as
exact or silently selected by the executor. `exact` remains the default;
`quality`, `balanced`, and `fast` are caller-visible bounded policies with
independently reported telemetry.

### Scale-aware candidate completion

The fixed-cap failure has a single cause: build-time pruning removed documents
that a deeper query cut could never recover. Increasing `query_cut` to 24 or
32 improved overlap only partially on TREC-COVID and Quora. Increasing the
per-list cap recovered overlap, but an absolute cap retained 97% of NFCorpus
postings and only 23% of Quora postings. A shared absolute cap is therefore
not a coherent cross-corpus policy.

The replacement builder selects the smallest per-list cap that retains 45%
of the complete corpus posting mass, with a minimum cap of 1,000 so inexpensive
surfaces are not needlessly pruned. It uses no qrels or dataset identity. The
query candidate uses `query_cut=16`, `heap_factor=0.7`, and exact completion of
up to `8 * k` candidates. The reported latency includes both candidate search
and completion against the original sparse authority.

| Surface | Docs | Cap | Retained mass | Exact p50 | Candidate p50 | Speed | O@100 | Delta NDCG@10 | Delta MAP@100 | Delta Recall@100 | Delta MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FIQA | 57,638 | 1,000 | 45.45% | 16.473 ms | 3.480 ms | 4.73x | 0.997593 | +0.000000 | +0.000005 | -0.000600 | +0.000006 |
| NFCorpus | 3,633 | 1,000 | 97.15% | 0.805 ms | 1.051 ms | 0.77x | 0.999845 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| SciFact | 5,183 | 1,000 | 97.38% | 1.158 ms | 1.390 ms | 0.83x | 0.999967 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| Arguana | 8,674 | 1,000 | 83.49% | 2.298 ms | 1.795 ms | 1.28x | 0.999793 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| SciDocs | 25,657 | 1,000 | 70.77% | 6.014 ms | 2.736 ms | 2.20x | 0.999940 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| TREC-COVID | 171,331 | 2,081 | 45.01% | 41.998 ms | 4.580 ms | 9.17x | 0.995000 | +0.000000 | +0.000347 | +0.000749 | +0.000000 |
| Quora | 522,931 | 3,943 | 45.00% | 64.605 ms | 2.572 ms | 25.12x | 0.996434 | +0.000018 | -0.000004 | -0.000100 | -0.000003 |

Across all seven rows, mean O@100 is 0.998367. The all-row geometric-mean
speedup is 2.95x; the four rows with at least 25,000 documents give 7.00x.
The negative speedup on NFCorpus and SciFact is a required routing signal, not
a failed representation result. The product executor must compare estimated
exact work with bounded candidate work and select exact whenever the query is
already cheap.

SeismicWave's 10-neighbor graph was also evaluated rather than assumed. It
adds about 1.15 MB on FIQA, 3.86 MB on TREC-COVID, and 12.42 MB on Quora. For
the quality candidate it changes TREC-COVID O@100 from 0.9950 to 0.9988 and
Quora from 0.996434 to 0.998951, while adding about 0.47 ms and 0.65 ms p50.
The graph is a real quality extension, but the graph-free candidate already
passes the representation-oracle quality gates. The first product slice will
therefore omit the graph and its publication lifecycle. It remains eligible
only if native validation exposes row-level harm that candidate completion
alone cannot repair.

The standalone Seismic index includes its own forward score store. Space-C
used about 171 MB on FIQA, 96 MB on SciDocs, and 296 MB on TREC-COVID. Evoke
must not copy that score authority: the product accelerator may store only
cluster summaries and candidate document references, while exact candidate
scores come from the existing packed v7 stream. A DotVByte variant reduced the
large FIQA oracle by only 2.5% while lowering O@100 from 0.99938 to 0.99071
under the tested setting; that trade is rejected for the first product slice.

### Native prototype and lifecycle boundary

`evoke_semantic_accelerator` is the derived metadata and query module used by
the PostgreSQL extension. It has checksummed little-endian term, directory,
and forward formats; sorted term ownership; outward-quantized maximum
summaries; bounded document deduplication; and an exact-score callback.
Unit coverage includes round trip, corruption, malformed ranges, conservative
quantization, duplicate query terms, negative-query rejection, callback
failure, non-finite scores, and candidate oversampling. Every selected term
now visits clusters in decreasing summary-score order. The candidate
multiplier keeps a lower intermediate heap threshold when sparse summaries
omit query coordinates, but returns only the requested `k` documents after
exact scoring.

The module is linked by both CMake and PGXS. Manifest v13 binds a derived
directory to the exact source authority. Publication streams one immutable
object per accelerated term and bounded forward chunk rather than building a
second serialized corpus copy. Query execution loads only selected term
objects and required forward chunks. A block-major route groups selected-term
clusters by forward chunk, opens each chunk once, and exact-scores candidates
from the unified additive contribution stream. Strictly increasing query atom
IDs use a linear sparse merge; arbitrary query order retains the exact
binary-search scorer.

The maintenance worker builds the accelerator only from a sealed, converged
root, without rerunning the encoder. Before loading the sealed snapshot it
estimates workspace from physical postings, logical postings, and document
slots, and applies `evoke.maintenance_rebuild_memory_budget`. An insufficient
budget returns `accelerator_memory_budget`, leaves the exact root readable,
does not busy-loop an immediate work hint, and is retried by periodic
reconciliation after capacity changes. A successful root change invalidates
the derived reference atomically; later convergence republishes it.

The first full FIQA C-kernel probe also rejected an important wrong product
shape. Building all 28,893 non-empty terms at the 1,000-document cap processed
4,987,950 document references in 174.76 seconds with 96.2 MB peak RSS. Memory
was bounded, but the row-major artifact estimate was 169.55 MB because it
contained 18,387,815 summary entries. That fails the derived-byte gate before
publication and must not be repaired by hiding the bytes in shared memory.

The same probe on only the 256 highest-DF terms took 8.82 seconds and estimated
7.99 MB for 256,000 document references. Those terms cover 30.70% of FIQA's
corpus posting mass and 88.34% of mean query top-16 posting work. This suggests
that geometry belongs only on terms that dominate work; low-DF residual terms
are cheaper and safer on the exact packed route.

A qrels-free selector now chooses the smallest high-DF set covering a target
fraction of corpus posting mass. At a 45% target it selects only 0.9%-1.5% of
the vocabulary on the seven saved surfaces while covering most observed query
work:

| Surface | Selected terms | Vocabulary | Mean query-work coverage | Median |
| --- | ---: | ---: | ---: | ---: |
| FIQA | 552 | 1.10% | 93.98% | 95.51% |
| NFCorpus | 630 | 1.25% | 80.74% | 88.46% |
| SciFact | 772 | 1.54% | 87.96% | 91.52% |
| Arguana | 520 | 1.03% | 86.12% | 89.44% |
| SciDocs | 752 | 1.50% | 90.93% | 93.34% |
| TREC-COVID | 669 | 1.33% | 97.50% | 97.67% |
| Quora | 451 | 0.90% | 86.09% | 89.14% |

NFCorpus and SciFact remain exact-route controls because their complete query
work is already cheaper than geometric dispatch. For heavy surfaces, the next
quality oracle must combine geometric candidates from selected terms with
exact residual candidates and full-query exact completion. A partial
accelerator is not allowed to silently ignore an unselected term inside the
declared query budget. Terms below the public `query_cut` do not generate
candidates, but still participate in exact completion; this is the same
bounded contract as the existing quality oracle, not an exact-mode claim.

The C builder and query implementation now close that partial-accelerator
oracle on all 648 FIQA queries. Both rows use `query_cut=16`, a 1,000-document
retained-list cap, `centroid_fraction=0.02`, `summary_energy=0.10`,
`heap_factor=0.7`, an 800-document geometric heap, exact posting union for
unselected active terms, and full-query completion. Selection is based only on
corpus DF; qrels are used only for final reporting.

| Corpus-mass target | Terms | Serialized bytes | Mean residual posting work | Mean union candidates | O@100 | Delta NDCG@10 | Delta MAP@100 | Delta Recall@100 | Delta MRR@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30.7% | 256 | 7,978,428 | 11.60% | 14,946.6 | 0.998596 | +0.000000 | -0.000021 | -0.000772 | +0.000000 |
| 45.0% | 552 | 17,602,500 | 5.99% | 8,313.4 | 0.995247 | +0.000000 | -0.000024 | -0.000909 | +0.000000 |

The result separates representation quality from product cost. The 30.7% row
examines, per query, about 24,299 summary entries, 5,941 retained document
references, and 20,085 exact residual postings instead of 212,106 exact
top-16 postings. This is a 4.21x reduction in primitive examinations before
accounting for their different costs. The 45% row gives a slightly better
4.44x work proxy and a smaller completion set, but costs 2.21x more bytes and
has lower exact membership overlap. Neither Python/ctypes latency is a product
number: the oracle computes a full sparse score vector to service the C exact-
score callback. Native timing must use page-backed candidate completion.

The first page-native build should therefore start near the 256-term point.
It loses only about 5% of the 552-term work reduction, has better membership
fidelity, and is the only measured row plausibly inside the 25% derived-byte
gate. The 552-term point remains a candidate only after a page format with
delta-coded summary term IDs demonstrates the byte gate against the actual
packed root. A fixed-width row-major summary is not the final disk layout.

The same C path passes a larger TREC-COVID check without changing the policy.
At 171,331 documents, the 30.7% selector chooses 305 terms and produces a
7,326,172-byte artifact in 8.30 seconds. It contains 305,000 retained document
references and 747,723 summaries. Across all 50 queries, exact residual union
retains 0.9926 mean O@100 and exact-top-100 candidate coverage; 49 queries have
complete candidate coverage. NDCG@10 and MRR@20 are unchanged, MAP@100 changes
by +0.000095, and Recall@100 changes by +0.000292. The primitive work proxy is
about 18,421 summaries, 6,617 retained references, and 36,106 residual exact
postings per query versus 685,128 exact top-16 postings, an 11.21x reduction.
This is not native latency, but it rejects the hypothesis that the FIQA result
was caused only by its smaller corpus.

#### Cohesive block-metadata side oracle

A second metadata level helps only when its children are cohesive under the
same score geometry used by the query. Three controls make this boundary
explicit.

- A 13-level source-order hierarchy on the Shadow ArXiv root had an
  oracle-selected median sparse-metadata ratio of 0.5316, but the hard
  `machine learning` query still required 0.9544. Fixed hierarchy levels also
  regressed high-DF queries. Source order is not semantic locality.
- Exact coordinate-maximum block bounds examined 99.996% of FIQA postings;
  exact centroid-radius bounds examined 100%. The summaries are safe, but the
  union of active sparse coordinates and broad cluster radii makes them
  noncompetitive.
- Equal-budget global SimHash grouping reduced summary hits by 9.69% but
  increased posting work by 5.83%, with unchanged top-100 quality. Hash
  proximity is therefore not a sufficient construction objective.

The positive control uses true term-local centroid grouping, as in
[Seismic](https://arxiv.org/abs/2404.18812): rank cohesive block summaries,
retain a fixed candidate set, and then exact-score those documents from the
complete vector. It is an approximate candidate route with exact completion,
not a safe block upper bound. A current-source side oracle produced the
following results:

| Surface | Candidate search p50 | Exact completion p50 | Total p50 | Exact p50 | Mean O@100 | Official metrics |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FIQA | 2.75 ms | 20.36 ms | 23.01 ms | 23.92 ms | 0.999844 | NDCG/MAP/Recall/MRR equal |
| TREC-COVID | 4.92 ms | 39.78 ms | 44.62 ms | 55.39 ms | 0.997800 | NDCG/MRR equal; MAP/Recall do not regress |

FIQA p95 is worse than exact, so the product router must still select exact on
cheap work. TREC-COVID gives the first same-policy positive crossover. The
standalone sidecar sizes are diagnostic and are not acceptable as another
stored score authority.

This experiment also exposed an implementation mismatch. The reusable C term
builder already implemented centroid geometry, but the page-native maintenance
producer grouped documents by source document block. The producer now streams
the existing forward transpose into the scalable term-local centroid builder
and serializes its result. Publication remains one root-derived accelerator
directory under the existing worker, directory CAS, root invalidation, and
retirement paths. No second lifecycle or encoder pass was added. The complete
mutable SAE lifecycle passes all 34 gates after this change.

#### Native FIQA geometric-accelerator closure

The complete 57,638-document FIQA root was then used for a page-native A/B
without rebuilding the corpus. The old query binary and the current
block-major binary read the same published accelerator, so this comparison
isolates query execution rather than encoder or builder changes. Each row is
the distribution across the 32 evaluation queries with three repetitions.

| Route | Old p50 | Current p50 | Current p95 | Mean O@100 | Minimum O@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact packed | - | 61.488 ms | 65.860 ms | 1.000000 | 1.000000 |
| accelerator exact completion (`h0`) | 5,219.894 ms | 634.352 ms | 660.714 ms | 1.000000 | 1.000000 |
| bounded (`h0.7`) | 415.646 ms | 341.335 ms | 363.269 ms | 0.999375 | 0.990000 |
| bounded (`h1.0`) | 411.860 ms | 340.856 ms | 351.000 ms | 0.999375 | 0.990000 |

The implementation change is substantial: exact accelerator completion is
8.23x faster at p50 and 11.23x faster at p95 than the old term-major callback
path. It opens each forward block once, linearly merges sorted query atoms,
and avoids exact-scoring a document again through the residual surface. The
bounded route improves by about 18% at p50 and 17% at p95.

This is nevertheless a negative product-routing result for FIQA. Its native
exact route examines only about 212,000 query postings and completes in about
62 ms. Accelerator `h0` remains 10.3x slower, while the bounded route remains
5.55x slower. The residual union still covers about 70,879 postings and 31,122
documents across every one of the 15 forward chunks. Geometry cannot amortize
that fixed candidate-completion work on this control surface.

The resulting rule is structural rather than dataset-specific: estimate the
exact packed work first and do not admit the accelerator when the complete
query is already cheaper than its cluster, forward-chunk, and residual work.
FIQA closes as the cheap-query exact-route control. No further heap-factor or
FIQA-specific tuning is justified. PubMed/ArXiv qualification must establish
the opposite side of the crossover on high-DF queries before this accelerator
can become a product route.

The retained raw A/B evidence is outside the repository:

```text
/Volumes/Betty/Tmp/evoke-fiqa-accelerator-v8.json
/Volumes/Betty/Tmp/evoke-fiqa-accelerator-v15-reuse.json
/tmp/evoke-product-v15-rss-smoke.json
```

The product integration is a root-versioned, disposable accelerator for a
converged neutral fold:

1. Build from already completed semantic postings; never rerun model inference.
2. Admit publication before loading the sealed snapshot and stream page
   objects with bounded serialized buffers. Large builders may use a generous
   explicit maintenance budget, but may not create unbounded backend state.
3. Keep L0 and unconverged segments on the exact packed route.
4. Discard the accelerator on root mismatch, corruption, crash, or failed
   publication; exact v7 remains immediately queryable.
5. Let compaction/fold replace the derived accelerator atomically with its root
   rather than maintaining a second mutation lifecycle.

Physical publication uses one bounded immutable page object per selected term,
bounded document-major forward chunks, and one small directory object. A query
reads only the objects for accelerated query terms and the forward chunks that
contain exact-scored candidates. This reuses the checksummed object-page
reader and avoids serializing a corpus-sized second in-memory copy. The
directory binds every object to the exact authority from which it was derived.
The native executor aggregates selected clusters by forward block, loads each
forward chunk once, and linearly merges sorted query atoms with each candidate
row. Query validation is performed once rather than once per candidate. A
query-local scored-document bitmap also prevents residual terms from scoring a
document a second time after the accelerator already computed its complete
unified score. These are exact work reductions; they do not change candidate
selection or score order.

TREC-COVID confirms that this decomposition is practical rather than merely
architectural. With the existing v2 codec, the median one-term object is
24,268 bytes, p95 is 29,278 bytes, and the maximum is 33,004 bytes. The 305
separate term objects total 7,352,924 bytes, only 0.37% above the monolithic
7,326,172-byte artifact. The small page-local overhead is therefore preferable
to a second corpus-sized allocation or a new giant-object streaming protocol.
The bounded term-object codec now also rejects an unexpected source authority
or term id during attach, before its clusters can enter candidate generation.

Publication itself creates a descendant manifest, so source identity cannot be
defined as the checksum of that new root. The safe contract is a derived-only
descendant: its directory records the parent authority identity, and the new
manifest is allowed to add only the accelerator reference. Any later CRUD or
ordinary maintenance manifest clears that reference. The exact packed path
remains authoritative and immediately queryable until a worker publishes a
new accelerator after convergence. This makes staleness fail closed without
teaching foreground mutation code how to update clusters.

This follows the convergent-index contract: short-lived mutations may be
slower but exact, while the worker progressively restores the high-performance
fold. It also avoids the failure of corpus-global IVF by constructing geometry
inside each posting list, where the useful locality was observed.

### Exact/certified executor boundary

The following progressive executor remains the route for callers that require
an exact or per-query certified result. The geometric accelerator changes the
candidate order and average work, but it does not remove the need for this
contract when approximation is unacceptable. It has four cooperating stages.

1. **Estimate work before scoring.** Use metadata already available after term
   planning: query terms and runs, logical postings, super references, mutable
   L0 presence, requested `k`, resident metadata state, and required scratch
   bytes. Start with a calibrated monotone linear cost model, not a learned
   gate:

   ```text
   W_hat(q) = c_s * super_refs + c_b * candidate_blocks
            + c_p * decoded_postings + c_d * exact_doc_lookups
   ```

   Coefficients are host-level microbenchmark constants. They describe units
   of CPU work, not relevance, and can be recalibrated without dataset-specific
   ranking tuning.

2. **Seed an exact threshold.** Merge a bounded number of high-impact semantic
   entries from a root-current derived prefix. Complete every selected document
   with exact lookups against all query terms, then insert those exact scores
   into the top-k heap. The resulting threshold is a real lower bound on the
   final kth score, not a prediction. A weak seed costs some lookups but cannot
   change results. A strong seed lets BMP skip uncompetitive blocks immediately.
   The first implementation is a read-only oracle; product publication is
   allowed only after it proves useful. Lexical scores may guide candidate
   order, but must not be an unsafe semantic skipping threshold: unconstrained
   BM25 guidance is known to lose relevance when lexical and learned scores are
   misaligned
   ([Dual Skipping Guidance](https://arxiv.org/abs/2204.11154),
   [2GTI](https://arxiv.org/abs/2305.01203)).

3. **Traverse progressively and keep a residual ledger.** Continue normal BMP
   in decreasing block-bound order. Exact mode stops only at the existing
   rank-safe condition. Bounded mode additionally enforces both the absolute
   score-error budget `E(q)` and a semantic support-retention floor. It keeps
   an oversampled candidate heap and completes those candidates exactly. If
   `tau_k` is the exact kth candidate score, `U_next` is the maximum retained-
   term bound of any unvisited block, and `rho_L` is the approximate score at
   the oversampled candidate boundary, then the returned top-k is certified
   exact whenever:

   ```text
   tau_k > max(U_next + E(q), rho_L + E(q))
   ```

   The first term bounds unseen blocks. The second bounds scored documents
   outside the candidate set. If the certificate fails, the executor either
   continues exactly or follows an explicit caller policy; it must not silently
   widen epsilon. The oracle may initially retain the current corpus-sized
   score array. Product promotion additionally requires block-local b16 score
   accumulation plus a bounded candidate table so selected-block execution no
   longer reserves one float per corpus document in every PostgreSQL backend.

4. **Admit concurrent work by cost, not query count.** Reserve shared semantic
   work units and query-memory bytes before the expensive phase. Cheap queries
   can run concurrently; one predicted 25-million-posting query cannot consume
   the same reservation as a small query. At stage boundaries, a query tops up
   its reservation from observed work. If capacity is unavailable it waits in
   an interruptible, aging queue, uses only a caller-declared bounded mode, or
   respects `statement_timeout`. BM25-only queries use a separate lexical
   budget so semantic overload cannot regress the BM25 baseline. Response-time
   admission with separate query classes and starvation protection follows the
   same systems principle evaluated by
   [Bouncer](https://arxiv.org/abs/2312.15123).

The public quality contract should expose mathematically meaningful controls,
not an estimated overlap promise:

| Mode | Contract |
| --- | --- |
| `exact` | Byte-identical top-k; work may queue or reach `statement_timeout` |
| `certified` | May prune aggressively, but returns only after the exact top-k certificate passes; otherwise continues exact |
| `bounded` | Explicit maximum score-error ratio, minimum support ratio, work target, and unresolved-result policy |

`Recall@k` and overlap remain validation metrics, not per-query guarantees. A
score-error bound alone does not imply a recall bound when the kth score margin
is small.

### Physical-layout boundary

The experiments change the physical-layout priority. Consecutive document-ID
blocks and query-term cuts retain too much corpus coverage. A global IVF loses
the term-local neighborhoods. Per-term geometric clusters preserve those
neighborhoods and are now the first bounded-mode layout to productize.

This does not change the durable authority. The accelerator is equivalent to
a root-scoped hot fold: optional, derived, immutable after publication, and
rebuildable from the exact packed stream. The exact path never depends on it.
The mutable index therefore does not need to update geometric clusters in the
foreground. New lexical data is immediately visible; semantic completion and
normal segment convergence remain exact; a later fold can replace the
accelerator atomically.

An impact-prefix seed remains useful for `certified` mode because it can raise
the exact heap threshold without changing results. It is no longer expected to
solve bounded high-DF candidate generation by itself. Pair-prefix caches are
deprioritized: they add combinatorial publication and invalidation cost before
the simpler per-term geometry has been tested natively.

SeismicWave's kNN expansion is also not part of the first slice. The completed
oracle proves that it can recover neighbors after cluster pruning, but the
graph-free eight-way completion already passes the seven-surface gate. The
graph would add another root-scoped publication artifact before native evidence
requires it. Document-ordered topical segments remain a later SLA option supported
by [Anytime Ranking](https://arxiv.org/abs/2104.08976), not a reason to reorder
the exact v7 authority.

### Phased evidence gates

1. **Closed: local control audit.** Fixed DF pruning and fixed query cuts do not
   reduce touched documents or latency enough. Do not reopen them without a new
   physical ordering.
2. **Closed: global IVF audit.** It cannot provide both small candidate sets
   and high exact-top-100 coverage. Do not tune its cluster count further.
3. **Closed: cross-surface representation oracle.** Per-term geometry with a
   45% posting-mass target, minimum cap 1,000, query cut 16, heap factor 0.7,
   and eight-way exact completion passes the shared quality gate on FIQA,
   NFCorpus, SciFact, Arguana, SciDocs, TREC-COVID, and the 522,931-document
   Quora surface. Cheap-query routing remains mandatory.
4. **Closed: isolated C format/query prototype.** Serialization, validation,
   candidate ordering, exact callback scoring, failure cleanup, C unit,
   compiler-warning, and sanitizer gates pass.
5. **Closed: root-derived product builder.** The deterministic per-term C
   kernel, qrels-free high-DF selector, compact forward transpose, streaming
   object writer, root fingerprint, and pre-snapshot memory admission are
   integrated. Build time, final bytes, and publication workspace are capacity
   planning telemetry, not quality gates. The hard constraints are bounded by
   an explicit maintenance budget, no model reinference, and no second mutable
   score authority.
6. **Closed locally: native lifecycle.** Missing, corrupt, stale, L0-active,
   unsupported, and memory-blocked states fall back exactly. Local semantic
   CRUD, invalidation, worker convergence, republish without inference,
   restart, VACUUM/reclamation, and REINDEX gates pass. Replica and large-root
   evidence remain part of gates 8 and 9 rather than this local closure.
7. **Closed for the FIQA control; scale crossover remains open.** Native FIQA
   retains exact top-k in `h0`, and the bounded preset retains 0.999375 mean
   O@100. The block-major executor materially reduces accelerator work, but
   both routes lose to the approximately 62 ms exact packed query because the
   control surface is too cheap. The product rule is therefore to reject the
   accelerator before scoring on low estimated exact work. Gate 9 must show at
   least 3x lower semantic kernel p50 and at least 2x lower p95 on the
   high-DF heavy-query decile before promotion. Report every row; macro gains
   cannot hide a TREC-like loss.
8. **Open: high-concurrency closure.** Run mixed BM25/SAE load at concurrency
   1/8/32 with cancellation. Require bounded RSS, no starvation, no more than
   5% BM25-only p99 regression, and materially higher SAE throughput than the
   current approximately 42.5-QPS FIQA ceiling.
9. **Open: PubMed/ArXiv scale.** Build out of place, then measure build time,
   index bytes, publication peak RSS versus the configured admission estimate,
   cold/hot p50/p95/p99, candidates, clusters opened, exact lookups, forward
   chunk reads, and fallback rate. Larger storage or a slower one-time build is
   acceptable when query latency improves and lifecycle bounds remain
   explicit. Unbounded publication memory, query-sized backend state, or an
   unreadable exact root is rejected.

These gates preserve unique failure attribution. A future regression at gate
5 is a builder or publication-memory problem. Failure at gate 7 is a native
candidate, exact-scoring, or routing problem. Passing per-query work but
failing gate 8 is a scheduler or shared-resource problem. No new model training
is justified until these engine causes have been resolved independently.

## Why Not Another Cache or Hot Fold

Hot fold and OS page-cache prefetch remain useful secondary optimizations:

- hot fold reduces the number of immutable segment surfaces;
- prefetch lowers cold page-fault latency;
- resident metadata avoids repeated header decoding.

None changes the mathematical amount of score accumulation. They should be
applied after the exact executor reduces competitive work. Otherwise they only
make a 25-million-posting scan somewhat warmer.

## Large-Scale Build and Query Gate

The bounded-execution implementation sequence is tracked in
[Semantic Accelerator Bounded Execution](semantic-accelerator-bounded-execution.md).
It treats SeismicWave-style neighbor expansion as optional metadata inside the
existing root-derived accelerator publication, not as another index authority
or mutation lifecycle.

The next PubMed/ArXiv build must use an isolated out-of-place index and the
normal convergent lifecycle. It is not necessary to minimize index bytes or
one-time build duration before query qualification. The required sequence is:

1. Set `evoke.maintenance_rebuild_memory_budget` to a deliberate host budget.
   Record the estimator, PostgreSQL backend peak RSS, runtime-server peak RSS,
   elapsed encoder time, and accelerator publication time separately. The
   product benchmark samples the build backend and runtime workers during
   `CREATE INDEX`, reporting initial, final, and peak RSS per process.
2. Complete the semantic root once. Accelerator publication must consume the
   existing postings and must not increase the runtime encoder run counter.
3. Confirm the published source manifest and authority checksum, restart
   PostgreSQL, and repeat exact query probes. The derived path may be dropped
   or rebuilt; the exact root must remain readable throughout.
4. Run cold and hot query matrices for exact packed, accelerator `h0`, and the
   explicit bounded preset. `h0` must be byte-identical to exact top-k. Report
   p50/p95/p99, postings, documents scored, clusters opened/skipped, residual
   postings/documents, forward chunk reads, fallback, and overlap.
5. Apply insert, update, and delete batches. The first query after mutation
   must be exact, the stale accelerator must be unreachable, and normal worker
   convergence must republish without model inference for unchanged completed
   documents.
6. Exercise cancellation during publication, crash/restart before and after
   manifest CAS, VACUUM/reclamation, and a root race. No orphan may remain
   reachable and no busy-loop may follow a memory-admission block.

Large-scale promotion requires query-time work and p95 improvement on the
heavy semantic queries. Build duration and derived bytes are reported for
operations planning, but they do not veto a faster exact or bounded query
path. Peak publication memory exceeding the configured budget, loss of exact
fallback, or unpredictable worker retry behavior does veto promotion.

The current candidate is therefore ready for an isolated large-root build,
not for default query routing. The 83-gate page-native suite and 31-gate SAE
lifecycle suite pass locally. A product benchmark smoke also verifies that
build RSS is sampled separately for the PostgreSQL backend and each runtime
worker. The FIQA control requires exact routing; only representative PubMed or
ArXiv high-DF evidence can promote the accelerator.

## Release Gates

The packed v7 stream is the exact default and does not depend on the geometric
accelerator. Its remaining release work is representative PubMed/ArXiv
rebuild, replica qualification, and p99/RSS measurement rather than another
format migration.

The bounded geometric path may be installed only after gates 5-9 above pass.
Even then:

- `exact` remains the default mode;
- `quality`, `balanced`, and `fast` require an explicit index or query policy;
- every bounded result reports the selected policy, clusters opened,
  candidates exact-scored, and whether exact fallback was used;
- root mismatch or lifecycle uncertainty always falls back to exact;
- no dataset-specific preset or silent auto-tuning is permitted.

The exact b16 selectivity and packed block-skipping results remain structural
improvements: exact pruning does not require a second impact copy. The newer
oracle adds the missing high-DF result: when consecutive document ranges are
not selective, term-local geometry can reduce candidate work by multiples at
a small, measurable quality cost. The next task is to carry that derived
metadata through the convergent fold lifecycle, not to add another cache or
loss function.

## Related Work Boundary

- [BMP](https://arxiv.org/abs/2405.01117) is the direct rank-safe algorithmic
  basis for small document blocks and forward scoring.
- [Dynamic Superblock Pruning](https://arxiv.org/abs/2504.17045) supports a
  two-level directory, but the PubMed evidence shows coarse bounds alone are
  too loose.
- [Efficiency Optimizations for Superblock-based Sparse Retrieval](https://arxiv.org/abs/2602.02883)
  supports lightweight superblock traversal and traversal-aware compression.
- [Approximate Cluster-Based Sparse Retrieval](https://arxiv.org/abs/2404.08896)
  supports segmented bounds, but its approximate controls are not enabled in
  this exact product path.
- [Seismic](https://arxiv.org/abs/2404.18812) is the basis for per-term
  geometric clusters, sparse maximum summaries, static posting pruning, and
  candidate score completion. The
  [official implementation](https://github.com/TusKANNy/seismic) is used only
  as the external representation oracle; Evoke must retain its own root and
  lifecycle contract.
- [SeismicWave](https://arxiv.org/abs/2408.04443) supports ordered cluster
  traversal and optional kNN expansion. Only ordered traversal is present in
  the first C prototype; graph expansion remains behind a separate memory and
  maintenance gate.
- [Bridging Dense and Sparse MIPS](https://arxiv.org/abs/2309.09013) motivates
  sparse IVF and cluster pruning, but the FIQA audit rejects one corpus-global
  IVF as the Evoke product structure.
- [Anytime Ranking](https://arxiv.org/abs/2104.08976) supports query-time
  latency/quality policies over document-ordered topical ranges. Evoke exposes
  a bounded policy instead of pretending that an approximate route is exact.
- The [official BMP implementation](https://github.com/pisa-engine/BMP) is the
  closest implementation reference. Evoke must still adapt the design to
  PostgreSQL MVCC, convergent segments, folds, and exact fallback.
