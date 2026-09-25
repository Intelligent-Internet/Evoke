# ParadeDB Tantivy Fork Analysis

Date: 2026-04-02

This note reviews
[`paradedb/tantivy`](https://github.com/paradedb/tantivy) from the
perspective of `evoke`.

The goal is not to reproduce Tantivy or turn `evoke` into a Lucene-style
segment engine. The goal is to identify where Tantivy's speed comes from,
separate inherited Tantivy strengths from ParadeDB-specific changes, and
record what is worth learning.

The initial version of this note was based on source inspection only.
It now also records the follow-up `evoke` branch exploration, the
full local `main` vs branch validation, and the cloud PG18 rerun used
to refresh the published `evoke` benchmark columns.

## Scope and Sources

Primary sources inspected:

- [`paradedb/tantivy` repository](https://github.com/paradedb/tantivy)
- [`README.md`](https://github.com/paradedb/tantivy/blob/main/README.md)
- [`ARCHITECTURE.md`](https://github.com/paradedb/tantivy/blob/main/ARCHITECTURE.md)
- [`CHANGELOG.md`](https://github.com/paradedb/tantivy/blob/main/CHANGELOG.md)
- [`Cargo.toml`](https://github.com/paradedb/tantivy/blob/main/Cargo.toml)
- [`src/collector/top_score_collector.rs`](https://github.com/paradedb/tantivy/blob/main/src/collector/top_score_collector.rs)
- [`src/collector/sort_key_top_collector.rs`](https://github.com/paradedb/tantivy/blob/main/src/collector/sort_key_top_collector.rs)
- [`src/collector/filter_collector_wrapper.rs`](https://github.com/paradedb/tantivy/blob/main/src/collector/filter_collector_wrapper.rs)
- [`src/query/range_query/range_query_fastfield.rs`](https://github.com/paradedb/tantivy/blob/main/src/query/range_query/range_query_fastfield.rs)
- [`src/index/merge_optimized_inverted_index_reader.rs`](https://github.com/paradedb/tantivy/blob/main/src/index/merge_optimized_inverted_index_reader.rs)
- [`src/schema/document/default_document.rs`](https://github.com/paradedb/tantivy/blob/main/src/schema/document/default_document.rs)

## First Clarification: What This Repository Actually Is

This repository presents itself as Tantivy. The package metadata still points
at Quickwit's Tantivy homepage and repository, and the README still describes
it as a Rust full-text search library strongly inspired by Lucene.

That matters because the main performance story here is not "ParadeDB added a
few local speed hacks". The main performance story is that this codebase
inherits the performance properties of Tantivy itself: segment-oriented
indexing, mmap-backed readers, block-compressed postings, fast fields,
specialized collectors, and aggressive query-path specialization.

I did not find evidence in the current repository metadata or docs that this
repository is specifically presented as a `bm_search` fork. The safer and
more accurate description is:

- it is a ParadeDB-maintained Tantivy fork
- it is used as part of ParadeDB's broader search stack
- most core search-performance advantages come from Tantivy's own
  architecture

## Executive Summary

The biggest performance levers visible in `paradedb/tantivy` are:

1. specialized top-k collectors instead of generic heap-heavy ranking
2. broad use of fast-field and columnar execution paths for sort, filter, and
   range operations
3. mmap-centric, compact on-disk reader design with low anonymous-memory cost
4. segment and merge engineering that keeps both search and indexing scalable
5. block-oriented compression and SIMD-friendly inner loops in low-level data
   structures

The most important competitive conclusion for `evoke` is:

- Tantivy is fast because it is engineered as a complete search engine core,
  not because of one isolated trick.
- The parts most worth learning from are not the full segment architecture,
  but the way it specializes execution for top-k, filtering, sorting, and
  range queries.
- The current ParadeDB fork does not appear to derive its speed from a large
  set of ParadeDB-only performance patches. Most of the important speed story
  is upstream Tantivy design.

## Where the Performance Mostly Comes From

### 1. Top-k collection is treated as a dedicated hot path

This is one of the clearest performance themes in the code and changelog.

The changelog explicitly calls out:

- "Faster TopN: replace BinaryHeap with TopNComputer"
- `collect_block`
- fast-field sorting in `TopDocs`
- string fast-field support in `TopDocs`

The collector implementation reflects this. In
`src/collector/top_score_collector.rs`, the top-k path is not a naive
"collect everything and sort later" design. The collector comment states that
it repeatedly truncates with a median-based strategy after `K * 2` documents
and uses pattern-defeating quicksort, targeting roughly `O(N + K)` behavior.

`src/collector/sort_key_top_collector.rs` extends the same idea to sort-key
collection. It uses `TopNComputer` directly and merges only bounded top-k
state across segments.

This matters because top-k is the core user-facing query shape. Tantivy makes
that path first-class instead of treating it as a generic collector problem.

What to learn:

- collector design matters as much as scoring speed
- top-k needs specialized bounded-state machinery
- sorted top-k should have dedicated code paths, not generic post-processing

### 2. Fast fields are used aggressively to avoid the text path when possible

Tantivy's fast fields are its equivalent of doc values / columnar
random-access structures. In the architecture doc, they are treated as
central to ranking, aggregation, and post-filtering.

The most concrete example is
`src/query/range_query/range_query_fastfield.rs`. The file explicitly says
that if a fast-field implementation is available, the range query should use
that path instead of the default term-dictionary plus postings path.

That is a major performance lever. It means Tantivy does not force every
filter-like operation through the inverted text machinery. It dispatches to a
cheaper execution engine when field layout permits it.

`src/collector/filter_collector_wrapper.rs` shows the same philosophy on the
collector side: filtering can be performed against fast-field values and even
uses `collect_block` to pass filtered doc blocks downstream.

The changelog reinforces this theme with items like:

- faster fast-field range queries using SIMD
- `FastFieldRangeQuery`
- string fast-field support for `TopDocs`
- faster aggregation block fetch / full-column fast paths
- optional indices for sparse multivalued columns

What to learn:

- build more explicit field-layout-aware execution paths
- keep filter/sort/range optimization separate from pure text scoring
- where possible, route work to side structures rather than the main posting
  path

### 3. Reader-side layout is optimized for mmap and compact access

The architecture doc is explicit that Tantivy readers are designed to use
very little anonymous memory and read data straight from mmapped files.

That yields several advantages:

- low startup cost
- cheap reader reloads
- reduced heap pressure
- better cache locality on repeated searches

The README highlights this directly with:

- mmap directory
- tiny startup time
- compressed document store
- fast fields
- SIMD integer compression

On top of that, the changelog notes:

- `CompactDoc` replacing `TantivyDocument` with much smaller storage and
  similar performance
- faster sstable loading using FST-backed indices

The code aligns with this:

- `src/schema/document/default_document.rs` aliases `TantivyDocument` to
  `CompactDoc`
- `src/index/merge_optimized_inverted_index_reader.rs` uses buffered slices to
  reduce merge-time I/O overhead without loading everything into memory

What to learn:

- performance is heavily affected by read-path layout and memory behavior
- compact intermediate and stored representations can matter even when scoring
  math is unchanged
- low-memory reader design is a real performance feature, not just an
  operational convenience

### 4. Segment and merge engineering are part of the speed story

Tantivy's architecture doc makes it clear that the engine is built around
immutable segments, cheap snapshot searchers, and background merge work.

This matters because search performance is not only about per-query scoring.
It is also about:

- keeping segment counts under control
- reducing tombstones
- avoiding reader and merge I/O pathologies
- making indexing and searching coexist without constant full rewrites

The code contains merge-specific reader implementations and substantial merger
logic. The changelog also mentions many indexing and merge improvements,
including:

- faster indexing via term hashmaps in fastfield writers
- reduced indexing allocations and memory usage
- merge-related optimizations in inverted and columnar paths
- shared search executor

What to learn:

- read performance and write/merge discipline are coupled
- keeping background maintenance efficient is part of keeping queries fast
- even a read-focused engine benefits from explicit merge-optimized readers and
  maintenance paths

### 5. SIMD and blockwise compression are useful, but they are not the whole story

The README advertises SIMD integer compression, and the changelog mentions:

- SIMD linear search within blocks
- faster fast-field range queries using SIMD

This absolutely helps. But the more important observation is that SIMD is only
one layer in a broader block-oriented design:

- postings are block-compressed
- fast fields are bitpacked / columnar
- collectors support block collection
- query paths are rewritten to use block-friendly side structures

So the speed story is not "SIMD made everything fast". The speed story is
more like:

- engineer the data layout and query plan so that hot loops become cheap,
  bounded, and SIMD-friendly

What to learn:

- SIMD is most effective after the execution path and data layout are already
  well-specialized
- do not mistake one low-level optimization for the entire engine advantage

## What Seems ParadeDB-Specific vs. Inherited From Tantivy

Based on the repository state inspected here, the fork currently looks much
closer to "ParadeDB-maintained Tantivy" than to "a heavily rewritten new
search engine core".

Notable evidence:

- package metadata still points to Quickwit's Tantivy repository
- README and architecture docs still describe Tantivy itself
- the latest visible commit on the fork was Czech stemmer / stopword support,
  not a performance-core rewrite

That does not mean ParadeDB gets no benefit from maintaining the fork. It
means the bulk of the performance advantage should be credited to Tantivy's
existing architecture and long-running upstream engineering, not assumed to
be ParadeDB-only invention.

For competitive analysis, this matters because it changes the lesson:

- if ParadeDB is fast, one big reason is that it is standing on a strong
  search-engine core
- the relevant comparison target is not just ParadeDB's SQL APIs but also the
  broader Tantivy engine design underneath them

## What `evoke` Can Learn

### 1. Specialize top-k harder

Tantivy treats top-k as a dedicated algorithmic problem. `evoke` should
continue moving in that direction for ranked scan paths.

Useful directions:

- tighter bounded top-k structures
- more dedicated sorted top-k paths
- more aggressive block-oriented collector interfaces

### 2. Be clearer about fast paths

Tantivy has a strong pattern of: if field layout enables a better path, use
it explicitly.

For `evoke`, the lesson is not "become a full fast-field engine". The
lesson is:

- make execution-path dispatch more explicit
- explain when a filtered/ranked path is hitting a cheaper plan
- keep pushing native filtered ranked scans where the AM can exploit structure

### 3. Improve filter/sort side-structure strategy

Tantivy's range, sort, and filter story is strong because it has side
structures designed for them. That suggests continued work on `evoke`
side data for:

- filtered top-k
- ordered retrieval
- query-shape-aware execution

without changing the core BM25 contract.

### 4. Keep storage-layout performance in scope

Tantivy gains from compact readers, compact documents, buffered merge readers,
and memory-mapped access. For `evoke`, this reinforces that:

- on-disk layout and deserialization shape are performance-critical
- index loading and scan-start costs deserve just as much attention as pure
  scoring loops

### 5. Separate worth learning from worth adopting

Tantivy's whole design is coherent, but it is coherent for a segment-based
search engine library. `evoke` should learn from:

- execution specialization
- collector design
- side structures
- maintenance/read coupling

without trying to absorb:

- the full segment architecture
- the full docstore/fast-field system design
- the whole engine-level abstraction surface

## What Actually Converted Into Wins On This Branch

The follow-up branch did not just stop at code reading. We ran a long series
of focused local A/B checks and only kept changes that stayed positive after
reruns.

The ideas that actually converted into stable wins were:

- harder top-k specialization, especially on sorted / bounded result paths
- candidate-restricted ranking for filtered ordered scans
- narrower boolean candidate planning for:
  - single `MUST`
  - two `MUST`
  - positive `OR` / `AND` AST shapes
- verified-query candidate restriction instead of broad full-corpus ranking
- phrase candidate pruning for:
  - raw phrase queries
  - phrase scans
  - positive AST phrase planning
  - boolean-AST phrase scans
- lighter phrase-side ranking / verification structure, including token-length
  caching and lazy verification in scan-oriented paths

That is important because it confirms the code-reading conclusion:

- the right lessons from Tantivy were execution specialization and bounded
  collectors
- the wins came from narrowing candidate sets earlier and ranking less work
- the wins did **not** require turning `evoke` into a segment engine

The ideas that did **not** survive repeated validation were also revealing:

- generic parsed-query caches
- full exact-vocabulary lookup structures
- clause workspace reuse
- sparse workspace reuse in candidate gathering
- tiny local sort / stack-buffer tweaks
- generic verifier swaps such as `array_create_iterator(...)`
- direct `text[]` payload verification in the current storage layout

So the practical conclusion is sharper than the initial repository review:

- Tantivy-style thinking helped
- but only when translated into narrow PostgreSQL-native fast paths
- broad “adopt the engine internals” moves were either irrelevant to the
  current architecture or actively unhelpful

## Final Local Main-vs-Branch Result

After the focused micro-benchmark loop was exhausted, the branch was
validated against `main` on the full official 15-dataset BEIR list using
the local PostgreSQL benchmark harness.

Raw comparison data is archived in:

- [`tantivy-followup-local-main-vs-branch-2026-04-02`](../../performance/data/raw/tantivy-followup-local-main-vs-branch-2026-04-02)
- [`summary.md`](../../performance/data/raw/tantivy-followup-local-main-vs-branch-2026-04-02/summary.md)
- [`summary.json`](../../performance/data/raw/tantivy-followup-local-main-vs-branch-2026-04-02/summary.json)

Compared code states:

- baseline `main`: `f200342`
- candidate branch: `47f811f`

Aggregate outcome:

- `evoke_ids`
  - median query delta: `+18.60%`
  - mean query delta: `+20.98%`
  - wins: `14/15`
  - median build delta: `+2.69%`
- `evoke_text`
  - median query delta: `+24.52%`
  - mean query delta: `+18.84%`
  - wins: `13/15`
  - median build delta: `+2.16%`

Per-dataset query deltas:

| Dataset | ids delta | text delta |
|---|---:|---:|
| `arguana` | `-7.18%` | `-37.14%` |
| `climate-fever` | `+5.92%` | `+20.70%` |
| `cqadupstack` | `+31.05%` | `+35.62%` |
| `dbpedia-entity` | `+23.71%` | `+25.88%` |
| `fever` | `+11.42%` | `+11.91%` |
| `fiqa` | `+42.20%` | `+24.52%` |
| `hotpotqa` | `+9.10%` | `+8.49%` |
| `msmarco` | `+11.19%` | `+11.40%` |
| `nfcorpus` | `+14.86%` | `+33.58%` |
| `nq` | `+18.60%` | `+34.88%` |
| `quora` | `+66.36%` | `+66.58%` |
| `scidocs` | `+28.02%` | `+30.67%` |
| `scifact` | `+25.48%` | `+29.73%` |
| `trec-covid` | `+27.95%` | `-25.46%` |
| `webis-touche2020` | `+5.98%` | `+11.22%` |

This final check matters because it validates that the branch wins were
not confined to synthetic harnesses. The strongest gains carried over to
larger, more representative corpora such as:

- `quora`
- `cqadupstack`
- `dbpedia-entity`
- `fiqa`
- `nq`

and the large corpora still remained positive on both indexed paths:

- `fever`
- `hotpotqa`
- `msmarco`

### Interpreting the Small-Dataset Jitter

The final table still has a few regressions:

- `arguana / ids`: `-7.18%`
- `arguana / text`: `-37.14%`
- `trec-covid / text`: `-25.46%`

These are real benchmark outcomes, but they should be interpreted with
care.

`arguana` is a tiny corpus:

- `8,674` docs
- `1,406` queries

At that scale the benchmark is already in the sub-millisecond to
low-millisecond range. For example:

- `arguana / ids`
  - `0.685 ms -> 0.738 ms` average latency
- `arguana / text`
  - `0.852 ms -> 1.355 ms` average latency

That means fixed PostgreSQL call overhead, `text[]` token materialization,
and executor-side constants become a much larger fraction of total time
than they are on the medium and large datasets. The branch optimizations
mainly improved candidate narrowing and bounded ranking, so the branch is
expected to help more once ranking work is a dominant cost.

`trec-covid` is noisy for a different reason:

- `171,332` docs
- only `50` queries

Its text-path regression corresponds to:

- `3.63 ms -> 4.87 ms` average latency

That is a visible slowdown, but it is also a benchmark slice where a very
small number of slower samples can move the final percentage a lot.

So the correct reading is not "the branch regressed generally". The
correct reading is:

- the branch is strongly positive on medium and large datasets
- the branch is also positive on almost all small datasets
- the remaining losses sit on very small or statistically noisy slices
  where fixed overheads dominate the actual ranking work

## Cloud PG18 Matrix Refresh Against the Documented Baseline

After the local `main` vs branch validation was complete, the same
branch was rerun on Google Cloud in the project PG18 matrix shape, but
only for the two `evoke` paths:

- `evoke_ids`
- `evoke_text`

That rerun was then merged back into the existing project-wide PG18
matrix by replacing only those `30` cells and carrying forward the other
`45` cells (`upstream bm25s`, `pg_search`, `vchord_bm25`) unchanged.

Raw rerun data is archived in:

- [`pg18-beir-psql-only-tantivy-2026-04-02`](../../performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02)
- [`comparison-vs-doc-baseline.md`](../../performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02/comparison-vs-doc-baseline.md)
- [`comparison-vs-doc-baseline.json`](../../performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02/comparison-vs-doc-baseline.json)

Aggregate outcome versus the previously documented `2026-03-31` matrix:

- `evoke_ids`
  - median query delta: `+37.93%`
  - mean query delta: `+31.24%`
  - wins: `13/15`
  - median build delta: `-3.58%`
- `evoke_text`
  - median query delta: `+16.12%`
  - mean query delta: `+27.62%`
  - wins: `15/15`
  - median build delta: `-3.07%`

Per-dataset query deltas versus the documented matrix:

| Dataset | ids old | ids new | ids delta | text old | text new | text delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | `1141.17` | `1402.63` | `+22.91%` | `1094.53` | `1112.01` | `+1.60%` |
| `climate-fever` | `39.66` | `57.78` | `+45.69%` | `38.27` | `50.75` | `+32.59%` |
| `cqadupstack` | `321.27` | `443.13` | `+37.93%` | `233.90` | `438.42` | `+87.44%` |
| `dbpedia-entity` | `88.66` | `128.19` | `+44.58%` | `66.33` | `91.19` | `+37.49%` |
| `fever` | `67.94` | `97.56` | `+43.61%` | `72.05` | `80.15` | `+11.24%` |
| `fiqa` | `984.26` | `1409.52` | `+43.21%` | `900.86` | `1186.41` | `+31.70%` |
| `hotpotqa` | `56.61` | `55.40` | `-2.14%` | `47.53` | `49.86` | `+4.90%` |
| `msmarco` | `59.73` | `96.67` | `+61.84%` | `46.53` | `82.13` | `+76.50%` |
| `nfcorpus` | `3382.96` | `3373.94` | `-0.27%` | `3234.63` | `3326.96` | `+2.85%` |
| `nq` | `115.96` | `174.34` | `+50.35%` | `116.98` | `176.69` | `+51.04%` |
| `quora` | `448.25` | `637.98` | `+42.33%` | `445.23` | `619.64` | `+39.17%` |
| `scidocs` | `1373.32` | `1835.85` | `+33.68%` | `1432.67` | `1614.92` | `+12.72%` |
| `scifact` | `2203.32` | `2557.47` | `+16.07%` | `2181.21` | `2240.18` | `+2.70%` |
| `trec-covid` | `160.74` | `191.94` | `+19.41%` | `133.04` | `154.48` | `+16.12%` |
| `webis-touche2020` | `75.82` | `82.97` | `+9.43%` | `69.74` | `74.10` | `+6.25%` |

Per-dataset build deltas versus the documented matrix:

| Dataset | ids build old | ids build new | ids build delta | text build old | text build new | text build delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | `108.91` | `100.56` | `-7.67%` | `175.48` | `167.50` | `-4.54%` |
| `climate-fever` | `54940.95` | `52735.13` | `-4.01%` | `94879.89` | `91728.48` | `-3.32%` |
| `cqadupstack` | `5869.12` | `5827.17` | `-0.71%` | `10134.97` | `9291.73` | `-8.32%` |
| `dbpedia-entity` | `28447.66` | `26370.41` | `-7.30%` | `46473.55` | `45047.60` | `-3.07%` |
| `fever` | `54558.31` | `52830.13` | `-3.17%` | `95004.10` | `94593.96` | `-0.43%` |
| `fiqa` | `691.70` | `671.76` | `-2.88%` | `1052.00` | `1059.31` | `+0.69%` |
| `hotpotqa` | `31402.50` | `30792.23` | `-1.94%` | `49490.63` | `50378.30` | `+1.79%` |
| `msmarco` | `57120.97` | `56057.55` | `-1.86%` | `93096.14` | `87827.69` | `-5.66%` |
| `nfcorpus` | `85.21` | `81.54` | `-4.31%` | `122.17` | `113.12` | `-7.41%` |
| `nq` | `25100.79` | `23712.07` | `-5.53%` | `40360.51` | `40803.48` | `+1.10%` |
| `quora` | `801.58` | `787.85` | `-1.71%` | `1073.90` | `1064.56` | `-0.87%` |
| `scidocs` | `409.43` | `394.77` | `-3.58%` | `757.69` | `633.01` | `-16.46%` |
| `scifact` | `105.91` | `98.73` | `-6.78%` | `156.15` | `150.09` | `-3.88%` |
| `trec-covid` | `2781.23` | `2652.98` | `-4.61%` | `4382.88` | `4605.91` | `+5.09%` |
| `webis-touche2020` | `10025.63` | `9842.92` | `-1.82%` | `16121.97` | `16510.61` | `+2.41%` |

This cloud rerun matters because it validates the branch against the
same environment and same matrix shape that back the public project
benchmark pages. It also sharpens the small-dataset interpretation from
the local study:

- the largest and most operationally relevant datasets stayed strongly
  positive
- the two cloud regressions on the `ids` path were both tiny:
  `nfcorpus -0.27%` and `hotpotqa -2.14%`
- every `text[]` slice stayed positive in the cloud rerun

That is consistent with the earlier conclusion that the remaining
negative or flat points sit near the boundary where fixed PostgreSQL
overhead and small sample sizes can dominate the actual ranking work.

## Branch Closeout

This closes the Tantivy-inspired branch-local exploration cycle.

The main conclusion is now stable:

- the cheap, local ideas worth absorbing from the Tantivy comparison
  have already been absorbed on this branch
- the surviving wins are execution-shape specializations, not
  architectural imitation
- the remaining work is no longer "one more Tantivy-like fast path"

What remains worth doing belongs to the normal project roadmap rather
than this competitive-analysis thread:

- deeper storage and representation work
- stronger heap and `text[]` fetch redesign
- more architectural query-state reuse

Those are still plausible performance directions, but they are larger
design cycles in their own right. They should be evaluated as general
`evoke` future work, not as one more extension of this Tantivy
study.

## Not Suitable for This Project

### 1. Do not turn into a Tantivy clone

`evoke` is valuable partly because it remains a `bm25s`-aligned,
PostgreSQL-native extension with explicit SQL retrieval APIs. Replacing that
with a Tantivy-shaped engine would dissolve the project into something else.

### 2. Do not import the whole segment engine as a default answer

Segments, merge policies, and searcher snapshots are powerful, but they carry
substantial architectural commitments. They should not be copied merely
because they work well in Tantivy.

### 3. Do not over-interpret SIMD as the main lesson

The main lesson is execution-path specialization and data layout. SIMD helps,
but only after the rest of the engine is already structured around hot,
block-oriented loops.

## Net Assessment

The main performance advantage visible in `paradedb/tantivy` comes from a
mature search-engine core with several reinforcing properties:

- specialized top-k collectors
- fast-field / columnar dispatch for non-text work
- mmap-friendly compact reader design
- disciplined segment and merge engineering
- low-level block and SIMD optimizations

The most important competitive takeaway for `evoke` is therefore not
"reproduce Tantivy". It is:

- keep the BM25 core contract
- keep the PostgreSQL-native explicit API model
- invest harder in top-k specialization, filter/sort fast paths, and
  storage-layout efficiency
- treat broader engine architecture differences as context, not a blueprint

In short: Tantivy is fast mostly because it is an aggressively engineered
search core. The useful lessons for `evoke` are execution specialization
and layout discipline, not wholesale architectural imitation.
