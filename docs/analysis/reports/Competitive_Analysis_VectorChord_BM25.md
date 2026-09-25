# VectorChord-BM25 Analysis

Date: 2026-03-23

## Scope

This note evaluates
[`tensorchord/VectorChord-bm25`](https://github.com/tensorchord/VectorChord-bm25)
as a source of ideas for `evoke`.

The goal is not to reproduce its implementation. The goal is to separate:

- ideas that improve `evoke` without breaking its design goals
- ideas that are only useful in a future major format revision
- ideas that are good for VectorChord-BM25 but are not a good fit here

The analysis is based on the upstream repository at commit
`c610b7ab06ac1fc4a4ae2d24cd91616a9987efd9`.

## Executive Summary

VectorChord-BM25 is a more mature PostgreSQL-native BM25 access method,
but it solves a somewhat different problem from `evoke`.

It is built around:

- a typed sparse `bm25vector` value
- an index-bound `bm25query` value
- a true access-method-first query surface
- Block-WeakAnd over compressed postings
- incremental inserts, deletes, vacuum, and segment sealing
- an optional external tokenizer ecosystem

That gives it several real strengths:

- a cleaner sparse-vector API for pretokenized corpora
- better support for mutable corpora
- a native ranked-scan surface that is explicit about index identity
- a real filtered top-k prefilter path

But it also makes different tradeoffs:

- it is not a port of `bm25s`
- it is not built around eager sparse scoring
- it accepts a much more complex on-disk AM and segment architecture
- it depends on a separate tokenizer extension for its richer text stack

For `evoke`, the main conclusion is:

- do not replace the current `bm25s`-aligned retrieval core
- do learn from VectorChord-BM25's API discipline, mutable-index
  architecture, and correctness-testing style
- do consider selective future work around prepared sparse queries,
  generic filtered top-k scans, and stronger differential fuzzing

## What VectorChord-BM25 Actually Is

### Core data model

The extension centers on a sparse `bm25vector` SQL type rather than
directly indexing raw `text[]` or `int4[]`.

At the storage boundary, a `bm25vector` is:

- sorted term IDs
- per-term frequencies
- an explicit `doc_len`

Relevant source:

- `src/datatype/bm25vector.rs`
- `src/datatype/memory_bm25vector.rs`
- `src/datatype/text_bm25vector.rs`

This is an important API choice. It makes the ingestion contract
explicit: the extension wants a sparse term-frequency vector, not just a
bag of repeated token IDs.

### Query model

VectorChord-BM25 also makes the query object explicit.

It defines a composite `bm25query` type:

- `index_oid regclass`
- `query_vector bm25vector`

and a helper:

- `to_bm25query(index_oid, query_vector)`

Scoring is then exposed as:

- `embedding <&> to_bm25query(...)`

Relevant source:

- `src/datatype/functions.rs`
- `src/sql/finalize.sql`

This is a strong API idea. BM25 scores depend on corpus statistics, so
the query is explicitly bound to an index identity.

### Access method and ranking path

VectorChord-BM25 implements a real PostgreSQL access method with:

- `amcanorderbyop = true`
- `amgettuple`
- `ambuild`, `aminsert`, `ambulkdelete`, `amvacuumcleanup`

Relevant source:

- `src/index/am.rs`
- `src/index/scan.rs`
- `src/index/build.rs`
- `src/index/insert.rs`
- `src/index/vacuum.rs`

Its ranked retrieval path is based on:

- compressed postings
- block metadata
- fieldnorm quantization
- Block-WeakAnd early termination

Relevant source:

- `src/algorithm/block_wand.rs`
- `src/segment/posting/*.rs`
- `src/segment/field_norm.rs`

This is fundamentally different from `evoke`, which is still
organized around the `bm25s` eager sparse-scoring core.

### Mutable index architecture

VectorChord-BM25 supports ongoing writes with a two-part layout:

- a growing append-only segment
- a sealed inverted segment

As inserts arrive, vectors are appended to the growing segment. When the
growing segment passes a size threshold, it is sealed and merged into
the main inverted structure.

Deletes are tracked through a delete bitmap and vacuum refreshes term
statistics.

Relevant source:

- `src/segment/growing.rs`
- `src/segment/sealed.rs`
- `src/index/insert.rs`
- `src/index/vacuum.rs`

This is a real strength of the project, but it is also the single
biggest architectural difference from `evoke`.

### Tokenization and text processing

Modern VectorChord-BM25 no longer implements its main tokenizer stack in
this repository. It delegates that to `pg_tokenizer.rs`.

The README demonstrates:

- BERT tokenization
- custom analyzers
- stopwords
- stemming
- Chinese `jieba`
- Japanese `lindera`

But the current BM25 extension itself is primarily the ranking/index
layer. The richer tokenizer story lives in a separate extension.

Relevant source:

- upstream `README.md`
- `tests/init.sql`

## Comparison With `evoke`

### Where `evoke` is stronger

`evoke` is stronger when the priority is:

- fidelity to upstream `bm25s`
- a clear canonical exact BM25 function path
- high throughput on the existing serialized eager-score core
- explicit separation between canonical BM25 APIs and convenience SQL
  surfaces

In other words, `evoke` is currently the better implementation if
the project goal remains:

- "port `bm25s` into PostgreSQL without losing its core behavior"

### Where VectorChord-BM25 is stronger

VectorChord-BM25 is stronger when the priority is:

- pretokenized sparse-vector ingestion
- incremental inserts and deletes without forced rebuild semantics
- access-method-native ranked queries as the primary API
- clear index-bound query objects
- top-k correctness under extra SQL filters

In other words, it behaves more like:

- "a PostgreSQL-native BM25 indexing system"

than:

- "a `bm25s` port"

That distinction matters. It explains why some of its best ideas are
good future extensions for `evoke`, but not good replacements for
the current core.

## What `evoke` Should Learn Now

### 1. Prepared sparse query and vector APIs

This is the most useful API idea to borrow.

VectorChord-BM25's `bm25vector` and `bm25query` make two things
explicit:

- a sparse term-frequency representation is a first-class value
- corpus-dependent BM25 queries should be explicitly index-bound

`evoke` should consider an optional future surface like:

- a sparse term-frequency input type for precomputed corpora
- a prepared query value bound to one index

Benefits for `evoke`:

- less query marshalling for repeated searches
- fewer repeated token-to-ID resolution steps
- cleaner API semantics than relying only on raw arrays or raw text
- a possible foundation for a stricter BM25 operator surface

This fits the current project well because it can be added without
replacing the canonical `rowset` APIs.

### 2. Generic filtered top-k prefiltering

This is the most important execution-path idea to borrow.

VectorChord-BM25 has a prefilter path that checks candidate tuples
against MVCC visibility and ordinary quals during ranked retrieval.

Relevant source:

- `src/index/scan.rs`
- `tests/sqllogictest/prefilter.slt`

The key value is not the exact implementation. The key value is the
semantics:

- with `ORDER BY rank LIMIT k` plus extra filters, the index scan should
  try to return the top `k` qualifying rows, not just the top `k`
  global rows later filtered away

`evoke` currently has filtered ordered-scan integration centered on
`@@`. VectorChord-BM25 suggests a broader future direction:

- support simple extra quals during ranked scans
- keep LIMIT semantics correct under filtering
- do it inside the native AM path, not through SQL/SPI

This is genuinely useful and performance-relevant.

### 3. Stronger differential and fuzz testing

VectorChord-BM25 includes randomized correctness testing that compares:

- indexed ranked retrieval
- brute-force / disabled-index retrieval

under random insert/delete/select sequences.

Relevant source:

- `src/tests/fuzz.rs`

This is immediately useful for `evoke`.

The current `evoke` test story is already much better than where it
started, but a dedicated randomized differential harness would still be
valuable for:

- ranked scan correctness
- filtered ordered scans
- phrase verification paths
- stale/update edge cases
- future prepared-query surfaces

This is low-risk and high-value.

### 4. Cleaner ingestion path for pretokenized corpora

VectorChord-BM25's sparse vector contract is a real usability win for
large pretokenized corpora.

Right now, `evoke` accepts:

- `int4[]`
- `text[]`
- raw query text through parser helpers

That is practical, but repeated token IDs are a verbose carrier for
large precomputed corpora.

An optional sparse term-frequency carrier in `evoke` would help:

- large documents
- offline tokenization pipelines
- ingestion from external ML/tokenizer systems
- lower write amplification at the SQL boundary

This should remain optional. It should not replace the current array
surfaces.

## What `evoke` Should Learn Later, If Ever

### 1. Online mutable segment architecture

VectorChord-BM25's growing/sealed segment design is real engineering,
not a gimmick. It is probably the strongest answer in this analysis to:

- "How do we make BM25 updates work continuously inside PostgreSQL?"

But this is not a small feature. It is a major storage-model rewrite.

For `evoke`, this only makes sense as a future mutable-storage line of work if
the project decides that:

- online updates are more important than strict serialized `bm25s`
  storage fidelity

That tradeoff should be explicit before any implementation starts.

### 2. Block-max / WAND style pruning in a new index format

The Block-WeakAnd machinery is good engineering for compressed inverted
postings.

But `evoke` should not graft it onto the current core just because
it looks faster in the abstract. The current core is built around
`bm25s` eager sparse scoring, not a WAND iterator model.

If `evoke` ever adds a second storage engine optimized for:

- online updates
- much larger dynamic corpora
- stronger rank-time pruning

then VectorChord-BM25's block metadata and WAND pruning become highly
relevant.

For the current storage engine, they are not a drop-in win.

## Not Suitable for This Project

### 1. Replacing the current core with Block-WeakAnd

This would move `evoke` away from its stated purpose.

The project is valuable precisely because it is a `bm25s`-aligned
PostgreSQL port. Replacing its retrieval core with a very different
rank-time algorithm would undercut that.

### 2. Mandatory dependence on an external tokenizer extension

VectorChord-BM25's split between ranking and tokenization is reasonable
for that ecosystem, but `evoke` should not make that a hard
requirement.

The current built-in helper layer for tokenization, normalization,
stemming, diacritic folding, raw queries, highlighting, and snippets is
already useful and lightweight.

External tokenizer integration can be explored later as an optional
interoperability path, not as a forced dependency.

### 3. Negative-score ranking as the main API convention

VectorChord-BM25 negates BM25 scores so plain ascending `ORDER BY`
returns the best rows first.

That is pragmatic for its operator surface, but it is not a good fit for
`evoke`, which already distinguishes:

- canonical BM25 scores in `rowset`
- `<=>` as an ordered-scan/operator surface

Keeping positive canonical scores is clearer here.

### 4. Its cost-estimation and hook strategy as-is

VectorChord-BM25's cost estimation is intentionally simple, and its
executor hook is tightly tied to its AM setup.

Relevant source:

- `src/index/am.rs`
- `src/index/hook.rs`

The underlying lesson is useful: filtered ranked scans may need closer
executor/AM coordination.

The implementation itself should not be copied blindly.

## API Lessons

The most useful API lesson is not a specific operator. It is this:

- make corpus-dependent query semantics explicit

VectorChord-BM25 does this better than most BM25-in-Postgres projects.

Its `bm25query` type clearly says:

- this query is only meaningful against this index

That is better than pretending an arbitrary operator always has
standalone meaning.

For `evoke`, the most promising API follow-up is:

- an optional prepared-query or sparse-query value bound to an index
- while keeping `rowset` as the canonical exact BM25 path

That would improve both API clarity and repeated-query efficiency.

## Performance Lessons

The most important performance lesson is not "use WAND everywhere".

The real lessons are:

1. sparse term-frequency values are a good ingestion boundary
2. filtered top-k needs to happen inside the ranked scan, not after it
3. mutable corpora need a storage model designed for writes, not just
   query-time caching
4. differential testing matters when ranked scans, MVCC, deletes, and
   LIMIT interact

For the current `evoke` architecture, the best near-term
performance lessons are therefore:

- reduce repeated query preparation through prepared sparse queries
- broaden safe in-scan prefiltering
- strengthen randomized differential validation before adding more AM
  features

Not:

- replace the existing eager sparse-scoring engine

## Recommended Follow-Ups For `evoke`

### Worth adding to future plans

1. Optional prepared/index-bound query values.
2. Optional sparse term-frequency SQL input type for pretokenized
   corpora.
3. Broader native filtered ranked scans beyond the current `@@`-centered
   path.
4. Dedicated differential fuzz testing for indexed vs brute-force paths.
5. A future storage-engine research track for online updates, only if
   the project chooses to move beyond strict `bm25s` storage fidelity.

### Not worth adding to future plans

1. Replacing the current core with Block-WeakAnd.
2. Making external tokenization mandatory.
3. Reorienting the canonical API around negative scores.

## Bottom Line

VectorChord-BM25 is worth studying, but not suitable for wholesale adoption.

The best parts for `evoke` are:

- sparse-vector and prepared-query API discipline
- filtered top-k correctness inside the AM
- mutable-index storage ideas for a possible future storage revision
- stronger randomized differential testing

The parts to resist are:

- abandoning the current `bm25s`-aligned core
- turning `evoke` into a different search engine under the same
  name

The right way to learn from VectorChord-BM25 is to absorb its best
PostgreSQL-native ideas while keeping `evoke` anchored to its own
goal: a fast, explicit, semantically disciplined PostgreSQL port of
`bm25s`.
