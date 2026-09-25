# `pg_bm25s` Study

Date: 2026-03-21

This document consolidates the earlier `pg_bm25s` reports into one study:

- project-level design comparison
- same-machine PostgreSQL extension benchmark comparison
- the later alignment work that imported the worthwhile ideas into
  `evoke`

Compared projects:

- `evoke`: the source tree containing this report; its planned public
  repository is not yet provisioned.
- `pg_bm25s`: historical internal checkout; no public repository URL is
  available
- `pg_bm25s` branch: `perf/term-id-resolution`

## Scope

This study is about one question:

What should `evoke` learn from the separate `pg_bm25s`
implementation, and what should it explicitly avoid adopting?

The comparison is based on:

- source-level review of both codebases
- SQL surface and access-method comparison
- same-machine localhost PostgreSQL benchmark runs
- the later implementation and benchmarking of the useful ideas inside
  `evoke`

Raw benchmark files used in the final alignment/comparison stage:

- `pg-extension-focused-comparison-current-2026-03-21.json`
- `native-orderby-focus-topk1000-2026-03-21.json`
- `native-orderby-focus-topk10-2026-03-21.json`

## Executive Summary

`pg_bm25s` is broader as a SQL product.

`evoke` is stronger as a `bm25s`-aligned retrieval engine.

That remained true even after a fair PostgreSQL-native comparison on the
same localhost machine. The useful ideas from `pg_bm25s` were mostly in
the PostgreSQL integration layer:

- native ordered scans
- operator/planner-visible ranked retrieval
- SQL ergonomics around `@@` and `<=>`

Those ideas were worth adopting and were implemented in `evoke`.
But the results also showed their boundary: the new SQL-native ordered
scan path is useful, correct, and planner-visible, yet it is slower than
the existing `rowset` retrieval path.

So the final conclusion is:

- keep `evoke` as the main engine architecture
- treat `pg_bm25s` mainly as a source of SQL-surface and planner
  integration ideas
- do not adopt its SPI-heavy storage and retrieval model

## High-Level Comparison

| Area | `evoke` | `pg_bm25s` |
| --- | --- | --- |
| Core storage model | Serialized CSC payload stored inside the PostgreSQL index relation | Terms/postings stored in ordinary heap tables |
| Retrieval hot path | Pure C over cached deserialized BM25 state | SPI/SQL over heap-backed auxiliary index tables |
| Upstream `bm25s` fidelity | High | Medium-high for formulas, lower at full-system level |
| Planner/operator integration | Now supports `<=>`, ordered scans, and index-aware `@@` | Broader operator surface existed earlier |
| Incremental maintenance | Conservative stale-and-refresh model | More ambitious incremental/approximate maintenance |
| Feature breadth | Narrower but cleaner | Broader and more productized |
| Benchmark evidence | Strong and current in this repo | Mixed; later measured fairly through PostgreSQL |

## Design Comparison

### 1. Storage architecture

`evoke` stores its primary search state inside the index relation:

- metadata page
- `doc_id -> ctid` mapping
- serialized BM25 payload

This gives it a cleaner "real PostgreSQL index" shape:

- ordinary PostgreSQL persistence
- ordinary physical durability and replication behavior
- hot retrieval path stays in C over compact cached state

`pg_bm25s` stores the core search state in heap-backed helper tables.
That makes the data more SQL-visible and easier to inspect, but it also
means:

- the search state is outside the index relation
- the access method is more of a wrapper than the sole storage owner
- SQL/SPI overhead remains structurally important

For a long-term `bm25s` port, `evoke` has the better storage base.

### 2. Query execution model

`evoke` scores directly in C from cached deserialized index state.
Most of the earlier performance work in this repo removed:

- per-query payload reload
- per-query full deserialization
- dense full-document work on the common sparse path

That left mostly PostgreSQL-side overhead:

- relation open/close
- array deconstruction and marshalling
- visibility checks
- tuple materialization

`pg_bm25s` does much more work in SQL/SPI:

- fetch metadata through SPI
- query terms/postings through SQL
- expand prefixes via SQL
- aggregate matches in SQL
- filter dead rows through SQL-visible checks
- verify some phrase paths by rereading and retokenizing documents

That makes it flexible, but it also makes the hot path heavier.

### 3. Maintenance model

`evoke` is intentionally conservative:

- `INSERT` / `UPDATE` mark the index stale
- stale indexes reject ranked search
- exact rebuild happens through refresh / reindex
- delete correctness is protected at read time, but exact corpus stats
  after deletes still need rebuild

`pg_bm25s` is more ambitious:

- tries to maintain terms/postings incrementally
- keeps serving while approximate maintenance state exists
- offers a more continuously-updatable database feel

This is a real strength for `pg_bm25s`, but it comes with more nuanced
correctness and performance semantics.

### 4. SQL UX and feature surface

This is where `pg_bm25s` is clearly ahead as a product:

- raw-text indexing
- richer query syntax
- required / excluded / prefix / phrase terms
- `@@` and `<=>`
- stemming and stopwords
- highlight/snippet support

`evoke` was initially narrower and more retrieval-engine-facing.
Over the later alignment work it absorbed the useful SQL-side ideas
without changing the underlying engine design:

- raw-query AST
- `@@`
- highlight/snippet helpers
- `<=>`
- ordered AM scans
- combined `@@ + ORDER BY <=>` index scans

## Same-Machine PostgreSQL Benchmark Comparison

Source results JSON:

- `pg-extension-focused-comparison-current-2026-03-21.json`

This comparison used the fair path:

- both implementations ran inside the local PostgreSQL instance
- the same localhost machine was used throughout
- the same tokenized BEIR corpus was used
- `pg_bm25s` was measured via `CREATE INDEX USING bm25s`
- `evoke` used its PostgreSQL-native `rowset` search path

Scope of the focused completed subset:

- datasets:
  `arguana, fiqa, nfcorpus, scidocs, scifact, trec-covid, webis-touche2020`
- `top_k = 1000`

### Query Summary

`min`, `median`, and `max` below are per-dataset ratios against
same-machine local upstream Python `bm25s`.

| Path | At or above local upstream | Min vs local upstream | Median vs local upstream | Max vs local upstream |
| --- | ---: | ---: | ---: | ---: |
| `evoke ids` | `0/7` | `0.29x` | `0.48x` | `0.98x` |
| `evoke text[]` | `0/7` | `0.37x` | `0.58x` | `0.79x` |
| `pg_bm25s` | `0/7` | `0.00x` | `0.01x` | `0.15x` |

### Index Build Summary

`build` means index construction time only.

| Path | Total build vs local upstream |
| --- | ---: |
| `evoke ids` | `0.47x` |
| `evoke text[]` | `0.49x` |
| `pg_bm25s` | `27.11x` |

### Dataset Table

| Dataset | Docs | Queries | `bm25s` local QPS | `evoke ids` QPS | `ids / local` | `evoke text[]` QPS | `text[] / local` | `pg_bm25s` QPS | `pg_bm25s / local` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | 8,674 | 1,406 | 1513.39 | 1115.73 | 0.74x | 602.43 | 0.40x | 19.05 | 0.01x |
| `fiqa` | 57,638 | 648 | 1280.28 | 1102.06 | 0.86x | 794.43 | 0.62x | 7.55 | 0.01x |
| `nfcorpus` | 3,633 | 323 | 4953.49 | 4878.90 | 0.98x | 1834.45 | 0.37x | 748.95 | 0.15x |
| `scidocs` | 25,657 | 1,000 | 2180.66 | 1054.34 | 0.48x | 1728.96 | 0.79x | 28.03 | 0.01x |
| `scifact` | 5,183 | 300 | 3661.23 | 1746.35 | 0.48x | 2795.01 | 0.76x | 120.84 | 0.03x |
| `trec-covid` | 171,332 | 50 | 491.49 | 188.37 | 0.38x | 287.02 | 0.58x | 2.42 | 0.00x |
| `webis-touche2020` | 382,545 | 49 | 352.43 | 103.15 | 0.29x | 157.71 | 0.45x | 1.48 | 0.00x |

### Readout

The result is not close:

- `evoke ids` was consistently far faster than `pg_bm25s`
- `evoke text[]` was also consistently far faster than `pg_bm25s`
- `pg_bm25s` lagged heavily in both build and query throughput

The performance implication is simple:

- `pg_bm25s` has useful SQL ideas
- but its database-native hot path is not a competitive retrieval engine
  architecture for `evoke`

### Large-Set Note

Several large official BEIR datasets were omitted from the direct
comparison table because `pg_bm25s` native build was not practically
comparable on localhost:

- `climate-fever`
- `cqadupstack`
- `dbpedia-entity`
- `fever`
- `hotpotqa`
- `msmarco`
- `nq`
- `quora`

The clearest case was `climate-fever`, where native
`CREATE INDEX USING bm25s` remained active for roughly 15 minutes before
it was stopped.

## What Was Worth Learning

After reviewing `pg_bm25s` and the `perf/term-id-resolution` branch, the
useful ideas all turned out to be in the PostgreSQL integration layer,
not the core retrieval engine:

1. Native ordered scan support through `<=>` and `amgettuple`
2. Correct ordered-scan executor contract on PostgreSQL 17
3. Real planner-visible SQL integration instead of function-only search

Those ideas were later implemented in `evoke`.

## What Was Explicitly Not Adopted

The following `pg_bm25s` design choices were intentionally rejected:

- SPI-heavy search execution
- heap-table-backed terms/postings storage
- query-time SQL joins as the main retrieval path
- phrase verification by rereading and retokenizing stored documents as
  the general engine model
- generic CTID-hash accumulation in place of the existing `doc_id`
  workspace
- approximate maintenance as the default semantic contract

These either underperform the current `evoke` core path or solve
problems that `evoke` already handles better in its own design.

## What Changed In `evoke` After The Alignment

The alignment pass added:

- native `<=>` operators for `int4[]` and `text[]`
- ordered scans through the `evoke` access method
- PostgreSQL 17 ordered-scan state handling
- regression coverage for
  `ORDER BY ... <=> ... ASC LIMIT ...`

This means `evoke` now supports both:

- high-throughput function/result APIs
- SQL-native ranked retrieval through the planner

## Post-Alignment Benchmark Result

The new native ordered scan path is correct and useful for SQL
integration, but it is not the fastest query path in `evoke`.

### `top_k = 1000`

| Dataset | Local upstream `bm25s` QPS | `evoke` ids result QPS | `evoke` ids `<=>` QPS | ids `<=>` vs ids result | `evoke` text result QPS | `evoke` text `<=>` QPS | text `<=>` vs text result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | 1507.42 | 887.57 | 73.27 | 0.08x | 1107.07 | 13.55 | 0.01x |
| `scifact` | 3454.73 | 3104.14 | 514.53 | 0.17x | 2464.90 | 105.82 | 0.04x |

### `top_k = 10`

| Dataset | Local upstream `bm25s` QPS | `evoke` ids result QPS | `evoke` ids `<=>` QPS | ids `<=>` vs ids result | `evoke` text result QPS | `evoke` text `<=>` QPS | text `<=>` vs text result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `scifact` | 9451.81 | 4282.78 | 3120.54 | 0.73x | 10004.81 | 2663.38 | 0.27x |

### Meaning

The alignment result was clear:

- native `<=>` works correctly
- it is only used when queries explicitly ask for
  `ORDER BY ... <=> ... ASC LIMIT ...`
- it does not replace or slow down
  the `evoke_query_*` rowset APIs
- `evoke_query_ids(...)` and `evoke_query_tokens(...)`
  remain the canonical exact BM25 APIs
- the ordered AM path is slower mainly because PostgreSQL executes it as
  tuple-by-tuple index-scan output rather than the batched array return
  shape used by `rowset`

So native ordered scan is a SQL ergonomics feature, not a benchmark-path
replacement.

## Final Position

After the design review, the fair localhost extension benchmark, and the
later alignment work, the conclusion is stable:

- `evoke` should remain the primary long-term foundation
- `pg_bm25s` should be treated mainly as a reference for SQL surface,
  planner integration, and user-facing search UX
- `evoke_query_ids(...)` and
  `evoke_query_tokens(...)` should remain the preferred
  high-throughput APIs
- native `<=>` support should stay, because it makes the extension more
  usable inside ordinary SQL
- the `pg_bm25s` storage and SPI-heavy retrieval architecture should not
  be copied into `evoke`

## Practical Recommendation

Use:

- `evoke_query_ids(...)` for the fastest `int4[]` retrieval
- `evoke_query_tokens(...)` for the fastest `text[]`
  retrieval
- `ORDER BY column <=> query ASC LIMIT k` when SQL-native ranked queries
  matter more than absolute throughput

Treat `pg_bm25s` as a source of feature and integration ideas, not as the
retrieval-engine architecture to inherit.
