# Unified Delta Cache Design And Acceptance

Date: 2026-07-26
Updated: 2026-07-27

> Historical baseline: the backend-local SAE fallback documented below was
> removed by the 2026-07-27 product closure. Current SAE creation and query
> require the shared runtime and a positive shared arena; foreground queries
> never build a private unified cache or run the full-corpus overlay.

## Product Decision

The product owns one `evoke` index relation, one durable mutation delta, and one
generation, transaction, maintenance, and publication lifecycle.

The query acceleration layer is reconstructible:

- `LEXICAL_DELTA_CACHE` serves `sae=false`.
- `UNIFIED_DELTA_CACHE` serves `sae=true`.

Neither cache is a sidecar index. A cache has no relation, WAL stream,
scheduler, generation number, or publication lifecycle of its own. Its exact
identity is derived from the owning index locator, generation, cache epoch,
delta extent and counters, and runtime contract signature.

BM25-only lexical caches may use the selected local or shared cache tier. The
model-backed unified cache is stricter: it must be published in the
postmaster-owned shared arena and is never built privately by a query backend.
The relation remains the only durable source of truth.

## Durable And Volatile Data

The compacted unified base stores the atom-oriented metadata needed to evaluate
new mutations without enumerating every base document:

- total document length and live-document metadata;
- lexical posting rows containing document ordinal and term frequency;
- semantic frontier postings and rank-level candidates;
- exact impact heads, document vectors, TIDs, and document lengths.

The existing relation-owned mutation delta stores complete versioned records:

- document TID, length, lexical term frequencies, and semantic frontier;
- tombstones and replacement-version information;
- the owning generation and delta extent.

The reconstructible unified cache contains mutation-proportional decoded data:

- live delta ordinals, TIDs, lengths, and document-local values;
- lexical and semantic atom posting lists;
- semantic rank-level candidates;
- tombstoned base ordinals;
- rank counts, effective document metadata, and bounded head patches.

The cache does not copy the base document set. Tombstone adjustments inspect
only referenced base documents.

## Exact Query Path

For an exact visible delta identity, a query:

1. attaches the exact current shared cache, waking its targeted publisher and
   waiting when necessary;
2. computes effective live counts, document lengths, lexical document
   frequencies, and semantic rank budgets;
3. traverses compacted base and delta postings into one score accumulator;
4. filters tombstones and stale tuple versions;
5. selects the exact top-k with the native score, TID, and document-ordinal
   tie-break.

The base path reuses monotonic semantic heads where the exact cutoff permits it.
Lexical head patches and semantic additions are stored in the same unified
cache payload. The final top-k uses a bounded best-first heap rather than
sorting every scored document.

The former full-corpus overlay is not a product fallback. It is available only
behind the explicit differential-test oracle used by the acceptance suite.

## Consistency And Failure Semantics

- `realtime`: committed changes are immediately searchable through exact
  heap-MVCC-filtered base-plus-delta state; bounded background maintenance
  converges the generation.
- `eventual`: committed complete mutation records remain exactly searchable
  through the unified cache while maintenance compacts them later.
- `manual`: changes mark the index stale and become visible after explicit
  maintenance, matching the existing product contract.

Cache publication is immutable and single-flight. An identity change makes an
old entry ineligible; leases protect it until readers release it. A PostgreSQL
restart or physical standby never replays cache bytes. The timer worker or a
query-triggered targeted publisher reconstructs the exact cache from the
relation-owned delta.

Model-backed index creation and query require a positive shared arena. Missing
shared runtime capacity is a fail-closed configuration error, not permission to
create a backend-local delta cache.

If a shared payload cannot be admitted, diagnostics expose the admission miss.
The product does not silently omit mutations or switch to a base full scan.

## Performance Acceptance

The refreshed benchmark used 4,096 compacted base documents, one fixed query,
`top_k=20`, five independent latency trials, and 101 warm queries per trial.
The live eventual index and a compacted reference index remained in the same
temporary cluster on independent connections. Their trial order alternated, and
the reported ratio is the median paired p95 ratio. This removes the systematic
cache-temperature bias caused by measuring every delta state before rebuilding
every compacted reference.

| Delta docs | Warm p50 ms | Warm p95 ms | p95/compacted | Base full scans | Exact |
|---:|---:|---:|---:|---:|:---:|
| 0 | 0.828 | 1.068 | 0.9628x | 0 | yes |
| 1 | 0.940 | 1.207 | 1.1270x | 0 | yes |
| 32 | 0.950 | 1.252 | 1.1472x | 0 | yes |
| 256 | 0.961 | 1.268 | 1.1300x | 0 | yes |
| 1000 | 1.052 | 1.366 | 1.1847x | 0 | yes |

The old full-overlay oracle scanned all 4,096 base documents at
`delta=1000` and measured 15.171 ms warm p95. The accepted query path measured
1.366 ms, an 11.1061x speedup. Maximum observed score error was
`1.7881393432617188e-7`, below the `1e-6` gate, with identical IDs and order.

All hard performance gates passed:

- base-only p95 regression: `0.9628x`, limit `1.03x`;
- delta 1-32 maximum p95 ratio: `1.1472x`, limit `1.15x`;
- threshold p95 ratio: `1.1847x`, limit `1.30x`;
- full-overlay speedup: `11.1061x`, minimum `10x`;
- base full-scan counter: zero at every delta level.

The artifact records the tested source, installed package, model, and host-load
provenance. Host load rose from `10.355` to `14.326` during the paired run, so
the ratios are retained as the acceptance measure rather than interpreting
absolute latency as an unloaded-host capacity claim.

The memory scaling probe used 256, 1,024, and 4,096 base documents with
32 delta documents. Retained backend memory growth was zero for every size,
exactness passed, and no base full scan occurred.

## Concurrency And Lifecycle Acceptance

- After an explicit cache clear, the targeted background publisher restored one
  exact cache in 70.016 ms. All 32 concurrent backends then attached it, did no
  foreground build, and produced one exact ranking digest.
- A PostgreSQL restart cleared volatile cache state; the background publisher
  restored it before the first query, and cold/warm exact comparisons passed.
- The current mutable lifecycle suite passed 58/58 gates, including INSERT, indexed
  UPDATE, HOT update, DELETE, VACUUM, maintenance, REINDEX, restart, immediate
  crash recovery, rollback, savepoints, failed rebuild rollback, bounded
  failed-batch memory, atomic delta/metapage WAL, online tail handoff, BM25
  lightweight-fold races, semantic base/delta snapshot races, and DROP.
- Physical replication passed for BM25 and SAE. A hot standby rebuilt the
  unified cache from replicated relation pages and returned the same results.
- Shared-preload oversized admission and pressure eviction lifecycle tests
  passed.
- Runtime provider, privilege, required-shared-runtime, restart, cancellation,
  and retained-memory soak gates passed.

## Build And Product Validation

- The PostgreSQL 18 production build and install completed successfully.
- Normal CTest and the ASAN/UBSAN CTest build both passed.
- The current installed build passed the 58-gate mutable lifecycle, physical
  replication, corruption fail-closed, concurrent DDL/rewrite, same-index
  concurrency, and 1,000-request runtime resource-soak gates.
- Full `pytest` passed 80/80. Python compilation, shell syntax, extension schema
  smoke, temporary-cluster regression, convergence inventory, and
  `git diff --check` passed.

The package-bound PostgreSQL 17/18 and PostgreSQL 18 container qualification is
intentionally a clean-commit release execution gate. The maturity runner
rejects a dirty source tree and requires the staged package, installed package,
and tested commit to share one fingerprint. This report does not weaken that
contract or reinterpret the current development worktree as release evidence.

Valgrind is not installed on the validation host. Apple's AddressSanitizer does
not implement LeakSanitizer, so the sanitizer run used
`detect_leaks=0:halt_on_error=1`; address and undefined-behavior checks passed.
The lifecycle and resource-soak suites remain the retained-memory evidence on
this host.

## Operational Diagnostics

`evoke_generation_cache_state_json(...)` exposes:

- `shared_unified_delta_cache_current` and loading state;
- unified cache entry count and arena bytes;
- exact relation debt and generation identity;
- cache admission state, misses, and eviction counters;
- query trace counters for cache build/hit/miss, decoded postings, candidate
  postings, scored documents, and prohibited base/full scans.

These are read-only diagnostics. No additional public search or lifecycle API
was added.

## Raw Evidence

The immutable acceptance artifacts and manifest are stored under:

`docs/performance/data/raw/unified-delta-cache-2026-07-26/`

The directory includes the old oracle, historical product gate, memory,
lifecycle, replication, and shared-preload lifecycle evidence. The refreshed
paired acceptance artifact is
`evoke-delta-cache-product-acceptance-paired-5x101.json`.
