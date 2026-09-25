# BM25 Page-Native Regression Audit

## Scope

This audit compares the March 2026 `psql_bm25s` SciFact benchmark with the
current Evoke BM25-only product path on PostgreSQL 18.6. The fixture contains
5,183 documents, 1,109 queries, 812,074 corpus tokens, 10,343 query tokens,
and a 26,559-term vocabulary. Both indexes use the same English stopword and
stemming tokenizer, Lucene BM25 with `k1=1.5`, `b=0.75`, `delta=0.5`, and
`top_k=1000`.

The reconstructed corpus token arrays match the installed fixture exactly.
The regression was not caused by dataset, tokenizer, or BM25 parameter drift.

## Root Cause

The original implementation retained one contiguous CSR image per backend.
The convergent index moved durable authority to relation pages and initially
routed every BM25 search through the generic page-native scorer. Its result
projection then reopened a fixed-size document COW block for every ranked hit.
At `k=1000`, native scoring took about 6.2 ms while repeated
document-slot-to-TID projection consumed another 57.4 ms.

The benchmark root had also been reused by a prior mutation test. Although the
table still contained exactly 5,183 rows, the index high-water mark contained
5,187 slots and four retired incarnations. A clean `REINDEX` restored a sealed
5,183-slot generation. Benchmark fixtures must not be mutated before a
performance run.

Three corrections were made:

1. page-native top-k projection groups ranked documents by COW block and loads
   each block once while retaining MVCC/HOT visibility checks;
2. when postmaster shared runtime is unavailable, a pure BM25 relation that
   fits `ii42.workspace_cache_bytes` may use a bounded resident sparse scorer;
3. when shared runtime is available, selected converged BM25 and SAE roots use
   the same pointer-free exact-root resident fold, otherwise both remain
   page-native;
4. a sealed generation whose stable incarnation order is exactly document-id
   order uses an 8-byte compact top-k item. Mutable generations retain the full
   incarnation key.

The resident scorer is only a query cache. Relation pages remain the sole
durable authority and rebuild, mutation, WAL, and maintenance lifecycles are
unchanged.

## Hot-Query Matrix

The historical row is the checked-in official SciFact result. The current row
uses the same 1,109 official queries, two warm passes, three measured rounds,
and a clean root. The current SQL aggregates the public SRF so that all 1,000
rows are consumed. The compact evidence summary is stored in
[`bm25-page-native-regression-2026-08-18.json`](../data/diagnostics/bm25-page-native-regression-2026-08-18.json).

| Route | Mean | p50 | p95 | QPS |
| --- | ---: | ---: | ---: | ---: |
| Historical official `psql_bm25s` | 0.454 ms | 0.451 ms | 0.525 ms | 2,203.3 |
| Evoke before repair | 63.430 ms | 63.370 ms | 65.896 ms | 15.8 |
| Evoke after repair | 0.448 ms | 0.435 ms | 0.558 ms | 2,234.1 |

Against the official historical anchor, repaired Evoke is 1.4% faster by mean
and 3.5% faster at p50; p95 is 6.3% slower. The original 110x regression is
removed.

A same-cluster, same-SRF-shape diagnostic measured the old binary at
0.314 ms mean and current Evoke at 0.448 ms. That difference is not an SAE
cost and is not a page scan: the current implementation additionally enforces
stable incarnation ordering and returns the PostgreSQL-snapshot-visible
HOT-chain TID. The historical result-array benchmark also used a different
public result shape and cannot isolate those costs.

## Correctness

Current bounded-resident and forced page-native execution produced identical
`ctid`, document id, rank, and score for 101 sampled official queries at
top1000. Maximum score error was zero. CRUD/maintenance checks also kept the
two execution paths equivalent after update, delete, insert, and segment
rotation.

The old binary is not a complete correctness oracle. On a 100-query sample
against upstream Python BM25S:

| Engine | Mean overlap@1000 | Minimum overlap@1000 | Exact top-10 queries |
| --- | ---: | ---: | ---: |
| Historical `psql_bm25s` | 0.5527 | 0.2620 | 59/100 |
| Current Evoke | 0.9877 | 0.4020 | 100/100 |

Scores for documents returned by both engines agree within about `4e-6`.
One reproduced old mismatch occurred at rank 67 for `k=1000`; requesting all
5,183 rows put the same document at rank 67 with the matching score. The old
limited top-k path therefore cannot be restored as a performance shortcut.

One documented boundary remains: under a concurrent update, a repeatable-read
transaction retains heap visibility but may observe generation-level score and
top-k membership drift between statements. Evoke intentionally pins one
current ranking root per statement rather than retaining historical ranking
roots for every transaction snapshot. Snapshot-invisible tuple versions remain
excluded. This is independent of the resident/page-native parity repair.

## SAE Impact

SAE did not cause this BM25 regression because every failing benchmark used
`sae=false`. SAE did, however, share two structural costs:

- page-native result projection used the same document COW lookup and therefore
  benefits from block-grouped projection;
- fragmented roots add extent-transition work to both lexical and semantic
  queries until maintenance converges them.

SAE still has additional costs that this repair does not remove: encoder
latency, high-DF semantic postings, bounded accelerator decisions, and shared
runtime scheduling. Those must be measured separately from the repaired BM25
baseline.

## Product Boundary

- `ii42.workspace_cache_bytes = 0` forces page-native BM25 execution.
- With no postmaster shared runtime, the default 32 MB budget admits only a
  pure BM25 relation whose complete physical size fits the bound. A zero or
  negative budget disables this fallback.
- When shared runtime is available, BM25 and SAE both use shared resident-fold
  or page-native execution; neither admits a backend-local index snapshot.
- SAE, field-aware special scoring, masks, and large BM25 indexes keep their
  page-native/shared-runtime lifecycle.
- The compact tie-break route is used only when the sealed incarnation order
  is provably identical to document-id order.
- Performance qualification requires a clean, converged root and a locked SQL
  result shape; a reused mutation fixture is not a valid baseline.
