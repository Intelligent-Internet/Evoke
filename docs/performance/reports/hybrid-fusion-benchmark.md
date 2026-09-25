# Hybrid Fusion Benchmark

> **Owner-only diagnostic benchmark.** The measured fusion helpers are not
> granted to `PUBLIC` and are not an application query route. Applications use
> `evoke_query(...)`.

This benchmark validates the first hybrid BM25/vector late-fusion layer on a
local PostgreSQL 18 instance. The vector side uses synthetic vector-like
distance candidates so the benchmark does not require `pgvector` or
VectorChord.

Raw data:

- [Hybrid fusion diagnostics](../data/diagnostics/hybrid-fusion-current.json)
- [C vs SQL reference diagnostics](../data/diagnostics/hybrid-fusion-c-vs-sql-reference.json)
- [BM25 field-helper branch check](../data/diagnostics/sql-field-helpers-hybrid-branch-check.json)

## Setup

- documents: 20,000 for hybrid fusion
- BM25-only branch check: 4,000 documents
- queries: `bird`, `cat`, `bird OR cat`, `policy`
- returned top-k: 10
- repeats: 5 per query and variant
- candidate counts: 100, 500, 1000, 5000
- PostgreSQL: local Homebrew PostgreSQL 18
- CPU path: scalar fallback on this machine

## Hybrid Fusion Results

| candidate_k | variant | mean ms | p50 ms |
|---:|---|---:|---:|
| 100 | BM25 weighted queries | 0.713 | 0.595 |
| 100 | Hybrid RRF, two BM25 sources | 3.045 | 2.930 |
| 100 | Hybrid RRF, BM25 + synthetic vector | 11.486 | 9.315 |
| 100 | Hybrid score minmax, BM25 + synthetic vector | 9.194 | 9.201 |
| 500 | BM25 weighted queries | 1.571 | 1.497 |
| 500 | Hybrid RRF, two BM25 sources | 8.153 | 8.196 |
| 500 | Hybrid RRF, BM25 + synthetic vector | 16.287 | 16.259 |
| 500 | Hybrid score minmax, BM25 + synthetic vector | 18.819 | 16.563 |
| 1000 | BM25 weighted queries | 2.646 | 2.462 |
| 1000 | Hybrid RRF, two BM25 sources | 14.543 | 14.502 |
| 1000 | Hybrid RRF, BM25 + synthetic vector | 25.178 | 25.196 |
| 1000 | Hybrid score minmax, BM25 + synthetic vector | 25.457 | 25.487 |
| 5000 | BM25 weighted queries | 12.513 | 12.434 |
| 5000 | Hybrid RRF, two BM25 sources | 63.344 | 63.257 |
| 5000 | Hybrid RRF, BM25 + synthetic vector | 100.749 | 100.770 |
| 5000 | Hybrid score minmax, BM25 + synthetic vector | 100.725 | 100.614 |

## C Fast Path Versus SQL Reference

The owner-only fusion function runs the normalization, duplicate handling,
per-row aggregation, and final ordering in C. The SQL reference implementation
is retained for diagnostics. On the same 20,000-document synthetic benchmark,
the C path keeps the same result ordering while avoiding the expensive SQL
window/group pipeline:

| candidate_k | variant | C mean ms | SQL reference mean ms | speedup |
|---:|---|---:|---:|---:|
| 100 | RRF, BM25 + synthetic vector | 8.953 | 21.050 | 2.4x |
| 100 | score minmax, BM25 + synthetic vector | 8.867 | 19.434 | 2.2x |
| 500 | RRF, BM25 + synthetic vector | 16.552 | 193.720 | 11.7x |
| 500 | score minmax, BM25 + synthetic vector | 16.898 | 193.783 | 11.5x |
| 1000 | RRF, BM25 + synthetic vector | 25.847 | 722.111 | 27.9x |
| 1000 | score minmax, BM25 + synthetic vector | 26.313 | 742.017 | 28.2x |

## BM25-Only Branch Check

The existing BM25 field-helper benchmark still returns equivalent results
across all variants. This confirms the hybrid feature is additive and does not
rewrite the existing BM25-only fusion path.

| variant | mean ms | p50 ms | equivalent |
|---|---:|---:|---|
| baseline fused | 0.221 | 0.131 | true |
| weighted queries | 0.275 | 0.238 | true |
| field queries | 0.238 | 0.213 | true |
| search fields named | 0.253 | 0.224 | true |
| search fields short | 0.264 | 0.237 | true |

## Interpretation

The hybrid implementation is still late fusion and still accepts generic
candidate arrays, so it keeps the core extension independent from vector
extensions. The expensive fusion work itself is now C-backed, which makes
candidate pools in the low thousands practical for database-internal weighted
top-k.

For the API boundary and execution model behind these measurements, see
[Hybrid Fusion Engine](../../hybrid-fusion-engine.md).

The existing BM25 weighted-query diagnostic remains much faster for BM25-only
comparisons. Applications use `evoke_query(...)` regardless of whether the
index is exact BM25 or semantic-enabled unified posting.

If hybrid search becomes a very high-QPS primary path with candidate pools far
above 5000 per source, the next optimization should be a streaming/table-source
API that avoids building one large composite candidate array before fusion.
