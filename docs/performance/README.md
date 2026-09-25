# Performance and Benchmarks

This page is the default performance entry point for `ii42`.

## Current Guides And Dated Evidence

- [Testing and Validation](../testing-and-validation.md): current release gates and reproduction commands.
- [Evoke System Technical Report (Beta 1)](../technical-report-evoke-system.md): current architecture and engineering evidence.
- [Evoke Model Technical Report (Beta 1)](../technical-report-evoke-model.md): semantic relevance experiments and their scope.
- [Maintenance Policy Tuning](maintenance-policy-tuning.md): supported policy choices and current scheduling behavior.
- [BM25 Page-Native Regression](reports/bm25-page-native-regression-2026-08-18.md): dated lexical regression evidence for the page-native transition.
- [Mutable Maintenance Benchmarks](mutable-maintenance-benchmarks.md): historical lexical maintenance checkpoint.
- [Semantic Accelerator Bounded Execution](reports/semantic-accelerator-bounded-execution.md): historical accelerator milestone, not the latest query contract.

The matrix below is frozen exact-BM25 cross-engine evidence: throughput,
construction, and storage were measured in March/April 2026; relevance was
measured in May 2026. It does not benchmark the later `sae = true` unified
posting route or certify the current Beta 1 binary. The
[Enlightenment P2 report](reports/enlightenment-p2-production-qualification-2026-07-22.md)
is dated evidence from a superseded catalog/runtime candidate, not current
release qualification. Current lifecycle and package gates are defined in
[Testing and Validation](../testing-and-validation.md). Do not combine numbers
from these different hardware, version, and workload surfaces into one
like-for-like performance claim.

The 2026-08-26 planner-native Commons qualification snapshot is documented in the
[Shadow Evoke 0.2.5 report](reports/csg-beta-natural-search-shadow-2026-08-26.md).
It records an API/exactness/no-regression gate on that deployment, not a replacement
for the frozen cross-engine matrix below.

## Frozen Lexical Cross-Engine Reference

The PG18 `15 x 5` BEIR matrix compares:

- upstream Python `bm25s`
- `ii42 ids`
- `ii42 text[]`
- ParadeDB `pg_search`
- TensorChord `vchord_bm25`

The `ii42` labels below are normalized names for the earlier `psql_bm25s`
implementation; raw archives preserve the original engine names and run
provenance. Renaming a label does not rerun a benchmark against Evoke.

The corresponding current owner-only exact BM25 diagnostic APIs are regression
anchors, not recommended application entrypoints:

- `ii42_query_ids(...)`
- `ii42_query_tokens(...)`

Applications use the public [`ii42_query(...)` family](../api-reference.md).

The published matrix on this page is intentionally based on the
pretokenized index paths:

- `ii42 ids` uses `int4[]`
- `ii42 text[]` uses `text[]`

Scalar `text` and `varchar` source columns are supported in the
extension, but they are not the basis of the public `2026-04-02`
cross-engine matrix. See [Supported Input Types](../input-types.md) for
the type-by-type contract and trade-offs.

Raw data used by this page:

- [Token-ID results](data/canonical/official-beir-ids-current-2026-04-02.json)
- [Token-stream results](data/canonical/official-beir-text-current-2026-04-02.json)
- [Upstream Python results](data/diagnostics/official-beir-upstream-current-2026-04-02.json)
- [PG18 comparison](data/diagnostics/official-beir-pg18-comparison-current-2026-04-02.json)
- [Complete engine matrix](data/diagnostics/pg18-beir-extension-matrix-current-2026-04-02.json)
- [Relevance matrix](data/diagnostics/pg18-beir-quality-matrix-current-2026-05-06.json)

Feature-specific local checks:

- [Hybrid Fusion Benchmark](reports/hybrid-fusion-benchmark.md)
- [Convergent Query-State Matrix](reports/convergent-query-state-matrix-2026-07-31.md)
- [Impact-Banded Residual Authority Oracle](reports/impact-banded-residual-authority-oracle-2026-08-27.md)

Full raw per-task archives:

- [2026-03-31 full matrix](data/raw/pg18-beir-extension-matrix-2026-03-31/)
- [2026-04-02 lexical rerun](data/raw/pg18-beir-psql-only-tantivy-2026-04-02/)

Validation status:

- `75/75` dataset-engine cells in the rolled-up frozen matrix were
  checked against the backing raw archives
- `30/75` refreshed `ii42` cells come from the
  `2026-04-02` Google Cloud rerun
- `45/75` carried-forward upstream / `pg_search` / `vchord_bm25` cells
  remain pinned to the stable `2026-03-31` PG18 matrix
- the frozen matrix and the raw archives match exactly on:
  - `stats`
  - `build_ms`
  - `query.count`
  - `query.qps`

## How To Read This Section

This comparison is based on the frozen like-for-like PG18 matrix,
not the older localhost desktop run.

That distinction matters:

- the older 2026-03-21 "current" files were a localhost M2 Ultra study
- the 2026-03-31 files established the stable GCP PG18 `15 x 5` matrix
- the 2026-04-02 refresh reran only `ii42 ids` and
  `ii42 text[]` on the same GCP shape
- the refreshed 2026-04-02 matrix is the source for the lexical
  cross-engine comparison reproduced here

The published BM25S paper QPS table is still useful as historical
background, but it is not the headline comparison here. This experiment's
official cache uses larger query sets on several datasets. A new release
requires fresh same-shape qualification rather than inheriting these results.

## Scope

The frozen read-performance comparison combines two aligned GCP PG18 runs:

- the stable `2026-03-31` full `15 x 5` matrix
- the `2026-04-02` rerun that refreshed only
  `ii42 ids` and `ii42 text[]`
- all 15 official BEIR subsets used in the BM25S benchmark set
- the local-uploaded dataset cache recorded with those runs
- `top_k = 1000`

Benchmark scope:

- cloud: Google Cloud
- zone: `us-east4-a`
- machine type: `n2-standard-16`
- PostgreSQL: `18`
- dataset delivery: local uploaded cache, not runtime download
- upstream path: Python `bm25s` carried forward from the stable
  `2026-03-31` matrix
- PostgreSQL paths:
  - `ii42_query_ids(...)` refreshed on `2026-04-02`
  - `ii42_query_tokens(...)` refreshed on `2026-04-02`
  - ParadeDB `pg_search` carried forward from `2026-03-31`
  - TensorChord `vchord_bm25` carried forward from `2026-03-31`

For the lexical extension, the recorded exact-path configuration was:

- `method = 'lucene'`
- `idf_method = 'lucene'`
- `k1 = 1.5`
- `b = 0.75`
- `delta = 0.5`
- `consistency = 'manual'`

## SIMD Microbenchmark

For direct AVX2-vs-fallback measurement of the core scoring path, use
the local microbenchmark target:

```bash
cmake -S . -B build_tmp -DCMAKE_BUILD_TYPE=Release
cmake --build build_tmp -j
II42_SIMD_MODE=scalar ./build_tmp/ii42_simd_bench
II42_SIMD_MODE=avx2 ./build_tmp/ii42_simd_bench
```

Notes:

- `II42_SIMD_MODE=auto` is the default.
- The benchmark prints both `simd_env` and `simd_active`.
- If `simd_env=avx2` but hardware support is unavailable, the active
  path remains `scalar`.
- Building on machines without AVX2 support is allowed. The build emits
  a warning, and the resulting extension uses the scalar path on that
  machine unless AVX2-capable hardware is available at runtime.
- Optional positional args are:
  `./build_tmp/ii42_simd_bench [num_docs] [timed_iters] [warmup_iters]`

## Query Summary

`min`, `median`, and `max` below mean the per-dataset ratio of engine
QPS against upstream Python `bm25s` across the full 15-dataset suite.

| Path | At or above upstream | Min vs upstream | Median vs upstream | Max vs upstream |
| --- | ---: | ---: | ---: | ---: |
| `ii42 ids` | `12/15` | `0.35x` | `3.97x` | `60.10x` |
| `ii42 text[]` | `11/15` | `0.31x` | `3.93x` | `51.06x` |
| `pg_search` | `3/15` | `0.01x` | `0.17x` | `2.76x` |
| `vchord_bm25` | `7/15` | `0.07x` | `0.54x` | `11.31x` |

## Index Build Summary

`build` here means index construction time only.

| Path | Total build ms | Relative to upstream |
| --- | ---: | ---: |
| upstream Python `bm25s` | `848046.35` | `1.00x` |
| `ii42 ids` | `262955.79` | `0.31x` |
| `ii42 text[]` | `443975.35` | `0.52x` |
| `pg_search` | `356944.25` | `0.42x` |
| `vchord_bm25` | `739014.63` | `0.87x` |

## Dataset Table

| Dataset | Docs | Queries | upstream `bm25s` QPS | `ii42 ids` QPS | `ii42 text[]` QPS | `pg_search` QPS | `vchord_bm25` QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | 8,674 | 1,406 | 1158.34 | 1402.63 | 1112.01 | 115.94 | 78.77 |
| `climate-fever` | 5,416,593 | 1,535 | 3.04 | 57.78 | 50.75 | 2.84 | 5.25 |
| `cqadupstack` | 457,199 | 13,145 | 111.56 | 443.13 | 438.42 | 13.99 | 60.51 |
| `dbpedia-entity` | 4,635,922 | 467 | 3.47 | 128.19 | 91.19 | 5.21 | 23.66 |
| `fever` | 5,416,568 | 123,142 | 3.15 | 97.56 | 80.15 | 5.62 | 12.13 |
| `fiqa` | 57,638 | 6,648 | 810.51 | 1409.52 | 1186.41 | 17.76 | 190.57 |
| `hotpotqa` | 5,233,329 | 97,852 | 4.16 | 55.40 | 49.86 | 3.42 | 9.31 |
| `msmarco` | 8,841,823 | 509,962 | 1.61 | 96.67 | 82.13 | 4.44 | 18.20 |
| `nfcorpus` | 3,633 | 3,237 | 3155.35 | 3373.94 | 3326.96 | 1132.17 | 1252.75 |
| `nq` | 2,681,468 | 3,452 | 10.55 | 174.34 | 176.69 | 6.28 | 21.96 |
| `quora` | 522,931 | 15,000 | 90.56 | 637.98 | 619.64 | 13.26 | 154.36 |
| `scidocs` | 25,657 | 1,000 | 1203.09 | 1835.85 | 1614.92 | 17.89 | 367.04 |
| `scifact` | 5,183 | 1,109 | 2964.86 | 2557.47 | 2240.18 | 500.04 | 629.42 |
| `trec-covid` | 171,332 | 50 | 210.50 | 191.94 | 154.48 | 8.75 | 75.66 |
| `webis-touche2020` | 382,545 | 49 | 240.36 | 82.97 | 74.10 | 8.14 | 86.04 |

## Scale Trend

![QPS vs dataset scale](reports/pg18-qps-vs-dataset-scale-2026-04-02.svg)

## Index Build Matrix

Query throughput is still the headline metric, but build time matters
for every bulk load, refresh, and reproducible end-to-end deployment.
The matrix below reports index construction time only. It excludes query
execution and VM orchestration overhead so the comparison stays focused
on the actual indexing cost of each engine.

The same dataset order is used as the query matrix above. That makes it
easy to compare the two dimensions directly: one table answers "how fast
is query execution after the index exists?" and the second answers "how
expensive is it to get to that state?".

| Dataset | Docs | Queries | upstream `bm25s` build | `ii42 ids` build | `ii42 text[]` build | `pg_search` build | `vchord_bm25` build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nfcorpus` | 3,633 | 3,237 | 167ms | 82ms | 113ms | 254ms | 326ms |
| `scifact` | 5,183 | 1,109 | 224ms | 99ms | 150ms | 326ms | 449ms |
| `arguana` | 8,674 | 1,406 | 318ms | 101ms | 168ms | 376ms | 448ms |
| `scidocs` | 25,657 | 1,000 | 974ms | 395ms | 633ms | 490ms | 1.10s |
| `fiqa` | 57,638 | 6,648 | 1.88s | 672ms | 1.06s | 705ms | 1.32s |
| `trec-covid` | 171,332 | 50 | 6.67s | 2.65s | 4.61s | 2.86s | 3.80s |
| `webis-touche2020` | 382,545 | 49 | 21.18s | 9.84s | 16.51s | 10.65s | 13.39s |
| `cqadupstack` | 457,199 | 13,145 | 15.68s | 5.83s | 9.29s | 7.21s | 19.56s |
| `quora` | 522,931 | 15,000 | 5.80s | 788ms | 1.06s | 936ms | 1.69s |
| `nq` | 2,681,468 | 3,452 | 1.2m | 23.71s | 40.80s | 31.46s | 55.57s |
| `dbpedia-entity` | 4,635,922 | 467 | 1.7m | 26.37s | 45.05s | 40.73s | 1.9m |
| `hotpotqa` | 5,233,329 | 97,852 | 1.8m | 30.79s | 50.38s | 41.97s | 1.9m |
| `fever` | 5,416,568 | 123,142 | 2.6m | 52.83s | 1.6m | 1.2m | 2.7m |
| `climate-fever` | 5,416,593 | 1,535 | 2.7m | 52.74s | 1.5m | 1.2m | 2.5m |
| `msmarco` | 8,841,823 | 509,962 | 3.2m | 56.06s | 1.5m | 1.2m | 1.7m |

## Build Trend

![Index build time vs dataset scale](reports/pg18-build-vs-dataset-scale-2026-04-02.svg)

Build-time trend highlights:

- `ii42 ids` is the clear build-time winner. It is faster than
  upstream on `15/15` datasets and has the lowest median build ratio at
  `0.34x` of upstream.
- `ii42 text[]` is also consistently cheaper than upstream. It is
  faster on `15/15` datasets with a median build ratio of `0.56x`.
- `pg_search` is generally competitive on build time and still beats
  upstream on `12/15` datasets, but it does not carry that advantage
  over to query throughput.
- `vchord_bm25` has a more mixed build story. It beats upstream on
  `7/15` datasets, but its median build ratio is just above parity and
  it climbs sharply on several larger datasets.
- Both lexical paths have higher suite-median query ratios than the compared
  PostgreSQL plugins and build faster than upstream on all 15 datasets.
  This is not a per-dataset query win: `vchord_bm25` is faster than both on
  `webis-touche2020` in this matrix.

## Index Size Matrix

The matrix also records PostgreSQL index relation size as `build_bytes`.
The upstream Python `bm25s` path is omitted here because it is not a
PostgreSQL index relation and does not report an equivalent byte count.

| Dataset | Docs | `ii42 ids` size | `ii42 text[]` size | `pg_search` size | `vchord_bm25` size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nfcorpus` | 3,633 | 4.60 MiB | 4.80 MiB | 4.98 MiB | 165.37 MiB |
| `scifact` | 5,183 | 6.09 MiB | 6.35 MiB | 5.70 MiB | 227.52 MiB |
| `arguana` | 8,674 | 8.51 MiB | 8.73 MiB | 6.43 MiB | 205.01 MiB |
| `scidocs` | 25,657 | 24.81 MiB | 25.43 MiB | 19.67 MiB | 510.23 MiB |
| `fiqa` | 57,638 | 42.96 MiB | 43.62 MiB | 25.55 MiB | 533.44 MiB |
| `trec-covid` | 171,332 | 151.95 MiB | 153.91 MiB | 77.53 MiB | 1.49 GiB |
| `webis-touche2020` | 382,545 | 484.02 MiB | 488.45 MiB | 290.91 MiB | 3.02 GiB |
| `cqadupstack` | 457,199 | 323.90 MiB | 334.80 MiB | 204.12 MiB | 6.92 GiB |
| `quora` | 522,931 | 52.29 MiB | 52.95 MiB | 28.55 MiB | 602.66 MiB |
| `nq` | 2,681,468 | 1.34 GiB | 1.35 GiB | 725.37 MiB | 7.75 GiB |
| `dbpedia-entity` | 4,635,922 | 1.50 GiB | 1.54 GiB | 1022.44 MiB | 22.13 GiB |
| `hotpotqa` | 5,233,329 | 1.56 GiB | 1.59 GiB | 1.11 GiB | 20.70 GiB |
| `fever` | 5,416,568 | 2.66 GiB | 2.69 GiB | 1.70 GiB | 25.47 GiB |
| `climate-fever` | 5,416,593 | 2.66 GiB | 2.69 GiB | 1.70 GiB | 25.47 GiB |
| `msmarco` | 8,841,823 | 2.98 GiB | 3.00 GiB | 1.65 GiB | 12.59 GiB |

## Index Size Trend

![Index size vs dataset scale](reports/pg18-index-size-vs-dataset-scale-2026-04-02.svg)

## Quality Matrix

The 2026-05-06 quality matrix is a local PG18 relevance run over the same 15 BEIR
datasets. It measures `NDCG@10`, `MAP@100`, `Recall@100`, and
`Precision@10` with `top_k = 100`.

This table uses qrels-bearing queries only. That means the evaluated query
count can be smaller than the full query count shown in the QPS table.

The local machine had all five comparison engines available for this
relevance run: upstream Python `bm25s`, `ii42 ids`,
`ii42 text[]`, `pg_search`, and `vchord_bm25`.

The primary chart is an absolute-score heatmap rather than a dataset-scale
line chart. Each cell is the metric value for one engine on one dataset.
Bold cells are within `0.001` of the best engine for that dataset and
metric.

![Quality score heatmap](reports/pg18-quality-score-heatmap-2026-05-06.svg)

| Dataset | Docs | Eval queries | Engine | NDCG@10 | MAP@100 | Recall@100 | Precision@10 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `nfcorpus` | 3,633 | 323 | upstream `bm25s` | 0.3230 | 0.1533 | 0.2474 | 0.2319 |
| `nfcorpus` | 3,633 | 323 | `ii42 ids` | 0.3235 | 0.1535 | 0.2503 | 0.2322 |
| `nfcorpus` | 3,633 | 323 | `ii42 text[]` | 0.3235 | 0.1535 | 0.2504 | 0.2322 |
| `nfcorpus` | 3,633 | 323 | `pg_search` | 0.3215 | 0.1523 | 0.2486 | 0.2313 |
| `nfcorpus` | 3,633 | 323 | `vchord_bm25` | 0.3209 | 0.1518 | 0.2468 | 0.2303 |
| `scifact` | 5,183 | 300 | upstream `bm25s` | 0.6863 | 0.6439 | 0.9127 | 0.0907 |
| `scifact` | 5,183 | 300 | `ii42 ids` | 0.6863 | 0.6439 | 0.9127 | 0.0907 |
| `scifact` | 5,183 | 300 | `ii42 text[]` | 0.6863 | 0.6439 | 0.9127 | 0.0907 |
| `scifact` | 5,183 | 300 | `pg_search` | 0.6819 | 0.6397 | 0.9127 | 0.0900 |
| `scifact` | 5,183 | 300 | `vchord_bm25` | 0.6766 | 0.6350 | 0.9127 | 0.0893 |
| `arguana` | 8,674 | 1,406 | upstream `bm25s` | 0.3655 | 0.2524 | 0.9659 | 0.0760 |
| `arguana` | 8,674 | 1,406 | `ii42 ids` | 0.3656 | 0.2524 | 0.9659 | 0.0760 |
| `arguana` | 8,674 | 1,406 | `ii42 text[]` | 0.3656 | 0.2524 | 0.9659 | 0.0760 |
| `arguana` | 8,674 | 1,406 | `pg_search` | 0.3060 | 0.2100 | 0.9161 | 0.0654 |
| `arguana` | 8,674 | 1,406 | `vchord_bm25` | 0.3597 | 0.2484 | 0.9580 | 0.0749 |
| `scidocs` | 25,657 | 1,000 | upstream `bm25s` | 0.1578 | 0.1077 | 0.3646 | 0.0816 |
| `scidocs` | 25,657 | 1,000 | `ii42 ids` | 0.1578 | 0.1077 | 0.3646 | 0.0816 |
| `scidocs` | 25,657 | 1,000 | `ii42 text[]` | 0.1578 | 0.1077 | 0.3646 | 0.0816 |
| `scidocs` | 25,657 | 1,000 | `pg_search` | 0.1567 | 0.1066 | 0.3607 | 0.0805 |
| `scidocs` | 25,657 | 1,000 | `vchord_bm25` | 0.1561 | 0.1066 | 0.3616 | 0.0802 |
| `fiqa` | 57,638 | 648 | upstream `bm25s` | 0.2514 | 0.2041 | 0.5593 | 0.0699 |
| `fiqa` | 57,638 | 648 | `ii42 ids` | 0.2514 | 0.2041 | 0.5593 | 0.0699 |
| `fiqa` | 57,638 | 648 | `ii42 text[]` | 0.2514 | 0.2041 | 0.5593 | 0.0699 |
| `fiqa` | 57,638 | 648 | `pg_search` | 0.2504 | 0.2034 | 0.5589 | 0.0688 |
| `fiqa` | 57,638 | 648 | `vchord_bm25` | 0.2517 | 0.2036 | 0.5542 | 0.0698 |
| `trec-covid` | 171,332 | 50 | upstream `bm25s` | 0.5988 | 0.3354 | 0.1121 | 0.6500 |
| `trec-covid` | 171,332 | 50 | `ii42 ids` | 0.5994 | 0.3353 | 0.1121 | 0.6500 |
| `trec-covid` | 171,332 | 50 | `ii42 text[]` | 0.5994 | 0.3353 | 0.1121 | 0.6500 |
| `trec-covid` | 171,332 | 50 | `pg_search` | 0.5903 | 0.3240 | 0.1104 | 0.6420 |
| `trec-covid` | 171,332 | 50 | `vchord_bm25` | 0.5895 | 0.3271 | 0.1108 | 0.6460 |
| `webis-touche2020` | 382,545 | 49 | upstream `bm25s` | 0.3259 | 0.2105 | 0.5557 | 0.3041 |
| `webis-touche2020` | 382,545 | 49 | `ii42 ids` | 0.3259 | 0.2106 | 0.5557 | 0.3041 |
| `webis-touche2020` | 382,545 | 49 | `ii42 text[]` | 0.3259 | 0.2106 | 0.5557 | 0.3041 |
| `webis-touche2020` | 382,545 | 49 | `pg_search` | 0.3347 | 0.2120 | 0.5600 | 0.3102 |
| `webis-touche2020` | 382,545 | 49 | `vchord_bm25` | 0.3379 | 0.2140 | 0.5589 | 0.3122 |
| `cqadupstack` | 457,199 | 13,145 | upstream `bm25s` | 0.2994 | 0.2723 | 0.5543 | 0.0488 |
| `cqadupstack` | 457,199 | 13,145 | `ii42 ids` | 0.2994 | 0.2723 | 0.5543 | 0.0488 |
| `cqadupstack` | 457,199 | 13,145 | `ii42 text[]` | 0.2994 | 0.2723 | 0.5543 | 0.0488 |
| `cqadupstack` | 457,199 | 13,145 | `pg_search` | 0.3005 | 0.2735 | 0.5508 | 0.0489 |
| `cqadupstack` | 457,199 | 13,145 | `vchord_bm25` | 0.3007 | 0.2734 | 0.5503 | 0.0490 |
| `quora` | 522,931 | 10,000 | upstream `bm25s` | 0.8045 | 0.7630 | 0.9771 | 0.1216 |
| `quora` | 522,931 | 10,000 | `ii42 ids` | 0.8056 | 0.7643 | 0.9775 | 0.1218 |
| `quora` | 522,931 | 10,000 | `ii42 text[]` | 0.8056 | 0.7643 | 0.9775 | 0.1218 |
| `quora` | 522,931 | 10,000 | `pg_search` | 0.8075 | 0.7662 | 0.9786 | 0.1223 |
| `quora` | 522,931 | 10,000 | `vchord_bm25` | 0.8069 | 0.7658 | 0.9770 | 0.1220 |
| `nq` | 2,681,468 | 3,452 | upstream `bm25s` | 0.2849 | 0.2409 | 0.7430 | 0.0521 |
| `nq` | 2,681,468 | 3,452 | `ii42 ids` | 0.2849 | 0.2408 | 0.7430 | 0.0521 |
| `nq` | 2,681,468 | 3,452 | `ii42 text[]` | 0.2849 | 0.2408 | 0.7430 | 0.0521 |
| `nq` | 2,681,468 | 3,452 | `pg_search` | 0.2933 | 0.2487 | 0.7506 | 0.0532 |
| `nq` | 2,681,468 | 3,452 | `vchord_bm25` | 0.2935 | 0.2491 | 0.7494 | 0.0532 |
| `dbpedia-entity` | 4,635,922 | 400 | upstream `bm25s` | 0.2801 | 0.2113 | 0.4472 | 0.2658 |
| `dbpedia-entity` | 4,635,922 | 400 | `ii42 ids` | 0.2803 | 0.2110 | 0.4472 | 0.2658 |
| `dbpedia-entity` | 4,635,922 | 400 | `ii42 text[]` | 0.2803 | 0.2110 | 0.4472 | 0.2658 |
| `dbpedia-entity` | 4,635,922 | 400 | `pg_search` | 0.2837 | 0.2169 | 0.4538 | 0.2683 |
| `dbpedia-entity` | 4,635,922 | 400 | `vchord_bm25` | 0.2845 | 0.2172 | 0.4536 | 0.2685 |
| `hotpotqa` | 5,233,329 | 7,405 | upstream `bm25s` | 0.5689 | 0.4858 | 0.7586 | 0.1199 |
| `hotpotqa` | 5,233,329 | 7,405 | `ii42 ids` | 0.5689 | 0.4859 | 0.7586 | 0.1199 |
| `hotpotqa` | 5,233,329 | 7,405 | `ii42 text[]` | 0.5689 | 0.4859 | 0.7586 | 0.1199 |
| `hotpotqa` | 5,233,329 | 7,405 | `pg_search` | 0.5927 | 0.5094 | 0.7760 | 0.1242 |
| `hotpotqa` | 5,233,329 | 7,405 | `vchord_bm25` | 0.5884 | 0.5050 | 0.7716 | 0.1234 |
| `fever` | 5,416,568 | 6,666 | upstream `bm25s` | 0.4811 | 0.4284 | 0.8494 | 0.0712 |
| `fever` | 5,416,568 | 6,666 | `ii42 ids` | 0.4811 | 0.4284 | 0.8494 | 0.0712 |
| `fever` | 5,416,568 | 6,666 | `ii42 text[]` | 0.4811 | 0.4284 | 0.8494 | 0.0712 |
| `fever` | 5,416,568 | 6,666 | `pg_search` | 0.5125 | 0.4598 | 0.8633 | 0.0744 |
| `fever` | 5,416,568 | 6,666 | `vchord_bm25` | 0.5121 | 0.4593 | 0.8638 | 0.0744 |
| `climate-fever` | 5,416,593 | 1,535 | upstream `bm25s` | 0.1361 | 0.1020 | 0.3666 | 0.0429 |
| `climate-fever` | 5,416,593 | 1,535 | `ii42 ids` | 0.1361 | 0.1020 | 0.3666 | 0.0429 |
| `climate-fever` | 5,416,593 | 1,535 | `ii42 text[]` | 0.1361 | 0.1020 | 0.3666 | 0.0429 |
| `climate-fever` | 5,416,593 | 1,535 | `pg_search` | 0.1421 | 0.1064 | 0.3867 | 0.0456 |
| `climate-fever` | 5,416,593 | 1,535 | `vchord_bm25` | 0.1403 | 0.1046 | 0.3784 | 0.0449 |
| `msmarco` | 8,841,823 | 43 | upstream `bm25s` | 0.4005 | 0.3212 | 0.4248 | 0.5767 |
| `msmarco` | 8,841,823 | 43 | `ii42 ids` | 0.3996 | 0.3225 | 0.4247 | 0.5767 |
| `msmarco` | 8,841,823 | 43 | `ii42 text[]` | 0.3996 | 0.3225 | 0.4247 | 0.5767 |
| `msmarco` | 8,841,823 | 43 | `pg_search` | 0.4097 | 0.3526 | 0.4506 | 0.5907 |
| `msmarco` | 8,841,823 | 43 | `vchord_bm25` | 0.4093 | 0.3346 | 0.4393 | 0.5837 |

Quality readout:

- All five engines sit in a close relevance band on this BM25 quality
  matrix. Average `NDCG@10` ranges from `0.3976` to `0.4019` across the
  compared engines.
- `ii42 ids` and `ii42 text[]` remain quality-neutral exact
  PostgreSQL paths in this run. Their largest absolute metric difference
  from the Python reference implementation is below `0.0030`, while their
  engineering cost profile is covered by the QPS, build-time, and
  index-size matrices above.
- `pg_search` and `vchord_bm25` are also competitive on relevance in this
  local quality run, but they have different throughput and storage trade-
  offs in the main performance matrix.
- These quality metrics do not replace the QPS, build-time, or index-size
  matrix above. They only show that the compared engines are retrieving
  similarly relevant top-100 candidates under the BEIR qrels.

## Readout

- `ii42 ids` has the strongest suite-median PostgreSQL result in this snapshot. It beats
  upstream on `12/15` datasets and has the highest median ratio.
- `ii42 text[]` is also strong. It beats upstream on `11/15`
  datasets and sits almost level with `ids` on the suite median,
  though it still has the higher total build cost.
- `pg_search` remains much slower on the full suite. It clears upstream
  on only `3/15` datasets and has the weakest median ratio.
- `vchord_bm25` is materially stronger than `pg_search`, but still lags
  behind both `ii42` paths on the suite median.
- The largest workloads remain the clearest signal. On `msmarco`,
  `ii42 ids` reached `96.67` QPS, `ii42 text[]` reached
  `82.13` QPS, `vchord_bm25` reached `18.20` QPS, `pg_search` reached
  `4.44` QPS, and upstream Python `bm25s` reached `1.61` QPS.
- Build performance is also in good shape. Both `ii42` paths are
  substantially cheaper than upstream on total build time, with `ids`
  remaining the clear winner.

## Practical Conclusion

- Use the public `ii42_query(...)` API and choose source types according to
  [Supported Input Types](../input-types.md), not an owner-only diagnostic API
  merely because it labels a benchmark series.
- This historical comparison covers pretokenized `int4[]` and `text[]`, not
  raw-text tokenization or the semantic runtime. Its suite-median throughput ranking
  is token IDs, token streams, `vchord_bm25`, then `pg_search`; individual
  workloads can have a different ordering.
- Qualify a new package with current correctness, lifecycle, and workload
  tests. Keep the frozen matrix as historical evidence rather than a current
  latency or storage guarantee.
- A small retained historical localhost comparison set for the
  `pg_bm25s` study still exists because that comparison was not
  superseded by a newer like-for-like rerun. Those files are historical
  supporting evidence, not the default benchmark reference.
