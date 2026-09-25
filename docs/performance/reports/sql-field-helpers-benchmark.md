# SQL Field Helper Benchmark

Date: 2026-03-26

This benchmark is the focused validation slice for the structured
field-aware SQL helpers added on the `codex/sql-enhancements` branch.

It compares:

- explicit baseline fusion with `evoke_fusion(...)`
- `evoke_fusion_query_weighted(...)`
- `evoke_fusion_query_fields(...)`
- `evoke_fusion_query(field_names[], ...)`
- `evoke_fusion_query(index_names[], ...)`

The benchmark uses:

- 4000 synthetic documents
- two indexed fields:
  - `title_tokens`
  - `body_tokens`
- 4 representative raw queries:
  - `bird`
  - `cat`
  - `bird OR cat`
  - `policy`
- `k = 10`
- `candidate_k = 20`
- 30 repeats per query and per variant

Source data:

- [`../data/diagnostics/sql-field-helpers-2026-03-26.json`](../data/diagnostics/sql-field-helpers-2026-03-26.json)
- [`../../../scripts/benchmark_sql_field_helpers.py`](../../../scripts/benchmark_sql_field_helpers.py)

## Result Equivalence

All helper surfaces matched the explicit baseline exactly for:

- `doc_ids`
- `scores`

Equivalence summary:

- `baseline_fused`: `true`
- `weighted_queries`: `true`
- `field_queries`: `true`
- `search_indexes_named`: `true`
- `search_indexes_short`: `true`

## Latency Summary

Median latency (`p50_ms`):

- `baseline_fused`: `0.369 ms`
- `weighted_queries`: `0.419 ms`
- `field_queries`: `0.608 ms`
- `search_indexes_short`: `0.625 ms`
- `search_indexes_named`: `0.693 ms`

Interpretation:

- the structured helper layers do add wrapper overhead
- the overhead stayed sub-millisecond on this focused slice
- the more explicit field-aware surfaces were slower than direct
  weighted-query fusion, but not by enough to suggest a broken plan or
  a semantic fallback

## Practical Read

This result is strong enough for the branch goal:

- field-aware helpers remain exact
- field names are metadata only
- post-retrieval fusion semantics stayed unchanged
- the new surfaces behave like convenience layers, not hidden alternate
  scoring engines

This does not replace broader query-path benchmarking. It is the narrow
validation slice that protects the field-aware SQL contract itself.
