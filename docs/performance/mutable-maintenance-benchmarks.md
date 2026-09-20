# Mutable Maintenance Benchmarks

This page summarizes the **2026-03-23 lexical maintenance checkpoint**. Its
measurements are historical, not a fresh Beta 1 benchmark or a Sparse Semantic Retrieval
(SSR) performance claim. Current lifecycle behavior is specified in
[Maintenance Lifecycle](../maintenance-lifecycle.md).

## Scope

Representative maintenance checkpoint:

- 5,000 initial documents
- 100 inserts
- 100 updates
- 100 deletes
- 200 measured exact BM25 queries
- measured owner-diagnostic query path (not the current application API):
  `ii42_query_ids(...)`

Primary source report:

- [Index maintenance benchmark](reports/index-maintenance-benchmark.md).

## Summary Table

| Slice | Result |
| --- | --- |
| Automatic read path before writes | `23.4k QPS` |
| Deferred insert commit | `2.26 ms` |
| First exact read after deferred insert | `2.79 ms` |
| Deferred update commit | `2.55 ms` |
| First exact read after deferred update | `2.70 ms` |
| Deferred delete commit | `0.83 ms` |
| Delete vacuum | `5.87 ms` |
| First exact read after deferred delete | `3.03 ms` |

## Interpretation

At this checkpoint, the tested maintenance design demonstrated:

- exact reads remain available
- write cost is far lower than eager rebuild-only maintenance
- deferred state stays bounded and introspectable
- `DELETE` remains visibility-correct immediately and becomes
  corpus-statistics-exact through normal `VACUUM`

This is not a claim that every workload is universally faster. It is a
claim that the tested design offered a practical exact maintenance
path without forcing full rebuilds on every write.

## Related Stability Coverage

Current release qualification separately covers:

- restart
- crash recovery
- physical replication
- concurrent writer stress

See:

- [Testing and Validation](../testing-and-validation.md)
- [Maintenance Policy Tuning](maintenance-policy-tuning.md)

Supporting archived material is stored in:

- [`reports/index-maintenance-benchmark.md`](reports/index-maintenance-benchmark.md)
- [`data/diagnostics/auto-maintenance-2026-03-23.json`](data/diagnostics/auto-maintenance-2026-03-23.json)
