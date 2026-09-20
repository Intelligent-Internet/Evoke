# Maintenance Policy Tuning

This is current operational guidance. The archived sweeps linked below record
older scheduler experiments; their threshold presets are not current
reloptions or supported recommendation profiles.

## Current Recommendations

`ii42_index_policy_recommend(index, profile)` is advisory. It returns options;
it does not alter an index or tune the scheduler automatically. The implementation
is [`ii42_am_get_policy_recommendation`](../../src/ii42_am_options.c).

| Index mode | Accepted profile | Recommended consistency |
| --- | --- | --- |
| BM25 | omitted, `balanced`, `query_first` | `realtime` |
| BM25 | `write_tolerant_query_first`, `write_first` | `eventual` |
| Sparse Semantic Retrieval (SSR, `sae = true`) | omitted or any of the above names | `eventual`, reported as `balanced` |

The old `small_mixed_churn`, `heavy_mixed_churn`, `heavy_insert_skew`, and
`longrun_mixed_churn` labels identify experiments, not current SQL profiles.
See [Index Policy](../index-policy.md) for visibility and
[Index Parameters](../index-parameters.md) for supported options.

## Scheduling And Serving

Normal defaults are one maintenance worker, a `60s` maintenance discovery
interval, and a `1h` low-debt maintenance interval. The separate preload probe
interval is `1s`. These are scheduling controls, not query deadlines or serving
artifact TTLs. Low-volume debt can converge periodically without crossing an
emergency threshold; hard pressure and missing artifacts remain actionable.
Compatible serving baselines have no maximum age and remain available while
replacement work proceeds. Do not trigger `REINDEX` simply because delta exists.

Set `ii42.maintenance_worker_limit` and memory budgets from measured CPU,
memory, and I/O headroom. Increasing background parallelism can hurt foreground
latency even when readers do not wait for a build lock. Compare warm-query
latency, root/accelerator identities, completed work, bytes rewritten, and
private/shared memory across read-only and mixed-read/write runs.

## Worker Isolation

Workers discover candidates under short catalog snapshots, reserve maintenance
ownership, and release discovery state before long work. Structural maintenance
reuses persisted postings; it does not re-encode unchanged source documents.
Semantic completion uses bounded document batches. Accelerator scope capture
reads included heap values under short MVCC snapshots of at most 256 root
documents, including safe TOAST access.

Long accelerator preparation is separate from checked publication. It does not
hold a query-blocking relation fence for the entire build. Short publication
and reader-safe page-reuse fences still exist, and a single admitted action can
take longer than the scheduler's between-action time budget. For the precise
lock, snapshot, retry, and reclamation boundaries, use
[Maintenance Lifecycle](../maintenance-lifecycle.md).

## Historical Evidence

These experiments informed earlier policy choices. Preserve their dates,
source versions, workloads, and limitations rather than applying their presets
to the current engine:

- [Policy sweep](reports/maintenance-policy-sweep.md)
- [Policy matrix](reports/maintenance-policy-matrix.md)
- [Repeatability](reports/maintenance-policy-repeatability.md)
- [Long-run benchmark](reports/maintenance-longrun-benchmark.md)
- [Production-shaped traces](reports/maintenance-trace-profiles.md)
- [Raw diagnostics](data/diagnostics/)
