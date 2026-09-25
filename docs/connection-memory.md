# Connection Memory And Prewarming

Evoke separates durable index state, shared execution state, and query-local
scratch:

```text
relation-owned page-native postings
    + PostgreSQL shared buffers / OS page cache
    + bounded postmaster-owned runtime and residency
    + bounded backend query scratch
```

This model applies to BM25 and semantic-enabled indexes. Sparse Semantic Retrieval
(SSR) additionally requires the shared runtime for text encoding; application
backends never own tokenizers or model sessions.

## Ownership

| State | Owner | Lifetime |
| --- | --- | --- |
| Checked root, document COW, postings, linked L0 | Index relation | WAL-durable |
| Relation-page cache | PostgreSQL shared buffers and OS | Disposable |
| Runtime queues, root markers, HOT_FOLD, exact-root resident fold | Shared preload arena | Postmaster |
| Tokenizers and ONNX sessions | Runtime workers | Worker/session LRU |
| Ranking workspace and matching-L0 projection | Query backend | Query or bounded idle cache |
| No-shared-runtime BM25 snapshot | Query backend | Optional, bounded by workspace policy |
| Explicit BM25 `weight_mask` snapshot | Query backend | Optional small-index compatibility path, bounded by workspace policy |

No semantic posting image or model session is retained per connection. A
selected, fully converged BM25 or semantic index may have one pointer-free
exact-root fold in the postmaster arena. Backends lease that shared image and
retain only query-local projections. When the postmaster shared runtime is
available, BM25 and SSR use this same dispatcher and no backend-local index
snapshot is admitted for ordinary queries. Only a pure BM25 deployment without
shared runtime may use the bounded fallback when its physical size fits the
workspace budget. The explicit BM25-only `weight_mask` API is the sole
shared-runtime exception because its input already spans the complete document
slot space; it is rejected unless the same finite physical-size bound admits
the snapshot.
RSS can count shared mappings in every backend, so use proportional set size
and private dirty/anonymous memory when diagnosing connection growth.

## Query Workspace

| Setting | Default | Meaning |
| --- | ---: | --- |
| `ii42.workspace_cache_bytes` | `32MB` | Maximum completed-query workspace retained by one backend and admission bound for the no-shared-runtime BM25 fallback or explicit BM25 `weight_mask` snapshot. |
| `ii42.workspace_idle_timeout` | `60s` | Idle interval before retained workspace is released on the backend's next Evoke use. |

`workspace_cache_bytes = 0` releases workspace after every query and disables
both snapshot paths. `-1` removes the general workspace-retention cap but does
not admit a backend-local index snapshot because snapshot admission must have a
finite positive bound. It is intended only for controlled benchmarks or a
small fixed pool.
`workspace_idle_timeout = -1` disables idle expiry.

Backend snapshot entries are leased only while a fallback query is active.
An idle entry is reused across indexes, so a long-lived connection does not
retain one complete snapshot for every small index it has queried. The entry
objects themselves have stable addresses. Simultaneously leased snapshots are
charged together against `workspace_cache_bytes` and have a separate 64-entry
hard limit. Once a concurrent high-water mark drains, excess idle entries drop
their snapshots; a backend retains at most one idle fallback snapshot.

An active query may temporarily exceed the retention budget; excess scratch is
released when the query ends. Same-transaction reads project only matching
linked-L0 records under the current snapshot and do not construct a private
index copy.

## Shared Semantic Runtime

The relevant postmaster settings are:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `ii42.shared_runtime_size` | `0` | Shared runtime/residency arena; must be positive for SSR. The arena is not posting authority. |
| `ii42.runtime_worker_count` | `2` | Parallel inference workers, from 1 through 16. |
| `ii42.runtime_reserve_query_lane` | `on` | Reserve one runtime lane for foreground queries; disable only for controlled offline rebuilds. |
| `ii42.runtime_max_batch_size` | `128` | Local runtime text batch limit and default remote service batch cap. This is a deployment throughput knob, not model identity. |
| `ii42.runtime_document_pipeline_depth` | `16` | Builder-level active document runtime batches per backend, from 1 through 4096. Effective depth is capped by the sum of the local runtime outstanding window and configured remote accelerator `weight` windows. Completed batches may retire out of order into a bounded reorder buffer before the builder applies them to the index in original sequence order. |
| `ii42.runtime_accelerators` | `[]` | JSON array of optional remote Evoke runtime services. Service objects accept optional positive integer `weight` as a per-service outstanding request window hint from 1 through 4096 and optional positive integer `max_batch_size` as that service's request cap; a service without `weight` starts as one schedulable slot, and shared accelerator metrics are capped at 64 services. |
| `ii42.runtime_liveness_timeout` | `5min` | Per-run liveness guard for local runtime execution and already-submitted remote accelerator requests; `0` disables it. Remote connect/send I/O still uses short transport timeouts. |
| `ii42.control_database` | `postgres` | Optional override for the stable, connectable database used for runtime initialization and cluster maintenance discovery. |
| `ii42.onnxruntime_session_cache_size` | `1` | Sessions retained per worker; `0` releases after each request. |
| `ii42.onnxruntime_intra_op_threads` | `0` | Automatic per-worker CPU allocation; positive values set an explicit cap. |
| `ii42.onnxruntime_document_cpu_mem_arena` | `off` | Retain CPU provider document-mode arenas for higher offline rebuild throughput on memory-rich hosts. |
| `ii42.sae_transaction_mutation_max_bytes` | `64MB` | Per-transaction retained source-text budget before lexical L0 serialization. |

With multiple runtime workers, automatic CPU allocation prevents nested ONNX
thread pools from consuming every PostgreSQL CPU. Each worker has an
independent bounded session LRU, so the runtime memory term is approximately:

```text
runtime_worker_count * measured session RSS per worker
```

Measure this with the production checkout and provider. Model file size is not
a reliable expanded-session estimate.

## Build And Maintenance Memory

Semantic `CREATE INDEX`, `REINDEX`, and background completion send bounded
document batches to the shared worker pool. Application query encoding remains
single-text and immediate.

A full builder uses `ii42.runtime_document_pipeline_depth` (default `16`),
capped by the available local and remote outstanding-request windows. Completed
batches can release runtime handles out of order into a bounded reorder buffer;
the builder applies results in source-row order. A slow earlier batch can still
cause backpressure when that buffer fills. With query-lane reservation enabled
and at least two healthy local workers, document work leaves capacity for query
inference. One-worker mode serializes document and query inference.

Large builds do not materialize all postings in backend memory:

- document authority stays proportional to document count;
- decoder rowsets are released after each batch;
- posting streams and compact payloads use PostgreSQL temporary files;
- `tuplesort` and `work_mem` bound in-memory ordering;
- completed pages are copied into the relation before one checked root switch.

Capacity-plan `temp_tablespaces` for the posting streams and external-sort
runs. Automatic maintenance respects
`ii42.maintenance_rebuild_memory_budget`; an over-budget optional action keeps
the readable root and reports a blocker rather than forcing swap.

## Prewarming

A converged index selected by `auto_preload > 0`, or explicitly preloaded, uses
an exact-root resident fold when its source relation fits the global
`ii42.shared_runtime_size` arena, the priority-aware arena admission policy,
and host materialization headroom. `ii42.prewarm_max_bytes` does not cap shared
residency; it bounds only relation-page warming work. BM25 and SSR use the same
format and scorer. Indexes not admitted to the shared arena record a
checked-root marker and warm relation pages through PostgreSQL shared buffers.
They use a bounded roots-and-payload pass so startup cannot cycle the entire
shared-buffer cache.
It may also warm an exact term-local HOT_FOLD.

```sql
SELECT ii42_index_preload(
    'docs_search_idx'::regclass
);
```

The exact resident fold is disposable and exact-root keyed. A mutation
invalidates that exact projection and returns affected queries to page-native
execution until the same preload worker publishes its replacement. This rule
does not invalidate a compatible durable semantic accelerator: that artifact
remains an older baseline and queries revalidate its bounded candidate set.
Small post-baseline additions may wait for the next debt-driven refresh.
Run warmup after a restart or major rebuild before opening a large application
pool.

BM25 can operate without shared preload. Every semantic-enabled index requires:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '1GB'
```

`64MB` remains sufficient for minimal runtime smoke tests. GB-scale values are
supported for explicit resident-index policy; the configured arena is reserved
at postmaster start. Size it from measured `resident_fold_bytes`, runtime state,
and safety headroom. Durable postings remain in relation pages.

## Capacity Formula

Use this planning model:

```text
required RAM ~= PostgreSQL baseline
             + shared_buffers and OS page cache
             + shared runtime/residency arena
             + worker count * measured session RSS
             + connection count * retained workspace budget
             + operating-system safety margin
```

For large connection pools, reduce `workspace_cache_bytes` before reducing the
shared runtime required by SSR. Increase worker count only after measuring both
throughput and session RSS.

## Diagnostics

```sql
SELECT ii42_index_status('docs_search_idx'::regclass);

SELECT ii42_index_runtime_state(
    'docs_search_idx'::regclass
);

SELECT ii42_index_runtime_state_json(
    'docs_search_idx'::regclass
);
```

The text and JSON diagnostics report current runtime, root-marker, workspace,
and residency state from one C snapshot collector. They do not describe a
second index lifecycle.

For controlled cold-start testing, a privileged operator may call:

```sql
SELECT ii42_runtime_cache_clear();
```

This clears disposable workspace, markers, and derived residency only. It does
not change the checked root, postings, linked L0, model contract, or results.

See [Shared Runtime And Residency](shared-runtime-and-residency.md),
[Semantic Runtime](examples/semantic-runtime.md), and
[Semantic Index Operations](examples/semantic-index-operations.md).
