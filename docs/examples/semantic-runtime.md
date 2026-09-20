# Semantic Runtime

The shared runtime converts text into sparse atoms for Sparse Semantic Retrieval
(SSR, `sae = true`) indexes.
It is an execution service for one relation-owned index lifecycle, not a model
database or posting authority.

SSR is eventual-only: foreground writes publish lexical state and pending
document identity; shared workers perform bounded semantic completion.

## Ownership

- `shared_preload_libraries = 'ii42'` starts the supervisor and runtime pool.
- A positive `ii42.shared_runtime_size` provides bounded queues,
  root markers, and optional residency.
- Runtime workers alone own tokenizers and ONNX sessions.
- Application backends submit bounded requests and retain no model session.
- Durable document atoms remain in relation pages and linked L0.

## Scheduling

Each worker runs one inference request at a time. Workers execute in parallel.
Application query requests are single-text; build and maintenance use bounded
document batches.

Document work has bounded queue occupancy and backpressure. With
`ii42.runtime_reserve_query_lane = on` and at least two healthy workers,
the scheduler preserves query admission/execution capacity
rather than allowing one build to occupy every lane. One-worker mode is
supported but necessarily serializes query and document inference.

Cancellation requests cooperative ONNX termination. The liveness timeout is a
worker guard, not a SQL deadline. A provider that fails to return is isolated by
recycling its worker while other workers remain available.

## Model Sessions

Each worker owns an independent bounded session LRU. Session identity includes
the canonical checkout signature and worker identity, so artifact replacement
or operating-system PID reuse cannot expose an incompatible session.

Every request revalidates the checkout around inference. Validation metadata
uses a separate bounded cache, but each hit still restats bound artifacts.

`ii42.onnxruntime_session_cache_size = 0` releases a session after each request.
Positive values retain at most the configured number per worker. Capacity-test
the product of worker count and model-session RSS.

## CPU Threading

With multiple workers and `ii42.onnxruntime_intra_op_threads = 0`, II-42
reserves CPU capacity for PostgreSQL and divides an inference budget across
workers. A positive value explicitly caps each worker's ONNX intra-op pool.

This prevents nested model thread pools from erasing request-level parallelism.

## Minimal Configuration

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
ii42.runtime_worker_count = 2
ii42.runtime_liveness_timeout = '5min'
ii42.onnxruntime_session_cache_size = 1
ii42.onnxruntime_intra_op_threads = 0
```

The official package installs the milestone checkout in PostgreSQL's shared
data directory. `ii42.sae_model_path` remains an optional server-wide override;
`model_path` on an index has higher precedence.

Workers use the standard `postgres` database for catalog-safe runtime
initialization and application-database discovery; they do not pin every
application database. Set `ii42.control_database` only when `postgres` is
unavailable or a different stable, connectable database is preferred.

`ii42.runtime_liveness_timeout` also bounds an already-submitted remote
accelerator request. Remote connection and send I/O keep short transport
timeouts so an unreachable accelerator can be skipped without waiting for the
full liveness window.

## Contract Handoff

Query encoding returns the runtime-contract signature used for inference.
`ii42_query(...)` opens only a root with the same signature. A runtime-contract
change between encoding and scoring fails closed and requires a new query.
Ordinary publication of a compatible root is not itself a contract mismatch;
a checked compatible serving baseline may remain usable across such changes.
Query atoms from one model contract are never scored against another.

## Product SQL

```sql
CREATE INDEX docs_search_idx
ON docs USING ii42 (body)
WITH (sae = true);

SELECT ii42_index_status('docs_search_idx'::regclass);
SELECT * FROM ii42_query('docs_search_idx'::regclass, 'query', 20);
DROP INDEX docs_search_idx;
```

## Failure Rules

Missing artifacts, digest mismatch, unsupported ABI, invalid tensor output,
empty atom output, unavailable worker, cancellation, or liveness expiry return
a clear error or readiness blocker. Shared-memory ABI mismatch requires a clean
postmaster restart. None creates a backend-local fallback or alternate scorer.

See [Shared Runtime And Residency](../shared-runtime-and-residency.md).
