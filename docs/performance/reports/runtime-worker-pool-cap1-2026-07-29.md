# Runtime Worker Pool CAP-1 Qualification

Date: 2026-07-29

## Decision

Accept the bounded two-worker runtime pool as the product default. It removes
the single-worker serialization point, preserves unloaded latency, provides a
real query lane during document encoding, and isolates an individual worker
failure.

Do not promote four workers as the default for this model and host. Four
workers improved saturated throughput, but missed the predeclared `2x`
single-worker promotion floor and consumed about four times the single-worker
model RSS. No deadline, queue-priority delay, or extension-specific timeout is
used as a substitute for inference capacity.

## Tested System

- Host: Apple Silicon, 24 logical CPUs, 64 GiB memory.
- PostgreSQL: 18.4.
- ONNX Runtime: CPU provider, 1.28.0.
- Checkout:
  `evoke_p2_p22_nfcorpus_v2_smoke`, 382 MiB on disk.
- Runtime response slots: 8.
- Runtime queue slots: 64.
- Requests per client-count phase: 20 rounds.
- Client counts: 1, 8, 16, 32, and 64.
- Query result contract and native ranking parity were covered by the runtime
  and product-path smokes in addition to the throughput harness.

The benchmark includes PostgreSQL client admission and result delivery. At
more than eight clients, excess callers wait at the bounded response-slot
admission boundary rather than allocating unbounded shared result buffers.

## Throughput And Latency

The default automatic CPU-thread policy keeps the ONNX Runtime default for a
single worker, assigns six intra-op threads to each of two workers, and assigns
three to each of four workers on this 24-logical-CPU host.

| Workers | Effective ORT threads | Clients | QPS | Request p95 (ms) | Aggregate worker RSS (KiB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | ORT default | 1 | 115.08 | 8.67 | 1,605,456 |
| 1 | ORT default | 8 | 252.58 | 28.48 | 1,605,456 |
| 1 | ORT default | 16 | 278.80 | 51.64 | 1,605,456 |
| 1 | ORT default | 32 | 286.69 | 101.58 | 1,605,456 |
| 1 | ORT default | 64 | 282.73 | 207.74 | 1,605,456 |
| 2 | 6 | 1 | 118.22 | 7.07 | 3,216,384 |
| 2 | 6 | 8 | 356.43 | 19.02 | 3,216,384 |
| 2 | 6 | 16 | 409.71 | 33.43 | 3,216,384 |
| 2 | 6 | 32 | 418.57 | 67.23 | 3,216,384 |
| 2 | 6 | 64 | 438.21 | 129.20 | 3,216,384 |
| 4 | 3 | 1 | 80.82 | 8.95 | 6,443,600 |
| 4 | 3 | 8 | 337.56 | 18.43 | 6,443,600 |
| 4 | 3 | 16 | 436.17 | 30.16 | 6,443,600 |
| 4 | 3 | 32 | 498.33 | 54.04 | 6,443,600 |
| 4 | 3 | 64 | 547.29 | 98.09 | 6,443,600 |

At 64 clients, two workers delivered `+55.0%` throughput and reduced request
p95 by `37.8%` relative to one worker. Its unloaded p95 also improved from
8.67 ms to 7.07 ms.

Four workers delivered `1.936x` the single-worker saturated throughput and
reduced p95 by `52.8%`. This is useful scale-out, but it does not pass the
predeclared `2x` promotion floor. Its unloaded p95 increased by only `3.3%`,
which passes the latency guard.

Explicit four-worker thread limits did not improve the result:

| Workers | ORT threads | 64-client QPS | 64-client p95 (ms) |
| ---: | ---: | ---: | ---: |
| 4 | 2 | 485.29 | 114.41 |
| 4 | 3, automatic | 547.29 | 98.09 |
| 4 | 4 | 477.33 | 115.81 |

This rejects ORT thread-budget tuning as the missing route to the `2x` floor.
Adaptive request micro-batching was not added: the measured two-worker pool
already removes serialization without introducing an idle-query batching
delay, while implementing cross-request result splitting would enlarge the
runtime ABI and fault surface. It can be reconsidered only with a separate,
predeclared throughput target.

## Correctness And Availability

The following current-tree gates passed:

- two long query batches were observed executing on two workers at once;
- a cold worker stole a second same-model request while the affinity worker
  retained one request, so model affinity does not serialize a backlog;
- alternating two checkout aliases caused bounded session loads rather than
  unbounded LRU reloads;
- with two concurrent document batches, at most one worker encoded documents
  and the other remained available for an online query;
- backend cancellation released only the caller response ownership and did
  not stop the pool;
- terminating the worker that owned an in-flight document request failed that
  request, while the surviving worker returned identical product query rows;
- postmaster restarted the failed slot and restored a two-worker healthy pool;
- the 120-round, two-worker soak completed every request, retained both
  workers, grew by 4,288 KiB during warmup, and had a final tail slope of
  0.29 KiB per iteration.

## Capacity Contract

`evoke.runtime_worker_count` and
`evoke.onnxruntime_session_cache_size` are restart-only bounds. Each worker owns
its own bounded session cache, so model memory scales approximately with:

```text
runtime workers x cached sessions per worker x measured session footprint
```

There is no portable pre-load estimator for ONNX Runtime's expanded session
RSS. The product therefore does not claim a false automatic byte budget.
Operators must qualify the chosen worker and session-cache counts with the
actual checkout and provider. The measured default here is about 3.1 GiB for
two workers; four workers are about 6.1 GiB.

## Raw Evidence

- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers1.json`
- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers2.json`
- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers4.json`
- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers4-t2.json`
- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers4-t4.json`
- `docs/performance/data/raw/runtime-worker-pool-2026-07-29/evoke-cap1-workers2-soak120.json`

These artifacts qualify the current working tree. Release evidence must be
rerun and bound to the final commit and staged package fingerprint.
