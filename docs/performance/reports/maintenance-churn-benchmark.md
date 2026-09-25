# Maintenance Churn Benchmark

Date: 2026-03-24

> Historical benchmark. The later unified-lifecycle qualification closed its
> correctness and stability follow-ups. Threshold selection remains an
> operator workload policy, not an unresolved engine defect.

## Scope

This benchmark measures sustained mixed write churn on the same local
PostgreSQL instance and the same `evoke` build.

Workload:

- 5,000 initial documents
- 6 churn cycles
- each cycle:
  - 50 inserts
  - 50 updates
  - 50 deletes
  - one `VACUUM`
- 100 measured BM25 queries after each cycle
- `int4[]` / `evoke_query_ids(...)`

Compared maintenance modes:

- `docs_auto_eager`
  - exact eager rebuilds at commit
- `docs_auto_threshold_count`
  - `auto_rebuild_threshold = 1000`
- `docs_auto_threshold_tuned`
  - `auto_rebuild_threshold = 1000`
  - `auto_rebuild_delta_bytes = 50000`
  - `auto_rebuild_churn_ratio = 0.05`

Raw data is stored in the corresponding diagnostics JSON.
The benchmark now records the effective index maintenance policy through
`evoke_index_details(regclass)` for each tested mode.

## Summary

| Mode | Effective policy | Median QPS | Min QPS | Max QPS | Avg commit ms | Avg vacuum ms | Final rebuilds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager auto | threshold=`0`, delta-bytes=`0`, churn=`0` | 17156.33 | 7322.03 | 19661.32 | 12.18 | 23.44 | 13 |
| Threshold count | threshold=`1000`, delta-bytes=`0`, churn=`0` | 8041.34 | 7346.30 | 13200.30 | 2.62 | 7.89 | 2 |
| Threshold tuned | threshold=`1000`, delta-bytes=`50000`, churn=`0.05` | 19009.93 | 6778.24 | 26030.39 | 8.49 | 4.92 | 4 |

## Interpretation

The current tuned deferred policy is the strongest balance in this run.

- `docs_auto_threshold_tuned` kept the highest median query throughput.
- It rebuilt more often than the count-only policy, but far less often
  than eager maintenance.
- Its average vacuum cost was much lower than eager maintenance, and its
  steady-state QPS stayed closer to the best cycles.
- Its policy was also the cleanest practical balance in this run:
  the churn-ratio guard kept deferred state from persisting all the way
  to the end of the benchmark.

The count-only deferred policy is cheaper on writes, but lets the
overlay grow too long.

- `docs_auto_threshold_count` had the lowest average commit cost.
- But it allowed pending state to accumulate through the end:
  - final state:
    `pending_writes=100, pending_deletes=100, delta_records=200, delta_bytes=12400`
- That reduced median QPS relative to the tuned mode.

Eager maintenance still gives the cleanest state, but pays for it.

- `docs_auto_eager` rebuilt on every cycle and after every vacuum path.
- It kept exactness simple, but average commit and vacuum times were the
  highest.
- Its read throughput was also the least stable in this run.

## Main Finding

The new consolidation controls are doing useful work.

- `auto_rebuild_delta_bytes` and `auto_rebuild_churn_ratio` help prevent
  long-lived deferred state from growing until it drags the query path
  down.
- The tuned mode still consolidated infrequently, but it did not leave
  a deferred overlay behind at the end of the run.
- That was a better balance than either rebuilding constantly or
  allowing the cheaper count-only policy to stay deferred too long.

That is the current practical takeaway for the current maintenance
design:

- the bounded deferred overlay path is good enough for mixed churn
- consolidation policy matters materially
- a small number of well-chosen rebuilds is better than either extreme

## Next Step

The next useful benchmark work is no longer another hand-edited one-off
run. The harness is now parameterized, so the next step is a broader
matrix:

- larger corpora
- different insert/update/delete ratios
- different threshold and churn-ratio settings
- longer cycle counts

That should tell us whether the current tuned policy is generally good,
or just good for this local workload.

## Heavier Checkpoint

A second diagnostic run used the same harness with a heavier local
workload:

- 10,000 initial documents
- 8 churn cycles
- each cycle:
  - 100 inserts
  - 100 updates
  - 100 deletes
  - one `VACUUM`
- 150 measured BM25 queries after each cycle

Raw data is stored in the corresponding heavier diagnostics JSON.

| Mode | Effective policy | Median QPS | Min QPS | Max QPS | Avg commit ms | Avg vacuum ms | Final rebuilds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager auto | threshold=`0`, delta-bytes=`0`, churn=`0` | 5075.41 | 2435.50 | 15180.34 | 34.18 | 49.39 | 17 |
| Threshold count | threshold=`1000`, delta-bytes=`0`, churn=`0` | 9300.07 | 3709.95 | 13904.29 | 22.51 | 20.67 | 3 |
| Threshold tuned | threshold=`1000`, delta-bytes=`50000`, churn=`0.05` | 9060.96 | 2228.93 | 19045.33 | 55.84 | 19.54 | 5 |

This heavier run is useful because it does not fully agree with the
smaller benchmark.

- The tuned policy still kept exact state cleaner and ended with no
  deferred overlay.
- But the count-only policy slightly beat it on median QPS in this
  heavier local workload.
- The tuned policy paid more rebuild cost here, which raised commit
  latency materially.

So the current conclusion is narrower than before:

- the design is working
- deferred exact overlays plus bounded consolidation are viable
- but the current tuned thresholds are not yet proven globally optimal
- policy tuning remains workload-dependent and needs a broader matrix
