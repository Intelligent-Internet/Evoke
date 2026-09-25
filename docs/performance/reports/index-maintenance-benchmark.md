# Index Maintenance Benchmark

Date: 2026-03-23

> Historical benchmark. Its original follow-up scope was closed by the
> 2026-07-27 unified-lifecycle qualification in
> [Eventual SAE Release Readiness History](../../archive/engineering/eventual-sae-release-readiness-history.md).
> The section below records what remained at the time; it is not the current
> product planning queue.

## Scope

This benchmark compares the maintenance variants of the same
`evoke` build on the same local PostgreSQL instance:

- default realtime maintenance: `consistency = 'realtime'`
- manual refresh mode: `consistency = 'manual'`
- threshold-deferred automatic maintenance:
  `auto_rebuild_threshold > 0`

The benchmark setup was:

- 5,000 initial documents
- 100 inserted documents
- 100 updated documents
- 100 deleted documents
- 200 measured queries
- `top_k = 20`
- `int4[]` / `evoke_query_ids(...)`

Raw output is stored in
`auto-maintenance-2026-03-23.json`.

## Results

| Mode | Phase | Avg ms | P50 ms | P95 ms | QPS |
| --- | --- | ---: | ---: | ---: | ---: |
| Manual | Before writes | 0.0714 | 0.0348 | 0.2264 | 14011.21 |
| Auto | Before writes | 0.0427 | 0.0375 | 0.0684 | 23407.72 |
| Manual | After insert + refresh | 0.0563 | 0.0425 | 0.1150 | 17751.97 |
| Auto | After insert | 0.0506 | 0.0432 | 0.0732 | 19774.11 |
| Threshold auto | After insert | 0.0650 | 0.0384 | 0.1431 | 15388.39 |
| Manual | After update + refresh | 0.0423 | 0.0307 | 0.0913 | 23652.94 |
| Auto | After update | 0.0395 | 0.0360 | 0.0569 | 25286.46 |
| Threshold auto | After update | 0.0433 | 0.0400 | 0.0635 | 23083.06 |
| Manual | After delete, before vacuum | 0.0415 | 0.0379 | 0.0641 | 24097.35 |
| Auto | After delete, before vacuum | 0.0433 | 0.0386 | 0.0673 | 23102.82 |
| Manual | After delete + refresh | 0.0458 | 0.0382 | 0.0623 | 21834.48 |
| Auto | After delete + vacuum | 0.0469 | 0.0401 | 0.0706 | 21315.26 |
| Threshold auto | After delete + vacuum | 0.0601 | 0.0382 | 0.1440 | 16641.26 |

Write-side timing:

- manual insert batch + commit: `2.34 ms`
- automatic insert batch + commit: `14.44 ms`
- thresholded insert batch + commit: `2.26 ms`
- manual update batch + commit: `7.88 ms`
- automatic update batch + commit: `12.81 ms`
- thresholded update batch + commit: `2.55 ms`
- automatic insert + update in one top-level transaction: `18.15 ms`
- manual delete batch + commit: `0.76 ms`
- automatic delete batch + commit: `0.96 ms`
- thresholded delete batch + commit: `0.83 ms`
- manual vacuum after deletes: `4.88 ms`
- automatic vacuum after deletes: `23.89 ms`
- thresholded vacuum after deletes: `5.87 ms`

The new maintenance transaction metric is important: batching an
insert and an update into one transaction now costs less than two
separate realtime-maintenance write batches, because the branch rebuilds once
at commit instead of once per statement.

Threshold scheduling result:

- insert-only deferred maintenance now stays deferred across reads:
  - maintenance state after insert commit:
    `rebuilds=1, pending_writes=100, pending_deletes=0, delta_records=100, delta_bytes=12000, stale=false`
  - first canonical query after that deferred insert commit:
    `2.79 ms`
  - maintenance state after the query is unchanged:
    `rebuilds=1, pending_writes=100, pending_deletes=0, delta_records=100, delta_bytes=12000, stale=false`
- update-heavy deferred maintenance now uses the same bounded overlay
  strategy:
  - maintenance state after update commit:
    `rebuilds=1, pending_writes=98, pending_deletes=0, delta_records=98, delta_bytes=11760, stale=false`
  - first canonical query after that deferred update commit:
    `2.70 ms`
  - maintenance state after the query is unchanged:
    `rebuilds=1, pending_writes=98, pending_deletes=0, delta_records=98, delta_bytes=11760, stale=false`
- delete-heavy deferred maintenance now uses exact vacuum-driven
  tombstones:
  - maintenance state after delete + vacuum:
    `rebuilds=1, pending_writes=0, pending_deletes=95, delta_records=95, delta_bytes=380, stale=false`

The current design now also exposes a second consolidation control:

- `auto_rebuild_delta_bytes`

This reloption forces commit-time consolidation once the persisted delta
payload crosses a byte threshold, even if the tuple-count threshold has
not yet been reached. Regression coverage now includes a case where one
deferred insert stays below the tuple-count threshold but exceeds the
delta-byte threshold and therefore rebuilds at commit.

The current design now also exposes a third consolidation control:

- `auto_rebuild_churn_ratio`

This reloption forces commit-time consolidation once pending
write/delete activity reaches a configured fraction of the current
indexed corpus. It is mainly intended for smaller or high-churn tables
where tuple-count and byte-size thresholds alone can defer too long.
Regression coverage now includes a case where one deferred insert on a
small table remains below the tuple-count and byte-size thresholds but
still consolidates because the churn ratio crosses the configured limit.
  - first canonical query after that deferred delete maintenance:
    `3.03 ms`
  - maintenance state after the query is unchanged:
    `rebuilds=1, pending_writes=0, pending_deletes=95, delta_records=95, delta_bytes=380, stale=false`

This is now the intended threshold tradeoff: inserts, updates, and
vacuum-driven deletes all use the new bounded persisted overlay and
avoid first-read rebuilds, while keeping the canonical query path exact.

Behavioral difference:

- manual mode query after insert: `evoke index is stale`
- manual mode query after update: `evoke index is stale`
- manual mode query after delete + vacuum: `evoke index is stale`
- automatic mode query after insert: succeeds immediately
- automatic mode query after update: succeeds immediately
- automatic mode query after delete + vacuum: succeeds immediately

## Interpretation

The important result is that the query path stayed in the same class of
latency and throughput after adding automatic maintenance. The hot
retrieval path still uses the same cached deserialized index state, and
the new insert-only threshold path now keeps exact reads cheap too.

The cost moved to writes and vacuum, which is intentional for this
maintenance-first slice. Realtime maintenance still performs exact full
rebuilds for the conservative paths, but the current design now has two
important improvements on top of that:

- batched `INSERT` / `UPDATE` rebuild scheduling across one top-level
  transaction
- persisted insert/delete overlays that build exact in-memory overlays
  instead of forcing first-read rebuilds

That is still slower than the old stale-then-refresh mode for writes,
but it preserves:

- exact query semantics after `INSERT` / `UPDATE`
- exact query semantics for write-then-read transactions through
  flush-on-read
- exact canonical query semantics for bounded deferred pure-insert
  states without first-read rebuilds
- exact canonical query semantics for bounded deferred delete states
  after vacuum-driven tombstone materialization
- exact post-delete corpus statistics after `VACUUM`
- normal PostgreSQL index storage and WAL behavior
- backend-local cache correctness through persisted cache epochs

The main remaining limitation is no longer basic update handling.
Instead, it is long-run churn behavior: how overlay size, repeated
updates, and repeated vacuum cycles should consolidate over time without
giving back the current read-path advantage.

## Restart Smoke

The main repository also includes a dedicated restart smoke script.

It validates that threshold-deferred insert, update, and delete states
survive a PostgreSQL restart and that the first canonical query after
restart returns exact results through the persisted overlay path.

## Crash And Replication Smoke

The main repository also includes dedicated crash-recovery and physical
replication smoke scripts.

These validate that the same deferred insert/update/delete states
survive:

- `pg_ctl stop -m immediate` crash recovery for deferred inserts and
  deferred updates, plus checkpointed immediate-crash validation for
  vacuum-materialized tombstones
- local physical replication to a hot standby

The replication smoke also verifies that a standby can answer canonical
BM25 queries exactly from replicated deferred insert, update, and delete
states without forcing a rebuild.

## Historical Follow-Up Scope

At the time of this benchmark, the unresolved gap was no longer basic
durability for the insert/update/delete deferred slice. The proposed follow-up
work was broader:

- richer long-run write/read benchmarking under sustained churn
- stronger consolidation policies once churn or deferred overlay size
  grows beyond the current threshold model
