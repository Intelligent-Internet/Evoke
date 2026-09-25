# Semantic Index Operations

An `sae = true` index is one page-native Evoke relation. Lexical and semantic
atoms share one checked root, linked L0, scorer, mutation lifecycle, and
maintenance authority.

## Product Surface

| Operation | Interface |
| --- | --- |
| Build | `CREATE INDEX ... USING evoke ... WITH (sae = true)` |
| Query | `evoke_query(...)` |
| Inspect | `evoke_index_options(...)`, `evoke_index_status(...)` |
| Maintain | `evoke_index_maintain(...)` |
| Schedule | `evoke_index_maintain_due(...)` |
| Rebuild | PostgreSQL `REINDEX` |
| Remove | PostgreSQL `DROP INDEX` |

Applications do not create, publish, or query semantic state separately.
Per-index control functions enforce PostgreSQL ownership. The global scheduler
is revoked from `PUBLIC` and belongs only to a trusted maintenance role.

## Build

Configure the shared runtime before creating the index. Release packages use
their digest-locked milestone checkout unless the index or server selects a
different qualified checkout:

```sql
CREATE INDEX docs_search_idx
ON docs USING evoke (body)
WITH (
    sae = true,
    consistency = eventual,
    auto_preload = 1
);
```

Sparse Semantic Retrieval (SSR) is eventual-only. Omitting `consistency`
selects `eventual`; explicit `realtime` and `manual` are rejected. BM25-only
indexes retain their supported consistency policies.

Use ordinary `CREATE INDEX` for an offline build. Use PostgreSQL's standard
concurrent form outside an explicit transaction when the table must remain
writable:

```sql
CREATE INDEX CONCURRENTLY docs_search_idx
ON docs USING evoke (body)
WITH (sae = true);
```

Monitor long builds with `pg_stat_progress_create_index`. The initial build
uses PostgreSQL's heap-build visibility protocol and publishes one checked
root. Heap scan completion is not publication completion. It does not create
an application-visible model object or second index.

## Progressive Consistency

Foreground insert and update perform no document-model inference. They append
exact lexical postings plus semantic-pending document identity to the linked
L0. The committed row is lexical-searchable immediately through the exact
route. A compatible bounded accelerator may temporarily omit that post-baseline
row. Shared workers later encode a bounded pending batch and append semantic
atoms for the same validated document version.

The exact query route reads the checked root, immutable extents or folds, and
the snapshot-visible linked-L0 projection through one scorer. The default
accelerator may instead keep using a compatible older baseline and recheck its
bounded candidates against current rows. An update, delete, `REINDEX`, or
model-contract change prevents stale completion from becoming active.

The worker completes actionable semantic work, but urgent lexical sealing can
take priority and does not wait for inference. Mutation maintenance processes
changed versions; optional accelerator preparation can traverse a complete
stored baseline and capture included heap values. Full corpus inference occurs
during initial build and explicit `REINDEX`, not during accelerator refresh.

## CRUD

No application hook is required:

```sql
INSERT INTO docs VALUES (42, 'new semantic document');
UPDATE docs SET body = 'updated semantic document' WHERE id = 42;
DELETE FROM docs WHERE id = 42;
VACUUM docs;
```

Mutation, semantic completion, dead-tuple retirement, and page reuse all
publish through the relation's document-COW authority. Transaction rollback
cannot expose partial lexical or semantic state.

## Maintenance

```sql
SELECT evoke_index_maintain('docs_search_idx'::regclass);
SELECT evoke_index_try_maintain('docs_search_idx'::regclass);
```

Mutation seals and selected compaction use bounded work. Accelerator
preparation can be longer and uses short snapshots for included heap values;
it does not re-encode unchanged rows. `VACUUM` records reclaimable state; maintenance
advances reader-safe frontiers, reuses retired pages, and truncates obsolete
tail pages when safe.

Model-contract changes fail closed until explicit rebuild:

```sql
REINDEX INDEX docs_search_idx;
```

## Readiness

```sql
SELECT evoke_index_options('docs_search_idx'::regclass);
SELECT evoke_index_status('docs_search_idx'::regclass);
```

`query_ready` is a correctness/readiness gate, not a warm-latency guarantee.
Before admitting performance-sensitive traffic, also inspect `performance_ready`
and warm state and run representative bounded queries. Diagnose:

- index type and `sae_enabled`;
- root and physical-layout validity;
- pending semantic count, age, bytes, and convergence;
- bounded runtime-contract and model identity;
- active posting and document counts;
- maintenance failures and the reported blocker.

`ready_baseline_delta` and `baseline_current=false` can describe healthy
service from a compatible older accelerator. They do not require disabling it
or rebuilding the index. Inspect warm metadata, candidate quality, latency,
and actual background progress separately. See
[Maintenance lifecycle](../maintenance-lifecycle.md).

`evoke_index_status(...)` deliberately does not hash every model artifact or walk
the complete generation. Use `evoke_index_audit(...)` for that explicit heavy
qualification step, not for routine polling.

```sql
-- Explicit release/incident qualification, not a frequent health check.
SELECT evoke_index_audit('docs_search_idx'::regclass);
```

Applications still query only through `evoke_query(...)`. Runtime queue and
worker telemetry are privileged operational diagnostics, not a second API.

If maintenance workers are unavailable but query inference remains healthy,
committed changes remain lexical-visible through the exact route and semantic
debt accumulates in linked L0. A compatible accelerator can keep serving its
prior baseline, with declared post-baseline omissions, while maintenance is
restored. SSR queries still need a live shared runtime to encode query text:
an unavailable inference pool fails closed, even with a resident accelerator.
Search never falls back to an independent BM25 index.

## Backup And Replication

All durable index state lives in relation pages and follows PostgreSQL WAL,
backup, crash recovery, physical replication, `REINDEX`, and `DROP INDEX`.
Install compatible extension packages and the same immutable model contract on
every node. Match PostgreSQL/extension versions and verify each installed binary;
node-local runtime providers and capacity settings may differ with hardware.

The primary is the only writer and root publication authority. A standby replays relation
pages and warms derived shared residency locally; it does not run competing
semantic maintenance while in recovery. After replay, require matching status,
`query_ready = true`, and representative ordered search results before serving
traffic. Logical subscribers build their own index from replicated source
rows.

## Memory And Security

SSR requires `shared_preload_libraries = 'evoke'` and a positive
`evoke.shared_runtime_size`. This is the bounded shared runtime and residency
arena. Runtime workers own tokenizer and ONNX sessions; query backends keep
only bounded request and scoring scratch.

Server-local model paths require trusted deployment ownership. Grant
applications table access and `evoke_query(...)`; do not grant internal runtime
or payload helpers.

## Teardown

```sql
DROP INDEX docs_search_idx;
```

The relation owns the complete durable lifecycle, so no separate cleanup job
or external-object removal follows the drop.

See [Semantic Runtime](semantic-runtime.md),
[Shared Runtime And Residency](../shared-runtime-and-residency.md), and
[Maintenance Lifecycle](../maintenance-lifecycle.md).
