# Index Policy

`consistency` controls foreground mutation behavior. It does not choose a
different index or scorer. Automatic policies share the page-native v3 lifecycle;
manual refresh has an explicit heap-rebuild path.

## Policy Matrix

| Mode | `realtime` | `eventual` | `manual` |
| --- | --- | --- | --- |
| BM25 | Supported; default | Supported | Supported |
| `sae = true` / Sparse Semantic Retrieval (SSR) | Rejected | Supported; default | Rejected |

SSR is eventual-only because document inference belongs to shared workers, not
foreground DML.

## BM25 Policies

### Realtime

Use for ordinary mutable tables when newly committed lexical evidence must be
searchable immediately. Writes append exact relation-owned posting changes;
bounded maintenance later improves physical read shape.

```sql
CREATE INDEX docs_body_idx
ON docs USING evoke (body)
WITH (consistency = 'realtime');
```

### Eventual

This policy uses durable linked L0 and automatic background convergence. In
the current native implementation, both automatic BM25 policies append changed
lexical evidence and the exact route reads its committed, visible projection.
`eventual` is not a switch to a second delayed lexical index. Physical
convergence can lag; status exposes the outstanding work.

```sql
CREATE INDEX docs_body_idx
ON docs USING evoke (body)
WITH (consistency = 'eventual');
```

### Manual

Use for static corpora or deployments with explicit refresh scheduling. DML
marks the index stale; automatic due-index discovery does not maintain it.
Explicit maintenance may scan and rebuild from the heap under stronger
relation locks, so schedule it as a rebuild rather than a bounded background
seal.

```sql
CREATE INDEX docs_body_idx
ON docs USING evoke (body)
WITH (consistency = 'manual');
```

## Semantic Policy

Omitting `consistency` with `sae = true` selects `eventual`:

```sql
CREATE INDEX docs_semantic_idx
ON docs USING evoke (body)
WITH (sae = true);
```

Foreground semantic mutation does three bounded things:

1. records the exact document version;
2. appends transaction-owned lexical evidence, visible according to commit;
3. records semantic-pending state.

It performs no document inference. The worker then completes a bounded batch,
and later attempts may seal, compact, fold, or reclaim the same relation-owned
state. Lexical and semantic evidence are never published as independent
indexes.

If semantic completion cannot progress, the bounded mutation/admission limits
eventually reject additional work rather than allowing unbounded memory or
storage growth. Existing readable state remains authoritative.

## Automatic Maintenance

The built-in selector distinguishes urgent work from optional convergence:

1. protect finite active/pending frontiers and relieve blocked sealing;
2. complete actionable semantic versions;
3. coalesce ordinary structural work, including periodic small debt;
4. prepare eligible accelerators, folds, and warm images;
5. reclaim pages when reader fencing permits reuse.

Each selected publication captures a checked source and frontier, prepares
unreachable objects, and revalidates before its WAL-logged root switch.
Concurrent active-L0 appends are preserved. Optional semantic accelerator
preparation can scan a complete sealed baseline and included heap values;
bounded batches do not make that total work constant or incremental. A separate
build lock avoids duplicating it, while hard frontier pressure can take priority.

Compatible serving baselines do not expire when delta appears. The periodic
low-debt interval is a work trigger, not a maximum stale age. See
[Maintenance lifecycle](maintenance-lifecycle.md) for the detailed selection
and publication rules.

Worker hints reduce latency but are not durable authority. Periodic catalog
reconciliation rediscovers automatic-policy debt after restart or lost hints.

## Operator Functions

```sql
SELECT evoke_index_status('docs_body_idx'::regclass);
SELECT * FROM evoke_index_details('docs_body_idx'::regclass);
SELECT evoke_index_policy_recommend(
    'docs_body_idx'::regclass,
    'balanced'
);

SELECT evoke_index_try_maintain('docs_body_idx'::regclass);
SELECT evoke_index_maintain('docs_body_idx'::regclass);
SELECT * FROM evoke_index_maintain_due(4);
```

- Prefer `try_maintain` for unattended work because it does not wait on a busy
  publication boundary.
- Use `maintain` when the caller is allowed to wait.
- `maintain_due` is revoked from `PUBLIC`; run it as a trusted role that owns
  its target indexes.
- `evoke_index_policy_recommend(...)` is advisory and does not mutate options.
  For SSR it always recommends eventual.

The built-in worker already handles automatic policies. `pg_cron` is optional
and useful only when an operator wants an additional time-based wakeup:

```sql
SELECT evoke_index_maintain_due(2);
```

List manual indexes explicitly; they are never selected by `maintain_due`.

## Preload Policy

`auto_preload` is independent from `consistency`:

```sql
ALTER INDEX docs_body_idx SET (auto_preload = 10);
```

It prioritizes best-effort relation-page warmup, exact-root resident-fold
admission, and optional term-local HOT_FOLD work. It does not change freshness,
scoring, or durable state. Eviction may change cold latency only.

Semantic indexes require shared preload for model execution regardless of
`auto_preload`; the option controls proactive warming, not runtime existence.

## `VACUUM`, `REINDEX`, And Drop

- `VACUUM` supplies exact dead-TID retirement. Use
  `VACUUM (INDEX_CLEANUP ON)` when testing immediate retirement convergence.
- `REINDEX` is the explicit whole-corpus rebuild for changed physical options,
  model contract, repair, or operator-requested compaction.
- mutation seals and compaction do not re-encode unchanged documents;
  optional accelerator preparation can read a corpus-sized stored baseline;
- PostgreSQL `DROP INDEX` removes either BM25 or semantic-enabled index and all
  relation-owned payloads through the ordinary lifecycle.

## Operational Defaults

- Start BM25 with its `realtime` default; do not assume that changing the
  policy alone selects a cheaper posting or query implementation.
- Use BM25 `manual` only when an external refresh contract is intentional.
- Use the default eventual-only semantic policy; size shared runtime and worker
  throughput from measured completion backlog.
- Monitor `evoke_index_status(...)` for readiness and blocker state, then use
  `evoke_index_details(...)` for diagnosis.
- Treat warm residency as acceleration, never as correctness evidence.

For exact reloptions and GUCs see [Index Parameters](index-parameters.md). For
storage mechanics see [Maintenance Lifecycle](maintenance-lifecycle.md).
