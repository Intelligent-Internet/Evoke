# API Reference

Evoke has one index lifecycle and one overloaded `evoke_query(...)` product
family. Scalar overloads compose semantic ranking with ordinary table SQL;
overloads with an explicit `k` return hit rows. Index options select exact BM25
or the Sparse Semantic Retrieval (SSR) unified-posting path internally.

## Application API

### Create

```sql
CREATE INDEX index_name
ON table_name USING evoke (column_name [, ...])
WITH (...);
```

`sae = false` is the default. `sae = true` selects the semantic-enabled
contract and is eventual-only. See [Index Parameters](index-parameters.md).

Single-column indexes support `int4[]`, `text[]`, `varchar[]`, `text`, or
`varchar`. Multicolumn indexes require homogeneous `text[]`, `varchar[]`,
`text`, or `varchar` columns. The default multicolumn shape fuses columns into
one logical document. `field_aware = true` preserves field identity for BM25;
it can also be combined with `sae = true` to preserve both lexical and semantic
field identity in one unified posting index.

SSR indexes may declare ordinary table columns with PostgreSQL `INCLUDE`.
Included columns are non-scoring scope dimensions: they do not enter BM25,
semantic encoding, field weights, or document length. On a converged root,
exact scalar `eq`/`in`/`range` and array `overlap` predicates can resolve
through same-root scope postings; string columns also support `ILIKE`. A
compatible published scope remains usable while linked-L0 and sealed delta
work converges. Returned candidates are rechecked against current heap rows,
while post-baseline matches may be temporarily absent. Unsupported predicate
shapes fall back to the current SQL predicate resolver. When every structured
predicate is scope-backed, the bounded serving-scope route does not materialize
the complete current matching universe; after current-row recheck it may return
fewer than `k` until background convergence publishes a newer scope.

See [Getting Started](getting-started.md) for the canonical first-use flow and
[Supported Input Types](input-types.md) for every supported index shape.

### Planner-Native Semantic Search

```sql
evoke_query(index_name regclass, query_text text) RETURNS real

evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[]
) RETURNS real
```

The two- and four-argument scalar `evoke_query(...)` overloads are planner
markers, not row-local scoring functions. They must appear as the only
descending sort key over one base table with a bounded `LIMIT`. PostgreSQL
owns final `WHERE` evaluation under the statement snapshot. For predicates
without an eligible same-root scope, PostgreSQL supplies the complete visible
TID subset and the custom scan ranks inside it. Eligible `INCLUDE` predicates
may instead use a compatible published scope baseline. Every returned row is
rechecked against the current snapshot, but post-baseline matches may be absent
when the bounded probe fills the limit. If recheck cannot fill the requested
limit, planner-native execution discards the probe and falls back to the
complete current subset.

```sql
SELECT source.*,
       evoke_query('docs_search_idx'::regclass, 'graph retrieval') AS score
FROM docs AS source
WHERE source.publish_date >= DATE '2026-01-01'
  AND source.categories && ARRAY['cs.LG']
ORDER BY score DESC
LIMIT 20;
```

The current path rejects joins, row-dependent marker arguments, ascending or
secondary ordering, unbounded ranking, row locking, `WITH TIES`, RLS, and
partitioned-parent global ranking. The marker fails closed if PostgreSQL cannot
use the Evoke custom executor.

### Explicit Hit Search

```sql
evoke_query(
    index_name regclass,
    query_text text,
    k int4,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
```

Field-aware indexes also expose an overload through the same product name:

```sql
evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    k int4
)
RETURNS SETOF evoke_result_hit
```

SSR indexes expose predicate-defined subset-ranking overloads:

```sql
evoke_query(
    index_name regclass,
    query_text text,
    filters jsonb,
    k int4 DEFAULT 10
)

evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    filters jsonb,
    k int4 DEFAULT 10
)

evoke_query(
    index_name regclass,
    query_text text,
    allowed_tids tid[],
    k int4 DEFAULT 10
)

evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    allowed_tids tid[],
    k int4 DEFAULT 10
)
```

Each predicate-defined overload returns `SETOF evoke_result_hit`.

The JSON overload is the explicit structured-predicate API. Filters are ANDed
by column. Each column specifies exactly one operation: `eq`, `in`, `overlap`,
`ilike`,
`ilike_any`, or `range`; range accepts `gt`, `gte`, `lt`, and `lte` bounds.
`ilike` accepts one PostgreSQL pattern and `ilike_any` accepts an OR-list of
patterns; on a string-array column they match individual elements.

An eligible same-root scope may rank a bounded published baseline and recheck
every returned row against the active snapshot. Returned membership is current,
but the matching universe may omit post-baseline rows and a fully scope-backed
request may return fewer than `k`. Without an eligible scope, Evoke may use a
bounded ranked-prefix probe; if that probe is insufficient, PostgreSQL resolves
the complete current predicate set and can use ordinary B-tree, GIN, BRIN, or
suitable trigram/expression indexes. For a nonempty `overlap` operand, the SQL
resolver includes the equivalent `cardinality(column) > 0` condition so a
matching partial GIN index remains usable. All routes select top-k inside their
selected candidate universe. Scope and SQL-subset routes are predicate-first;
the bounded prefix route is explicitly a global overfetch probe followed by
current predicate recheck, not an exact resumable filtered iterator. Collection
operands are limited to 4,096 values; use a selective table predicate rather
than transporting a large application-owned ID list.

The `tid[]` overload remains a low-level exact-membership boundary: no result
can come from outside the supplied set. Ranking still uses the index's selected
exact or bounded-approximate route, so a stale accelerator may omit an allowed
post-baseline row. Generate TIDs in the same statement and do not persist them
across table rewrites. Null and empty sets return no rows. By default filter
metadata remains PostgreSQL-owned.
An SSR index can opt frequently used exact dimensions into its existing root
with `INCLUDE`; this adds no generation, worker, compaction, or fold lifecycle.
Predicate-defined filtered top-k is reserved for SSR unified indexes;
pure BM25 continues to use PostgreSQL's ordinary predicate path.

The default overload searches all fields with weight `1.0`. The field-aware
overload searches the selected unique fields and applies each weight to the
field's complete unified contribution:
`sum(weight * (BM25 + semantic))`. Weights must be finite and non-negative.

`evoke_result_hit` contains:

| Field | Meaning |
| --- | --- |
| `ctid` | Physical row identity for joining to the indexed table. |
| `doc_id` | Index-local document slot. Do not persist it as row identity. |
| `score` | Query-time score produced by the selected index contract. |

Example:

```sql
SELECT source.id, source.body, hit.score
FROM evoke_query(
    'docs_body_idx'::regclass,
    'postgres index maintenance',
    20
) AS hit
JOIN docs AS source ON source.ctid = hit.ctid
ORDER BY hit.score DESC, source.id;
```

For BM25, omitted normalization arguments inherit the named index options.
For `sae = true`, all BM25-only overrides and `weight_mask` are rejected;
normalization belongs to the model checkout. Field weights remain valid on a
semantic-enabled field-aware index because they scale complete field-local
lexical and semantic evidence after model encoding.

`weight_mask` is an exact-BM25 diagnostic surface and is inherently a
document-slot-sized operation. It is therefore admitted only when the
physical index and all other active fallback snapshots fit the finite positive
per-backend `evoke.workspace_cache_bytes` budget. Ordinary queries should omit
it and use the shared resident-fold/page-native route.

The caller needs `SELECT` on the indexed table. `evoke_query(...)` rejects
row-level-security tables and partitioned parent indexes. An explicit TID set
provides predicate-defined subset ranking, but it is not an RLS policy boundary
and cannot combine independently ranked child corpora into one global top-k.

### Inspect

```sql
evoke_index_options(index_name regclass) RETURNS jsonb
evoke_index_status(index_name regclass) RETURNS jsonb
evoke_index_audit(index_name regclass) RETURNS jsonb
evoke_index_details(index_name regclass) RETURNS TABLE (
    index_name regclass,
    source_type text,
    docs int8,
    index_bytes int8,
    pages int8,
    stale bool,
    consistency text,
    rebuilds int8,
    pending_writes int8,
    pending_deletes int8,
    delta_records int8,
    delta_bytes int8
)
```

- `evoke_index_options(...)` reports effective type, source shape, reloptions,
  and semantic configuration. Semantic indexes report
  `semantic_impact_precision`, `semantic_alpha_mass`, and an `exact` or
  `approximate` `semantic_accuracy_profile`; defaults are `f32` and `1.0`.
  The packed semantic authority always uses 64-document blocks; precision is
  selectable per index, but block geometry is not.
- `evoke_index_status(...)` is the application readiness surface. Check
  `query_ready` and `blocker` rather than interpreting internal counters.
  `query_usable` and `query_ready` mean the exact fallback remains correct.
  For semantic indexes, `performance_ready` additionally requires either a
  serving semantic accelerator or a current exact-root resident fold. A
  serving accelerator may report `state=ready_baseline_delta` and
  `baseline_current=false`: the immutable baseline remains authoritative while
  bounded overfetch and current-row validation reject stale candidates. Newer
  rows may be temporarily omitted under the declared approximate profile. This
  is an expected online state, not a fallback or an invalid accelerator, and it
  has no maximum serving age. Small debt is still scheduled periodically by
  `evoke.maintenance_low_debt_interval_ms`; record and byte high-water marks
  bypass that interval. Status exposes `periodic_refresh_eligible` separately
  from immediate `refresh_due`. Failed publication or a concurrent builder is
  retried no sooner than `evoke.maintenance_timer_interval_ms`; that internal
  cooldown applies only to accelerator construction and never invalidates the
  serving baseline. It becomes
  `state=ready` and `baseline_current=true` after sealing and derived
  publication catch up. A compatible manifest seal does not clear
  `query_metadata_warm`; that marker follows the unchanged accelerator
  directory and baseline sequence while exact manifest projections converge
  in the background. A false marker in this state therefore indicates startup,
  shared-runtime admission pressure, or a genuine serving-authority change,
  not ordinary baseline drift. This keeps the established query-readiness contract
  while preventing a relation-sized semantic fallback from passing
  performance qualification. `performance_blocker` identifies outstanding
  convergence work even when the existing baseline remains usable. If
  `auto_preload > 0`, it also reports
  `query_metadata_not_warm` until accelerator metadata is warm or an exact
  resident fold is current. Its
  generation projection is bounded to root metadata even on relation-sized
  indexes; `diagnostics_complete=false` and null reachability/reclaim fields
  mean that no full storage walk was requested. A semantic accelerator reports
  authenticated fixed-header metadata and `directory_bytes`; aggregate
  artifact `bytes` remains null because calculating it requires walking every
  child reference.
- `evoke_index_audit(...)` is the explicit heavy integrity surface. It validates
  the complete generation closure and SHA-256 hashes every semantic model artifact.
  Do not call it from readiness polling or request paths.
- `evoke_index_details(...)` exposes operator-oriented root, mutation,
  maintenance, and builder details.

These functions enforce access to the indexed table. Status validates bounded
manifest/runtime identity but deliberately reports `model_artifacts_valid` as
null. Audit validates server-local model bytes under the guarded
extension-owner boundary; callers cannot supply arbitrary paths.

### Maintain

```sql
evoke_index_refresh(index_name regclass)
evoke_index_maintain(index_name regclass)
evoke_index_try_maintain(index_name regclass)
evoke_index_maintain_due(max_indexes integer DEFAULT 1)
```

- `evoke_index_maintain(...)` may wait and performs one needed bounded action or
  returns a no-op.
- `evoke_index_try_maintain(...)` avoids waiting on a busy publication boundary
  and returns retryable no-op results when necessary. This is non-blocking lock
  admission, not a deadline on the maintenance action once admitted.
- `evoke_index_maintain_due(...)` uses the same native selector for a bounded
  number of automatic-policy indexes. It is revoked from `PUBLIC` and is meant
  for a trusted maintenance role.
- `evoke_index_refresh(...)` is an explicit operator refresh surface. Use
  `REINDEX` when options or model contract changed.

Per-index mutation and maintenance require index ownership. Semantic
completion, sealing, compaction, fold, and reclamation all act on the same
page-native v3 root and linked L0.

### Drop

```sql
DROP INDEX index_name;
```

PostgreSQL relation lifecycle is authoritative for both modes. All Evoke
payloads are owned by the index relation, so there is no semantic side object
or external cleanup step.

## Text Utilities

The public value-local helpers are:

- `evoke_tokenize_text(text, ...)`;
- `evoke_normalize_tokens(text[], ...)`;
- `evoke_highlight(text[] | text | varchar, query_text, ...)`;
- `evoke_snippet(text[] | text | varchar, query_text, ...)`.

They operate on supplied values. They do not perform index retrieval.

## BM25 Planner Surface

BM25 indexes support PostgreSQL operator integration:

- `tokens @@ 'query text'` for `text[]` and `varchar[]` predicates;
- `value @@@ evoke_prepared_query(...)` for owner diagnostics and scalar text;
- `ORDER BY value <=> query_tokens ASC LIMIT k` for index-ordered retrieval.

`@@` is a boolean predicate, not a ranking API. `<=>` has index ranking
semantics only when PostgreSQL chooses an actual `evoke` index scan. Application
code that needs an explicit hit set can use `evoke_query(...)`; semantic table
queries should prefer `evoke_query(...)`.

These operators are BM25 surfaces. They do not dispatch to semantic scoring.

## Owner-Only BM25 Diagnostics

The extension owner can use exact BM25 functions for regression, benchmark,
and implementation diagnostics:

- `evoke_query_ids(...)`;
- `evoke_query_tokens(...)`;
- `evoke_prepared_query(...)`, `evoke_order_tokens(...)`, and local match/score
  helpers;
- single-index token-level field-weight helpers.

These functions are revoked from `PUBLIC`. Exact BM25 rowset and scoring
helpers reject `sae = true` indexes. None may become an alternate application
API.
Use PostgreSQL `EXPLAIN (FORMAT JSON)` directly when validating planner paths.
Evoke does not wrap or execute caller-supplied SQL text.

## Public Composition APIs

The `evoke_fusion_*` family combines top-k results from multiple independently
maintained `evoke` indexes. The `evoke_hybrid_*` family combines Evoke candidates
with externally retrieved candidates such as vector-index distances. These
families are granted to `PUBLIC`; each Evoke source still enforces source-table
`SELECT` through `evoke_query(...)`.

Fusion hit rows are keyed by `ctid`, so both `evoke_fusion_*` and `evoke_hybrid_*`
sources must belong to the same base table and SQL snapshot. For different
tables or partitions, map hits to stable document IDs and aggregate in
application SQL outside these helpers. Both families fuse finite source
prefixes, not the complete matching universes; candidate limits can affect
recall. See [Multi-Index Fusion](multi-index-fusion.md) and
[Hybrid Fusion Engine](hybrid-fusion-engine.md).

Composition is a product layer above single-index retrieval. It does not alter
an index's unified posting layout, mutation lifecycle, maintenance policy, or
native scorer. Use planner-native SQL or `evoke_query(...)` when one index is
sufficient; use fusion or hybrid APIs only when the product intentionally
combines independent indexes or retrieval engines.

## Runtime And Residency Diagnostics

Runtime and residency surfaces include:

```sql
evoke_index_runtime_state(index_name regclass)
evoke_index_runtime_state_json(index_name regclass)
evoke_index_preload(index_name regclass)
evoke_runtime_cache_clear()
evoke_runtime_service_status()
```

The state functions require `SELECT` on the indexed table.
`evoke_index_preload(...)` requires index ownership, while cache clearing and
runtime-service status are extension-owner diagnostics revoked from `PUBLIC`.
These functions report or control checked-root markers, relation page warming,
optional HOT_FOLD and exact-root resident-fold state, bounded workspace, and
shared runtime state. They do not expose posting storage or a second mutation
authority. `evoke_index_runtime_state_json(...)` reports
`resident_fold_current`, `resident_fold_entries`, and `resident_fold_bytes`.
The text and JSON forms use the same C snapshot collector; JSON diagnostics do
not parse the human-readable state string.

`evoke_index_preload(...)` first attempts to publish one pointer-free exact-root
fold when the index is converged and the complete image fits
`evoke.shared_runtime_size`. Its result then reports
`tier=shared_resident_fold`, `prewarm_scope=exact`, and `resident_bytes`.
Otherwise it performs exact relation-page warming within
`evoke.prewarm_max_bytes` or bounded roots-and-payload warming for a larger
index. Query readers still validate the checked root; durable authority never
moves out of the index relation.

`evoke_runtime_cache_clear()` is revoked from `PUBLIC`. Clearing disposable
residency may change cold latency but cannot change results.

## Privilege Summary

| Surface | Intended caller |
| --- | --- |
| `evoke_query` | Application role with source-table `SELECT`. |
| `evoke_fusion_*`, `evoke_hybrid_*` | Application role; source queries retain their own authorization. |
| Options/status/details | Role allowed to inspect the source relation. |
| Per-index maintenance | Index owner. |
| PostgreSQL `DROP INDEX` | Index owner under the normal PostgreSQL lifecycle. |
| `evoke_index_maintain_due` | Trusted maintenance role. |
| Text utilities | Application role. |
| Runtime state | Role allowed to inspect the source relation. |
| Per-index preload | Index owner. |
| Exact BM25, model, cache, and runtime internals | Extension owner/diagnostics. |

Internal functions ending in `_internal` are implementation boundaries. Do not
grant them to application roles.

## See Also

- [Function Index](functions.md)
- [Query Semantics](query-semantics.md)
- [Index Parameters](index-parameters.md)
- [Index Policy](index-policy.md)
- [Semantic Query API](examples/semantic-query-api.md)
