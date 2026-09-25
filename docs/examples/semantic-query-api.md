# Semantic Query API

Semantic-enabled indexes support natural PostgreSQL ranking:

```sql
SELECT source.*,
       ii42_query('docs_search_idx'::regclass, 'query text') AS score
FROM docs AS source
WHERE source.publish_date >= DATE '2026-01-01'
ORDER BY score DESC
LIMIT 20;
```

The scalar `ii42_query(...)` overload above is a planner marker. One Evoke
`CustomScan` evaluates the
ordinary relation predicate, encodes once, and ranks only the matching rows
through the same unified root and scorer. It does not evaluate the model once
per source row.

The explicit hit API works for semantic-enabled and BM25 indexes:

```sql
ii42_query(
    index_name regclass,
    query_text text,
    k int4,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF ii42_result_hit
```

## Explicit Hit Dispatch

`ii42_query(...)` reads the stored index options:

- `sae = false`: exact BM25 path;
- `sae = true`: shared query encoder plus one unified page-native scorer.

Semantic search rejects BM25-only overrides and `weight_mask`. The model
checkout owns query normalization, atom identity, and score calibration.

## Explicit Hit Result

`ii42_result_hit` contains `ctid`, index-local `doc_id`, and `score`.

```sql
SELECT source.id, hit.score
FROM ii42_query('docs_search_idx'::regclass, 'query text', 20) AS hit
JOIN docs AS source ON source.ctid = hit.ctid
ORDER BY hit.score DESC, source.id;
```

Join by `ctid` in the same statement. Do not persist `ctid` or `doc_id` as a
business identifier.

## Readiness

```sql
SELECT ii42_index_status('docs_search_idx'::regclass);
```

Semantic search requires a valid checked root, shared runtime, model checkout,
artifact identity, and matching runtime-contract signature. Status performs a
bounded readiness check and deliberately does not hash every artifact. Run
`ii42_index_audit(...)` explicitly for deep generation and model-artifact
qualification. Any detected mismatch fails closed; it does not fall back to
BM25.

## Progressive Visibility

Sparse Semantic Retrieval (SSR) is eventual-only and lexical-first. A committed
changed row can be lexical-searchable through the exact route before shared
workers add semantic atoms. Both states live under the same root/linked-L0
authority. A compatible bounded accelerator may continue serving its older
baseline and temporarily
omit post-baseline rows while workers converge.

Heap MVCC hides aborted, superseded, and deleted rows. Maintenance completion
cannot attach stale model output to a replaced document version.

## Security And Limits

- Caller needs `SELECT` on the indexed table.
- Row-level-security tables are rejected.
- Partitioned parent indexes are not globally queryable.
- Planner-native ranking currently accepts one base table, descending rank,
  and a bounded `LIMIT`; unsupported joins, secondary ordering, and row locks
  fail closed.
- Internal semantic encoder/scorer functions remain revoked from `PUBLIC`.
- Exact-BM25 operators and diagnostics do not dispatch to semantic scoring.

Teardown uses PostgreSQL `DROP INDEX`; all Evoke payloads are relation-owned.

See [API Reference](../api-reference.md) and
[Query Semantics](../query-semantics.md).
