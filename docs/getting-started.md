# Getting Started

This is the canonical first-use path for Evoke. It starts with the default
exact BM25 mode, then shows the optional semantic mode. Multicolumn,
field-aware, filtered, and model-specific choices are linked at the end rather
than mixed into the basic lifecycle.

## 1. Install The Extension

Install a package that matches the PostgreSQL major, operating system, and
architecture. Then create the extension in the application database:

```sql
CREATE EXTENSION IF NOT EXISTS ii42;
```

No separate schema is required for ordinary use. Custom schema placement is
optional; it is required only for specific deployment layouts such as the
side-by-side [`psql_bm25s` migration](upgrading.md). Evoke is intentionally
not relocatable after creation.

## 2. Create A Table And BM25 Index

```sql
CREATE TABLE docs (
    id bigint PRIMARY KEY,
    title text NOT NULL,
    body text NOT NULL
);

INSERT INTO docs (id, title, body) VALUES
    (1, 'PostgreSQL indexes', 'transaction-safe index maintenance'),
    (2, 'Sparse retrieval', 'semantic search with unified postings'),
    (3, 'Cooking', 'pasta with tomato sauce');

CREATE INDEX docs_body_idx
ON docs USING ii42 (body);
```

`sae = false` is the default. This index uses exact Lucene-style BM25 and the
ordinary PostgreSQL relation lifecycle.

## 3. Search

The explicit hit overload returns an index-local hit set for either BM25 or
semantic mode:

```sql
SELECT source.id, source.title, hit.score
FROM ii42_query(
    'docs_body_idx'::regclass,
    'reliable database search',
    10
) AS hit
JOIN docs AS source ON source.ctid = hit.ctid
ORDER BY hit.score DESC, source.id;
```

Join by `ctid` in the same statement, then retain the table primary key. Do not
persist `ctid` or the returned `doc_id` as application identity. BM25 also
keeps its native operator and ordered index-scan surfaces; see
[Query Semantics](query-semantics.md).

## 4. Optional Semantic Index

Semantic mode adds model-generated sparse atoms to the same lexical posting
index. It is lexical-first and eventual-only: foreground writes publish
lexical evidence, and shared workers complete semantic evidence later.

Before PostgreSQL starts, configure the shared runtime:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
ii42.runtime_worker_count = 2
```

Restart PostgreSQL after changing `shared_preload_libraries`. Release packages
include the qualified default model checkout. Source installs must provision a
qualified checkout at the compiled default path or set `ii42.sae_model_path`.
Use the public [Beta 1 model download and source-install instructions](examples/semantic-model-checkout.md#download-the-default-model).

Create the optional semantic index:

```sql
CREATE INDEX docs_semantic_idx
ON docs USING ii42 (body)
WITH (sae = true);
```

The same explicit query shape works with `docs_semantic_idx`. Planner-native
semantic ranking can also combine normal table predicates with one bounded
descending score:

```sql
SELECT source.id,
       source.title,
       ii42_query(
           'docs_semantic_idx'::regclass,
           'semantic retrieval'
       ) AS score
FROM docs AS source
ORDER BY score DESC
LIMIT 10;
```

## 5. Check Readiness

```sql
SELECT ii42_index_options('docs_semantic_idx'::regclass);
SELECT ii42_index_status('docs_semantic_idx'::regclass);
```

Require `query_ready = true` before serving traffic. A non-zero semantic
pending count is a valid progressive state when status reports no blocker; the
row remains lexical-searchable through the exact route while workers converge.
The default bounded accelerator may temporarily omit post-baseline rows under
its declared approximate profile.

For latency-sensitive traffic, also qualify the requested warm/performance
state and representative queries; `query_ready` is not a latency guarantee.
See [Semantic Operations](examples/semantic-index-operations.md).

An index owner may request maintenance with non-blocking lock admission for an
operational check. Once admitted, the selected action can still take time:

```sql
SELECT ii42_index_try_maintain('docs_semantic_idx'::regclass);
```

The built-in worker normally performs this work. Do not call maintenance after
every application write.

## 6. Write, Rebuild, And Remove

Ordinary table writes need no extension-specific hook:

```sql
INSERT INTO docs VALUES
    (4, 'Incremental search', 'lexical first semantic completion');

UPDATE docs
SET body = 'bounded convergent semantic maintenance'
WHERE id = 2;

DELETE FROM docs WHERE id = 3;
VACUUM docs;
```

Use `REINDEX` after changing a physical index option or model contract, or for
an explicit whole-corpus rebuild:

```sql
REINDEX INDEX docs_semantic_idx;
```

PostgreSQL owns teardown. Every Evoke payload is relation-owned, so ordinary
`DROP INDEX` removes the complete index without sidecar cleanup:

```sql
DROP INDEX docs_semantic_idx;
DROP INDEX docs_body_idx;
```

## Common Errors

| Error or symptom | What to check |
| --- | --- |
| Semantic search reports that the shared runtime is unavailable | Configure `shared_preload_libraries`, a positive shared runtime size, and restart PostgreSQL. Verify the configured control database is connectable. |
| Model or artifact mismatch | Poll `ii42_index_status(...)`, then run `ii42_index_audit(...)` explicitly. `REINDEX` if the immutable model contract changed. |
| A joined query returns fewer than `k` rows | Check whether a SQL filter was applied after hit generation. For semantic indexes, put it on the base table with planner-native `ORDER BY ii42_query(...) DESC LIMIT k`, or use a supported filtered overload. A scope-backed approximate baseline can also underfill during convergence; see the query contract. |
| Search is rejected on an RLS table or partitioned parent | Those shapes cannot provide a safe globally exact `ii42_query(...)` result in the current release. |

## Next Reading

- [API Reference](api-reference.md)
- [Supported Input Types](input-types.md)
- [Multicolumn Indexes](multicolumn-indexes.md)
- [Field-Aware Indexes](field-aware-indexes.md)
- [Index Parameters](index-parameters.md)
- [Semantic Operations](examples/semantic-index-operations.md)
- [Query Semantics](query-semantics.md)
