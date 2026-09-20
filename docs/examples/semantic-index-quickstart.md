# Semantic Index Quickstart

Start with [Getting Started](../getting-started.md) for the default BM25 path.
This focused example adds semantic postings to BM25 in the same relation-owned
index with `sae = true`.

## 1. Create A Table

```sql
CREATE EXTENSION ii42;

CREATE TABLE docs (
    id bigint PRIMARY KEY,
    body text NOT NULL
);

INSERT INTO docs VALUES
    (1, 'postgres index maintenance and transaction safety'),
    (2, 'semantic retrieval with sparse postings'),
    (3, 'cooking pasta with tomato sauce');
```

## 2. Enable Semantic Runtime

Release packages include the default model checkout. Source installations must
first [download and provision the model](semantic-model-checkout.md#download-the-default-model).
Before starting
PostgreSQL, enable the shared runtime:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
```

## 3. Create And Query The Index

Restart PostgreSQL, then create the unified index:

```sql
CREATE INDEX docs_semantic_idx
ON docs USING ii42 (body)
WITH (sae = true);

SELECT source.id, source.body, hit.score
FROM ii42_query(
    'docs_semantic_idx'::regclass,
    'database search',
    10
) AS hit
JOIN docs AS source ON source.ctid = hit.ctid
ORDER BY hit.score DESC, source.id;
```

`sae = true` does not create a vector side index. Lexical and semantic
postings share one checked root and scorer. Sparse Semantic Retrieval (SSR) is
eventual-only: mutation is lexical-first and semantic evidence is completed by
shared workers.

Check readiness with:

```sql
SELECT ii42_index_status('docs_semantic_idx'::regclass);
```

Require `query_ready = true` before serving traffic. For production warm-query
qualification, also follow the [operations readiness checks](semantic-index-operations.md#readiness).

Remove the index through the ordinary PostgreSQL lifecycle:

```sql
DROP INDEX docs_semantic_idx;
```

## Next Steps

- [Index Parameters](../index-parameters.md) explains runtime precision,
  posting precision, alpha profiles, model overrides, preload, memory, and
  other advanced index configuration.
- [Getting Started](../getting-started.md) covers the canonical first-use
  lifecycle.
- [Multicolumn Indexes](../multicolumn-indexes.md) and
  [Field-Aware Indexes](../field-aware-indexes.md) cover advanced index shapes.
- [Semantic Index Operations](semantic-index-operations.md) covers CRUD,
  eventual completion, maintenance, rebuild, replication, and teardown.
- [Query Semantics](../query-semantics.md) covers natural ranked SQL,
  filtering, explicit-hit search, and current query limits.
- [Migration](../upgrading.md) covers the only historical product migration:
  `psql_bm25s` to current II42.
