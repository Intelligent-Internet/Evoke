# Multi-Index Fusion

This guide combines candidate sets from separate indexes. For multiple fields
inside one index, use [Field-Aware Indexes](field-aware-indexes.md) instead.
The `evoke_fusion_*` family identifies rows by `ctid`: all input indexes must
refer to the same base table and be queried within the same SQL snapshot.
Do not fuse TIDs from different tables or partitions as if they were document
IDs. The [hybrid API](hybrid-search.md) also uses same-table TIDs. For
cross-table or document/chunk fusion, first map hits to stable document IDs
and aggregate them in application SQL outside these TID-keyed helpers.

Each source contributes only its retrieved candidate prefix. Increasing
`candidate_k` improves coverage but does not guarantee the global fused top-k:
a document below every source's cutoff can still have a high combined score.

The public `evoke_fusion_*` APIs combine candidates from multiple independent
Evoke indexes. This is a product composition layer above single-index search;
it does not change the storage, mutation, maintenance, or scorer of any source
index.

If you want one BM25 index over several `text[]` columns instead, see
[Multicolumn Indexes](multicolumn-indexes.md).

Use one multicolumn field-aware index when one lifecycle is preferable. Use
this composition layer when sources need independent indexes, weights, or
maintenance policies.

## When To Use This

Use this pattern when:

- one logical document has multiple indexed fields
- fields should contribute with different weights
- an application needs one final ranked result set
- the query-time score should stay attached to the current search

Typical examples:

- `title` gets the highest weight
- `abstract` gets a medium weight
- `body` gets a lower weight

## Core Model

The stable contract is:

1. each source has its own BM25 or semantic-enabled `evoke` index
2. each source runs its own native top-k retrieval
3. scores are fused only after those top-k result sets exist

That means:

- there is no hidden cross-field scoring engine
- field names are metadata, not implicit scoring signals
- public `evoke_query(...)` remains each source's retrieval foundation
- one single-index fusion mode is a different feature with different
  semantics; see [Multicolumn Indexes](multicolumn-indexes.md)

## Schema Shape

One practical schema shape is:

```sql
CREATE TABLE docs (
    id bigint PRIMARY KEY,
    title_tokens text[] NOT NULL,
    abstract_tokens text[] NOT NULL,
    body_tokens text[] NOT NULL
);

CREATE INDEX docs_title_bm25_idx
    ON docs USING evoke (title_tokens);

CREATE INDEX docs_abstract_bm25_idx
    ON docs USING evoke (abstract_tokens);

CREATE INDEX docs_body_bm25_idx
    ON docs USING evoke (body_tokens);
```

Each field is indexed independently. That keeps the storage and
retrieval contract simple and makes weighting explicit at query time.

## Level 1: Explicit Baseline Fusion

This is the most explicit shape. It is also the easiest one to reason about.

```sql
WITH fused AS (
    SELECT *
    FROM evoke_fusion(
        ARRAY(
            SELECT h
            FROM evoke_query(
                'docs_title_bm25_idx'::regclass,
                'cancer therapy',
                20
            ) AS h
        ),
        3.0,
        ARRAY(
            SELECT h
            FROM evoke_query(
                'docs_abstract_bm25_idx'::regclass,
                'cancer therapy',
                20
            ) AS h
        ),
        1.5,
        10
    )
)
SELECT d.id, d.title_tokens, h.score
FROM fused AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

Logic:

- retrieve native top-k from one index
- retrieve native top-k from another index
- weight the scores explicitly
- fuse the result sets
- join back to rows through `ctid`

This is the reference mental model for the higher-level helpers below.

## Level 2: Structured Field Queries

When an application has several weighted fields, the structured surface is:

- `evoke_fusion_field_query(...)`
- `evoke_fusion_query_fields(...)`

```sql
SELECT d.id, d.title_tokens, h.score
FROM evoke_fusion_query_fields(
    ARRAY[
        evoke_fusion_field_query(
            'title',
            'docs_title_bm25_idx'::regclass,
            'cancer therapy',
            3.0
        ),
        evoke_fusion_field_query(
            'abstract',
            'docs_abstract_bm25_idx'::regclass,
            'cancer therapy',
            1.5
        ),
        evoke_fusion_field_query(
            'body',
            'docs_body_bm25_idx'::regclass,
            'cancer therapy',
            1.0
        )
    ]::evoke_result_fusion_field_query[],
    10,
    30,
    NULL
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

What this buys you:

- field names are carried explicitly
- each field still owns one weighted prepared query
- the array is structured instead of being only positional

What it does not change:

- native retrieval still happens per field/index
- fusion still happens after retrieval
- the returned score is still query-scoped

## Level 3: Convenience Search Fields

If several fields share the same raw query text and normalization
options, the shortest surface is:

- `evoke_fusion_query(...)`

Named form:

```sql
SELECT d.id, d.title_tokens, h.score
FROM evoke_fusion_query(
    ARRAY['title', 'abstract', 'body']::text[],
    ARRAY[
        'docs_title_bm25_idx'::regclass,
        'docs_abstract_bm25_idx'::regclass,
        'docs_body_bm25_idx'::regclass
    ],
    'cancer therapy',
    ARRAY[3.0, 1.5, 1.0]::real[],
    10,
    30,
    NULL,
    true,
    ARRAY['and', 'or']::text[],
    true,
    false
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

Short form:

```sql
SELECT d.id, d.title_tokens, h.score
FROM evoke_fusion_query(
    ARRAY[
        'docs_title_bm25_idx'::regclass,
        'docs_abstract_bm25_idx'::regclass,
        'docs_body_bm25_idx'::regclass
    ],
    'cancer therapy',
    ARRAY[3.0, 1.5, 1.0]::real[],
    10,
    30,
    NULL
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

Recommendation:

- use the named form in long-lived application SQL
- keep the short form for smaller SQL snippets or quick experiments

## Understanding `k` And `candidate_k`

The multi-index composition helpers expose two important limits:

- `k`
  - final result size
- `candidate_k`
  - per-field retrieval size before fusion

Practical rule:

- start with `candidate_k >= k`
- use a larger `candidate_k` when several fields may surface different
  documents that need to survive into the fused top-k

If `candidate_k` is too small, fusion can miss documents that would have
scored well after combining fields.

## Getting Back Rows And Scores

The intended product pattern is:

1. get query-scoped hit rows
2. join hits back to rows

That is why the examples use:

- `evoke_fusion_query_fields(...)`
- `evoke_fusion_query(...)`
- `JOIN ... ON d.ctid = h.ctid`

This keeps score attached to the current query execution. It also avoids
the high-risk pattern of a generic scalar `score(id)` call evaluated row
by row later.

## Syntax Summary

Use these in increasing order of abstraction:

- `evoke_fusion(...)`
  - most explicit
- `evoke_fusion_weighted_query(...)`
  - weighted prepared query
- `evoke_fusion_field_query(...)`
  - weighted prepared query with a field name
- `evoke_fusion_query_weighted(...)`
  - fusion from weighted prepared queries
- `evoke_fusion_query_fields(...)`
  - fusion from named field queries
- `evoke_fusion_query(...)`
  - convenience form for shared query text

## Design Notes

The important design boundaries are:

- no hidden planner magic
- no row-by-row generic rescoring
- no silent change to BM25 ranking semantics
- no requirement that composition helpers behave like another
  extension's API

This is an application composition layer built on top of public
`evoke_query(...)`, not a second search engine inside the extension.
