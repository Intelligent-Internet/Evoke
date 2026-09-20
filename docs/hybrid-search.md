# Hybrid Vector/II-42 Search

The public `ii42_hybrid_*` APIs combine candidates from II-42 and external
retrieval engines such as pgvector or VectorChord. This composition layer is a
product capability above the unified single-index design; it does not split or
modify an II-42 index internally.

Hybrid search combines BM25 or unified Sparse Semantic Retrieval (SSR)
candidates with external vector candidates inside PostgreSQL. A single SSR
index already combines lexical and sparse semantic evidence; this API is only
for additional independent sources.
The implementation is intentionally a late
fusion layer: each source keeps its own best index access path, and
`ii42` only combines already-retrieved candidates.

This keeps the core extension independent from `pgvector`, VectorChord, and
other vector extensions. Vector candidates are supplied as ordinary SQL rows
containing `ctid`, a raw distance or score, rank, weight, and normalization
metadata.

## When To Use It

Use hybrid composition when an application needs these capabilities:

- lexical precision from exact BM25 fields such as title, abstract, or body
- semantic recall from vector embeddings
- one final top-k ordering inside PostgreSQL
- explicit weights and explainable per-source contributions

Do not use hybrid search when a single BM25 index or a single vector index is
already the complete ranking signal. In that case, the direct index path is
simpler and faster.

## Candidate Model

The fusion layer operates on `ii42_result_hybrid_candidate` values:

- `source_name`: stable source label such as `title`, `body`, or `embedding`
- `ctid`: row identity for same-table fusion within one statement
- `raw_value`: BM25 score, vector distance, or another source-local value
- `source_rank`: one-based rank inside that source
- `weight`: source weight
- `normalizer`: score normalization strategy
- `direction`: whether higher or lower raw values are better

The output is a set of `ii42_result_hybrid_hit` rows with:

- final fused score
- source count
- source names
- raw values
- normalized scores
- weighted scores
- source ranks

All candidate TIDs must refer to the same base table and SQL snapshot. TIDs can
collide across tables and partitions, and these helpers carry no table OID or
application document ID. For cross-table or document/chunk fusion, map each
source hit to a stable document ID and combine it in application SQL instead.

The per-source fields are part of the product contract because hybrid ranking
is only useful
when the final order can be inspected and tuned.

`ii42_hybrid_fuse_candidates(...)` uses a C fast path for normalization,
de-duplication, grouping, and final ordering. Use `ii42_query(...)` for each
II-42 source and hybrid fusion only when independent retrieval engines are
intentionally combined.

For implementation boundaries, performance expectations, and validation
coverage, see [Hybrid Fusion Engine](hybrid-fusion-engine.md).

## Default Fusion: RRF

The default fusion method is reciprocal rank fusion:

```text
source contribution = weight / (rrf_k + source_rank)
```

The default `rrf_k` is `60`.

RRF is the safest default because it does not compare raw BM25 scores with
raw vector distances. BM25 and vector distances live on different scales, and
vector distance is usually lower-is-better while BM25 is higher-is-better.

## Score Fusion

Advanced users can use `fusion => 'score'` with explicit normalization.

Supported normalizers:

- `identity`: use the raw score directly; with `lower_is_better`, the value is
  negated
- `negative_distance`: use `-distance`
- `inverse_distance`: use `1 / (epsilon + distance)`
- `minmax`: normalize each source over its retrieved candidates
- `zscore`: z-score each source over its retrieved candidates
- `rank`: use `1 / source_rank`

Degenerate cases are handled deliberately:

- empty inputs return no rows
- single-row `minmax` sources normalize to `1.0`
- zero-variance `zscore` sources normalize to `0.0`
- NULL, NaN, and infinite raw values are ignored

For mixed BM25/vector workloads, prefer `rrf` first. Use score fusion only
after measuring whether a specific normalization improves result quality.

For the historical SQL-first dual-engine design that motivated this API
surface, see the archived
[Hybrid Vector/BM25 Search Use-Case Design](research-sae/reports/designs/ii42-hybrid-vector-bm25-use-case-design.md).

## BM25 Sources

BM25 candidates can be created directly from a `ii42` index:

```sql
WITH bm25_candidates AS (
    SELECT c
    FROM ii42_hybrid_bm25_candidates(
        'title',
        'docs_title_bm25_idx'::regclass,
        'how to use a computer',
        2.0,
        1000
    ) AS c
)
SELECT h.*
FROM ii42_hybrid_fuse_candidates(
    ARRAY(SELECT c FROM bm25_candidates),
    20,
    'rrf'
) AS h;
```

This is still late fusion. The BM25 index produces candidates first; the
hybrid layer only combines the resulting rows.

## Vector Sources

Vector candidates are supplied by SQL. The core extension does not require
the `vector` type.

With VectorChord or pgvector installed, the vector side can look like this:

```sql
WITH vector_candidates AS (
    SELECT ii42_hybrid_vector_candidate(
        'embedding',
        d.ctid,
        (d.embedding <-> '[0.1,0.2,0.3]'::vector)::real,
        row_number() OVER (
            ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
        )::int4,
        1.0,
        'minmax'
    ) AS c
    FROM docs AS d
    ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
    LIMIT 1000
)
SELECT h.*
FROM ii42_hybrid_fuse_candidates(
    ARRAY(SELECT c FROM vector_candidates),
    20,
    'rrf'
) AS h;
```

For RRF, the vector normalizer is not used; only rank and weight matter. For
score fusion, vector distances should normally use `lower_is_better` through
`ii42_hybrid_vector_candidate(...)`, with `minmax`,
`negative_distance`, or `inverse_distance`.

## Mixed BM25 And Vector Example

```sql
WITH title_candidates AS (
    SELECT c
    FROM ii42_hybrid_bm25_candidates(
        'title',
        'docs_title_bm25_idx'::regclass,
        'how to use a computer',
        2.0,
        1000
    ) AS c
),
body_candidates AS (
    SELECT c
    FROM ii42_hybrid_bm25_candidates(
        'body',
        'docs_body_bm25_idx'::regclass,
        'how to use a computer',
        1.8,
        1000
    ) AS c
),
vector_candidates AS (
    SELECT ii42_hybrid_vector_candidate(
        'embedding',
        d.ctid,
        (d.embedding <-> '[0.1,0.2,0.3]'::vector)::real,
        row_number() OVER (
            ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
        )::int4,
        1.0,
        'minmax'
    ) AS c
    FROM docs AS d
    WHERE d.published_at >= DATE '2022-01-01'
      AND d.published_at < DATE '2024-01-01'
    ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
    LIMIT 1000
),
hybrid_hits AS (
    SELECT *
    FROM ii42_hybrid_fuse_candidates(
        ARRAY(
            SELECT c FROM title_candidates
            UNION ALL
            SELECT c FROM body_candidates
            UNION ALL
            SELECT c FROM vector_candidates
        ),
        20,
        'rrf'
    )
)
SELECT d.*, h.score, h.source_names, h.normalized_scores
FROM hybrid_hits AS h
JOIN docs AS d ON d.ctid = h.ctid
WHERE d.published_at >= DATE '2022-01-01'
  AND d.published_at < DATE '2024-01-01'
ORDER BY h.score DESC, d.id;
```

The final `WHERE` keeps the returned rows aligned with the requested time
window, but runs after fusion's `k` cutoff and can underfill. Push the predicate
into each source when supported. For an SSR source, use a filtered
`ii42_query(...)` overload or planner-native query before constructing hybrid
candidates; increasing each source's candidate budget is only a recall tradeoff.

## Filtering And Recall

Late fusion has a known recall tradeoff: filters applied after each source
has produced candidates can remove many candidates. If the filter is narrow,
the final result may miss rows that would have appeared with a larger
candidate pool.

Finite source prefixes also cannot guarantee global fused top-k, even without
filters: an omitted document can have a high combined score. Measure coverage
as candidate budgets change.

Do not concatenate TIDs from several partitions into one fusion call. A
partition-local call is valid only when every source uses that same child
table; cross-partition ranking needs stable document-ID aggregation outside
the TID-keyed API.

## Optional VectorChord Smoke

Run this only in a database that already has VectorChord or another
pgvector-compatible index installed:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS vchord;

EXPLAIN
WITH vector_candidates AS (
    SELECT ii42_hybrid_vector_candidate(
        'embedding',
        d.ctid,
        (d.embedding <-> '[0.1,0.2,0.3]'::vector)::real,
        row_number() OVER (
            ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
        )::int4,
        1.0,
        'minmax'
    ) AS c
    FROM docs AS d
    ORDER BY d.embedding <-> '[0.1,0.2,0.3]'::vector
    LIMIT 1000
)
SELECT *
FROM ii42_hybrid_fuse_candidates(
    ARRAY(SELECT c FROM vector_candidates),
    20,
    'rrf'
);
```

The plan should show the vector index path for the candidate CTE. The hybrid
function should only appear after the vector candidate query has produced its
top-k rows.
