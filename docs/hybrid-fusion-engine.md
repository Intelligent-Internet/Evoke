# Hybrid Fusion Engine

The public hybrid fusion engine is a composition layer above independent
retrieval sources. Applications use `evoke_query(...)` for each Evoke source
and use this engine only when they intentionally combine sources.

The hybrid fusion engine is the implementation layer behind
[Hybrid Vector/Evoke Search](hybrid-search.md). It lets PostgreSQL combine
BM25 or unified Sparse Semantic Retrieval (SSR) candidates, vector candidates,
and other ranked candidate sources into one weighted top-k result set.

The feature is intentionally a late-fusion layer. Each retrieval source keeps
its own best access path:

- `evoke` indexes produce BM25 or unified lexical/semantic candidates.
- Vector extensions such as `pgvector` or VectorChord produce vector
  candidates.
- Ordinary SQL can produce any additional ranked candidate source.

Evoke only owns the final candidate normalization, weighting,
de-duplication, fusion, and ordering step.

## What Is Complete

The current implementation includes the pieces needed for hybrid ranking
inside PostgreSQL:

- A generic candidate type, `evoke_result_hybrid_candidate`.
- A generic hit type, `evoke_result_hybrid_hit`.
- BM25 candidate constructors and a direct BM25 index adapter.
- Vector candidate constructors that accept ordinary SQL distances.
- A public C-backed fusion function,
  `evoke_hybrid_fuse_candidates(...)`.
- Reciprocal-rank fusion through `fusion => 'rrf'`.
- Weighted score fusion through `fusion => 'score'`.
- Explicit normalizers: `identity`, `negative_distance`,
  `inverse_distance`, `minmax`, `zscore`, and `rank`.
- Per-result debug arrays for source names, raw values, normalized scores,
  weighted scores, and source ranks.
- Regression coverage that does not require a vector extension.
- A non-`public` schema smoke test for fresh installation and schema-safe
  helper resolution.
- A local benchmark that compares the C fast path with the SQL reference.

The core extension has no hard dependency on `pgvector`, VectorChord, or any
specific vector type. Vector support is optional because the vector side is
just SQL that emits hybrid candidates.

## Execution Model

The fusion function accepts an array of candidate rows:

```sql
evoke_hybrid_fuse_candidates(
    candidates evoke_result_hybrid_candidate[],
    k integer,
    fusion text DEFAULT 'rrf',
    rrf_k real DEFAULT 60,
    epsilon real DEFAULT 1e-6
)
```

Internally, the C fast path performs these steps:

1. Parse and validate candidate rows.
2. Ignore candidates with NULL, NaN, or infinite raw values.
3. Assign deterministic source-local ranks when the input rank is not usable.
4. Compute per-source statistics needed by score normalizers.
5. Normalize each source-local value.
6. De-duplicate candidates by `(source_name, ctid)`.
7. Aggregate weighted source contributions by `ctid`.
8. Sort final hits by fused score and deterministic `ctid` tie-breaker.
9. Return the top `k` rows with debug arrays.

Every `ctid` must identify a row in the same base table and SQL snapshot.
The candidate type has no table OID or document-ID field. Cross-table or
cross-partition sources require application SQL to map and aggregate stable
document IDs, not a mixed TID array.

This keeps expensive windowing, grouping, and final sorting out of ordinary
SQL plans while preserving the same observable result semantics as the SQL
reference implementation.

## Fusion Methods

### RRF

`rrf` is the default and recommended first choice for mixed BM25/vector
ranking:

```text
source contribution = weight / (rrf_k + source_rank)
```

RRF avoids comparing BM25 scores with vector distances directly. This is the
safer default because BM25 is normally higher-is-better, vector distance is
normally lower-is-better, and their raw scales are unrelated.

### Score Fusion

`score` fusion is for workloads that have chosen an explicit normalization
strategy:

```text
source contribution = weight * normalized_score
```

Use it when you have measured that a normalizer gives stable ranking quality
for your corpus. Do not assume raw BM25 scores and vector distances are
directly comparable.

## Normalizers

| Normalizer | Typical use |
| --- | --- |
| `identity` | Already comparable higher-is-better scores. |
| `negative_distance` | Lower-is-better distances that should become higher-is-better scores. |
| `inverse_distance` | Distance signals where close neighbors should dominate smoothly. |
| `minmax` | Per-source score fusion when only candidate-local ranges are available. |
| `zscore` | Per-source score fusion when candidate values have meaningful variance. |
| `rank` | Score fusion that should behave closer to rank-only fusion. |

Degenerate cases are explicit:

- Empty inputs return no rows.
- Single-row `minmax` sources normalize to `1.0`.
- Zero-variance `zscore` sources normalize to `0.0`.
- NULL, NaN, and infinite raw values are ignored.

## Product Query Shape

For RAG-style search with SQL filters, keep retrieval source ownership clear:

```sql
WITH title_candidates AS (
    SELECT c
    FROM evoke_hybrid_bm25_candidates(
        'title',
        'docs_title_bm25_idx'::regclass,
        'how to use a computer',
        2.0,
        1000
    ) AS c
),
vector_candidates AS (
    SELECT evoke_hybrid_vector_candidate(
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
    FROM evoke_hybrid_fuse_candidates(
        ARRAY(
            SELECT c FROM title_candidates
            UNION ALL
            SELECT c FROM vector_candidates
        ),
        20,
        'rrf'
    )
)
SELECT d.*, h.score, h.source_names, h.weighted_scores
FROM hybrid_hits AS h
JOIN docs AS d ON d.ctid = h.ctid
WHERE d.published_at >= DATE '2022-01-01'
  AND d.published_at < DATE '2024-01-01'
ORDER BY h.score DESC, d.id;
```

The final filter keeps returned rows aligned with the requested SQL predicate,
but can remove rows after the fused top-k cutoff. Push filters into each
source's supported retrieval path where possible. Increasing `candidate_k`
can improve coverage but cannot guarantee global fused top-k from truncated
source prefixes. See [Filtering and Recall](hybrid-search.md#filtering-and-recall).

## Performance Boundary

The recorded local benchmark covers candidate counts up to `5000` and shows
a large improvement over its SQL reference. This is dated evidence for that
candidate-only workload, not a latency guarantee for source retrieval,
filtering, or a current deployment.

The feature is still late fusion over a materialized candidate array. If a
future workload needs very high QPS with candidate pools far above `5000` per
source, the next likely optimization is a streaming or table-source API that
avoids building one large composite array before fusion.

See [Hybrid Fusion Benchmark](performance/reports/hybrid-fusion-benchmark.md)
for measured results.

## Usage Guidance

Applications may use the C-backed function directly:

```sql
evoke_hybrid_fuse_candidates(...)
```

Recommended defaults:

- Use `rrf` first for mixed BM25/vector ranking.
- Start with weights that express source importance, not raw score scale.
- Use `score` only after selecting and benchmarking a normalizer.
- Keep vector retrieval in the vector extension's own indexed SQL path.
- Keep Evoke retrieval in `evoke_query(...)` or its public candidate adapters.
- Use the debug arrays to inspect why a document won.

## Validation

Current validation covers:

- fixed regression cases for RRF and score fusion
- normalizer edge cases
- empty input
- non-finite candidate values
- BM25 adapter output
- C fast path versus SQL reference comparison
- schema-qualified extension usage
- fresh installation in both `public` and non-`public` schemas
- local C-versus-SQL benchmark results

See also:

- [Hybrid Vector/Evoke Search](hybrid-search.md)
- [API Reference](api-reference.md#public-composition-apis)
- [Query Semantics](query-semantics.md#prepared-diagnostics-and-product-composition)
- [Testing and Validation](testing-and-validation.md)
- [Hybrid Fusion Benchmark](performance/reports/hybrid-fusion-benchmark.md)
