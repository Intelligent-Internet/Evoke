# Field-Aware Indexes

`evoke` provides two field-aware product shapes:

- public fusion of separate single-column indexes
- one multicolumn index with `field_aware = true`

The second shape preserves field identity inside one physical index without
building one index per field. Applications query it through
`evoke_query(...)` with equal or explicit field weights. It supports exact
BM25 and semantic-enabled unified postings through the same public API.

## When To Use This

Use a field-aware index when:

- the table has several searchable text-like fields
- one index object is operationally simpler than several field indexes
- applications need weights such as `title^3 + body^1`
- BM25F-style per-field length normalization is not required
- a Sparse Semantic Retrieval (SSR) index should keep lexical field identity while sharing one
  physical index and lifecycle

To weight independently maintained indexes at query time, keep separate field
indexes and use the public composition helpers documented in
[Multi-Index Fusion](multi-index-fusion.md).
For columns that are simply one logical document, use the default fused
multicolumn index documented in
[Multicolumn Indexes](multicolumn-indexes.md).

## Supported Column Types

Field-aware indexes are available for multicolumn indexes whose
indexed columns are all one of these homogeneous text-like types:

- `text[]`
- `varchar[]`
- `text`
- `varchar`

Scalar `text` and `varchar` fields are tokenized with the index text options
before the field-scoped tokens are stored.

## Build An Index

Pretokenized fields:

```sql
CREATE INDEX docs_field_bm25_idx
    ON docs USING evoke (title_tokens, body_tokens)
    WITH (field_aware = true);
```

Scalar text fields:

```sql
CREATE INDEX docs_text_field_bm25_idx
    ON docs USING evoke (title, body)
    WITH (
        field_aware = true,
        text_lowercase = true,
        text_fold_diacritics = true
);
```

Semantic-enabled field-aware index:

```sql
CREATE INDEX docs_text_field_semantic_idx
    ON docs USING evoke (title, body)
    WITH (
        sae = true,
        field_aware = true
    );
```

This remains one relation-owned unified posting index with one mutation and
maintenance lifecycle. Foreground writes publish field-scoped lexical atoms
immediately; the shared worker batch-encodes changed nonempty fields and
publishes one semantic completion for the document version asynchronously.

`field_aware = true` is opt-in. Without it, the same multicolumn index is a
fused-document index: columns are appended into one logical token stream and
field identity is not preserved.

## Application Query

Applications use the same product entrypoint as every other index:

```sql
SELECT d.id, h.score
FROM evoke_query(
    'docs_field_bm25_idx'::regclass,
    'bird migration',
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

This searches every field with equal weight.

To search a subset of fields or apply explicit weights, use the public
field-aware overload:

```sql
SELECT d.id, h.score
FROM evoke_query(
    'docs_text_field_semantic_idx'::regclass,
    'bird migration',
    ARRAY['title', 'body'],
    ARRAY[3.0, 1.0]::real[],
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

The score is:

```text
sum(field_weight * (field_BM25 + field_semantic))
```

This is one native accumulation over one posting index. It is not score fusion
between independent BM25 and semantic indexes. Field names must be unique and
belong to the index. Weights must be finite and non-negative. Passing `NULL`
weights uses `1.0` for every selected field.

The query expands lexical and, for SSR, semantic atoms into field-scoped terms
with the selected field weight, then uses the same route dispatcher as a
single-column index. BM25 uses resident-fold or page-native execution; SSR can
use the published semantic accelerator. It does not materialize one corpus score array per field
or retain a private index snapshot in each backend when shared runtime exists.

## Owner-Only Token Diagnostics

Extension owners may use `evoke_field_aware_query_tokens(...)` for token-array
diagnostics:

```sql
SELECT d.id, h.score
FROM evoke_field_aware_query_tokens(
    'docs_field_bm25_idx'::regclass,
    ARRAY['bird'],
    ARRAY['title_tokens', 'body_tokens'],
    ARRAY[3.0, 1.0]::real[],
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

For scalar `text` or `varchar` diagnostics, use
`evoke_field_aware_query(...)` so the query text is tokenized with
the index text options before field-aware retrieval:

```sql
SELECT d.id, h.score
FROM evoke_field_aware_query(
    'docs_text_field_bm25_idx'::regclass,
    'bird migration',
    ARRAY['title', 'body'],
    ARRAY[3.0, 1.0]::real[],
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

Pass `NULL` for weights, or omit the argument, to use equal weights.

## Generic Query Behavior

The public `evoke_query(...)` family searches all indexed fields with equal
weight by default. Its field-aware overload selects a field subset and optional
weights for either BM25 or SSR indexes.

The owner-only `evoke_query_tokens(...)` diagnostic also searches every indexed
field with equal weight on BM25 indexes. Explicit field-aware diagnostics can
select fields and weights; they are not the public semantic query route.

For BM25, complex raw-query semantics such as phrase and boolean filters remain
available on non-field-aware fused indexes. Field-aware BM25 generic raw queries
reject complex forms rather than silently using non-field-aware planning.
In SSR mode, input is model-compiled text, not the BM25 raw-query language; its
tokenizer and normalization come from the model checkout rather than BM25 text
reloptions.

## Internal Semantics

The field-aware engine stores field-scoped lexical atoms in one posting
payload.
For example:

- `title_tokens = ARRAY['bird']` becomes one internal token namespace
- `body_tokens = ARRAY['bird']` becomes a different internal token namespace
- a query over both fields scores both streams and sums the weighted scores

For `sae = true`, lexical and semantic atoms both use one namespace per indexed
field. The shared runtime batch-encodes each changed nonempty field and maps
the result into that field's semantic namespace. Query lexical and semantic
atoms are copied into every selected field namespace, multiplied by the same
field weight, and accumulated by the same native scorer.

Semantic admission is also field-local. The configured document semantic
budget ratio is applied independently to each field's mapped lexical atom
count, then the retained field atoms are merged into the document's one
unified posting row. This prevents a long field from consuming a short field's
semantic budget before query-time weights are applied. The summed budget stays
approximately proportional to the document's total mapped lexical atoms.

This field expansion does not create per-field roots, generations, pending
queues, or maintenance policies. A row has one document version and one
semantic-pending identity; the worker merges the completed fields and publishes
one atomic completion for that version. `REINDEX` encodes the full corpus, while
ordinary maintenance only encodes changed document versions.

This gives one multicolumn index with field-aware lexical evidence. It is not
BM25F:

- document length normalization uses the combined indexed document length
- there is no per-field length normalization
- there is no per-field explain output
- phrase and boolean verification are not field-isolated in the generic path

The implementation keeps the single-column hot path and the default
multicolumn fused-document path separate. Field-aware behavior is used only
when `field_aware = true` is set on a multicolumn text-like index.

## Choosing A Multi-Field Shape

Use separate field indexes when:

- field-local phrase or verification semantics are required
- per-field explainability matters
- isolated rebuild and maintenance per field is more important than having
  one index object

Use a default multicolumn fused-document index when:

- the columns are really one logical search document
- field weights are not needed
- phrase matching across column boundaries is acceptable

Use `field_aware = true` when:

- one multicolumn index object is preferred
- lexical and semantic field identity or query-time field weights are required
- BM25F-style per-field normalization is not required

## Current Limits

The current field-aware engine deliberately does not implement:

- BM25F scoring
- per-field length normalization
- per-field explain output
- SQL operator scans over multicolumn indexes
- field-isolated complex raw-query planning

Those limits are part of the current contract, not unfinished work in this
feature.
