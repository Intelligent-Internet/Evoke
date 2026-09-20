# Multicolumn Indexes

This page describes the current multicolumn `ii42` index feature.

The goal is to support one index over several homogeneous text-like columns
without turning the feature into a hidden tax on the existing single-column
hot path. The fused-document shape is available for BM25 and `sae = true`.

## What This Feature Is

A multicolumn fusion index lets one `ii42` index treat multiple
`text[]`, `varchar[]`, `text`, or `varchar` columns as one BM25
document.

Example:

```sql
CREATE TABLE docs (
    id bigint PRIMARY KEY,
    title_tokens text[],
    body_tokens text[]
);

CREATE INDEX docs_fused_bm25_idx
    ON docs USING ii42 (title_tokens, body_tokens);
```

In the current implementation:

- the indexed columns must all share one type:
  - `text[]`
  - `varchar[]`
  - `text`
  - `varchar`
- the columns are fused in index-definition order
- `NULL` indexed columns are skipped
- the fused token stream is what BM25 sees for build, refresh, and
  direct regclass query APIs
- scalar `text` and `varchar` columns are tokenized with the index text
  options before fusion

This means the feature behaves like a single BM25 document assembled
from multiple tokenized sources.

## Semantic-Enabled Fusion

Use the same column list with `sae = true`:

```sql
CREATE TABLE docs_text (
    id bigint PRIMARY KEY,
    title text,
    body text
);

CREATE INDEX docs_text_semantic_idx
    ON docs_text USING ii42 (title, body)
    WITH (sae = true);
```

This remains one relation-owned unified posting index. Lexical processing
fuses columns in definition order. The semantic compiler receives one model
input: scalar text columns are joined in definition order, while token arrays
are joined into one token stream. `NULL` columns are skipped.

Applications search with the same `ii42_query(...)` call used by a
single-column Sparse Semantic Retrieval (SSR) index. SSR is eventual-only, and
foreground writes remain lexical-first while shared workers complete the changed
document version.

By default, semantic-enabled multicolumn indexes do not preserve lexical field
identity. Add `field_aware = true` to place each field's lexical and semantic
atoms in separate namespaces while retaining one native scorer and lifecycle.
Public `ii42_query(...)` supports equal or explicit whole-field weights.

## What This Feature Is Not

The default multicolumn index is not a full field-aware BM25F design.

It does not keep separate:

- per-field scores
- per-field length normalization
- per-field explain output
- per-field query masks inside one index

If you need field identity inside one index, use
the single-index field-aware mode enabled by `field_aware = true` and
documented in [Field-Aware Indexes](field-aware-indexes.md). To combine
independent indexes with custom weights, use the public composition APIs in
[Multi-Index Fusion](multi-index-fusion.md).

## Field-Aware Index

For multicolumn `text[]`, `varchar[]`, `text`, or `varchar` indexes, the
index can preserve field identity inside one BM25 payload:

```sql
CREATE INDEX docs_field_bm25_idx
    ON docs USING ii42 (title_tokens, body_tokens)
    WITH (field_aware = true);
```

The same shape supports the unified semantic index:

```sql
CREATE INDEX docs_field_semantic_idx
    ON docs_text USING ii42 (title, body)
    WITH (sae = true, field_aware = true);
```

Its lexical and semantic namespaces remain field-scoped. The shared runtime
batch-encodes nonempty fields and publishes one completion for each changed
document version.

This mode stores field-scoped internal tokens. Applications query it through
`ii42_query(...)`, which uses equal field weights by default. Applications
can select a subset and set whole-field weights with its public overload:

```sql
SELECT d.id, h.score
FROM ii42_query(
    'docs_field_semantic_idx'::regclass,
    'bird migration',
    ARRAY['title', 'body'],
    ARRAY[3.0, 1.0]::real[],
    10
) AS h
JOIN docs_text AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

The score is `sum(weight * (field_BM25 + field_semantic))`. Owner-only
`ii42_field_aware_query(...)` and `ii42_field_aware_query_tokens(...)` remain
exact-BM25 diagnostic surfaces.

`ii42_query(...)` searches all indexed fields with equal weight on
`field_aware = true` indexes for token and simple raw term queries. Its public
field-aware overload can select a field subset or custom per-field weights.
For BM25 indexes, complex raw-query semantics such as phrases and boolean
filters remain available on non-field-aware fused indexes. In SSR mode, text is
compiled by the model contract, not interpreted as the BM25 raw-query language.

This is one BM25 payload with field-scoped terms, not BM25F and not
multiple internal sub-indexes. Combined document length
normalization still applies.

## Current Query Surface

The explicit-hit API for multicolumn fusion indexes is:

- `ii42_query(...)`

An SSR multicolumn index also supports planner-native
`ORDER BY ii42_query(...) DESC LIMIT k`. BM25 fusion indexes retain the explicit
hit route because the existing ordinary BM25 ordering operators are
single-column surfaces.

Example application query:

```sql
SELECT d.id, h.score
FROM ii42_query(
    'docs_fused_bm25_idx'::regclass,
    'cat bird',
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

Owner-only exact diagnostics may additionally use `ii42_query_tokens(...)`
and prepared-query helpers built on the same regclass retrieval path. For
example:

```sql
SELECT d.id, h.score
FROM ii42_query_tokens(
    'docs_fused_bm25_idx'::regclass,
    ARRAY['cat', 'bird']::text[],
    10
) AS h
JOIN docs AS d ON d.ctid = h.ctid
ORDER BY h.score DESC, d.id;
```

## Current Operator Limitation

Multicolumn fusion indexes do not currently support the SQL operator
scan surfaces:

- `@@`
- `@@@`
- `<=>`

Those operator paths remain single-column surfaces in this version.

Reason:

- the operator scan path still assumes one indexed heap attribute for
  planner-visible filtering and ordering
- keeping those paths single-column avoids pushing extra complexity into
  the existing fast path

So the practical rule is:

- use `ii42_query(...)` for multicolumn fusion indexes
- use operator scans on single-column indexes
- on `field_aware = true` multicolumn indexes, `ii42_query(...)` searches all
  fields with equal weight for token and simple raw term queries
- use the public field-aware `ii42_query(...)` overload for custom whole-field
  weights or a field subset

## Build And Refresh Semantics

The fused token stream is materialized during:

- initial index build
- foreground mutation into the relation-owned linked L0
- explicit `REINDEX`

That keeps the storage format aligned with the existing single-column
core design. The current implementation does not introduce a second
posting format just for multicolumn indexes.

For scalar `text` and `varchar` columns:

- each non-NULL indexed column is tokenized with the index text options
- the resulting token streams are appended in index-column order
- the fused token stream is what gets indexed, refreshed, and verified

## Phrase And Boolean Semantics

Phrase and boolean matching operate over the fused token stream.

That means:

- phrase matching sees the fused token order
- phrase boundaries are not field boundaries
- `"cat bird"` can match if `cat` is at the end of one indexed column
  and `bird` is at the start of the next one in the fused order

This is a deliberate tradeoff. The feature is a fused-document index,
not a field-isolated query model.

If that phrase behavior is not acceptable for your application, keep
separate field indexes and use weighted multi-index fusion instead.

## When To Use It

Use a multicolumn fusion index when:

- one logical search document is already spread across several tokenized
  columns
- you want one index object instead of several field indexes
- you do not need separate field weights inside that one index
- you are happy with fused-document phrase semantics

Typical examples:

- `title_tokens + body_tokens`
- `title_text + body_text`
- `headline_tokens + summary_tokens + body_tokens`
- several chunked text arrays that are already conceptually one search
  document

## When Not To Use It

Do not use the default fused-document multicolumn index when:

- you need `title^3 + body^1` style weighting inside one query
- you need field-isolated phrase semantics
- you want SQL operator scans over the same index

To compose separate single-column indexes, use the public product APIs:

- `ii42_fusion_query_weighted(...)`
- `ii42_fusion_query_fields(...)`
- `ii42_fusion_query(...)`

For the lowest-overhead route, use one multicolumn index and
`ii42_query(...)`. If field identity is more important than BM25F-style field
normalization, use `field_aware = true`; the public field-aware overload can
select fields and apply custom whole-field weights.
See [Multi-Index Fusion](multi-index-fusion.md) and
[Field-Aware Indexes](field-aware-indexes.md).
