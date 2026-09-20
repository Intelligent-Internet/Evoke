# Supported Input Types

`ii42` supports five indexed source-column types:

- `int4[]`
- `text[]`
- `varchar[]`
- `text`
- `varchar`

In BM25 mode they all feed the same index core, through two
different input models:

- pretokenized inputs owned by the application:
  - `int4[]`
  - `text[]`
  - `varchar[]`
- scalar text inputs tokenized at the index boundary:
  - `text`
  - `varchar`

## Overview

| Source type | Input model | Best fit | Main trade-off |
| --- | --- | --- | --- |
| `int4[]` | pre-encoded token IDs | exact retrieval without text processing | requires an external vocabulary or token-ID pipeline |
| `text[]` | pretokenized text tokens | explicit token control with strong exact performance | application must materialize tokens |
| `varchar[]` | pretokenized text tokens | same use case as `text[]` for schemas that already use `varchar[]` | application must materialize tokens |
| `text` | raw scalar text | easiest onboarding from ordinary PostgreSQL schemas | indexing and some verification paths pay tokenization cost inside the extension |
| `varchar` | raw scalar text | same as `text` when schema already uses `varchar` | indexing and some verification paths pay tokenization cost inside the extension |

For `sae = true`, the source must be text-like: `text`, `varchar`, `text[]`,
or `varchar[]`. The model receives one text representation derived from those
values. `int4[]` remains a single-column BM25 input and is not a semantic model
input.

## How `ii42` Supports Them

In BM25 mode, for `int4[]`, `text[]`, and `varchar[]`, the index receives the caller's
token stream directly:

- `int4[]` passes pre-encoded token IDs
- `text[]` passes pretokenized text
- `varchar[]` follows the same text-token path after adapting each array
  element to the text-like token interface used by the index

For scalar `text` and `varchar`, the extension tokenizes at the index
boundary and then lowers the result to the same internal token-stream
model. That keeps one BM25 scoring and postings core while still
letting ordinary SQL schemas start from raw text columns.

The scalar text pipeline uses the same project text-processing path used
by the SQL helpers:

- Unicode-aware tokenization
- NFC normalization
- ICU word-break segmentation
- Unicode case folding
- optional stopword filtering
- optional English Porter stemming
- optional Latin-diacritic folding

For the index-level scalar text parameters, see
[Index Parameters](index-parameters.md#bm25-text-processing).

In Sparse Semantic Retrieval (SSR) mode, the frozen model tokenizer and
normalization contract govern both lexical and semantic atoms. Text arrays are
joined into model input; they do not bypass model tokenization or preserve
arbitrary application tokens as model IDs. See
[Semantic Model Checkout](examples/semantic-model-checkout.md).

## Selection Guidance

### `int4[]`

This path avoids text tokenization inside the extension and gives the
application complete ownership of vocabulary and token-ID assignment.

### `text[]`

This path keeps the token stream explicit and avoids scalar retokenization.

Use it when:

- the application already tokenizes documents
- token boundaries must stay explicit
- phrase and verification-heavy paths should avoid retokenizing raw text

### `varchar[]`

This follows the same pretokenized path as `text[]`. It mainly exists so
existing schemas do not need to rewrite arrays just to use the index.
Its behavior and expected performance profile should track `text[]`
closely.

### `text` and `varchar`

These are the easiest way to get started because the schema can index an
ordinary text column directly. The trade-off is that the extension must
tokenize during indexing, refresh, and some exact verification paths.

That makes scalar text columns the best choice when:

- ease of adoption matters more than explicit token materialization
- the dataset is moderate enough that extra tokenization CPU is
  acceptable
- the application wants a direct SQL-column search surface

Pretokenized arrays can reduce indexing CPU when the application already owns
tokenization. Actual throughput depends on corpus shape, token distribution,
PostgreSQL configuration, and hardware; use the current benchmark harness
before choosing a type for performance alone.

## Multicolumn Fusion

Multicolumn fusion indexes currently support:

- `text[]`
- `varchar[]`
- `text`
- `varchar`

Scalar multicolumn fusion tokenizes each indexed scalar column with the
index text options before fusing the resulting token stream. See
[Multicolumn Indexes](multicolumn-indexes.md) for the
current rules and recommended usage.

The same homogeneous multicolumn text-like shapes support `sae = true`. The
columns form one lexical document and one semantic model input by default.
With `field_aware = true`, lexical and semantic atoms retain separate field
namespaces. The shared runtime batch-encodes nonempty fields, while the index
keeps one document version, root, maintenance lifecycle, and scorer.

## Practical Guidance

Use:

- `text` or `varchar` when you want the easiest schema-level start
- `text[]` or `varchar[]` when you already own tokenization and want the
  clearest text-token contract
- `int4[]` when you can own stable token IDs upstream and want to avoid text
  processing inside the extension

Related docs:

- [API Reference](api-reference.md)
- [Query Semantics](query-semantics.md)
- [Performance and Benchmarks](performance/README.md)
- [Multicolumn Indexes](multicolumn-indexes.md)
