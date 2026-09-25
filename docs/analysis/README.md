# Competitive Analysis

This page summarizes what `evoke` learned from nearby projects and
what it explicitly treats as not suitable for this project.

This is historical comparative evidence, not the current product contract.
Use the [API Reference](../api-reference.md),
[Query Semantics](../query-semantics.md), and
[Architecture](../architecture-and-design.md) for current behavior.

## pg_bm25s

Useful ideas:

- PostgreSQL-facing SQL surface design
- ordered-scan integration lessons
- practical operator ergonomics

Not suitable for this project:

- SQL/SPI-heavy retrieval architecture
- table-backed core search path

Outcome:

- `evoke` absorbed the worthwhile PostgreSQL integration ideas
  while keeping the retrieval core on the native C path

Detailed notes are archived in
[`reports/Competitive_Analysis_pg_bm25s.md`](reports/Competitive_Analysis_pg_bm25s.md).

## VectorChord-BM25

Useful ideas:

- cleaner prepared or index-bound query values
- stronger filtered top-k execution concepts
- mutable-corpus engineering discipline
- stronger differential testing expectations

Not suitable for this project:

- replacing the `bm25s`-aligned core with Block-WeakAnd
- making an external tokenizer mandatory
- changing the project into a different retrieval engine

Outcome:

- it influenced future API and maintenance directions, not the current
  BM25 core contract

Detailed notes are archived in
[`reports/Competitive_Analysis_VectorChord_BM25.md`](reports/Competitive_Analysis_VectorChord_BM25.md).

## ParadeDB

Useful ideas:

- clearer API layering
- stronger filtered ranked scan story
- better documentation around top-k execution shapes
- better query-builder ergonomics
- score-carrying table-returning APIs instead of forcing applications to
  reconstruct scores row by row
- prepared or index-bound query values as a safer way to express parsed
  query state

Not suitable for this project:

- whole-engine takeover around Tantivy and custom scans
- planner-dependent semantics that drift away from the current BM25
  contract
- generic scalar `score(id)` compatibility, because it encourages
  row-by-row scoring and weakens the exact top-k path
- full drop-in query-language emulation, because it would add parser,
  planner, and execution burden without improving the core BM25 contract
- silent semantic drift in tokenization, normalization, or ranking just to
  imitate a friendlier SQL surface

Outcome:

- it influenced both design direction and concrete branch work:
  - stronger top-k specialization
  - narrower candidate planning
  - phrase-path pruning and verification improvements
- the SQL compatibility review was closed out by absorbing the useful API
  shape while rejecting drop-in compatibility:
  - `rowset` remains the canonical exact BM25 contract
  - `@@` remains a document-match predicate, not a scoring API
  - `<=>` remains an ordered-retrieval surface only when PostgreSQL uses a
    real `evoke` index path
  - SQL wrappers, prepared/index-bound query values, filtered ranked
    helpers, score-carrying result APIs, field-aware query helpers, and
    the field-aware multicolumn engine were added as explicit
    layers over the current retrieval core
  - generic scalar `score(id)` and planner-only hidden semantics were
    deliberately not added
- the full local `main` vs branch closeout stayed positive on the full
  15-dataset BEIR list:
  - `evoke_ids`: median `+18.60%`
  - `evoke_text`: median `+24.52%`
- the follow-up cloud PG18 rerun used to refresh the published matrix
  also stayed positive overall:
  - `evoke_ids`: median `+37.93%` vs the documented baseline
  - `evoke_text`: median `+16.12%` vs the documented baseline
- it did not justify replacing the PostgreSQL-native, `bm25s`-aligned
  core architecture

Detailed notes are archived in
[`reports/Competitive_Analysis_ParadeDB.md`](reports/Competitive_Analysis_ParadeDB.md).

Detailed notes on ParadeDB's Tantivy fork are archived in
[`reports/Competitive_Analysis_ParadeDB_Tantivy.md`](reports/Competitive_Analysis_ParadeDB_Tantivy.md).

The final raw comparison data for the branch closeout is archived in:

- [`../performance/data/raw/tantivy-followup-local-main-vs-branch-2026-04-02`](../performance/data/raw/tantivy-followup-local-main-vs-branch-2026-04-02)
- [`../performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02`](../performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02)
