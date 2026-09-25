# ParadeDB Analysis

Date: 2026-03-23

This note reviews [`paradedb/paradedb`](https://github.com/paradedb/paradedb)
from the perspective of `evoke`.

The goal is not to reproduce ParadeDB's architecture. The goal is to identify
what is genuinely useful for a `bm25s`-aligned PostgreSQL extension and
separate that from ideas that would pull `evoke` away from its current
design goals.

## Scope

This review focused on ParadeDB's BM25-related pieces:

- SQL API and search syntax
- planner and executor integration
- Top-K and filtered query execution
- tokenizer and query-builder design
- write/update model where it materially affects read performance

Primary upstream sources:

- [ParadeDB repository](https://github.com/paradedb/paradedb)
- [How Text Search Works](https://docs.paradedb.com/documentation/full-text/overview)
- [How Advanced Query Functions Work](https://docs.paradedb.com/documentation/query-builder/overview)
- [Filtering](https://docs.paradedb.com/documentation/filtering)
- [Top K](https://docs.paradedb.com/documentation/sorting/topk)
- [Architecture](https://docs.paradedb.com/welcome/architecture)

## What ParadeDB Is

ParadeDB is not a `bm25s` port. It is a broader PostgreSQL search and
analytics system built around `pg_search`, `pgrx`, Tantivy, custom scan
hooks, and a BM25 covering index.

That distinction matters. `evoke` is centered on a `bm25s`-aligned C
core with PostgreSQL-native persistence and explicit SQL retrieval APIs.
ParadeDB is centered on a broader operator-driven and planner-driven search
system, with heavier control over planning and execution.

So ParadeDB is relevant less as a drop-in technical blueprint and more as a
source of mature API, planner, and operational ideas.

## What ParadeDB Does Well

### 1. Strong API layering

ParadeDB has a clear split between:

- simple operators for common query shapes
- richer typed query-builder functions
- an internal structured query representation

Examples:

- `|||` for match disjunction
- `&&&` for match conjunction
- `###` for phrase
- `@@@` for richer query-builder driven search

Under the surface, ParadeDB rewrites those surfaces into a typed query object
and then into Tantivy queries. The main lesson is not the exact operator set.
The lesson is that a mature search extension benefits from having:

- a canonical internal query representation
- multiple SQL entry points layered on top of it
- a clear distinction between convenience syntax and exact execution

This is useful to `evoke`. We already have a real raw-query AST and a
canonical `rowset` path. ParadeDB reinforces that the next useful step
is not more ad hoc operators, but better prepared and index-bound query
values, plus cleaner API layering around the existing core.

### 2. Top-K is treated as a first-class execution mode

ParadeDB's documentation is unusually explicit about when its fast Top-K path
can be used. It documents the required SQL shape and encourages users to
verify it with `EXPLAIN`.

That is a strong design choice for a search extension. It helps users reason
about performance and avoids the common problem where a feature exists but is
too opaque to use well.

The implementation side also reflects this priority:

- custom scan hooks are deeply integrated
- Top-K has explicit execution nodes
- there is dedicated logic for segmented Top-K and pruning
- filter pushdown and sort requirements are part of the design, not an
  afterthought

For `evoke`, the useful lesson is not "adopt ParadeDB's custom scan
stack". The useful lesson is:

- make filtered ranked scans a first-class optimization target
- make fast-path requirements easy to explain and inspect
- add clearer introspection for when ordered scans or filtered ranked scans
  are or are not active

### 3. Filter pushdown is broad and practical

ParadeDB does a good job of treating filtering as part of the search problem,
not as something to tack on after ranking. It supports pushdown over indexed
non-text fields and certain literal-tokenized text fields.

This matters because the common real-world query is not just "search, then
rank". It is "search, filter, sort, limit".

`evoke` already moved in this direction with:

- index-aware `@@`
- combined `@@` plus ordered `<=>` scan support

ParadeDB shows that this direction is correct, and that there is still room
to broaden it. The most useful takeaway is to continue expanding native
filtered ranked scans inside the access method, without falling back to
SQL-heavy execution.

### 4. Its operational documentation is honest and useful

ParadeDB documents:

- read tuning
- write tuning
- parallel worker expectations
- autovacuum implications
- segment count tradeoffs
- logical replication workflow
- highlighting cost caveats

This is one of the strongest non-code aspects of the project. A mature search
extension needs not only a fast engine, but also clear operational guidance.

For `evoke`, this suggests two worthwhile directions:

- better performance-path observability and explainability
- clearer documentation around read/write tradeoffs and future mutable-index
  research

### 5. Query-builder ergonomics are stronger than its operator surface alone

The best ParadeDB API ideas are not the symbols themselves. They are:

- explicit typed query-builder functions
- index-bound query wrappers like `with_index(...)`
- a broader query vocabulary without overloading one operator too much

This matters because corpus-dependent ranking semantics are easier to reason
about when the query can explicitly name or bind an index context.

This aligns closely with a future `evoke` direction:

- prepared query values
- index-bound query values
- richer but still explicit query construction

## Not Suitable for This Project

### 1. Do not replace the `bm25s`-aligned core

ParadeDB's engine is built around Tantivy and a different storage and
execution model. That is not a small implementation detail. It is a
different system.

`evoke` should not absorb this by degrees until it silently stops being a
`bm25s`-aligned PostgreSQL port.

### 2. Do not adopt the full custom-scan takeover model

ParadeDB uses custom scans much more aggressively than `evoke`.
That makes sense for ParadeDB's goals, but it would be too disruptive as a
default direction for this project.

`evoke` should learn from ParadeDB's filtered Top-K and planner
observability ideas, but it should not blindly inherit a large custom scan
stack unless there is a clear, measured gain that the current AM path cannot
achieve.

### 3. Do not adopt the covering-index model as the default

ParadeDB's covering BM25 index is a coherent choice for its system, but it
comes with strong schema coupling and reindex requirements.

That model is not obviously wrong, but it is not an automatic fit for
`evoke`, whose current strength is a narrower, more explicit, and more
portable `bm25s`-aligned design.

### 4. Do not adopt operator semantics that depend on planner rewriting

Some ParadeDB operators are really planner-rewrite entry points and panic if
they are evaluated directly in the wrong shape.

That is acceptable inside ParadeDB's architecture, but it is not a good
semantic model for `evoke`. Our SQL surfaces should stay explicit about
which path is canonical and should degrade in a controlled, documented way
rather than rely on planner-only magic.

### 5. Do not let external tokenizer ecosystems become mandatory

ParadeDB's tokenizer surface is broad and powerful. That is a strength.
But it comes with a heavier dependency story.

`evoke` can learn from the idea of tokenizer profiles and richer
normalization objects without making a large external tokenizer stack a hard
dependency.

## Net Assessment For `evoke`

ParadeDB is valuable mainly in four areas:

1. API and syntax design
2. filtered Top-K execution priorities
3. planner/explain observability
4. operational maturity around read/write/search tradeoffs

The most useful borrowings are therefore:

- prepared and index-bound query values
- clearer layering between convenience syntax and canonical query objects
- broader native filtered ranked scans
- better explainability for fast paths
- optional tokenizer-profile style APIs
- future research on mutable/online-update storage only as an explicit
  storage-revision track

The main things to reject are:

- replacing the current C retrieval core
- importing a full custom scan takeover as the default plan
- adopting a covering-index model as the new baseline
- treating planner rewrite tricks as the core SQL semantic contract

## Resulting Direction

After comparing ParadeDB to the current `evoke` design, the right
strategy is:

- keep the existing `bm25s`-aligned C core
- keep `rowset` as the canonical exact BM25 retrieval path
- continue improving native ranked and filtered scans inside the current AM
- improve the SQL API and query-value model
- improve observability and performance-path documentation
- treat mutable index/storage redesign as explicit future research, not as a
  silent migration

That is the part of ParadeDB that is worth learning from.
