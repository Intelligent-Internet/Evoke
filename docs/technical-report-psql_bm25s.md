# psql_bm25s Technical Report

Date: 2026-05-07

This report is frozen evidence. It is not the current product contract or a
current semantic-performance claim. A revised report requires a deliberately
rerun experiment matrix and current raw artifacts.

For the independent Beta 1 successor covering the complete BM25 plus Sparse Semantic Retrieval (SSR) system, read the
[Evoke System Technical Report](technical-report-ii42-system.md). Historical
measurements and implementation descriptions below retain their original scope;
they are not current operating instructions.

Scope note (2026-07-26): this report is the lexical BM25 foundation and
mutable-index design record. The current product additionally supports
`sae = true` as one relation-owned unified posting index. For the current
single API and lifecycle contract, read
[Architecture and Design](architecture-and-design.md),
[API Reference](api-reference.md), and the
[semantic index quickstart](examples/semantic-index-quickstart.md).

## Abstract

`ii42` is a PostgreSQL extension for exact BM25-family lexical
retrieval. The project is inspired by the `bm25s` technical report and
its central observation: much of BM25 retrieval cost can be moved from
query time to index time by eagerly computing per-term document
contributions and storing them in sparse form. This project keeps that
core ranking contract where it matters most, then extends it into a
database-native setting with native PostgreSQL index storage, SQL
query surfaces, crash recovery, physical replication, and maintenance
mechanisms for mutable workloads. This draft describes the algorithmic
foundation inherited from `bm25s`, the PostgreSQL-specific design
choices in `ii42`, and the main architectural enhancement over
the original static-corpus model: a more maintainable storage and
maintenance design for frequent `INSERT`, `UPDATE`, and `DELETE`. It
also describes the mainline fusion surfaces used by production search
systems: multicolumn BM25 indexes, explicit multi-index score fusion,
field-aware query surfaces, and SQL-level composition with external
vector indexes.

## 1. Introduction

Lexical retrieval remains important because it is exact, transparent,
training-free, and operationally predictable. The `bm25s` report shows
that a Python implementation can become dramatically faster if it
reformulates BM25 retrieval around eager sparse scoring rather than
computing token contributions only at query time.

That result is important, but a database extension faces an additional
set of constraints that are mostly outside the scope of the original
report:

- indexes must live inside PostgreSQL storage
- writes must interact with transactions
- restart and crash recovery must preserve index state
- physical replication must reproduce index state correctly
- SQL users need explicit query surfaces and operational controls
- row churn must be handled without collapsing read performance

`ii42` is designed around those database constraints while
keeping the BM25 scoring model aligned with the `bm25s` family.

## 2. Algorithmic Foundation from BM25S

The `bm25s` report describes two core implementation ideas that are
directly relevant here.

We use the following notation:

- $C$: indexed collection
- $N = |C|$: number of indexed documents
- $D \in C$: a document
- $|D|$: document length
- $L_{\mathrm{avg}}$: average document length
- $Q = (q_1, \dots, q_{|Q|})$: tokenized query
- $\mathrm{df}(t)$: document frequency of token $t$
- $\mathrm{tf}(t, D)$: term frequency of token $t$ in document $D$
- $m_t$: postings length of token $t$, equal to
  $\mathrm{df}(t)$ in the sparse index
- $L = \sum_{D \in C} |D|$: total corpus token count
- $U = \sum_{D \in C} u_D$: total number of distinct term-document
  pairs, where $u_D$ is the number of unique terms in document $D$

### 2.1 Eager sparse scoring

Instead of waiting for a query to arrive and then computing all token
contributions on demand, `bm25s` precomputes the score contribution of
each indexed term for each document at index time. Terms that do not
occur in a document contribute zero for the sparse variants, so the
result can be stored sparsely.

At retrieval time, a query is evaluated by:

- selecting the sparse rows corresponding to query terms
- summing those rows across the query dimension
- applying top-k selection to the resulting document score vector

This changes the hot path from repeated BM25 arithmetic toward sparse
row access plus summation.

### 2.2 Sparse storage and top-k retrieval

The `bm25s` report uses sparse matrix storage and specifically notes
the practicality of CSC-style storage for efficient access. The report
also emphasizes that top-k selection does not need a full sort; a
selection strategy such as partial partitioning is sufficient for the
exact top-k set.

These two ideas are central to the lexical foundation described in this report:

- sparse storage is the core representation;
- BM25 retrieval preserves exact score and top-k semantics;
- the hot path uses sparse access rather than SQL/SPI scoring.

The later Evoke semantic accelerator is a different, bounded-candidate execution
contract. Its design and separately dated qualification evidence belong to the
[Evoke system report](technical-report-ii42-system.md), not to the lexical
benchmark results in this historical report.

### 2.3 Variant support and non-occurrence adjustments

The `bm25s` report also describes how sparsity extends naturally to
Lucene, Robertson, and ATIRE style variants, and how BM25L and BM25+
can still be supported exactly by shifting scores relative to a
non-occurrence baseline. That observation matters because it means the
project can preserve exact variant semantics without giving up sparse
retrieval.

`ii42` keeps this alignment for the supported variants:

- `robertson`
- `lucene`
- `atire`
- `bm25l`
- `bm25+`

### 2.4 Scoring equations

Following the `bm25s` report, the total query score is:

$$
B(Q, D) = \sum_{i=1}^{|Q|} S(q_i, D)
$$

with

$$
S(t, D) = \mathrm{IDF}(t, C) \cdot \mathrm{TFC}(t, D)
$$

For Lucene-style BM25, the current implementation uses:

$$
\mathrm{IDF}_{\mathrm{lucene}}(t, C) =
\log\left(1 + \frac{N - \mathrm{df}(t) + 0.5}{\mathrm{df}(t) + 0.5}\right)
$$

$$
\mathrm{TFC}_{\mathrm{lucene}}(t, D) =
\frac{\mathrm{tf}(t, D)}
{\mathrm{tf}(t, D) + k_1
\left(1 - b + b \cdot \frac{|D|}{L_{\mathrm{avg}}}\right)}
$$

This is the normalized Lucene-style term factor used by the current
code path. The ATIRE branch keeps the more familiar explicit
$(k_1 + 1)$ numerator.

For BM25L and BM25+, sparsity is preserved by shifting against a
non-occurrence baseline. Define:

$$
S_{\theta}(t) = S(t, \varnothing)
$$

where $\varnothing$ is an empty document. Then define the
differential score:

$$
S_{\Delta}(t, D) = S(t, D) - S_{\theta}(t)
$$

and rewrite the total score as:

```math
B(Q, D) =
\sum_{i=1}^{|Q|} S_{\Delta}(q_i, D)
+ \sum_{i=1}^{|Q|} S_{\theta}(q_i)
```

The second term depends only on the query, not on the document, so it
does not change the ordering of documents within that query.

**Proposition 1.**
For a fixed query `Q`, replacing the per-term score `S(t, D)` by the
differential score `S_Δ(t, D)` and adding back the query-only constant
sum of non-occurrence baselines preserves the exact ranking order of
all documents.

**Proof.**
Let:

```math
K(Q) = \sum_{i=1}^{|Q|} S_{\theta}(q_i)
```

Then for every document `D`:

```math
B(Q, D) = B_{\Delta}(Q, D) + K(Q)
```

where:

```math
B_{\Delta}(Q, D) = \sum_{i=1}^{|Q|} S_{\Delta}(q_i, D)
```

For any two documents `D_a` and `D_b`:

```math
B(Q, D_a) \ge B(Q, D_b)
\iff
B_{\Delta}(Q, D_a) + K(Q) \ge B_{\Delta}(Q, D_b) + K(Q)
```

Subtracting the same constant `K(Q)` from both sides yields:

```math
B_{\Delta}(Q, D_a) \ge B_{\Delta}(Q, D_b)
```

So the ordering is unchanged. This is why BM25L and BM25+ can remain
exact while still using sparse differential storage.

### 2.5 Complexity inherited from eager sparse scoring

In the current implementation, term counting inside one document sorts
the document token IDs and collapses adjacent equal values. Therefore,
if the corpus contains documents $D_1, \dots, D_N$, the document-
local counting cost is:

$$
\sum_{i=1}^{N} O(|D_i| \log |D_i|)
$$

The passes that compute corpus statistics and write CSC-style sparse
storage are linear in the number of distinct term-document pairs:

$$
O(U)
$$

So the practical index-build complexity is:

$$
O\left(\sum_{i=1}^{N} |D_i| \log |D_i| + U\right)
$$

and the storage complexity is:

$$
O(U + N + |V|)
$$

where $|V|$ is vocabulary size.

For a query $Q$, the sparse accumulation work over postings is:

$$
O\left(\sum_{t \in Q'} m_t\right)
$$

where $Q'$ is the filtered in-vocabulary query.

However, the exact current implementation also allocates and zeroes a
full score buffer of length $N$, so the practical query-time score
construction cost is:

$$
O\left(N + \sum_{t \in Q'} m_t\right)
$$

The current exact top-k routine uses a bounded heap of size $k$, so
selection is:

$$
O(N \log k)
$$

Therefore the practical canonical retrieval complexity is:

$$
O\left(
N + \sum_{t \in Q'} m_t + N \log k
\right)
$$

**Proposition 2.**
The bounded-heap top-k selection used by the current implementation
returns the exact top-$k$ set under the implementation's total order:
descending score, then ascending document ID as the deterministic
tie-break. It does not need to fully sort all $N$ documents.

**Proof sketch.**
Scan the document scores once while maintaining a min-heap of size at
most $k$.

- While the heap contains fewer than $k$ elements, insert the next
  score.
- Once the heap is full, compare each new score $s$ with the heap
  minimum $h_{\min}$. If $s \le h_{\min}$, discard it. If
  $s > h_{\min}$, remove $h_{\min}$ and insert $s$.

After processing any prefix of the score array, the heap contains the
best $k$ items seen so far under that total order. This is immediate by
induction on the number of processed scores: the invariant is trivially
true before
the heap is full, and once full, any discarded score is no larger than
the smallest retained score, so it cannot belong to the top-$k$ of
the processed prefix. When scores tie, the comparator prefers the
smaller document ID, which makes the retained set deterministic. Any
inserted score strictly improves the retained
set by replacing its current minimum. Therefore, after the full scan,
the heap contains exactly the global top-$k$ scores. A final sort
of those $k$ heap elements produces the exact ranked result.

## 3. Project Scope and Design Goals

The goal of `ii42` is not to reproduce the original Python API.
The goal is to preserve the retrieval contract while embedding it into
PostgreSQL as a first-class index access method.

The main design goals are:

1. exact BM25-family semantics on the canonical retrieval path
2. PostgreSQL-native durability and physical replication
3. SQL-visible indexing and retrieval
4. high read performance as the top priority
5. more maintainable behavior under row churn than a purely static
   offline artifact

The design explicitly accepts that write-side maintenance may be more
expensive if that is required to protect exact reads and stable query
throughput.

## 4. System Architecture

The implementation is organized as four layers.

### 4.1 BM25 core

Files:

- `src/ii42_core.c`
- `src/ii42_core.h`

Responsibilities:

- BM25-family scoring logic
- corpus statistics
- sparse postings and score storage
- exact top-k retrieval
- bounded and subset-aware collector paths for ordered scans
- exact-score rescoring support for mutable maintenance

This is the layer that stays closest to the `bm25s` retrieval model.

### 4.2 Stable binary storage

Files:

- `src/ii42_storage.c`

Responsibilities:

- portable little-endian serialization
- storage for the internal `ii42_index` SQL type
- storage for payloads inside PostgreSQL index relations

The mainline design extends the earlier single-payload model with the
exact statistics needed for more maintainable mutable workloads:

- per-posting term frequency
- per-document document length
- per-term document frequency

These are not a change in retrieval semantics. They are extra state
needed to keep exact scoring possible while maintenance becomes more
database-friendly.

### 4.3 PostgreSQL bindings

Files:

- `src/ii42_pg.c`

Responsibilities:

- SQL type I/O
- standalone builder helpers
- standalone top-k helpers
- utility functions around the serialized storage type

The extension currently supports five indexed source-column types:

- `int4[]`
- `text[]`
- `varchar[]`
- `text`
- `varchar`

These fall into two groups:

- pretokenized inputs supplied by the caller:
  - `int4[]`
  - `text[]`
  - `varchar[]`
- scalar text inputs tokenized at the index boundary:
  - `text`
  - `varchar`

This keeps one BM25 storage and scoring core while still allowing
ordinary SQL schemas to start from raw text columns. The pretokenized
paths remain the more explicit and more performance-oriented contract
when the application already owns tokenization or token-ID assignment.

### 4.4 PostgreSQL access method

Files:

- `src/ii42_am.c`

Responsibilities:

- `CREATE INDEX USING ii42`
- canonical exact retrieval over stored indexes
- ordered scans through `<=>`
- predicate scans through `@@`
- candidate-set planning for boolean, verified, and phrase queries
- candidate-restricted ranking and deferred ranking for large subsets
- multicolumn fusion and field-aware query helpers
- phrase and grouped-query pruning before heap verification
- mutable-workload maintenance
- maintenance introspection and recommendation

For `int4[]`, `text[]`, and `varchar[]`, the access method consumes the
stored token or token-ID stream directly. For scalar `text` and
`varchar`, it normalizes and tokenizes at the index boundary, then feeds
the same internal token-stream model to the BM25 core. Phrase and
verified text paths on scalar columns re-tokenize the heap value using
the index-bound text options before final verification; array-backed
paths avoid that retokenization step.

The same boundary rule is also used for multicolumn fusion indexes.
Homogeneous `text[]`, `varchar[]`, `text`, and `varchar` index columns
can be fused into one BM25 document in index-column order. For scalar
fusion, each source column is tokenized with the index text options
before the fused token stream is appended to the shared BM25 build path.

## 5. PostgreSQL Adaptation of the Retrieval Model

### 5.1 Native index relation

Unlike an offline matrix stored outside the database, `ii42`
stores its state inside a PostgreSQL index relation. This makes the
index participate in ordinary PostgreSQL durability mechanisms:

- WAL logging
- crash recovery
- restart persistence
- physical replication

This is the main structural step from a fast retrieval library toward a
database index.

### 5.2 Canonical retrieval APIs

The extension defines two canonical exact retrieval surfaces:

- `ii42_query_ids(...)`
- `ii42_query_tokens(...)`

These functions are the clearest definition of the exact BM25 contract
in this project and are the primary performance path. Other SQL
surfaces exist for usability, but the `rowset` functions are the
right reference point for exactness and benchmarking.

Even when an index is built over scalar `text` or `varchar`, the index
still lowers the source column to the same internal token-stream model.
Raw-text search helpers exist for ergonomics, but the ranking core
remains token- and corpus-statistics-driven.

This separation was reinforced by the broader research notes: richer
operator surfaces are useful, but the system still needs one explicit
canonical retrieval contract.

### 5.3 SQL convenience surfaces

On top of the canonical path, the extension adds:

- raw-query retrieval
- grouped boolean parsing
- prefix handling
- bounded phrase verification
- document-match predicates through `@@`
- ordered scan integration through `<=>`
- tokenization, normalization, stemming, and highlighting helpers

These surfaces are designed so that convenience features do not replace
the exact canonical path as the semantic center of the system.

The emphasis on SQL-visible ranked scans and operator ergonomics was
strengthened by studying mature PostgreSQL search systems. The project
kept those ideas, but attached them to the native access-method path
instead of a SQL- or planner-heavy retrieval core.

### 5.4 Fusion and hybrid SQL composition

The mainline extension supports two BM25 fusion patterns.

The first pattern is a multicolumn fusion index. A homogeneous
multicolumn index over `text[]`, `varchar[]`, `text`, or `varchar`
columns can be treated as one BM25 document. The indexed columns are
fused in index-definition order, scalar text columns are tokenized with
the index text options, and the resulting token stream enters the same
BM25 build path as a single-column document. This is a storage and
query-surface convenience; it does not change the BM25 scoring formula.

The second pattern is public score fusion across separate indexes. Functions
such as `ii42_fusion_query(...)` and `ii42_fusion_query_fields(...)` let
applications, regression suites, and benchmark harnesses
retrieve field-specific candidates from independent BM25 indexes,
apply query-time weights, and return one ranked result set. This is a
composition layer above independently maintained indexes; one-index queries
still use `ii42_query(...)`.

For `field_aware = true` multicolumn indexes, one posting payload preserves
field-scoped lexical terms and, in `sae = true` / SSR mode, field-scoped
semantic atoms.
Applications query all indexed fields with equal weight or use the public
`ii42_query(...)` overload for explicit whole-field weights. Each weight
scales the field's combined BM25 and semantic evidence in one native
accumulator.
Semantic admission applies the configured budget ratio per field before the
retained atoms are merged, so fields do not compete for budget before their
query-time weights are known.
This is still one PostgreSQL-native index, not a separate BM25F engine and not
multiple hidden sub-indexes.

The public hybrid fusion surfaces also keep historical BM25/vector comparisons
reproducible. `ii42` does not implement dense
vector indexing; that role belongs to pgvector, VectorChord, or another
vector access method. A product SQL statement can therefore combine:

- relational filters
- BM25 candidates from `ii42`
- semantic candidates from a vector index
- SQL-side normalization, gating, or weighted score fusion
- a final ordered candidate set for comparison or AI reranking

This remains useful for controlled AI and RAG comparisons, but it is
not the semantic-enabled product route. An SSR index (`sae = true`) compiles
lexical and model evidence into one relation-owned unified posting
generation, which applications query through `ii42_query(...)`.

The detailed user-facing behavior is documented in
[Multicolumn Indexes](multicolumn-indexes.md),
[Multi-Index Fusion](multi-index-fusion.md), and
[Field-Aware Indexes](field-aware-indexes.md).

### 5.5 Canonical query complexity inside PostgreSQL

The PostgreSQL embedding preserves the same native sparse scoring core,
but adds database-specific fixed costs:

- SQL argument unmarshalling
- relation and cache validation
- MVCC visibility checks
- tuple materialization for SQL return values

Those costs are intentionally kept outside the ranking core. The exact
owner-only `rowset` diagnostics therefore still follow the same
algorithmic structure as the standalone core:

$$
O\left(
N + \sum_{t \in Q'} m_t + N \log k
\right)
$$

plus PostgreSQL-specific constant factors for executor interaction and
result materialization.

### 5.6 Execution specialization without semantic drift

The formulas above define the retrieval contract. The current project
does not change those formulas in order to get speedups. The newer
performance work instead specializes the execution paths that decide
which documents reach ranking and how much ranking work is actually
performed.

The main practical patterns are:

- bounded collectors on ordered and deferred-ranking paths
- candidate-restricted ranking when filters or grouped queries already
  expose a narrow candidate set
- narrower boolean planning for common `MUST`, `MUST OR`, and positive
  AST shapes
- phrase and verified-query paths that prune candidate sets before
  verification instead of ranking broad positive supersets

These changes matter because PostgreSQL convenience surfaces are not all
the same shape. A small phrase query, a filtered ordered scan, and a
full-corpus exact `rowset` lookup all preserve the same BM25-family
semantics, but they should not necessarily pay the same planning or
ranking costs.

So the design rule is:

- keep the canonical BM25 score definition fixed
- narrow candidate sets as early as correctness allows
- use bounded ranking paths when the query shape already imposes a
  natural bound

This is the main execution-design refinement that emerged after the
initial mainline implementation.

This was also the clearest practical lesson absorbed from studying
mature search systems: the largest wins came from bounded collectors,
earlier candidate narrowing, and query-shape-aware execution, not from
changing the BM25 formulas themselves.

This distinction explains a recurring benchmark pattern: the extension
remains strong on larger or heavier workloads, while fixed database
overheads become more visible on lighter, very high-QPS workloads.

## 6. Mainline Enhancement: Maintainable Mutable Indexes

The most important architectural enhancement over the original static
`bm25s` storage model is mutable-workload maintenance.

### 6.1 Problem

The original eager sparse idea is strongest when the corpus is stable.
Inside PostgreSQL, however, indexes must survive frequent:

- `INSERT`
- `UPDATE`
- `DELETE`

If every change forced a full rebuild immediately, operational cost
would be too high. If changes were ignored, the index would stop
behaving like a real database index.

### 6.2 Exact-stats payload

To support more flexible maintenance without changing BM25 semantics,
the payload stores enough exact statistics to support exact rescoring
and exact overlays:

- posting-level term frequency
- document length
- term-level document frequency

This is the key storage extension that turns a static sparse payload
into something that can support exact maintenance decisions.

### 6.3 Persisted maintenance metadata

The index metapage tracks maintenance state such as:

- pending writes
- pending deletes
- rebuild count

This metadata lets the index expose its state through SQL and survive
restart or replication with the rest of the relation.

### 6.4 Frontier-bounded deferred maintenance

The mainline index can run in either the manual consistency policy or
automatic maintenance policies. Automatic v3 maintenance is bounded by fixed
active and pending linked-L0 frontiers. Exact reads combine immutable extents
with the snapshot-visible frontier; maintenance timing never hides committed
lexical state.

For stale-tolerant knowledge-base workloads, the same maintenance model can be
configured with an opt-in query-first eventual policy:

- `consistency = 'eventual'`

This policy keeps foreground queries from triggering unbounded rebuild work.
Queries read immutable extents and the snapshot-visible linked L0 as one exact
surface. Work hints, periodic reconciliation, and one bounded action per worker
round control convergence without a foreground cache heuristic.

Semantic-enabled eventual indexes specialize this contract without introducing
a side index. A writer appends exact lexical postings plus a durable
semantic-pending mutation identity and performs no document-model inference.
Search may be lexical-first briefly. The ordinary maintenance worker completes
semantic postings in bounded batches, attaches each completion only to its
exact tuple/version mutation, and then performs larger generation compaction.
Base, pending lexical records, semantic completions, and tombstones remain one
logical unified query surface throughout progressive convergence.

The recommended automatic deployment path loads `ii42` through
`shared_preload_libraries`. In that mode, a timer-based generic catch-up
supervisor wakes bounded workers, each worker maintains at most one due
index, and advisory locks prevent duplicate work on the same index.
Deployments without shared preload can still use lightweight touch
wakeups or explicit maintenance helpers. PostgreSQL `VACUUM` remains
part of the normal delete-cleanup lifecycle, but corpus-statistics
convergence is owned by the maintenance path rather than by manual
VACUUM as a synchronization mechanism.

The concrete `CREATE INDEX ... WITH (...)` parameters are documented in
[Index Parameters](index-parameters.md).

The broader decision to make mutable behavior policy-bounded and
explicitly database-facing was strengthened by the research notes,
which highlighted that maintenance strategy and read performance cannot
be treated as independent concerns.

### 6.5 Exact base-plus-delta overlays

The project extends the static payload model with exact bounded
overlays for:

- inserted rows
- updated rows
- delete tombstone state from PostgreSQL heap-cleanup paths

Canonical reads remain exact by combining:

- the persisted base payload
- bounded deferred delta state
- visibility-aware delete handling

This keeps the read path exact without forcing immediate full rebuilds
for every small mutation. This statement describes the canonical packed and
page-native authority. The default semantic accelerator is a derived
bounded-approximate route: a compatible stale baseline revalidates old
candidates but may omit post-baseline additions until a high-water or periodic
low-debt refresh.

### 6.6 Operational introspection

The extension exposes maintenance state and policy helpers, including:

- `ii42_index_details(regclass)`
- `ii42_index_policy_recommend(regclass, profile text)`
- `ii42_index_refresh(regclass)`
- `ii42_index_maintain(regclass)`
- `ii42_index_try_maintain(regclass)`
- `ii42_index_maintain_due(max_indexes integer DEFAULT 1)`

This is an explicitly database-oriented addition. The intent is to make
index maintenance observable and tunable rather than hidden.

For query-first eventual indexes, the non-blocking maintenance path stages a
replacement payload before taking the final swap lock. The swap is conditional:
if an in-flight writer exists or the lock is busy, the scheduler retries later.
Append-only delta records that arrive while the replacement is being built are
carried forward after the swap. Non-tail-compatible metapage changes still
discard the staged payload. This keeps the long heap scan and BM25 build out of
foreground query and write paths while preserving a valid previous base index.

That operational emphasis was reinforced by studying mutable search
indexes, which showed that operational systems need clear
maintenance-state visibility rather than opaque background behavior.

### 6.7 Exactness of the mutable overlay design

The mutable-maintenance layer can be stated as a simple correctness
proposition.

**Proposition 3.**
If the persisted base payload represents collection
$C_{\mathrm{base}}$, the delta area represents all pending inserted
or updated documents, and the tombstone/replacement bookkeeping removes
all deleted or superseded documents, then the overlay index built from
those components is equivalent to rebuilding from scratch over the
current visible collection $C_{\mathrm{current}}$.

**Proof sketch.**
The overlay construction reconstructs the base corpus, appends every
pending delta document, and excludes any tuple marked deleted or
replaced by a newer visible version. The resulting document multiset is
exactly the visible corpus. The overlay index is then rebuilt using the
same deterministic BM25-family build routine and the same parameter
tuple. Therefore the resulting corpus statistics, sparse payload, and
query scores are exactly those of a fresh rebuild over
$C_{\mathrm{current}}$.

This is why the maintenance design is described as an enhancement in
maintainability rather than a change in scoring semantics.

### 6.8 Complexity of the mutable enhancements

The mutable-maintenance additions introduce two kinds of cost:

1. incremental write bookkeeping
2. occasional exact overlay or consolidation work

#### 6.8.1 Incremental write cost

For an inserted or updated document $D$, the deferred write path
stores a delta record containing the new token array plus heap-TID
metadata. At the logical level, this bookkeeping is linear in the
document representation:

$$
O(|D|)
$$

This is substantially cheaper than forcing an eager full rebuild over
the corpus:

$$
O\left(\sum_{i=1}^{N} |D_i| \log |D_i| + U\right)
$$

#### 6.8.2 Overlay construction cost

When deferred maintenance remains within configured bounds, the first
exact read may build an overlay from:

- the persisted base payload
- the persisted delta records
- the tombstone/replacement set

Let:

- $N_b$: number of base documents
- $U_b$: base sparse nonzero count
- $L_b$: total token count reconstructed from the base payload
- $N_{\Delta}$: number of pending delta documents
- $L_{\Delta}$: total token count of delta documents

The overlay path must:

1. load deferred state
2. reconstruct the base corpus from the sparse payload
3. exclude deleted or replaced documents
4. rebuild an exact index on the resulting visible corpus

The base reconstruction step is not free. In the current code it
allocates explicit document arrays from `doc_lengths`, then walks the
CSC postings and expands each posting by its stored term frequency.
That contributes at least:

$$
O(N_b + U_b + L_b)
$$

The subsequent exact rebuild over the visible overlay corpus costs:

```math
O\left(
\sum_{D \in C_{\mathrm{overlay}}} |D| \log |D|
+ U_{\mathrm{overlay}}
\right)
```

So a safer practical bound for the full overlay path is:

```math
O\left(
N_b + U_b + L_b
+ \sum_{D \in C_{\mathrm{overlay}}} |D| \log |D|
+ U_{\mathrm{overlay}}
\right)
```

where $C_{\mathrm{overlay}}$ is the visible corpus after applying
deltas and tombstones.

In practice, the rebuild term is usually the dominant one for ordinary
document lengths, but the reconstruction term is still part of the real
cost and should be stated explicitly. This is exactly why the design
bounds deferred maintenance by policy. Exact overlays are correct, but
they must remain bounded to stay operationally useful.

The current page-native product path no longer executes this
full-corpus overlay. The compacted base carries atom-oriented metadata and the
relation-owned linked L0 carries lexical-pending, semantic-completion, and
retirement records. Queries pin one checked root, traverse immutable posting
cursors, and project only matching snapshot-visible L0 records without
enumerating all base documents or building a complete delta image. The
full-overlay implementation is retained only as a differential test oracle.

#### 6.8.3 Physical frontiers as the bound

V3 does not expose a rebuild threshold. Fixed page and record ceilings bound
each active and pending linked-L0 frontier. Maintenance rotates a nonempty
active frontier, seals the transaction-safe pending frontier, and performs at
most one structural or semantic action per selected worker round. This turns a
potentially unbounded maintenance problem into a bounded state machine without
making correctness depend on operator tuning.

### 6.9 Storage overhead of exact-stats support

Compared with a minimal sparse payload containing only:

- sparse scores
- document indices
- CSC pointers

the enhanced payload adds:

- one term-frequency integer per sparse nonzero
- one document-length integer per document
- one document-frequency integer per vocabulary term

Asymptotically this does not change the storage class:

$$
O(U + N + |V|)
$$

It increases the constant factors, but not the order. This is an
important design property: the project pays extra space for exact
mutable maintenance, but it does not leave the sparse-index regime.

## 7. Correctness, Durability, and Replication

The design treats PostgreSQL durability as a first-class requirement.

Current mainline behavior includes:

- persisted index pages in PostgreSQL storage
- WAL-logged index updates
- restart-safe maintenance metadata
- crash-recovery coverage
- physical replication coverage

The project also includes stress and smoke validation around:

- restart
- crash recovery
- physical replication
- concurrent maintenance behavior

This is a major difference from treating the retrieval structure as an
external artifact managed outside the database.

## 8. Performance Model

The project keeps the same general performance philosophy as `bm25s`:

- do expensive scoring work ahead of query time where possible
- keep the hot retrieval path in native code
- preserve exact top-k as the default behavior

At the same time, the database setting introduces overheads that do not
exist in a standalone retrieval library:

- executor and function-call overhead
- SQL marshalling and tuple materialization
- MVCC visibility checks
- maintenance coordination

The implementation therefore separates performance concerns into two
categories:

1. canonical read performance on `rowset` retrieval
2. write/maintenance cost under row churn

The project benchmarks both dimensions in the repository documentation.
Read performance remains the primary optimization target; maintenance is
designed to be more flexible, but not at the expense of collapsing the
exact read path.

### 8.1 Current benchmark authority

The current cross-engine benchmark authority is the refreshed PG18
`15 x 5` BEIR matrix published on `2026-04-02`. It compares:

- upstream Python `bm25s`
- `ii42 ids`
- `ii42 text[]`
- ParadeDB `pg_search`
- TensorChord `vchord_bm25`

Run shape:

- all 15 BEIR datasets used in the BM25S benchmark set
- Google Cloud `us-east4-a`
- `n2-standard-16`
- PostgreSQL `18`
- local uploaded dataset cache
- `top_k = 1000`
- base matrix: `2026-03-31` full `15 x 5`, one dataset-engine task per VM
- refresh matrix: `2026-04-02` `ii42 ids/text[]` rerun on the same
  GCP PG18 shape

The public matrix is intentionally anchored to the pretokenized input
paths:

- `ii42 ids` uses `int4[]`
- `ii42 text[]` uses `text[]`

Scalar `text` and `varchar` source-column support is part of the
extension design, but it is not the basis of the current cross-engine
headline table.

The rolled-up current matrix was checked against the paired raw archives
in:

- `docs/performance/data/raw/pg18-beir-extension-matrix-2026-03-31/`
- `docs/performance/data/raw/pg18-beir-psql-only-tantivy-2026-04-02/`

The refreshed matrix keeps `45/75` upstream / `pg_search` /
`vchord_bm25` cells from the stable `2026-03-31` PG18 run and replaces
`30/75` `ii42` cells with the `2026-04-02` rerun. All `75/75`
dataset-engine cells matched exactly on:

- `stats`
- `build_ms`
- `query.count`
- `query.qps`

This matrix is now the project's authoritative database-engine
comparison. Older same-machine localhost engine-comparison raw files
were removed from the working tree once this PG18 matrix became the
default reference.

### 8.2 Current PG18 cross-engine matrix

Current query-throughput matrix, with dataset size and QPS:

| Dataset | Docs | Queries | upstream `bm25s` QPS | `ii42 ids` QPS | `ii42 text[]` QPS | `pg_search` QPS | `vchord_bm25` QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `arguana` | 8,674 | 1,406 | 1158.34 | 1402.63 | 1112.01 | 115.94 | 78.77 |
| `climate-fever` | 5,416,593 | 1,535 | 3.04 | 57.78 | 50.75 | 2.84 | 5.25 |
| `cqadupstack` | 457,199 | 13,145 | 111.56 | 443.13 | 438.42 | 13.99 | 60.51 |
| `dbpedia-entity` | 4,635,922 | 467 | 3.47 | 128.19 | 91.19 | 5.21 | 23.66 |
| `fever` | 5,416,568 | 123,142 | 3.15 | 97.56 | 80.15 | 5.62 | 12.13 |
| `fiqa` | 57,638 | 6,648 | 810.51 | 1409.52 | 1186.41 | 17.76 | 190.57 |
| `hotpotqa` | 5,233,329 | 97,852 | 4.16 | 55.40 | 49.86 | 3.42 | 9.31 |
| `msmarco` | 8,841,823 | 509,962 | 1.61 | 96.67 | 82.13 | 4.44 | 18.20 |
| `nfcorpus` | 3,633 | 3,237 | 3155.35 | 3373.94 | 3326.96 | 1132.17 | 1252.75 |
| `nq` | 2,681,468 | 3,452 | 10.55 | 174.34 | 176.69 | 6.28 | 21.96 |
| `quora` | 522,931 | 15,000 | 90.56 | 637.98 | 619.64 | 13.26 | 154.36 |
| `scidocs` | 25,657 | 1,000 | 1203.09 | 1835.85 | 1614.92 | 17.89 | 367.04 |
| `scifact` | 5,183 | 1,109 | 2964.86 | 2557.47 | 2240.18 | 500.04 | 629.42 |
| `trec-covid` | 171,332 | 50 | 210.50 | 191.94 | 154.48 | 8.75 | 75.66 |
| `webis-touche2020` | 382,545 | 49 | 240.36 | 82.97 | 74.10 | 8.14 | 86.04 |

Scale-versus-throughput view:

![QPS vs dataset scale](performance/reports/pg18-qps-vs-dataset-scale-2026-04-02.svg)

Summary readout:

- `ii42 ids` beat upstream on `12/15` datasets and had the
  strongest median query-throughput ratio at `3.97x`.
- `ii42 text[]` also beat upstream on `11/15` datasets, with a
  `3.93x` median ratio and a higher build cost than `ids`.
- `pg_search` beat upstream on `3/15` datasets and had the weakest
  suite median at `0.17x`.
- `vchord_bm25` beat upstream on `7/15` datasets and materially
  outperformed `pg_search`, but still trailed both `ii42` paths
  on median throughput.

Index construction should be read alongside query throughput, because
the build side determines how expensive bulk loads, refreshes, and
repeatable deployments are. The next matrix reports index-construction
time only; it excludes query execution and orchestration overhead.

Current build-time matrix:

| Dataset | Docs | Queries | upstream `bm25s` build | `ii42 ids` build | `ii42 text[]` build | `pg_search` build | `vchord_bm25` build |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `nfcorpus` | 3,633 | 3,237 | 167ms | 82ms | 113ms | 254ms | 326ms |
| `scifact` | 5,183 | 1,109 | 224ms | 99ms | 150ms | 326ms | 449ms |
| `arguana` | 8,674 | 1,406 | 318ms | 101ms | 168ms | 376ms | 448ms |
| `scidocs` | 25,657 | 1,000 | 974ms | 395ms | 633ms | 490ms | 1.10s |
| `fiqa` | 57,638 | 6,648 | 1.88s | 672ms | 1.06s | 705ms | 1.32s |
| `trec-covid` | 171,332 | 50 | 6.67s | 2.65s | 4.61s | 2.86s | 3.80s |
| `webis-touche2020` | 382,545 | 49 | 21.18s | 9.84s | 16.51s | 10.65s | 13.39s |
| `cqadupstack` | 457,199 | 13,145 | 15.68s | 5.83s | 9.29s | 7.21s | 19.56s |
| `quora` | 522,931 | 15,000 | 5.80s | 788ms | 1.06s | 936ms | 1.69s |
| `nq` | 2,681,468 | 3,452 | 1.2m | 23.71s | 40.80s | 31.46s | 55.57s |
| `dbpedia-entity` | 4,635,922 | 467 | 1.7m | 26.37s | 45.05s | 40.73s | 1.9m |
| `hotpotqa` | 5,233,329 | 97,852 | 1.8m | 30.79s | 50.38s | 41.97s | 1.9m |
| `fever` | 5,416,568 | 123,142 | 2.6m | 52.83s | 1.6m | 1.2m | 2.7m |
| `climate-fever` | 5,416,593 | 1,535 | 2.7m | 52.74s | 1.5m | 1.2m | 2.5m |
| `msmarco` | 8,841,823 | 509,962 | 3.2m | 56.06s | 1.5m | 1.2m | 1.7m |

Scale-versus-build-time view:

![Index build time vs dataset scale](performance/reports/pg18-build-vs-dataset-scale-2026-04-02.svg)

Build-side readout:

- `ii42 ids` was faster than upstream on `15/15` datasets and had
  the strongest build-side median ratio at `0.34x` of upstream.
- `ii42 text[]` was also faster than upstream on `15/15`
  datasets, with a `0.56x` median build ratio.
- `pg_search` beat upstream on `12/15` datasets and remained
  competitive on build time, even though it lagged badly on query
  throughput.
- `vchord_bm25` beat upstream on `7/15` datasets. Its build profile was
  mixed overall and drifted back toward or above upstream on several
  larger datasets.

Total build-time summary:

- upstream Python `bm25s`: `848046.35 ms`
- `ii42 ids`: `262955.79 ms`
- `ii42 text[]`: `443975.35 ms`
- `pg_search`: `356944.25 ms`
- `vchord_bm25`: `739014.63 ms`

Current relevance-quality matrix:

The quality matrix is a local PG18 relevance run over the same 15 BEIR
datasets. It measures `NDCG@10`, `MAP@100`, `Recall@100`, and
`Precision@10` with `top_k = 100`.

This table uses qrels-bearing queries only. That means the evaluated
query count can be smaller than the full query count shown in the QPS
table.

The local machine had all five comparison engines available for this
relevance run: upstream Python `bm25s`, `ii42 ids`,
`ii42 text[]`, `pg_search`, and `vchord_bm25`.

The primary chart is an absolute-score heatmap rather than a
dataset-scale line chart. Each cell is the metric value for one engine
on one dataset. Bold cells are within `0.001` of the best engine for
that dataset and metric.

![Quality score heatmap](performance/reports/pg18-quality-score-heatmap-2026-05-06.svg)

For the complete per-dataset metric table, see the
[performance benchmark reference](performance/README.md#quality-matrix).

Quality readout:

- All five engines sit in a close relevance band on this BM25 quality
  matrix. Average `NDCG@10` ranges from `0.3976` to `0.4019` across
  the compared engines.
- `ii42 ids` and `ii42 text[]` remain quality-neutral exact
  PostgreSQL paths in this run. Their largest absolute metric
  difference from the Python reference implementation is below
  `0.0030`, while their engineering cost profile is covered by the QPS,
  build-time, and index-size matrices.
- `pg_search` and `vchord_bm25` are also competitive on relevance in
  this local quality run, but they have different throughput and
  storage trade-offs in the main performance matrix.
- These quality metrics do not replace the QPS, build-time, or
  index-size matrices. They only show that the compared engines are
  retrieving similarly relevant top-100 candidates under the BEIR qrels.

## 9. Current Limitations

The present design still has boundaries.

- Canonical exact retrieval is strongest on the explicit `rowset`
  APIs. SQL convenience layers are broader but not always equally cheap.
- `<=>` only matches the exact BM25 ordering when PostgreSQL is
  executing a real `ii42` index scan.
- Post-delete cleanup still follows PostgreSQL's heap lifecycle, but
  exact corpus-statistics convergence is handled by automatic or
  explicit index maintenance. Operators should keep autovacuum and the
  `ii42` maintenance worker path healthy rather than treating
  manual `VACUUM` as the synchronization primitive.
- The current text processing stack is practical and configurable, but
  it is not intended to replace a full external search-text ecosystem.
- Hybrid BM25/vector retrieval is a SQL composition pattern. Dense
  vector indexing is intentionally delegated to vector extensions such
  as pgvector or VectorChord.
- Large-query-set upstream runs can dominate total benchmark wall-clock
  even after the PostgreSQL engines have already completed.

## 10. Conclusion

`ii42` should be understood as a PostgreSQL-native extension that
inherits the most important retrieval insight from `bm25s` and then
extends it for database reality.

The inherited idea is:

- eager sparse scoring with exact BM25-family semantics

The main project-specific contribution is:

- turning that retrieval model into a durable, replicated,
  SQL-addressable, and more maintainable PostgreSQL index

In that sense, `ii42` is not merely a wrapper around `bm25s`.
It is a database-oriented system design that preserves the BM25
contract while adding the storage, maintenance, and operational
properties that a PostgreSQL extension must have.

## 11. Other Engineering Research

This report keeps the main technical narrative focused on the project's
retrieval model, storage design, maintenance strategy, fusion surfaces,
and measured benchmark position. The implementation was still informed
by careful reading of other strong open-source systems.

We are grateful to the maintainers and contributors of these open-source
projects. Their public designs, implementations, benchmarks, and
documentation helped sharpen this project's thinking, and they provided
substantial practical inspiration for many details in API design,
execution-path shaping, maintenance visibility, hybrid retrieval
composition, and performance optimization.

In particular, the project studied
[VectorChord-BM25](https://github.com/tensorchord/VectorChord-bm25),
[ParadeDB](https://github.com/paradedb/paradedb), and
[ParadeDB's Tantivy fork](https://github.com/paradedb/tantivy). The
useful outcome was not a feature-by-feature comparison table. It was a
set of engineering lessons around bounded collectors, candidate
narrowing, mutable-index maintenance, SQL ergonomics, and hybrid
retrieval composition.

## 12. References

- BM25S technical report:
  <https://arxiv.org/abs/2407.03618>
- BM25S codebase:
  <https://github.com/xhluca/bm25s>
- `ii42` source repository: this development tree; the planned public
  repository is not yet provisioned.
- `ii42` performance benchmark reference:
  [Performance and Benchmarks](performance/README.md)
