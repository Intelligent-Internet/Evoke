# Query Semantics

## Canonical Retrieval

For a Sparse Semantic Retrieval (SSR) index, enabled with `sae = true`,
canonical retrieval is ordinary table SQL:

```sql
SELECT d.*,
       evoke_query('docs_body_idx'::regclass, 'query text') AS score
FROM docs AS d
WHERE d.publish_date >= DATE '2026-01-01'
ORDER BY score DESC
LIMIT 20;
```

The stored `sae` option selects exact BM25 or unified lexical/semantic scoring.
Applications do not select a scorer. Semantic-enabled mutation is
eventual-only. Every route pins checked page-native authority. Exact execution
includes snapshot-visible linked L0; the bounded accelerator route may use a
compatible older serving baseline and omit newer additions until convergence.
Returned rows are still revalidated under the current statement snapshot.

The two- and four-argument scalar `evoke_query(...)` overloads are planner
markers. They are never evaluated once per row. One Evoke `CustomScan` evaluates
ordinary PostgreSQL predicates under the statement snapshot, encodes the query
once, and returns subset top-k from the same root and scorer used by the
explicit-hit overloads.

The explicit hit API returns `ctid`, index-local `doc_id`, and `score`. Join by
`ctid` inside the same statement when that API is required:

```sql
SELECT d.id, d.body, hit.score
FROM evoke_query('docs_body_idx'::regclass, 'query text', 20) AS hit
JOIN docs AS d ON d.ctid = hit.ctid
ORDER BY hit.score DESC, d.id;
```

`doc_id` is an internal slot and may change after rebuild. It is not a durable
application identifier.

## BM25 Semantics

BM25 scores come from the selected `method`, `idf_method`, `k1`, `b`, and
`delta` index contract. Omitted query normalization arguments inherit scalar
text index options. Explicit overrides are useful for diagnostics but should
not silently differ from indexing policy.

The extension-owner exact support functions are:

- `evoke_query_ids(...)` for `int4[]`;
- `evoke_query_tokens(...)` for text token arrays.

They are revoked from `PUBLIC`, reject semantic-enabled indexes, and exist for
regression/benchmark isolation rather than application routing.

## Semantic Semantics

For `sae = true`, one query encoder emits weighted atoms defined by the model
checkout. Lexical and semantic postings contribute to one score accumulator
and one top-k result. There is no ANN query or post-retrieval fusion stage.

On `field_aware = true` indexes, lexical and semantic atoms are namespaced per
field. The ordinary `evoke_query(index, query, k)` overload searches every
field with weight `1.0`; the field-aware overload selects fields and computes
`sum(field_weight * (field_BM25 + field_semantic))`. Both forms traverse the same
physical posting index and use the same accumulator.

BM25-only query overrides and `weight_mask` are rejected. Query encoding,
document encoding, atom identity, normalization, and scoring profile must share
the same validated contract. A mismatch makes `evoke_index_status(...)`
non-ready and search fails closed.

A newly committed semantic row may initially have lexical evidence only. This
is a valid progressive state, not a second index. Shared workers later append
semantic completion for the same document version.

During semantic catch-up, the exact page-native route keeps committed-row
membership and lexical evidence snapshot-correct, but the unified score and
top-k may differ from their settled values because matching semantic evidence
is not available yet. The default accelerated route has the additional bounded
approximation described below: a compatible serving generation may omit a
post-baseline insert or improvement until the next threshold-triggered or
periodic refresh.
Both routes revalidate returned candidates against the current COW version,
TID, predicate, and heap snapshot, so deleted, replaced, or otherwise invalid
rows cannot leak from an older accelerator. A semantic completion for an
obsolete row version must never attach.

## MVCC And Freshness

Returned rows follow the statement snapshot:

- aborted and rolled-back writes are invisible;
- committed upserts are visible according to the selected consistency policy;
- superseded and deleted heap tuples cannot be returned;
- `VACUUM (INDEX_CLEANUP ON)` supplies exact dead-TID retirement so corpus
  statistics and reusable storage converge.

One search statement pins its root and heap snapshot while maintenance
publishes a new descendant. A later statement sees the then-current committed
root and visible L0 frontier. Query execution never mixes roots within one
statement.

Evoke does not retain a historical ranking root for every transaction
snapshot. Two statements in one long `REPEATABLE READ` transaction therefore
keep identical heap visibility but may use different current index roots and
corpus statistics after concurrent commits. That can change scores or top-k
membership, but snapshot-invisible tuple versions are still rejected. An
application that requires ranking to remain fixed across statements must
materialize the first statement's results.

## Ordering And Ties

Explicit-hit `evoke_query(..., k, ...)` returns the index top-k and permits an
application-side secondary key. Planner-native scalar `evoke_query(...)`
currently accepts only its score as the sort key; a second key fails closed
rather than changing top-k semantics.

```sql
ORDER BY hit.score DESC, source.id
```

Do not assume `ctid` or `doc_id` is stable across `VACUUM FULL`, table rewrite,
or `REINDEX`.

## PostgreSQL Operators

### `@@`

For `text[]` and `varchar[]`, `document @@ 'query text'` is a boolean
document-match predicate. It does not carry the corpus-level top-k score.

PostgreSQL already defines `text @@ text`; Evoke does not overload that raw
scalar shape. Owner diagnostics can use the prepared-query `@@@` surface.

### `<=>`

`ORDER BY value <=> query ASC LIMIT k` is the planner-visible BM25 ordering
surface. It has full index ranking semantics only when PostgreSQL selects a
real `evoke` index scan. A scalar evaluation outside that scan is a local
distance calculation, not a corpus retrieval oracle.

These operators are BM25-specific and do not dispatch to semantic scoring.

## Prepared Diagnostics And Product Composition

Prepared-query and ranked-query helpers are owner-only diagnostics. The public
`evoke_query(...)` overload provides custom field weighting inside one
field-aware index. Fusion and hybrid helpers are public composition APIs above
one or more independently queried indexes. Neither category changes a source
index's page-native scorer or lifecycle.

- prepared values bind normalization and index context;
- ranked values bundle prepared input, order tokens, `k`, and optional mask;
- field-aware search varies whole-field weights inside one BM25 or SSR index;
- fusion/hybrid APIs combine already retrieved candidate rows.

Application code should prefer planner-native SQL for one semantic index and
use `evoke_query(...)` when it explicitly needs hit rows. Use fusion or hybrid
composition only when the application deliberately combines multiple Evoke
indexes or an Evoke source with another retrieval engine.

## Weight Masks

In BM25 diagnostics, `weight_mask` is defined over the pinned root's document
slot domain. Slots introduced later in linked L0 have no valid mask entry and
are omitted until a descendant root exposes a maskable domain. Current
retirements still apply.

Semantic-enabled search does not accept `weight_mask`.

## Filtering

SSR indexes support predicate-defined filtered top-k through ordinary
PostgreSQL predicates:

```sql
SELECT source.id,
       source.body,
       evoke_query('docs_body_idx'::regclass, 'graph retrieval') AS score
FROM docs AS source
WHERE source.publish_date >= DATE '2026-01-01'
  AND source.categories && ARRAY['cs.LG']
ORDER BY score DESC
LIMIT 20;
```

PostgreSQL owns predicate parsing, permissions, types, MVCC, and final predicate
evaluation. When every ANDed predicate is a built-in `=`, `&&`, `>`, `>=`, `<`,
`<=`, or direct scalar `ILIKE` operation on an Evoke index `INCLUDE` column, the planner
can translate it to the scope-posting format stored with the published semantic
baseline. Evoke then runs one same-root filtered request for up to four times
`k`, fetches those rows under the statement snapshot, and evaluates the original
PostgreSQL qual before returning any row. This avoids materializing a
corpus-scale visible-TID set for broad supported filters.

The scope artifact is serving-state, not a foreground maintenance barrier. If
the root has newer L0 or sealed delta work, planner-native search may continue
using the older published scope baseline. Rows whose current values no longer
match are removed by the PostgreSQL recheck; newly matching or newly inserted
post-baseline rows may be absent until background convergence publishes a newer
baseline. This is the same bounded-staleness authority used by ordinary
accelerated semantic search. Every returned row is verified against the
current statement snapshot, while the candidate universe may be the older
published baseline.

If an expression is unsupported, a predicate column is not in `INCLUDE`, the
scope artifact is unavailable, or the rechecked probe cannot fill the requested
limit, Evoke discards the probe and falls back to complete visible-TID subset
scoring. `EXPLAIN (ANALYZE)` reports scope eligibility, one probe count,
candidate and matching counts, completion, and fallback. The explicit
structured-predicate API below uses the same serving-baseline contract.

### Structured Predicate API

The structured JSON `evoke_query(...)` overload provides explicit subset top-k
workflows. Its supported operations are:

| Operation | JSON shape | Typical use |
| --- | --- | --- |
| `eq` | `{"document_id":{"eq":"..."}}` | document-local chunks, jurisdiction |
| `in` | `{"journal":{"in":["A","B"]}}` | finite scopes |
| `overlap` | `{"categories":{"overlap":["cs.LG"]}}` | PostgreSQL array metadata |
| `ilike` | `{"journal":{"ilike":"%Lancet%"}}` | one text pattern |
| `ilike_any` | `{"organizations":{"ilike_any":["%NVIDIA%","%Google%"]}}` | OR-list of text patterns |
| `range` | `{"publish_date":{"range":{"gte":"2026-01-01"}}}` | date or numeric bounds |

Columns are ANDed. Values are converted through the indexed table's actual row
type, so applications do not pass SQL fragments or type names. Unknown columns,
unknown operations, malformed values, and unsupported operand types fail
closed. `ilike` and `ilike_any` accept PostgreSQL ILIKE patterns on string
columns; on string-array columns they match individual elements. An empty
`in`, `overlap`, or `ilike_any` set returns no rows. Collection operands are
limited to 4,096 values so an application cannot accidentally turn one search
call into an unbounded JSON expansion.

When all predicates are covered by an eligible same-root scope posting, the
structured API ranks inside that published serving baseline for up to four
times `k`, then evaluates the complete JSON predicate against those candidate
TIDs under the statement
snapshot. Stale members are removed before return. Newly matching or newly
inserted post-baseline rows may be absent until background convergence, exactly
as for planner-native search. If current recheck cannot fill `k`, this bounded
route returns the remaining verified rows rather than materializing the complete
current matching universe. Scope and accelerator refresh improve recall in the
background without becoming a foreground latency barrier.

When no eligible scope posting is available, Evoke first probes the SQL
membership resolver with a 65,536-match limit plus one overflow witness. If
the probe completes within the limit, its TIDs supply the scoring subset. The
limit bounds collected matches, not scanned heap rows or execution time.
If the probe overflows, Evoke may test a bounded prefix of the ordinary ranked
query when the physical route supports it. A successful prefix has current
predicate membership; an insufficient or unavailable prefix leads to full SQL
membership resolution. The prefix is a bounded candidate source, not an exact
resumable score iterator. Partially scope-backed filters can also require SQL
resolution of the complete predicate.

When no eligible same-root scope posting exists, the metadata predicate runs
through PostgreSQL under the same statement snapshot. Add ordinary metadata
indexes where selectivity warrants them, for example B-tree on
`document_id`/date, GIN on exact category arrays, and trigram/expression indexes
for selective ILIKE predicates. Those indexes remain PostgreSQL-owned and do
not create a second Evoke generation or worker.
For a nonempty `overlap` operand, the resolver also emits the logically implied
`cardinality(column) > 0` predicate. This preserves membership exactly and lets
PostgreSQL use a partial array GIN index declared with the same predicate.
The resolver stores one compact 64-bit key per match; it does not build a SQL
`tid[]` value. Unknown or broad predicates use a bounded SPI cursor. When a
same-root scope bitmap supplies a `work_mem`-bounded result ceiling, Evoke uses
one materialized SPI execution so PostgreSQL can retain a parallel predicate
plan, then intersects the exact TIDs with the scope bitmap.

An SSR index can place frequently used exact dimensions in the same Evoke root:

```sql
CREATE INDEX docs_search_idx
ON docs USING evoke (title, abstract)
INCLUDE (publish_date, categories)
WITH (sae = true, field_aware = true);
```

Included columns are not scoring fields and are never sent to the encoder.
After convergence, scalar `eq`/`in`/`range`/`ilike`/`ilike_any`, array
`overlap`, and string-array `ilike`/`ilike_any` resolve directly to
document-slot postings from the published serving accelerator directory when
the published value dictionary is within the bounded scope-reader limits.
Range bounds use the column type's PostgreSQL B-tree comparator and collation
rather than dictionary byte order. ILIKE uses PostgreSQL's text matcher and the
column collation over the published values; it does not approximate the pattern.
Predicates are ANDed by bitmap intersection and collection values are unioned
before that intersection. Publication, retirement, checksum validation, WAL,
reclamation, and rollback use the same root and manifest as the semantic
forward stream. There is no side relation, independent cache, or second worker
lifecycle.

Scope postings remain serving artifacts while linked L0 or a newer sealed delta
exists. Returned candidates are checked against current heap metadata, while
post-baseline matches may wait for worker convergence. This keeps foreground
latency independent from scope publication and uses the same bounded-staleness
contract as the semantic accelerator. Unsupported data types, oversized
range/pattern dictionaries, columns not declared with `INCLUDE`, and an
unavailable or partially resolved scope use the exact SQL resolver. A fully
scope-backed structured request remains on its bounded serving-baseline route
even when current recheck leaves fewer than `k` rows.

For the Commons access patterns, use the following physical metadata indexes:

| Scope | Filter | PostgreSQL index |
| --- | --- | --- |
| chunks in one document | `document_id` / `policy_ca_doc_id` `eq` | B-tree |
| publication window | `publish_date` `range` | B-tree or BRIN |
| partial-date interval overlap | `date_start` `lte` plus `date_end` `gte` | B-tree on both normalized bounds |
| exact category membership | `categories` `overlap` | GIN array index |
| fuzzy journal or organization | scalar `ilike` / `ilike_any` | converged Evoke scope; `pg_trgm` is the mutable/fallback path |

Fuzzy matching over array elements is exact but cannot use an ordinary array
GIN index. Prefer exact `overlap`; if fuzzy array matching is a frequent
requirement, expose a normalized scalar/generated metadata column with a
trigram index. This remains table metadata, not an Evoke side index.
Likewise, expression-heavy partial dates should be normalized into start/end
metadata columns so two ordinary range predicates remain planner-visible.

On a converged semantic root, the existing accelerator publication writes a
compact, checksummed TID-to-document-slot child from the document TIDs already
collected during that publication pass. It shares the accelerator generation,
retirement, WAL, and reclamation lifecycle; it is not another index generation
or worker. The preload worker may project this child into the shared arena.
A cold query reads the compact child directly when that projection is absent,
so it does not scan the full document COW stream. Linked L0 and older roots keep
the same exact membership through the checked document-stream fallback.

The default bounded-approximate filtered scorer deliberately bypasses the
term-major resident fold. Although a resident-fold bitmap intersection is
exact, opening its query-term cluster metadata before applying a statement-
local bitmap performed more work than the page-native forward route on
selective production filters. The resident fold therefore remains the
unfiltered peak path. Once a scope or explicit allowed set has been resolved,
subset top-k chooses a forward-row/transpose accelerator view or the exact
filtered BMP path described below. This is distinct from the structured JSON
global-prefix fallback used when a wide predicate has no eligible scope.

For `auto_preload > 0`, the existing preload worker prepares a shared projection
of this directory for large page-native generations after restart. A resident
fold carries an equivalent TID projection, but filtered scoring still uses the
page-native route and the durable compact child remains the cold fallback.
Setting
`evoke.maintenance_worker_limit = 0` disables both proactive preload and normal
background convergence. In that diagnostic configuration, a current root still
serves filtered queries from its compact child but does not publish shared
state; production deployments should leave maintenance enabled.

If the metadata predicate matches `M` rows, the resolver stores `8 * M` bytes
of TID keys and sorts them in place in `O(M log M)` time. PostgreSQL metadata
index access precedes that step. Each key is resolved through the shared
projection or durable root child in `O(log N)`, then represented by a compact
document-slot bitmap. This avoids an `O(N)` corpus scan per filtered query.

When the current semantic forward accelerator is eligible, Evoke scores only the
allowed rows from its bounded int8 forward stream. This preserves exact
predicate membership and computes top-k inside the subset, but scores follow the
same bounded-approximate contract as the ordinary default accelerator. Disabling
the accelerator selects the exact packed scorer. The filtered exact path derives
allowed 64-document blocks and 1,024-document superblocks from the root-stable
document-slot bitmap before reading posting records. It materializes lexical
document lengths only for allowed documents instead of walking the whole corpus.

A compatible accelerator may continue serving as `ready_baseline_delta` with no
maximum age. It runs once with bounded overfetch, rejects baseline candidates
whose current COW version, TID, or predicate membership changed, and does not
open post-baseline postings. Consequently, membership of returned rows remains
valid while a small newly inserted or improved document may be absent until
the next threshold-triggered or periodic sealed-debt refresh. This background
schedule does not impose an expiry on the serving baseline.

After a serving scope or exact SQL subset has been resolved, the filtered path
reads the forward-chunk references and one fixed model-vocabulary work table
from the tail of the same checksummed root directory; the term directory
remains lazy.
The table is accumulated while the existing forward artifacts are validated
and records each term's complete sparse ranges or dense document lanes. It is
therefore independent of forward-chunk count and does not create another
artifact or publication lifecycle.

Eligible filters choose between two subset-scoring accelerator views. Sparse
subsets use document-major forward rows; chunks containing multiple allowed
documents load one row-offset directory and reuse one-block row windows instead
of issuing two tiny random reads per document. Broader subsets use the
forward-transposed route so query terms are scored without walking every
allowed row. Direct work is estimated from each active chunk's complete posting
count scaled by selected rows. Transpose work scales the query terms' root work
by active-chunk document coverage and charges one additional lane operation per
active document for score initialization and reduction. Scratch remains bounded
by the largest active chunk plus fixed workspace. This work-count model removes
the old linear planning scan and is deterministic for the selected candidate
set, but it is not yet a full physical cost model: production-scale evidence
shows that fixed
per-chunk transpose metadata and page I/O can dominate query-term work. Route
selection therefore remains a qualification item rather than a stable
cross-root performance guarantee.
For a large cold subset, the document-major route submits bounded PostgreSQL
prefetch batches for the exact forward-row metadata and payload pages it is
about to score. Metadata residency is not treated as proof that row payloads
are resident. The later checked read still validates every consumed page, so
prefetch changes latency only and cannot change membership or scores. This is
a cold-path safety improvement, not a complete physical solution: a selective
filter whose rows are dispersed across a large relation can still be limited
by random page reads, while the warm path still performs linear row scoring.
These structures remain inside the published semantic accelerator object and
share its root identity, publication, retirement, and reclamation lifecycle;
filtering does not create a side index or persistent cache. Without an eligible
accelerator, the exact posting path skips posting blocks that contain no
allowed document. Dense high-DF packed runs carry an adaptive fine-block
membership bitset in the same exact semantic section. The scorer intersects it
with the allowed-block bitmap before reading exact signed bounds, then uses the
bit rank to address the canonical compact bound record. Sparse runs keep the
ordered reference stream when a dense membership directory would cost more.
This directory is exact query metadata, not accelerator score authority; the
approximate accelerator cannot satisfy an exact continuation proof.

For the current exact fallback, selected dense-membership ranks sharing one
immutable relation payload page reuse a checked packed-reference page window.
This removes millions of one-reference page reads without changing bounds or
scores. It does not remove the cross-run fragmentation of the current format.
Further bound-directory layout experiments are tracked in
[Model Planning](model-planning.md#2-reduce-high-df-posting-cost-without-regressing-serving);
they are not a shipped query route or a promise of corpus-independent latency.

The filtered exact score workspace follows the same physical subset. It stores
one 16-score block for each admitted b16 block and maps document slots through
a statement-local compact-block directory. The compact form is admitted only
when that directory plus its scores is smaller than a corpus-wide score array;
broad filters therefore retain the ordinary dense workspace. This is query
memory only: it is released with the statement and does not create backend-
local index-sized state or another persistent artifact.

The filtered direct-flat route retains a query-local cache only for selected
superblock record addresses. This avoids repeating page-native super-reference
searches when several competitive fine blocks belong to the same query term and
superblock. It is not a score cache or an alternate authority: exact impacts
still come from the canonical packed record, and the cache is generation-pinned,
bounded by the existing statement work limit, and released with the query.

The scorer accumulates only matching documents, so `k` is selected inside the
published filtered baseline rather than globally and then reduced by `WHERE`.
Eligible scope predicates therefore avoid enumerating a current corpus-scale
allowed set. Predicates without scope support use a bounded global-prefix probe
or the SQL membership route and score only the resulting subset.
Declared `INCLUDE` scope postings remove PostgreSQL predicate and TID-to-slot
resolution from eligible exact filters. Any future block bounds must remain
part of the same semantic accelerator root and lifecycle rather than becoming
a separately maintained artifact.

For an SSR field-aware index, use the overload with field names,
weights, filters, and `k`. Field weights are applied before subset top-k.

The `tid[]` overload remains available when an application already owns an
exact statement-local set. The set is a hard membership boundary, while ranking
uses the same exact or bounded-approximate route as the other overloads. A stale
accelerator can therefore omit an allowed post-baseline row, but cannot return
an out-of-set row. The API exposes physical row identities directly.

Heap TIDs are statement-local physical identities. Do not cache them across
`VACUUM FULL`, `CLUSTER`, table rewrites, or similar physical rewrites. A null
or empty TID set returns no rows; use the ordinary overload for unfiltered
search. Pure BM25 filtered top-k is not exposed by this beta contract.

Row-level security is unsupported. Filtering an already ranked top-k result
cannot guarantee complete or non-leaking RLS semantics. Use a non-RLS search
table or a policy-filtered materialized search relation.

Partitioned parent indexes are also unsupported for global ranking. Each child
owns independent corpus statistics, document slots, and TID space.

## Readiness And Failure

Before serving traffic:

```sql
SELECT evoke_index_status('docs_body_idx'::regclass);
```

Search fails closed for invalid physical roots, incompatible reloptions,
missing semantic runtime, model/artifact mismatch, unsupported source shapes,
RLS, or partitioned-parent use. It never falls back from semantic scoring to a
different BM25-only route.

PostgreSQL `DROP INDEX` removes the same relation-owned index authority queried
above, including every Evoke payload.

See [API Reference](api-reference.md), [Index Policy](index-policy.md), and
[Semantic Query API](examples/semantic-query-api.md).
