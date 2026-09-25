# Maintenance Lifecycle

Every Evoke index has one relation-owned search and mutation authority. BM25
and semantic-enabled indexes share the same page-native root, linked L0,
maintenance selector, COW publication, and reclamation rules.

## Durable Invariants

- One checked metapage selects one COW manifest root.
- Immutable extents, folds, document versions, and linked L0 are relation pages
  covered by WAL.
- Lexical and semantic postings share one atom namespace and publication
  boundary.
- Query backends read the checked root and snapshot-visible L0 through
  PostgreSQL shared buffers.
- Warm markers and HOT_FOLD are disposable acceleration, never authority.
- Mutation convergence uses bounded document batches, selected extents, and
  bounded reclamation. Optional accelerator construction can scan a complete
  sealed posting baseline and fetch its indexed `INCLUDE` values in bounded
  snapshot batches; one action is not a constant-time operation.
- Corpus-wide document re-encoding belongs to `CREATE INDEX`, explicit
  `REINDEX`, or an explicit rebuild path, not automatic accelerator refresh.

Unsafe or incompatible work retries or fails closed. It never publishes a
partial root or falls back to another storage layout.

## Independent Progress Axes

The lifecycle exposes four related but independent states:

1. **Visibility and MVCC correctness** come from the checked root plus the
   snapshot-visible linked L0. The exact route observes committed lexical state
   without waiting for semantic completion, accelerator refresh, or prewarming.
   The default stale-baseline accelerator may intentionally omit a small newer
   delta until the next checkpoint; it still rejects retired or replaced rows.
2. **Semantic completeness** advances when shared workers replace pending
   identities with model output for the same document version. This is the
   only background transition that adds the configured semantic signal and
   therefore improves the index toward its target ranking contract.
3. **Accelerator freshness** advances when maintenance derives a newer
   immutable baseline from already completed postings. A stale compatible
   baseline remains usable indefinitely: queries run it once, overfetch a
   bounded candidate headroom, and revalidate those candidates against current
   document authority. Small newer deltas may be absent. Refresh reduces this
   declared approximation drift, but grants no row visibility and never runs
   model inference.
4. **Warmness and residency** change only where reconstructible query data is
   served from. They affect cold and warm latency, not rows, scores, or
   durable convergence.

These states must not gate each other. Workers may consume CPU, I/O, and the
configured maintenance workspace, so zero resource contention is not a
promise. The contract is behavioral isolation: queries keep a pinned readable
authority, foreground commits keep appending, and a blocked or failed optional
refresh yields without stopping semantic completion or L0 convergence.

## Foreground Mutation

Automatic-policy changes append transaction-tagged records to linked L0 under
the same root authority before commit. Readers and maintenance distinguish
committed, unresolved, and aborted records; physical append is not a grant of
visibility.

For BM25, records carry exact lexical change state required by the selected
consistency policy. For semantic-enabled indexes, a new version carries exact
lexical state and semantic-pending identity. Foreground DML performs no model
inference.

Transaction abort, savepoint rollback, and prepared-transaction rollback leave
no visible mutation. Commit makes the new version visible under PostgreSQL
MVCC. Later maintenance consumes each eligible record exactly once.

## Semantic Completion

Shared workers select bounded semantic-pending document versions, encode their
source text, and append matching completion records. Before append and before
root publication, the worker revalidates:

- the current document/version identity;
- the captured root and linked-L0 frontier;
- the model checkout and runtime-contract signature;
- retirement or supersession state.

Stale model output is discarded and retried; it cannot attach to a reused
document slot. Repeated row-local failures enter bounded quarantine while the
row remains lexical-searchable through the exact route and status remains
non-converged. A compatible bounded accelerator may temporarily omit that
post-baseline row.

## Bounded Maintenance Selector

One maintenance attempt chooses at most one bounded action: satisfy a
query-visibility obligation, complete one semantic batch, seal one linked-L0
interval, refresh one semantic accelerator baseline, compact one extent set,
advance one term-local fold, or reclaim retired objects after reader fencing.

Action ordering preserves authority rather than forcing every derived object
to be current. An occupied pending L0 and urgent structural work precede
accelerator refresh. Semantic completion is normally preferred, but a due
accelerator receives one independent build opportunity even while semantic
debt remains actionable. The build reads one immutable completed-posting
snapshot; later semantic output stays in L0 and makes that baseline stale
without invalidating it. The active L0 checkpoints immediately at 65,536
records or 512 pages. Below those limits, coalesced debt becomes eligible for a
low-priority checkpoint after `evoke.maintenance_low_debt_interval_ms`, one hour
by default. A compatible accelerator refresh likewise starts immediately at
65,536 sealed records or 512 MiB, or periodically for smaller sealed debt.
Successful sealing or accelerator publication starts a fresh low-debt interval,
which avoids a continuous stream of tiny corpus-derived rebuilds.
Corpus-sized accelerator work uses a separate per-index build lock and releases
the urgent maintenance lock and discovery reservation before scanning. While a
build is active, queries, foreground appends, semantic-completion appends, and a
safe active-to-pending rotation continue, but other immutable-root publications
for that index defer. If both L0 frontiers fill, urgent sealing wins and the
accelerator retries rather than blocking writers. This bounds wasted rebuilds
without coupling foreground availability to derived convergence.
An unsuccessful accelerator attempt establishes a per-index retry cooldown. A
lost root CAS, busy reader fence, concurrent builder, resource block, or build
error cannot turn the one-second preload supervisor tick into continuous corpus
scans. A successful or no-longer-due attempt clears the retry gate. The
cooldown uses `evoke.maintenance_timer_interval_ms`; it suppresses only another
accelerator attempt, so semantic completion, L0 rotation, sealing, and other
urgent structural work remain independently eligible.

The selector applies urgent state and database-local fairness without creating
separate per-feature schedulers. Worker hints accelerate discovery; periodic
catalog reconciliation guarantees eventual rediscovery after restart or lost
hints.

## COW Publication

Maintenance captures a root and eligible frontier, builds immutable descendants
outside current authority, validates their complete closure, and publishes one
small WAL-logged metapage switch.

- A crash before the switch leaves the old root authoritative.
- A crash after the switch exposes the complete descendant.
- Readers that pinned the old root keep it until query completion.
- Commits after the captured frontier stay in linked L0 for a later attempt.
- A changed root or busy non-blocking fence turns the action into retryable
  work, not mixed-root publication.

## Compaction And Fold

Sealing turns a complete L0 interval into immutable posting extents. Compaction
merges selected extents under fixed input bounds. Term-local fold produces an
exact contiguous read shape for hot or structurally expensive terms.

Folds are derived from durable posting state. A valid fold and its uncovered
tail return exactly the same scores as the fragmented representation. Losing a
fold changes read cost only; it does not require repair.

Compaction can absorb an older per-term fold watermark into a wider segment.
That stale boundary remains valid only when the COW tail descriptor proves the
covered-versus-tail split. Promoting an existing minor fold to a major fold
preserves that authenticated watermark and rewrites no tail extent; advancing
a fold over new extents still requires a current manifest segment boundary.

The optional HOT_FOLD projection is a volatile residency of the durable fold.
It follows the same exactness rule.

## `VACUUM` And Reclamation

Heap MVCC hides superseded and deleted rows immediately. PostgreSQL `VACUUM`
later supplies exact dead-TID decisions to the access method. Evoke records
logical retirement in the same document COW and posting lifecycle.

PostgreSQL may skip an index access method during `VACUUM` when
`INDEX_CLEANUP AUTO` decides cleanup is not yet worthwhile. Query visibility
remains correct through heap MVCC, but exact retirement statistics and physical
page reuse then wait for a later cleanup pass. Operations and qualification
that require immediate retirement convergence must use
`VACUUM (INDEX_CLEANUP ON)`; routine autovacuum may remain adaptive.

Logical retirement and physical page reuse are separate:

- a retired document stops contributing to current search/statistics after the
  retirement publication;
- old pages remain protected while any reader can still reference them;
- ordinary COW replacement is written append-only, without a relation fence;
- overflow publication uses a non-blocking reader fence only to revalidate and
  exchange the root, then releases it before WAL marker/FSM handoff;
- if that fence is busy, maintenance yields and returns its precisely tracked,
  never-published staging pages to the FSM.

Reclamation becomes due when either 8 retired ranges accumulate or their
combined size reaches 8,192 blocks (64 MiB at the standard block size). One
maintenance action returns at most 8,192 blocks to the FSM. This bounds the
reclaimed volume per action, not an absolute lock-duration or I/O-time limit.
A large accelerator replacement can reclaim over repeated actions. Pure reclamation
successors inherit a still-current accelerator by authority checksum, so page
reuse cannot force queries back to the exact full-posting route. A small final
manifest range remains below both thresholds.

Prepared-handoff paths stage their immutable closure before acquiring the
reader fence and do not rerun that COW build after fencing. Reclamation-only
publication similarly prepares its identity manifest first and can release
the fence before the FSM handoff. Direct reuse-arena paths have a stronger
requirement: the acquired fence covers every reused-page write and the
replacement-root publication. Conditional acquisition yields to existing
readers, but a successfully held fence can delay new readers. See
[the reclamation implementation](../src/evoke_am_reclamation.c); the fence is
not universally limited to a metapage exchange.

Because pure reclamation changes only page ownership, not query authority,
publication rekeys safe shared entries such as validation state, hot folds, and
the warm marker while the reader fence is held. Document lengths and the TID
lookup use stable relation keys but authenticate their serving authority inside
the payload. Accelerator-bound metadata therefore survives compatible manifest
successors; exact-root metadata is retired when its authority changes. The
pointer-free resident fold embeds the exact root identity and is always retired
and rebuilt rather than relabelled.

The unified warm marker has a narrower serving-authority identity. For an
accelerator query path it authenticates the accelerator directory object and
baseline sequence; otherwise it authenticates the exact manifest and root ID.
A compatible sealed successor keeps the same accelerator marker while exact
manifest projections converge independently. Replacing the accelerator,
changing the exact BM25 root, reindexing, or changing the relation locator
invalidates the marker. Invalid-marker cleanup retires the entry through its
held lease, so a concurrent worker cannot publish a new authority marker and
then have an older observer accidentally remove it.

If two preload attempts meet at the same manifest, the loser observes the
document-length projection as loading and retries instead of publishing an
incomplete unified marker. If publication completes between lookup and reserve,
the loser reattaches the winning entry. This keeps `query_metadata_warm` stable
across worker and operator-triggered preload races.

No query or maintenance path allocates a corpus-sized deleted-row bitmap.

## Build And Repair

Explicit `CREATE INDEX` and `REINDEX` are controlled whole-corpus operations.
They select a standard, compact, spill, or semantic-stream builder according to
input shape and memory admission. Temporary-file-backed builders keep large
posting streams bounded.

Automatic maintenance does not enter this ladder for ordinary convergence. If
an automatic rebuild-like action exceeds
`evoke.maintenance_rebuild_memory_budget`, it preserves the readable root and
reports a blocker instead of forcing host swap.

The semantic query accelerator is durable derived relation data, not another
mutation lifecycle. Publication reads existing completed postings and never
reruns the encoder. Once published, it remains an immutable query baseline
across compatible linked-L0 and sealed successors. A stale-baseline query runs
the accelerator once, rejects changed or retired baseline rows, and does not
open exact post-baseline posting streams. Its extra candidate count is exactly
`min(changed_documents, min(max(k, 64), 4096))`: small requests reserve up to
64 replacements, ordinary requests allow up to one additional `k` window, and
all requests stop at 4,096 extra candidates. Foreground query work therefore
does not grow linearly with accumulated mutation debt. If more high-ranking
baseline rows changed than that bounded headroom can replace, or newer documents
would rank, result and score drift is part of the declared approximate profile.

A worker builds the replacement against a pinned immutable manifest without
changing foreground authority, then swaps the complete artifact atomically.
Posting and transpose preparation holds neither a transaction ID nor an MVCC
snapshot horizon. Scope preparation reads indexed `INCLUDE` values under a
current MVCC snapshot in batches of at most 256 root documents because
PostgreSQL requires a registered snapshot to fetch external TOAST data. Each
batch releases that snapshot before the next batch, so the corpus-sized pass
does not pin one VACUUM horizon. The bounded publication phase may assign the
short write transaction ID needed for the checked root switch, but it does not
retain a snapshot horizon.
It does not require writers to stop. A first or contract-incompatible artifact
is scheduled immediately. A compatible stale artifact has no serving-age
deadline: it remains usable until a complete replacement publishes. Its sealed
debt refreshes immediately at the record or byte high-water mark and otherwise
becomes low-priority work after the configured low-debt interval. The worker
starts with no pending L0 and gives up the urgent maintenance lock for the
corpus-sized build. New writes and semantic completion may append; other
immutable-root publications defer until the replacement publishes. The
independent build lock prevents duplicate scans, and the existing compatible
baseline remains the query path throughout.

Auto-preload also treats the current readable root independently from optional
accelerator freshness. It first publishes metadata for the actual serving
authority and warms its bounded page set. Maintenance hints independently track
both immediate high-water work and periodic low debt. A slow or blocked refresh
therefore cannot keep
`query_metadata_warm=false` or make startup query latency depend on rebuild
completion. When an existing accelerator serves
`ready_baseline_delta`, manifest seals preserve its authenticated warm marker;
its baseline-sized document lengths and TID projection remain authenticated by
the same accelerator authority instead of flapping with every manifest.

The durable accelerator TID directory describes the immutable baseline, not
later sealed documents. A filtered stale-baseline query may attach that
baseline-sized directory, but every matching slot is revalidated against the
current document COW TID and current predicate scope before admission. This
prevents false-positive membership from updates or deletes. New post-baseline
documents may remain absent until checkpoint and refresh, matching the declared
approximate profile. When no accelerator is eligible, preload instead builds
exact-root metadata for the page-native route.

Pending L0 still seals before accelerator work. Actionable semantic completion
normally runs first, but cannot starve a threshold-triggered or periodic
refresh: the scheduler selects one accelerator build and then lets semantic
workers continue during its retry window. The resulting baseline contains all
semantic postings completed in its immutable source snapshot. Semantic output
published later remains L0 debt under the declared approximate profile.
Convergence assumes maintenance capacity can eventually seal pending debt; if
the write rate permanently exceeds that capacity, linked L0 remains the exact
availability authority and status stays non-converged rather than claiming a
current accelerator.

The worker estimates document metadata, vocabulary selection, largest-term,
bounded tuplesort, and publication workspace before streaming the sealed root.
It reads one term at a time and spills the document transpose to PostgreSQL
temporary files, so total corpus posting bytes are not resident. If the estimate
exceeds the same maintenance budget it reports `accelerator_memory_budget` or
`accelerator_scope_memory_budget`, preserves the previous baseline, and relies
on normal periodic reconciliation rather than an immediate retry loop.

The accelerator directory records a builder-policy identity and its retained-
document bound. Current policy 7 uses directory v10, signed-int8
forward/transpose format v6, and same-root scope v6 metadata when scope columns
are declared. It keeps at most 64 seed documents per selected term. The default
query path uses a `0.7` cluster heap factor, `64x` candidate headroom, bounded
residual candidates, and cross-term residual accumulation before direct-row
scoring.
This is a bounded candidate surface, not a copy of each source posting list.
A missing or older identity is reported as `stale_policy`, is excluded from
approximate execution, and is replaced by the same worker path.
The old directory and children enter the normal manifest retirement inventory
after replacement. If a fragmented legacy artifact
cannot fit in that bounded inventory, publication briefly takes the existing
reader fence and transfers its old page ranges through the normal FSM handoff.
Small and current artifacts stay on the ordinary lock-light path. Initial
`CREATE INDEX` and `REINDEX` publish only the exact root; they do not retain
corpus-sized build sources to construct this disposable artifact
synchronously.

Cross-term accumulation uses a dense float workspace and is admitted only
inside the 64 MiB query working-set bound (about 16.8 million document slots).
Larger roots or allocation failure automatically use the exact packed scorer.
The bounded summary, graph, sketch, DF-pruning, and semantic-error-budget
variants remain diagnostics because their row-level loss has not passed the
product gate; they are not silently stacked into the default route.

## Restart And Replication

WAL carries relation pages and root publication. Restart rebuilds disposable
worker and residency state from the checked root.

Physical standbys query replayed page-native roots and may prewarm relation
pages. They do not perform primary-side durable maintenance while in recovery.
Promotion resumes normal maintenance from the replayed root and linked L0.

Logical replication copies table rows, not index relations. Build the Evoke
index independently on the subscriber.

## Operator Surface

```sql
SELECT evoke_index_status('docs_search_idx'::regclass);
SELECT * FROM evoke_index_details('docs_search_idx'::regclass);
SELECT evoke_index_try_maintain('docs_search_idx'::regclass);
SELECT evoke_index_maintain('docs_search_idx'::regclass);
SELECT * FROM evoke_index_maintain_due(2);
DROP INDEX docs_search_idx;
```

Use `try_maintain` for unattended work and `maintain` when waiting is allowed.
The built-in worker already handles automatic policies; an external scheduler
is optional.

A built-in worker may service several due indexes or several successive
bounded actions for one index during one launch. Every action keeps its own
transaction and repeats candidate selection and non-blocking lock admission.
Between actions, the launch checks a budget of 256 successful actions or one
second of maintenance work, whichever comes first. It also stops when an action
cannot make progress. This is not a one-second deadline for a single admitted
action: an accelerator build can take longer. The budget avoids repeated
one-launch-per-term overhead without allowing an unlimited action loop.

`evoke_index_status(...)` is safe for routine readiness polling and does not walk
the full relation. Storage qualification uses `evoke_index_audit(...)`
explicitly; its cost is proportional to index size and it should not be placed
in dashboards or health checks. The `_internal` generation-audit helper is an
extension implementation boundary, not the operator entrypoint.

## Qualification

Release gates cover:

- exact rows, scores, and order against an independent page-native oracle;
- commit, abort, savepoint, two-phase commit, old snapshot, and TID reuse;
- semantic completion, quarantine, retry, and supersession;
- crash/cancel before and after root publication;
- concurrent reader, writer, maintenance, `VACUUM`, and DDL activity;
- page retirement, reader fencing, reuse, and fixed-live-set storage plateau;
- restart, physical replication, standby prewarm, and promotion;
- bounded backend memory and worker-owned model sessions;
- BM25 exactness and performance with semantic mode disabled.

See [Index Policy](index-policy.md), [Shared Runtime And Residency](shared-runtime-and-residency.md),
and [Testing And Validation](testing-and-validation.md).
