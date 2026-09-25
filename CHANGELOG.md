# Changelog

This changelog records the public product lineage, not every
internal research milestone or intermediate Evoke tag.

The product has two generations in this release lineage:

1. `psql_bm25s` `v0.4.11` is the first public prerelease baseline.
2. Evoke `0.2.5` is the second generation, documented as **Beta 1**.

The version number was reset when the extension, access method, package,
library, SQL namespace, and configuration namespace changed from
`psql_bm25s` to `evoke`. Intermediate Evoke tags belong to the development of
the second generation; they are not separate product baselines in this file.

## [Evoke 0.2.5] - 2026-09-04

Status: published as Beta 1 under the `v0.2.5` tag and promoted to the default
GitHub Latest release on 2026-09-08. Beta 1 describes product maturity; the
GitHub pre-release flag is no longer used for this default release channel.
The public release includes PostgreSQL 17 and 18
Linux x86-64 packages and a PostgreSQL 18 Docker image, with ONNX Runtime
1.29.0 and the frozen default model. The `v0.2.5-rc1` tag is an earlier
candidate snapshot, not the published Beta 1 package.
As part of the public brand transition, current release ZIPs, Docker image
names, repository names, and model-card references use Evoke. The SQL
extension, access method, shared library, function namespace, GUC namespace,
and on-disk compatibility identity remain `evoke` for this release.

Comparison baseline: [`psql_bm25s` `v0.4.11`], released on 2026-05-11.
This section describes user-visible differences from that baseline.

### Preserved From `psql_bm25s`

- Kept a PostgreSQL-native index access method with ordinary
  `CREATE INDEX`, `REINDEX`, `VACUUM`, and `DROP INDEX` lifecycle behavior.
- Kept exact, corpus-statistics-based BM25 as the default index mode. The
  default scoring profile remains Lucene-style BM25 and IDF with `k1 = 1.5`
  and `b = 0.75`.
- Kept the `robertson`, `lucene`, `atire`, `bm25l`, and `bm25+` BM25/IDF
  families and the existing text-normalization controls.
- Kept single-column `int4[]`, `text[]`, `varchar[]`, `text`, and `varchar`
  inputs, plus homogeneous multicolumn text fusion and opt-in
  `field_aware = true` retrieval.
- Kept owner-only exact token-ID and token-stream query paths, now named
  `evoke_query_ids(...)` and `evoke_query_tokens(...)`, as BM25 regression and
  benchmark anchors.
- Kept the BM25 `@@` and `<=>` operator surfaces for supported index shapes,
  with Evoke types and operator classes replacing the old prefixed objects.
- Kept transactional `INSERT`, indexed-column `UPDATE`, `DELETE`, rollback,
  crash recovery, and PostgreSQL physical-replication requirements.
- Kept `realtime`, `eventual`, and `manual` BM25 consistency policies and
  explicit maintenance entrypoints.

### Changed

- Renamed the product and every installation boundary from `psql_bm25s` to
  `evoke`: extension, access method, shared library, SQL types and functions,
  operator classes, and GUCs. Evoke does not install aliases in the old
  namespace.
- Documented the source-preserving migration route from `psql_bm25s` and the
  legacy beta `ii42` access method to the current `evoke` package. The route
  rebuilds from source tables instead of reinterpreting old physical pages.
- Updated the published Hugging Face model card and frozen checkout contract
  to use Evoke model identifiers, `evoke_model_v1`, the
  `evoke_p2_unified_text_atoms_v2` runtime ABI, and
  Sparse Semantic Retrieval (SSR) wording.
- Made the overloaded `evoke_query(...)` family the recommended application
  retrieval surface. Explicit-`k` overloads return hit rows; scalar overloads
  support planner-native ranked SQL. Owner-only exact BM25 diagnostic
  functions remain available for regression and benchmark isolation.
- Added scope-posting admission for planner-native and structured semantic
  queries with simple predicates on Evoke index `INCLUDE` columns, including direct
  scalar `ILIKE`. One same-root, overfetched filtered request is rechecked under
  the statement snapshot. Planner-native requests retain their complete
  visible-TID fallback when a scope probe cannot fill the SQL limit. Structured
  requests whose predicates are all represented in scope postings complete on
  the bounded serving-scope candidates, even if current recheck returns fewer
  than `k`; partial, unsupported, or unavailable scope requests retain the SQL
  resolver. A compatible serving scope may remain usable while delta maintenance converges,
  so post-baseline matching rows can be temporarily absent under the normal
  approximate semantic contract. Filter child selection also rejects index-only
  paths, which cannot provide the heap TIDs required by fallback subset scoring.
- Replaced the serialized eager sparse generation and bounded delta/full
  rebuild maintenance model with one checked, page-native relation root.
  Committed changes enter a linked L0, and bounded workers seal, compact,
  fold, complete semantic rows, and reclaim retired pages.
- Replaced generation-cache-specific controls with one shared runtime and
  residency arena. BM25 can use an exact resident fold or page-native
  execution; semantic indexes require a positive shared runtime. Shared
  resident-fold admission uses the global arena and priority policy rather
  than the separate per-index relation-page warming budget.
- Made one relation-owned lifecycle authoritative for both BM25-only and
  semantic-enabled indexes. Semantic completion encodes changed document
  versions rather than the unchanged corpus. Derived accelerator construction
  still reads immutable postings and can scan indexed `INCLUDE` values in
  bounded heap-snapshot batches.
- Materialized packed semantic payload runs through their bounded BMP posting
  stream during structural maintenance. Semantic postings are not duplicated
  in generic payload arrays; lexical runs retain their physical serialized
  offsets.
- Moved prepared append-only COW construction outside the relation reader
  fence, leaving checked publication and page-retirement handoff at the
  authority boundary. Failed publication recycles its never-visible staging
  pages. Direct reuse-arena paths must instead hold the acquired fence across
  reused-page writes and publication. Non-blocking admission reduces
  foreground interference; it does not promise zero lock or I/O contention.
- Made mandatory maintenance fair across indexes. Pending L0, semantic
  completion, required structural folds, and semantic-accelerator publication
  share a cursor-rotated tier, so a continuously written root cannot starve a
  converged root that still needs its requested performance path.
- Split background discovery from execution. A worker uses a short catalog
  snapshot to reserve one root, commits that snapshot, then performs each
  preload or maintenance action in a separate transaction. Immutable posting
  work does not need a corpus-long MVCC snapshot. Heap-dependent semantic
  completion and accelerator scope capture use bounded snapshot batches,
  including registered snapshots for external TOAST reads.
- Let one maintenance worker launch drain successive actions, checking a
  one-second or 256-successful-action budget between actions. A selected
  corpus-sized action can exceed one second. Candidate selection, fairness,
  transaction boundaries, and non-blocking lock admission are repeated for
  every action, avoiding one-launch-per-term backlogs after large compactions.
- Kept a published semantic accelerator as an immutable baseline while linked
  L0 and sealed successors advance. Queries revalidate candidate versions,
  TIDs, and predicates under the statement snapshot rather than merging an
  ever-growing post-baseline posting tail in the foreground. New or changed
  matches can be temporarily absent. The stale route uses one
  accelerator pass with query-relative overfetch capped at 4,096 extra
  candidates, so foreground work does not grow linearly with mutation debt.
  Background maintenance can publish a replacement sealed baseline while
  active L0 continues accepting writes; checked publication preserves that
  frontier or retries after a conflicting publication. A capacity or
  reader-fence deferral does not expire the serving baseline or prevent urgent
  L0 work. Query authority changes only through a
  completed atomic publication, and delta scoring may conservatively drift
  until that replacement becomes current.
- Scoped document lengths and TID lookup to the immutable manifest, while the
  query-warm marker authenticates the actual serving authority: accelerator
  directory plus baseline sequence, or exact manifest plus root ID. Ordinary
  linked-L0 writes no longer discard warm query state or force a corpus
  document-length walk. Compatible seals preserve
  `query_metadata_warm=true` for `ready_baseline_delta` while exact manifest
  projections converge independently. A real accelerator replacement,
  reindex, or exact-root change still fails closed and requires a new bounded
  warmup. Invalid-marker cleanup uses the held registry lease so an older
  observer cannot retire a concurrently published replacement.
  Baseline-only accelerator TID directories are now rejected after a manifest
  seal unless the baseline is current and both visible-document and
  document-slot coverage remain exact.
  Owner-only exact structured-filter diagnostics use the current document COW
  fallback until preload publishes a complete current-manifest projection.
  The default product route may instead keep a compatible serving baseline and
  temporarily omit post-baseline matches while convergence proceeds.
  Maintenance rotates and seals active semantic debt before rebuilding an
  accelerator that already covers the manifest, while still allowing refresh
  across newer ingress after sealed authority advances. Preload authenticates
  structured-filter scope artifacts with the validated accelerator source
  authority.
  Scope construction reads external TOAST values under 256-document MVCC
  batches, preventing invalid snapshot-free TOAST access without retaining one
  corpus-wide VACUUM horizon.
  Concurrent preload publication now treats document-length metadata as
  loading rather than incomplete, retries the work hint, and reattaches a
  publisher that wins the reserve race. It no longer publishes a false
  `query_metadata_warm=false` marker at that boundary.
- Defined query visibility, semantic completeness, accelerator freshness, and
  shared residency as independent lifecycle progress axes. Committed rows do
  not wait for optional derived state; only semantic completion adds model
  output for the matching version; accelerator refresh performs no inference.
  Appends and urgent frontier work remain available during derived construction;
  conflicting immutable-root publications coordinate with its build lock.
  A successful
  active-ingress refresh requeues the preserved L0 so automatic convergence
  does not depend on a later catalog-reconciliation pass. A transient empty
  immutable baseline, such as after TID retirement while its replacement is
  still in L0, skips accelerator construction and lets core convergence run
  before the derived baseline is retried. Low mutation debt now enters
  periodic background convergence after
  `evoke.maintenance_low_debt_interval_ms` (one hour by default), while record
  and byte high-water marks remain immediate. This interval schedules work; it
  never expires a serving root or compatible accelerator. Successful sealing
  or accelerator publication restarts the interval to avoid continuous tiny
  rebuilds. Failed, raced, or concurrent accelerator attempts now receive a
  per-index retry cooldown based on `evoke.maintenance_timer_interval_ms`; the
  cooldown does not delay semantic completion, L0 rotation, sealing, or query
  service. A due accelerator receives one build opportunity even under
  continuous semantic debt, preventing semantic ingress from starving derived
  refresh. Successful or no-longer-due attempts clear the retry gate.
- Made semantic query execution prefer that published accelerator over a
  disposable exact resident fold. Auto-preload now publishes compact query
  metadata before relation-sized residency and skips the fold when the
  accelerator is eligible, so restart warmup and replacement construction do
  not redirect or block foreground semantic queries.
- Made corpus-sized semantic-accelerator construction release the per-index
  maintenance lock and the worker's cross-transaction discovery reservation
  after taking an independent non-blocking build lock. Foreground appends,
  semantic-completion appends, and safe active-to-pending rotation continue
  while the prior compatible baseline remains queryable. Other immutable-root
  publications normally defer to avoid repeated wasted builds; urgent sealing
  wins when both L0 frontiers fill, and checked accelerator publication retries
  safely. Auto-preload now warms the readable root before merely hinting
  that a stale accelerator should be refreshed, so optional derived work
  cannot leave query metadata cold after restart. Worker cleanup verifies that
  it still owns the cross-transaction reservation before releasing it, avoiding
  a second unlock after accelerator construction has already released it.
- Made release-zip construction own its checksum-locked ONNX Runtime SDK.
  Direct packaging now installs and selects the repository-pinned 1.29 SDK by
  default, records its prefix, and rejects an explicit prefix with another
  version instead of inheriting the host's `pkg-config` selection.
- Kept stable logical, physical source, and packed query-local posting offsets
  separate. Structural folds validate against the COW term directory while
  bounded materialized scoring reads the split lexical and semantic streams
  instead of treating a zero-based packed-BMP view as a changed disk plan.
- Made COW compaction preserve every active term-fold coverage boundary. A
  current-format root written by an earlier 0.2.5 candidate can recover in
  place only when its COW tail directory proves the exact covered-versus-tail
  classification; ambiguous or non-COW straddles still fail closed. Existing
  qualified roots therefore do not require `REINDEX` for this writer defect.
- Allowed an existing minor fold whose historical watermark was absorbed into
  a wider segment to promote in place to a major fold. Promotion merges two
  already validated fold objects and preserves the COW tail unchanged; new
  fold coverage still requires a current manifest boundary.
- Allowed a replacement range that crosses a fold watermark only when every
  affected posting is already covered and therefore omitted from the next
  term record. Any retained tail posting that crosses the watermark still
  fails closed. This prevents covered-only compaction from repeatedly
  deferring to a fold with no tail while semantic completion remains pending.
- Classified a retained term run in a single segment whose sequence range
  crosses its fold watermark as an unsafe compaction boundary rather than
  malformed storage. A run omitted from the authoritative COW tail remains
  safely covered by its fold, so valid covered-only roots can compact in
  place while retained-tail crossings still fail closed.
- Retuned retired-range reclamation for the compact COW lexicon. Fewer packed
  ranges now wake the existing reader-fenced, bounded reclamation path before
  a small semantic root can retain a fourth complete physical generation.
- Made current physical roots fail closed when their catalog, storage format,
  runtime contract, or model identity is unsupported. Source rows remain the
  rebuild authority for unsupported formats; the narrow same-format COW
  boundary proof above is not a cross-format compatibility reader.

### Added

- Added optional unified lexical and semantic posting with `sae = true`.
  Lexical terms and model-produced semantic atoms share one relation, posting
  namespace, scorer, mutation log, compaction path, and reclamation lifecycle.
- Added eventual semantic completion. Foreground writes become lexically
  searchable first; shared workers append semantic completion or quarantine
  state to the same linked L0.
- Added planner-native filtered semantic top-k through scalar
  `evoke_query(...)` markers and a PostgreSQL `CustomScan`. Ordinary SQL
  predicates define the visible subset before Evoke ranks it.
- Added explicit filtered-hit overloads and same-root scope postings for
  declared `INCLUDE` metadata. Compatible serving scopes remain eligible with
  unsealed L0. Unsupported or unavailable scopes use bounded SQL membership
  probing or complete subset resolution as described in
  [Query Semantics](docs/query-semantics.md).
- Added immutable semantic posting profiles: `f32`, `fp16`, and `u8` impact
  precision plus `semantic_alpha_mass`. The exact product default is `f32`
  with alpha `1.0`; lower-precision or lower-alpha profiles are explicit
  approximate choices. Exact stored impacts do not make the bounded semantic
  candidate route globally exact.
- Added a shared ONNX Runtime 1.29/API 29 execution contract, bounded local
  runtime workers, optional compatible remote accelerators, batch document
  encoding, and model/runtime identity validation. Build and background
  semantic-completion batches use the same bounded remote-capable pipeline;
  the local fallback reserves a foreground query lane when reservation is
  enabled and at least two runtime workers are configured.
- Added bounded readiness and inspection surfaces through
  `evoke_index_options(...)`, `evoke_index_status(...)`, and
  `evoke_index_details(...)`, with `evoke_index_audit(...)` reserved for a full
  integrity walk. Semantic status distinguishes correctness readiness from
  fast-path and requested-prewarm qualification.
- Added package-bound qualification for schema placement, source-table
  migration, BM25 and semantic CRUD, two-phase commit, `VACUUM`, restart and
  crash recovery, physical replication, concurrent DDL, runtime ownership,
  storage convergence, cancellation, and backend-memory stability.
- Published the frozen [Evoke Model Beta 1 checkout](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
  with an immutable download revision and checksums. The documentation now
  separates model and plugin-system technical reports, current application
  contracts, future planning, and dated historical evidence.

### Removed From The Public Surface

- Removed the installed `psql_bm25s_*` namespace from the Evoke package. An
  old `psql_bm25s` installation may remain online separately during migration,
  but Evoke does not provide cross-name aliases.
- Removed the old SQL-text fast-path advice, plan, and explain wrappers.
  PostgreSQL planning and the native Evoke scan paths own route selection.
- Removed the old public `filter_query`, `query_prepared`, and
  `ranked_query` convenience layers. Their supported use cases are covered by
  `evoke_query(...)`, ordinary SQL predicates, or owner-only exact BM25
  diagnostics.
- Replaced `generation_cache_clear`, `generation_cache_state`, and
  `generation_cache_preload` with the current runtime-state, cache-clear, and
  index-preload surfaces.
- Removed readers and upgrade shims for old `psql_bm25s` relation pages and
  experimental Evoke physical generations.

### Migration Boundary

- There is no in-place `ALTER EXTENSION` or relation-page conversion from
  `psql_bm25s` to Evoke. The extension identities and physical formats differ.
- Preserve the source table and build a new `USING evoke` index. Keep the old
  index online until result parity, CRUD, readiness, restart, and replication
  checks pass and the rollback window closes.
- Install the two extensions in separate schemas while they coexist because
  some same-signature operators cannot share one schema.
- Preserve indexed keys, `INCLUDE` columns, partial predicates, text/scoring
  options, field-aware settings, and maintenance policy when generating the
  reviewed rebuild plan.
- See [Migrate From `psql_bm25s`](docs/upgrading.md) for the supported
  side-by-side procedure and validation gates.

### Performance Claim Boundary

- This release does not claim that every query is faster than
  `psql_bm25s` `v0.4.11`. Index-size, build-throughput, exact-BM25 latency,
  semantic latency, and application/API latency are separate measurements.
- The frozen exact-BM25 BEIR matrix remains historical comparison evidence.
  Current release claims must use the same corpus, query shape, PostgreSQL
  major, cache state, binary/catalog, and result-correctness contract.
- `sae = false` remains the default, so semantic runtime and posting options
  do not silently alter a BM25-only index.

## [`psql_bm25s` 0.4.11] - 2026-05-11

Status: first public prerelease baseline for this product lineage.

The baseline shipped a PostgreSQL-native exact BM25 access method with five
input types, multicolumn and field-aware retrieval, SQL operators, canonical
token-ID and token-stream APIs, scalar-text helpers, mutable-index maintenance,
shared generation caching, crash recovery, and physical-replication support.
It did not include model-backed semantic postings or the Evoke page-native
unified lifecycle.

The upstream `v0.4.11` release is the authoritative baseline. The generated
`0.4.10 -> 0.4.11` SQL upgrade stated that the SQL surface was unchanged; the
tagged package is the source-package baseline used by the Evoke migration
smoke.

[Evoke 0.2.5]: https://github.com/Intelligent-Internet/Evoke/releases/tag/v0.2.5
[`psql_bm25s` 0.4.11]: https://github.com/Intelligent-Internet/Evoke/releases/tag/v0.4.11
[`psql_bm25s` `v0.4.11`]: https://github.com/Intelligent-Internet/Evoke/releases/tag/v0.4.11
