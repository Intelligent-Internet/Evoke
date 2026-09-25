# Beta Correctness And Stability Audit

Date: 2026-08-26

## Scope

This audit freezes product query behavior at source baseline `810374f7`. It
does not change route selection, scoring, cache sizing, index formats, worker
policy, or performance thresholds. The review is limited to correctness,
lifecycle closure, failure safety, resource ownership, and product boundaries.

The product still has one relation-owned root, one mutation lifecycle, one
maintenance authority, and one public `evoke_query(...)` entry point. SSR
remains lexical-first and eventual-only. Shared workers own model sessions;
backends do not retain index-sized semantic state.

## Review Result

No product C or SQL correctness defect was reproduced. Recent scope v6,
prefetch, prewarm, and query-route changes preserve the existing root and
worker authority. Their readers validate object references, checksums, source
generation identity, row ranges, and current scope identity before use.

One test-coverage gap was found: structured filters and long
`REPEATABLE READ` snapshots were each covered independently, but not together.
The convergent SSR lifecycle now verifies that a metadata update is visible to
a fresh statement while an older transaction keeps the old metadata view. It
also verifies that the temporary metadata value disappears after restoration
and maintenance convergence. This adds no product path or compatibility code.

## Reproduced Evidence

The outputs used for this source audit are retained under
`/private/tmp/evoke-correctness-*`. They are disposable local audit artifacts,
not packaged release evidence. The principal result files are
`evoke-correctness-sae-lifecycle-rr2.json`,
`evoke-correctness-unified-lifecycle.json`,
`evoke-correctness-2pc.json`, `evoke-correctness-vacuum-frontier.json`,
`evoke-correctness-replication.json`, and the matching build, pytest, inventory,
regression, runtime, preload, schema, privilege, and corruption logs.

| Surface | Result |
| --- | --- |
| CMake core build and C test | 1/1 passed |
| Python suite | 437 passed, 1 skipped |
| Product convergence inventory | passed |
| Unified lifecycle | 74/74 passed |
| Convergent SSR lifecycle | 71/71 passed after the new snapshot gate |
| Transactional delta and 2PC | 21/21 passed |
| Empty and UNLOGGED lifecycle | 7/7 passed |
| Same-index reader/writer concurrency | passed |
| Concurrent DDL and rewrite | 8/8 passed |
| VACUUM frontier and reclamation | 10/10 passed |
| Runtime service, cancellation, and restart | passed |
| Shared preload and auto-preload | passed |
| Schema and privilege boundaries | passed |
| Payload corruption fail-closed and rebuild | passed |
| Physical replication | passed |
| Isolated extension regression | passed |
| ASAN and UBSAN core test | 1/1 passed |

The lifecycle evidence covers commit, abort, savepoint, prepared transaction,
HOT and non-HOT update, delete, TID replacement, lexical-first visibility,
semantic completion, stale output rejection, immediate crash recovery,
reader-safe reclamation, `REINDEX`, cold restart, runtime worker restart,
replication, structured filtering, and bounded storage/RSS plateaus.

## Product Boundary

This audit qualifies current source behavior in isolated PostgreSQL 18 test
clusters. It does not prove an immutable release package, installed host
catalog, or production full-root performance. Those claims require a clean
ORT 1.29 package fingerprint and a separate Shadow package/root matrix.

The stopped Commons performance plan remains stopped. No further route ratio,
threshold, cache, or scorer micro-tuning is implied by this audit.

### Read-Only Shadow Transition Check

A read-only check on 2026-08-26 found PostgreSQL 18.4 and Evoke 0.2.4 on
Shadow. `ii_dev` is the only database with the extension and owns 14 Evoke
indexes, all valid and ready. All nine declared Commons product roots report a
healthy current generation and accelerator policy 7. The six product roots
with included scope columns report scope v6 and `scope_current=true`; the
remaining three intentionally have no scope object. No product root therefore
requires an index rebuild before the bounded scope v2-v5 and accelerator policy
5-6 readers are removed.

The installed module SHA-256 is
`a415dd9b921586b5499b076d5522d53e00e4f8474455c7cebfedbd2e7091acf0`,
and it resolves `libonnxruntime.so.1`. This identifies the current host binary
but does not prove that it came from a clean immutable package bound to the
current source, SQL catalog, model manifest, and ORT 1.29 package fingerprint.
That package/provenance gate remains open and does not require an index rebuild
as long as the cleaned binary preserves the current on-disk format.

## Minimum Deployment Boundary

The current source is suitable for a limited beta only when the deployed
package, catalog, and roots are the same current generation. Before admission,
the target must prove:

- the package is built against the pinned ORT 1.29 runtime and its installed C
  module and SQL catalog identify that package;
- every product root is valid and query-ready, has no ambiguous maintenance
  task, and reports accelerator policy 7;
- a root with included scope columns reports scope v6 and `scope_current=true`;
- restart, cancellation, one mutable CRUD cycle, and representative filtered
  and unfiltered exact-result probes pass on the installed package.

Scope v2-v5 readers, accelerator policies 5-6, and the synthetic fixed-32
extent fixture remain bounded transition code. Current writers never publish
them and currentness gates reject them for qualification. They should be
removed only after the Shadow current-generation package/root gate; removing
them before that evidence would turn a controlled rebuild boundary into an
unverified migration failure.

This boundary accepts the documented warm-query latency limitations. It does
not accept stale roots, mixed catalogs, unbounded backend RSS, incorrect
filtered membership, or a query path that silently uses a stale accelerator.
