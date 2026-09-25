# Testing And Validation

Evoke validation is layered so a scorer unit test cannot hide an invalid
PostgreSQL lifecycle. Product acceptance uses the installed or explicitly
staged extension, one page-native index path, the public explicit-hit
`ii42_query(..., k, ...)` route, and planner-native scalar `ii42_query(...)`
SQL.

## Fast Checks

```bash
python3 -m pytest -q
python3 scripts/test_product_convergence_inventory.py
python3 -m compileall -q scripts tests

find scripts packaging \
    -type f -name '*.sh' -print0 \
    | xargs -0 -n1 bash -n

git diff --check
```

`pytest` owns deterministic Python units for packaging, native evaluation,
release provenance, and maturity-runner composition. It does not claim
PostgreSQL lifecycle coverage. The inventory gate enforces one
rank/search/status/maintenance/drop lifecycle, eventual-only Sparse Semantic Retrieval
(SSR), current page-native documentation, and absence of split storage authorities.

All `scripts/test_*.py` entrypoints belong to the canonical product maturity
suite. Focused invocation is supported for diagnosis, but standalone smoke
wrappers are not a second acceptance surface. `scripts/benchmark_*.py` and the
native qrels tools produce performance or quality evidence; they are not
correctness gates unless the maturity runner invokes them explicitly.

## Build And Core Tests

```bash
cmake -S . -B build_tmp -DCMAKE_BUILD_TYPE=Release
cmake --build build_tmp --parallel 4
ctest --test-dir build_tmp --output-on-failure
```

These C tests are independent of the PostgreSQL extension build. Build the
extension with the pinned ONNX Runtime and PostgreSQL development files using
the [contributing workflow](../CONTRIBUTING.md#build-from-source). Do not reuse
a CMake cache copied from a differently located checkout; build directories are
local generated state, not release inputs.

Run PostgreSQL regressions in an isolated temporary cluster so a resident
preloaded library cannot mask the candidate build:

```bash
python3 scripts/test_extension_regression_temp_pg.py \
    --pg-bin /opt/homebrew/opt/postgresql@18/bin
```

`make installcheck` remains appropriate for a fresh CI cluster.

## Unified Lifecycle

```bash
python3 scripts/test_unified_index_lifecycle_smoke.py \
    --model-path /path/to/model-checkout \
    --mixed-soak-cycles 10 \
    --concurrent-crud-cycles 8 \
    --concurrent-readers 4 \
    --concurrent-writers 2 \
    --soak-queries 100 \
    --output /tmp/ii42-unified-lifecycle.json
```

To bind the run to staged source, provide both `--extension-libdir` and
`--extension-control-dir`. The harness rejects a partial binding and records
the resolved package roots.

The lifecycle gate must prove:

1. BM25 and `sae = true` publish one checked relation root and use
   `ii42_query(...)`.
2. Lexical and semantic atoms occupy one posting namespace; no second index,
   registry, or application-managed payload exists.
3. SSR is eventual-only: foreground DML publishes lexical evidence and pending
   identity with zero document inference; shared workers complete semantics.
4. Commit, abort, savepoint, two-phase commit, old snapshots, HOT/non-HOT
   updates, delete, and TID reuse obey heap MVCC. Structured filters must keep
   metadata membership exact across same-transaction changes and concurrent
   `REPEATABLE READ` snapshots.
5. Semantic completion consumes bounded linked-L0 intervals and does not
   re-encode unchanged or already completed document versions. Structural
   maintenance reuses persisted postings. Derived scope construction may read
   corpus-wide INCLUDE metadata in bounded snapshot batches; this is not a
   full-corpus semantic re-encoding pass.
6. `VACUUM (INDEX_CLEANUP ON)` supplies immediate exact dead-TID retirement
   and reader-safe reclamation. `INDEX_CLEANUP AUTO` may defer physical
   convergence without changing MVCC-visible query correctness.
7. Concurrent reader, writer, maintenance, fold, and root publication cannot
   expose mixed-root results.
8. Cold restart, immediate crash, `REINDEX`, and physical replay preserve
   readiness and the selected exact or bounded-approximate ranking contract.
9. Shared workers own model sessions; backends retain no index-sized semantic
   state. Warm metadata remains attached across active linked-L0 ingress when
   its serving authority is unchanged. A compatible older accelerator baseline
   and its projections remain usable until replacement or a genuine authority
   incompatibility, not merely until the next manifest publication.
10. A failed build or model-contract mismatch preserves the previous readable
    root and fails closed.
11. `ii42_index_status(...)` exposes readiness, semantic debt, convergence,
    and blockers without a relation-sized or model-sized walk.
    `ii42_index_audit(...)` separately validates COW closure, reachability,
    reclaim markers, derived accelerator objects, and model SHA-256 identities.
12. PostgreSQL `DROP INDEX` removes the complete relation-owned index lifecycle
    without an external cleanup step.

Any failed invariant is a product failure, not a known-warning baseline.

## Focused Mutation And Concurrency Gates

```bash
python3 scripts/test_transactional_delta_lifecycle.py \
    --model-path /path/to/model-checkout

python3 scripts/test_same_index_writer_concurrency_temp_pg.py \
    --model-path /path/to/model-checkout

python3 scripts/test_eventual_semantic_maintenance_fairness.py \
    --model-path /path/to/model-checkout
```

These isolate transaction callbacks, writer serialization, semantic
completion and accelerator-publication fairness, restart reconciliation,
query admission while a writer is open, and bounded storage under sustained
CRUD. The fairness gate includes a continuously written high-OID root and a
lower-OID root waiting for its accelerator, so pending-L0 urgency cannot bypass
cross-index rotation. During the accelerator build it also samples the
builder's advisory locks, probes same-root maintenance, writes to that root,
and requires structural progress before checked accelerator publication
retries. This prevents a transaction or cross-transaction discovery lock from
serializing mutation convergence behind corpus-sized derived work. The staged
same-index writer gate owns both writer and public-query admission coverage so
those contracts cannot drift across separate database fixtures. It records a
single-reader warm baseline, a parallel-reader control, concurrent
reader/worker activity, and post-drain query latency distributions so a
correctness pass cannot hide a query stall. Reader waits longer than
`deadlock_timeout` are also checked against PostgreSQL `log_lock_waits`;
maintenance may compete for CPU and I/O but must not serialize a public query
behind a heavyweight lock. The gates must compare
result identities, order, and scores with the independent oracle rather than
merely checking that SQL completed.

The fairness gate drains both axes separately: semantic/L0 debt must reach
zero first, and every admitted semantic accelerator must then advance from a
compatible baseline-delta state to a current baseline. This prevents an early
correctness pass from hiding a permanently stale performance path. Accelerator
prepare samples may not hold `backend_xid`. Posting and transpose phases may
not hold `backend_xmin`; scope extraction may expose it only for the bounded
256-document MVCC batches needed to read external TOAST values. The atomic
publication may briefly assign a write XID but must not hold `backend_xmin`.

Release qualification repeats the convergent lifecycle and concurrent-reader
gate. Latency evidence must include p50, p95, and maximum values before, during,
and after maintenance. Hardware-specific ratios are reported rather than
encoded as portable correctness assertions; statement timeouts, empty results,
runtime failures, or failure to return to a stable post-drain range are release
failures.

## Page-Native Exactness And Memory

```bash
python3 scripts/test_unified_index_lifecycle_smoke.py \
    --model-path /path/to/model-checkout \
    --output /tmp/ii42-page-native-lifecycle.json

python3 scripts/test_backend_memory_ownership.py \
    --output /tmp/ii42-backend-memory.json

python3 scripts/test_convergent_segment_read_smoke.py
python3 scripts/test_convergent_vacuum_frontier_smoke.py
```

Acceptance requires:

- exact result rows/order and documented float tolerance;
- checked-root plus snapshot-visible linked-L0 parity;
- exactness before and after seal, compaction, fold, `VACUUM`, and restart;
- bounded query-specific L0 projection and no corpus scan;
- no foreground document inference;
- bounded backend private memory and worker-owned model sessions;
- storage plateau for a fixed live set after reader-safe reclamation.

For an accelerator-eligible index, the unified lifecycle smoke reaches
`state=ready`, `baseline_current=true`, and `scope_current=true` before it
accepts `no_pending`. Core semantic debt reaching zero is not by itself a
complete derived-performance convergence result. Stability soak baselines are
taken only after both axes reach that idle state. Long-snapshot checks compare
sealed authority rather than manifest identity because accelerator publication
and active-L0 rotation may safely advance the checked root while the old MVCC
horizon remains protected.

The same smoke stops immediately after a pending-L0 seal and requires the
unchanged accelerator to report `state=ready_baseline_delta`,
`baseline_current=false`, and a valid `query_metadata_warm=true` marker before
derived convergence continues. This catches manifest churn that incorrectly
invalidates serving query metadata. Planner-native and structured filters must
reject stale baseline TIDs after UPDATE or DELETE, while a newly inserted
post-baseline row may remain absent. The background due selector must leave
sub-threshold debt alone before the configured low-debt interval, then rotate
it after that interval without running inference or invalidating the serving
marker. Explicit per-index maintenance may advance it immediately. A forced
test threshold then
seals the delta, refreshes the accelerator, and requires the new row to appear
with the marker valid
throughout.

The backend-memory gate uses allocator-zone and private-writable ownership from
`vmmap` on macOS. On Linux it uses PSS from `smaps_rollup` as an observational
resident metric and counts only writable private mappings from `smaps` for the
hard ownership gate. Linux does not expose the macOS allocator-zone split, so
that unavailable metric remains explicit rather than being reported as zero.

Cache clear, eviction, and cold warmup may change latency only. Disposable
residency is never correctness evidence.

## Shared Runtime And Preload

```bash
python3 scripts/test_shared_preload_lifecycle_closure.py \
    --bindir /opt/homebrew/opt/postgresql@18/bin

python3 scripts/test_shared_preload_auto_preload.py \
    --bindir /opt/homebrew/opt/postgresql@18/bin

python3 scripts/test_runtime_service_required_smoke.py
python3 scripts/test_runtime_service_privilege_smoke.py
python3 scripts/test_runtime_service_restart_smoke.py
```

These gates require semantic operation to fail closed without the shared
runtime, enforce privilege boundaries, validate worker restart/liveness, warm
relation pages through PostgreSQL shared buffers, and retain only bounded
markers or derived residency in the Evoke arena.

## Schema And Package Boundaries

```bash
python3 scripts/test_extension_schema_smoke.py \
    --pg-bin /opt/homebrew/opt/postgresql@18/bin

python3 scripts/run_product_maturity_suite.py \
    --package-root /path/to/staged/package \
    --source-package-root /path/to/psql_bm25s-package \
    --model-path /path/to/model-checkout \
    --skip-benchmark \
    --output /tmp/ii42-product-maturity.json
```

The package suite verifies the extension binary, control file, install SQL,
bundled runtime, licenses, and `BUILD-INFO.txt` before database tests. A staged
package must remain byte-identical throughout qualification. Restart,
independent golden, low-memory maintenance, backend RSS, semantic fairness,
shared-preload, and replication gates run by default. Use
`--skip-restart-smoke` only for focused diagnosis, never for release
qualification.

The only historical product migration gate is source-table migration from a
self-consistent `psql_bm25s` package:

```bash
python3 scripts/test_psql_bm25s_source_migration_smoke.py \
    --source-package-root /path/to/psql_bm25s-package \
    --extension-libdir /path/to/ii42-stage/pkglibdir \
    --extension-control-dir /path/to/ii42-stage/sharedir \
    --output /tmp/ii42-source-migration.json
```

It keeps both products queryable during side-by-side validation, exercises
CRUD, and removes the old extension only after result and dependency checks.
Intermediate Evoke experiment catalogs are not a compatibility target.

## Restart And Replication

`scripts/test_replication_lifecycle_smoke.py` must run BM25 and semantic-enabled
indexes through create, CRUD, maintenance, `VACUUM`, restart, `REINDEX`, replay,
promotion readiness, and drop. Primary and standby must agree on checked-root
contract, document count, readiness, and representative ordered results.

Logical replication copies source rows rather than index relations and is
validated by independently building the subscriber index.

## Performance Qualification

Performance work is a separate evidence gate. It must measure the same
installed page-native product path used by applications and keep these surfaces
separate:

- BM25 query throughput and tail latency;
- semantic query encoding plus posting traversal;
- build and `REINDEX` encoding throughput;
- eventual completion latency under sustained writes;
- batch versus single-text atom/weight parity;
- worker/session RSS, backend private memory, and storage plateau;
- cold versus warm relation-page behavior.

Query encoding remains single-text. Only build and maintenance document work
may batch. Do not copy historical measurements into product claims; publish
new numbers only with a reproducible workload, package fingerprint, hardware,
provider, raw artifacts, and current dataset.

See [Performance Evidence](performance/README.md).

## Native Quality Evaluation

Quality evaluation must query relation-owned indexes through
`ii42_query(...)` and join TIDs back to the corpus table:

```bash
python3 scripts/evaluate_ii42_native_qrels.py \
    --dsn 'host=/path/to/socket port=55432 dbname=postgres' \
    --dataset scifact \
    --schema bench \
    --table docs \
    --id-column id \
    --index bm25=bench.docs_body_idx \
    --index semantic=bench.docs_semantic_idx \
    --queries-jsonl /path/to/queries.jsonl \
    --qrels-json /path/to/qrels.json \
    --k 1000 \
    --output-json /tmp/scifact-native-qrels.json
```

The evaluator rejects invalid or non-query-ready indexes. A sampled candidate
surface is a canary, not a substitute for the official full corpus.

See [Contributing](../CONTRIBUTING.md), [Architecture](architecture-and-design.md),
and [Semantic Index Operations](examples/semantic-index-operations.md).
