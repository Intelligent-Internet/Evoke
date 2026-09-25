# Shadow Natural Search Qualification

Date: 2026-08-26

Status: qualified for the Evoke 0.2.5 natural `evoke_query` API on the Shadow
primary and physical standby. This closes the API migration and same-root
deployment gate. It does not reopen broad performance tuning.

## Product Contract

Evoke now uses one public overloaded name for both query styles:

- `evoke_query(index, query [, fields, weights]) RETURNS real` is the scalar
  marker used in `ORDER BY ... DESC LIMIT k`.
- `evoke_query(index, query, k, ...) RETURNS SETOF evoke_result_hit` remains the
  explicit-hit API.

There is no separate public rank function, and the former internal ranked-query
helper is absent from the current catalog. Supported natural SQL is planned as
`Custom Scan (Evoke Search)` so PostgreSQL predicates and MVCC visibility become
inputs to the existing unified scorer.

## Immutable Package

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `a04f5753b6e10b8382b521577485da4ba75ffa6535e375096e9c9a50da38f728` |
| Linux PG18 package | `95ec8d3e8b0638fade6633957d056bdb1565851d9dd65117f6f7fe2d42ebe7b6` |
| `evoke.so` | `2962ab00227737dcaeead42dffd7642d07fec51aa221b16c085b93bc9dec6eae` |
| ONNX Runtime 1.29.0 | `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca` |
| Model manifest | `dd0993f0638a3f683fbb6475a8509c228d8f6547aab90a0b762b1ff6fd7d0041` |
| Installed 0.2.5 SQL | `229a6db7538a1b359764fe4f7960a2e04ab77350309572a3c9b08daa1abd67d0` |

The package is based on commit
`8e240cb373c846c0bf57c1533b4284695ecbb4bc`; the source archive records the
uncommitted planner-native change set independently. The catalog update from
0.2.4 to 0.2.5 is SQL-only and did not rebuild or republish an index.

## Shadow Pair

| Property | Primary | Standby |
| --- | --- | --- |
| Host | `172.31.19.105` | `172.31.24.73` |
| Recovery mode | false | true |
| Extension | 0.2.5 | 0.2.5 |
| Valid/ready Evoke indexes | 14/14 | 14/14 |
| Binary SHA | `2962ab...eae` | `2962ab...eae` |

All primary and standby Evoke relfilenodes were identical. Replication was
streaming asynchronously with zero replay lag at the final snapshot. Logs
after the final postmaster starts contained no crash, SIGBUS, PANIC, missing
relation, or repeated worker failure.

Physical standbys do not have the main fork of an unlogged relation. Auto
preload and reconciliation now skip unlogged Evoke indexes during recovery,
preventing repeated attempts to open absent forks while retaining logged
product-index preload.

## Correctness

The natural planner path and explicit allowed-TID oracle ran in the same
statement snapshot:

| Case | Allowed | Natural | Explicit | TID diff | Predicate violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| arXiv 2024 + `cs.LG` | 28,341 | 20 | 20 | 0 | 0 |
| PubMed 2024+ + journal `nature` | 39,614 | 20 | 20 | 0 | 0 |

A 50 ms cancellation interrupted a PubMed planner-native query, after which
the same connection successfully returned 20 arXiv results. A read-only query
on the standby also used `Custom Scan (Evoke Search)` and returned the expected
20 filtered hits.

## Warm Performance

Measurements used `EXPLAIN (ANALYZE, TIMING OFF, FORMAT JSON)`. One warm-up was
discarded and the following five runs supplied p50/min/max, except the PubMed
filtered p50, which used ten measured runs after warm-up.

| Query | p50 | Min | Max |
| --- | ---: | ---: | ---: |
| arXiv natural, unfiltered | 207.325 ms | 205.276 ms | 211.308 ms |
| arXiv explicit hits | 211.264 ms | 208.852 ms | 214.944 ms |
| arXiv natural, filtered | 221.836 ms | 215.048 ms | 225.151 ms |
| PubMed natural, unfiltered | 442.112 ms | 439.036 ms | 456.896 ms |
| PubMed explicit hits | 442.545 ms | 439.316 ms | 455.022 ms |
| PubMed natural, filtered | 784.495 ms | 771.529 ms | 806.867 ms |

The natural unfiltered path is within one percent of the explicit-hit route on
both corpora. This is the relevant no-regression gate for planner integration.
The broader PubMed predicate remains more expensive than unfiltered retrieval,
but it no longer uses an application JSON carrier or post-filter semantics and
is outside this bounded API-closure performance scope.

## Validation And Deployment Notes

Repository qualification passed 442 Python tests with one skip, the focused
planner contract suite, PG18 compilation, CMake/CTest, product convergence
inventory, Python compilation, and `git diff --check`.

An initial deployment copied over a mapped `evoke.so` and caused an old primary
process SIGSEGV and an old standby process SIGBUS. This was a non-atomic binary
installation error, not a planner query failure. The final package was written
to a same-directory temporary file, atomically renamed into place, and followed
by PostgreSQL restarts. Both final postmasters then passed the gates above.

## Release Boundary

Commons should migrate supported query families to ordinary SQL predicates and
scalar `evoke_query(...)`, while retaining explicit-hit calls only where the
application deliberately needs a hit set. Unsupported query shapes fail closed
instead of executing the scalar marker once per row. No root rebuild, new
artifact, second worker lifecycle, or index-format compatibility layer was
introduced by this release.
