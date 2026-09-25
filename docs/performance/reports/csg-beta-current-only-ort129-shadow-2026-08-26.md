# Shadow Current-Only ORT 1.29 Closure

Date: 2026-08-26

Status: completed for immutable packaging, current-only reader removal, and
same-root Shadow deployment. This report does not close the remaining Commons
latency, concurrency, cancellation, or full-root RSS gates.

## Source And Package

The current-only Linux PostgreSQL 18 package was built from source commit
`852e28f1adf9643a74725cfaf052dea3af080ef4`. The source archive was clean and
recorded as `source-archive`; the build did not depend on an untracked checkout.

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `158abc76ee4ffbd108f176d3cea2ba8c3b2dc0064cac0bff3d344a0cd0b04e68` |
| Package archive | `8c36c8a84a23313f8120e8b32227bdcd1fbd0308367cd0969fd7dc9f1d8829c0` |
| `evoke.so` | `4e7d6b6d5ef412e96737e375dc3801b88540b2117eef9596e72af300315ac1ff` |
| ONNX Runtime 1.29.0 | `5715f06d8992ca8eeeddcce43df3a7d38f97d537052126f558e912cb312460ca` |
| `evoke.control` | `ca157ce9a4c1f597137975635928998264ea22fa5d9fa05d2119010eda7ab99b` |
| Install SQL | `a755eb5d40bd87360a26a2a3c913b74cd9d6e393e540eb76a387a23c51eccdee` |
| Model manifest | `dd0993f0638a3f683fbb6475a8509c228d8f6547aab90a0b762b1ff6fd7d0041` |
| Document compiler | `9e04d78c9cb14a73dbda4dba66d0e14ce0840d76ff6ab2ff6bdee8e4671cb79f` |
| Query compiler | `d93d38d1e9d34e26a08b9b87ccf4013a2d7357002c09e797abbe4aea7349f9b3` |

The staged model contained 13 manifest-authorized artifacts and 400,908,113
bytes. An obsolete installed manifest backup was moved into the deployment
backup. The installed model checkout then matched the package byte-for-byte and
passed milestone-model validation.

Package-isolated lifecycle qualification passed 71/71 gates using the bundled
binary, ORT runtime, SQL, and model. The repository validation for the reader
cleanup also passed the C build, CTest, 437 Python tests with one skip, product
convergence inventory, and `git diff --check`.

## Current-Only Reader Boundary

The package removes query support and synthetic success fixtures for:

- scope codecs v2 through v5;
- semantic accelerator policies 5 and 6.

The only accepted scope codec is v6, and the only queryable accelerator policy
is 7. The accelerator directory retirement decoder remains because directory
format reclamation is separate from query-policy compatibility.

The fixed-32-extent synthetic compatibility contract was not part of this
change. Current term-COW leaves remain variable-width and can legitimately have
an actual maximum of 32.

## Shadow Deployment

Shadow `ii_dev` was the only database using Evoke. Before deployment, all 14
indexes were valid and ready; all nine Commons product accelerators used policy
7, and all six scoped product roots used current scope v6.

PostgreSQL was stopped, the immutable package was installed, and PostgreSQL was
restarted. The new postmaster started at `2026-08-26 07:05:40.656052+00`. The
service is active, extension version is 0.2.4, runtime build information reports
API 29, and the linked runtime probe reports ORT 1.29.0.

All 14 root generation identifiers were unchanged before and after deployment.
No index rebuild, semantic inference, root publication, or accelerator
republication occurred. The existing roots were therefore reused exactly as
intended by the unchanged disk format.

The nine Commons roots were explicitly preloaded after restart. Their query
metadata became warm without warming additional pages, confirming that the
resident artifacts and page cache remained usable across the binary upgrade.

## Query Smoke

Pre- and post-deployment ArXiv unfiltered and date-category searches returned
50 hits with identical top-k signatures. The filtered path had zero predicate
violations and passed its oracle. Post-deployment PubMed smoke also completed
with stable results and zero predicate violations.

| Query | First query | Warm p50 | Correctness |
| --- | ---: | ---: | --- |
| ArXiv unfiltered | 2,699.280 ms | 242.546 ms | stable; pre/post signature identical |
| ArXiv date + category | 154.587 ms | 156.526 ms | oracle passed; zero violations |
| PubMed unfiltered | 7,145.679 ms | 417.867 ms | stable; zero violations |
| PubMed date + category | 4,698.278 ms | 450.888 ms | oracle passed; zero violations |

The deployment smoke is successful for package identity, runtime availability,
root reuse, and query correctness. It does not pass the complete release
performance matrix: restart-first-query limits still fail for ArXiv and
PubMed, and PubMed warm unfiltered remains above the 300 ms application gate.
Those are retained CQ-4/CQ-7 performance items, not ORT or format migration
failures.

## Remaining Boundary

The two bounded closure tasks are complete. The next-version `QDIR-RSS-1` item
remains open: the accelerator directory owns approximately
`4 * (document_count + forward_chunk_count)` bytes per concurrent query. It is
bounded and released, but production-sized concurrency RSS still needs a
separate gate. This deployment does not reopen performance micro-tuning.
