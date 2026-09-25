# CQ-3 full PubMed same-root qualification

These SQL fixtures qualify the filtered exact-BMP scorer against one immutable
current-format PubMed root. They intentionally use one physically dispersed
0.6%, 10%, and 50% filter generated from `pmid`, plus an unfiltered control, so
baseline and candidate binaries see identical query and membership inputs.

Run them only through `scripts/qualify_same_root_binary_ab.py`. The runner uses
a private mount namespace, guards the production postmaster and installed
library, refuses an active index build, freezes maintenance through a
postmaster-only override during both measurements, and restores the isolated
baseline or candidate requested by the caller without changing its persistent
maintenance configuration.
The caller must pass
`--qualified-index=bench.pubmed_full_v2_idx`. Before setup, the runner requires
that index and its heap to be permanent, query-ready, and backed by a non-empty
Evoke generation with immutable segments and posting records. An unlogged
surface is invalid because crash recovery can reset the heap and index while
leaving stale catalog statistics that resemble a completed build.
The fixtures also record bounded generation identity before and after each
binary. Any root, contract, accelerator-source, or relation-size change fails
the comparison, so background publication cannot contaminate a same-root A/B.

The comparison fails closed unless all four routes return 50 hits, every rank
has the same document ordinal and `float8send(score)` bytes, the 0.6% filter
performs less bound and posting work than the unfiltered control, and repeated-
query PostgreSQL memory contexts and Linux backend RSS reach bounded plateaus.
Every filtered control must remain on the exact BMP route without fallback,
stay within the product's 64 MiB statement-work limit, and avoid considering
blocks or documents outside the allowed subset. Product latency SLOs remain in
the normal-route Commons matrix rather than this forced exact-path fixture.
These are physical qualification fixtures, not application-level Commons
predicates; the full Commons matrix remains a separate CQ-7 gate.

Pass `--normal-probe-sql=normal_probe.sql` to run the candidate binary through
the automatic product router and forced direct-row and transpose diagnostics.
Each mode measures all four filter sets and records ranked-prefix probe work,
forward layout work, exact-BMP work, all accelerator memory owners, backend
RSS, and overlap with the forced exact candidate. The forced modes are physical
A/B controls only. The automatic same-root matrix remains the CQ-3A
route-selection authority; a forced route alone cannot promote a product path.

Use `run_product_auto.sql` with `compare_product_auto.sql` when a change affects
the automatic product route itself. The runner executes both binaries on the
same persistent root, requires all 200 ranked TIDs and score bytes to match,
keeps five latency samples per filter and binary, and verifies that generation
identity and relation bytes remain stable. This is the preferred fast gate for
route-cost changes; the forced exact-BMP fixture remains the representation and
memory qualification gate.

After the streamed full-root rebuild finishes, run
`postbuild_equivalence.sql` before any swap. The baseline root was encoded by
ORT 1.26 and the candidate by ORT 1.29, so this is a runtime-migration gate,
not the same-input writer byte oracle. With the semantic accelerator disabled,
it requires five complete top-100 result sets, exact membership, at most one
rank of document movement, and at most 0.2% relative score drift. Both roots
are queried twice and must be independently rank- and score-bit stable. The
stream writer remains subject to the separate C oracle that compares the old
and new serializers with identical lexical and semantic inputs. The default
roots are `bench.pubmed_full_v2_idx` and
`bench.pubmed_full_cq3e_streamed_idx`; pass `-v baseline_index=... -v
candidate_index=...` to qualify another pair.
