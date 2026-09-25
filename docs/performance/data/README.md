# Benchmark Data Layout

This directory is organized around the current benchmark authority and
its raw backing archive.

Quick links:

- [canonical/](canonical/)
- [diagnostics/](diagnostics/)
- [raw/](raw/)

## `canonical/`

These are the benchmark raw data files that should be treated as the
current project reference:

- `official-beir-ids-current-2026-04-02.json`
  Current PG18 BEIR run for the `ids` path using the refreshed
  `2026-04-02` Google Cloud rerun.
- `official-beir-text-current-2026-04-02.json`
  Current PG18 BEIR run for the `text[]` path using the refreshed
  `2026-04-02` Google Cloud rerun.

## `diagnostics/`

These files contain the broader comparison matrices, supporting
diagnostics, and historical side studies that sit next to the canonical
per-path raw data:

- `official-beir-upstream-current-2026-04-02.json`
  Current upstream `bm25s` slice carried forward into the refreshed
  `2026-04-02` PG18 matrix bundle.
- `official-beir-pg18-comparison-current-2026-04-02.json`
  Refreshed PG18 comparison file combining upstream `bm25s`,
  `evoke ids`, and `evoke text[]`.
- `pg18-beir-extension-matrix-current-2026-04-02.json`
  Refreshed `15 x 5` PG18 matrix combining upstream `bm25s`,
  `evoke ids`, `evoke text[]`, `pg_search`, and
  `vchord_bm25`.
- `official-beir-local-upstream-current-2026-03-21.json`
  Historical localhost upstream rerun retained because the later
  `pg_bm25s` study still depends on that same-machine reference.
- `official-beir-local-machine-comparison-current-2026-03-21.json`
  Historical merged localhost comparison retained for the same reason.
- `pg-bm25s-focused-comparison-2026-03-21.json`
  Historical same-machine `pg_bm25s` comparison subset.
- `pg-extension-focused-comparison-current-2026-03-21.json`
  Historical merged comparison file combining local upstream,
  `evoke`, and `pg_bm25s` on the completed localhost subset.
- `native-orderby-focus-topk1000-2026-03-21.json`
  Focused localhost benchmark comparing the native `<=>` ordered-scan
  path with the existing `rowset` APIs at `top_k = 1000`.
- `native-orderby-focus-topk10-2026-03-21.json`
  Focused localhost benchmark comparing the same paths at `top_k = 10`
  to show how the ordered-scan overhead behaves at smaller limits.
- `scifact-ids-spotcheck-2026-03-21.json`
- `scifact-text-spotcheck-2026-03-21.json`
- `scifact-upstream-spotcheck-2026-03-21.json`
- `convergent-query-states-2026-07-31/`
  Three independent isolated PostgreSQL 18 runs of the native static,
  three-segment, and workload-folded query-state matrix. This is local
  structural evidence, not the qualification-host release matrix.

The refreshed `2026-04-02` PG18 matrix files are now the preferred
reference when the question is "what is the current cross-engine
benchmark status?".

That refresh only replaced the two evoke columns. The raw archive keeps its
original pre-rebrand result filenames so the historical archive remains
byte-for-byte traceable.

- `30` new raw cells from the `2026-04-02` Google Cloud rerun supply
  `evoke ids` and `evoke text[]`
- the other `45` raw cells (`upstream bm25s`, `pg_search`,
  `vchord_bm25`) are intentionally carried forward from the stable
  `2026-03-31` PG18 matrix

Most older same-machine engine-comparison raw files were removed from
the working tree once the PG18 matrix became the project-wide
authority. The retained `2026-03-21` localhost comparison files remain
because the `pg_bm25s` historical study still depends on them and they
have not been replaced by a newer like-for-like rerun.

The current rolled-up matrix has been checked against the backing raw
archives cell by cell:

- `75/75` dataset-engine cells matched
- matching fields:
  - `stats`
  - `build_ms`
  - `query.count`
  - `query.qps`

## `raw/`

These directories contain the raw per-task benchmark artifacts that sit
behind the current rollups.

- `pg18-beir-extension-matrix-2026-03-31/`
  Full raw archive for the final `15 x 5` PG18 extension matrix,
  including:
  - `results/<dataset>/<engine>.json`
  - `manifest.json`
  - `launch.json`
  - `summary.json`
  - `status-matrix.json`
  - `status-matrix.md`
- `pg18-beir-psql-hardening-partial-2026-03-31/`
  Partial raw archive for the
  `chore/bm25s-simd-pg-common-hardening` branch rerun using the same
  PG18/GCP setup as the current matrix authority, including:
  - `results/<dataset>/<engine>.json`
  - `manifest.json`
  - `launch.json`
  - `summary.json`
  - `status-matrix.json`
  - `status-matrix.md`
  - `comparison-current.json`
  - `comparison-so-far.md`
  This branch-evaluation rerun was stopped early and is retained for
  PR review and follow-up analysis. It does not replace the
  `2026-03-31` full matrix as the current project-wide authority.
- `pg18-beir-psql-only-tantivy-2026-04-02/`
  Raw archive for the follow-up Google Cloud rerun that refreshed only
  the evoke ids and text[] columns. The per-dataset result files keep
  their original pre-rebrand filenames inside this raw archive.
  - `manifest.json`
  - `launch.json`
  - `summary.json`
  - `status-matrix.json`
  - `status-matrix.md`
  - `comparison-vs-doc-baseline.json`
  - `comparison-vs-doc-baseline.md`
- `tantivy-followup-local-main-vs-branch-2026-04-02/`
  Full local `main` vs `codex/tantivy-followup-planning` comparison
  used to close out the Tantivy-inspired optimization study, including:
  - `main.json`
  - `branch.json`
  - `summary.json`
  - `summary.md`
  - `README.md`

No separate general historical raw engine-comparison archive is kept in
the working tree. The retained 2026-03-21 localhost comparison JSONs
listed above are the exception because they still back the `pg_bm25s`
historical study.
