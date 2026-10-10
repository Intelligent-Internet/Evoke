# Research Workspace Archive: 2026-10-10

## Scope and Product Boundary

This archive retains 105 previously untracked research files from the local
workspace: 15 reports, 62 source/configuration/launcher files, and 28 tests,
totalling 1,137,259 bytes. The
[checksum inventory](../../source/snapshots/2026-10-10-workspace.json) records
their original paths, retained paths, sizes, and SHA-256 digests. Payload bytes
are preserved, including historical names and experiment assumptions; the new
archive notes and index updates are separate editorial additions.

The product baseline is commit
`2b1bd90441893936ea912eeb633bdef853622f6c` in `Evoke-dev`. This archival batch
does not change the extension's C implementation, SQL/API, build or packaging
configuration, release workflows, model assets, index lifecycle, or deployment
settings. It does not characterize earlier commits as research-only.

## Retained Work

| Area | Evidence retained | Interpretation |
| --- | --- | --- |
| NG80-NG83 | Campaign/data preparation, dense and sparse fitting, fusion, diagnostic and audit runners, matching tests | Research tooling remains at its existing `scripts/` and `tests/` paths to preserve imports and frozen source-copy conventions. |
| NG84-NG87 | Teacher preparation, diagnosis, intervention, training and evaluation runners, matching tests | An archive of the explored supervision and loss pipeline, not a new training run or product promotion. |
| NG88 | Evaluation/math sources, tests, and corrected [progress record](../ng0001-ng0099/ng0088-progress.zh.md) | Partial evaluation stopped at resource admission; completion is not claimed. |
| M217-M225 | Teacher, pseudo-label, router, ensemble and action-distillation reports and available sources | Results are retained as reported; archival did not rerun their metrics. |
| M226-M229 | Score-geometry, qrels-free/row-aware preflights and [query-rich sidecar results](../m0200-m0299/ii42-m229-query-rich-score-geometry-report.md) | Research milestones and promotion terminology describe experiment contracts, not a deployed or independently qualified product. |
| M230-M231 | [Dual-output design/result ledger](../m0200-m0299/ii42-m230-unified-dual-output-model-plan.md), [sequence-adapter report](../m0200-m0299/ii42-m231-sequence-branch-adapter-report.md), available sources and launchers | Preserve the compression results, including negative results. Later results supersede the initial M230 schedule. M232 is a proposed follow-up, not a completed experiment here. |
| M233 | EmbeddingGemma teacher-preflight source and launcher | Source-only evidence; no completion report or receipt was available in this workspace inventory. |
| Block-Max WAND prototype | [Illustrative source and limitations](../../source/prototypes/README.md) | Synthetic exploration, not validated product performance or a correctness proof. |

The [M-series source map](../../source/m0200-m0299/README.md) links individual
reports and identifies missing dependencies. M233 and the standalone WAND
prototype were relocated out of the product tooling directory. NG runners were
not relocated: their imports and frozen experiment layouts depend on the
existing paths, and CI excludes their tests from product unit-test selection.

## Verified NG88 State

A read-only check of the retained remote controller receipt on 2026-10-10
confirmed `passed=false`, completed phases `encode-base-0` and `encode-base-1`,
and `ResourceBusy: resources busy; no worker started`. The receipt was written
at 2026-09-20 07:54:23 UTC, with automatic retry and automatic training disabled.
The two base-document shards cover 290,609 documents. Base-query encoding,
candidate A/B evaluation, rankings, and final review are not complete.

The input-manifest digest and both completion-receipt digests match the values
recorded in the [progress record](../ng0001-ng0099/ng0088-progress.zh.md).
This is receipt verification, not a fresh full-payload integrity audit or a
re-evaluation of the models. This task did not restart or modify the run.

## Reproduction and Validation

The original 1,595-file external source manifest remains unchanged. This new
snapshot is not a replacement for that archive or a self-contained training
environment. Nine direct M-series imports are absent from the current checkout;
their exact source versions, transitive dependencies, datasets, model revisions,
splits and runtime environment must be restored before replay. See the
[replay limits](../../source/m0200-m0299/README.md#replay-limits).

Historical launchers retain host-specific paths and installation/queue
behavior, including M233's unpinned dependency and unbounded wait. They are
preserved for inspection, not approved for execution. No launcher, GPU workload,
locked-test inference, deployment or release synchronization was started by
this archival task.

The NG80-NG88/review unit tests passed locally: **284 passed, 7 skipped**.
All seven skips explicitly reserve Torch execution for Linux/Lambda2. Product
Python unit tests, selected with the CI exclusions for NG research, passed:
**677 passed, 1 skipped**; the skipped page-cache test requires
`POSIX_FADV_DONTNEED`, unavailable on this host. The initial attempt in the research environment
could not collect product tests because `psycopg` was absent; the successful
run used the existing system Python with that dependency, without modifying
the research environment.

The product-convergence inventory, local navigation targets, payload digests,
Python AST parsing and shell syntax checks passed. The full staged whitespace
check reports 16 existing trailing-whitespace lines in the byte-preserved WAND
prototype only; all other files pass. No whitespace exemption or CI rule was
added. C builds, database integration and archived model metrics were not rerun
for this research-only archive.

These checks validate source organization and utilities, not the accuracy of archived models,
the completeness of remote evaluations, or product release qualification.
The checksum inventory covers only the 105 original payload files, not this
new report or the new navigation notes.

Large datasets, checkpoints and full run outputs remain in external artifact
storage. Small source files, research conclusions and traceability records are
retained in Git. This archive does not authorize deleting any external evidence.
