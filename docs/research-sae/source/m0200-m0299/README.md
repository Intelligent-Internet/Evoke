# M217-M233 Retained Research Source

Archived from the local workspace on 2026-10-10. These files retain historical
II-42 names and experiment identifiers; they are not Evoke runtime modules,
supported command-line tools, or product-release inputs.

## Source Map

| Experiment | Retained source | Report |
| --- | --- | --- |
| M217 | External pairwise-head runner; not present in this snapshot | [Pseudo global head](../../reports/m0200-m0299/ii42-m217-rankt5-pseudo-global-head-report.md) |
| M218-M219 | `research_sae_m218_safe_rankt5_global_head.py`; M219 reuses this family | [Safe head](../../reports/m0200-m0299/ii42-m218-safe-rankt5-global-head-report.md), [transfer](../../reports/m0200-m0299/ii42-m219-rankt5-transfer-global-head-report.md) |
| M220-M222 | Runtime-confidence, row-aware router, and row-risk pseudo-label modules | [M220](../../reports/m0200-m0299/ii42-m220-runtime-confidence-gate-report.md), [M221](../../reports/m0200-m0299/ii42-m221-row-aware-router-report.md), [M222](../../reports/m0200-m0299/ii42-m222-row-risk-pseudolabel-head-report.md) |
| M223-M226 | Supervised bridge, teacher ensemble, action distillation, and score geometry modules | [M223](../../reports/m0200-m0299/ii42-m223-supervised-teacher-bridge-head-report.md), [M224](../../reports/m0200-m0299/ii42-m224-teacher-ensemble-bridge-head-report.md), [M225](../../reports/m0200-m0299/ii42-m225-action-distill-bridge-head-report.md), [M226](../../reports/m0200-m0299/ii42-m226-score-geometry-bridge-head-report.md) |
| M227-M229 | Qrels-free geometry, row-aware geometry, query-rich score geometry modules | [M227](../../reports/m0200-m0299/ii42-m227-qrels-free-geometry-preflight-report.md), [M228](../../reports/m0200-m0299/ii42-m228-row-aware-geometry-preflight-report.md), [M229](../../reports/m0200-m0299/ii42-m229-query-rich-score-geometry-report.md) |
| M230-M231 | Unified dual-output encoder, including the sequence adapter mode; launchers under `launchers/` | [M230](../../reports/m0200-m0299/ii42-m230-unified-dual-output-model-plan.md), [M231](../../reports/m0200-m0299/ii42-m231-sequence-branch-adapter-report.md) |
| M233 | EmbeddingGemma teacher-preflight module and launcher | Source-only snapshot; no result report or completion receipt was available in this workspace inventory |

M230 retains its historical `plan` filename because it is an append-only
design/result ledger with existing citations. Its later A/B/C results and the
M231 report supersede the initial proposed schedule. M232 is mentioned as a
possible next step, not represented here as implemented or completed.

## Replay Limits

This snapshot is **not a self-contained training environment**. Direct imports
of the following modules are not supplied by the current Git checkout:

- `research_sae_m1970_compact_dense_cascade_oracle`
- `research_sae_m1971_compact_dense_fusion_oracle`
- `research_sae_m201_top256_distill_projection`
- `research_sae_m202_shared_top256_residual_head`
- `research_sae_m210_nonlinear_mini_vector_head`
- `research_sae_m211_query_gated_mini_vector`
- `research_sae_m213_qrels_safe_mode_selector`
- `research_sae_m216_dense_pairwise_global_head`
- `research_sae_m217_rankt5_pairwise_global_head`

The first two names occur in the original [source archive manifest](../manifest.jsonl).
The remaining exact module versions must be recovered from their recorded
experiment source bundles before replay; similarly numbered old modules are
not substitutes. Transitive dependencies, model revisions, datasets, splits,
and environment versions must also be checked. No missing module was replaced
with a fallback during this archival pass.

Launchers retain host paths and installation/queue behavior as historical
evidence. In particular, M233 references unpinned Transformers source and an
unbounded resource wait; it must not be run as a current approved controller.
Syntax checking a launcher does not make these behaviors safe or reproducible.
New II-42 work remains restricted to explicitly admitted physical GPU 0 unless
the user changes that policy.

Source bytes and the three original-to-retained path mappings are recorded in
the [workspace checksum inventory](../snapshots/2026-10-10-workspace.json).
The report metrics were not rerun or independently requalified during archival.
