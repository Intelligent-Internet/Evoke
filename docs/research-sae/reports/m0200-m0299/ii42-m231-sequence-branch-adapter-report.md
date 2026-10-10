# II-42 M231 Sequence Branch Adapter Report

## Context

M229B is the saved query-performance sidecar milestone before query-cost
control: `query_rich_512x2 + m220c_joint,w=4.5`, mean over seeds 301/313/337
MAP 0.366415 / Recall 0.639995, deltas vs PPLX dense +0.064950 / +0.029258,
with no dense-unsafe rows.

M230 tested whether this sidecar behavior can be compressed into a unified
dual-output encoder while keeping the NG80 sparse-S posting path frozen.

| Endpoint | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M229B sidecar mean | 0.366415 | 0.639995 | +0.064950 | +0.029258 | none |
| M221 qrels-free retained endpoint | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M230A frozen mean readout | 0.300202 | 0.617215 | -0.001263 | +0.006479 | Arguana, FiQA, TREC-COVID |
| M230B richer frozen readout | 0.290998 | 0.602481 | -0.010466 | -0.008256 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| M230C token-summary MLP | 0.297280 | 0.609905 | -0.004185 | -0.000832 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |

M230A/B/C all preserved posting invariance, but none recovered the mini-dense
quality. M230C improved over M230B but stayed below M230A and repeated the
expanded unsafe-row pattern. This closes the plain frozen readout and flattened
feature-adapter path.

## M231A Plan

M231A is a structural test before guarded trunk adaptation. It keeps the NG80
SAE/posting path frozen, reuses the M230C frozen token-summary feature cache,
and replaces the flat MLP readout with a token-aware branch adapter.

The adapter reads top-8 frozen token hidden vectors selected by hidden norm,
plus pooled/global summary features. It applies a learned attention pooling
branch separately for document and query mini-dense heads, then emits a 32d
INT4 correction vector. The posting head still reads the original frozen SAE
path, so posting invariance must remain exact.

Active run:

- tmux session: `ii42_m231a_sequence_adapter_gpu0`
- Output directory:
  `/home/huoju/leask/runs/ii42-m231-sequence-branch-adapter-v1/runs/m231a-ng80-s-sequence-adapter-32d-int4-gpu0-v1`
- Log:
  `/home/huoju/leask/runs/ii42-m231-sequence-branch-adapter-v1/logs/m231a-sequence-branch-adapter-gpu0-v1.log`
- Script:
  `/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts/research_sae_m230_unified_dual_output_encoder.py`
- Launcher:
  `/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts/launch_m231a_sequence_branch_adapter.sh`
- Feature cache:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/feature-cache/m230c-ng80-s-token-summary-v1`
- Head kind: `token_adapter`.
- Feature taps: `mean,max,first,token_topk_hidden,sparse_summary`.
- Teacher source: M229B seed301/313/337 `query_rich_512x2` checkpoints.
- Fixed weights: `0.25,0.5,0.75,1.0,1.25,1.5`.
- Routers: `fixed,family_dense_half,m220c_joint`.

Promotion gate:

- Posting invariance remains exact.
- Best deployable variant beats M221 MAP and Recall, or materially improves
  over M230A while not expanding unsafe rows.
- Strong promotion requires dense-safety and recovery of a meaningful share of
  the M229B sidecar gain.

Stop condition:

If M231A remains below M221 or keeps the five-row unsafe pattern, the frozen
NG80 representation is exhausted for this compression route. The next credible
experiment is guarded top-block adaptation with an explicit sparse-posting
preservation loss, not more frozen feature/readout variants.

## M231A Result

M231A completed on lambda2 GPU0 on 2026-10-02. It reused all seven M230C
feature-cache files and preserved posting invariance, but it did not improve
the unified mini-dense result.

Best deployable variant:

| Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed | 0.5 | 0.294077 | 0.605069 | -0.007388 | -0.005667 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |

Row deltas at the best point:

| Dataset | dMAP vs dense | dRecall vs dense | Safe vs dense |
| --- | ---: | ---: | --- |
| Arguana | -0.004129 | -0.002141 | no |
| FiQA | -0.073216 | -0.063822 | no |
| NFCorpus | -0.011015 | -0.020471 | no |
| SciDocs | -0.016158 | -0.023117 | no |
| SciFact | +0.035374 | +0.016000 | yes |
| TREC-COVID | -0.012120 | -0.010874 | no |
| Webis-Touche2020 | +0.029547 | +0.064753 | yes |

Comparison:

| Endpoint | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 retained qrels-free | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M230A frozen mean readout | 0.300202 | 0.617215 | -0.001263 | +0.006479 | Arguana, FiQA, TREC-COVID |
| M230C token-summary MLP | 0.297280 | 0.609905 | -0.004185 | -0.000832 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |
| M231A token adapter | 0.294077 | 0.605069 | -0.007388 | -0.005667 | Arguana, FiQA, NFCorpus, SciDocs, TREC-COVID |

Decision:

M231A confirms that frozen NG80 sparse-S representations are not enough for
this compression route, even with a token-aware branch adapter. The route loses
too much MAP/Recall and remains row-unsafe. Do not run more frozen
feature/readout variants.

The next structural route is M232 guarded top-block adaptation. It must include
an explicit sparse-posting preservation loss before launch: posting support
overlap, posting score drift, active-count drift, and macro retrieval safety
must be measured against the frozen NG80 baseline. M232 should not start until
that guarded script is implemented and syntax-checked.
