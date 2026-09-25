# II-42 M200 Top256 Tiny INT4 Residual Floor Report

Date: 2026-09-22

## Question

Evaluate whether `score top256 + 32d INT4` is a viable next-step design for the
next-generation retrieval stack:

- frozen first-stage candidate source: M190 latent postings + BM25;
- no ANN path;
- only rerank the baseline tail after the protected head;
- target: an extremely small dense residual vector that can be emitted together
  with the sparse posting head by a future dual-head encoder.

This run is a no-training floor. It tests whether the target has enough signal
before spending training budget.

## Prior Context

M1970/M1971 showed that compact dense correction is structurally useful inside a
fixed candidate set, with 256d packed INT4 retaining most of the exact dense tail
gain. M1972/M1973 then tightened the inference contract: correction should be
local to the baseline top256, not applied across the full candidate union.

Earlier dense-tail training attempts such as M401 and M730 should not be treated
as identical to this route. They failed broader learned dense-tail or query-side
compiler gates. M200 is narrower: it first asks whether a top256-local document
code has enough recoverable signal before training a dual-head encoder.

## Implementation Notes

Added `scripts/research_sae_m200_top256_tiny_int4_floor.py`.

The first version still used the old full-corpus dense score matrix path for the
exact ceiling. That reproduced the FiQA OOM failure mode from the older M1971
screening script on a 24GB GPU. The script was corrected to compute exact dense
scores only for each query's baseline top256 window.

Execution was moved from spark-1 to lambda2 as requested:

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Runner environment: `/home/huoju/venv/vllm-env`
- Output root:
  `/home/huoju/leask/runs/ii42-m200-tiny-residual-floor-v1`

Spark-1 was used only as the source for existing frozen artifacts. No experiment
compute was run on spark-1 after the handoff to lambda2.

## Full7 Results

All results are M200B top256-only. The correction scope is baseline ranks
`21..256`, with ranks `1..20` protected. Dense weight is fixed at `0.5`; no qrel
selection is used.

| Dataset | Exact dMAP | Exact dRecall | 32d dMAP | 32d dRecall | 64d dMAP | 64d dRecall | 256d dMAP | 256d dRecall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | +0.000348 | +0.004996 | -0.000017 | +0.002855 | +0.000142 | +0.004283 | +0.000336 | +0.004283 |
| FiQA | +0.003060 | +0.024603 | +0.001764 | +0.015400 | +0.002180 | +0.018220 | +0.002804 | +0.022852 |
| NFCorpus | +0.003306 | +0.009822 | -0.001451 | -0.004415 | -0.000884 | -0.002358 | +0.002343 | +0.007679 |
| SciDocs | +0.002452 | +0.021550 | +0.000244 | +0.004700 | +0.000847 | +0.008950 | +0.001911 | +0.016633 |
| SciFact | +0.000175 | +0.003333 | -0.000142 | -0.004000 | -0.000148 | -0.004000 | +0.000075 | +0.003333 |
| TREC-COVID | +0.009286 | +0.010116 | +0.002660 | +0.002608 | +0.003971 | +0.003845 | +0.008181 | +0.009005 |
| Webis-Touche2020 | -0.001743 | -0.001706 | -0.000087 | +0.004414 | +0.000241 | +0.005064 | -0.000304 | +0.000415 |

Macro over the seven rows:

| Variant | Bytes/doc | dMAP | dRecall | Unsafe rows |
| --- | ---: | ---: | ---: | --- |
| Exact top256 dense | full dense | +0.002412 | +0.010388 | Webis has negative exact correction |
| 16d INT4 | 12 | -0.000075 | +0.000110 | NFCorpus, SciDocs, SciFact |
| 32d INT4 | 20 | +0.000424 | +0.003080 | NFCorpus, SciFact |
| 64d INT4 | 36 | +0.000907 | +0.004858 | NFCorpus, SciFact |
| 256d INT4 | 132 | +0.002192 | +0.009171 | none |

Observed scoring p95 for the local top256 scoring kernel stayed below 0.32 ms
per query in this Python/NumPy harness. The candidate source remains the
dominant latency surface.

## Interpretation

`32d INT4` is a good design target, but it is not yet a safe no-training
deployment target. On the full7 gate it has positive macro Recall, but its MAP
retention is weak and it regresses NFCorpus and SciFact. That means the current
PCA-style 32d projection is losing local rank information that matters for
scientific/biomedical style corpora.

`64d INT4` is better but still not robust enough as a direct product default,
because it shares the same unsafe rows. It is a plausible distillation target,
not a final target yet.

`256d INT4` remains the current safe teacher. It preserves most of the top256
exact dense correction gain with no unsafe full7 rows and a small enough payload
for top256-local scoring.

Webis is special: exact dense top256 correction itself is negative on MAP and
Recall. This means a future residual head needs an activation/gating policy; it
should not always apply dense correction blindly on every domain/query.

## Decision

Do not immediately train a production 32d INT4 dual-head model.

Proceed structurally in this order:

1. Freeze the M200B top256-only protocol as the evaluation target.
2. Treat 256d INT4 as the teacher/reference residual head.
3. Train a compact residual head against the top256-local teacher, first at 64d
   as the minimum plausible safe capacity.
4. Add a query/domain activation gate before attempting 32d as a product target.
5. Revisit 32d only after the learned projection/head fixes NFCorpus/SciFact
   sign errors under the same top256-only gate.

## Next Experiment

M201 should generate a top256-local distillation dataset:

- input candidates from frozen M190 latent postings + BM25;
- labels from exact dense top256 correction, not full-corpus dense retrieval;
- protect top20 from dense perturbation;
- include a no-op/gating label for queries where exact dense correction is
  neutral or harmful;
- train the residual vector head with listwise/pairwise ordering loss over ranks
  `21..256`;
- report 64d and 32d INT4 separately, with 256d INT4 as teacher.

The core pass condition should be strict: no more NFCorpus/SciFact regressions
worse than `-0.002`, and at least 60% MAP/Recall retention versus the 256d
teacher on macro full7.
