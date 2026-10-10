# II-42 M219 RankT5-Safe Transfer Global Head Report

Date: 2026-09-28

## Question

M218B confirmed that the safe RankT5-pseudo route has a seed-stable macro win
under the product contract, but the gain is small and still row-unsafe. M219A
tests whether the deployable student is under-transferring the useful teacher
signal by reducing the M210 distillation constraint and making the pairwise
training target more boundary-oriented.

## Protocol

Run family:
`m219a-<config>-pk16-ns32-nk96-blendd0p5-32d-int4-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base initialization: M210A residual MLP checkpoint
- Teacher cache: completed M217A RankT5 top256 cache
- Teacher policy: `blend_d0p5`
- Product contract: `32d INT4`, 20 bytes/doc payload, score top256 candidates
- Fixed runtime weights: `0.75, 1.0, 1.25, 1.5`
- Epochs: `16`
- Seed: `218`
- Pairwise target: `positive_k=16`, `negative_start_rank=32`,
  `negative_k=96`
- Config A: `distill_weight=0.05`
- Config B: `distill_weight=0.10`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Locked-test access: none

## Results

M219A completed on 2026-09-28 02:54 EDT. Both transfer configs failed to improve
over M218A or the M218B seed-stability mean.

Best deployable variant per config:

| Config | Best weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `distill_weight=0.05` | 0.75 | 0.300389 | 0.612722 | -0.001076 | +0.001986 | Arguana, FiQA, SciDocs, TREC-COVID |
| `distill_weight=0.10` | 1.00 | 0.300314 | 0.614345 | -0.001151 | +0.003608 | Arguana, FiQA, SciDocs, TREC-COVID |

Reference points:

| Run | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense |
| --- | ---: | ---: | ---: | ---: |
| M218A best | 0.302526 | 0.612611 | +0.001062 | +0.001875 |
| M218B mean best-per-seed | 0.302231 | 0.613316 | +0.000766 | +0.002579 |
| M219A `dw0p05` best | 0.300389 | 0.612722 | -0.001076 | +0.001986 |
| M219A `dw0p10` best | 0.300314 | 0.614345 | -0.001151 | +0.003608 |

Per-row anatomy for `dw0p05`, best fixed `weight=0.75`:

| Dataset | dMAP vs dense | dRecall vs dense | Dense-safe |
| --- | ---: | ---: | --- |
| Arguana | -0.018550 | -0.003569 | no |
| FiQA | -0.043182 | -0.036884 | no |
| NFCorpus | +0.005419 | +0.002149 | yes |
| SciDocs | -0.009714 | -0.017533 | no |
| SciFact | +0.039992 | +0.016667 | yes |
| TREC-COVID | -0.004371 | -0.004633 | no |
| Webis-Touche2020 | +0.022877 | +0.057703 | yes |

Per-row anatomy for `dw0p10`, best fixed `weight=1.0`:

| Dataset | dMAP vs dense | dRecall vs dense | Dense-safe |
| --- | ---: | ---: | --- |
| Arguana | -0.023928 | -0.004996 | no |
| FiQA | -0.036870 | -0.031838 | no |
| NFCorpus | +0.008770 | +0.006546 | yes |
| SciDocs | -0.008227 | -0.013333 | no |
| SciFact | +0.038039 | +0.016667 | yes |
| TREC-COVID | -0.003062 | -0.003693 | no |
| Webis-Touche2020 | +0.017224 | +0.055907 | yes |

## Interpretation

M219A is a negative transfer result. Lowering the distillation weight and moving
the pairwise contrast farther down the boundary did not recover additional MAP.
It improved Recall in the `distill_weight=0.10` config, but MAP fell below PPLX
dense and below the M218B mean.

This suggests the current limiter is not simply M210 regularization or weak
boundary pressure. The persistent pattern is now clear:

- The `32d INT4` payload is sufficient when qrels-shaped supervision exists.
- A simple fixed-weight top256 operator is sufficient once the head is good.
- RankT5-derived weak labels provide a stable but small macro lift.
- The current RankT5-safe labels do not supply enough useful signal for FiQA and
  SciDocs, and student transfer tweaks can make that worse.

## Decision

Do not run M219 seed stability. M219A did not improve over M218A or the M218B
mean. The current RankT5-safe weak-supervision route is close to its practical
limit under this objective family.

The next useful step should introduce a genuinely stronger supervision source
or row-aware pseudo-label objective, rather than more tuning of this transfer
setup. Good candidates are reranker ensembles, LLM pairwise judgments on the
top100 boundary, click-derived labels, or a qrels-free row-risk model that
explicitly suppresses unsafe pseudo-actions for FiQA/SciDocs-like rows.
