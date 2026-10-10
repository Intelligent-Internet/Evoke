# II-42 M218 Safe RankT5-Pseudo Global Head Report

Date: 2026-09-27

## Question

M217A showed that ordinary RankT5 pseudo-labels contain macro relevance signal,
but both the teacher ceiling and deployable `32d INT4` student were row-unsafe.
M218A tests whether conservative qrels-free teacher blending/gating plus longer
convergence can transfer a safer RankT5 signal into the 20-byte product
mini-vector head.

## Protocol

Run:
`m218a-safe-rankt5-policies-32d-int4-epoch16-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base initialization: M210A residual MLP checkpoint
- Teacher cache: completed M217A RankT5 top256 cache
- Candidate window: frozen top256 from the existing retrieval stack
- Product contract: `32d INT4`, 20 bytes/doc payload, score top256 candidates
- Teacher policies: `blend_d1p0`, `blend_d0p5`, `dense_top128_gate`,
  `rank_dense_consensus`
- Fixed runtime weights: `0.75, 1.0, 1.25, 1.5, 2.0`
- Epochs: `16`
- Seed: `218`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Locked-test access: none

## Status

M218A completed on 2026-09-27 04:24 EDT. Runtime was about 2324 seconds. No
M218A tmux session or Python process remained alive after completion; GPU0 was
idle. The run produced all four policy checkpoints and wrote `summary.json` and
`summary.md`.

Final policy losses:

| Policy | Samples | Loss | Pair loss | Distill loss |
| --- | ---: | ---: | ---: | ---: |
| `blend_d1p0` | 3722 | 0.026986 | 0.006903 | 0.107318 |
| `blend_d0p5` | 3722 | 0.047641 | 0.013043 | 0.160479 |
| `dense_top128_gate` | 3722 | 0.080193 | 0.023539 | 0.271882 |
| `rank_dense_consensus` | 3722 | 0.051311 | 0.014558 | 0.198325 |

## Results

Best deployable `student_32d_int4` variants:

| Policy | Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `blend_d0p5` | 1.25 | 0.302526 | 0.612611 | +0.001062 | +0.001875 | Arguana, FiQA, SciDocs, TREC-COVID |
| `dense_top128_gate` | 1.00 | 0.302363 | 0.612387 | +0.000898 | +0.001650 | Arguana, FiQA, SciDocs, TREC-COVID |
| `dense_top128_gate` | 1.25 | 0.302135 | 0.610751 | +0.000670 | +0.000014 | Arguana, FiQA, SciDocs, TREC-COVID |
| `blend_d0p5` | 1.00 | 0.301947 | 0.612434 | +0.000482 | +0.001697 | Arguana, FiQA, SciDocs, TREC-COVID |
| `blend_d0p5` | 1.50 | 0.301750 | 0.612056 | +0.000285 | +0.001319 | Arguana, FiQA, SciDocs, TREC-COVID |

M218A is the first qrels-free RankT5-pseudo student in this branch to beat PPLX
dense on both macro MAP and Recall, but it does not pass the row-safety
condition. No deployable variant had zero dense-unsafe rows.

Per-row anatomy for the best deployable variant, `blend_d0p5`, fixed
`weight=1.25`:

| Dataset | dMAP vs dense | dRecall vs dense | Dense-safe |
| --- | ---: | ---: | --- |
| Arguana | -0.023614 | -0.002141 | no |
| FiQA | -0.026694 | -0.037411 | no |
| NFCorpus | +0.008253 | +0.007165 | yes |
| SciDocs | -0.005359 | -0.012683 | no |
| SciFact | +0.039707 | +0.016667 | yes |
| TREC-COVID | -0.003926 | -0.003956 | no |
| Webis-Touche2020 | +0.019063 | +0.045482 | yes |

Best non-deployable teacher ceilings:

| Policy | Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `blend_d0p5` | 2.00 | 0.323842 | 0.620149 | +0.022377 | +0.009412 | Arguana, FiQA, SciDocs |
| `blend_d1p0` | 2.00 | 0.322954 | 0.620079 | +0.021490 | +0.009343 | FiQA, SciDocs |
| `blend_d0p5` | 1.50 | 0.321970 | 0.619703 | +0.020505 | +0.008966 | Arguana, FiQA, SciDocs |
| `blend_d1p0` | 1.50 | 0.320692 | 0.619167 | +0.019227 | +0.008431 | FiQA, SciDocs |

No teacher-ceiling variant was row-safe. `blend_d1p0` removes Arguana from the
best teacher unsafe set, but FiQA and SciDocs remain below the dense row floor.

## Interpretation

M218A is a partial positive result. Conservative teacher policies and longer
training recover enough signal for a qrels-free `32d INT4` student to edge past
PPLX dense at macro level, but the improvement is carried by NFCorpus, SciFact,
and Webis. The same hard rows continue to lose: Arguana, FiQA, SciDocs, and
TREC-COVID.

The failure is not primarily convergence. The student tracks the available
teacher signal better than M217A, yet the teacher ceilings themselves are still
row-unsafe. That means a runtime gate over the current teacher family can reduce
damage only by suppressing corrections; it cannot create the missing FiQA and
SciDocs relevance signal.

## M218A Decision

Do not launch M218 seed stability. The pass condition required a deployable
`32d INT4` student that beats PPLX dense on macro MAP and Recall with no
dense-unsafe rows, and M218A misses row safety.

The next justified step should change the weak-supervision source or add a
stronger row-aware safety target before student training. A narrow runtime gate
is not the main route because the best teacher ceilings are still unsafe on
FiQA and SciDocs.

## M218B Seed Stability

After M218A, the continuation criterion was relaxed to a macro-first exploratory
criterion: row failures are tolerated if all seeds beat PPLX dense on macro MAP
and Recall and mean dMAP is at least `+0.0005`. M218B repeats the most useful
M218A policies, `blend_d0p5` and `dense_top128_gate`, over seeds `301`, `313`,
and `337`.

Run family:
`m218b-seed-<seed>-safe-rankt5-32d-int4-epoch16-gpu0-v1`

Best deployable variant per seed:

| Seed | Policy | Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 301 | `dense_top128_gate` | 1.00 | 0.301940 | 0.613415 | +0.000475 | +0.002678 | Arguana, FiQA, SciDocs, TREC-COVID |
| 313 | `blend_d0p5` | 1.25 | 0.302744 | 0.613498 | +0.001279 | +0.002761 | Arguana, FiQA, SciDocs, TREC-COVID |
| 337 | `blend_d0p5` | 1.25 | 0.302011 | 0.613035 | +0.000546 | +0.002298 | Arguana, FiQA, SciDocs, TREC-COVID |
| mean | mixed | mixed | 0.302231 | 0.613316 | +0.000766 | +0.002579 | Arguana, FiQA, SciDocs, TREC-COVID |

M218B confirms the macro win is seed-stable under this exploratory criterion.
Every seed has positive macro MAP and Recall deltas vs PPLX dense. The effect is
still small, but it is no longer a single-seed accident.

The row anatomy remains essentially unchanged. The persistent unsafe rows are
Arguana, FiQA, SciDocs, and TREC-COVID. FiQA is the largest unresolved failure:
the best seeds remain around `-0.024` to `-0.033` dMAP and around `-0.034` to
`-0.036` dRecall vs dense.

## M218B Decision

Continue to M219A. The next question is whether the student is under-transferring
the usable part of the RankT5-safe teacher. Keep the teacher family fixed to
`blend_d0p5`, reduce the M210 distillation constraint, and make the pairwise
target more boundary-oriented:

- Config A: `distill_weight=0.05`, `positive_k=16`,
  `negative_start_rank=32`, `negative_k=96`
- Config B: `distill_weight=0.10`, `positive_k=16`,
  `negative_start_rank=32`, `negative_k=96`

If M219A improves the macro win beyond M218B, run seed stability for the best
transfer config. If it does not improve, the current RankT5-safe weak-supervision
route is close to its practical limit and needs a stronger weak-supervision
source or row-aware pseudo-label objective.
