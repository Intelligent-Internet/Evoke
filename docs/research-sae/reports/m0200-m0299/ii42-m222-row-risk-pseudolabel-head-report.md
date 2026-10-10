# II-42 M222 Row-Risk Pseudo-Label Head Report

Date: 2026-09-28

## Purpose

M222 moves past scalar gate tuning. M221 retained a stronger qrels-free runtime
route, but the same dense-unsafe rows persisted: Arguana, FiQA, SciDocs, and
TREC-COVID. M222 tests whether that failure pattern can be converted into a
weak row-family prior during pseudo-label construction, while keeping the same
`32d INT4` product contract and the M218 training budget.

This is a weak-supervision diagnostic, not a pure qrels-free endpoint: the
risk-family list comes from prior non-locked diagnostic results. The training
still uses cached RankT5, dense, and local baseline scores as pseudo teachers;
qrels are otherwise used only for final metrics.

Current retained baseline:

| Route | MAP | Recall | dMAP vs PPLX dense | dRecall vs PPLX dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 `family_dense_half` | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |

## M222A Plan

Train two narrow row-risk pseudo-label policies, with no changes to optimizer,
epoch budget, model size, quantization, or candidate protocol:

- `row_prior_dense_anchor`: on risk families, pull pseudo labels toward dense
  and local baseline while still retaining a weak RankT5 term
- `row_prior_consensus_anchor`: on risk families, require stronger agreement
  with dense or local baseline before a document can become a positive

Shared settings:

- train/eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- risk-family prior: Arguana, FiQA, SciDocs, TREC-COVID
- product contract: `32d INT4`
- epochs: `16`
- distill weight: `0.2`
- pair sampling: `positive_k=8`, `negative_start_rank=16`, `negative_k=64`
- runtime evaluation: `fixed`, `family_dense_half`, and `m220c_joint`
- fixed weights: `1.25,1.375,1.5`
- overlap-k: `24`

Decision rule:

- If M222A improves macro MAP/Recall over M221 while preserving or reducing the
  unsafe-row set, continue with seed stability.
- If M222A reduces unsafe rows while keeping positive macro deltas, continue
  the row-safety branch even if MAP is slightly below M221.
- If it neither improves macro nor reduces unsafe rows, stop this weak row-prior
  branch. The next route should use a stronger external supervision source or a
  different row-aware objective, not more RankT5-cache tuning.

## M222A Result

M222A completed both weak row-risk pseudo-label policies under the unchanged
M218 budget and `32d INT4` product contract.

Best deployable variant:

| Policy | Router | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `row_prior_consensus_anchor` | `family_dense_half` | 1.25 | 0.301160 | 0.613896 | -0.000304 | +0.003159 | Arguana, FiQA, SciDocs, TREC-COVID |

Best non-deployable teacher ceiling:

| Policy | Weight | MAP | Recall | dMAP vs dense | dRecall vs dense | Unsafe rows |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `row_prior_dense_anchor` | 1.50 | 0.314408 | 0.617156 | +0.012943 | +0.006419 | Arguana, FiQA, SciDocs, TREC-COVID |

Policy diagnostics:

| Policy | Mean active fraction | Arguana | FiQA | SciDocs | TREC-COVID |
| --- | ---: | ---: | ---: | ---: | ---: |
| `row_prior_dense_anchor` | 0.839983 | 0.704812 | 0.735098 | 0.719891 | 0.720078 |
| `row_prior_consensus_anchor` | 0.764766 | 0.593474 | 0.581260 | 0.577297 | 0.601328 |

Best deployable row deltas versus PPLX dense:

| Dataset | dMAP | dRecall | dMAP vs fixed hybrid | Safe vs dense |
| --- | ---: | ---: | ---: | --- |
| Arguana | -0.032789 | -0.002141 | -0.011066 | no |
| FiQA | -0.039089 | -0.039325 | +0.040893 | no |
| NFCorpus | +0.007569 | +0.008395 | +0.018240 | yes |
| SciDocs | -0.008478 | -0.012717 | +0.010856 | no |
| SciFact | +0.045349 | +0.016667 | +0.012109 | yes |
| TREC-COVID | -0.005963 | -0.005557 | +0.012784 | no |
| Webis | +0.031271 | +0.056791 | +0.000195 | yes |

## M222 Decision

M222A does not meet either continuation condition:

- It does not improve macro MAP/Recall over the retained M221 endpoint.
  M221 is MAP 0.305818 / Recall 0.615731, while M222A best deployable is
  MAP 0.301160 / Recall 0.613896.
- It does not reduce the unsafe-row set. Arguana, FiQA, SciDocs, and
  TREC-COVID remain unsafe.

The teacher ceiling still has macro headroom, but it also remains row-unsafe.
That means simple row-family dense anchoring weakens the student without solving
the actual row-safety problem. The failure is especially clear on Arguana and
FiQA: the deployable student loses more MAP than M221 while preserving the same
unsafe rows.

Conclusion: stop this weak row-prior pseudo-label branch. The retained
deployable endpoint remains M221:

- M218B seed 313 `blend_d0p5`
- `family_dense_half`
- overlap-k `24`
- weight `1.375`
- MAP 0.305818 / Recall 0.615731
- deltas vs PPLX dense: +0.004354 / +0.004994

The next justified breakthrough would need a stronger supervision source or a
different objective that directly targets row failure modes. More anchoring
against the current RankT5/dense/baseline cache is not justified by M222A.
