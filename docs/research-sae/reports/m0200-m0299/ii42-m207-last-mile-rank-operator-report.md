# II-42 M207 Last-Mile Rank Operator Report

Date: 2026-09-23

## Question

M203 proved that exact dense scores inside the BM25+SSR top256 candidate window
can beat PPLX dense on macro MAP and Recall. M204-M206 then showed that global
projection heads trained to mimic dense scores, top-rank pairs, or blended final
scores do not transfer that advantage into a compact mini-vector.

M207 treats the mini-vector as a last-mile sorting signal rather than a dense
score surrogate. Before training another head, it tests whether rank-based or
gated local operators have a better exact ceiling and a cleaner deployment
contract.

## Protocol

Run:
`m207a-full7-rank-rrf-gated-exact-top256-v1`

- Host: `lambda2`
- Tmux session: `ii42_m207a_rank_operator_scout`
- Candidate source: frozen M190/SSR latent postings plus BM25
- Eval rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID,
  Webis-Touche2020
- Qrels: final metrics only
- No training, no locked-test access

Operator families:

- `score_minmax`: M203-style score blend reference
- `rank_linear`: baseline rank plus dense rank over the local window
- `rrf`: reciprocal-rank fusion over baseline rank and dense rank
- `score_margin_gate`: apply score blend only when dense top-vs-rank16 margin
  clears a threshold

Sweeps:

- Protected heads: `0, 5`
- Correction windows: `100, 256`
- Score weights: `0.75, 1.0, 1.25, 1.5, 2.0`
- Rank weights: `0.25, 0.5, 0.75, 1.0, 1.5, 2.0`
- RRF weights: `0.5, 1.0, 1.5, 2.0`
- RRF `k`: `10, 30, 60`
- Dense-margin gates: `0.05, 0.1, 0.15, 0.2, 0.3`

## Decision Rule

If a rank/gated exact operator beats PPLX dense with a larger or safer margin
than M203, use it as the M208 mini-vector training target. If none beat M203,
keep the M203 product operator as the ceiling but shift training toward
query/row-adaptive gating rather than another global score-mimic objective.

## M207A Results

M207A completed in 1205 seconds and evaluated 192 exact local operators.

Top exact variants:

| Rank | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Unsafe dense rows |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | score minmax, head0, window256, w2 | 0.304746 | 0.613773 | +0.003281 | +0.003036 | FiQA, SciDocs |
| 2 | score minmax + margin gate 0.05 | 0.304746 | 0.613773 | +0.003281 | +0.003036 | FiQA, SciDocs |
| 3 | score minmax + margin gate 0.1 | 0.304616 | 0.613592 | +0.003151 | +0.002855 | FiQA, SciDocs, TREC-COVID |
| 4 | RRF, head0, window256, w2, k10 | 0.304472 | 0.615018 | +0.003007 | +0.004282 | FiQA, SciDocs, TREC-COVID |
| 5 | RRF, head0, window256, w2, k30 | 0.304094 | 0.615227 | +0.002629 | +0.004490 | FiQA, SciDocs, TREC-COVID |
| 12 | M203 reference score minmax, head0, window256, w1.5 | 0.303368 | 0.614099 | +0.001903 | +0.003362 | Arguana, FiQA, SciDocs, TREC-COVID |

Best variant by family:

| Family | Operator | dMAP vs dense | dRecall vs dense | Beats dense | Unsafe dense rows |
| --- | --- | ---: | ---: | --- | --- |
| score minmax | head0, window256, w2 | +0.003281 | +0.003036 | yes | FiQA, SciDocs |
| score margin gate | head0, window256, w2, gate0.05 | +0.003281 | +0.003036 | yes | FiQA, SciDocs |
| RRF | head0, window256, w2, k10 | +0.003007 | +0.004282 | yes | FiQA, SciDocs, TREC-COVID |
| rank linear | head0, window256, w2 | +0.000633 | +0.001427 | yes | Arguana, FiQA, SciDocs |

Selected row anatomy:

| Dataset | score w2 dMAP | score w2 dRecall | RRF w2 k10 dMAP | RRF w2 k10 dRecall |
| --- | ---: | ---: | ---: | ---: |
| Arguana | -0.001656 | -0.000714 | -0.001765 | -0.000714 |
| FiQA | -0.019031 | -0.025672 | -0.007279 | -0.023383 |
| NFCorpus | +0.005188 | +0.001147 | +0.004977 | +0.001519 |
| SciDocs | -0.003464 | -0.007417 | -0.001177 | -0.008017 |
| SciFact | +0.022670 | +0.020000 | +0.010374 | +0.016667 |
| TREC-COVID | -0.001765 | -0.001013 | -0.002525 | -0.002128 |
| Webis-Touche2020 | +0.021027 | +0.034923 | +0.018445 | +0.046026 |

## Interpretation

M207 changes the next step. The M203 operator was not the true exact limit for
top256; increasing score-blend pressure from `1.5` to `2.0` improves macro MAP
from `+0.001903` to `+0.003281` versus PPLX dense and reduces unsafe rows from
four to two. The simple dense-margin gate does not add value at low threshold:
`gate=0.05` fires on every query and is identical to ungated score blend.

RRF is the more interesting training target even though it is second on MAP.
It beats PPLX dense on MAP, Recall, NDCG, and MRR, and its contract is rank
based rather than score calibrated. That should be easier for a compact
mini-vector to learn: the head only needs to order the top256 candidates well
enough for reciprocal-rank fusion, not reproduce dense score magnitudes.

## Decision

Proceed to M208 with a rank-distillation objective and evaluate the deployable
`32d INT4` head under the M207-selected RRF operator:
`protected_head=0`, `correction_window=256`, `weight=2.0`, `rrf_k=10`.
Keep `64d` and `float32` only as diagnostics. Use the exact `score_minmax w2`
result as the ceiling/control, not as the first learned objective.
