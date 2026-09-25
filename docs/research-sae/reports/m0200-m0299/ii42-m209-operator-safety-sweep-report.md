# II-42 M209 Learned Mini-Vector Operator Safety Sweep Report

Date: 2026-09-23

## Question

M207 proved the top256 last-mile operator still has exact headroom, while M208
showed that aggressive learned RRF rank distillation can make the mini-vector
worse than earlier M204/M205 results. M209 tests whether the degradation is
mainly caused by an unsafe operator rather than absence of learned signal.

The experiment trains a stable M204-style shared score head, then evaluates the
same learned mini-vector under safer deployable operators: score blend,
mini-vector margin gate, baseline ambiguity gate, and promotion-only.

## Protocol

Run:
`m209a-stable-score-head-operator-safety-full7-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Epochs: `30`
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Objective: M204-style dense score-shape distillation
- Operators: score minmax, correction-margin gate, baseline-ambiguity gate,
  promotion-only
- Qrels: final metrics only

## Decision Rule

If a `32d INT4` safer operator beats PPLX dense, move to seed stability. If a
safer operator recovers M204/M205 quality but still misses dense, use that
operator as the base for M209B action distillation. If none recover quality,
treat the global linear mini-vector head as the bottleneck and move to a
non-linear or mixture head.

## M209A Results

M209A completed in 4718 seconds and evaluated 240 operators over `32d/64d`
and `float32/INT4` variants. No `32d INT4` operator beat PPLX dense.

Best `32d INT4` variants:

| Rank | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | score minmax, head0, window256, w1.25 | 0.297693 | 0.609346 | -0.003772 | -0.001391 | +0.008534 | +0.008107 |
| 2 | correction margin gate, head0, window256, w1.25, gate0.1 | 0.297330 | 0.609361 | -0.004135 | -0.001376 | +0.008170 | +0.008123 |
| 3 | score minmax, head0, window256, w1.0 | 0.297039 | 0.609107 | -0.004426 | -0.001630 | +0.007880 | +0.007869 |
| 4 | score minmax, head0, window256, w1.5 | 0.296830 | 0.609537 | -0.004635 | -0.001200 | +0.007671 | +0.008299 |

Best diagnostic variants by dimension/mode:

| Variant | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | score minmax, head0, window256, w1.25 | 0.297693 | 0.609346 | -0.003772 | -0.001391 | +0.008534 | +0.008107 |
| 32d float32 | score minmax, head0, window256, w1.0 | 0.298289 | 0.610131 | -0.003176 | -0.000606 | +0.009130 | +0.008893 |
| 64d INT4 | score minmax, head0, window256, w1.5 | 0.299439 | 0.612094 | -0.002026 | +0.001357 | +0.010279 | +0.010856 |
| 64d float32 | score minmax, head0, window256, w1.5 | 0.300520 | 0.612939 | -0.000945 | +0.002202 | +0.011361 | +0.011701 |

Per-row anatomy for the best `32d INT4` operator:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.273221 | 0.999286 | -0.031399 | -0.000714 | -0.009676 | +0.004996 |
| FiQA | 0.416765 | 0.788488 | -0.040884 | -0.040380 | +0.039098 | +0.026699 |
| NFCorpus | 0.167760 | 0.322631 | -0.000252 | -0.004333 | +0.010419 | +0.007568 |
| SciDocs | 0.155130 | 0.480133 | -0.007724 | -0.015700 | +0.011610 | +0.022350 |
| SciFact | 0.743666 | 0.983333 | +0.030043 | +0.020000 | -0.003198 | +0.003333 |
| TREC-COVID | 0.134764 | 0.164869 | -0.004020 | -0.002485 | +0.014727 | +0.013199 |
| Webis-Touche2020 | 0.192543 | 0.526678 | +0.027831 | +0.033873 | -0.003245 | -0.021393 |

## Interpretation

M209A confirms the safer-operator hypothesis partially. The best safer
operator recovers from M208's collapse and lands near the earlier M204/M205
quality band, but still misses PPLX dense. The selected deployable point is
`32d INT4`, `score_minmax`, `head=0`, `window=256`, `weight=1.25`.

The sweep also shows that simple gates are not enough. Correction-margin gates
mostly fire on nearly all queries and do not materially improve the frontier.
Baseline-ambiguity and promotion-only operators did not enter the top
`32d INT4` group. The useful learned signal is present, but it is not being
trained around the right action boundary.

The diagnostic `64d float32` result is close to dense MAP and beats dense
Recall, which means the route is not dead. The remaining problem is how to
train a deployable `32d INT4` head to make the exact operator's promotion and
demotion decisions rather than mimic dense scores globally.

## Decision

Proceed to M209B action distillation using the M209A selected operator as the
deployment target: `score_minmax`, `head=0`, `window=256`, `weight=1.25`.
Train against exact teacher promotion/demotion actions around the top100
boundary instead of whole-window dense score shape. Keep `32d INT4` as the
primary target and `64d/float32` as diagnostics.

## M209B Plan

M209B follows the M209A branch where safer operators recovered the M204/M205
quality band but did not beat dense. The new target is action distillation: train
on exact teacher promotion/demotion decisions around the top100 boundary under
`score_minmax`, `head=0`, `window=256`, `teacher_weight=1.25`.

Run:
`m209b-action-distill-full7-32-64d-float32-int4-gpu0-v1`

Decision rule:

- if `32d INT4` beats PPLX dense on macro MAP and Recall, move to seed
  stability;
- if action distillation beats M209A but still misses dense, inspect row-level
  failures before deciding between a gate or non-linear head;
- if it does not beat M209A, stop linear-head experiments and move to a
  non-linear or mixture head.

## M209B Results

M209B completed in 2538 seconds. It did not beat M209A. The best `32d INT4`
action-distilled point was `weight=1.0`, with MAP 0.287471 and Recall 0.606115.
That is worse than M209A's selected `32d INT4` operator by MAP -0.010222 and
Recall -0.003231, and remains below PPLX dense by MAP -0.013993 and Recall
-0.004622.

`32d INT4` variants:

| Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 0.287471 | 0.606115 | -0.013993 | -0.004622 | -0.001688 | +0.004877 |
| 1.25 | 0.282422 | 0.603972 | -0.019043 | -0.006765 | -0.006737 | +0.002734 |
| 1.5 | 0.278059 | 0.602271 | -0.023406 | -0.008466 | -0.011100 | +0.001032 |

Best diagnostic variants:

| Variant | Weight | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | 1.0 | 0.287471 | 0.606115 | -0.013993 | -0.004622 | -0.001688 | +0.004877 |
| 32d float32 | 1.0 | 0.286553 | 0.605173 | -0.014912 | -0.005564 | -0.002606 | +0.003934 |
| 64d INT4 | 1.0 | 0.288745 | 0.608765 | -0.012720 | -0.001972 | -0.000414 | +0.007526 |
| 64d float32 | 1.0 | 0.290256 | 0.608879 | -0.011209 | -0.001858 | +0.001097 | +0.007640 |

## M209B Interpretation

Action distillation did not solve the last-mile mini-vector bottleneck. It made
the `32d INT4` head substantially worse than the M209A score-shape head and did
not even preserve the M204/M205 quality band. Increasing the deployment weight
amplified the error, which matches the earlier diagnosis that noisy learned
signals become harmful when used as an unconditional top256 reranker.

The diagnostic `64d float32` result is also below M209A's `32d INT4`, so this is
not primarily an INT4 quantization loss. The failure is in the linear action
head/objective combination: it does not learn a stable local promotion/demotion
surface from the current features.

## M209B Decision

Stop this linear-head action-distillation branch. The next useful path should
change the model class rather than keep sweeping weights or losses on the same
global linear mini-vector head. The most defensible follow-up is a small
non-linear or mixture head trained for the same deployment contract, with
explicit safeguards that only apply the mini-vector correction where confidence
is high.
