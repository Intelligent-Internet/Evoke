# II-42 M213 Qrels-Supervised Safe Mode Selector Report

Date: 2026-09-24

## Question

M212A showed that dense-teacher agreement is not enough. Even the fixed
mini-vector mode oracle missed PPLX dense MAP for deployable `32d INT4`.

M213A changes the supervision target before spending compute on a true
multi-expert head. It keeps the M210A mini-vector fixed and trains a query-level
mode selector using train qrels to choose the correction strength that maximizes
per-query AP@100, with a small Recall tie-breaker. This is a supervised
diagnostic, not a qrels-free benchmark.

## Protocol

Run:
`m213a-qrels-safe-mode-selector-m210a-full7-32-64d-int4-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base mini-vector: M210A residual MLP checkpoints
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Mode weights: `0.0, 0.5, 0.75, 1.0, 1.25, 1.5`
- Selector: query-level MLP over qrels-free baseline/correction features
- Selector labels: train-qrels AP@100 plus `0.05 * Recall@100`
- Diagnostics: per-query qrels oracle selector is non-deployable
- Locked-test access: none

## Decision Rule

If a deployable `32d INT4` qrels-supervised selector beats PPLX dense on macro
MAP and Recall, prepare seed stability and consider moving supervision into a
true multi-expert head. If only the qrels oracle beats dense, the bottleneck is
selector learnability/features. If even the qrels oracle fails to beat dense,
stop the fixed M210A mini-vector path and only continue with a genuinely new
multi-expert representation if its diagnostic objective is different.

## M213A Results

M213A completed in 1030 seconds. The best deployable `32d INT4` selector was
`mode_soft`, with MAP 0.299537 and Recall 0.613452. This is the best deployable
`32d INT4` result in the M210-M213 sequence so far, but it still misses PPLX
dense MAP by -0.001927 while beating dense Recall by +0.002715.

Best deployable `32d INT4` variants:

| Rank | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mode soft | 0.299537 | 0.613452 | -0.001927 | +0.002715 | +0.010378 | +0.012213 |
| 2 | mode hard | 0.299138 | 0.612077 | -0.002327 | +0.001340 | +0.009979 | +0.010839 |
| 3 | mode confident 0.4 | 0.296503 | 0.609441 | -0.004962 | -0.001296 | +0.007344 | +0.008203 |
| 4 | mode confident 0.5 | 0.295643 | 0.608716 | -0.005822 | -0.002021 | +0.006484 | +0.007478 |
| 5 | mode confident 0.6 | 0.293886 | 0.606287 | -0.007579 | -0.004450 | +0.004727 | +0.005048 |

Best diagnostic variants:

| Variant | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | mode soft | 0.299537 | 0.613452 | -0.001927 | +0.002715 | +0.010378 | +0.012213 |
| 32d float32 | mode hard | 0.299079 | 0.612640 | -0.002385 | +0.001903 | +0.009920 | +0.011402 |
| 64d INT4 | mode soft | 0.300296 | 0.612993 | -0.001169 | +0.002256 | +0.011137 | +0.011755 |
| 64d float32 | mode hard | 0.301228 | 0.612050 | -0.000237 | +0.001314 | +0.012069 | +0.010812 |

Qrels-oracle diagnostics:

| Variant | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 oracle | mode oracle | 0.316319 | 0.621989 | +0.014854 | +0.011253 | +0.027160 | +0.020751 |

Per-row anatomy for the `32d INT4` qrels oracle:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Baseline rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.305416 | 0.999286 | +0.000796 | -0.000714 | +0.022519 | +0.004996 | 0.000000 |
| FiQA | 0.438058 | 0.801213 | -0.019591 | -0.027655 | +0.060391 | +0.039424 | 0.424383 |
| NFCorpus | 0.177573 | 0.341380 | +0.009561 | +0.014416 | +0.020232 | +0.026316 | 0.334365 |
| SciDocs | 0.165899 | 0.493000 | +0.003045 | -0.002833 | +0.022379 | +0.035217 | 0.346000 |
| SciFact | 0.779723 | 0.986667 | +0.066100 | +0.023333 | +0.032860 | +0.006667 | 0.806667 |
| TREC-COVID | 0.137229 | 0.166874 | -0.001554 | -0.000480 | +0.017193 | +0.015204 | 0.060000 |
| Webis-Touche2020 | 0.207045 | 0.567565 | +0.042333 | +0.074760 | +0.011257 | +0.019494 | 0.428571 |

## M213A Interpretation

This is the first result that clearly says the fixed M210A mini-vector path is
not dead. Under qrels-safe oracle selection, even deployable `32d INT4`
mini-vector scores can beat PPLX dense by a large macro margin. The remaining
gap is selector learnability: the deployable selector moves in the right
direction but recovers only a small fraction of oracle headroom.

The label distribution explains the gap. For `32d INT4`, qrels-safe labels pick
baseline-only about 53% of the time and strong correction `w1.5` about 18% of
the time. The previous dense-teacher selector almost never chose baseline. So
the real target is not dense mimicry; it is a safety policy that often abstains
and only applies correction on rows where it improves AP.

Selector accuracy is still low, about 42% for `32d INT4`. That makes a narrow
follow-up justified: test whether row-family conditioning makes the selector
learnable before moving to an expensive true multi-expert representation.

## M213A Decision

Continue with a narrow M213B selector-learnability diagnostic. Keep the M210A
mini-vector fixed, keep the same supervised diagnostic framing, and add explicit
train-row conditioning to the selector features. If M213B still fails to recover
substantial oracle headroom, the next step should be a true multi-expert head
rather than more selector variants over the same score.

## M213B Results

Run:
`m213b-qrels-safe-mode-selector-onehot-m210a-full7-32-64d-int4-gpu0-v1`

M213B completed in 1021 seconds. It keeps the M210A mini-vector fixed and adds
`train_onehot` row-family conditioning to the selector features. This is still a
supervised diagnostic, because selector labels come from train-qrels AP@100 with
the Recall tie-breaker.

This is the first fixed-M210 `32d INT4` mini-vector selector result that beats
PPLX dense on both macro MAP and macro Recall. The best deployable diagnostic
variant is `mode_hard`, with MAP 0.302958 and Recall 0.613410. That is
+0.001494 MAP and +0.002673 Recall vs PPLX dense, and +0.013799 MAP and
+0.012172 Recall vs fixed BM25+SSR.

Best deployable diagnostic `32d INT4` variants:

| Rank | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mode hard | 0.302958 | 0.613410 | +0.001494 | +0.002673 | +0.013799 | +0.012172 |
| 2 | mode confident 0.4 | 0.301695 | 0.612098 | +0.000230 | +0.001361 | +0.012535 | +0.010860 |
| 3 | mode soft | 0.301067 | 0.613123 | -0.000397 | +0.002386 | +0.011908 | +0.011885 |
| 4 | mode confident 0.5 | 0.300765 | 0.608069 | -0.000700 | -0.002668 | +0.011606 | +0.006831 |
| 5 | mode confident 0.6 | 0.298916 | 0.605985 | -0.002549 | -0.004751 | +0.009757 | +0.004747 |

Best diagnostic variants across dimensions and quantization:

| Variant | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | Payload |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | mode hard | 0.302958 | 0.613410 | +0.001494 | +0.002673 | 20 B/doc |
| 32d float32 | mode hard | 0.302096 | 0.613865 | +0.000631 | +0.003128 | 128 B/doc |
| 64d INT4 | mode soft | 0.301502 | 0.612186 | +0.000037 | +0.001449 | 36 B/doc |
| 64d float32 | mode hard | 0.303873 | 0.614005 | +0.002408 | +0.003268 | 256 B/doc |

Per-row anatomy for the best `32d INT4` one-hot selector:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Mean alpha | Baseline rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.282393 | 0.996431 | -0.022228 | -0.003569 | -0.000505 | +0.002141 | 0.322091 | 0.543897 |
| FiQA | 0.427564 | 0.794921 | -0.030085 | -0.033947 | +0.049896 | +0.033131 | 0.971065 | 0.123457 |
| NFCorpus | 0.171195 | 0.333179 | +0.003183 | +0.006216 | +0.013854 | +0.018116 | 1.105263 | 0.024768 |
| SciDocs | 0.160172 | 0.487050 | -0.002682 | -0.008783 | +0.016652 | +0.029267 | 1.113750 | 0.017000 |
| SciFact | 0.760374 | 0.980000 | +0.046751 | +0.016667 | +0.013510 | +0.000000 | 0.350000 | 0.613333 |
| TREC-COVID | 0.134716 | 0.165171 | -0.004068 | -0.002183 | +0.014679 | +0.013501 | 1.180000 | 0.000000 |
| Webis-Touche2020 | 0.184297 | 0.537120 | +0.019585 | +0.044315 | -0.011491 | -0.010951 | 1.158163 | 0.000000 |

Selector training improved materially over M213A. For `32d INT4`, final train
accuracy rose from about 42% to 50.9%, while the qrels-safe label distribution
remained similar: baseline-only about 53%, and strong correction `w1.5` about
18%. The one-hot feature did not create new oracle headroom; it made a larger
fraction of existing headroom learnable.

The non-deployable `32d INT4` qrels oracle remains the endpoint reference:
MAP 0.316319, Recall 0.621989, dMAP +0.014854 and dRecall +0.011253 vs PPLX
dense.

## M213B Interpretation

M213B changes the conclusion from "fixed M210 mini-vector may have headroom" to
"fixed M210 mini-vector can beat dense under a supervised safety selector." The
payload remains in the intended product band at `32d INT4` and 20 bytes/doc.

The result is not yet a product claim. It is a supervised diagnostic with
train-row one-hot conditioning and qrels-derived selector labels. The macro win
also hides row-level safety issues: Arguana, FiQA, SciDocs, and TREC-COVID are
still unsafe vs dense under the -0.002 row floor, while NFCorpus, SciFact, and
Webis carry the macro gain.

## M213B Decision

Continue with seed stability before changing the representation. The next run
should keep the exact M213B protocol but restrict to deployable `32d INT4` and
repeat selector seeds. If the macro dense win is stable, the next structural
step is to move from a row-conditioned selector over one shared mini-vector into
a true multi-expert or row-family-conditioned head. If it is not stable, the
selector is still the bottleneck.

## M213C Results

Run family:
`m213c-seed-<seed>-qrels-safe-onehot-32d-int4-gpu0-v1`

M213C repeats the exact M213B supervised diagnostic protocol, restricted to the
deployable `32d INT4` output contract, across selector seeds `301`, `313`, and
`337`.

The result is stable. All three seeds beat PPLX dense on both macro MAP and
macro Recall with the same winning selector family, `mode_hard`.

| Seed | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Unsafe dense rows |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 301 | mode hard | 0.302939 | 0.613215 | +0.001474 | +0.002478 | +0.013780 | +0.011977 | Arguana, FiQA, SciDocs, TREC-COVID |
| 313 | mode hard | 0.302272 | 0.612997 | +0.000807 | +0.002260 | +0.013113 | +0.011758 | Arguana, FiQA, SciDocs, TREC-COVID |
| 337 | mode hard | 0.302514 | 0.613690 | +0.001049 | +0.002953 | +0.013355 | +0.012452 | Arguana, FiQA, SciDocs, TREC-COVID |
| mean | mode hard | 0.302575 | 0.613300 | +0.001110 | +0.002564 | +0.013416 | +0.012062 | Arguana, FiQA, SciDocs, TREC-COVID |

Seed-level selector training accuracy is also stable but not high: 47.4%,
48.7%, and 46.9%. That is enough to recover a repeatable macro win, but still
far from the qrels-oracle endpoint.

## M213C Interpretation

M213C confirms the M213B finding is not a lucky selector seed. Under supervised
safe-MAP labels plus train-row one-hot conditioning, the fixed M210A `32d INT4`
mini-vector consistently beats PPLX dense macro MAP and Recall.

The remaining weakness is row safety and learnable capacity, not the 20-byte
payload itself. The same four rows remain unsafe vs dense under the -0.002 row
floor across all seeds. Meanwhile, NFCorpus, SciFact, and Webis consistently
carry the macro gain. This means the next useful experiment should not be
another selector variant over the same shared score; it should make the
mini-vector representation itself row-family aware, or split it into a true
multi-expert head, while keeping the deployed contract at `32d INT4`.

## M213C Decision

Continue to a structural M214 experiment. The goal is to test whether moving the
row-family information from the selector layer into the mini-vector head can
reduce unsafe dense rows while keeping the M213C macro win. The product-facing
contract remains unchanged: score only the top256 candidates with a `32d INT4`
payload.
