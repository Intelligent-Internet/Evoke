# II-42 M211 Query-Gated Mini-Vector Report

Date: 2026-09-23

## Question

M210A showed that residual MLP capacity changes the shape of the result but
does not solve the core problem. `32d INT4` improves Recall over PPLX dense but
still misses dense MAP, while `64d float32` is close to dense MAP. Row anatomy
shows the same mini-vector correction helps some datasets and hurts others.

M211A tests the next structural hypothesis: the deployable path needs a
query-level gate or mixture controller that decides when the mini-vector
correction should be applied. The mini-vector itself is fixed from M210A. The
gate is trained from dense-teacher agreement features, not qrels.

## Protocol

Run:
`m211a-query-gated-m210a-full7-32-64d-int4-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base mini-vector: M210A residual MLP checkpoints
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Score weights: `0.5, 0.75, 1.0, 1.25`
- Gate: small query-level MLP over qrels-free baseline/correction features
- Gate label: learned correction is closer than baseline to the exact dense
  top256 teacher by an agreement margin
- Diagnostics: teacher-agreement oracle gate is reported as non-deployable
- Qrels: final metrics only

## Decision Rule

If a deployable `32d INT4` gated variant beats PPLX dense on macro MAP and
Recall, prepare seed stability. If the oracle gate beats dense but the learned
gate does not, improve gate features/training. If neither beats M210A, stop this
teacher-agreement gate route and move to a trained mixture expert or a different
supervision target.

## M211A Results

M211A completed in 1197 seconds. The best deployable `32d INT4` result was a
learned soft gate over `score_minmax, head=0, window=256, w=1.25`, with MAP
0.298132 and Recall 0.612283. This improves over M210A's best `32d INT4`
unconditional point, but it still misses PPLX dense MAP.

Best deployable `32d INT4` variants:

| Rank | Gate | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | learned soft | score minmax, head0, window256, w1.25 | 0.298132 | 0.612283 | -0.003333 | +0.001546 | +0.008973 | +0.011045 |
| 2 | learned soft | score minmax, head0, window256, w1.0 | 0.298089 | 0.612077 | -0.003376 | +0.001340 | +0.008930 | +0.010839 |
| 3 | ungated | score minmax, head0, window256, w0.75 | 0.297471 | 0.611828 | -0.003993 | +0.001091 | +0.008312 | +0.010590 |
| 4 | learned hard 0.4 | score minmax, head0, window256, w1.0 | 0.297453 | 0.611616 | -0.004012 | +0.000880 | +0.008294 | +0.010378 |
| 5 | ungated | score minmax, head0, window256, w1.0 | 0.297203 | 0.612312 | -0.004261 | +0.001575 | +0.008044 | +0.011074 |

Best diagnostic variants:

| Variant | Gate | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | learned soft | score minmax, head0, window256, w1.25 | 0.298132 | 0.612283 | -0.003333 | +0.001546 | +0.008973 | +0.011045 |
| 32d float32 | learned hard 0.4 | score minmax, head0, window256, w1.0 | 0.298444 | 0.611585 | -0.003021 | +0.000849 | +0.009285 | +0.010347 |
| 64d INT4 | ungated | score minmax, head0, window256, w1.0 | 0.298972 | 0.612274 | -0.002493 | +0.001537 | +0.009813 | +0.011035 |
| 64d float32 | ungated | score minmax, head0, window256, w1.0 | 0.300663 | 0.612764 | -0.000801 | +0.002027 | +0.011504 | +0.011526 |

Teacher-agreement oracle gate diagnostics:

| Variant | Operator | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 oracle | score minmax, head0, window256, w1.25 | 0.298987 | 0.614276 | -0.002478 | +0.003539 | +0.009828 | +0.013038 |
| 32d float32 oracle | score minmax, head0, window256, w1.0 | 0.299191 | 0.613882 | -0.002274 | +0.003145 | +0.010032 | +0.012644 |
| 64d INT4 oracle | score minmax, head0, window256, w1.0 | 0.300017 | 0.614605 | -0.001448 | +0.003868 | +0.010857 | +0.013367 |
| 64d float32 oracle | score minmax, head0, window256, w1.0 | 0.301331 | 0.615118 | -0.000134 | +0.004381 | +0.012171 | +0.013879 |

Per-row anatomy for the best deployable `32d INT4` variant:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Mean alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.277960 | 0.997859 | -0.026661 | -0.002141 | -0.004938 | +0.003569 | 0.433638 |
| FiQA | 0.415005 | 0.791806 | -0.042644 | -0.037062 | +0.037337 | +0.030017 | 0.831153 |
| NFCorpus | 0.171320 | 0.332099 | +0.003309 | +0.005135 | +0.013979 | +0.017035 | 0.800819 |
| SciDocs | 0.155569 | 0.485517 | -0.007285 | -0.010317 | +0.012049 | +0.027733 | 0.686163 |
| SciFact | 0.748789 | 0.980000 | +0.035165 | +0.016667 | +0.001925 | +0.000000 | 0.715842 |
| TREC-COVID | 0.132668 | 0.163453 | -0.006116 | -0.003901 | +0.012632 | +0.011783 | 0.780569 |
| Webis-Touche2020 | 0.185615 | 0.535249 | +0.020903 | +0.042444 | -0.010173 | -0.012822 | 0.864241 |

## M211A Interpretation

Query gating helps, but not enough. The best learned soft gate improves the
deployable `32d INT4` MAP over M210A by about +0.000661 and over M209A by about
+0.000439, while keeping Recall above dense. However, dense MAP is still ahead
by -0.003333.

The oracle gate is the important diagnostic. Even with non-deployable access to
dense-teacher agreement labels, `32d INT4` remains below dense MAP by -0.002478.
This means the failure is not just a weak learned gate. The gate has some
headroom, but the teacher-agreement gating target is not sufficient to close the
dense MAP gap under the current single-correction mini-vector path.

The per-row pattern remains the same: the route helps NFCorpus, SciFact, and
Webis, but still loses too much on Arguana, FiQA, SciDocs, and TREC-COVID. The
gate reduces some damage, but does not learn a decisive abstention policy for
the harmful rows.

## M211A Decision

Do not launch another run on this exact teacher-agreement query gate. The next
route should change supervision or structure: either a trained mixture expert
with explicit row/query mode selection, or a different target that optimizes
safe MAP retention rather than dense-teacher agreement alone.
