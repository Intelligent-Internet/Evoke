# II-42 M212 Query Mode Selector Report

Date: 2026-09-23

## Question

M211A showed that a binary learned soft gate improves the deployable `32d INT4`
mini-vector path but still misses PPLX dense MAP. The teacher-agreement oracle
gate also missed dense MAP, suggesting that on/off gating is too coarse.

M212A tests a more expressive query-level controller: choose one correction
mode per query from a small set of strengths, including baseline-only. The
mini-vector is fixed from M210A; only the query mode selector is trained.

## Protocol

Run:
`m212a-query-mode-selector-m210a-full7-32-64d-int4-gpu0-v1`

- Host: `lambda2`
- GPU visibility: `CUDA_VISIBLE_DEVICES=0`
- Base mini-vector: M210A residual MLP checkpoints
- Train rows: Arguana, FiQA, NFCorpus, SciDocs, SciFact, TREC-COVID
- Eval rows: train rows plus Webis-Touche2020
- Dimensions: `32, 64`
- Quantization/eval modes: `float32`, `INT4`
- Mode weights: `0.0, 0.5, 0.75, 1.0, 1.25, 1.5`
- Teacher: exact dense top256 operator with `teacher_weight=2.0`
- Selector: small query-level MLP over qrels-free baseline/correction features
- Diagnostics: per-query teacher-agreement oracle selector is non-deployable
- Qrels: final metrics only

## Decision Rule

If a deployable `32d INT4` selector beats PPLX dense on macro MAP and Recall,
prepare seed stability. If the oracle selector beats dense but the learned
selector does not, improve selector features/training. If the oracle selector
still misses dense, stop this fixed-mini-vector selector family and move to a
different supervision target or a true multi-expert head.

## M212A Results

M212A completed in 1070 seconds. The best deployable `32d INT4` selector was
`mode_soft`, with MAP 0.297693 and Recall 0.612779. It improves Recall over
M211A but loses MAP relative to M211A, and it still misses PPLX dense MAP by
-0.003771.

Best deployable `32d INT4` variants:

| Rank | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | mode soft | 0.297693 | 0.612779 | -0.003771 | +0.002043 | +0.008534 | +0.011541 |
| 2 | mode hard | 0.297685 | 0.613467 | -0.003779 | +0.002730 | +0.008526 | +0.012228 |
| 3 | mode confident 0.4 | 0.293547 | 0.608910 | -0.007918 | -0.001827 | +0.004388 | +0.007672 |
| 4 | mode confident 0.5 | 0.291218 | 0.606221 | -0.010247 | -0.004516 | +0.002059 | +0.004983 |
| 5 | mode confident 0.6 | 0.289595 | 0.603602 | -0.011870 | -0.007135 | +0.000436 | +0.002364 |

Best diagnostic variants:

| Variant | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 | mode soft | 0.297693 | 0.612779 | -0.003771 | +0.002043 | +0.008534 | +0.011541 |
| 32d float32 | mode hard | 0.297557 | 0.611997 | -0.003907 | +0.001260 | +0.008398 | +0.010758 |
| 64d INT4 | mode hard | 0.299287 | 0.612264 | -0.002178 | +0.001528 | +0.010128 | +0.011026 |
| 64d float32 | mode soft | 0.300729 | 0.612579 | -0.000735 | +0.001842 | +0.011570 | +0.011341 |

Mode-oracle diagnostics:

| Variant | Selector | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32d INT4 oracle | mode oracle | 0.299152 | 0.613910 | -0.002312 | +0.003173 | +0.009993 | +0.012672 |
| 32d float32 oracle | mode oracle | 0.299133 | 0.614608 | -0.002331 | +0.003872 | +0.009974 | +0.013370 |
| 64d INT4 oracle | mode oracle | 0.299820 | 0.613962 | -0.001645 | +0.003225 | +0.010661 | +0.012724 |
| 64d float32 oracle | mode oracle | 0.301673 | 0.614154 | +0.000208 | +0.003418 | +0.012513 | +0.012916 |

Per-row anatomy for the best deployable `32d INT4` selector:

| Dataset | MAP@100 | Recall@100 | dMAP vs dense | dRecall vs dense | dMAP vs fixed | dRecall vs fixed | Mean alpha |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Arguana | 0.275730 | 0.998572 | -0.028890 | -0.001428 | -0.007168 | +0.004283 | 0.779843 |
| FiQA | 0.421832 | 0.793542 | -0.035817 | -0.035326 | +0.044165 | +0.031753 | 1.106742 |
| NFCorpus | 0.172008 | 0.330461 | +0.003997 | +0.003497 | +0.014668 | +0.015397 | 1.064996 |
| SciDocs | 0.156878 | 0.486000 | -0.005975 | -0.009833 | +0.013359 | +0.028217 | 1.036702 |
| SciFact | 0.741199 | 0.980000 | +0.027576 | +0.016667 | -0.005664 | +0.000000 | 1.029351 |
| TREC-COVID | 0.133191 | 0.163364 | -0.005592 | -0.003990 | +0.013155 | +0.011694 | 1.079284 |
| Webis-Touche2020 | 0.183014 | 0.537516 | +0.018302 | +0.044712 | -0.012775 | -0.010555 | 1.193297 |

## M212A Interpretation

The multi-mode selector did not improve the deployable frontier. It shifted
toward stronger correction and increased Recall, but MAP fell below M211A's
learned-soft gate. Confidence fallback variants were worse, which means the
selector confidence is not a useful abstention signal in this feature space.

The oracle result is the key diagnostic. The `32d INT4` mode oracle still misses
dense MAP by -0.002312, so the problem is not only that the learned selector is
weak. Even the qrels-free exact-teacher mode choice is not enough to make the
fixed M210 mini-vector path beat dense. The only variant that beats dense on
both MAP and Recall is `64d float32` oracle, which is diagnostic and outside the
deployable payload contract.

The selector labels also explain the behavior: baseline-only is almost never
chosen for `32d INT4` training, while high correction weights dominate. This
teacher-agreement target prefers dense-like movement even when that movement
does not preserve MAP against final qrels.

## M212A Decision

Stop this fixed-mini-vector selector family. The next useful route needs a
different supervision target or a true multi-expert head, not another selector
over the same M210A mini-vector scores. In particular, the target should encode
safe MAP retention or row/query mode specialization rather than dense-teacher
agreement alone.
