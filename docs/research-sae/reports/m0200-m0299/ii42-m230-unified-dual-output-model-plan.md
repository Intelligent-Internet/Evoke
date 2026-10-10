# II-42 M230 Unified Dual-Output Model Plan

## Status

M230 is the next architectural step after the M229 query-rich
score-geometry milestone. The goal is to move from a mini-dense sidecar
correction path to one unified text encoder that emits both:

- SAE posting output for the unified sparse index.
- 32d INT4 mini-dense output for the rerank/correction signal.

M229 remains the saved query-performance reference while M230 tests whether
that capability can be compressed into the same model that produces SAE
postings.

## Current Evidence

M229B is the current best query-performance endpoint before query-cost control:

| Endpoint | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 qrels-free runtime route | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M226B supervised score geometry mean | 0.360175 | 0.639834 | +0.058710 | +0.029097 | none |
| M229B query-rich mean | 0.366415 | 0.639995 | +0.064950 | +0.029258 | none |
| M229B best seed301 | 0.367166 | 0.640347 | +0.065701 | +0.029611 | none |

M229 is not yet the final unified model. It trains `doc_model` and
`query_model` PyTorch heads and evaluates them through compact 32d INT4
correction rows plus the M220/M221 runtime router. It is therefore still a
research sidecar path, even though the document-side mini-vector storage
contract is deployable.

The true unified product target must instead be one model/runtime artifact
whose document forward pass produces a single payload record containing sparse
postings and the compact mini-dense vector.

## Product Contract Target

The unified encoder should expose a document path like:

```text
document text
  -> shared encoder trunk
  -> SAE posting head
       int32 atom_ids[]
       float32 atom_weights[]
       uint16 active_count
       float32 norm
  -> mini-dense head
       int4[32] vector payload
       scale/zero metadata needed for exact product scoring
```

The query path should mirror the same trunk with query-specific heads:

```text
query text
  -> shared encoder trunk
  -> query SAE posting head
  -> query mini-dense vector
```

This should compile into the existing unified posting runtime family rather
than creating a new sidecar table. The mini-dense payload can be attached to
the same document record used by `unified_posting`.

## Main Hypothesis

The M229 gain comes from continuous supervised score geometry and moderate
query-side capacity. The open question is whether that geometry is available
from the SAE encoder's hidden state.

If the SAE trunk already contains enough signal, a mini-dense head can be
attached with little or no posting regression. If not, a shared-trunk multitask
run is needed, and the key risk is destroying SAE posting selectivity.

## M230A: Frozen SAE Trunk Readout Probe

Purpose:

Test whether the existing SAE encoder can emit a useful 32d mini-dense vector
without changing the sparse posting behavior.

Design:

- Start from the current mature NG80 sparse encoder checkpoint, preferably the
  stronger sparse endpoint used by the unified posting path.
- Freeze the transformer trunk and SAE posting head.
- Add a small mini-dense projection head from hidden pooled states.
- Train only the mini-dense head against the M226/M229 mean-z supervised
  score-geometry target.
- Keep the document-side mini-vector at 32d INT4.
- Evaluate with the same M220/M221 router family used by M229.

Success criteria:

- No SAE posting output drift, because trunk and posting head are frozen.
- Dense-safe mini-dense result beats PPLX dense on macro MAP and Recall.
- Preferably recovers at least half of the M229B MAP gain over PPLX dense.

Decision:

- If M230A is close to M229B, promote frozen-trunk dual-output as the safest
  unification route.
- If M230A is positive but materially behind M229B, continue to M230B
  multitask fine-tuning.
- If M230A is weak, do not spend time on query-cost control yet; the model
  needs shared representation training.

## M230B: Shared-Trunk Multitask Compression

Purpose:

Train one encoder to preserve SAE postings while learning the mini-dense
score-geometry behavior.

Losses:

- Sparse posting preservation loss against the existing NG80/S-P sparse
  endpoint outputs.
- Mini-dense pair loss on supervised candidate windows.
- Mini-dense score-geometry loss against the M214B seed301/313/337 mean-z
  teacher ensemble.
- Drift guard on posting active-count, support overlap, and posting score
  quality.

Training shape:

- Initialize from NG80 sparse encoder.
- Add 32d document and query mini-dense heads.
- Keep the first run conservative: low trunk learning rate, stronger head
  learning rate, and periodic sparse replay checks.
- Use the same train/eval split rules as the M226/M229 bridge. Qrels are used
  only for final metrics and teacher diagnostics, not for post-hoc tuning.

Success criteria:

- Mini-dense quality is in the M226/M229 band and dense-safe.
- SAE posting metrics regress only within a predeclared tolerance.
- No row becomes dense-unsafe under the -0.002 row floor.

## M230C: Unified Runtime Packing

Purpose:

Turn the best M230A/B model into a product-like artifact.

Deliverable:

- One checkpoint.
- One tokenizer/config.
- One forward API returning sparse atom ids/weights and 32d mini-dense vector.
- A compiled manifest compatible with the unified posting lifecycle, extended
  with mini-dense metadata.

Validation:

- Byte/payload determinism.
- Document update/reindex lifecycle smoke test.
- Query parity between offline PyTorch scoring and product runtime scoring.
- Posting-only, mini-dense-only, and fused retrieval metrics.

## Why This Is The Right Next Move

M227/M228 showed that qrels-free RankT5/dense/baseline geometry still has
macro headroom but remains row-unsafe. M226/M229 showed that the product
mini-vector head can carry much stronger dense-safe behavior when the teacher
geometry is aligned. Therefore the next structural question is no longer
whether 32d INT4 has enough capacity; it does. The question is whether that
capacity can live inside the same encoder that emits SAE postings.

The first M230 experiment should answer this directly before any more
query-cost work:

```text
Can the SAE encoder hidden state read out the M229/M226 mini-dense geometry?
```

If yes, the project has a clean path to a unified dual-output model. If no, the
next breakthrough needs shared-trunk multitask training, not more sidecar
rerank tuning.

## Immediate Next Step

Build `M230A` as a frozen-trunk readout probe on lambda2 GPU0:

- Inputs: NG80 sparse checkpoint and M226/M229 score-geometry supervision.
- Output: one frozen SAE encoder plus trained 32d mini-dense readout head.
- No posting mutation in M230A.
- Report both mini-dense quality and exact posting invariance.

Only after M230A establishes the readout feasibility should M230B unfreeze any
part of the trunk.

## Active M230A Run

M230A has been implemented and launched as the first fusion-route probe.

- tmux session: `ii42_m230a_unified_dual_output_gpu0`
- Output directory:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/runs/m230a-ng80-s-frozen-dual-output-32d-int4-gpu0-v1`
- Log:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/logs/m230a-unified-dual-output-encoder-gpu0-v1.log`
- Script:
  `/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts/research_sae_m230_unified_dual_output_encoder.py`
- Local source:
  `docs/research-sae/source/m0200-m0299/research_sae_m230_unified_dual_output_encoder.py`
- Launcher:
  `docs/research-sae/source/m0200-m0299/launchers/launch_m230a_unified_dual_output_encoder.sh`

This run freezes the NG80 `sparse-S-v1/train-3072` SAE encoder and trains only
the 32d mini-dense readout heads. The saved checkpoint records the SAE encoder
identity and verifies that the SAE parameter hash is unchanged after feature
encoding and mini-dense training.

## M230A Result

M230A completed on lambda2 GPU0 at 2026-10-01 19:55 America/Toronto.

Best deployable variant:

- Router: `family_dense_half`
- Fixed weight: `2.5`
- MAP: 0.300202
- Recall: 0.617215
- Delta vs PPLX dense: MAP -0.001263, Recall +0.006479
- Dense-unsafe rows: Arguana, FiQA, TREC-COVID
- SAE posting invariance: verified by unchanged SAE parameter hash

The frozen SAE trunk can recover a useful recall signal, but it does not carry
the M226/M229 score geometry strongly enough to preserve MAP. It also remains
row-unsafe. This fails the M230A continuation criterion for direct frozen-trunk
unification.

Comparison against retained endpoints:

| Endpoint | MAP | Recall | dMAP vs dense | dRecall vs dense | Dense-unsafe rows |
| --- | ---: | ---: | ---: | ---: | --- |
| M221 qrels-free runtime route | 0.305818 | 0.615731 | +0.004354 | +0.004994 | Arguana, FiQA, SciDocs, TREC-COVID |
| M226B supervised score geometry mean | 0.360175 | 0.639834 | +0.058710 | +0.029097 | none |
| M229B query-rich mean | 0.366415 | 0.639995 | +0.064950 | +0.029258 | none |
| M230A frozen SAE readout best | 0.300202 | 0.617215 | -0.001263 | +0.006479 | Arguana, FiQA, TREC-COVID |

Interpretation:

The result is a useful negative structural probe. The current NG80 sparse SAE
hidden state is not enough as a frozen representation for the M229 mini-dense
head. A simple dual-head attachment would sacrifice the sidecar quality. The
next possible route is M230B shared-trunk multitask training with explicit
posting drift guards, not frozen readout or more scalar router tuning.

## M230A Lessons

What M230A kept:

- It implemented the first real dual-output prototype shape: one SAE encoder
  identity plus doc/query mini-dense readout heads.
- It verified exact SAE posting invariance by keeping the NG80 sparse encoder
  frozen and checking the parameter hash before and after feature extraction
  and training.
- It recovered some useful recall signal: best dRecall vs PPLX dense was
  +0.006479.
- It produced strong local gains on some rows, especially Scifact
  (dMAP +0.034263 / dRecall +0.020000), Webis-Touche2020
  (dMAP +0.031076 / dRecall +0.055266), and NFCorpus
  (dMAP +0.005075 / dRecall +0.020446).

What M230A lost:

- It failed macro MAP: best dMAP was -0.001263, far below M221 and far below
  the M229B sidecar milestone.
- FiQA collapsed hardest: dMAP -0.068666 and dRecall -0.047268.
- TREC-COVID stayed unsafe: dMAP -0.005389 and dRecall -0.006398.
- Arguana was also unsafe on MAP: dMAP -0.004085.
- The M220/M221 router family did not rescue the frozen readout; higher weights
  monotonically hurt MAP after the best `family_dense_half,w=2.5` point.

Likely cause:

The frozen NG80 SAE encoder was trained to emit sparse posting fields. Its
mean-pooled hidden state is not aligned to the continuous M226/M229 score
geometry. M229's gain comes from a dense-feature mini-vector path plus a
moderate query-rich adapter, not from a generic low-dimensional projection of
the SAE hidden mean. Therefore the direct frozen readout is too weak to guard
the sidecar result.

## Revised Path Toward The Goal

The target remains one unified model, but the next step should preserve the
M229 sidecar behavior explicitly. M230A trained against the upstream M214
mean-z teacher geometry. The next run should distill the actual M229B
deployable behavior, because that is the sidecar result we need to keep.

### M230B: Frozen Trunk, Richer Mini-Dense Branch

Purpose:

Test whether the SAE trunk contains enough information if the mini-dense branch
uses richer frozen features than plain hidden mean pooling.

Changes from M230A:

- Keep the SAE encoder and posting head frozen, so posting invariance remains
  exact.
- Use richer readout features: mean pool, max pool, CLS/first-token pool, and
  selected sparse-head summary statistics rather than only mean-pooled hidden.
- Train a `query_rich_512x2`-style mini-dense branch from those features.
- Primary target: M229B seed301/313/337 deployable correction rows or score
  geometry.
- Secondary target: M214B mean-z teacher geometry, used as a regularizer.

Promotion gate:

- Posting invariance remains exact.
- Dense-safe macro result beats M221.
- Preferably recovers at least half of the M229B dMAP gain; otherwise treat it
  as partial evidence only.

Completed run:

- tmux session: `ii42_m230b_richer_m229_distill_gpu0`
- Output directory:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/runs/m230b-ng80-s-richer-m229-distill-32d-int4-gpu0-v1`
- Log:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/logs/m230b-richer-m229-distill-gpu0-v1.log`
- Feature cache:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/feature-cache/m230b-ng80-s-richer-hidden-v1`
- Teacher source: M229B seed301/313/337 `query_rich_512x2` checkpoints.
- Feature taps: `mean,max,first,sparse_summary`.
- Mini-dense branch: 32d INT4 readout with hidden dim 1024 and depth 3.

Result:

- Posting invariance remained exact: SAE parameter hash was unchanged before
  feature extraction, after feature extraction, and after readout training.
- Best deployable variant was `family_dense_half,w=1.5`.
- MAP 0.290998 / Recall 0.602481.
- Deltas vs PPLX dense: dMAP -0.010466 / dRecall -0.008256.
- Dense-unsafe rows: Arguana, FiQA, NFCorpus, SciDocs, and TREC-COVID.
- It is weaker than M230A, the retained qrels-free M221 endpoint, and the
  M229B sidecar milestone.

Interpretation:

M230B is a clean negative. Richer frozen SAE features and direct M229B behavior
distillation did not recover the mini-dense sidecar. The evidence now points
away from simple frozen-trunk readout as the compression path. The NG80 SAE
hidden space can support byte-stable postings, but its frozen representation is
not aligned enough to host the M229/M226 score geometry through a shallow
mini-dense branch.

This run was still a frozen-posting-path probe. It did not mutate the NG80 SAE
encoder or posting head.

### M230C: Branch-Local Adapter, Posting Path Frozen

Purpose:

Because M230B was too weak, the next plausible route is branch-local adapter
capacity for the
mini-dense head without changing the posting path.

Design:

- The posting head continues to read the original SAE hidden states.
- The mini-dense head gets a small adapter over the same token hidden states.
- The adapter is active only for mini-dense output, not for posting output.
- This is still a unified model/runtime artifact because one text forward
  emits both outputs, but posting behavior remains protected.

Promotion gate:

- SAE posting output is byte-identical or score-identical to the frozen
  baseline.
- Mini-dense result is dense-safe and approaches the M229B band.
- M230C should not be launched as another readout sweep. It should be a
  structural test: the posting path must remain frozen/guarded, while the
  mini-dense path receives extra branch-local adaptation.

Completed run:

- tmux session: `ii42_m230c_token_adapter_gpu0`
- Output directory:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/runs/m230c-ng80-s-token-summary-adapter-32d-int4-gpu0-v1`
- Log:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/logs/m230c-token-summary-adapter-gpu0-v1.log`
- Feature cache:
  `/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1/feature-cache/m230c-ng80-s-token-summary-v1`
- Teacher source: M229B seed301/313/337 `query_rich_512x2` checkpoints.
- Feature taps:
  `mean,max,first,token_topk_hidden,sparse_summary`.
- Token summary: top 8 non-padding token hidden vectors by hidden norm.
- Mini-dense branch: 32d INT4 readout with hidden dim 2048 and depth 3.
- Fixed weights: `0.5,1.0,1.5,2.0,2.5`.
- Routers: `fixed,family_dense_half,m220c_joint`.

Result:

- Posting invariance remained exact.
- Best deployable variant was `fixed,w=0.5`.
- MAP 0.297280 / Recall 0.609905.
- Deltas vs PPLX dense: dMAP -0.004185 / dRecall -0.000832.
- Dense-unsafe rows: Arguana, FiQA, NFCorpus, SciDocs, and TREC-COVID.
- Row deltas at the best point:
  - Arguana dMAP -0.005581 / dRecall -0.001428.
  - FiQA dMAP -0.067385 / dRecall -0.055952.
  - NFCorpus dMAP -0.006664 / dRecall +0.000199.
  - SciDocs dMAP -0.011961 / dRecall -0.010167.
  - SciFact dMAP +0.041537 / dRecall +0.020000.
  - TREC-COVID dMAP -0.010975 / dRecall -0.009989.
  - Webis-Touche2020 dMAP +0.031735 / dRecall +0.051510.

Interpretation gate:

- If M230C beats M221 while preserving no worse unsafe rows, continue to a
  narrower branch-adapter refinement.
- If M230C remains below M221 or repeats M230B's expanded unsafe-row pattern,
  frozen SAE representation plus branch-local feature adaptation should be
  treated as exhausted. The next credible direction would be guarded top-block
  adaptation with explicit posting preservation loss, not more readout features.

Decision:

M230C does not beat M221 and repeats the expanded unsafe-row pattern from
M230B. It is better than M230B on macro MAP/Recall, but it is still worse than
M230A and far below the sidecar M229B endpoint. This closes the frozen-SAE
readout/feature-adapter path. Further progress toward a unified dual-output
model requires controlled trunk adaptation with explicit sparse-posting
preservation, or a different unified pretraining objective. More frozen feature
taps should not be run.

### M230D: Shared-Trunk Multitask Only If Needed

Purpose:

Unfreeze trunk parameters only if frozen richer readouts and branch-local
adapters cannot recover the sidecar result.

Guardrails:

- Low learning rate on only the top transformer block(s).
- Strong sparse posting preservation loss against NG80/S-P outputs.
- Active-count, support-overlap, score, and row-safety drift checks.
- No promotion unless posting regression stays within a predeclared tolerance.

This is the riskiest route and should not be the next immediate step.
