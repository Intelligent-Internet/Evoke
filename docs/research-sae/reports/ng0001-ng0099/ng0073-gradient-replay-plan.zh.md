# NG-0073: Actual TRAIN Gradient Direction and Coverage Replay

## Status and Question

Pre-execution protocol, 2026-09-13 UTC. NG71 D improved the three-domain exposed-DEV macro but failed the NQ quality floor; NG72 found TRAIN sentinel front-rank losses dominated by common matching weights, not only support disappearance. Neither observation establishes which optimizer mechanism caused a lost ordering.

NG73 asks whether the actual frozen D objective directly requested smaller margins on pairs whose correct initial ordering was subsequently lost. Compare all four outcome groups, not only failures. This is a diagnosis, not a new training round or a quality claim.

## Frozen Inputs and Surface

- Replay all 192 D updates, 768 example exposures, 384 unique TRAIN pilot queries, using each example's actual stored scores and the correct A0/A96 reference witness pool.
- Exclude the disjoint TRAIN sentinel from gradient examples. No DEV or LOCKED_TEST inference or scores are read for this analysis.
- Join NG72 pilot positive/rival rows, including original and added positives, all known gold exclusions, initial/final stable-ID order, and baseline top-10/top-100 membership.
- Preserve pool absence and teacher-ineligible pairs. They are not hard negatives and do not receive invented direct gradients.
- Verify the NG71 input anchor, relevant source/dependency hashes, predecessor phase receipts, NG72 terminal anchor, and every consumed input before and after execution. Freeze this source, tests and protocol into a fresh external NG73 unit.

## Computation and Interpretation

For pair margin `z = (s_positive - s_rival) / T_student`, coefficient `c` and soft target `t`, the own-pair derivative is:

```text
dL / d(s_positive - s_rival) = c * (sigmoid(z) - t) / T_student
```

Positive values request shrinking that margin under independent score-space descent; negative values request expansion. Use an explicit `1e-12` numerical stationary interval. Replay NumPy and Torch derivatives and compare the recorded four-example-normalized score gradients with the existing NG71 tolerance (`rtol=2e-5, atol=2e-7`) and loss tolerance (`rtol=2e-6, atol=2e-7`).

Also record `g_positive - g_rival` after summing all objective pairs. This describes aggregate score-space pressure, not the actual parameter-induced score change. Query/document encoders share parameters, and AdamW/VJP/cross-query interference are not reconstructed here. No causal attribution of final rank flips follows from this replay alone.

Output `lost`, `retained`, `gained`, `stayed_behind` and all-pair controls. Record raw pair-exposure counts, direct derivative mass, pool absence/ineligibility, head membership and aggregate pressure. For balanced contributions, divide by each positive's full fixed rival universe, then average positives/exposures within query and queries within domain. Do not conditionally renormalize only the lost subset. Original/added-positive analyses are separately identified. Every actual supervised pair must appear in the NG72 rival union, otherwise fail closed.

## Resource and Decision Gates

CPU only, four threads, at most 900 seconds and 4 GiB process-tree RSS; retain at least 8 GiB host available RAM and 10 GiB free disk. ClearML offline actual-start/close receipts, process exit/group closure, SHA manifest and explicit no-inference flags are required. Do not rerun failed attempts in place.

If direct shrink is meaningfully present, a one-sided teacher-margin floor becomes a candidate, not an accepted fix: stop penalizing a student merely for exceeding an already-correct soft teacher margin. If it is absent or weak in the relevant evidence, do not force that mechanism; investigate indirect shared-parameter drift or supervision coverage instead. A subsequent uniform-versus-metric and retention-off-versus-on controlled pilot must retain the mature base, candidate pools, data, steps and quality floors. No automatic scale-up or production promotion.

The metric weighting is motivated by [LambdaLoss](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/), not a guarantee for this custom balanced soft-target variant. [SPLADE-v3](https://arxiv.org/html/2403.06789v1) provides empirical motivation for treating teacher calibration, ranking loss and mature initialization as separate controlled factors; its reported weights and gains are not transplanted into this protocol.
