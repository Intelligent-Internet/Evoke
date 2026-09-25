# Maintenance Policy Repeatability

Date: 2026-03-24

## Scope

This report checks how stable the policy-matrix winners are across
multiple seeds.

Raw data is stored in the corresponding diagnostics JSON.

Run shape:

- the same five policy presets as the earlier matrix
- the same five churn scenarios
- `repeat_count = 3`
- seed offsets applied across repeats

## Main Finding

The single-run matrix was directionally useful, but not stable enough to
justify a high-confidence one-size-fits-all deferred recommendation.

Repeatability read:

- `query_first`
  - still remains the clearest guidance
  - eager wins most often overall
- `small_mixed_churn`
  - `tuned_current` remains the strongest deferred preset
- `heavy_mixed_churn`
  - no single deferred policy wins consistently across repeats
- `heavy_insert_skew`
  - `bytes_40000` won more often in repeats than the earlier
    single-run `tuned_relaxed` result

## Practical Impact

The recommendation helper should stay conservative.

- `query_first`
  - high confidence
- `small_mixed_churn`
  - medium confidence
- `balanced`
  - medium confidence
- `heavy_mixed_churn`
  - low confidence
- `write_tolerant_query_first`
  - medium confidence

That confidence guidance is now reflected in
`evoke_index_policy_recommend(...)`.

## Targeted Follow-Up

Two heavier 5-repeat follow-ups were run sequentially to decide whether
more skew-specific recommendation profiles were justified.

Raw data:

- heavy-update repeat diagnostics JSON
- heavy-delete repeat diagnostics JSON

Results:

- `heavy_update_skew`
  - overall winner: `tuned_current` in `3/5` repeats
  - best deferred winner: `tuned_current` in `4/5` repeats
  - outcome: no new profile was added
- `heavy_delete_skew`
  - overall winner: `eager` in `4/5` repeats
  - best deferred winner: `tuned_current` in `5/5` repeats
  - outcome: no new profile was added because existing
    `query_first`/`balanced` guidance already covers this split

That leaves `heavy_insert_skew` as the only skewed workload with enough
repeatable evidence to justify a dedicated recommendation profile.

## Stability Note

While collecting the earlier heavier repeats, concurrent heavy
benchmark runs intermittently hit `invalid evoke index metapage`.

Follow-up hardening on the branch serializes maintenance writers and
rebuilds with a maintenance-only relation lock. A dedicated smoke script
exists in the main repository so the same style of concurrent pressure
can be re-run on demand.

After that hardening, broader concurrent stress runs passed at:

- `rounds=2, repeat_count=2`
- `rounds=3, repeat_count=3`
- `rounds=5, repeat_count=3`
- `rounds=6, repeat_count=4`

Additional cross-skew concurrent pressure also passed with:

- `--pair heavy_delete_skew,heavy_insert_skew`
- `--pair heavy_update_skew,heavy_mixed`
- `rounds=3, repeat_count=3`

One later failure during simultaneous heavy stress runs was traced to
benchmark-harness database-name collisions rather than a new engine-side
metapage corruption. The stress harness now injects a unique `run_id`
per matrix process so concurrent runs no longer share benchmark DB
names.

A broader family runner also exists in the main repository.
The current branch passed its full default family set:

- `matrix_default_soak`
- `matrix_cross_skew`
- `longrun_pair`
