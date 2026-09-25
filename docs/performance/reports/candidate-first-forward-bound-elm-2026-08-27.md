# Candidate-first and forward-bound closure

Date: 2026-08-27

## Decision

Keep the current float32 forward-bound format and policy 7. Use the existing
automatic route split: direct forward rows for extremely small allowed sets,
the exact transposed stream for medium and broad filters, and float32 bounds
only when the physical-cost model selects the bounded proof route.

Do not promote candidate-prefix completion, exact global prefix, BMP filtered
scoring, adaptive set-bit iteration, uint8 bounds, or bfloat16 bounds. Every
rejected route either failed exactness or increased measured product latency.

No rejected format remains in the product implementation. Elm was restored to
policy 7 after the experiment.

## Qualification surface

- Host: Elm
- PostgreSQL: 18.6
- Runtime: immutable ONNX Runtime 1.29 package
- Fixture: `evoke_cf_eval.pubmed_250k`
- Documents: 250,000 logged rows
- Index: `evoke_cf_eval.pubmed_250k_evoke_idx`
- Queries: three semantic queries
- Exact ladder: seven filters spanning 178 to 249,854 allowed rows
- Product comparison: warm medians after one warm-up execution

The fixture is durable across restart. Accelerator-only publication reused the
existing root and semantic postings; it did not re-encode documents or rebuild
the main index.

## Exactness and route elimination

Candidate-prefix completion was not exact: 18 of 21 ladder rows matched, while
three returned 49 of the required 50 documents. Increasing the prefix from 128
to 256 and 512 did not repair those rows. The accelerator prefix is therefore a
candidate seed, not an exact ordered iterator.

An exact global prefix was also rejected. At a 10% filter it took about 728 ms,
versus roughly 100 ms for the current exact filtered route. BMP was slower than
the accelerator transpose at 10%, 50%, and near-full selectivity. Adaptive
set-bit iteration did not improve the quiet same-root ladder. Coarser block
hierarchies reduced metadata but increased row work; block size 8 remained the
best total-cost point.

These results close the candidate-first line unless a future representation
provides a genuinely exact, resumable global score iterator. Threshold changes
cannot make the current prefix exact.

## Conservative-bound compression

Both compressed formats were encoded as conservative upper bounds and passed
the 18-case oracle exactness matrix.

| Format | Bound metadata reduction | 2.5% scattered total proxy reduction | Competitive row inflation |
| --- | ---: | ---: | ---: |
| Per-term uint8 scale | 59.9% | 46.4% | 1.0% |
| Upward bfloat16 | 40.0% | 30.9% | 1.0% |

The oracle result did not translate into product latency.

| Product format | Exact rows | Median | Mean | p95 | Mean versus float |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float32 policy 7 | 21/21 | 259.08 ms | 205.74 ms | 298.05 ms | baseline |
| Uint8 policy 8 experiment | 21/21 | 273.88 ms | 215.38 ms | 315.60 ms | 1.042x |
| Bfloat16 policy 9 experiment | 21/21 | 261.71 ms | 221.68 ms | 336.23 ms | 1.053x |

Uint8 paid a per-entry scale multiplication and widened conservative bounds.
Bfloat16 removed that multiplication but remained slower because the hot path
is CPU/decode dominated rather than metadata-I/O dominated. Reducing serialized
bytes alone is not a valid promotion criterion.

## Route crossover

Forced-route measurements used identical result sets for direct, transpose,
and bounded execution.

| Allowed rows | Direct float | Transpose float | Bounded float | Best route |
| ---: | ---: | ---: | ---: | --- |
| 178 | 67.13 ms | 68.19 ms | 75.33 ms | direct |
| 25,078 | 188.73 ms | 100.20 ms | 118.07 ms | transpose |
| 125,021 | 684.84 ms | 251.57 ms | 270.84 ms | transpose |
| 249,854 | 1,064.60 ms | 298.83 ms | 316.58 ms | transpose |

This is the useful combined path: document-major direct scoring only for tiny
subsets, then term-major transpose as selectivity broadens. The current cost
model already selects this crossover closely enough that a new selectivity
threshold would be micro-tuning rather than a structural improvement.

## Stop condition

Stop metadata quantization and candidate-prefix tuning. A new experiment is
justified only if it changes one of the underlying physical capabilities:

1. an exact resumable score-ordered iterator;
2. block metadata that avoids decoding competitive postings rather than merely
   shrinking metadata bytes; or
3. a same-root layout whose hot-loop CPU cost is demonstrably lower in an
   interleaved same-binary A/B.

All future promotions must preserve exact subset top-k, use the same durable
root and binary for comparison, report decoded postings/blocks/bytes as well as
latency, and pass the broad ladder rather than a selective-only smoke.

## Evidence

Raw JSON is stored under
`docs/performance/data/raw/planner-native-candidate-first-elm-2026-08-27/`.
The directory contains the float, uint8, and bfloat16 product ladders, both
compression oracles, and the forced-route matrices.
