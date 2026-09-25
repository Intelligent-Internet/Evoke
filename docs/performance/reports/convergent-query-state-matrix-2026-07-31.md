# Convergent Query-State Matrix

Date: 2026-07-31

## Scope

This focused local diagnostic measures the same public
`evoke_query_ids(...)` top-100 query in five exact representations:

1. a one-segment static rebuild;
2. the same 40,000-document corpus represented by one base plus two sealed
   immutable segments;
3. the fragmented index after a query-heat-admitted, term-local COW workload
   neutral fold;
4. the convergent index after an epoch-bound impact specialization;
5. the legacy contiguous materialized-impact generation used as the frozen
   peak-performance control.

It tests the product contract that committed updates may temporarily add
bounded query work, while an unchanged hot working set converges back to the
static peak without a whole-corpus fold. It is not the qualification-host
release matrix and does not replace the official BEIR benchmark.

## Reproduction

The checked-in harness creates an isolated PostgreSQL 18 cluster, prevents the
background worker from racing the measured index, and alternates the control
and treatment order inside five trials:

```bash
python3 scripts/benchmark_convergent_query_states.py \
  --extension-libdir <staged-pg-libdir> \
  --extension-control-dir <staged-pg-sharedir> \
  --base-docs 30000 \
  --delta-docs 5000 \
  --warmup 16 \
  --samples 96 \
  --trials 5 \
  --output <result.json>
```

The command was run in three independent clusters. Every run used:

- PostgreSQL `18.4`;
- an arm64 macOS host with 24 logical CPUs;
- commit `eef1bdcc841ce856d09bf0ef31b00400714912ec`;
- a clean product source tree;
- staged library SHA-256
  `3771fbe3b284499d91fde3a822201e6a60d5306c023f0fac481b4b5d1263127d`;
- 40,000 synthetic four-token documents;
- one hot term present in all documents;
- one native scorer and one native top-k, with no result-list fusion.

Raw results:

- [run 1](../data/diagnostics/convergent-query-states-2026-07-31/run-1.json)
- [run 2](../data/diagnostics/convergent-query-states-2026-07-31/run-2.json)
- [run 3](../data/diagnostics/convergent-query-states-2026-07-31/run-3.json)
- [impact closure, 40k](../data/diagnostics/convergent-query-states-2026-07-31/run-v3-impact-40k.json)
- [impact closure, 250k](../data/diagnostics/convergent-query-states-2026-07-31/run-v3-impact-250k.json)

## Exactness And Maintenance

All three runs passed every gate:

- static, fragmented, and folded result ids were identical;
- maximum absolute score difference was `0.0`;
- the fragmented state contained exactly three visible immutable segments;
- query telemetry observed all three hot-term extents;
- workload maintenance consumed three extents and reduced the query surface
  by two;
- the fold remained under the same per-index maintenance guard;
- no whole-segment compaction or corpus rebuild occurred.

Each fold compiled 40,000 postings with `coverage_sequence=40000`. The
reported `input_bytes=10440728` belongs to the selected hot posting key, not
to the complete corpus. Each publication reused 49 currently unreachable
blocks. The relation still retained three immutable segments: the fold
changed only that term's authoritative read shape.

## Latency Matrix

Ratios below are paired against the static control measured in the same
phase. The table reports the three-run median of each per-run ratio.

| State | Mean | p50 | p95 | p99 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three extents / static | `1.029x` | `1.034x` | `1.042x` | `1.032x` | `0.972x` |
| Folded / concurrent static | `1.000x` | `1.003x` | `1.004x` | `1.013x` | `1.000x` |
| Folded / fragmented, normalized to static | `0.972x` | `0.980x` | `0.985x` | `0.971x` | `1.029x` |

Per-run p50 and p95 ratios:

| Run | Fragmented p50 | Fragmented p95 | Folded p50 | Folded p95 |
| --- | ---: | ---: | ---: | ---: |
| 1 | `1.034x` | `1.042x` | `1.003x` | `1.004x` |
| 2 | `1.022x` | `1.018x` | `1.002x` | `1.003x` |
| 3 | `1.055x` | `1.058x` | `1.034x` | `1.097x` |

The three-extent p95 remains inside the design's four-extent `1.10x`
temporary-degradation limit in every run. The folded p50 and QPS return to
the concurrent static repeatability interval. Tail percentiles remain noisy
on this shared local host, but the static-normalized fold improved p50 in
all three runs and recovered a median `2.9%` QPS over the fragmented state.

## Materialized-Impact Peak Closure

The final gate compares the selected impact-specialized COW surface directly
with a legacy materialized-impact index built over the same heap. Each trial
interleaves control and treatment queries. The gate requires exact ids and
scores, mean/p50/p95 no worse than `1.05x`, p99 no worse than `1.10x`, and QPS
at least `0.95x`.

| Corpus | Visible segments before fold | Mean | p50 | p95 | p99 | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40k | 3 | `0.976x` | `0.978x` | `0.973x` | `0.980x` | `1.024x` |
| 250k | 5 | `0.977x` | `0.968x` | `1.007x` | `0.991x` | `1.023x` |

Every ratio is impact-specialized divided by materialized control, except QPS
where values above one are faster. Both runs returned the same 100 ids with
maximum absolute score difference `0.0`. At 250k, median-trial mean latency
was `0.682 ms` for the impact-specialized read image versus `0.698 ms` for the
legacy materialized generation.

The five observed segments in the 250k run are valid. The two logical delta
batches crossed the bounded byte-pressure frontier and were split into four
immutable segments. The benchmark therefore validates that the observed
physical fanout is at least the logical minimum and requires workload
telemetry to cover every actual segment; it does not assume one segment per
batch.

The strict fast path is deliberately narrow. It is selected only when every
queried in-range term resolves to lexical-impact extents under the exact
positive-IDF contract, both linked-L0 frontiers are empty, and the impact
statistics epoch is current. Neutral, semantic, mixed, weighted, stale, or
unsupported shapes use the existing exact global scorer. A separate 44-gate
native lifecycle run proved that INSERT and RETIRE immediately disable the
read image; bounded COW maintenance rebuilds it; retired slots remain excluded;
and cache clear plus immediate-stop restart preserve exact rows and scores.

## Product Interpretation

This evidence supports the convergent design:

- foreground lexical writes can remain immediate and append-oriented;
- queries over newly sealed segments stay exact with bounded temporary cost;
- persistent term-local fold recovers hot-path locality after the workload
  proves that the rewrite is worthwhile;
- fold is a performance-only COW publication, never a second index or a
  correctness dependency;
- impact specialization is a disposable, epoch-bound read image over the same
  term authority, not another lifecycle or index;
- cold terms remain bounded fragmented instead of forcing a complete corpus
  fold;
- eventual SAE inference remains a separate mandatory worker stage ahead of
  optional workload optimization.

This closes the focused converged-hot-path no-regression gate against the
legacy materialized-impact control. It does not close the complete release
qualification matrix. Remaining work includes active-L0/fold-plus-tail timing,
cold-to-hot transition cost, mixed query distributions, concurrent writes,
CPU and I/O accounting, SAE-enabled query states, and the fixed
qualification-host static matrix.
