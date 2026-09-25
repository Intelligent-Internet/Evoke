# II-42 M1930B One-Index Additive Closure Report

## Result

M1930B established the first exact engineering closure for the new lexical-semantic residual route:

```text
exact BM25 postings
  + fixed-budget M1914 semantic postings in a disjoint namespace
  -> one sparse query vector
  -> one additive dot product
  -> one physical posting relation
```

There is no ANN retrieval, RRF, post-hoc reranker, or second query engine in this result.

The globally selected deterministic configuration was `b1_qall_s0.5`: semantic postings equal the lexical posting count, all semantic query atoms remain active, and the global semantic scale is `3.14080329`.

| Surface | NDCG@10 | MAP@100 | Recall@100 | MRR@20 | CUB@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.386478 | 0.296311 | 0.644862 | 0.419139 | 0.762057 |
| full M1914 | 0.452702 | 0.353850 | 0.725207 | 0.490606 | 0.860291 |
| M1930B | 0.453310 | 0.352296 | 0.721610 | 0.489030 | 0.857232 |

M1930B preserves nearly all of the unpruned M1914 quality while reducing semantic postings to a lexical-equivalent budget. It therefore proves representation and index closure, but not yet a quality breakthrough over the mature semantic parent.

## Generalization Gate

The strict leave-one-dataset-out gate did not pass. ArguAna was the only heldout failure: the train-selected configuration lost `0.012295` NDCG@10 and `0.011371` MRR@20 against its better parent. The heldout macro remained strong (`0.451245` NDCG, `0.351531` MAP, `0.717958` Recall, `0.487539` MRR, `0.856475` CUB), but the row-level failure correctly prevented residual training.

Decision: `stop_before_residual_training`.

## Native Replay

The selected surface was published to one PostgreSQL posting relation and evaluated through the existing native query path.

| Dataset | Offline/native match | Mean latency | P95 latency | Postings |
| --- | --- | ---: | ---: | ---: |
| NFCorpus | exact | 11.076 ms | 23.192 ms | 950,298 |
| SciFact | exact | 21.342 ms | 34.267 ms | 1,263,308 |

The normalized PostgreSQL relation proves score and lifecycle compatibility. Its row storage is not the final compact UBMX/BMP byte estimate because normalized tuple overhead dominates this canary representation.

## Conclusion

M1930B rules out the engine and additive score contract as blockers. Its remaining defect is source calibration: one global semantic scale is too small for ArguAna but too large for several other rows. M1931 addresses that issue without changing document postings or adding a second retrieval stage.
