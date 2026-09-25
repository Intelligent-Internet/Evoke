# Scoring Profile Reference

The scoring profile is a digest-validated checkout artifact. It defines how the
encoder's sparse atoms are normalized, admitted, and scored. Query-time dense
retrieval is not an input. The following is an abbreviated excerpt from the
[Beta 1 checkout](semantic-model-checkout.md#download-the-default-model), not a
complete replacement profile or an application-level tuning configuration:

```json
{
  "profile": "p2_unified_sparse_dot_v1",
  "version": 1,
  "score_inputs": ["lexical_postings", "semantic_postings"],
  "score": "sum(query_impact * document_impact)",
  "dense_runtime_dependency": false,
  "document_compiler": {
    "semantic_budget_ratio_to_lexical": 1.125,
    "semantic_pruning": "m1933_balanced_prune_rows"
  },
  "query_compiler": {
    "mode": "rms",
    "scale_constraint_order": "multiply_then_clip"
  }
}
```

The manifest names the profile and declares the artifact path and SHA-256. The
index validates the manifest, artifact, atom namespace, and profile as one
checkout contract. See the model technical report for
[the complete scoring design](../technical-report-evoke-model.md).
The full artifact also records calibration provenance and encoder settings.
Its `candidate_k` records the evaluation configuration, not the SQL API's
default result limit. Encoder `max_length` is a window limit; P2.2 document
compilation handles long input in windows rather than truncating the whole
document to that size.

```sql
SELECT ii42_index_options('docs_search_idx'::regclass);
SELECT ii42_index_status('docs_search_idx'::regclass);
SELECT * FROM ii42_query('docs_search_idx'::regclass, 'query', 20);
DROP INDEX docs_search_idx;
```

Changing a profile requires a model checkout update followed by `REINDEX`; it
is not an independent corpus or scorer publication.
