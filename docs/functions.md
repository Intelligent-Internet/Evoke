# Function Index

This is a compact lookup. [API Reference](api-reference.md) is authoritative
for contracts, privileges, and examples.

SSR means Sparse Semantic Retrieval, the semantic-enabled `sae = true`
product path.

## Product Functions

| Function | Purpose |
| --- | --- |
| `evoke_query(...)` | Scalar 2/4-argument overloads provide planner-native semantic ranking; `k`-bearing overloads return explicit hits for either index mode, field-aware weighting, JSON compatibility filters, or low-level `tid[]` subsets. |
| `evoke_index_options(regclass)` | Report effective index configuration. |
| `evoke_index_status(regclass)` | Report query readiness and blocker state. |
| `evoke_index_audit(regclass)` | Run the explicit heavy integrity and model-artifact audit. |
| `evoke_index_details(regclass)` | Report operator-level physical and maintenance state. |
| `evoke_index_policy_recommend(regclass, text)` | Return advisory BM25/SSR policy options. |
| `evoke_index_refresh(regclass)` | Request explicit index refresh. |
| `evoke_index_maintain(regclass)` | Perform one blocking maintenance attempt. |
| `evoke_index_try_maintain(regclass)` | Skip a busy maintenance lock; an admitted maintenance action can still take time. |
| `evoke_index_maintain_due(integer)` | Maintain a bounded set of due owned indexes. |
| `evoke_fusion_query(...)` | Search and weight multiple independent Evoke indexes. |
| `evoke_fusion_query_fields(...)` | Compose named, weighted Evoke sources. |
| `evoke_hybrid_bm25_candidates(...)` | Adapt an Evoke source to hybrid candidates. |
| `evoke_hybrid_fuse_candidates(...)` | Fuse Evoke and external candidate sets. |

SSR is eventual-only. Single-index functions act on one page-native v3 index
relation; composition functions combine independently maintained sources above
that layer. There is no model-specific search or maintenance API. Removal uses
PostgreSQL `DROP INDEX`.

## Public Text Utilities

- `evoke_tokenize_text(...)`;
- `evoke_normalize_tokens(...)`;
- `evoke_highlight(...)`;
- `evoke_snippet(...)`.

## Public Composition Functions

The `evoke_fusion_*` and `evoke_hybrid_*` families are public product APIs. They
compose independently retrieved candidate sets above the single-index layer;
they do not add another lifecycle or change any source index.

## Owner Diagnostic Functions

The exact rowset, prepared-query, token-level field-weight, and local match/score
families are owner-only diagnostics. Principal names are:

- `evoke_query_ids(...)` and `evoke_query_tokens(...)`;
- `evoke_prepared_query(...)`, `evoke_order_tokens(...)`;
- `evoke_field_aware_query(...)` and
  `evoke_field_aware_query_tokens(...)`;
- local prepared-query match and score helpers.

They are revoked from `PUBLIC`, do not replace `evoke_query(...)`, and reject
semantic-enabled indexes where an exact-BM25 diagnostic state is required.
Use PostgreSQL `EXPLAIN (FORMAT JSON)` directly for planner diagnostics.

## Runtime And Build Diagnostics

Runtime-state diagnostics require `SELECT` on the indexed table:

- `evoke_index_runtime_state(...)`;
- `evoke_index_runtime_state_json(...)`.

Explicit `evoke_index_preload(...)` requires index ownership.

Extension-owner diagnostics revoked from `PUBLIC` are:

- `evoke_runtime_cache_clear()`;
- `evoke_runtime_service_status()`;
- `evoke_onnxruntime_probe()`.

`evoke_onnxruntime_build_info()` exposes non-mutating linked-runtime build
metadata. These functions describe page-native root/residency and
shared-runtime state, not posting storage.

All `_internal` functions are implementation boundaries and must remain
unavailable to application roles.
