# Function Index

This is a compact lookup. [API Reference](api-reference.md) is authoritative
for contracts, privileges, and examples.

SSR means Sparse Semantic Retrieval, the semantic-enabled `sae = true`
product path.

## Product Functions

| Function | Purpose |
| --- | --- |
| `ii42_query(...)` | Scalar 2/4-argument overloads provide planner-native semantic ranking; `k`-bearing overloads return explicit hits for either index mode, field-aware weighting, JSON compatibility filters, or low-level `tid[]` subsets. |
| `ii42_index_options(regclass)` | Report effective index configuration. |
| `ii42_index_status(regclass)` | Report query readiness and blocker state. |
| `ii42_index_audit(regclass)` | Run the explicit heavy integrity and model-artifact audit. |
| `ii42_index_details(regclass)` | Report operator-level physical and maintenance state. |
| `ii42_index_policy_recommend(regclass, text)` | Return advisory BM25/SSR policy options. |
| `ii42_index_refresh(regclass)` | Request explicit index refresh. |
| `ii42_index_maintain(regclass)` | Perform one blocking maintenance attempt. |
| `ii42_index_try_maintain(regclass)` | Skip a busy maintenance lock; an admitted maintenance action can still take time. |
| `ii42_index_maintain_due(integer)` | Maintain a bounded set of due owned indexes. |
| `ii42_fusion_query(...)` | Search and weight multiple independent Evoke indexes. |
| `ii42_fusion_query_fields(...)` | Compose named, weighted Evoke sources. |
| `ii42_hybrid_bm25_candidates(...)` | Adapt an Evoke source to hybrid candidates. |
| `ii42_hybrid_fuse_candidates(...)` | Fuse Evoke and external candidate sets. |

SSR is eventual-only. Single-index functions act on one page-native v3 index
relation; composition functions combine independently maintained sources above
that layer. There is no model-specific search or maintenance API. Removal uses
PostgreSQL `DROP INDEX`.

## Public Text Utilities

- `ii42_tokenize_text(...)`;
- `ii42_normalize_tokens(...)`;
- `ii42_highlight(...)`;
- `ii42_snippet(...)`.

## Public Composition Functions

The `ii42_fusion_*` and `ii42_hybrid_*` families are public product APIs. They
compose independently retrieved candidate sets above the single-index layer;
they do not add another lifecycle or change any source index.

## Owner Diagnostic Functions

The exact rowset, prepared-query, token-level field-weight, and local match/score
families are owner-only diagnostics. Principal names are:

- `ii42_query_ids(...)` and `ii42_query_tokens(...)`;
- `ii42_prepared_query(...)`, `ii42_order_tokens(...)`;
- `ii42_field_aware_query(...)` and
  `ii42_field_aware_query_tokens(...)`;
- local prepared-query match and score helpers.

They are revoked from `PUBLIC`, do not replace `ii42_query(...)`, and reject
semantic-enabled indexes where an exact-BM25 diagnostic state is required.
Use PostgreSQL `EXPLAIN (FORMAT JSON)` directly for planner diagnostics.

## Runtime And Build Diagnostics

Runtime-state diagnostics require `SELECT` on the indexed table:

- `ii42_index_runtime_state(...)`;
- `ii42_index_runtime_state_json(...)`.

Explicit `ii42_index_preload(...)` requires index ownership.

Extension-owner diagnostics revoked from `PUBLIC` are:

- `ii42_runtime_cache_clear()`;
- `ii42_runtime_service_status()`;
- `ii42_onnxruntime_probe()`.

`ii42_onnxruntime_build_info()` exposes non-mutating linked-runtime build
metadata. These functions describe page-native root/residency and
shared-runtime state, not posting storage.

All `_internal` functions are implementation boundaries and must remain
unavailable to application roles.
