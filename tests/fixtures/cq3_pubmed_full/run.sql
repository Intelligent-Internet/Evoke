\set ON_ERROR_STOP on
SET statement_timeout = '60s';
SET evoke.test_disable_semantic_accelerator = on;
SET evoke.test_force_semantic_bmp = on;
SET application_name = :'route';

DELETE FROM bench.cq3_full_results
WHERE route = current_setting('application_name');
DELETE FROM bench.cq3_full_hits
WHERE route = current_setting('application_name');
DELETE FROM bench.cq3_full_root_identity
WHERE route = current_setting('application_name');

WITH source AS (
    SELECT evoke_index_generation_status_internal(
        'bench.pubmed_full_v2_idx'::regclass
    )::jsonb AS status
)
INSERT INTO bench.cq3_full_root_identity(route, checkpoint, identity)
SELECT
    current_setting('application_name'),
    'before',
    jsonb_build_object(
        'generation_id', status->>'generation_id',
        'contract_signature', status->>'contract_signature',
        'valid', status->'valid',
        'runtime_contract_matches', status->'runtime_contract_matches',
        'health', status->>'health',
        'accelerator_source_manifest_id',
            status#>'{semantic_accelerator,source_manifest_id}',
        'accelerator_builder_policy_id',
            status#>'{semantic_accelerator,builder_policy_id}',
        'accelerator_state',
            status#>>'{semantic_accelerator,state}',
        'relation_bytes',
            pg_relation_size('bench.pubmed_full_v2_idx'::regclass)
    )
FROM source;

SELECT format(
    'SELECT bench.cq3_full_capture_probe(%L, %s);',
    filter_set.name,
    attempt.attempt_index
)
FROM bench.cq3_full_filter_sets AS filter_set
CROSS JOIN generate_series(1, 10) AS attempt(attempt_index)
ORDER BY filter_set.name, attempt.attempt_index
\gexec

SELECT format(
    'SELECT bench.cq3_full_capture_hits(%L);',
    filter_set.name
)
FROM bench.cq3_full_filter_sets AS filter_set
ORDER BY filter_set.name
\gexec

WITH source AS (
    SELECT evoke_index_generation_status_internal(
        'bench.pubmed_full_v2_idx'::regclass
    )::jsonb AS status
)
INSERT INTO bench.cq3_full_root_identity(route, checkpoint, identity)
SELECT
    current_setting('application_name'),
    'after',
    jsonb_build_object(
        'generation_id', status->>'generation_id',
        'contract_signature', status->>'contract_signature',
        'valid', status->'valid',
        'runtime_contract_matches', status->'runtime_contract_matches',
        'health', status->>'health',
        'accelerator_source_manifest_id',
            status#>'{semantic_accelerator,source_manifest_id}',
        'accelerator_builder_policy_id',
            status#>'{semantic_accelerator,builder_policy_id}',
        'accelerator_state',
            status#>>'{semantic_accelerator,state}',
        'relation_bytes',
            pg_relation_size('bench.pubmed_full_v2_idx'::regclass)
    )
FROM source;

SELECT
    route,
    filter_name,
    count(*) AS attempts,
    min(hit_count) AS minimum_hits,
    round(
        percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms)::numeric,
        3
    ) AS p50_ms,
    round(
        percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms)::numeric,
        3
    ) AS p95_ms,
    min(backend_memory_bytes) AS minimum_backend_memory_bytes,
    max(backend_memory_bytes) AS maximum_backend_memory_bytes,
    min(backend_rss_bytes) AS minimum_backend_rss_bytes,
    max(backend_rss_bytes) AS maximum_backend_rss_bytes,
    max((trace->>'semantic_bmp_ref_reads')::int8) AS bound_refs,
    max((trace->>'semantic_bmp_super_ref_reads')::int8) AS super_refs,
    max((trace->>'semantic_bmp_record_reads')::int8) AS records,
    max((trace->>'semantic_bmp_postings_examined')::int8)
        AS semantic_postings,
    max((trace->>'semantic_bmp_query_bytes')::int8) AS query_bytes
FROM bench.cq3_full_results
WHERE route = current_setting('application_name')
GROUP BY route, filter_name
ORDER BY filter_name;

SELECT route, checkpoint, identity
FROM bench.cq3_full_root_identity
WHERE route = current_setting('application_name')
ORDER BY checkpoint;
