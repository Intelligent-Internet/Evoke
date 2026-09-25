\set ON_ERROR_STOP on
SET statement_timeout = '60s';
SET evoke.test_disable_semantic_accelerator = off;
SET evoke.test_force_semantic_bmp = off;
SELECT set_config('evoke.test_filtered_forward_route', 'auto', false);
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
CROSS JOIN generate_series(1, 5) AS attempt(attempt_index)
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
        'relation_bytes',
            pg_relation_size('bench.pubmed_full_v2_idx'::regclass)
    )
FROM source;

SELECT
    route,
    filter_name,
    count(*) AS attempts,
    round(
        percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms)::numeric,
        3
    ) AS p50_ms,
    round(
        percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms)::numeric,
        3
    ) AS p95_ms,
    array_agg(DISTINCT trace->>'query_route') AS query_routes,
    max((trace->>'accelerator_forward_bytes')::int8) AS forward_bytes,
    max((trace->>'ranked_prefix_probe_documents_examined')::int8)
        AS prefix_documents,
    max((trace->>'ranked_prefix_probe_postings_examined')::int8)
        AS prefix_postings,
    max((trace->>'memory_bytes')::int8) AS memory_bytes
FROM bench.cq3_full_results
WHERE route = current_setting('application_name')
GROUP BY route, filter_name
ORDER BY filter_name;
