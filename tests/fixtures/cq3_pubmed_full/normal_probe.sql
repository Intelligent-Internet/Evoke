\set ON_ERROR_STOP on
SET statement_timeout = '60s';
SET evoke.test_disable_semantic_accelerator = off;
SET evoke.test_force_semantic_bmp = off;
SELECT set_config(
    'evoke.test_filtered_forward_route',
    :'forward_route',
    false
);
SET application_name = :'route';

DELETE FROM bench.cq3_full_results
WHERE route = current_setting('application_name');
DELETE FROM bench.cq3_full_hits
WHERE route = current_setting('application_name');

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

WITH route_summary AS (
    SELECT
        route,
        filter_name,
        count(*) AS attempts,
        bool_and(hit_count = 50) AS topk_complete,
        array_agg(DISTINCT trace->>'query_route') AS query_routes,
        round(
            percentile_cont(0.5) WITHIN GROUP (
                ORDER BY elapsed_ms
            )::numeric,
            3
        ) AS p50_ms,
        round(
            percentile_cont(0.95) WITHIN GROUP (
                ORDER BY elapsed_ms
            )::numeric,
            3
        ) AS p95_ms,
        max((trace->>'accelerator_forward_bytes')::int8) AS forward_bytes,
        max((trace->>'accelerator_forward_postings_examined')::int8)
            AS forward_postings,
        max((trace->>'ranked_prefix_probe_attempts')::int4)
            AS prefix_attempts,
        max((trace->>'ranked_prefix_probe_documents_examined')::int8)
            AS prefix_documents,
        max((trace->>'ranked_prefix_probe_postings_examined')::int8)
            AS prefix_postings,
        max((trace->>'ranked_prefix_probe_memory_bytes')::int8)
            AS prefix_memory_bytes,
        max((trace->>'semantic_bmp_ref_reads')::int8) AS bound_refs,
        max((trace->>'semantic_bmp_postings_examined')::int8)
            AS semantic_postings,
        max((trace->>'documents_examined')::int8) AS documents_examined,
        max((trace->>'document_block_reads')::int8)
            AS document_block_reads,
        max((trace->>'positive_document_count')::int8)
            AS positive_documents,
        max((trace->>'zero_score_documents_added')::int8)
            AS zero_score_documents_added,
        max((trace->>'zero_score_cow_objects_loaded')::int8)
            AS zero_score_cow_objects_loaded,
        max((trace->>'zero_score_cow_records_examined')::int8)
            AS zero_score_cow_records_examined,
        max((trace->>'zero_score_heap_peak')::int8)
            AS zero_score_heap_peak,
        max((trace->>'memory_bytes')::int8) AS memory_bytes,
        max((trace->>'semantic_accelerator_query_bytes')::int8)
            AS accelerator_query_bytes,
        max((trace->>'accelerator_owned_index_bytes')::int8)
            AS owned_index_bytes,
        max((trace->>'accelerator_membership_bytes')::int8)
            AS membership_bytes,
        max((trace->>'accelerator_candidate_scratch_bytes')::int8)
            AS candidate_scratch_bytes,
        max((trace->>'accelerator_forward_scratch_bytes')::int8)
            AS forward_scratch_bytes,
        max((trace->>'accelerator_residual_scratch_bytes')::int8)
            AS residual_scratch_bytes,
        max((trace->>'semantic_bmp_query_bytes')::int8)
            AS semantic_bmp_query_bytes,
        min(backend_rss_bytes) AS minimum_backend_rss_bytes,
        max(backend_rss_bytes) AS maximum_backend_rss_bytes
    FROM bench.cq3_full_results
    WHERE route = current_setting('application_name')
    GROUP BY route, filter_name
), compared AS (
    SELECT
        normal.filter_name,
        count(*) AS normal_hits,
        count(exact.doc_ord) AS shared_hits
    FROM bench.cq3_full_hits AS normal
    LEFT JOIN bench.cq3_full_hits AS exact
      ON exact.route = :'candidate_route'
     AND exact.filter_name = normal.filter_name
     AND exact.doc_ord = normal.doc_ord
    WHERE normal.route = current_setting('application_name')
    GROUP BY normal.filter_name
), overlap_summary AS (
    SELECT
        filter_name,
        normal_hits,
        shared_hits,
        round(shared_hits::numeric / NULLIF(normal_hits, 0), 6) AS overlap
    FROM compared
)
SELECT json_build_object(
    'suite', 'cq3_normal_route_probe',
    'route', current_setting('application_name'),
    'forward_route', current_setting('evoke.test_filtered_forward_route'),
    'candidate_route', :'candidate_route',
    'summaries', (
        SELECT json_agg(route_summary ORDER BY filter_name)
        FROM route_summary
    ),
    'overlap', (
        SELECT json_agg(overlap_summary ORDER BY filter_name)
        FROM overlap_summary
    )
);
