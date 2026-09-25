\set ON_ERROR_STOP on
SET statement_timeout = '10min';

DO $preflight$
DECLARE
    index_persistence "char";
    heap_persistence "char";
    index_valid bool;
    index_ready bool;
    index_live bool;
    generation_status jsonb;
BEGIN
    SELECT
        index_relation.relpersistence,
        heap_relation.relpersistence,
        index_catalog.indisvalid,
        index_catalog.indisready,
        index_catalog.indislive,
        evoke_index_generation_status_internal(
            index_relation.oid
        )::jsonb
    INTO
        index_persistence,
        heap_persistence,
        index_valid,
        index_ready,
        index_live,
        generation_status
    FROM pg_class AS index_relation
    JOIN pg_index AS index_catalog
      ON index_catalog.indexrelid = index_relation.oid
    JOIN pg_class AS heap_relation
      ON heap_relation.oid = index_catalog.indrelid
    WHERE index_relation.oid = 'bench.pubmed_full_v2_idx'::regclass;

    IF index_persistence <> 'p' OR heap_persistence <> 'p' THEN
        RAISE EXCEPTION
            'CQ-3 qualification requires a permanent heap and index';
    END IF;
    IF NOT index_valid OR NOT index_ready OR NOT index_live THEN
        RAISE EXCEPTION 'CQ-3 qualification index is not query-ready';
    END IF;
    IF NOT coalesce((generation_status->>'valid')::bool, false) OR
       coalesce((generation_status->>'docs')::int8, 0) <= 0 OR
       NOT (
           coalesce(
               (generation_status#>>'{primary,segment_count}')::int8,
               0
           ) > 0 OR (
               coalesce(
                   (generation_status#>>'{primary,fragmented}')::bool,
                   false
               ) AND
               coalesce(
                   (generation_status#>>'{primary,manifest_start_block}')::int8,
                   0
               ) > 0 AND
               coalesce(
                   (generation_status#>>'{primary,manifest_pages}')::int8,
                   0
               ) > 0 AND
               coalesce(
                   (generation_status#>>'{primary,physical_blocks}')::int8,
                   0
               ) > 1 AND
               coalesce(
                   (
                       generation_status#>>
                       '{primary,published_block_high_watermark}'
                   )::int8,
                   0
               ) > 1
           )
       ) OR
       coalesce(
           (generation_status#>>'{posting,record_count}')::int8,
           0
       ) <= 0 THEN
        RAISE EXCEPTION
            'CQ-3 qualification root is empty or invalid: %',
            generation_status;
    END IF;
END
$preflight$;

DROP TABLE IF EXISTS bench.cq3_full_filter_sets;
DROP TABLE IF EXISTS bench.cq3_full_query_vector;
DROP TABLE IF EXISTS bench.cq3_full_root_identity;
DROP TABLE IF EXISTS bench.cq3_full_results;
DROP TABLE IF EXISTS bench.cq3_full_hits;

CREATE TABLE bench.cq3_full_filter_sets(
    name text PRIMARY KEY,
    tids tid[]
);
WITH aggregated AS (
    SELECT
        array_agg(ctid ORDER BY ctid) FILTER (
            WHERE mod(pmid, 161) = 0
        ) AS p006,
        array_agg(ctid ORDER BY ctid) FILTER (
            WHERE mod(pmid, 10) = 0
        ) AS p100,
        array_agg(ctid ORDER BY ctid) FILTER (
            WHERE mod(pmid, 2) = 0
        ) AS p500
    FROM bench.pubmed_full
)
INSERT INTO bench.cq3_full_filter_sets(name, tids)
SELECT filter_set.name, filter_set.tids
FROM aggregated
CROSS JOIN LATERAL (
    VALUES
        ('p006', aggregated.p006),
        ('p100', aggregated.p100),
        ('p500', aggregated.p500)
) AS filter_set(name, tids);

INSERT INTO bench.cq3_full_filter_sets(name, tids)
VALUES ('unfiltered', NULL);

CREATE TABLE bench.cq3_full_query_vector AS
WITH encoded AS MATERIALIZED (
    SELECT evoke_encode_text_internal(
        'bench.pubmed_full_v2_idx'::regclass,
        'cancer immunotherapy biomarkers'
    ) AS value
)
SELECT
    ARRAY(
        SELECT atom.value::int4
        FROM jsonb_array_elements_text(encoded.value->'atoms')
            WITH ORDINALITY AS atom(value, ordinality)
        ORDER BY atom.ordinality
    ) AS atoms,
    ARRAY(
        SELECT weight.value::real
        FROM jsonb_array_elements_text(encoded.value->'weights')
            WITH ORDINALITY AS weight(value, ordinality)
        ORDER BY weight.ordinality
    ) AS weights,
    encoded.value->>'runtime_signature' AS signature
FROM encoded;

CREATE TABLE bench.cq3_full_results(
    route text NOT NULL,
    filter_name text NOT NULL,
    attempt int4 NOT NULL,
    elapsed_ms float8 NOT NULL,
    hit_count int4 NOT NULL,
    backend_memory_bytes int8 NOT NULL,
    backend_rss_bytes int8 NOT NULL,
    trace jsonb NOT NULL
);
CREATE TABLE bench.cq3_full_root_identity(
    route text NOT NULL,
    checkpoint text NOT NULL,
    identity jsonb NOT NULL,
    PRIMARY KEY (route, checkpoint)
);
CREATE TABLE bench.cq3_full_hits(
    route text NOT NULL,
    filter_name text NOT NULL,
    rank int4 NOT NULL,
    doc_ord int4 NOT NULL,
    score float8 NOT NULL
);

CREATE OR REPLACE FUNCTION bench.cq3_full_capture_probe(
    probe_filter_name text,
    probe_attempt int4
)
RETURNS void
LANGUAGE plpgsql
AS $function$
DECLARE
    started_at timestamptz;
    elapsed_ms float8;
    hit_count int4;
    backend_memory_bytes int8;
    backend_rss_bytes int8;
    trace_value jsonb;
BEGIN
    started_at := clock_timestamp();
    SELECT count(*)::int4
    INTO hit_count
    FROM bench.cq3_full_query_vector AS query_vector
    CROSS JOIN bench.cq3_full_filter_sets AS filter_set
    CROSS JOIN LATERAL evoke_index_semantic_query_native_internal(
        'bench.pubmed_full_v2_idx'::regclass,
        query_vector.atoms,
        query_vector.weights,
        ARRAY['title', 'abstract']::text[],
        ARRAY[1.0, 1.0]::real[],
        50,
        query_vector.signature,
        NULL,
        filter_set.tids,
        NULL
    ) AS hit
    WHERE filter_set.name = probe_filter_name;
    elapsed_ms := 1000 * extract(
        epoch FROM clock_timestamp() - started_at
    );
    trace_value := evoke_query_trace_internal();
    SELECT sum(total_bytes)::int8
    INTO backend_memory_bytes
    FROM pg_backend_memory_contexts;
    SELECT rss.value[1]::int8 * 1024
    INTO backend_rss_bytes
    FROM regexp_match(
        pg_read_file('/proc/self/status'),
        'VmRSS:[[:space:]]+([0-9]+) kB'
    ) AS rss(value);
    IF backend_rss_bytes IS NULL THEN
        RAISE EXCEPTION 'could not read backend VmRSS';
    END IF;
    INSERT INTO bench.cq3_full_results(
        route,
        filter_name,
        attempt,
        elapsed_ms,
        hit_count,
        backend_memory_bytes,
        backend_rss_bytes,
        trace
    )
    VALUES (
        current_setting('application_name'),
        probe_filter_name,
        probe_attempt,
        elapsed_ms,
        hit_count,
        backend_memory_bytes,
        backend_rss_bytes,
        trace_value
    );
END;
$function$;

CREATE OR REPLACE FUNCTION bench.cq3_full_capture_hits(
    probe_filter_name text
)
RETURNS void
LANGUAGE sql
AS $function$
    INSERT INTO bench.cq3_full_hits(route, filter_name, rank, doc_ord, score)
    SELECT
        current_setting('application_name'),
        filter_set.name,
        hit.rank,
        hit.doc_ord,
        hit.score
    FROM bench.cq3_full_query_vector AS query_vector
    CROSS JOIN bench.cq3_full_filter_sets AS filter_set
    CROSS JOIN LATERAL evoke_index_semantic_query_native_internal(
        'bench.pubmed_full_v2_idx'::regclass,
        query_vector.atoms,
        query_vector.weights,
        ARRAY['title', 'abstract']::text[],
        ARRAY[1.0, 1.0]::real[],
        50,
        query_vector.signature,
        NULL,
        filter_set.tids,
        NULL
    ) AS hit
    WHERE filter_set.name = probe_filter_name
$function$;

SELECT name, cardinality(tids) AS allowed_documents
FROM bench.cq3_full_filter_sets
ORDER BY name;
SELECT cardinality(atoms) AS atoms, signature
FROM bench.cq3_full_query_vector;
