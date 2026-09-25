\set ON_ERROR_STOP on

\if :{?baseline_index}
\else
    \set baseline_index bench.pubmed_full_v2_idx
\endif
\if :{?candidate_index}
\else
    \set candidate_index bench.pubmed_full_cq3e_streamed_idx
\endif

SET statement_timeout = '30min';
SET evoke.test_disable_semantic_accelerator = on;

CREATE TEMP TABLE cq3e_queries(query text PRIMARY KEY);
INSERT INTO cq3e_queries(query) VALUES
    ('cancer immunotherapy survival'),
    ('gene expression biomarker'),
    ('cardiovascular disease prevention'),
    ('machine learning clinical diagnosis'),
    ('public health intervention');

CREATE TEMP TABLE cq3e_hits AS
SELECT
    queries.query,
    variants.root,
    variants.replay,
    hit.rank_position,
    hit.doc_id,
    hit.score
FROM cq3e_queries AS queries
CROSS JOIN (
    VALUES
        ('old'::text, 1, :'baseline_index'::regclass),
        ('old'::text, 2, :'baseline_index'::regclass),
        ('new'::text, 1, :'candidate_index'::regclass),
        ('new'::text, 2, :'candidate_index'::regclass)
) AS variants(root, replay, index_name)
CROSS JOIN LATERAL evoke_query(
    variants.index_name,
    queries.query,
    100
) WITH ORDINALITY AS hit(ctid, doc_id, score, rank_position);

CREATE TEMP TABLE cq3e_comparison AS
WITH old_hits AS (
    SELECT query, rank_position, doc_id, score
    FROM cq3e_hits
    WHERE root = 'old' AND replay = 1
), new_hits AS (
    SELECT query, rank_position, doc_id, score
    FROM cq3e_hits
    WHERE root = 'new' AND replay = 1
), by_document AS (
    SELECT
        coalesce(old_hits.query, new_hits.query) AS query,
        old_hits.rank_position AS old_rank,
        new_hits.rank_position AS new_rank,
        old_hits.doc_id AS old_doc_id,
        new_hits.doc_id AS new_doc_id,
        old_hits.score AS old_score,
        new_hits.score AS new_score
    FROM old_hits
    FULL JOIN new_hits USING (query, doc_id)
), migration AS (
    SELECT
        query,
        count(*) FILTER (WHERE old_doc_id IS NOT NULL) AS old_count,
        count(*) FILTER (WHERE new_doc_id IS NOT NULL) AS new_count,
        count(*) FILTER (
            WHERE old_doc_id IS NULL OR new_doc_id IS NULL
        ) AS membership_diff,
        count(*) FILTER (WHERE old_rank <> new_rank) AS moved_documents,
        coalesce(max(abs(new_rank - old_rank)), 0) AS max_rank_displacement,
        count(*) FILTER (
            WHERE old_score IS NOT NULL
              AND new_score IS NOT NULL
              AND pg_catalog.float4send(old_score)
                  <> pg_catalog.float4send(new_score)
        ) AS score_bit_diff,
        coalesce(max(abs(new_score - old_score)), 0) AS max_abs_score_delta,
        coalesce(max(
            abs(new_score - old_score) /
                greatest(abs(old_score), 1.0e-12)
        ), 0) AS max_relative_score_delta
    FROM by_document
    GROUP BY query
), replay AS (
    SELECT
        coalesce(first.query, second.query) AS query,
        coalesce(first.root, second.root) AS root,
        count(*) FILTER (
            WHERE first.rank_position IS NULL
               OR second.rank_position IS NULL
               OR first.doc_id <> second.doc_id
        ) AS rank_identity_diff,
        count(*) FILTER (
            WHERE first.rank_position IS NOT NULL
              AND second.rank_position IS NOT NULL
              AND pg_catalog.float4send(first.score)
                  <> pg_catalog.float4send(second.score)
        ) AS score_bit_diff
    FROM (
        SELECT query, root, rank_position, doc_id, score
        FROM cq3e_hits
        WHERE replay = 1
    ) AS first
    FULL JOIN (
        SELECT query, root, rank_position, doc_id, score
        FROM cq3e_hits
        WHERE replay = 2
    ) AS second USING (query, root, rank_position)
    GROUP BY coalesce(first.query, second.query),
        coalesce(first.root, second.root)
), replay_summary AS (
    SELECT
        query,
        max(rank_identity_diff) FILTER (WHERE root = 'old') AS
            old_replay_rank_diff,
        max(score_bit_diff) FILTER (WHERE root = 'old') AS
            old_replay_score_diff,
        max(rank_identity_diff) FILTER (WHERE root = 'new') AS
            new_replay_rank_diff,
        max(score_bit_diff) FILTER (WHERE root = 'new') AS
            new_replay_score_diff
    FROM replay
    GROUP BY query
)
SELECT migration.*, replay_summary.old_replay_rank_diff,
    replay_summary.old_replay_score_diff,
    replay_summary.new_replay_rank_diff,
    replay_summary.new_replay_score_diff
FROM migration
JOIN replay_summary USING (query);

SELECT *
FROM cq3e_comparison
ORDER BY query;

SELECT
    count(*) = 5 AND bool_and(
        old_count = 100 AND
        new_count = 100 AND
        membership_diff = 0 AND
        max_rank_displacement <= 1 AND
        max_relative_score_delta <= 0.002 AND
        old_replay_rank_diff = 0 AND
        old_replay_score_diff = 0 AND
        new_replay_rank_diff = 0 AND
        new_replay_score_diff = 0
    ) AS migration_ok
FROM cq3e_comparison
\gset
\if :migration_ok
\else
    DO $failure$
    BEGIN
        RAISE EXCEPTION
            'CQ-3E qualification failed: runtime migration is unstable';
    END
    $failure$;
\endif

SELECT
    query,
    root,
    replay,
    count(*) AS hit_count,
    md5(string_agg(
        rank_position::text || ':' || doc_id::text || ':' ||
            encode(pg_catalog.float4send(score), 'hex'),
        ',' ORDER BY rank_position
    )) AS ranked_result_digest
FROM cq3e_hits
GROUP BY query, root, replay
ORDER BY query, root, replay;
