#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify product/runtime service privilege boundaries.',
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        **kwargs,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def psql(psql_bin: Path, socket_dir: Path, port: int, sql: str) -> str:
    result = run(
        [
            str(psql_bin),
            '-X',
            '-q',
            '-t',
            '-A',
            '-h',
            str(socket_dir),
            '-p',
            str(port),
            '-d',
            'postgres',
            '-v',
            'ON_ERROR_STOP=1',
        ],
        input=sql,
    )
    return result.stdout.strip()


def privilege_sql(model_path: Path) -> str:
    escaped_path = str(model_path).replace("'", "''")
    return f'''
    CREATE EXTENSION evoke;
    CREATE ROLE evoke_product_app LOGIN;
    CREATE ROLE evoke_no_table_access LOGIN;

    CREATE TABLE docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO docs (id, body) VALUES
        (1, 'alpha cuda'),
        (2, 'semantic gpu'),
        (3, 'optimization graph');

    CREATE INDEX docs_body_idx
    ON docs
    USING evoke (body)
    WITH (
        sae = true,
        model_path = '{escaped_path}'
    );

    CREATE TABLE rls_bm25_docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO rls_bm25_docs (id, body) VALUES
        (1, 'visible alpha token'),
        (2, 'hidden omega sentinel');
    CREATE INDEX rls_bm25_docs_idx
    ON rls_bm25_docs USING evoke (body);

    CREATE TABLE rls_model_docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO rls_model_docs (id, body) VALUES
        (1, 'visible alpha token'),
        (2, 'hidden omega sentinel');
    CREATE INDEX rls_model_docs_idx
    ON rls_model_docs
    USING evoke (body)
    WITH (
        sae = true,
        model_path = '{escaped_path}'
    );

    CREATE TABLE partitioned_docs (
        id int NOT NULL,
        body text NOT NULL
    ) PARTITION BY RANGE (id);
    CREATE TABLE partitioned_docs_p0
        PARTITION OF partitioned_docs FOR VALUES FROM (0) TO (100);
    INSERT INTO partitioned_docs (id, body) VALUES
        (1, 'partition visible alpha token'),
        (2, 'partition hidden omega sentinel');
    CREATE INDEX partitioned_docs_idx
        ON partitioned_docs USING evoke (body);

    CREATE TABLE rls_partitioned_docs (
        id int NOT NULL,
        body text NOT NULL
    ) PARTITION BY RANGE (id);
    CREATE TABLE rls_partitioned_docs_p0
        PARTITION OF rls_partitioned_docs FOR VALUES FROM (0) TO (100);
    INSERT INTO rls_partitioned_docs (id, body) VALUES
        (1, 'partition visible alpha token'),
        (2, 'partition hidden omega sentinel');
    CREATE INDEX rls_partitioned_docs_idx
        ON rls_partitioned_docs USING evoke (body);

    ALTER TABLE rls_bm25_docs ENABLE ROW LEVEL SECURITY;
    ALTER TABLE rls_model_docs ENABLE ROW LEVEL SECURITY;
    ALTER TABLE rls_partitioned_docs ENABLE ROW LEVEL SECURITY;
    CREATE POLICY rls_bm25_visible ON rls_bm25_docs
        FOR SELECT TO evoke_product_app USING (id = 1);
    CREATE POLICY rls_model_visible ON rls_model_docs
        FOR SELECT TO evoke_product_app USING (id = 1);
    CREATE POLICY rls_partitioned_visible ON rls_partitioned_docs
        FOR SELECT TO evoke_product_app USING (id = 1);

    GRANT SELECT ON docs TO evoke_product_app;
    GRANT SELECT ON
        rls_bm25_docs,
        rls_model_docs,
        partitioned_docs,
        rls_partitioned_docs
    TO evoke_product_app;
    CREATE SCHEMA app_owned AUTHORIZATION evoke_product_app;

    DO $$
    DECLARE
        service_status jsonb;
    BEGIN
        SELECT evoke_runtime_service_status() INTO service_status;
        IF (
            service_status->>'ready_for_text_encoding'
        )::boolean IS DISTINCT FROM true THEN
            RAISE EXCEPTION 'runtime service is not ready: %',
                service_status;
        END IF;
        IF service_status->>'queue_policy'
            <> 'bounded_affinity_worker_pool'
            OR service_status->>'queue_timeout_policy' <> 'none'
            OR (service_status->>'document_queue_limit')::int <> 1 THEN
            RAISE EXCEPTION
                'runtime queue policy contract changed: %',
                service_status;
        END IF;
        IF NOT (service_status ? 'canceled_requests')
            OR NOT (service_status ? 'orphan_responses')
            OR NOT (service_status ? 'worker_recoveries') THEN
            RAISE EXCEPTION
                'runtime recovery counters missing: %',
                service_status;
        END IF;
    END;
    $$;

    SET ROLE evoke_no_table_access;

    DO $$
    BEGIN
        PERFORM evoke_index_options('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'index options should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_status('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'index status should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_fusion_query(
            ARRAY['docs_body_idx'::regclass],
            'fusion without source table access',
            ARRAY[1.0]::real[],
            3
        );
        RAISE EXCEPTION
            'fusion query should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_options('partitioned_docs_idx'::regclass);
        RAISE EXCEPTION
            'partition options should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_status('partitioned_docs_idx'::regclass);
        RAISE EXCEPTION
            'partition status should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_encode_text_internal(
            'docs_body_idx'::regclass,
            'direct encoder without source table access'
        );
        RAISE EXCEPTION
            'direct encoder should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_encode_document_batch_internal(
            'docs_body_idx'::regclass,
            ARRAY['batch encoder without source table access']
        );
        RAISE EXCEPTION
            'batch encoder should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_runtime_state('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'cache state should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_shared_preload_resident(
            'docs_body_idx'::regclass
        );
        RAISE EXCEPTION
            'cache residency should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM * FROM evoke_index_details('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'index details should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_index_policy_recommend(
            'docs_body_idx'::regclass,
            'balanced'
        );
        RAISE EXCEPTION
            'index policy should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'query without source table access',
            3
        );
        RAISE EXCEPTION
            'unified search should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'direct BM25 query without source table access',
            3
        );
        RAISE EXCEPTION
            'direct BM25 query should require source table SELECT privilege';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_index_semantic_query_native_internal(
            'docs_body_idx'::regclass,
            ARRAY[0]::int4[],
            ARRAY[1.0]::real[],
            NULL::text[],
            NULL::real[],
            1,
            repeat('0', 32)
        );
        RAISE EXCEPTION
            'direct semantic scorer should require source table SELECT';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    RESET ROLE;

    SET ROLE evoke_product_app;

    DO $$
    DECLARE
        normal_rows jsonb;
        oracle_requested_rows jsonb;
    BEGIN
        PERFORM set_config(
            'evoke.test_unified_overlay_oracle',
            'off',
            true
        );
        SELECT coalesce(
            jsonb_agg(
                jsonb_build_array(
                    hit.ctid::text,
                    hit.doc_id,
                    hit.score
                ) ORDER BY hit.ordinality
            ),
            '[]'::jsonb
        )
        INTO normal_rows
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'application role oracle isolation',
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, ordinality);

        PERFORM set_config(
            'evoke.test_unified_overlay_oracle',
            'on',
            true
        );
        SELECT coalesce(
            jsonb_agg(
                jsonb_build_array(
                    hit.ctid::text,
                    hit.doc_id,
                    hit.score
                ) ORDER BY hit.ordinality
            ),
            '[]'::jsonb
        )
        INTO oracle_requested_rows
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'application role oracle isolation',
            3
        ) WITH ORDINALITY AS hit(ctid, doc_id, score, ordinality);

        IF oracle_requested_rows IS DISTINCT FROM normal_rows THEN
            RAISE EXCEPTION
                'test oracle GUC changed page-native product results: % vs %',
                normal_rows,
                oracle_requested_rows;
        END IF;
    END;
    $$;

    DO $$
    DECLARE
        function_signature text;
        restricted_functions text[] := ARRAY[
            'evoke_runtime_cache_clear()',
            'evoke_index_touch_maintenance()',
            'evoke_index_maintain_due(integer)',
            'evoke_index_try_maintenance_lock(regclass)',
            'evoke_index_maintenance_unlock(regclass)',
            'evoke_index_maintenance_lock_held(regclass)',
            'evoke_runtime_service_query_atoms(text,text)',
            'evoke_runtime_service_query_atoms(text,text,text)',
            'evoke_runtime_service_query_atoms_batch(text,text[])',
            'evoke_runtime_service_query_atoms_batch(text,text,text[])',
            'evoke_runtime_service_document_atoms_batch(text,text[])',
            'evoke_runtime_service_document_atoms_batch(text,text,text[])',
            'evoke_runtime_service_atoms_batch_internal(text,text,text[])',
            'evoke_runtime_service_atoms_batch_internal(text,text,text,text[])',
            'evoke_runtime_service_status()',
            'evoke_index_runtime_plan_internal(text,jsonb)',
            'evoke_index_options_internal(regclass)',
            'evoke_index_runtime_signature_internal(regclass)',
            'evoke_index_generation_signature_internal(regclass)',
            'evoke_index_generation_status_internal(regclass)',
            'evoke_index_generation_audit_internal(regclass)',
            'evoke_encode_text_internal(regclass,text)',
            'evoke_encode_document_batch_internal(regclass,text[])',
            'evoke_index_semantic_query_native_internal('
                || 'regclass,integer[],real[],text[],real[],integer,text,'
                || 'integer[],tid[],jsonb)',
            'evoke_query_semantic_internal('
                || 'regclass,text,integer,text[],real[],integer[],tid[],'
                || 'jsonb)',
            'evoke_query_internal('
                || 'regclass,text,text[],real[],integer,real[],boolean,'
                || 'text[],boolean,boolean,integer[],tid[],jsonb)',
            'evoke_query_bm25_internal('
                || 'regclass,text,integer,real[],boolean,text[],boolean,'
                || 'boolean)',
            'evoke_query_ids(regclass,integer[],integer,real[])',
            'evoke_prepared_query(regclass,text,boolean,text[],boolean,boolean)'
        ];
    BEGIN
        IF NOT has_function_privilege(
            current_user,
            'evoke_query(regclass,text,integer,real[],boolean,text[],boolean,boolean)',
            'EXECUTE'
        ) THEN
            RAISE EXCEPTION 'application role cannot execute evoke_query';
        END IF;
        IF NOT has_function_privilege(
            current_user,
            'evoke_query(regclass,text,text[],real[],integer)',
            'EXECUTE'
        ) THEN
            RAISE EXCEPTION
                'application role cannot execute weighted evoke_query';
        END IF;
        IF NOT has_function_privilege(
            current_user,
            'evoke_query(regclass,text,jsonb,integer)',
            'EXECUTE'
        ) THEN
            RAISE EXCEPTION
                'application role cannot execute filtered evoke_query';
        END IF;
        FOREACH function_signature IN ARRAY restricted_functions
        LOOP
            IF has_function_privilege(
                current_user,
                function_signature,
                'EXECUTE'
            ) THEN
                RAISE EXCEPTION
                    'application role can execute restricted function %',
                    function_signature;
            END IF;
        END LOOP;
    END;
    $$;

    DO $$
    DECLARE
        exposed_diagnostics text[];
        exposed_result_routes text[];
    BEGIN
        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO exposed_diagnostics
        FROM pg_catalog.pg_proc AS procedure
        JOIN pg_catalog.pg_extension AS extension
          ON extension.extnamespace = procedure.pronamespace
        WHERE extension.extname = 'evoke'
          AND procedure.proname = ANY (ARRAY[
              'evoke_query_ids',
              'evoke_query_tokens',
              'evoke_field_aware_query_tokens',
              'evoke_field_aware_query',
              'evoke_prepared_query',
              'evoke_order_tokens',
              'evoke_op_match_prepared_query',
              'evoke_op_match_prepared_query_scalar',
              'evoke_match_prepared_query',
              'evoke_match_query',
              'evoke_score_prepared_query',
              'evoke_score_query'
          ])
          AND pg_catalog.has_function_privilege(
              current_user,
              procedure.oid,
              'EXECUTE'
          );
        IF exposed_diagnostics IS NOT NULL THEN
            RAISE EXCEPTION
                'application role can execute diagnostic query functions: %',
                exposed_diagnostics;
        END IF;

        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO exposed_result_routes
        FROM pg_catalog.pg_proc AS procedure
        JOIN pg_catalog.pg_extension AS extension
          ON extension.extnamespace = procedure.pronamespace
        JOIN pg_catalog.pg_type AS result_type
          ON result_type.oid = procedure.prorettype
        WHERE extension.extname = 'evoke'
          AND procedure.proretset
          AND result_type.typnamespace = procedure.pronamespace
          AND result_type.typname = ANY (ARRAY[
              'evoke_result_hit',
              'evoke_result_hybrid_candidate',
              'evoke_result_hybrid_hit'
          ])
          AND procedure.proname <> ALL (ARRAY[
              'evoke_query',
              'evoke_fusion',
              'evoke_fusion_query',
              'evoke_fusion_query_fields',
              'evoke_fusion_query_weighted',
              'evoke_hybrid_bm25_candidates',
              'evoke_hybrid_fuse_candidates'
          ])
          AND pg_catalog.has_function_privilege(
              current_user,
              procedure.oid,
              'EXECUTE'
          );
        IF exposed_result_routes IS NOT NULL THEN
            RAISE EXCEPTION
                'application role can execute alternate top-k routes: %',
                exposed_result_routes;
        END IF;
    END;
    $$;

    DO $$
    DECLARE
        unavailable_product_functions text[];
    BEGIN
        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO unavailable_product_functions
        FROM pg_catalog.pg_proc AS procedure
        JOIN pg_catalog.pg_extension AS extension
          ON extension.extnamespace = procedure.pronamespace
        WHERE extension.extname = 'evoke'
          AND procedure.proname = ANY (ARRAY[
              'evoke_fusion_weighted_query',
              'evoke_fusion_weighted_queries',
              'evoke_fusion_field_query',
              'evoke_fusion_field_queries',
              'evoke_fusion',
              'evoke_fusion_query',
              'evoke_fusion_query_fields',
              'evoke_fusion_query_weighted',
              'evoke_hybrid_candidate',
              'evoke_hybrid_bm25_candidate',
              'evoke_hybrid_vector_candidate',
              'evoke_hybrid_bm25_candidates',
              'evoke_hybrid_fuse_candidates'
          ])
          AND NOT pg_catalog.has_function_privilege(
              current_user,
              procedure.oid,
              'EXECUTE'
          );
        IF unavailable_product_functions IS NOT NULL THEN
            RAISE EXCEPTION
                'application role cannot execute product composition APIs: %',
                unavailable_product_functions;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_preload(
            'docs_body_idx'::regclass
        );
        RAISE EXCEPTION
            'non-owner application role unexpectedly preloaded index cache';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_runtime_service_query_atoms(
            '{escaped_path}',
            'direct model path should stay privileged'
        );
        RAISE EXCEPTION 'direct runtime path should require superuser';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    DECLARE
        hit_count int;
        filtered_hit_count int;
        fusion_hit_count int;
        hybrid_hit_count int;
        index_options jsonb;
        index_status jsonb;
    BEGIN
        SELECT evoke_index_options('docs_body_idx'::regclass)
        INTO index_options;
        SELECT evoke_index_status('docs_body_idx'::regclass)
        INTO index_status;
        IF index_options->>'index_type' <> 'semantic'
            OR (index_status->>'query_ready')::boolean IS DISTINCT FROM true
            OR (
                index_status->>'runtime_signature_matches'
            )::boolean IS DISTINCT FROM true THEN
            RAISE EXCEPTION
                'authorized product inspection is not ready: %, %',
                index_options,
                index_status;
        END IF;

        SELECT count(*)
        INTO hit_count
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'ordinary product query uses runtime worker',
            3
        );

        IF hit_count = 0 THEN
            RAISE EXCEPTION
                'authorized unified product query returned no rows';
        END IF;

        SELECT count(*)
        INTO filtered_hit_count
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'ordinary product query uses runtime worker',
            '{{"id":{{"eq":1}}}}'::jsonb,
            3
        );
        IF filtered_hit_count <> 1 THEN
            RAISE EXCEPTION
                'authorized filtered product query returned % rows',
                filtered_hit_count;
        END IF;

        SELECT count(*)
        INTO fusion_hit_count
        FROM evoke_fusion_query(
            ARRAY[
                'docs_body_idx'::regclass,
                'docs_body_idx'::regclass
            ],
            'ordinary product fusion query',
            ARRAY[1.0, 0.5]::real[],
            3
        );
        IF fusion_hit_count = 0 THEN
            RAISE EXCEPTION
                'authorized product fusion query returned no rows';
        END IF;

        WITH candidates AS (
            SELECT array_agg(candidate) AS items
            FROM evoke_hybrid_bm25_candidates(
                'unified',
                'docs_body_idx'::regclass,
                'ordinary product hybrid query',
                1.0,
                3
            ) AS candidate
        )
        SELECT count(*)
        INTO hybrid_hit_count
        FROM candidates
        CROSS JOIN LATERAL evoke_hybrid_fuse_candidates(
            candidates.items,
            3
        );
        IF hybrid_hit_count = 0 THEN
            RAISE EXCEPTION
                'authorized product hybrid query returned no rows';
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_refresh('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'non-owner application role unexpectedly refreshed index';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_maintain('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'non-owner application role unexpectedly maintained index';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_try_maintain('docs_body_idx'::regclass);
        RAISE EXCEPTION
            'non-owner application role unexpectedly try-maintained index';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        EXECUTE 'DROP INDEX public.docs_body_idx';
        RAISE EXCEPTION
            'non-owner application role unexpectedly dropped index';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_query(
            'rls_bm25_docs_idx'::regclass,
            'hidden omega sentinel',
            10
        );
        RAISE EXCEPTION
            'BM25 product search should reject row-level security';
    EXCEPTION WHEN feature_not_supported THEN
        IF position('row-level security' IN SQLERRM) = 0 THEN
            RAISE EXCEPTION 'unexpected BM25 RLS error: %', SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_query_bm25_internal(
            'rls_bm25_docs_idx'::regclass,
            'hidden omega sentinel',
            10
        );
        RAISE EXCEPTION
            'application role should not execute direct BM25 diagnostics';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_query(
            'rls_model_docs_idx'::regclass,
            'hidden omega sentinel',
            10
        );
        RAISE EXCEPTION
            'model product search should reject row-level security';
    EXCEPTION WHEN feature_not_supported THEN
        IF position('row-level security' IN SQLERRM) = 0 THEN
            RAISE EXCEPTION 'unexpected model RLS error: %', SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_index_semantic_query_native_internal(
            'rls_model_docs_idx'::regclass,
            ARRAY[0]::int4[],
            ARRAY[1.0]::real[],
            NULL::text[],
            NULL::real[],
            10,
            repeat('0', 32)
        );
        RAISE EXCEPTION
            'application role should not use the internal semantic scorer';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_status('rls_model_docs_idx'::regclass);
        RAISE EXCEPTION
            'model status should reject row-level security';
    EXCEPTION WHEN feature_not_supported THEN
        NULL;
    END;
    $$;

    DO $$
    DECLARE
        partition_status jsonb;
    BEGIN
        SELECT evoke_index_status('partitioned_docs_idx'::regclass)
        INTO partition_status;
        IF partition_status->>'blocker'
                <> 'partitioned_parent_not_supported'
            OR (partition_status->>'query_ready')::boolean IS DISTINCT FROM
                false THEN
            RAISE EXCEPTION
                'partition parent status contract changed: %',
                partition_status;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_options('partitioned_docs_idx'::regclass);
        RAISE EXCEPTION
            'partition parent options should fail closed';
    EXCEPTION WHEN feature_not_supported THEN
        NULL;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_status('rls_partitioned_docs_idx'::regclass);
        RAISE EXCEPTION
            'partition status should reject row-level security';
    EXCEPTION WHEN feature_not_supported THEN
        IF position('row-level security' IN SQLERRM) = 0 THEN
            RAISE EXCEPTION
                'unexpected partition RLS status error: %',
                SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM evoke_index_options('rls_partitioned_docs_idx'::regclass);
        RAISE EXCEPTION
            'partition options should reject row-level security';
    EXCEPTION WHEN feature_not_supported THEN
        IF position('row-level security' IN SQLERRM) = 0 THEN
            RAISE EXCEPTION
                'unexpected partition RLS options error: %',
                SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        PERFORM *
        FROM evoke_encode_document_batch_internal(
            'docs_body_idx'::regclass,
            ARRAY['application role bulk runtime attempt']
        );
        RAISE EXCEPTION
            'application role should not use the build-only batch encoder';
    EXCEPTION WHEN insufficient_privilege THEN
        NULL;
    END;
    $$;

    CREATE TABLE app_owned.docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO app_owned.docs (id, body) VALUES
        (1, 'alpha cuda'),
        (2, 'semantic gpu');

    CREATE INDEX docs_bm25_idx
    ON app_owned.docs USING evoke (body);

    SELECT evoke_index_refresh(
        'app_owned.docs_bm25_idx'::regclass
    );
    SELECT evoke_index_maintain(
        'app_owned.docs_bm25_idx'::regclass
    );
    SELECT evoke_index_try_maintain(
        'app_owned.docs_bm25_idx'::regclass
    );

    DO $$
    DECLARE
        hit_count int;
    BEGIN
        SELECT count(*)
        INTO hit_count
        FROM evoke_query(
            'app_owned.docs_bm25_idx'::regclass,
            'alpha',
            2
        );
        IF hit_count = 0 THEN
            RAISE EXCEPTION
                'ordinary index owner could not search owned BM25 index';
        END IF;
        EXECUTE 'DROP INDEX app_owned.docs_bm25_idx';
        IF to_regclass('app_owned.docs_bm25_idx') IS NOT NULL THEN
            RAISE EXCEPTION
                'ordinary index owner could not drop owned BM25 index';
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        EXECUTE format(
            'CREATE INDEX docs_body_idx ON app_owned.docs '
            || 'USING evoke (body) '
            || 'WITH (sae = true, model_path = %L)',
            '{escaped_path}'
        );
        RAISE EXCEPTION
            'ordinary-owned SSR index should not read server-local model';
    EXCEPTION WHEN others THEN
        IF position(
            'must be owned by a superuser'
            IN SQLERRM
        ) = 0 THEN
            RAISE EXCEPTION
                'unexpected ordinary-owned index error: %',
                SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        IF to_regclass('app_owned.docs_body_idx') IS NOT NULL THEN
            RAISE EXCEPTION
                'failed ordinary-owned SSR build left a partial index';
        END IF;
    END;
    $$;

    RESET ROLE;
    '''


def main() -> None:
    args = parse_args()
    pg_bin = Path(args.pg_bin)
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    psql_bin = pg_bin / 'psql'
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
        extension_libraries = [
            args.extension_libdir / name
            for name in ('evoke.so', 'evoke.dylib')
        ]
        if not any(path.is_file() for path in extension_libraries):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )

    with tempfile.TemporaryDirectory(prefix='evoke_product_priv_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        port = 55433
        socket_dir.mkdir()
        model_path = args.model_path.expanduser().resolve()
        if not (model_path / 'manifest.json').is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {model_path}'
            )
        manifest = json.loads(
            (model_path / 'manifest.json').read_text(encoding='utf-8')
        )
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                '--model-path must use the current Evoke model contract'
            )

        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as f:
            f.write("\nshared_preload_libraries = 'evoke'\n")
            f.write("evoke.shared_runtime_size = '64MB'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                f.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(args.extension_control_dir).replace(
                    "'",
                    "''",
                )
                f.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            f.write("listen_addresses = ''\n")
            f.write("max_worker_processes = 16\n")

        started = False
        try:
            run([
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-l',
                str(log_path),
                '-o',
                f'-k {socket_dir} -p {port}',
                'start',
                '-w',
            ])
            started = True
            try:
                psql(psql_bin, socket_dir, port, privilege_sql(model_path))
            except Exception:
                if log_path.exists():
                    print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
                raise
        finally:
            if started:
                subprocess.run(
                    [str(pg_ctl), '-D', str(data_dir), 'stop', '-m', 'fast'],
                    text=True,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                )

    print('Runtime service privilege smoke passed')


if __name__ == '__main__':
    main()
