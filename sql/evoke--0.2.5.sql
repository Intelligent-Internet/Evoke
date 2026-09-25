CREATE TYPE evoke_index;

CREATE FUNCTION evoke_handler(internal)
RETURNS index_am_handler
AS 'MODULE_PATHNAME', 'evoke_handler'
LANGUAGE C;

CREATE FUNCTION evoke_in(cstring)
RETURNS evoke_index
AS 'MODULE_PATHNAME', 'evoke_in'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_out(evoke_index)
RETURNS cstring
AS 'MODULE_PATHNAME', 'evoke_out'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_recv(internal)
RETURNS evoke_index
AS 'MODULE_PATHNAME', 'evoke_recv'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_send(evoke_index)
RETURNS bytea
AS 'MODULE_PATHNAME', 'evoke_send'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE TYPE evoke_index (
    INPUT = evoke_in,
    OUTPUT = evoke_out,
    RECEIVE = evoke_recv,
    SEND = evoke_send,
    INTERNALLENGTH = variable,
    STORAGE = extended,
    ALIGNMENT = int4
);

CREATE TYPE evoke_result_hit AS (
    ctid tid,
    doc_id int4,
    score real
);

CREATE FUNCTION evoke_query(index_name regclass, query_text text)
RETURNS real
AS 'MODULE_PATHNAME', 'evoke_query_marker'
LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE
COST 1000000;

COMMENT ON FUNCTION evoke_query(regclass, text) IS
$evoke_comment$
Two-argument planner marker for natural Evoke ranked SQL. It fails closed unless
a planner-native Evoke CustomScan owns execution; it never performs row-local
scoring or falls back to a different ranking definition. Explicit-hit queries
use an overload with a required k argument.
$evoke_comment$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[]
)
RETURNS real
AS 'MODULE_PATHNAME', 'evoke_query_marker'
LANGUAGE C VOLATILE PARALLEL UNSAFE
COST 1000000;

COMMENT ON FUNCTION evoke_query(regclass, text, text[], real[]) IS
$evoke_comment$
Four-argument planner marker for field-aware natural Evoke ranked SQL. The
custom executor evaluates field expressions once per scan and uses the same
unified index root. Explicit-hit field queries also require k.
$evoke_comment$;

CREATE TYPE evoke_result_prepared_query AS (
    index_name regclass,
    query_text text,
    lowercase boolean,
    stopwords text[],
    stem_english boolean,
    fold_diacritics boolean
);

CREATE TYPE evoke_result_fusion_weighted_query AS (
    prepared_query evoke_result_prepared_query,
    weight real
);

CREATE TYPE evoke_result_fusion_field_query AS (
    field_name text,
    weighted_query evoke_result_fusion_weighted_query
);

CREATE TYPE evoke_result_hybrid_candidate AS (
    source_name text,
    ctid tid,
    raw_value real,
    source_rank int4,
    weight real,
    normalizer text,
    direction text
);

CREATE TYPE evoke_result_hybrid_hit AS (
    ctid tid,
    score real,
    source_count int4,
    source_names text[],
    raw_values real[],
    normalized_scores real[],
    weighted_scores real[],
    ranks int4[]
);

CREATE ACCESS METHOD evoke TYPE INDEX HANDLER evoke_handler;

COMMENT ON ACCESS METHOD evoke IS
'Unified sparse posting access method for exact BM25 and model-backed search';

CREATE FUNCTION evoke_index_details(index_name regclass)
RETURNS TABLE(
    index_name regclass,
    source_type text,
    docs int8,
    index_bytes int8,
    pages int8,
    stale bool,
    consistency text,
    rebuilds int8,
    pending_writes int8,
    pending_deletes int8,
    delta_records int8,
    delta_bytes int8
)
AS 'MODULE_PATHNAME', 'evoke_index_details'
LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION evoke_runtime_cache_clear()
RETURNS int4
AS 'MODULE_PATHNAME', 'evoke_runtime_cache_clear'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_runtime_state(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_index_runtime_state'
LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION evoke_index_try_maintenance_lock(index_name regclass)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_index_try_maintenance_lock'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_maintenance_unlock(index_name regclass)
RETURNS void
AS 'MODULE_PATHNAME', 'evoke_index_maintenance_unlock'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_maintenance_lock_held(index_name regclass)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_index_maintenance_lock_held'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_index_try_maintenance_lock(regclass) IS
$evoke_comment$
Try to acquire the evoke per-index maintenance lock used by C background
workers. SQL maintenance wrappers use this to deduplicate manual or external
scheduler calls with worker-driven generation lifecycle work. The lock is
session-scoped; internal callers that coordinate multiple statements must
release it explicitly from the same backend, including after statement errors.
$evoke_comment$;

COMMENT ON FUNCTION evoke_index_maintenance_unlock(regclass) IS
$evoke_comment$
Release the evoke per-index maintenance lock acquired by
evoke_index_try_maintenance_lock. This is an internal lifecycle coordination
helper for SQL maintenance wrappers and must run in the backend that acquired
the lock.
$evoke_comment$;

COMMENT ON FUNCTION evoke_index_maintenance_lock_held(regclass) IS
$evoke_comment$
Return whether the current backend already holds the evoke per-index maintenance
lock. Worker-facing maintenance helpers use this to enforce their caller-owned
lock contract instead of treating `caller_must_hold_maintenance_lock` as
metadata.
$evoke_comment$;

CREATE FUNCTION evoke_index_generation_status_internal(index_name regclass)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_index_generation_readiness_internal_c'
LANGUAGE C STABLE PARALLEL SAFE STRICT;

CREATE FUNCTION evoke_index_generation_audit_internal(
    index_name regclass
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_index_generation_audit_internal_c'
LANGUAGE C STABLE PARALLEL SAFE STRICT;

CREATE FUNCTION evoke_index_semantic_quarantine_internal(
    index_name regclass
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_index_semantic_quarantine_internal_c'
LANGUAGE C STABLE PARALLEL SAFE STRICT;

CREATE FUNCTION evoke_index_runtime_signature_internal(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_index_runtime_signature_internal_c'
LANGUAGE C VOLATILE PARALLEL UNSAFE STRICT;

CREATE FUNCTION evoke_index_generation_signature_internal(
    index_name regclass
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_index_generation_signature_internal_c'
LANGUAGE C STABLE PARALLEL SAFE STRICT;

CREATE FUNCTION evoke_packaged_model_path_internal()
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_packaged_model_path_internal_c'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION evoke_index_generation_status_internal(regclass) IS
$evoke_comment$
Read bounded generation metadata for the public product status path. This does
not perform complete COW closure, reachability, reclaim-marker, or semantic
accelerator directory audits.
$evoke_comment$;

COMMENT ON FUNCTION evoke_index_generation_audit_internal(regclass) IS
$evoke_comment$
Inspect the complete generation manifest owned by an evoke index. This explicit
deep diagnostic validates COW closure, reachable objects, reclaim markers, and
semantic accelerator directory state; its cost is proportional to index size.
$evoke_comment$;

COMMENT ON FUNCTION evoke_index_semantic_quarantine_internal(regclass) IS
$evoke_comment$
Inspect owner-only row identities and bounded retry metadata for semantic
completion quarantine. Application status exposes only aggregate quarantine
counts and lag; this diagnostic is not part of the public search API.
$evoke_comment$;

CREATE FUNCTION evoke_index_runtime_state_json(index_name regclass)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_index_runtime_state_json'
LANGUAGE C STABLE PARALLEL SAFE;

COMMENT ON FUNCTION evoke_index_runtime_state_json(regclass) IS
'Return one-snapshot structured page-native generation and runtime diagnostics.';

CREATE FUNCTION evoke_index_shared_preload_resident(index_name regclass)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_index_shared_preload_resident'
LANGUAGE C STABLE PARALLEL SAFE;

COMMENT ON FUNCTION evoke_index_shared_preload_resident(regclass) IS
'Return true when the exact current root has shared resident state or a warm marker.';

CREATE FUNCTION evoke_index_preload(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_index_preload'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_policy_recommend(
    index_name regclass,
    profile text DEFAULT 'balanced'
)
RETURNS TABLE(
    index_name regclass,
    profile text,
    confidence text,
    recommended_options text,
    recommended_consistency text,
    matches_current bool,
    refresh_now bool,
    docs int8,
    pending_total int8,
    reason text
)
AS 'MODULE_PATHNAME', 'evoke_index_policy_recommend'
LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION evoke_index_refresh(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_refresh_index'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_maintain(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_maintain_index'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_try_maintain(index_name regclass)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_try_maintain_index'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_touch_maintenance()
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_touch_maintenance'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

CREATE FUNCTION evoke_index_maintain_due(max_indexes integer DEFAULT 1)
RETURNS TABLE(index_oid regclass, result text)
AS 'MODULE_PATHNAME', 'evoke_maintain_due'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_index_maintain_due(integer) IS
$evoke_comment$
Maintain due automatic-consistency evoke indexes through the same candidate
classification, action priority, fairness, and non-blocking execution used by
the built-in worker. Convergent indexes advance one bounded semantic,
structural, or workload-fold action without scanning the corpus.
$evoke_comment$;

CREATE FUNCTION evoke_op_score_ids(doc_ids int4[], query_ids int4[])
RETURNS float8
AS 'MODULE_PATHNAME', 'evoke_score_ids_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_score_tokens(doc_tokens text[], query_tokens text[])
RETURNS float8
AS 'MODULE_PATHNAME', 'evoke_score_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_score_tokens(
    doc_tokens text[],
    query_tokens varchar[]
)
RETURNS float8
AS 'MODULE_PATHNAME', 'evoke_score_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_score_tokens(
    doc_tokens varchar[],
    query_tokens text[]
)
RETURNS float8
AS 'MODULE_PATHNAME', 'evoke_score_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_score_tokens(
    doc_tokens varchar[],
    query_tokens varchar[]
)
RETURNS float8
AS 'MODULE_PATHNAME', 'evoke_score_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_match_query_tokens(doc_tokens text[], query_text text)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_match_query_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_match_query_tokens(
    doc_tokens varchar[],
    query_text text
)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_match_query_tokens_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_score_scalar_text(
    doc_text text,
    query_tokens text[]
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT CASE
        WHEN COALESCE(array_ndims($2), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens(
            evoke_tokenize_text($1, true, NULL, false, false),
            $2
        )
    END
$$;

CREATE FUNCTION evoke_op_score_scalar_text(
    doc_text text,
    query_tokens varchar[]
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT CASE
        WHEN COALESCE(array_ndims($2), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens(
            evoke_tokenize_text($1, true, NULL, false, false),
            $2
        )
    END
$$;

CREATE FUNCTION evoke_op_score_scalar_text(
    doc_text varchar,
    query_tokens text[]
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT CASE
        WHEN COALESCE(array_ndims($2), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens(
            evoke_tokenize_text($1::text, true, NULL, false, false),
            $2
        )
    END
$$;

CREATE FUNCTION evoke_op_score_scalar_text(
    doc_text varchar,
    query_tokens varchar[]
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT CASE
        WHEN COALESCE(array_ndims($2), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens(
            evoke_tokenize_text($1::text, true, NULL, false, false),
            $2
        )
    END
$$;

CREATE OPERATOR <=> (
    LEFTARG = int4[],
    RIGHTARG = int4[],
    PROCEDURE = evoke_op_score_ids
);

CREATE OPERATOR <=> (
    LEFTARG = text[],
    RIGHTARG = text[],
    PROCEDURE = evoke_op_score_tokens
);

CREATE OPERATOR <=> (
    LEFTARG = text[],
    RIGHTARG = varchar[],
    PROCEDURE = evoke_op_score_tokens
);

CREATE OPERATOR <=> (
    LEFTARG = varchar[],
    RIGHTARG = text[],
    PROCEDURE = evoke_op_score_tokens
);

CREATE OPERATOR <=> (
    LEFTARG = varchar[],
    RIGHTARG = varchar[],
    PROCEDURE = evoke_op_score_tokens
);

CREATE OPERATOR <=> (
    LEFTARG = text,
    RIGHTARG = text[],
    PROCEDURE = evoke_op_score_scalar_text
);

CREATE OPERATOR <=> (
    LEFTARG = text,
    RIGHTARG = varchar[],
    PROCEDURE = evoke_op_score_scalar_text
);

CREATE OPERATOR <=> (
    LEFTARG = varchar,
    RIGHTARG = text[],
    PROCEDURE = evoke_op_score_scalar_text
);

CREATE OPERATOR <=> (
    LEFTARG = varchar,
    RIGHTARG = varchar[],
    PROCEDURE = evoke_op_score_scalar_text
);

CREATE OPERATOR @@ (
    LEFTARG = text[],
    RIGHTARG = text,
    PROCEDURE = evoke_op_match_query_tokens
);

CREATE OPERATOR @@ (
    LEFTARG = varchar[],
    RIGHTARG = text,
    PROCEDURE = evoke_op_match_query_tokens
);

CREATE OPERATOR CLASS evoke_text_array_ops
DEFAULT FOR TYPE text[] USING evoke AS
    OPERATOR 1 @@ (text[], text),
    OPERATOR 1 <=> (text[], text[]) FOR ORDER BY pg_catalog.float_ops,
    STORAGE text[];

CREATE OPERATOR CLASS evoke_varchar_array_ops
DEFAULT FOR TYPE varchar[] USING evoke AS
    OPERATOR 1 @@ (varchar[], text),
    OPERATOR 1 <=> (varchar[], text[]) FOR ORDER BY pg_catalog.float_ops,
    STORAGE varchar[];

CREATE OPERATOR CLASS evoke_text_ops
DEFAULT FOR TYPE text USING evoke AS
    OPERATOR 1 <=> (text, text[]) FOR ORDER BY pg_catalog.float_ops,
    STORAGE text;

CREATE OPERATOR CLASS evoke_varchar_ops
DEFAULT FOR TYPE varchar USING evoke AS
    OPERATOR 1 <=> (varchar, text[]) FOR ORDER BY pg_catalog.float_ops,
    STORAGE varchar;

CREATE OPERATOR CLASS evoke_int4_array_ops
DEFAULT FOR TYPE int4[] USING evoke AS
    OPERATOR 1 <=> (int4[], int4[]) FOR ORDER BY pg_catalog.float_ops,
    STORAGE int4[];

CREATE FUNCTION evoke_query_ids(
    index_name regclass,
    query_ids int4[],
    k int4 DEFAULT 10,
    weight_mask real[] DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
AS 'MODULE_PATHNAME', 'evoke_query_ids'
LANGUAGE C STABLE PARALLEL SAFE
ROWS 100;

CREATE FUNCTION evoke_query_tokens(
    index_name regclass,
    query_tokens text[],
    k int4 DEFAULT 10,
    weight_mask real[] DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
AS 'MODULE_PATHNAME', 'evoke_query_tokens'
LANGUAGE C STABLE PARALLEL SAFE
ROWS 100;

CREATE FUNCTION evoke_field_aware_query_tokens(
    index_name regclass,
    query_tokens text[],
    field_names text[],
    weights real[] DEFAULT NULL,
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
AS 'MODULE_PATHNAME', 'evoke_field_aware_query_tokens'
LANGUAGE C STABLE PARALLEL SAFE
ROWS 100;

CREATE FUNCTION evoke_field_aware_query(
    index_name regclass,
    query_text text,
    field_names text[],
    weights real[] DEFAULT NULL,
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_field_aware_query_tokens(
        $1,
        evoke_order_tokens($1, $2),
        $3,
        $4,
        $5
    )
$$;

CREATE FUNCTION evoke_query_bm25_internal(
    index_name regclass,
    query_text text,
    k int4 DEFAULT 10,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
AS 'MODULE_PATHNAME', 'evoke_query_bm25_internal'
LANGUAGE C STABLE PARALLEL SAFE
ROWS 100;

CREATE FUNCTION evoke_query_internal(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    k int4 DEFAULT 10,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL,
    allowed_doc_ids int4[] DEFAULT NULL,
    allowed_tids tid[] DEFAULT NULL,
    filters jsonb DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
DECLARE
    index_options jsonb;
    caller_name name;
    source_table regclass;
    source_row_security boolean := false;
    relation_kind "char";
    access_method name;
    sae_enabled boolean := false;
    field_aware boolean := false;
BEGIN
    SELECT
        relation.relkind,
        access_method.amname,
        indexed.indrelid::regclass,
        source.relrowsecurity
    INTO
        relation_kind,
        access_method,
        source_table,
        source_row_security
    FROM pg_catalog.pg_class AS relation
    LEFT JOIN pg_catalog.pg_am AS access_method
      ON access_method.oid = relation.relam
    LEFT JOIN pg_catalog.pg_index AS indexed
      ON indexed.indexrelid = relation.oid
    LEFT JOIN pg_catalog.pg_class AS source
      ON source.oid = indexed.indrelid
    WHERE relation.oid = index_name::oid;

    IF source_table IS NULL THEN
        RAISE EXCEPTION 'evoke indexed relation could not be resolved for %',
            index_name;
    END IF;

    caller_name := NULLIF(
        pg_catalog.current_setting('role', true),
        'none'
    );
    caller_name := COALESCE(caller_name, session_user::name);
    IF NOT pg_catalog.has_table_privilege(
        caller_name,
        source_table,
        'SELECT'
    ) THEN
        RAISE EXCEPTION
            'permission denied for evoke index %',
            index_name
            USING ERRCODE = 'insufficient_privilege',
                HINT = 'Grant SELECT on the indexed relation.';
    END IF;

    IF source_row_security THEN
        RAISE EXCEPTION
            'evoke search does not support row-level security on relation %',
            source_table
            USING ERRCODE = 'feature_not_supported',
                DETAIL =
                    'Filtering an already ranked top-k result cannot preserve '
                    || 'correct row-level-security ranking semantics.',
                HINT =
                    'Use a non-RLS search table or a security-filtered '
                    || 'materialized search relation.';
    END IF;

    IF relation_kind = 'I' THEN
        IF access_method IS DISTINCT FROM 'evoke' THEN
            RAISE EXCEPTION
                'index % must use evoke access method, got %',
                index_name,
                access_method;
        END IF;
        RAISE EXCEPTION
            'evoke partitioned parent index % is not queryable', index_name
            USING ERRCODE = 'feature_not_supported',
                DETAIL =
                    'Child indexes have independent corpus statistics, '
                    || 'document ordinals, and physical TID spaces.',
                HINT =
                    'Use one unpartitioned search table for globally ranked '
                    || 'results. Child indexes remain independently queryable.';
    END IF;

    index_options := evoke_index_options(index_name);
    sae_enabled := COALESCE(
        (index_options->>'sae_enabled')::boolean,
        false
    );
    field_aware := COALESCE(
        (index_options->>'field_aware')::boolean,
        false
    );

    IF field_names IS NULL AND field_weights IS NOT NULL THEN
        RAISE EXCEPTION 'field weights require field names';
    END IF;
    IF field_names IS NOT NULL AND NOT field_aware THEN
        RAISE EXCEPTION
            'field-aware search requires field_aware=true on index %',
            index_name;
    END IF;
    IF field_names IS NOT NULL AND (
        weight_mask IS NOT NULL
        OR lowercase IS NOT NULL
        OR stopwords IS NOT NULL
        OR stem_english IS NOT NULL
        OR fold_diacritics IS NOT NULL
    ) THEN
        RAISE EXCEPTION
            'field-aware evoke_query does not accept query overrides'
            USING HINT =
                'Configure text normalization on the index; field weights '
                || 'apply to the complete lexical and semantic field score.';
    END IF;

    IF sae_enabled THEN
        IF weight_mask IS NOT NULL
            OR lowercase IS NOT NULL
            OR stopwords IS NOT NULL
            OR stem_english IS NOT NULL
            OR fold_diacritics IS NOT NULL THEN
            RAISE EXCEPTION
                'sae-enabled evoke_query does not accept BM25 query overrides'
                USING HINT =
                    'Configure normalization and weighting in the model '
                    || 'manifest and index scoring profile.';
        END IF;

        IF NOT COALESCE(
            (index_options->>'semantic_configuration_ready')::boolean,
            false
        ) THEN
            RAISE EXCEPTION 'evoke SSR index % is not ready', index_name
                USING DETAIL = COALESCE(
                    index_options->>'semantic_configuration_error',
                    'default SSR model configuration is unavailable'
                ),
                HINT =
                    'Install the bundled milestone checkout or set model_path '
                    || 'on the index, then inspect evoke_index_status.';
        END IF;

        RETURN QUERY
        SELECT
            model_hit.ctid,
            model_hit.doc_ord AS doc_id,
            model_hit.score::real
        FROM evoke_query_semantic_internal(
            index_name,
            query_text,
            k,
            field_names,
            field_weights,
            allowed_doc_ids,
            allowed_tids,
            filters
        ) AS model_hit
        ORDER BY model_hit.rank;
        RETURN;
    END IF;

    IF allowed_doc_ids IS NOT NULL OR allowed_tids IS NOT NULL OR
        filters IS NOT NULL THEN
        RAISE EXCEPTION
            'filtered top-k currently requires sae=true'
            USING ERRCODE = 'feature_not_supported',
                HINT =
                    'Use an SSR-enabled unified index (sae=true) for '
                    || 'predicate-defined ranking.';
    END IF;

    IF field_names IS NOT NULL THEN
        RETURN QUERY
        SELECT *
        FROM evoke_field_aware_query(
            index_name,
            query_text,
            field_names,
            field_weights,
            k
        );
        RETURN;
    END IF;

    RETURN QUERY
    SELECT *
    FROM evoke_query_bm25_internal(
        index_name,
        query_text,
        k,
        weight_mask,
        lowercase,
        stopwords,
        stem_english,
        fold_diacritics
    );
END;
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    k int4,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        NULL,
        NULL,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8,
        NULL,
        NULL,
        NULL
    )
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    k int4
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        $3,
        $4,
        $5,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
    )
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    allowed_tids tid[],
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        NULL,
        NULL,
        $4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        CASE
            WHEN $3 IS NULL THEN ARRAY[]::int4[]
            ELSE NULL::int4[]
        END,
        $3,
        NULL
    )
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    allowed_tids tid[],
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        $3,
        $4,
        $6,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        CASE
            WHEN $5 IS NULL THEN ARRAY[]::int4[]
            ELSE NULL::int4[]
        END,
        $5,
        NULL
    )
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    filters jsonb,
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        NULL,
        NULL,
        $4,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        COALESCE($3, 'null'::jsonb)
    )
$$;

CREATE FUNCTION evoke_query(
    index_name regclass,
    query_text text,
    field_names text[],
    field_weights real[],
    filters jsonb,
    k int4 DEFAULT 10
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL SECURITY DEFINER VOLATILE PARALLEL UNSAFE
ROWS 100
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_query_internal(
        $1,
        $2,
        $3,
        $4,
        $6,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL,
        COALESCE($5, 'null'::jsonb)
    )
$$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    int4,
    real[],
    boolean,
    text[],
    boolean,
    boolean
) IS
$evoke_comment$
Search an evoke index with one string-first API. Ordinary indexes use exact
BM25 retrieval. Indexes created with sae=true use the model runtime and
unified-posting route, whose detailed result is projected to evoke_result_hit.
SSR normalization and weighting belong to the model manifest and scoring
profile; BM25-only query overrides are rejected instead of ignored. Semantic
mode never implicitly changes an index type: sae=true is required.
$evoke_comment$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    jsonb,
    int4
) IS
$evoke_comment$
Search an SSR-enabled unified index with structured predicate-defined top-k.
Filters are ANDed by column and support eq, in, overlap, ilike, ilike_any, and
range. An eligible same-root scope may rank a compatible published baseline;
every returned row is rechecked under the current MVCC snapshot, while
post-baseline matches may be temporarily absent. Otherwise Evoke uses a bounded
ranked-prefix probe or the current PostgreSQL predicate resolver. No separate
filter generation or worker lifecycle is created.
$evoke_comment$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    text[],
    real[],
    jsonb,
    int4
) IS
$evoke_comment$
Apply structured predicate-defined top-k to an SSR-enabled field-aware index.
Field weights apply to the complete lexical and semantic field contribution;
scope, ranked-prefix, or current SQL resolution defines the filtered candidate
universe before top-k selection. Serving-scope results are current-row checked
but may omit post-baseline matches.
$evoke_comment$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    tid[],
    int4
) IS
$evoke_comment$
Search an SSR-enabled unified index while restricting the ranking competition
to the supplied heap TIDs. Membership is a hard boundary, but ranking still uses
the selected exact or bounded-approximate index route; a stale accelerator may
omit an allowed post-baseline row. Build the TID set in the same statement and
snapshot;
do not persist heap TIDs across table rewrites, VACUUM FULL, or CLUSTER.
$evoke_comment$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    text[],
    real[],
    tid[],
    int4
) IS
$evoke_comment$
Search selected fields in an SSR-enabled field-aware index while ranking only
the supplied heap TIDs. Field weights apply to the complete lexical and
semantic contribution before predicate-defined subset top-k selection.
$evoke_comment$;

COMMENT ON FUNCTION evoke_query(
    regclass,
    text,
    text[],
    real[],
    int4
) IS
$evoke_comment$
Search selected fields through the same evoke product API. For field-aware
indexes each weight multiplies the complete field contribution after lexical
BM25 and SSR semantic postings are accumulated. BM25-only and SSR-enabled
indexes therefore share the same field selection and weighting semantics.
$evoke_comment$;



CREATE FUNCTION evoke_prepared_query(
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS evoke_result_prepared_query
AS 'MODULE_PATHNAME', 'evoke_prepare_query_resolved'
LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION evoke_fusion_weighted_query(
    prepared_query evoke_result_prepared_query,
    weight real DEFAULT 1.0
)
RETURNS evoke_result_fusion_weighted_query
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT ROW($1, COALESCE($2, 1.0))::evoke_result_fusion_weighted_query
$$;

CREATE FUNCTION evoke_fusion_weighted_query(
    index_name regclass,
    query_text text,
    weight real DEFAULT 1.0,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS evoke_result_fusion_weighted_query
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_fusion_weighted_query(
        ROW(
            $1,
            $2,
            $4,
            $5,
            $6,
            $7
        )::evoke_result_prepared_query,
        $3
    )
$$;

CREATE FUNCTION evoke_fusion_weighted_queries(
    index_names regclass[],
    query_text text,
    weights real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS evoke_result_fusion_weighted_query[]
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT COALESCE(
        array_agg(
            evoke_fusion_weighted_query(
                index_names[i],
                query_text,
                COALESCE(weights[i], 1.0),
                lowercase,
                stopwords,
                stem_english,
                fold_diacritics
            )
            ORDER BY i
        ),
        ARRAY[]::evoke_result_fusion_weighted_query[]
    )
    FROM generate_subscripts(COALESCE(index_names, ARRAY[]::regclass[]), 1) AS i
$$;

CREATE FUNCTION evoke_fusion_field_query(
    field_name text,
    weighted_query evoke_result_fusion_weighted_query
)
RETURNS evoke_result_fusion_field_query
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT ROW($1, $2)::evoke_result_fusion_field_query
$$;

CREATE FUNCTION evoke_fusion_field_query(
    field_name text,
    prepared_query evoke_result_prepared_query,
    weight real DEFAULT 1.0
)
RETURNS evoke_result_fusion_field_query
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_fusion_field_query(
        $1,
        evoke_fusion_weighted_query($2, $3)
    )
$$;

CREATE FUNCTION evoke_fusion_field_query(
    field_name text,
    index_name regclass,
    query_text text,
    weight real DEFAULT 1.0,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS evoke_result_fusion_field_query
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_fusion_field_query(
        $1,
        evoke_fusion_weighted_query(
            $2,
            $3,
            $4,
            $5,
            $6,
            $7,
            $8
        )
    )
$$;

CREATE FUNCTION evoke_fusion_field_queries(
    field_names text[],
    index_names regclass[],
    query_text text,
    weights real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS evoke_result_fusion_field_query[]
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT COALESCE(
        array_agg(
            evoke_fusion_field_query(
                COALESCE(field_names[i], format('field_%s', i)),
                index_names[i],
                query_text,
                COALESCE(weights[i], 1.0),
                lowercase,
                stopwords,
                stem_english,
                fold_diacritics
            )
            ORDER BY i
        ),
        ARRAY[]::evoke_result_fusion_field_query[]
    )
    FROM generate_subscripts(COALESCE(index_names, ARRAY[]::regclass[]), 1) AS i
$$;

CREATE FUNCTION evoke_order_tokens(
    prepared_query evoke_result_prepared_query
)
RETURNS text[]
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_tokenize_text(
        ($1).query_text,
        ($1).lowercase,
        ($1).stopwords,
        ($1).stem_english,
        ($1).fold_diacritics
    )
$$;

CREATE FUNCTION evoke_order_tokens(
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS text[]
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_order_tokens(
        evoke_prepared_query($1, $2, $3, $4, $5, $6)
    )
$$;
CREATE FUNCTION evoke_fusion(
    left_hits evoke_result_hit[],
    left_weight real,
    right_hits evoke_result_hit[],
    right_weight real,
    k int4 DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    WITH left_hits AS (
        SELECT
            (h).ctid AS ctid,
            (h).doc_id AS doc_id,
            (h).score::float8 * COALESCE($2, 1.0)::float8 AS score
        FROM unnest(COALESCE($1, ARRAY[]::evoke_result_hit[])) AS h
    ),
    right_hits AS (
        SELECT
            (h).ctid AS ctid,
            (h).doc_id AS doc_id,
            (h).score::float8 * COALESCE($4, 1.0)::float8 AS score
        FROM unnest(COALESCE($3, ARRAY[]::evoke_result_hit[])) AS h
    ),
    fused AS (
        SELECT
            ctid,
            MIN(doc_id) AS doc_id,
            SUM(score) AS score
        FROM (
            SELECT * FROM left_hits
            UNION ALL
            SELECT * FROM right_hits
        ) s
        GROUP BY ctid
    )
    SELECT ctid, doc_id, score::real
    FROM fused
    ORDER BY score DESC, ctid::text
    LIMIT COALESCE(
        $5,
        GREATEST(
            COALESCE(cardinality($1), 0),
            COALESCE(cardinality($3), 0)
        )
    )
$$;

CREATE FUNCTION evoke_fusion_query(
    field_names text[],
    index_names regclass[],
    query_text text,
    weights real[] DEFAULT NULL,
    k int4 DEFAULT 10,
    candidate_k int4 DEFAULT NULL,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_fusion_query_fields(
        evoke_fusion_field_queries(
            $1,
            $2,
            $3,
            $4,
            $8,
            $9,
            $10,
            $11
        ),
        $5,
        $6,
        $7
    )
$$;

CREATE FUNCTION evoke_fusion_query(
    index_names regclass[],
    query_text text,
    weights real[] DEFAULT NULL,
    k int4 DEFAULT 10,
    candidate_k int4 DEFAULT NULL,
    weight_mask real[] DEFAULT NULL,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_fusion_query_weighted(
        evoke_fusion_weighted_queries(
            $1,
            $2,
            $3,
            $7,
            $8,
            $9,
            $10
        ),
        $4,
        $5,
        $6
    )
$$;

CREATE FUNCTION evoke_fusion_query_fields(
    field_queries evoke_result_fusion_field_query[],
    k int4 DEFAULT 10,
    candidate_k int4 DEFAULT NULL,
    weight_mask real[] DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_fusion_query_weighted(
        COALESCE(
            ARRAY(
                SELECT (q).weighted_query
                FROM unnest(
                    COALESCE($1, ARRAY[]::evoke_result_fusion_field_query[])
                ) AS q
            ),
            ARRAY[]::evoke_result_fusion_weighted_query[]
        ),
        $2,
        $3,
        $4
    )
$$;

CREATE FUNCTION evoke_fusion_query_weighted(
    weighted_queries evoke_result_fusion_weighted_query[],
    k int4 DEFAULT 10,
    candidate_k int4 DEFAULT NULL,
    weight_mask real[] DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    WITH queries AS (
        SELECT
            (q).prepared_query AS prepared_query,
            COALESCE((q).weight, 1.0)::float8 AS weight
        FROM unnest(COALESCE($1, ARRAY[]::evoke_result_fusion_weighted_query[])) AS q
    ),
    hits AS (
        SELECT
            h.ctid,
            h.doc_id,
            h.score::float8 * q.weight AS score
        FROM queries q
        CROSS JOIN LATERAL evoke_query(
            (q.prepared_query).index_name,
            (q.prepared_query).query_text,
            COALESCE($3, $2),
            $4,
            (q.prepared_query).lowercase,
            (q.prepared_query).stopwords,
            (q.prepared_query).stem_english,
            (q.prepared_query).fold_diacritics
        ) AS h
    ),
    fused AS (
        SELECT
            ctid,
            MIN(doc_id) AS doc_id,
            SUM(score) AS score
        FROM hits
        GROUP BY ctid
    )
    SELECT ctid, doc_id, score::real
    FROM fused
    ORDER BY score DESC, ctid::text
    LIMIT $2
$$;

CREATE FUNCTION evoke_fusion_query_weighted(
    weighted_queries evoke_result_fusion_weighted_query[],
    query_text text,
    k int4 DEFAULT 10,
    candidate_k int4 DEFAULT NULL,
    weight_mask real[] DEFAULT NULL
)
RETURNS SETOF evoke_result_hit
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT *
    FROM evoke_fusion_query_weighted(
        ARRAY(
            SELECT evoke_fusion_weighted_query(
                ((q).prepared_query).index_name,
                $2,
                (q).weight,
                ((q).prepared_query).lowercase,
                ((q).prepared_query).stopwords,
                ((q).prepared_query).stem_english,
                ((q).prepared_query).fold_diacritics
            )
            FROM unnest(
                COALESCE($1, ARRAY[]::evoke_result_fusion_weighted_query[])
            ) AS q
        ),
        $3,
        $4,
        $5
    )
$$;

CREATE FUNCTION evoke_hybrid_candidate(
    source_name text,
    ctid tid,
    raw_value real,
    source_rank int4,
    weight real DEFAULT 1.0,
    normalizer text DEFAULT 'identity',
    direction text DEFAULT 'higher_is_better'
)
RETURNS evoke_result_hybrid_candidate
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    normalized_normalizer text := lower(COALESCE($6, 'identity'));
    normalized_direction text := lower(COALESCE($7, 'higher_is_better'));
BEGIN
    IF normalized_normalizer NOT IN (
        'identity',
        'negative_distance',
        'inverse_distance',
        'minmax',
        'zscore',
        'rank'
    ) THEN
        RAISE EXCEPTION 'unsupported evoke hybrid normalizer: %',
            $6
            USING HINT = 'Use identity, negative_distance, inverse_distance, minmax, zscore, or rank.';
    END IF;
    IF normalized_direction NOT IN ('higher_is_better', 'lower_is_better') THEN
        RAISE EXCEPTION 'unsupported evoke hybrid direction: %',
            $7
            USING HINT = 'Use higher_is_better or lower_is_better.';
    END IF;
    RETURN ROW(
        COALESCE(NULLIF($1, ''), 'source'),
        $2,
        $3,
        $4,
        COALESCE($5, 1.0),
        normalized_normalizer,
        normalized_direction
    )::evoke_result_hybrid_candidate;
END;
$$;

CREATE FUNCTION evoke_hybrid_bm25_candidate(
    source_name text,
    ctid tid,
    score real,
    source_rank int4,
    weight real DEFAULT 1.0,
    normalizer text DEFAULT 'identity'
)
RETURNS evoke_result_hybrid_candidate
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_hybrid_candidate(
        $1,
        $2,
        $3,
        $4,
        $5,
        COALESCE($6, 'identity'),
        'higher_is_better'
    )
$$;

CREATE FUNCTION evoke_hybrid_vector_candidate(
    source_name text,
    ctid tid,
    distance real,
    source_rank int4,
    weight real DEFAULT 1.0,
    normalizer text DEFAULT 'negative_distance'
)
RETURNS evoke_result_hybrid_candidate
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_hybrid_candidate(
        $1,
        $2,
        $3,
        $4,
        $5,
        COALESCE($6, 'negative_distance'),
        'lower_is_better'
    )
$$;

CREATE FUNCTION evoke_hybrid_bm25_candidates(
    source_name text,
    index_name regclass,
    query_text text,
    weight real DEFAULT 1.0,
    candidate_k int4 DEFAULT 100,
    normalizer text DEFAULT 'identity',
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS SETOF evoke_result_hybrid_candidate
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_hybrid_bm25_candidate(
        $1,
        h.ctid,
        h.score,
        row_number() OVER (ORDER BY h.score DESC, h.ctid::text)::int4,
        $4,
        $6
    )
    FROM evoke_query(
        $2,
        $3,
        GREATEST(COALESCE($5, 100), 0),
        NULL,
        $7,
        $8,
        $9,
        $10
    ) AS h
    ORDER BY h.score DESC, h.ctid::text
$$;

CREATE FUNCTION evoke_hybrid_fuse_candidates(
    candidates evoke_result_hybrid_candidate[],
    k int4 DEFAULT 10,
    fusion text DEFAULT 'rrf',
    rrf_k real DEFAULT 60.0,
    epsilon real DEFAULT 0.000001
)
RETURNS SETOF evoke_result_hybrid_hit
AS 'MODULE_PATHNAME', 'evoke_hybrid_fuse_candidates'
LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION evoke_onnxruntime_probe()
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_onnxruntime_probe'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_onnxruntime_probe() IS
'Probe whether this backend can dynamically load the ONNX Runtime C API.';

CREATE FUNCTION evoke_onnxruntime_build_info()
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_onnxruntime_build_info'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

COMMENT ON FUNCTION evoke_onnxruntime_build_info() IS
'Report whether evoke was compiled with ONNX Runtime C API build support.';

CREATE FUNCTION evoke_runtime_service_status()
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_status'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_status() IS
'Inspect the shared evoke SSR runtime worker, model ownership, bounded FIFO request queue, and recovery counters.';

CREATE FUNCTION evoke_runtime_service_query_atoms(
    model_path text,
    query_text text
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_query_atoms'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_query_atoms(text, text) IS
'Encode text through the shared evoke runtime used by semantic-enabled indexes.';

CREATE FUNCTION evoke_runtime_service_query_atoms(
    model_path text,
    runtime_precision text,
    query_text text
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_query_atoms'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_query_atoms(text, text, text) IS
'Encode text through the shared evoke runtime with an explicit precision contract.';

CREATE FUNCTION evoke_runtime_service_query_atoms_batch(
    model_path text,
    input_texts text[]
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_query_atoms_batch'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_query_atoms_batch(text, text[]) IS
'Encode an explicit query batch through the shared runtime for diagnostics.';

CREATE FUNCTION evoke_runtime_service_query_atoms_batch(
    model_path text,
    runtime_precision text,
    input_texts text[]
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_query_atoms_batch'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_query_atoms_batch(
    text, text, text[]
) IS
'Encode an explicit query batch through the shared runtime with an explicit precision contract.';

CREATE FUNCTION evoke_runtime_service_document_atoms_batch(
    model_path text,
    input_texts text[]
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_document_atoms_batch'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_document_atoms_batch(text, text[]) IS
'Encode an explicit document batch through the shared runtime for index build and rebuild operations.';

CREATE FUNCTION evoke_runtime_service_document_atoms_batch(
    model_path text,
    runtime_precision text,
    input_texts text[]
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_runtime_service_document_atoms_batch'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_runtime_service_document_atoms_batch(
    text, text, text[]
) IS
'Encode an explicit document batch through the shared runtime with an explicit precision contract.';

CREATE FUNCTION evoke_runtime_service_atoms_batch_internal(
    model_path text,
    runtime_precision text,
    request_mode text,
    input_texts text[]
)
RETURNS jsonb
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
BEGIN
    IF request_mode = 'document' THEN
        RETURN evoke_runtime_service_document_atoms_batch(
            model_path,
            runtime_precision,
            input_texts
        );
    ELSIF request_mode = 'query' THEN
        RETURN evoke_runtime_service_query_atoms_batch(
            model_path,
            runtime_precision,
            input_texts
        );
    END IF;

    RAISE EXCEPTION
        'unsupported evoke runtime request mode: %',
        COALESCE(request_mode, '<null>')
        USING ERRCODE = 'invalid_parameter_value',
            HINT = 'Use request_mode=document or request_mode=query.';
END;
$$;

COMMENT ON FUNCTION evoke_runtime_service_atoms_batch_internal(
    text, text, text, text[]
) IS
'Internal shared batch encoder entrypoint with an explicit precision contract.';

CREATE FUNCTION evoke_runtime_service_atoms_batch_internal(
    model_path text,
    request_mode text,
    input_texts text[]
)
RETURNS jsonb
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
BEGIN
    RETURN evoke_runtime_service_atoms_batch_internal(
        model_path,
        'fp16',
        request_mode,
        input_texts
    );
END;
$$;

COMMENT ON FUNCTION evoke_runtime_service_atoms_batch_internal(
    text, text, text[]
) IS
'Internal shared batch encoder entrypoint using the default fp16 precision.';

CREATE FUNCTION evoke_runtime_loader_status()
RETURNS jsonb
LANGUAGE SQL VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
    WITH probe AS (
        SELECT evoke_onnxruntime_probe() AS result
    ),
    build AS (
        SELECT evoke_onnxruntime_build_info() AS result
    )
    SELECT jsonb_build_object(
        'onnxruntime_probe', probe.result,
        'onnxruntime_available', probe.result LIKE 'available:%',
        'onnxruntime_build', build.result,
        'onnxruntime_build_enabled', build.result LIKE 'enabled:%',
        'onnxruntime_api_version',
            CASE
                WHEN build.result LIKE 'enabled:%'
                    THEN substring(build.result FROM 'api=([0-9]+)')
                ELSE NULL
            END,
        'onnxruntime_library',
            CASE
                WHEN probe.result LIKE 'available:%:version=%'
                    THEN substring(probe.result FROM '^available:(.*):version=')
                ELSE NULL
            END,
        'onnxruntime_version',
            CASE
                WHEN probe.result LIKE 'available:%:version=%'
                    THEN substring(probe.result FROM ':version=(.*)$')
                ELSE NULL
            END,
        'onnxruntime_error',
            CASE
                WHEN probe.result LIKE 'missing:%'
                    THEN substring(probe.result FROM '^missing:(.*)$')
                ELSE NULL
            END,
        'runtime_execution_linked', build.result LIKE 'enabled:%',
        'runtime_execution_error',
            CASE
                WHEN build.result LIKE 'enabled:%' THEN NULL
                ELSE 'ONNX Runtime query execution is not linked in this evoke build'
            END,
        'runtime_service', evoke_runtime_service_status()
    )
    FROM probe, build
$$;

COMMENT ON FUNCTION evoke_runtime_loader_status() IS
'Inspect optional runtime loader availability for semantic-enabled query execution.';

CREATE FUNCTION evoke_op_match_prepared_query(
    doc_tokens text[],
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_match_prepared_query_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_op_match_prepared_query(
    doc_tokens varchar[],
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_match_prepared_query_op'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION evoke_match_prepared_query(
    doc_tokens text[],
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_op_match_prepared_query(
        $1,
        $2
    )
$$;

CREATE FUNCTION evoke_match_prepared_query(
    doc_tokens varchar[],
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_op_match_prepared_query(
        $1,
        $2
    )
$$;

CREATE FUNCTION evoke_match_prepared_query(
    doc_text text,
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        evoke_tokenize_text(
            $1,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE FUNCTION evoke_match_prepared_query(
    doc_text varchar,
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        evoke_tokenize_text(
            $1::text,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE FUNCTION evoke_match_query(
    doc_tokens text[],
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS boolean
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_match_query(
    doc_tokens varchar[],
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS boolean
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_match_query(
    doc_text text,
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS boolean
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_match_query(
    doc_text varchar,
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS boolean
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_op_match_prepared_query_scalar(
    doc_text text,
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        evoke_tokenize_text(
            $1,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE FUNCTION evoke_op_match_prepared_query_scalar(
    doc_text varchar,
    prepared_query evoke_result_prepared_query
)
RETURNS boolean
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_match_prepared_query(
        evoke_tokenize_text(
            $1::text,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE OPERATOR @@@ (
    LEFTARG = text[],
    RIGHTARG = evoke_result_prepared_query,
    PROCEDURE = evoke_op_match_prepared_query
);

CREATE OPERATOR @@@ (
    LEFTARG = varchar[],
    RIGHTARG = evoke_result_prepared_query,
    PROCEDURE = evoke_op_match_prepared_query
);

CREATE OPERATOR @@@ (
    LEFTARG = text,
    RIGHTARG = evoke_result_prepared_query,
    PROCEDURE = evoke_op_match_prepared_query_scalar
);

CREATE OPERATOR @@@ (
    LEFTARG = varchar,
    RIGHTARG = evoke_result_prepared_query,
    PROCEDURE = evoke_op_match_prepared_query_scalar
);

ALTER OPERATOR FAMILY evoke_text_array_ops USING evoke
    ADD OPERATOR 1 @@@ (text[], evoke_result_prepared_query);

ALTER OPERATOR FAMILY evoke_varchar_array_ops USING evoke
    ADD OPERATOR 1 @@@ (varchar[], evoke_result_prepared_query);

ALTER OPERATOR FAMILY evoke_text_ops USING evoke
    ADD OPERATOR 1 @@@ (text, evoke_result_prepared_query);

ALTER OPERATOR FAMILY evoke_varchar_ops USING evoke
    ADD OPERATOR 1 @@@ (varchar, evoke_result_prepared_query);

CREATE FUNCTION evoke_score_prepared_query(
    doc_tokens text[],
    prepared_query evoke_result_prepared_query
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    WITH query_tokens AS (
        SELECT evoke_tokenize_text(
            ($2).query_text,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ) AS tokens
    )
    SELECT CASE
        WHEN COALESCE(array_ndims($1), 0) <> 1 THEN 0::float8
        WHEN COALESCE(array_ndims(tokens), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens($1, tokens)
    END
    FROM query_tokens
$$;

CREATE FUNCTION evoke_score_prepared_query(
    doc_tokens varchar[],
    prepared_query evoke_result_prepared_query
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    WITH query_tokens AS (
        SELECT evoke_tokenize_text(
            ($2).query_text,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ) AS tokens
    )
    SELECT CASE
        WHEN COALESCE(array_ndims($1), 0) <> 1 THEN 0::float8
        WHEN COALESCE(array_ndims(tokens), 0) <> 1 THEN 0::float8
        ELSE -evoke_op_score_tokens($1, tokens)
    END
    FROM query_tokens
$$;

CREATE FUNCTION evoke_score_prepared_query(
    doc_text text,
    prepared_query evoke_result_prepared_query
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        evoke_tokenize_text(
            $1,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE FUNCTION evoke_score_prepared_query(
    doc_text varchar,
    prepared_query evoke_result_prepared_query
)
RETURNS float8
LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        evoke_tokenize_text(
            $1::text,
            ($2).lowercase,
            ($2).stopwords,
            ($2).stem_english,
            ($2).fold_diacritics
        ),
        $2
    )
$$;

CREATE FUNCTION evoke_score_query(
    doc_tokens text[],
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS float8
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_score_query(
    doc_tokens varchar[],
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS float8
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_score_query(
    doc_text text,
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS float8
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_score_query(
    doc_text varchar,
    index_name regclass,
    query_text text,
    lowercase boolean DEFAULT NULL,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT NULL,
    fold_diacritics boolean DEFAULT NULL
)
RETURNS float8
LANGUAGE SQL STABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_score_prepared_query(
        $1,
        evoke_prepared_query($2, $3, $4, $5, $6, $7)
    )
$$;

CREATE FUNCTION evoke_normalize_tokens(
    tokens text[],
    lowercase boolean DEFAULT true,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text[]
AS 'MODULE_PATHNAME', 'evoke_normalize_tokens_sql'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_normalize_tokens(
    tokens varchar[],
    lowercase boolean DEFAULT true,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text[]
AS 'MODULE_PATHNAME', 'evoke_normalize_tokens_sql'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_tokenize_text(
    input_text text,
    lowercase boolean DEFAULT true,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text[]
AS 'MODULE_PATHNAME', 'evoke_tokenize_text_sql'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_highlight(
    doc_tokens text[],
    query_text text,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_highlight_tokens_cfg'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_highlight(
    doc_tokens varchar[],
    query_text text,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_highlight_tokens_cfg'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_highlight(
    doc_text text,
    query_text text,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_highlight(
        evoke_tokenize_text($1, $5, $6, $7, $8),
        $2,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8
    )
$$;

CREATE FUNCTION evoke_highlight(
    doc_text varchar,
    query_text text,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_highlight(
        evoke_tokenize_text($1::text, $5, $6, $7, $8),
        $2,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8
    )
$$;


CREATE FUNCTION evoke_snippet(
    doc_tokens text[],
    query_text text,
    max_tokens int4 DEFAULT 24,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_snippet_tokens_cfg'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_snippet(
    doc_tokens varchar[],
    query_text text,
    max_tokens int4 DEFAULT 24,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_snippet_tokens_cfg'
LANGUAGE C IMMUTABLE PARALLEL SAFE;

CREATE FUNCTION evoke_snippet(
    doc_text text,
    query_text text,
    max_tokens int4 DEFAULT 24,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_snippet(
        evoke_tokenize_text($1, $6, $7, $8, $9),
        $2,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8,
        $9
    )
$$;

CREATE FUNCTION evoke_snippet(
    doc_text varchar,
    query_text text,
    max_tokens int4 DEFAULT 24,
    start_tag text DEFAULT '<b>',
    end_tag text DEFAULT '</b>',
    lowercase boolean DEFAULT false,
    stopwords text[] DEFAULT NULL,
    stem_english boolean DEFAULT false,
    fold_diacritics boolean DEFAULT false
)
RETURNS text
LANGUAGE SQL IMMUTABLE PARALLEL SAFE
SET search_path FROM CURRENT
AS $$
    SELECT evoke_snippet(
        evoke_tokenize_text($1::text, $6, $7, $8, $9),
        $2,
        $3,
        $4,
        $5,
        $6,
        $7,
        $8,
        $9
    )
$$;

CREATE FUNCTION evoke_index_checkout_manifest_internal(
    checkout_path text
)
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_index_checkout_manifest_internal_c'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_index_checkout_manifest_internal(text) IS
'Internal manifest reader for the checkout owned by a semantic-enabled evoke index.';

CREATE FUNCTION evoke_checkout_manifest_signature_internal(
    manifest jsonb
)
RETURNS text
AS 'MODULE_PATHNAME', 'evoke_checkout_manifest_signature_internal_c'
LANGUAGE C IMMUTABLE PARALLEL SAFE STRICT;

COMMENT ON FUNCTION evoke_checkout_manifest_signature_internal(jsonb) IS
'Internal canonical SHA-256 identity for one validated checkout manifest.';

CREATE FUNCTION evoke_index_checkout_validate_internal(
    checkout_path text
)
RETURNS boolean
AS 'MODULE_PATHNAME', 'evoke_index_checkout_validate_internal_c'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_index_checkout_validate_internal(text) IS
'Internal deep SHA-256 audit for every artifact in a semantic-enabled evoke checkout.';

CREATE FUNCTION evoke_index_runtime_plan_internal(
    checkout_path text,
    runtime_overrides jsonb DEFAULT '{}'::jsonb
)
RETURNS jsonb
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    manifest jsonb;
    runtime text;
    encoder_relpath text;
    declared_address text;
    runtime_address text;
    runtime_address_type text;
    runtime_error text;
    runtime_service jsonb;
    checkout_signature text;
BEGIN
    IF NULLIF(checkout_path, '') IS NULL THEN
        RAISE EXCEPTION 'evoke checkout path must not be empty';
    END IF;
    runtime_overrides := COALESCE(runtime_overrides, '{}'::jsonb);
    IF jsonb_typeof(runtime_overrides) <> 'object' THEN
        RAISE EXCEPTION 'evoke runtime overrides must be a JSON object';
    END IF;

    manifest := evoke_index_checkout_manifest_internal(checkout_path);
    checkout_signature :=
        evoke_checkout_manifest_signature_internal(manifest);
    IF manifest ? 'runtime_parameters' THEN
        RAISE EXCEPTION
            'evoke runtime_parameters is not a model manifest field'
            USING HINT =
                'Move batching and provider settings to runtime server or '
                || 'PostgreSQL runtime configuration, then republish the '
                || 'checkout.';
    END IF;

    runtime := manifest->>'runtime';
    encoder_relpath := manifest#>>'{artifacts,query_encoder,path}';
    declared_address := COALESCE(
        manifest->>'runtime_address',
        manifest#>>'{runtime_endpoint,url}',
        manifest#>>'{runtime_connection,address}'
    );
    runtime_address := COALESCE(
        declared_address,
        CASE
            WHEN NULLIF(encoder_relpath, '') IS NOT NULL
                THEN checkout_path || '/' || encoder_relpath
            ELSE checkout_path
        END
    );
    runtime_address_type := CASE
        WHEN declared_address LIKE 'http://%'
            OR declared_address LIKE 'https://%' THEN 'remote_endpoint'
        WHEN NULLIF(encoder_relpath, '') IS NOT NULL
            THEN 'server_local_artifact'
        ELSE 'model_checkout'
    END;
    runtime_service := evoke_runtime_service_status();
    runtime_error := CASE
        WHEN runtime IS DISTINCT FROM 'onnxruntime' THEN
            format('unsupported evoke runtime: %s', runtime)
        WHEN runtime_address_type <> 'server_local_artifact' THEN
            'ONNX Runtime requires a server-local encoder artifact'
        WHEN evoke_onnxruntime_build_info() NOT LIKE 'enabled:%' THEN
            'ONNX Runtime execution is not linked in this evoke build'
        WHEN COALESCE(
            (runtime_service->>'ready_for_text_encoding')::boolean,
            false
        ) THEN NULL
        ELSE 'evoke runtime service is not ready for text encoding'
    END;

    RETURN jsonb_build_object(
        'model_id', manifest->>'model_id',
        'model_path', checkout_path,
        'manifest', manifest,
        'checkout_signature', checkout_signature,
        'runtime', runtime,
        'runtime_abi', manifest->>'runtime_abi',
        'model_format', manifest->>'model_format',
        'runtime_provider', 'evoke_runtime_service',
        'runtime_service', runtime_service,
        'runtime_address', runtime_address,
        'runtime_address_type', runtime_address_type,
        'runtime_options', runtime_overrides,
        'runtime_io', manifest->'runtime_io',
        'runtime_output', manifest->'runtime_output',
        'atom_space', manifest#>>'{index_compatibility,atom_space}',
        'scoring_profile',
            manifest#>>'{index_compatibility,scoring}',
        'can_execute_in_this_build', runtime_error IS NULL,
        'runtime_error', runtime_error
    );
END;
$$;

COMMENT ON FUNCTION evoke_index_runtime_plan_internal(text, jsonb) IS
'Internal runtime plan derived directly from an SSR-enabled index checkout.';

CREATE FUNCTION evoke_index_options_internal(
    index_name regclass
)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    access_method text;
    index_owner_is_superuser boolean := false;
    reloptions text[];
    option_text text;
    option_key text;
    option_value text;
    option_map jsonb := '{}'::jsonb;
    sae_explicit boolean := false;
    sae_enabled boolean := false;
    configured_model_path text;
    environment_model_path text;
    packaged_model_path text;
    effective_model_path text;
    model_path_source text := 'none';
    import_plan jsonb;
    semantic_configuration_error text;
    model_id text;
    atom_space text;
    scoring_profile text;
    consistency text;
    runtime_precision text := 'fp16';
    runtime_precision_source text := 'default';
    semantic_impact_precision text := 'f32';
    semantic_alpha_mass double precision := 1.0;
    runtime_signature text;
BEGIN
    SELECT access.amname, relation.reloptions, owner.rolsuper
    INTO access_method, reloptions, index_owner_is_superuser
    FROM pg_catalog.pg_class AS relation
    JOIN pg_catalog.pg_am AS access
      ON access.oid = relation.relam
    JOIN pg_catalog.pg_authid AS owner
      ON owner.oid = relation.relowner
    WHERE relation.oid = index_name::oid;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'evoke index % does not exist', index_name;
    END IF;
    IF access_method IS DISTINCT FROM 'evoke' THEN
        RAISE EXCEPTION
            'index % must use evoke access method, got %',
            index_name,
            access_method;
    END IF;

    FOREACH option_text IN ARRAY COALESCE(reloptions, ARRAY[]::text[])
    LOOP
        IF position('=' IN option_text) = 0 THEN
            CONTINUE;
        END IF;
        option_key := split_part(option_text, '=', 1);
        option_value := substring(
            option_text FROM position('=' IN option_text) + 1
        );
        option_map := option_map || jsonb_build_object(
            option_key,
            option_value
        );
    END LOOP;

    sae_explicit := option_map ? 'sae';
    sae_enabled := COALESCE((option_map->>'sae')::boolean, false);
    runtime_precision := lower(
        COALESCE(NULLIF(option_map->>'runtime_precision', ''), 'fp16')
    );
    runtime_precision_source := CASE
        WHEN option_map ? 'runtime_precision' THEN 'index'
        ELSE 'default'
    END;
    semantic_impact_precision := lower(COALESCE(
        NULLIF(option_map->>'semantic_impact_precision', ''),
        'f32'
    ));
    semantic_alpha_mass := COALESCE(
        NULLIF(option_map->>'semantic_alpha_mass', '')::double precision,
        1.0
    );
    IF runtime_precision NOT IN ('fp16', 'fp32') THEN
        semantic_configuration_error := format(
            'unsupported runtime_precision: %s',
            runtime_precision
        );
    END IF;
    IF semantic_impact_precision NOT IN ('f32', 'fp16', 'u8') THEN
        semantic_configuration_error := format(
            'unsupported semantic_impact_precision: %s',
            semantic_impact_precision
        );
    END IF;
    configured_model_path := NULLIF(option_map->>'model_path', '');
    environment_model_path := NULLIF(
        pg_catalog.current_setting('evoke.sae_model_path', true),
        ''
    );
    packaged_model_path := evoke_packaged_model_path_internal();
    IF sae_enabled THEN
        effective_model_path := COALESCE(
            configured_model_path,
            environment_model_path,
            packaged_model_path
        );
        model_path_source := CASE
            WHEN configured_model_path IS NOT NULL THEN 'index'
            WHEN environment_model_path IS NOT NULL THEN 'environment'
            WHEN packaged_model_path IS NOT NULL THEN 'package'
            ELSE 'none'
        END;

        IF effective_model_path IS NULL THEN
            semantic_configuration_error :=
                'no bundled or configured SSR model checkout is available';
        ELSIF index_owner_is_superuser IS DISTINCT FROM true THEN
            semantic_configuration_error :=
                'SSR indexes using server-local models must be owned by '
                || 'a superuser';
        ELSE
            BEGIN
                import_plan := evoke_index_runtime_plan_internal(
                    effective_model_path
                );
            EXCEPTION WHEN others THEN
                semantic_configuration_error := SQLERRM;
            END;

            IF import_plan IS NOT NULL
               AND NULLIF(option_map->>'model', '') IS NOT NULL
               AND option_map->>'model' <> import_plan->>'model_id' THEN
                semantic_configuration_error := format(
                    'index model override %s does not match checkout model %s',
                    option_map->>'model',
                    import_plan->>'model_id'
                );
            ELSIF import_plan IS NOT NULL
               AND NULLIF(option_map->>'atom_space', '') IS NOT NULL
               AND option_map->>'atom_space' <> import_plan->>'atom_space' THEN
                semantic_configuration_error := format(
                    'index atom_space override %s does not match checkout %s',
                    option_map->>'atom_space',
                    import_plan->>'atom_space'
                );
            ELSIF import_plan IS NOT NULL
               AND NULLIF(option_map->>'scoring_profile', '') IS NOT NULL
               AND option_map->>'scoring_profile'
                    <> import_plan->>'scoring_profile' THEN
                semantic_configuration_error := format(
                    'index scoring_profile override %s does not match checkout %s',
                    option_map->>'scoring_profile',
                    import_plan->>'scoring_profile'
                );
            END IF;
        END IF;
    END IF;

    IF semantic_configuration_error IS NULL THEN
        BEGIN
            runtime_signature :=
                evoke_index_runtime_signature_internal(index_name);
        EXCEPTION WHEN others THEN
            semantic_configuration_error := SQLERRM;
        END;
    END IF;

    model_id := COALESCE(
        NULLIF(option_map->>'model', ''),
        NULLIF(import_plan->>'model_id', '')
    );
    atom_space := COALESCE(
        NULLIF(option_map->>'atom_space', ''),
        NULLIF(import_plan->>'atom_space', '')
    );
    scoring_profile := COALESCE(
        NULLIF(option_map->>'scoring_profile', ''),
        NULLIF(import_plan->>'scoring_profile', '')
    );
    consistency := COALESCE(
        option_map->>'consistency',
        CASE WHEN sae_enabled THEN 'eventual' ELSE 'realtime' END
    );

    RETURN jsonb_build_object(
        'index_name', index_name::text,
        'access_method', access_method,
        'index_type', CASE
            WHEN sae_enabled THEN 'semantic'
            ELSE 'bm25'
        END,
        'sae_enabled', sae_enabled,
        'sae_explicit', sae_explicit,
        'sae_route_source', CASE
            WHEN sae_explicit THEN 'explicit'
            ELSE 'default'
        END,
        'consistency', consistency,
        'field_aware', COALESCE(
            (option_map->>'field_aware')::boolean,
            false
        ),
        'model', model_id,
        'model_path', effective_model_path,
        'model_path_source', model_path_source,
        'runtime_precision', runtime_precision,
        'runtime_precision_source', runtime_precision_source,
        'semantic_impact_precision', semantic_impact_precision,
        'semantic_alpha_mass', semantic_alpha_mass,
        'semantic_accuracy_profile', CASE
            WHEN semantic_alpha_mass = 1.0 THEN 'exact'
            ELSE 'approximate'
        END,
        'semantic_configuration_error', semantic_configuration_error,
        'model_runtime_error', import_plan->>'runtime_error',
        'runtime_signature', runtime_signature,
        'payload_owner', 'index_relation',
        'lifecycle', 'postgresql_index',
        'atom_space', atom_space,
        'scoring_profile', scoring_profile,
        'semantic_configuration_ready',
            semantic_configuration_error IS NULL
            AND model_id IS NOT NULL
            AND effective_model_path IS NOT NULL
            AND atom_space IS NOT NULL
            AND scoring_profile IS NOT NULL
            AND runtime_signature IS NOT NULL
    );
END;
$$;

COMMENT ON FUNCTION evoke_index_options_internal(regclass) IS
$evoke_comment$
Resolve the single relation-owned evoke index contract and, when sae=true, the
model checkout used to build its unified posting payload. It never creates or
discovers an external generation or sidecar.
$evoke_comment$;

CREATE FUNCTION evoke_index_options(index_name regclass)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    caller_name name;
    source_table regclass;
    source_row_security boolean := false;
    relation_kind "char";
    access_method name;
BEGIN
    SELECT
        relation.relkind,
        access_method.amname,
        indexed.indrelid::regclass,
        source.relrowsecurity
    INTO
        relation_kind,
        access_method,
        source_table,
        source_row_security
    FROM pg_catalog.pg_class AS relation
    LEFT JOIN pg_catalog.pg_am AS access_method
      ON access_method.oid = relation.relam
    LEFT JOIN pg_catalog.pg_index AS indexed
      ON indexed.indexrelid = relation.oid
    LEFT JOIN pg_catalog.pg_class AS source
      ON source.oid = indexed.indrelid
    WHERE relation.oid = index_name::oid;

    IF source_table IS NULL THEN
        RAISE EXCEPTION 'evoke indexed relation could not be resolved for %',
            index_name;
    END IF;

    caller_name := NULLIF(
        pg_catalog.current_setting('role', true),
        'none'
    );
    caller_name := COALESCE(caller_name, session_user::name);
    IF NOT pg_catalog.has_table_privilege(
        caller_name,
        source_table,
        'SELECT'
    ) THEN
        RAISE EXCEPTION
            'permission denied for evoke index %',
            index_name
            USING ERRCODE = 'insufficient_privilege',
                HINT = 'Grant SELECT on the indexed relation.';
    END IF;

    IF source_row_security THEN
        RAISE EXCEPTION
            'evoke search does not support row-level security on relation %',
            source_table
            USING ERRCODE = 'feature_not_supported',
                DETAIL =
                    'Filtering an already ranked top-k result cannot preserve '
                    || 'correct row-level-security ranking semantics.',
                HINT =
                    'Use a non-RLS search table or a security-filtered '
                    || 'materialized search relation.';
    END IF;

    IF relation_kind = 'I' THEN
        IF access_method IS DISTINCT FROM 'evoke' THEN
            RAISE EXCEPTION
                'index % must use evoke access method, got %',
                index_name,
                access_method;
        END IF;
        RAISE EXCEPTION
            'evoke partitioned parent index % has no physical generation',
            index_name
            USING ERRCODE = 'feature_not_supported',
                HINT =
                    'Inspect evoke_index_status(...) for the explicit blocker '
                    || 'and child index inventory.';
    END IF;

    RETURN evoke_index_options_internal(index_name);
END;
$$;

COMMENT ON FUNCTION evoke_index_options(regclass) IS
$evoke_comment$
Describe the unified evoke index contract. index_type is bm25 or semantic,
sae_enabled is the authoritative query and lifecycle route, and
sae_route_source records explicit or default selection. For sae=true,
model_path overrides evoke.sae_model_path, which overrides the bundled milestone
checkout. Remaining model identities are read from the checkout manifest. The
identity map and unified posting payload are owned by one index relation and
share one PostgreSQL index lifecycle.
$evoke_comment$;

CREATE FUNCTION evoke_index_runtime_options_internal(
    index_name regclass
)
RETURNS jsonb
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    options jsonb;
    runtime_plan jsonb;
    manifest jsonb;
BEGIN
    options := evoke_index_options_internal(index_name);
    IF options->>'access_method' IS DISTINCT FROM 'evoke' THEN
        RAISE EXCEPTION 'index % must use evoke access method', index_name;
    END IF;
    IF NOT COALESCE((options->>'sae_enabled')::boolean, false) THEN
        RAISE EXCEPTION 'index % is not SSR-enabled', index_name;
    END IF;
    IF NULLIF(options->>'semantic_configuration_error', '') IS NOT NULL THEN
        RAISE EXCEPTION '%', options->>'semantic_configuration_error';
    END IF;

    runtime_plan := evoke_index_runtime_plan_internal(
        options->>'model_path'
    );
    manifest := runtime_plan->'manifest';
    IF manifest->>'model_id' <> options->>'model'
        OR runtime_plan->>'atom_space' <> options->>'atom_space'
        OR runtime_plan->>'scoring_profile'
            <> options->>'scoring_profile' THEN
        RAISE EXCEPTION
            'evoke index and checkout runtime identities do not match';
    END IF;

    RETURN runtime_plan || jsonb_build_object(
        'index', options,
        'loader_status', CASE
            WHEN runtime_plan->>'runtime' = 'onnxruntime'
                THEN evoke_runtime_loader_status()
            ELSE NULL
        END
    );
END;
$$;

COMMENT ON FUNCTION evoke_index_runtime_options_internal(regclass) IS
'Internal validated runtime contract for a semantic-enabled evoke index.';

CREATE FUNCTION evoke_encode_text_internal(
    index_name regclass,
    input_text text
)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    runtime_config jsonb;
    manifest jsonb;
    runtime text;
    runtime_precision text;
    runtime_signature text;
    checkout_signature text;
    generation_signature text;
    onnx_result jsonb;
    atom_ids int4[];
    atom_weights real[];
    index_owner_is_superuser boolean := false;
BEGIN
    IF NULLIF(input_text, '') IS NULL THEN
        RAISE EXCEPTION 'evoke model input text must not be empty';
    END IF;

    /*
     * This helper runs with extension-owner privileges so it can reach the
     * bounded runtime service. Preserve the product API's source-table ACL
     * boundary even when the helper is invoked directly.
     */
    PERFORM evoke_index_options(index_name);

    runtime_config := evoke_index_runtime_options_internal(index_name);
    manifest := runtime_config->'manifest';
    runtime := runtime_config->>'runtime';
    runtime_precision := runtime_config#>>'{index,runtime_precision}';
    runtime_signature := runtime_config#>>'{index,runtime_signature}';
    checkout_signature := runtime_config->>'checkout_signature';
    generation_signature :=
        evoke_index_generation_signature_internal(index_name);
    IF generation_signature IS NOT NULL
        AND generation_signature IS DISTINCT FROM runtime_signature THEN
        RAISE EXCEPTION
            'evoke index generation does not match the configured index contract'
            USING DETAIL = format(
                'generation signature %s, runtime signature %s',
                COALESCE(NULLIF(generation_signature, ''), '<missing>'),
                runtime_signature
            ),
            HINT =
                'REINDEX the evoke index after changing model_path, '
                || 'runtime_precision, semantic_impact_precision, '
                || 'semantic_alpha_mass, the default '
                || 'model checkout, or any model artifact.';
    END IF;

    IF runtime = 'onnxruntime' THEN
        /*
         * The direct model-path runtime API stays superuser-only because it can
         * read server-local checkout files. The product path is narrower: a
         * superuser-created model index may expose text search to ordinary SQL
         * roles, but an ordinary role must not be able to create an arbitrary
         * model_path reloption and then use this SECURITY DEFINER wrapper to
         * read it through the runtime worker.
         */
        SELECT r.rolsuper
        INTO index_owner_is_superuser
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_roles AS r
          ON r.rolname = pg_catalog.pg_get_userbyid(c.relowner)
        WHERE c.oid = index_name;

        IF index_owner_is_superuser IS DISTINCT FROM true THEN
            RAISE EXCEPTION
                'ONNX-backed evoke model index % must be owned by a superuser',
                index_name
                USING HINT =
                    'Register the server-local model checkout and create the '
                    || 'semantic-enabled index as a superuser, then grant ordinary '
                    || 'SQL roles access to the product query surface.';
        END IF;

        onnx_result := evoke_runtime_service_query_atoms(
            runtime_config->>'model_path',
            runtime_precision,
            input_text
        );
        IF onnx_result->>'checkout_signature'
            IS DISTINCT FROM checkout_signature THEN
            RAISE EXCEPTION
                'evoke model checkout changed during query encoding'
                USING DETAIL = format(
                    'expected checkout signature %s, worker used %s',
                    COALESCE(checkout_signature, '<missing>'),
                    COALESCE(
                        onnx_result->>'checkout_signature',
                        '<missing>'
                    )
                ),
                HINT =
                    'Publish model checkouts atomically, then retry. REINDEX '
                    || 'the evoke index before querying a new checkout.';
        END IF;
        IF evoke_index_runtime_signature_internal(index_name)
            IS DISTINCT FROM runtime_signature THEN
            RAISE EXCEPTION
                'evoke model contract changed during query encoding'
                USING HINT =
                    'Retry after the checkout update completes, then REINDEX '
                    || 'the evoke index before querying the new model.';
        END IF;
        IF onnx_result->>'runtime_precision'
            IS DISTINCT FROM runtime_precision THEN
            RAISE EXCEPTION
                'evoke runtime precision changed during query encoding'
                USING DETAIL = format(
                    'expected precision %s, worker used %s',
                    COALESCE(runtime_precision, '<missing>'),
                    COALESCE(
                        onnx_result->>'runtime_precision',
                        '<missing>'
                    )
                ),
                HINT =
                    'REINDEX the evoke index with a single runtime_precision '
                    || 'setting.';
        END IF;
        IF jsonb_typeof(onnx_result->'atoms') <> 'array' THEN
            RAISE EXCEPTION 'ONNX runtime atoms must be an array';
        END IF;
        IF jsonb_typeof(onnx_result->'weights') <> 'array' THEN
            RAISE EXCEPTION 'ONNX runtime weights must be an array';
        END IF;

        SELECT array_agg(value::int4 ORDER BY ordinality)
        INTO atom_ids
        FROM jsonb_array_elements_text(onnx_result->'atoms')
            WITH ORDINALITY AS atom(value, ordinality);

        SELECT array_agg(value::real ORDER BY ordinality)
        INTO atom_weights
        FROM jsonb_array_elements_text(onnx_result->'weights')
            WITH ORDINALITY AS weight(value, ordinality);
    ELSE
        RAISE EXCEPTION 'unsupported evoke model runtime: %', runtime;
    END IF;

    IF cardinality(atom_ids) = 0
        OR cardinality(atom_ids) <> cardinality(atom_weights) THEN
        RAISE EXCEPTION
            'evoke model runtime returned invalid atom arrays: % atoms, % weights',
            cardinality(atom_ids),
            cardinality(atom_weights);
    END IF;

    RETURN jsonb_build_object(
        'atoms', to_jsonb(atom_ids),
        'weights', to_jsonb(atom_weights),
        'runtime', runtime,
        'runtime_precision', runtime_precision,
        'runtime_signature', runtime_signature,
        'model_id', runtime_config#>>'{manifest,model_id}',
        'input_length', length(input_text),
        'runtime_result', onnx_result
    );
END;
$$;

COMMENT ON FUNCTION evoke_encode_text_internal(regclass, text) IS
'Internal single-text encoder used by the latency-sensitive evoke query path.';

CREATE FUNCTION evoke_encode_document_batch_internal(
    index_name regclass,
    input_texts text[]
)
RETURNS TABLE(
    row_ordinal int4,
    atom_ids int4[],
    atom_weights real[]
)
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    runtime_config jsonb;
    manifest jsonb;
    runtime text;
    runtime_precision text;
    runtime_signature text;
    checkout_signature text;
    generation_signature text;
    onnx_result jsonb;
    result_item jsonb;
    input_count int4;
    max_batch_size int4;
    max_batch_bytes int4 := 1048575;
    chunk_start int4;
    chunk_end int4;
    chunk_bytes int4;
    text_bytes int4;
    chunk_count int4;
    batch_offset int4;
    caller_name name;
    index_owner name;
    index_owner_is_superuser boolean := false;
BEGIN
    IF array_ndims(input_texts) IS DISTINCT FROM 1
        OR array_lower(input_texts, 1) IS DISTINCT FROM 1
        OR cardinality(input_texts) IS NULL
        OR cardinality(input_texts) = 0 THEN
        RAISE EXCEPTION
            'evoke model input batch must be a non-empty one-dimensional array';
    END IF;
    /*
     * Build/rebuild callers must own the index. Query roles use the bounded
     * single-text path and cannot turn this helper into a bulk inference API.
     */
    PERFORM evoke_index_options(index_name);
    caller_name := NULLIF(
        pg_catalog.current_setting('role', true),
        'none'
    );
    caller_name := COALESCE(caller_name, session_user::name);
    SELECT r.rolname, r.rolsuper
    INTO index_owner, index_owner_is_superuser
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_roles AS r
      ON r.oid = c.relowner
    WHERE c.oid = index_name;
    IF index_owner IS NULL
        OR NOT pg_catalog.pg_has_role(
            caller_name,
            index_owner,
            'USAGE'
        ) THEN
        RAISE EXCEPTION
            'permission denied for evoke batch encoder on index %',
            index_name
            USING ERRCODE = 'insufficient_privilege',
                HINT =
                    'Batch encoding is reserved for the index owner during '
                    || 'build, REINDEX, and lifecycle maintenance.';
    END IF;

    input_count := cardinality(input_texts);
    FOR row_ordinal IN 1..input_count LOOP
        IF NULLIF(input_texts[row_ordinal], '') IS NULL THEN
            RAISE EXCEPTION
                'evoke model input text % must not be null or empty',
                row_ordinal;
        END IF;
    END LOOP;

    runtime_config := evoke_index_runtime_options_internal(index_name);
    manifest := runtime_config->'manifest';
    runtime := runtime_config->>'runtime';
    runtime_precision := runtime_config#>>'{index,runtime_precision}';
    runtime_signature := runtime_config#>>'{index,runtime_signature}';
    checkout_signature := runtime_config->>'checkout_signature';
    generation_signature :=
        evoke_index_generation_signature_internal(index_name);
    IF generation_signature IS NOT NULL
        AND generation_signature IS DISTINCT FROM runtime_signature THEN
        RAISE EXCEPTION
            'evoke index generation does not match the configured index contract'
            USING DETAIL = format(
                'generation signature %s, runtime signature %s',
                COALESCE(NULLIF(generation_signature, ''), '<missing>'),
                runtime_signature
            ),
            HINT =
                'REINDEX the evoke index after changing model_path, '
                || 'runtime_precision, semantic_impact_precision, '
                || 'semantic_alpha_mass, the default '
                || 'model checkout, or any model artifact.';
    END IF;

    IF runtime <> 'onnxruntime' THEN
        RAISE EXCEPTION 'unsupported evoke model runtime: %', runtime;
    END IF;

    IF index_owner_is_superuser IS DISTINCT FROM true THEN
        RAISE EXCEPTION
            'ONNX-backed evoke model index % must be owned by a superuser',
            index_name
            USING HINT =
                'Create the semantic-enabled index as a superuser before '
                || 'using the product query or maintenance surface.';
    END IF;

    /*
     * The C document runtime owns target-specific batching and bounded
     * concurrency. Keep only the public call-size and transport-byte bounds
     * here so maintenance can use all configured accelerators.
     */
    max_batch_size := 512;
    chunk_start := 1;
    WHILE chunk_start <= input_count LOOP
        chunk_end := chunk_start;
        chunk_bytes := 0;
        WHILE chunk_end <= input_count
            AND chunk_end < chunk_start + max_batch_size LOOP
            text_bytes := octet_length(input_texts[chunk_end]) + 1;
            IF text_bytes > max_batch_bytes THEN
                RAISE EXCEPTION
                    'evoke model input text % exceeds runtime transport limit',
                    chunk_end
                    USING DETAIL = format(
                        'text_bytes=%s total_limit=%s bytes',
                        text_bytes,
                        max_batch_bytes
                    );
            END IF;
            IF chunk_end > chunk_start
                AND chunk_bytes + text_bytes > max_batch_bytes THEN
                EXIT;
            END IF;
            chunk_bytes := chunk_bytes + text_bytes;
            chunk_end := chunk_end + 1;
        END LOOP;
        chunk_end := chunk_end - 1;
        chunk_count := chunk_end - chunk_start + 1;
        onnx_result := evoke_runtime_service_document_atoms_batch(
            runtime_config->>'model_path',
            runtime_precision,
            input_texts[chunk_start:chunk_end]
        );
        IF onnx_result->>'checkout_signature'
            IS DISTINCT FROM checkout_signature THEN
            RAISE EXCEPTION
                'evoke model checkout changed during batch encoding'
                USING DETAIL = format(
                    'expected checkout signature %s, worker used %s',
                    COALESCE(checkout_signature, '<missing>'),
                    COALESCE(
                        onnx_result->>'checkout_signature',
                        '<missing>'
                    )
                ),
                HINT =
                    'Publish model checkouts atomically, then retry. REINDEX '
                    || 'the evoke index before using a new checkout.';
        END IF;
        IF evoke_index_runtime_signature_internal(index_name)
            IS DISTINCT FROM runtime_signature THEN
            RAISE EXCEPTION
                'evoke model contract changed during batch encoding'
                USING HINT =
                    'Retry after the checkout update completes, then REINDEX '
                    || 'the evoke index before using the new model.';
        END IF;
        IF onnx_result->>'runtime_precision'
            IS DISTINCT FROM runtime_precision THEN
            RAISE EXCEPTION
                'evoke runtime precision changed during batch encoding'
                USING DETAIL = format(
                    'expected precision %s, worker used %s',
                    COALESCE(runtime_precision, '<missing>'),
                    COALESCE(
                        onnx_result->>'runtime_precision',
                        '<missing>'
                    )
                ),
                HINT =
                    'REINDEX the evoke index with a single runtime_precision '
                    || 'setting.';
        END IF;
        IF jsonb_typeof(onnx_result->'results') <> 'array'
            OR jsonb_array_length(onnx_result->'results') <> chunk_count THEN
            RAISE EXCEPTION
                'ONNX runtime batch result count does not match its input';
        END IF;

        FOR batch_offset IN 0..chunk_count - 1 LOOP
            result_item := onnx_result->'results'->batch_offset;
            IF jsonb_typeof(result_item->'atoms') <> 'array'
                OR jsonb_typeof(result_item->'weights') <> 'array' THEN
                RAISE EXCEPTION
                    'ONNX runtime batch row % has invalid atom arrays',
                    chunk_start + batch_offset;
            END IF;
            SELECT array_agg(value::int4 ORDER BY ordinality)
            INTO atom_ids
            FROM jsonb_array_elements_text(result_item->'atoms')
                WITH ORDINALITY AS atom(value, ordinality);
            SELECT array_agg(value::real ORDER BY ordinality)
            INTO atom_weights
            FROM jsonb_array_elements_text(result_item->'weights')
                WITH ORDINALITY AS weight(value, ordinality);
            IF cardinality(atom_ids) = 0
                OR cardinality(atom_ids) <> cardinality(atom_weights) THEN
                RAISE EXCEPTION
                    'evoke model runtime returned invalid batch row %',
                    chunk_start + batch_offset;
            END IF;
            row_ordinal := chunk_start + batch_offset;
            RETURN NEXT;
        END LOOP;
        chunk_start := chunk_end + 1;
    END LOOP;
END;
$$;

COMMENT ON FUNCTION evoke_encode_document_batch_internal(regclass, text[]) IS
'Internal bounded document batch encoder used by evoke build, REINDEX, lifecycle rebuild, and mutable SSR DML.';

CREATE FUNCTION evoke_index_semantic_query_native_internal(
    index_name regclass,
    query_atoms int4[],
    query_weights real[],
    field_names text[],
    field_weights real[],
    k int4,
    expected_generation_signature text,
    allowed_doc_ids int4[] DEFAULT NULL,
    allowed_tids tid[] DEFAULT NULL,
    filters jsonb DEFAULT NULL
)
RETURNS TABLE(
    rank int4,
    ctid tid,
    doc_ord int4,
    score float8,
    selected_terms int4,
    candidate_docs int4,
    scored_docs int4,
    generator_entry_visits int8,
    generator_decoded_postings int8,
    candidate_postings int8,
    rerank_binary_steps int8,
    rerank_slice_hits int8,
    rerank_score_terms int8,
    rerank_doc_terms int8,
    memory_bytes int8
)
AS 'MODULE_PATHNAME', 'evoke_index_semantic_query_native_internal'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_index_semantic_query_native_internal(
    regclass,
    int4[],
    real[],
    text[],
    real[],
    int4,
    text,
    int4[],
    tid[],
    jsonb
) IS
'Internal scorer over one convergent mixed-extent posting snapshot, pinned to the runtime signature used to encode the query.';

CREATE FUNCTION evoke_catalog_contract_internal()
RETURNS text
LANGUAGE sql IMMUTABLE PARALLEL SAFE
SET search_path = pg_catalog
AS $$
    SELECT 'evoke_catalog_v1'::text
$$;

COMMENT ON FUNCTION evoke_catalog_contract_internal() IS
'Return the current-only Evoke extension catalog contract.';

CREATE FUNCTION evoke_query_trace_internal()
RETURNS jsonb
AS 'MODULE_PATHNAME', 'evoke_query_trace_internal'
LANGUAGE C VOLATILE PARALLEL UNSAFE;

COMMENT ON FUNCTION evoke_query_trace_internal() IS
'Return bounded backend-local telemetry for the most recent semantic query in this session.';

CREATE FUNCTION evoke_query_semantic_internal(
    index_name regclass,
    query_text text,
    k int4 DEFAULT 20,
    field_names text[] DEFAULT NULL,
    field_weights real[] DEFAULT NULL,
    allowed_doc_ids int4[] DEFAULT NULL,
    allowed_tids tid[] DEFAULT NULL,
    filters jsonb DEFAULT NULL
)
RETURNS TABLE(
    rank int4,
    ctid tid,
    doc_ord int4,
    score float8,
    selected_terms int4,
    candidate_docs int4,
    scored_docs int4,
    generator_entry_visits int8,
    generator_decoded_postings int8,
    candidate_postings int8,
    rerank_binary_steps int8,
    rerank_slice_hits int8,
    rerank_score_terms int8,
    rerank_doc_terms int8,
    memory_bytes int8
)
LANGUAGE plpgsql VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    encoded jsonb;
    query_atoms int4[];
    query_weights real[];
    expected_generation_signature text;
    source_table regclass;
BEGIN
    IF k IS NULL OR k <= 0 THEN
        RAISE EXCEPTION 'evoke semantic query k must be positive';
    END IF;
    IF NULLIF(query_text, '') IS NULL THEN
        RAISE EXCEPTION 'evoke semantic query text must not be empty';
    END IF;
    encoded := evoke_encode_text_internal(index_name, query_text);
    SELECT array_agg(value::int4 ORDER BY ordinality)
    INTO query_atoms
    FROM jsonb_array_elements_text(encoded->'atoms')
        WITH ORDINALITY AS atom(value, ordinality);
    SELECT array_agg(value::real ORDER BY ordinality)
    INTO query_weights
    FROM jsonb_array_elements_text(encoded->'weights')
        WITH ORDINALITY AS weight(value, ordinality);
    expected_generation_signature := encoded->>'runtime_signature';
    IF NULLIF(expected_generation_signature, '') IS NULL THEN
        RAISE EXCEPTION
            'evoke query encoder did not return a generation signature';
    END IF;

    SELECT indrelid::regclass
    INTO source_table
    FROM pg_catalog.pg_index
    WHERE indexrelid = index_name::oid;
    IF source_table IS NULL THEN
        RAISE EXCEPTION 'evoke indexed relation could not be resolved for %',
            index_name;
    END IF;
    RETURN QUERY EXECUTE format(
        'SELECT '
        || 'q.rank, q.ctid, q.doc_ord, q.score, q.selected_terms, '
        || 'q.candidate_docs, q.scored_docs, '
        || 'q.generator_entry_visits, q.generator_decoded_postings, '
        || 'q.candidate_postings, q.rerank_binary_steps, '
        || 'q.rerank_slice_hits, q.rerank_score_terms, '
        || 'q.rerank_doc_terms, q.memory_bytes '
        || 'FROM evoke_index_semantic_query_native_internal('
        || '$1, $2, $3, $4, $5, $6, $7, $8, $9, $10) AS q '
        || 'JOIN %s AS source ON source.ctid = q.ctid '
        || 'ORDER BY q.rank',
        source_table
    ) USING
        index_name,
        query_atoms,
        query_weights,
        field_names,
        field_weights,
        k,
        expected_generation_signature,
        allowed_doc_ids,
        allowed_tids,
        filters;
END;
$$;

COMMENT ON FUNCTION evoke_query_semantic_internal(
    regclass,
    text,
    int4,
    text[],
    real[],
    int4[],
    tid[],
    jsonb
) IS
$evoke_comment$
Internal model-backed executor for evoke_query. Encoding, unified posting payload,
document ordinals, and physical TIDs all come from the active evoke index
generation. The checked page-native root is the sole scorer authority. Current
heap-row validation removes deleted or superseded candidate versions without a
separate identity or overlay lifecycle.
$evoke_comment$;

CREATE FUNCTION evoke_index_status(index_name regclass)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    options jsonb;
    details jsonb;
    generation jsonb;
    runtime_state jsonb;
    caller_name name;
    relation_kind "char";
    access_method name;
    source_table regclass;
    source_row_security boolean := false;
    partition_children jsonb;
    runtime_signature text;
    runtime_signature_error text;
    signature_matches boolean := true;
    index_valid boolean := false;
    index_ready boolean := false;
    query_usable boolean := false;
    query_ready boolean := false;
    performance_ready boolean := false;
    performance_blocker text := 'none';
    accelerator_ready boolean := false;
    query_warm_requested boolean := false;
    query_metadata_warm boolean := false;
    resident_fold_current boolean := false;
    blocker text := 'none';
BEGIN
    SELECT
        relation.relkind,
        access_method.amname,
        indexed.indrelid::regclass,
        source.relrowsecurity,
        indexed.indisvalid,
        indexed.indisready
    INTO
        relation_kind,
        access_method,
        source_table,
        source_row_security,
        index_valid,
        index_ready
    FROM pg_catalog.pg_class AS relation
    LEFT JOIN pg_catalog.pg_am AS access_method
      ON access_method.oid = relation.relam
    LEFT JOIN pg_catalog.pg_index AS indexed
      ON indexed.indexrelid = relation.oid
    LEFT JOIN pg_catalog.pg_class AS source
      ON source.oid = indexed.indrelid
    WHERE relation.oid = index_name::oid;

    IF relation_kind = 'I' THEN
        IF access_method IS DISTINCT FROM 'evoke' THEN
            RAISE EXCEPTION
                'index % must use evoke access method, got %',
                index_name,
                access_method;
        END IF;
        caller_name := NULLIF(
            pg_catalog.current_setting('role', true),
            'none'
        );
        caller_name := COALESCE(caller_name, session_user::name);
        IF NOT pg_catalog.has_table_privilege(
            caller_name,
            source_table,
            'SELECT'
        ) THEN
            RAISE EXCEPTION 'permission denied for evoke index %', index_name
                USING ERRCODE = 'insufficient_privilege',
                    HINT = 'Grant SELECT on the indexed relation.';
        END IF;
        IF source_row_security THEN
            RAISE EXCEPTION
                'evoke search does not support row-level security on relation %',
                source_table
                USING ERRCODE = 'feature_not_supported',
                    DETAIL =
                        'Filtering an already ranked top-k result cannot '
                        || 'preserve correct row-level-security ranking '
                        || 'semantics.',
                    HINT =
                        'Use a non-RLS search table or a security-filtered '
                        || 'materialized search relation.';
        END IF;

        WITH RECURSIVE descendants AS (
            SELECT inheritance.inhrelid
            FROM pg_catalog.pg_inherits AS inheritance
            WHERE inheritance.inhparent = index_name::oid
            UNION ALL
            SELECT inheritance.inhrelid
            FROM descendants
            JOIN pg_catalog.pg_inherits AS inheritance
              ON inheritance.inhparent = descendants.inhrelid
        )
        SELECT COALESCE(
            jsonb_agg(
                jsonb_build_object(
                    'index_name', child.oid::regclass::text,
                    'source_table', child_index.indrelid::regclass::text,
                    'index_valid', child_index.indisvalid,
                    'index_ready', child_index.indisready
                )
                ORDER BY child.oid::regclass::text
            ) FILTER (WHERE child.relkind = 'i'),
            '[]'::jsonb
        )
        INTO partition_children
        FROM descendants
        JOIN pg_catalog.pg_class AS child
          ON child.oid = descendants.inhrelid
        LEFT JOIN pg_catalog.pg_index AS child_index
          ON child_index.indexrelid = child.oid;

        RETURN jsonb_build_object(
            'api_version', 'evoke_index_v1',
            'index_name', index_name::text,
            'index_type', 'partitioned_evoke_unsupported',
            'sae_enabled', NULL,
            'query_usable', false,
            'query_ready', false,
            'performance_ready', false,
            'performance_blocker', 'partitioned_parent_not_supported',
            'blocker', 'partitioned_parent_not_supported',
            'index_valid', index_valid,
            'index_ready', index_ready,
            'options', NULL,
            'details', jsonb_build_object(
                'partitioned_parent', true,
                'source_table', source_table::text,
                'child_indexes', partition_children
            ),
            'generation', NULL,
            'runtime_signature', NULL,
            'runtime_signature_matches', false,
            'status_error',
                'partitioned parent indexes do not own one globally ranked '
                || 'evoke generation'
        );
    END IF;

    options := evoke_index_options(index_name);
    IF options->>'access_method' IS DISTINCT FROM 'evoke' THEN
        RAISE EXCEPTION
            'index % must use evoke access method, got %',
            index_name,
            options->>'access_method';
    END IF;

    SELECT to_jsonb(detail_row)
    INTO details
    FROM evoke_index_details(index_name) AS detail_row;

    SELECT indisvalid, indisready
    INTO index_valid, index_ready
    FROM pg_catalog.pg_index
    WHERE indexrelid = index_name::oid;

    BEGIN
        generation := evoke_index_generation_status_internal(index_name);
    EXCEPTION WHEN others THEN
        generation := jsonb_build_object('error', SQLERRM);
    END;
    IF generation#>>'{layout,storage}' = 'convergent_segments'
       AND generation->>'docs_scope' = 'sealed_generation' THEN
        -- Delta counters remain owned by this page-native generation snapshot.
        -- The separately collected details row may straddle worker publication.
        generation := generation || jsonb_build_object(
            'sealed_docs',
                COALESCE((generation->>'docs')::int8, 0),
            'docs_scope',
                'sealed_generation',
            'retirement_statistics',
                'vacuum_convergent'
        );
        details := details || jsonb_build_object(
            'docs', COALESCE((generation->>'docs')::int8, 0),
            'pending_writes',
                COALESCE((generation#>>'{delta,upserts}')::int8, 0),
            'pending_deletes',
                COALESCE((generation#>>'{delta,retirements}')::int8, 0),
            'delta_records',
                COALESCE((generation#>>'{delta,records}')::int8, 0),
            'delta_bytes',
                COALESCE((generation#>>'{delta,bytes}')::int8, 0)
        );
    END IF;
    runtime_signature := options->>'runtime_signature';
    runtime_signature_error := options->>'semantic_configuration_error';
    signature_matches := runtime_signature IS NOT NULL
        AND generation->>'contract_signature' IS NOT NULL
        AND runtime_signature = generation->>'contract_signature';

    IF COALESCE((options->>'sae_enabled')::boolean, false) THEN
        query_usable := index_valid
            AND index_ready
            AND COALESCE(
                (options->>'semantic_configuration_ready')::boolean,
                false
            )
            AND COALESCE((generation->>'valid')::boolean, false)
            AND COALESCE(
                (generation#>>'{posting,active}')::boolean,
                false
            )
            AND signature_matches;

        BEGIN
            runtime_state := evoke_index_runtime_state_json(index_name);
        EXCEPTION WHEN others THEN
            runtime_state := jsonb_build_object('error', SQLERRM);
        END;
        accelerator_ready := COALESCE(
            (generation#>>'{semantic_accelerator,eligible}')::boolean,
            false
        );
        query_warm_requested := COALESCE(
            (
                runtime_state#>>'{generation,auto_preload_priority}'
            )::int4,
            0
        ) > 0;
        query_metadata_warm := COALESCE(
            (
                runtime_state#>>'{shared_preload,query_metadata_warm}'
            )::boolean,
            false
        );
        resident_fold_current := COALESCE(
            (
                runtime_state#>>'{shared_preload,resident_fold_current}'
            )::boolean,
            false
        );
        performance_ready := query_usable
            AND (accelerator_ready OR resident_fold_current)
            AND (
                NOT query_warm_requested
                OR resident_fold_current
                OR query_metadata_warm
            );
        query_ready := query_usable;

        blocker := CASE
            WHEN NOT COALESCE(
                (options->>'semantic_configuration_ready')::boolean,
                false
            ) THEN 'semantic_configuration_unavailable'
            WHEN NOT index_valid OR NOT index_ready
                THEN 'physical_index_not_ready'
            WHEN generation ? 'error' THEN 'generation_status_unavailable'
            WHEN NOT COALESCE((generation->>'valid')::boolean, false)
                THEN 'generation_invalid'
            WHEN NOT COALESCE(
                (generation#>>'{posting,active}')::boolean,
                false
            )
                THEN 'unified_posting_not_ready'
            WHEN NOT signature_matches
                THEN 'runtime_generation_mismatch'
            WHEN NOT query_usable THEN 'semantic_index_not_ready'
            ELSE 'none'
        END;
        performance_blocker := CASE
            WHEN NOT query_usable THEN blocker
            WHEN NOT COALESCE(
                (generation#>>'{delta,semantic_completion,converged}')::boolean,
                false
            )
                THEN 'semantic_completion_pending'
            WHEN NOT accelerator_ready AND NOT resident_fold_current
                THEN 'semantic_accelerator_not_ready'
            WHEN query_warm_requested
                AND NOT resident_fold_current
                AND NOT query_metadata_warm
                THEN 'query_metadata_not_warm'
            ELSE 'none'
        END;
    ELSE
        query_usable := index_valid
            AND index_ready
            AND COALESCE((generation->>'valid')::boolean, false)
            AND signature_matches;
        query_ready := query_usable;
        performance_ready := query_usable;
        blocker := CASE
            WHEN runtime_signature_error IS NOT NULL
                THEN 'generation_contract_unavailable'
            WHEN NOT index_valid OR NOT index_ready
                THEN 'physical_index_not_ready'
            WHEN generation ? 'error' THEN 'generation_status_unavailable'
            WHEN NOT COALESCE((generation->>'valid')::boolean, false)
                THEN 'generation_invalid'
            WHEN NOT signature_matches
                THEN 'runtime_generation_mismatch'
            ELSE 'none'
        END;
        performance_blocker := blocker;
    END IF;

    RETURN jsonb_build_object(
        'api_version', 'evoke_index_v1',
        'index_name', index_name::text,
        'index_type', options->>'index_type',
        'sae_enabled', COALESCE(
            (options->>'sae_enabled')::boolean,
            false
        ),
        'query_usable', query_usable,
        'query_ready', query_ready,
        'performance_ready', performance_ready,
        'performance_blocker', performance_blocker,
        'blocker', blocker,
        'index_valid', index_valid,
        'index_ready', index_ready,
        'options', options,
        'details', details,
        'generation', generation,
        'runtime_signature', runtime_signature,
        'runtime_signature_matches', signature_matches,
        'model_artifacts_valid', NULL,
        'model_artifact_validation', CASE
            WHEN COALESCE((options->>'sae_enabled')::boolean, false)
                THEN 'explicit_audit_required'
            ELSE 'not_applicable'
        END,
        'status_error', COALESCE(
            generation->>'error',
            runtime_signature_error,
            options->>'semantic_configuration_error'
        )
    );
END;
$$;

COMMENT ON FUNCTION evoke_index_status(regclass) IS
$evoke_comment$
Inspect either bm25 or semantic through one product lifecycle API. The
generation object reports the primary identity/BM25 payload, unified posting
payload, and mutable delta section under one publication and rollback boundary.
$evoke_comment$;

CREATE FUNCTION evoke_index_audit(index_name regclass)
RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER VOLATILE PARALLEL UNSAFE
SET search_path FROM CURRENT
AS $$
DECLARE
    status jsonb;
    options jsonb;
    generation jsonb;
    generation_error text;
    artifact_validation_error text;
    artifacts_valid boolean;
    sae_enabled boolean := false;
BEGIN
    status := evoke_index_status(index_name);
    options := status->'options';
    sae_enabled := COALESCE(
        (options->>'sae_enabled')::boolean,
        false
    );

    BEGIN
        generation := evoke_index_generation_audit_internal(index_name);
    EXCEPTION WHEN others THEN
        generation_error := SQLERRM;
        generation := jsonb_build_object('error', generation_error);
    END;

    IF sae_enabled AND NULLIF(options->>'model_path', '') IS NOT NULL THEN
        BEGIN
            PERFORM evoke_index_checkout_validate_internal(
                options->>'model_path'
            );
            artifacts_valid := true;
        EXCEPTION WHEN others THEN
            artifacts_valid := false;
            artifact_validation_error := SQLERRM;
        END;
    END IF;

    RETURN jsonb_build_object(
        'api_version', 'evoke_index_audit_v1',
        'index_name', index_name::text,
        'passed',
            COALESCE((status->>'query_ready')::boolean, false)
            AND COALESCE((generation->>'valid')::boolean, false)
            AND generation_error IS NULL
            AND (
                NOT sae_enabled
                OR artifacts_valid IS TRUE
            ),
        'status', status,
        'generation', generation,
        'model_artifacts_valid', artifacts_valid,
        'model_artifact_error', artifact_validation_error,
        'audit_error', COALESCE(
            generation_error,
            artifact_validation_error
        )
    );
END;
$$;

COMMENT ON FUNCTION evoke_index_audit(regclass) IS
$evoke_comment$
Run the explicit relation-sized integrity audit for an evoke index. This walks
the complete generation closure and, for SSR indexes, SHA-256 validates every
model artifact. Use evoke_index_status for bounded readiness polling.
$evoke_comment$;

-- Keep direct runtime, exact rowset, and internal diagnostics outside the
-- application top-k privilege surface. Fusion and hybrid composition are
-- public product APIs layered above one or more independently authorized
-- evoke/vector query sources.
REVOKE EXECUTE ON FUNCTION evoke_query_ids(
    regclass, int4[], int4, real[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_query_tokens(
    regclass, text[], int4, real[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_field_aware_query_tokens(
    regclass, text[], text[], real[], int4
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_field_aware_query(
    regclass, text, text[], real[], int4
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_query_bm25_internal(
    regclass, text, int4, real[], boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_prepared_query(
    regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_order_tokens(
    evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_order_tokens(
    regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_op_match_prepared_query(
    text[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_op_match_prepared_query(
    varchar[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_op_match_prepared_query_scalar(
    text, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_op_match_prepared_query_scalar(
    varchar, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_prepared_query(
    text[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_prepared_query(
    varchar[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_prepared_query(
    text, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_prepared_query(
    varchar, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_query(
    text[], regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_query(
    varchar[], regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_query(
    text, regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_match_query(
    varchar, regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_prepared_query(
    text[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_prepared_query(
    varchar[], evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_prepared_query(
    text, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_prepared_query(
    varchar, evoke_result_prepared_query
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_query(
    text[], regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_query(
    varchar[], regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_query(
    text, regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_score_query(
    varchar, regclass, text, boolean, text[], boolean, boolean
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_cache_clear() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_touch_maintenance() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_maintain_due(integer) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_try_maintenance_lock(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_maintenance_unlock(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_maintenance_lock_held(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_onnxruntime_probe() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_status() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_query_atoms(text, text)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_query_atoms(text, text, text)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_query_atoms_batch(text, text[])
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_query_atoms_batch(
    text, text, text[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_document_atoms_batch(text, text[])
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_document_atoms_batch(
    text, text, text[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_atoms_batch_internal(
    text, text, text[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_service_atoms_batch_internal(
    text, text, text, text[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_runtime_loader_status() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_checkout_manifest_internal(text)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_checkout_manifest_signature_internal(jsonb)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_checkout_validate_internal(text)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_runtime_plan_internal(text, jsonb)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_options_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_runtime_options_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_runtime_signature_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_generation_signature_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_packaged_model_path_internal()
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_generation_status_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_generation_audit_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_semantic_quarantine_internal(regclass)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_encode_text_internal(regclass, text)
FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_encode_document_batch_internal(
    regclass, text[]
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_index_semantic_query_native_internal(
    regclass, int4[], real[], text[], real[], int4, text, int4[], tid[], jsonb
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_catalog_contract_internal() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_query_trace_internal() FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_query_semantic_internal(
    regclass, text, int4, text[], real[], int4[], tid[], jsonb
) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION evoke_query_internal(
    regclass, text, text[], real[], int4, real[], boolean, text[],
    boolean, boolean, int4[], tid[], jsonb
) FROM PUBLIC;
