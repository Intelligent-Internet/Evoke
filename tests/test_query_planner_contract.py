from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def current_sql() -> str:
    control = (REPO_ROOT / 'evoke.control').read_text(encoding='utf-8')
    version = control.split("default_version = '", 1)[1].split("'", 1)[0]
    return (REPO_ROOT / 'sql' / f'evoke--{version}.sql').read_text(
        encoding='utf-8'
    )


def test_top_k_query_functions_use_bounded_row_estimates() -> None:
    sql = current_sql()

    function_names = (
        'evoke_query_ids',
        'evoke_query_tokens',
        'evoke_field_aware_query_tokens',
        'evoke_field_aware_query',
    )

    for function_name in function_names:
        start = sql.index(f'CREATE FUNCTION {function_name}(')
        end = sql.find('\nCREATE FUNCTION ', start + 1)
        if end < 0:
            end = len(sql)
        assert 'ROWS 100' in sql[start:end]

    search_start = sql.index(
        'CREATE FUNCTION evoke_query(\n'
        '    index_name regclass,\n'
        '    query_text text,\n'
        '    k int4,'
    )
    search_end = sql.find('\nCREATE FUNCTION ', search_start + 1)
    assert 'ROWS 100' in sql[search_start:search_end]


def test_planner_native_probe_is_isolated_and_fail_closed() -> None:
    sql = current_sql()
    planner = (REPO_ROOT / 'src' / 'evoke_planner.c').read_text(
        encoding='utf-8'
    )

    assert 'CREATE FUNCTION evoke_query(index_name regclass' in sql
    assert 'COMMENT ON FUNCTION evoke_query(regclass, text, text[], real[])' in sql
    assert 'LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE' in sql
    assert 'PG_FUNCTION_INFO_V1(evoke_query_marker)' in planner
    assert 'strcmp(function_name, "evoke_query") == 0' in planner
    assert 'function->funcresulttype != FLOAT4OID' in planner
    assert 'create_upper_paths_hook = evoke_create_upper_paths' in planner
    assert 'evoke_previous_upper_paths_hook(' in planner
    assert 'stage != UPPERREL_ORDERED' in planner
    assert 'evoke.enable_planner_native' in planner
    assert 'true,\n        PGC_USERSET' in planner
    assert 'table_tuple_fetch_row_version(' in planner
    assert 'SPI_execute_with_args(' not in planner
    assert 'evoke_am_query_text(' in planner
    assert 'create_projection_path(' in planner
    assert 'path->custom_restrictinfo = list_copy(' in planner
    assert 'path->path.parent = output_relation' in planner
    assert 'path->path.pathtarget = root->upper_targets[UPPERREL_ORDERED]' in planner
    assert 'scan->scan.scanrelid = scan_relation_id' in planner
    assert 'SelfItemPointerAttributeNumber' in planner
    assert 'bms_num_members(root->all_baserels) != 1' in planner
    assert 'list_length(query->sortClause) != 1' in planner
    assert 'query->limitOption != LIMIT_OPTION_COUNT' in planner
    assert 'query->hasForUpdate' in planner
    assert 'query->hasRowSecurity' in planner
    assert 'contain_var_clause(argument)' in planner
    assert 'contain_volatile_functions(argument)' in planner
    assert 'root->glob->relationOids = lappend_oid(' in planner
    assert 'ExecInitNode(' in planner
    assert 'ExecReScan(' in planner
    assert 'ExecEndNode(' in planner
    assert 'getExtensionOfObject(ProcedureRelationId' in planner
    assert 'index->relam == evoke_am_oid' in planner
    assert 'index->indpred == NIL || index->predOK' not in planner
    assert "rows outside a partial index\n             * predicate" in planner
    assert 'evoke_am_sae_enabled(index_relation)' in planner


def test_planner_native_filter_uses_scope_then_exact_fallback() -> None:
    planner = (REPO_ROOT / 'src' / 'evoke_planner.c').read_text(
        encoding='utf-8'
    )

    assert 'evoke_build_filter_tid_path(' in planner
    assert 'evoke_planner_collect_allowed_tid_keys(' in planner
    assert 'while (IsA(tid_source, ProjectionPath))' in planner
    assert 'tid_source->pathtype == T_IndexOnlyScan' in planner
    assert 'evoke_filter_sort_unique_tid_keys(keys, count)' in planner
    assert 'state->filter_plan == NULL ? NULL : allowed_tid_keys' in planner
    assert 'allowed_tid_count,' in planner
    assert 'scan->scan.plan.qual = copyObject(filter_quals);' in planner
    assert 'Allowed TIDs' in planner
    assert 'EVOKE_PLANNER_SCOPE_OVERFETCH_MULTIPLIER 4' in planner
    assert 'evoke_planner_build_scope_filter_spec(' in planner
    assert 'evoke_planner_scope_attribute_is_included(' in planner
    assert 'get_func_namespace(get_opcode(' in planner
    assert 'variable_type != operand_type' in planner
    assert 'EVOKE_PLANNER_SCOPE_ILIKE' in planner
    assert 'strcmp(operator_name, "~~*") == 0' in planner
    assert 'evoke_planner_scope_string_type_supported(' in planner
    assert 'return "ilike";' in planner
    assert 'contain_var_clause(operand)' in planner
    assert 'contain_volatile_functions(operand)' in planner
    assert 'evoke_planner_build_scope_filter(' in planner
    assert 'evoke_planner_filter_scope_hits(' in planner
    assert 'ExecQual(state->filter_qual, expression_context)' in planner
    assert 'DatumGetTextPCopy(query_value)' in planner
    assert 'DatumGetArrayTypePCopy(field_names_value)' in planner
    assert 'state->scope_fallback = true;' in planner
    assert 'evoke_planner_collect_allowed_tid_keys(' in planner
    assert 'Scope Filter Eligible' in planner
    assert 'Scope Filter Complete' in planner
    assert 'Scope Filter Fallback' in planner

    am = (REPO_ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
    query_start = am.index('\nevoke_am_query_text(')
    query_end = am.index('\nstatic void', query_start)
    query = am[query_start:query_end]

    assert 'evoke_filter_try_scope_bitmap(' in query
    assert 'scope_trace.resolved_predicate_count ==' in query
    assert 'evoke_am_align_scope_filter_with_root(' in query
    assert 'scope_document_slot_count,' in query
    assert 'evoke_am_prepare_page_native_filtered_search_state_at_root(' in query
    assert 'query_executed = true;' in query

    scope_align_start = am.index('\nevoke_am_align_scope_filter_with_root(')
    scope_align_end = am.index('\nbool\nevoke_am_query_text(', scope_align_start)
    scope_align = am[scope_align_start:scope_align_end]
    assert 'repalloc(' in scope_align
    assert 'root->next_document_slot' in scope_align
    assert 'scope_document_slot_count & UINT64_C(7)' in scope_align
    assert 'filter_out->live_membership_verified = true;' in scope_align


def test_search_overloads_keep_one_unambiguous_product_name() -> None:
    sql = current_sql()

    assert (
        'CREATE FUNCTION evoke_query(index_name regclass, query_text text)\n'
        'RETURNS real'
    ) in sql
    assert (
        '    field_names text[],\n'
        '    field_weights real[]\n'
        ')\n'
        'RETURNS real\n'
        "AS 'MODULE_PATHNAME', 'evoke_query_marker'"
    ) in sql
    assert (
        '    query_text text,\n'
        '    k int4,\n'
        '    weight_mask real[] DEFAULT NULL'
    ) in sql
    assert (
        '    field_names text[],\n'
        '    field_weights real[],\n'
        '    k int4\n'
        ')\n'
        'RETURNS SETOF evoke_result_hit'
    ) in sql
    assert (
        'CREATE FUNCTION evoke_query(\n'
        '    index_name regclass,\n'
        '    query_text text,\n'
        '    k int4 DEFAULT 10,\n'
        '    weight_mask'
    ) not in sql
    assert (
        'CREATE FUNCTION evoke_query(\n'
        '    index_name regclass,\n'
        '    query_text text,\n'
        '    field_names text[],\n'
        '    field_weights real[] DEFAULT NULL,\n'
        '    k int4 DEFAULT 10\n'
        ')\n'
        'RETURNS SETOF evoke_result_hit'
    ) not in sql
