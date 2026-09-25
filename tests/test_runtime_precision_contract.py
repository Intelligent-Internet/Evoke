from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding='utf-8')


def current_sql() -> str:
    control = read_text('ii42.control')
    version = control.split("default_version = '", 1)[1].split("'", 1)[0]
    return read_text(f'sql/ii42--{version}.sql')


def test_runtime_precision_reloption_defaults_to_fp16() -> None:
    options_c = read_text('src/ii42_am_options.c')

    assert '"runtime_precision"' in options_c
    assert '{"fp16", II42_AM_RUNTIME_PRECISION_FP16}' in options_c
    assert '{"fp32", II42_AM_RUNTIME_PRECISION_FP32}' in options_c
    assert 'II42_AM_RUNTIME_PRECISION_FP16,\n        NULL,' in options_c
    sae_options = options_c.split(
        'static const char *sae_option_names[] = {',
        1,
    )[1].split('};', 1)[0]
    assert '"runtime_precision"' in sae_options
    assert '"model"' in sae_options


def test_runtime_precision_participates_in_index_contract() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'ii42_am_get_runtime_precision(indexRelation)' in am_c
    assert '"runtime_precision",\n            runtime_precision' in am_c
    assert (
        'config#>>\'{index,runtime_precision}\', '
        in am_c
    )
    assert (
        'ii42_runtime_service_submit_document_prefix_checkout_async(\n'
        '                    builder->model_path,\n'
        '                    builder->checkout_signature,\n'
        '                    builder->runtime_precision,'
        in am_c
    )


def test_document_runtime_pipeline_uses_reorder_buffer() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'ii42_am_semantic_completion' in am_c
    assert 'next_submit_sequence' in am_c
    assert 'next_apply_sequence' in am_c
    assert 'ii42_am_semantic_builder_find_free_inflight' in am_c
    assert 'ii42_am_semantic_builder_apply_ready_completions' in am_c
    assert 'builder->pending_count == 0' not in am_c
    assert 'inflight_head' not in am_c


def test_semantic_runtime_collects_large_pipelines_eagerly() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'collect_threshold = Max(collect_threshold / 16, 32);' in am_c
    assert 'collect_threshold = Min(collect_threshold, 256);' in am_c
    assert 'collect_threshold = Max(collect_threshold / 4, 1);' not in am_c


def test_semantic_runtime_completion_checks_readiness_first() -> None:
    am_c = read_text('src/ii42_am.c')

    assert (
        'if (!ii42_runtime_service_request_ready(&inflight->handle))\n'
        '    {\n'
        '        return false;\n'
        '    }\n'
        '    if (inflight->result_context == NULL)'
        in am_c
    )


def test_semantic_build_does_not_hold_spi_across_heap_scan() -> None:
    am_c = read_text('src/ii42_am.c')
    begin_start = am_c.index('ii42_am_semantic_builder_begin(')
    begin_end = am_c.index(
        '\nstatic void\nii42_am_semantic_builder_append_pair_row',
        begin_start
    )
    begin_body = am_c[begin_start:begin_end]

    assert 'SPIPlanPtr decode_plan' not in am_c
    assert 'builder->decode_plan' not in am_c
    assert 'builder->spi_connected' not in am_c
    assert 'SPI_prepare(' not in begin_body
    assert (
        'ii42_am_semantic_builder_load_contract(builder, schema_name);\n'
        '        ii42_am_semantic_builder_require_current_contract('
        in begin_body
    )
    assert 'SPI_finish();\n        spi_connected = false;' in begin_body


def test_semantic_runtime_results_use_short_lived_contexts() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'MemoryContext result_context;' in am_c
    assert '"ii42 semantic runtime result"' in am_c
    assert 'inflight->result_context = AllocSetContextCreate(' in am_c
    assert 'completion->result_context = inflight->result_context;' in am_c
    assert 'inflight->result_context = NULL;' in am_c
    assert 'MemoryContextDelete(parse_context);' in am_c
    assert (
        'ii42_am_semantic_builder_apply_result_in_context('
        in am_c
    )


def test_remote_request_is_canceled_before_result_context_deletion() -> None:
    am_c = read_text('src/ii42_am.c')
    start = am_c.index('ii42_am_semantic_inflight_clear(')
    end = am_c.index(
        '\nstatic bool\nii42_am_semantic_builder_apply_ready_completions',
        start,
    )
    body = am_c[start:end]

    cancel = body.index(
        'ii42_runtime_service_cancel_async(&inflight->handle);'
    )
    delete = body.index('MemoryContextDelete(inflight->result_context);')
    assert cancel < delete


def test_remote_accelerator_submission_does_not_swallow_query_cancel() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    for function_name, next_function_name in (
        (
            'ii42_runtime_service_submit_document_prefix_checkout_async(',
            'ii42_runtime_service_submit_document_checkout_async(',
        ),
        (
            'ii42_runtime_service_submit_document_checkout_async(',
            'ii42_runtime_service_submit_document_async(',
        ),
    ):
        start = semantic_c.index('\n' + function_name)
        end = semantic_c.index('\n' + next_function_name, start)
        body = semantic_c[start:end]
        catch = body.index('PG_CATCH();')
        rethrow = body.index(
            'caught_sqlerrcode == ERRCODE_QUERY_CANCELED',
            catch,
        )
        flush = body.index('FlushErrorState();', catch)

        assert rethrow < flush


def test_semantic_runtime_result_ownership_detaches_before_apply() -> None:
    am_c = read_text('src/ii42_am.c')
    retire_start = am_c.index(
        'ii42_am_semantic_builder_retire_inflight('
    )
    retire_end = am_c.index(
        '\nstatic ii42_am_semantic_inflight *\n'
        'ii42_am_semantic_builder_find_inflight',
        retire_start
    )
    retire_body = am_c[retire_start:retire_end]
    apply_start = retire_body.index(
        '    ii42_am_semantic_builder_apply_result('
    )
    before_apply = retire_body[:apply_start]
    apply_call = retire_body[apply_start:retire_body.index('    );', apply_start)]

    assert 'result_context = inflight->result_context;' in before_apply
    assert 'result_json = inflight->result_json;' in before_apply
    assert 'inflight->result_context = NULL;' in before_apply
    assert 'inflight->result_json = NULL;' in before_apply
    assert 'inflight->result_context' not in apply_call
    assert 'inflight->result_json' not in apply_call


def test_semantic_stream_uses_spill_lexical_builder() -> None:
    am_c = read_text('src/ii42_am.c')

    assert (
        'ii42_am_rebuild_builder_uses_compact_lexical_entries(\n'
        '    ii42_am_rebuild_builder builder'
        in am_c
    )
    assert (
        'builder == II42_AM_REBUILD_BUILDER_SEMANTIC_STREAM'
        in am_c
    )
    assert (
        'ii42_am_rebuild_builder_uses_spill_lexical_entries(\n'
        '    ii42_am_rebuild_builder builder'
        in am_c
    )
    assert (
        'if (ii42_am_rebuild_builder_uses_compact_lexical_entries(\n'
        '            build_state->rebuild_builder))'
        in am_c
    )
    assert (
        'if (ii42_am_rebuild_builder_uses_spill_lexical_entries(builder))'
        in am_c
    )


def test_semantic_stream_build_uses_row_scratch_context() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'MemoryContext row_context;' in am_c
    assert '"ii42 build row scratch"' in am_c
    assert 'ii42_am_build_callback_impl(' in am_c
    assert (
        'ii42_am_rebuild_builder_uses_compact_lexical_entries(\n'
        '            build_state->rebuild_builder))'
        in am_c
    )
    assert 'MemoryContextReset(build_state->row_context);' in am_c
    assert (
        'builder->doc_starts = MemoryContextAlloc(\n'
        '            builder->memory_context,'
        in am_c
    )
    assert 'build_context = build_state->memory_context;' in am_c


def test_semantic_segment_postings_use_bounded_sorted_stream() -> None:
    am_c = read_text('src/ii42_am.c')
    build_c = read_text('src/ii42_am_build.c')
    source_start = am_c.index('ii42_am_semantic_build_segment_source(')
    source_end = am_c.index(
        '\nstatic void\nii42_am_semantic_builder_finish',
        source_start
    )
    source_body = am_c[source_start:source_end]

    assert 'source.postings = malloc(' not in source_body
    assert 'source.postings = BufFileCreateTemp(false);' in source_body
    assert 'tuplesort_begin_heap(' in source_body
    assert 'tuplesort_puttupleslot(sort, input_slot);' in source_body
    assert 'BufFileWrite(source.postings, &posting, sizeof(posting));' in (
        source_body
    )
    assert 'BufFileClose(segment_source.postings);' in am_c
    assert 'BufFileClose(output->segment_semantic_postings);' in build_c
    assert 'free(segment_source.postings);' not in am_c
    assert 'free(output->segment_semantic_postings);' not in build_c


def test_semantic_publication_does_not_copy_sorted_postings() -> None:
    am_c = read_text('src/ii42_am.c')
    build_c = read_text('src/ii42_am_build.c')

    assert 'ii42_am_semantic_segment_posting_cmp' not in am_c
    assert 'ii42_initial_fold_stream_create(' in build_c
    assert 'ii42_am_semantic_posting_file_read' in build_c
    assert 'ii42_segment_payload_build_lexical(' not in build_c
    assert 'ii42_segment_payload_attach_semantic_sorted_reader(' not in build_c
    assert 'ii42_segment_payload_attach_semantic(' not in build_c


def test_publish_releases_sources_before_document_directory_write() -> None:
    build_c = read_text('src/ii42_am_build.c')
    segment_pages_c = read_text('src/ii42_segment_pages.c')
    producer_start = build_c.index('ii42_am_initial_fold_publish_next(')
    producer_end = build_c.index(
        '\nvoid\nii42_am_rebuild_output_release(',
        producer_start
    )
    producer_body = build_c[producer_start:producer_end]
    publish_start = build_c.index('ii42_am_publish_replacement_segments(')
    publish_end = build_c.index(
        '\nstatic uint64\nii42_am_budget_headroom_limit',
        publish_start
    )
    publish_body = build_c[publish_start:publish_end]

    stream_pos = publish_body.index('ii42_initial_fold_stream_create(')
    write_pos = publish_body.index(
        'ii42_segment_pages_write_streamed_initial_folded_bundle_fork('
    )
    next_pos = producer_body.index('ii42_initial_fold_stream_next(')
    release_pos = producer_body.index(
        'ii42_am_rebuild_output_release_materialized_sources('
    )
    streamed_folds_pos = segment_pages_c.index(
        'ii42_segment_pages_write_streamed_initial_folds('
    )
    records_pos = segment_pages_c.index(
        'ii42_segment_pages_write_document_directory_records(',
        streamed_folds_pos
    )

    assert stream_pos < write_pos
    assert next_pos < release_pos
    assert streamed_folds_pos < records_pos
    assert 'ii42_segment_payload_partition_contiguous(' not in publish_body
    assert 'output->doc_tids = NULL;' in build_c
    assert 'output->index_bytes = NULL;' in build_c
    assert 'output->segment_semantic_postings = NULL;' in build_c
    assert 'output->segment_semantic_input_fingerprints = NULL;' in build_c
    assert 'output->index_valid = false;' in build_c


def test_initial_publish_releases_consumed_cow_trees_immediately() -> None:
    segment_pages = read_text('src/ii42_segment_pages.c')
    functions = [
        (
            'ii42_segment_pages_write_document_directory(',
            'ii42_segment_pages_write_document_cow_objects_internal(',
            'ii42_document_cow_tree_free(&cleanup->document_cow_tree);',
        ),
        (
            'ii42_segment_pages_write_cow_term_directory(',
            'ii42_segment_pages_write_term_cow_objects_internal(',
            'ii42_term_cow_tree_free(&cleanup->term_cow_tree);',
        ),
        (
            'ii42_segment_pages_write_lexicon_lookup(',
            'ii42_segment_pages_write_lexicon_cow_objects_internal(',
            'ii42_lexicon_cow_tree_free(&cleanup->lexicon_cow_tree);',
        ),
        (
            'ii42_segment_pages_write_prefix_lookup(',
            'ii42_segment_pages_write_prefix_cow_objects_internal(',
            'ii42_prefix_cow_tree_free(&cleanup->prefix_cow_tree);',
        ),
    ]

    for function_name, write_call, release_call in functions:
        function_start = segment_pages.index(function_name)
        function_end = segment_pages.index('\nstatic ', function_start)
        function_body = segment_pages[function_start:function_end]
        assert function_body.index(write_call) < function_body.index(
            release_call
        )

    term_start = segment_pages.index(
        'ii42_segment_pages_write_cow_term_directory('
    )
    term_end = segment_pages.index('\nstatic ', term_start)
    term_body = segment_pages[term_start:term_end]
    assert term_body.index(
        'ii42_segment_pages_write_term_cow_objects_internal('
    ) < term_body.index(
        'ii42_term_directory_free(&cleanup->term_directory);'
    )


def test_explicit_build_releases_publish_sources_on_all_exit_paths() -> None:
    am_c = read_text('src/ii42_am.c')
    build_start = am_c.index('ii42_am_build_common(')
    build_end = am_c.index('\nstatic IndexBuildResult *\nii42_ambuild(', build_start)
    build_body = am_c[build_start:build_end]
    empty_start = am_c.index('ii42_ambuildempty(')
    empty_end = am_c.index(
        '\nstatic char *\nii42_am_compile_semantic_documents_query',
        empty_start
    )
    empty_body = am_c[empty_start:empty_end]

    assert 'PG_FINALLY();' in build_body
    assert 'ii42_am_rebuild_output_release(&replacement);' in build_body
    assert 'ii42_am_release_unused_malloc();' in build_body
    assert 'replacement.index = cleanup->index;' in empty_body
    assert 'ii42_index_init(&cleanup->index);' in empty_body
    assert 'PG_FINALLY();' in empty_body
    assert 'ii42_am_rebuild_output_release(&replacement);' in empty_body


def test_json_string_escapes_all_control_characters() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    append_start = semantic_c.index('ii42_append_json_string(')
    append_end = semantic_c.index(
        '\nstatic int\nii42_runtime_effective_max_batch_size',
        append_start
    )
    append_body = semantic_c[append_start:append_end]

    assert 'else if (*ptr < 0x20)' in append_body
    assert 'appendStringInfo(out, "\\\\u%04x", (unsigned int) *ptr);' in (
        append_body
    )


def test_async_accelerator_completion_uses_nonblocking_failover() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    header = read_text('src/ii42_runtime_service.h')

    assert 'II42_RUNTIME_SERVICE_VERSION 26' in header
    assert 'uint32 accelerator_retry_count;' in header
    assert 'TimestampTz accelerator_retry_after;' in header
    assert 'TimestampTz accelerator_deadline_at;' in header
    assert 'TimestampTz accelerator_local_fallback_deadline_at;' in header
    assert 'uint64 connect_failures;' in header
    assert 'uint64 read_failures;' in header
    assert 'uint64 status_failures;' in header
    assert 'uint64 backpressure_failures;' in header
    assert 'uint64 body_failures;' in header
    assert 'uint64 status_400_failures;' in header
    assert 'uint64 status_409_failures;' in header
    assert 'uint64 status_500_failures;' in header
    assert 'uint64 status_503_failures;' in header
    assert 'uint64 status_504_failures;' in header
    assert 'uint64 status_other_failures;' in header
    assert 'uint32 last_status_failure;' in header
    assert 'char last_status_body[256];' in header
    assert 'II42_RUNTIME_ACCELERATOR_RETRY_LIMIT 4' in semantic_c
    assert 'II42_RUNTIME_ACCELERATOR_ASYNC_RETRY_LIMIT 4' in semantic_c
    assert 'II42_RUNTIME_ACCELERATOR_ASYNC_RETRY_WAIT_MS 100' in semantic_c
    assert 'ii42_runtime_service_failover_accelerator_async' in semantic_c
    assert 'ii42_runtime_service_enqueue_local_failover_async' in semantic_c
    assert 'ii42_runtime_service_try_enqueue_prepared' in semantic_c
    try_complete_start = semantic_c.index(
        'ii42_runtime_service_try_complete_async('
    )
    try_complete_end = semantic_c.index(
        '\nchar *\nii42_runtime_service_wait_async',
        try_complete_start
    )
    try_complete_body = semantic_c[try_complete_start:try_complete_end]
    assert 'ii42_runtime_service_wait_accelerator(handle)' not in (
        try_complete_body
    )
    assert 'ii42_runtime_service_enqueue(' not in try_complete_body
    assert 'ii42_runtime_service_failover_accelerator_async(' in (
        try_complete_body
    )
    assert 'body = ii42_runtime_accelerator_http_take_body(handle, &status);' in (
        try_complete_body
    )
    assert 'status,\n                    body' in try_complete_body
    request_ready_start = semantic_c.index(
        'ii42_runtime_service_request_ready('
    )
    request_ready_body = semantic_c[request_ready_start:]
    assert (
        'if (handle->accelerator_fd < 0)\n'
        '        {\n'
        '            TimestampTz now = GetCurrentTimestamp();'
        in request_ready_body
    )
    assert 'now >= handle->accelerator_retry_after' in request_ready_body
    assert (
        'now >= handle->accelerator_local_fallback_deadline_at'
        in request_ready_body
    )
    assert (
        'return now >= handle->accelerator_retry_after;'
        in request_ready_body
    )
    local_failover_start = semantic_c.index(
        'ii42_runtime_service_enqueue_local_failover_async('
    )
    local_failover_end = semantic_c.index(
        '\nstatic bool\n'
        'ii42_runtime_service_failover_accelerator_async',
        local_failover_start
    )
    local_failover_body = semantic_c[
        local_failover_start:local_failover_end
    ]
    assert 'ii42_runtime_service_enqueue(' not in local_failover_body
    assert 'ii42_runtime_service_try_enqueue_prepared(' in local_failover_body
    assert 'connect_failures' in semantic_c
    assert 'read_failures' in semantic_c
    assert 'status_failures' in semantic_c
    assert 'backpressure_failures' in semantic_c
    assert 'body_failures' in semantic_c
    assert 'status_400_failures' in semantic_c
    assert 'status_409_failures' in semantic_c
    assert 'status_500_failures' in semantic_c
    assert 'status_503_failures' in semantic_c
    assert 'status_504_failures' in semantic_c
    assert 'status_other_failures' in semantic_c
    assert 'last_status_failure' in semantic_c
    assert 'last_status_body' in semantic_c
    assert 'entry->connect_failures++;' in semantic_c
    assert 'entry->read_failures++;' in semantic_c
    assert 'entry->status_failures++;' in semantic_c
    assert 'entry->backpressure_failures++;' in semantic_c
    assert 'entry->body_failures++;' in semantic_c
    assert 'entry->status_400_failures++;' in semantic_c
    assert 'entry->status_409_failures++;' in semantic_c
    assert 'entry->status_500_failures++;' in semantic_c
    assert 'entry->status_503_failures++;' in semantic_c
    assert 'entry->status_504_failures++;' in semantic_c
    assert 'entry->status_other_failures++;' in semantic_c
    assert 'entry->last_status_failure = (uint32) Max(status, 0);' in (
        semantic_c
    )
    assert 'strlcpy(\n                    entry->last_status_body,' in (
        semantic_c
    )
    assert 'ii42_append_json_string(json, entry->last_status_body);' in (
        semantic_c
    )
    failover_start = semantic_c.index(
        'ii42_runtime_service_failover_accelerator_async('
    )
    failover_end = semantic_c.index(
        '\nstatic char *\n'
        'ii42_runtime_service_wait_accelerator',
        failover_start
    )
    failover_body = semantic_c[failover_start:failover_end]
    assert 'II42_RUNTIME_ACCELERATOR_FAILURE_BODY' in failover_body
    assert 'handle,\n                    -1' in try_complete_body


def test_semantic_build_abort_cleans_remote_inflight_on_error() -> None:
    am_c = read_text('src/ii42_am.c')

    assert 'PG_CATCH();\n    {\n        ii42_am_semantic_builder_abort' in am_c
    assert (
        'ii42_am_semantic_builder_abort(&build_state);\n'
        '        ii42_index_free(&index);'
        in am_c
    )
    assert (
        'MemoryContextDelete(build_context);\n'
        '        ii42_am_release_unused_malloc();\n'
        '        PG_RE_THROW();'
        in am_c
    )


def test_accelerator_hot_path_avoids_select_fd_limit() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    assert '#include <poll.h>' in semantic_c
    assert 'FD_SETSIZE' not in semantic_c
    assert 'FD_SET' not in semantic_c
    assert 'select(' not in semantic_c
    assert 'handle->accelerator_checkout_signature' in semantic_c


def test_accelerator_idle_reuse_obeys_connection_close() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    header = read_text('src/ii42_runtime_service.h')

    assert 'bool accelerator_response_keep_alive;' in header
    assert 'pg_strncasecmp(line, "Connection:", 11)' in semantic_c
    assert 'strstr(connection, "close") != NULL' in semantic_c
    assert 'handle->accelerator_response_keep_alive = keep_alive;' in semantic_c
    assert 'ii42_runtime_accelerator_idle_connection_reusable' in semantic_c
    assert 'II42_RUNTIME_ACCELERATOR_IDLE_CONNECTIONS 32' in semantic_c
    assert (
        'II42_RUNTIME_ACCELERATOR_IDLE_CONNECTION_MAX_AGE_MS 30000'
        in semantic_c
    )
    assert 'MSG_PEEK' in semantic_c
    assert 'fcntl(fd, F_SETFL, flags | O_NONBLOCK)' in semantic_c
    assert (
        'if (errno == EAGAIN || errno == EWOULDBLOCK)\n'
        '            {\n'
        '                *complete_out =\n'
        '                    ii42_runtime_accelerator_http_response_complete'
        '(handle);\n'
        '                return true;'
        in semantic_c
    )
    assert (
        'if (handle->accelerator_response_keep_alive)\n'
        '                {\n'
        '                    ii42_runtime_accelerator_release_idle_connection'
        in semantic_c
    )


def test_runtime_server_bounds_keepalive_connections() -> None:
    server = read_text('src/ii42_runtime_server.cc')

    assert 'DEFAULT_CONNECTION_IDLE_TIMEOUT_S = 30' in server
    assert 'MAX_CONNECTIONS = 4096' in server
    assert 'SO_RCVTIMEO' in server
    assert 'SO_SNDTIMEO' in server
    assert 'struct ActiveConnectionGuard' in server
    assert 'active_connections->fetch_add(' in server
    assert 'previous_connections >= max_connections' in server
    assert '"connection capacity exhausted"' in server
    assert '--max-connections' in server
    assert '--connection-idle-timeout-s' in server


def test_accelerator_request_timeout_uses_liveness_guard() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    assert 'II42_RUNTIME_ACCELERATOR_TIMEOUT_SECONDS' not in semantic_c
    assert '#define II42_RUNTIME_ACCELERATOR_SOCKET_IO_TIMEOUT_MS 30000' in (
        semantic_c
    )
    assert (
        'ii42_runtime_accelerator_request_timeout_ms(void)\n'
        '{\n'
        '    return ii42_runtime_liveness_timeout_ms;\n'
        '}'
        in semantic_c
    )
    assert 'II42_RUNTIME_ACCELERATOR_STALE_INFLIGHT_MULTIPLIER 4' in semantic_c
    assert 'II42_RUNTIME_ACCELERATOR_STALE_INFLIGHT_MS' not in semantic_c
    assert 'return elapsed_ms >= (double) timeout_ms;' in semantic_c


def test_successful_accelerator_request_failures_backoff_softly() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    assert 'II42_RUNTIME_ACCELERATOR_HEALTHY_FAILURE_SOFT_LIMIT 2' in (
        semantic_c
    )
    assert 'ii42_runtime_accelerator_failure_backoff_ms' in semantic_c
    assert (
        'entry->consecutive_failures <\n'
        '            II42_RUNTIME_ACCELERATOR_HEALTHY_FAILURE_SOFT_LIMIT'
        in semantic_c
    )
    assert (
        'II42_RUNTIME_ACCELERATOR_HEALTHY_FAILURE_SOFT_LIMIT)\n'
        '    {\n'
        '        return 0;\n'
        '    }'
        in semantic_c
    )
    request_failure_start = semantic_c.index(
        'ii42_runtime_accelerator_note_request_failure('
    )
    request_failure_end = semantic_c.index(
        '\nstatic void\nii42_runtime_accelerator_note_backpressure',
        request_failure_start
    )
    request_failure_body = semantic_c[
        request_failure_start:request_failure_end
    ]
    assert 'entry->successes > 0' not in request_failure_body
    assert 'entry->consecutive_failures++' in request_failure_body
    assert 'entry->next_probe_at = GetCurrentTimestamp()' in (
        request_failure_body
    )


def test_async_accelerator_failover_has_terminal_deadline() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    start = semantic_c.index(
        'ii42_runtime_service_failover_accelerator_async('
    )
    end = semantic_c.index(
        '\nstatic char *\n'
        'ii42_runtime_service_wait_accelerator',
        start,
    )
    body = semantic_c[start:end]

    assert 'deadline_reached = handle->accelerator_deadline_at > 0' in body
    assert 'ii42_runtime_service_enqueue_local_failover_async(handle)' in body
    assert 'if (deadline_reached)' in body
    assert (
        'handle->accelerator_local_fallback_deadline_at == 0'
        in body
    )
    assert (
        'II42_RUNTIME_ACCELERATOR_LOCAL_FAILOVER_WAIT_MS * 1000'
        in body
    )
    assert (
        'now < handle->accelerator_local_fallback_deadline_at'
        in body
    )
    assert 'handle->accelerator_retry_after = now +' in body
    assert 'ii42_runtime_request_clear_accelerator(handle, false);' in body
    assert 'handle->active = false;' in body
    assert (
        '#define II42_RUNTIME_ACCELERATOR_LOCAL_FAILOVER_WAIT_MS'
        in semantic_c
    )


def test_accelerator_weight_is_slot_capacity() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    assert 'ii42_runtime_accelerator_default_remote_capacity' in semantic_c
    assert 'capacity = Max(target->weight, 1);' in semantic_c
    assert 'pipeline_depth * target_weight' not in semantic_c
    assert 'total_weight' not in semantic_c


def test_accelerator_target_json_has_bounded_process_cache() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    start = semantic_c.index(
        'ii42_runtime_accelerator_targets(\n'
    )
    end = semantic_c.index(
        '\nstatic bool\n'
        'ii42_runtime_accelerator_parse_url',
        start,
    )
    body = semantic_c[start:end]

    assert 'TopMemoryContext' in body
    assert '"ii42 accelerator target cache"' in body
    assert 'strcmp(cached_config, ii42_runtime_accelerators) != 0' in body
    assert 'MemoryContextReset(cache_context);' in body
    assert body.count('ii42_runtime_accelerator_targets_parse(') == 1
    assert 'MemoryContextStrdup(\n            caller_context,' in body


def test_accelerator_url_port_validation_precedes_free() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    start = semantic_c.index(
        'ii42_runtime_accelerator_parse_url(\n'
    )
    end = semantic_c.index(
        '\nstatic bool\n'
        'ii42_runtime_accelerator_idle_connection_reusable',
        start,
    )
    body = semantic_c[start:end]

    validation = body.index(
        "valid_port = errno == 0 && endptr != NULL && *endptr == '\\0'"
    )
    release = body.index('pfree(port_text);', validation)
    rejection = body.index('if (!valid_port)', release)

    assert validation < release < rejection
    assert '*endptr' not in body[release:]


def test_prefix_batch_scheduling_is_target_aware() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    am_c = read_text('src/ii42_am.c')

    assert 'ii42_runtime_accelerator_start_document_prefix_batch' in semantic_c
    assert 'ii42_runtime_accelerator_candidate_batch_size' in semantic_c
    assert 'best_batch_count' in semantic_c
    assert 'ii42_runtime_service_document_response_slots_locked();' in (
        semantic_c
    )
    assert 'ii42_runtime_accelerator_local_score(batch_count) == DBL_MAX' in (
        semantic_c
    )
    assert 'if (submitted_count == 0)' in am_c
    assert (
        'ii42_runtime_service_recommended_document_batch_size();\n'
        '    batch_count = Min('
        not in semantic_c
    )
    start_remote = semantic_c[
        semantic_c.index(
            'ii42_runtime_accelerator_start_document_batch_internal('
        ):semantic_c.index(
            '\nstatic int\n'
            'ii42_runtime_accelerator_start_document_batch',
            semantic_c.index(
                'ii42_runtime_accelerator_start_document_batch_internal('
            )
        )
    ]
    assert 'ii42_runtime_accelerator_local_score(' not in start_remote
    assert (
        'if (index == count)\n'
        '            {\n'
        '                continue;\n'
        '            }'
        in start_remote
    )


def test_sql_product_path_passes_index_precision_explicitly() -> None:
    sql = current_sql()

    assert 'runtime_precision text := \'fp16\';' in sql
    assert 'runtime_precision := runtime_config#>>\'{index,runtime_precision}\';' in sql
    assert (
        'onnx_result := ii42_runtime_service_query_atoms(\n'
        '            runtime_config->>\'model_path\',\n'
        '            runtime_precision,\n'
        '            input_text'
        in sql
    )
    assert (
        'onnx_result := ii42_runtime_service_document_atoms_batch(\n'
        '            runtime_config->>\'model_path\',\n'
        '            runtime_precision,\n'
        '            input_texts[chunk_start:chunk_end]'
        in sql
    )
    assert (
        'ii42_runtime_service_atoms_batch_internal(\n'
        '    model_path text,\n'
        '    runtime_precision text,'
        in sql
    )


def test_document_batch_uses_accelerated_bounded_pipeline() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    sql = current_sql()
    start = semantic_c.index(
        '\nii42_runtime_service_wait_document_pipeline('
    )
    end = semantic_c.index(
        '\nstatic char *\nii42_runtime_service_wait_local_document_chunks(',
        start,
    )
    pipeline = semantic_c[start:end]
    sql_start = sql.index(
        'CREATE FUNCTION ii42_encode_document_batch_internal('
    )
    sql_end = sql.index(
        '\nCOMMENT ON FUNCTION ii42_encode_document_batch_internal(',
        sql_start,
    )
    batch_encoder = sql[sql_start:sql_end]

    assert 'ii42_runtime_effective_document_pipeline_depth()' in pipeline
    assert (
        'ii42_runtime_service_submit_document_prefix_checkout_async('
        in pipeline
    )
    assert 'ii42_runtime_service_wait_async(handle)' in pipeline
    assert 'expected_counts[head]' in pipeline
    assert 'max_batch_size := 512;' in batch_encoder
    assert 'max_batch_bytes int4 := 1048575;' in batch_encoder
    assert 'max_batch_bytes int4 := 65535;' not in batch_encoder
    assert "current_setting('ii42.runtime_max_batch_size'" not in batch_encoder


def test_runtime_queue_session_and_response_are_precision_keyed() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    header = read_text('src/ii42_runtime_service.h')

    assert 'II42_RUNTIME_SERVICE_VERSION 26' in header
    assert 'char runtime_precision[II42_RUNTIME_SERVICE_PRECISION_MAX_BYTES];' in header
    assert 'entry->runtime_precision, runtime_precision' in semantic_c
    assert (
        'ii42_ort_require_runtime_precision('
        'active_provider, runtime_precision'
        ')' in semantic_c
    )
    assert 'runtime_parameters.provider' not in semantic_c
    assert 'requested_provider = pstrdup("auto");' in semantic_c
    assert 'ii42_ort_auto_provider(' in semantic_c
    assert '"TensorrtExecutionProvider"' in semantic_c
    assert '"trt_fp16_enable"' in semantic_c
    assert '"AllowLowPrecisionAccumulationOnGPU"' in semantic_c
    assert (
        '\\"fp16_fallback_order\\":[\\"tensorrt\\",\\"coreml\\",'
        '\\"cuda\\",\\"cpu\\"]'
        in semantic_c
    )
    assert (
        '\\"fp32_fallback_order\\":[\\"cuda\\",\\"tensorrt\\",'
        '\\"coreml\\",\\"cpu\\"]'
        in semantic_c
    )
    assert 'ii42 fp16 inference requires a fp16-capable provider' not in (
        semantic_c
    )
    assert 'appendStringInfoString(&json, "},\\"runtime_precision\\":");' in semantic_c


def test_runtime_server_requires_matching_precision() -> None:
    server_cc = read_text('src/ii42_runtime_server.cc')

    assert 'std::string runtime_precision = "fp16";' in server_cc
    assert 'model_.runtime_precision' in server_cc
    assert 'expected_runtime_precision' in server_cc
    assert '"runtime_precision"' in server_cc
    assert 'runtime precision mismatch' in server_cc


def test_onnxruntime_telemetry_is_disabled_before_initialization() -> None:
    semantic_c = read_text('src/ii42_semantic.c')
    server_cc = read_text('src/ii42_runtime_server.cc')
    entrypoint = read_text(
        'packaging/docker/runtime-gpu-aarch64/'
        'evoke-gpu-runtime-entrypoint'
    )

    api_start = semantic_c.index('ii42_ort_api(void)')
    api_end = semantic_c.index(
        '\nstatic void\nii42_ort_cache_release_entry',
        api_start,
    )
    api_body = semantic_c[api_start:api_end]
    server_start = server_cc.index('const OrtApi *ort_api()')
    server_end = server_cc.index('\nvoid ort_check(', server_start)
    server_body = server_cc[server_start:server_end]

    assert 'ii42_ort_prepare_process();' in api_body
    assert api_body.index('ii42_ort_prepare_process();') < api_body.index(
        'OrtGetApiBase()'
    )
    assert 'setenv("ORT_DISABLE_TELEMETRY", "1", 0)' in semantic_c
    assert server_body.index('ORT_DISABLE_TELEMETRY') < server_body.index(
        'OrtGetApiBase()'
    )
    assert (
        'export ORT_DISABLE_TELEMETRY=${ORT_DISABLE_TELEMETRY:-1}'
        in entrypoint
    )


def test_runtime_server_allows_overlong_semantic_inputs() -> None:
    server_cc = read_text('src/ii42_runtime_server.cc')

    assert 'constexpr size_t RUNTIME_TRANSPORT_MAX_BYTES = 1U << 20;' in (
        server_cc
    )
    assert 'constexpr size_t DEFAULT_MAX_REQUEST_BYTES = 8U << 20;' in (
        server_cc
    )
    assert 'constexpr size_t RUNTIME_TRANSPORT_MAX_BYTES = 65535;' not in (
        server_cc
    )


def test_remote_required_runtime_submit_waits_for_capacity() -> None:
    semantic_c = read_text('src/ii42_semantic.c')

    assert (
        '#define II42_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT 600'
        in semantic_c
    )
    assert (
        '#define II42_RUNTIME_ACCELERATOR_REQUIRED_RETRY_WAIT_MS 100'
        in semantic_c
    )
    assert (
        'ii42_runtime_accelerator_wait_required_retry(uint32 *waited_ms)'
        in semantic_c
    )
    assert (
        'retry_limit = II42_RUNTIME_ACCELERATOR_REQUIRED_WAIT_RETRY_LIMIT;'
        in semantic_c
    )
    assert semantic_c.count(
        'ii42_runtime_accelerator_wait_required_retry('
    ) >= 4
    assert '"no ii42 document runtime accelerator is available"' in semantic_c


def test_semantic_tokenizers_clip_overlong_inputs() -> None:
    runtime_c = read_text('src/ii42_p2_runtime.c')
    server_cc = read_text('src/ii42_runtime_server.cc')

    assert 'P2 semantic input exceeds the configured window limit' not in (
        runtime_c
    )
    assert 'semantic input exceeds the configured window limit' not in (
        server_cc
    )
    assert 'static bool\nii42_p2_token_builder_append' in runtime_c
    assert 'if (!ii42_p2_append_piece(' in runtime_c
    assert 'bool append_piece(' in server_cc
    assert 'if (!append_piece(' in server_cc


def test_large_rebuilds_publish_bounded_initial_fold_groups() -> None:
    segments_h = read_text('src/ii42_segments.h')
    am_build_c = read_text('src/ii42_am_build.c')
    segment_pages_h = read_text('src/ii42_segment_pages.h')
    segment_pages_c = read_text('src/ii42_segment_pages.c')

    assert (
        '#define II42_TERM_DIRECTORY_LEGACY_MAX_EXTENTS_PER_TERM UINT32_C(32)'
        in segments_h
    )
    assert (
        '#define II42_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM UINT32_C(64)'
        in segments_h
    )
    assert 'ii42_segment_payload_partition_contiguous(' not in am_build_c
    assert 'ii42_initial_fold_stream_create(' in am_build_c
    assert (
        'ii42_segment_pages_write_streamed_initial_folded_bundle_fork('
        in am_build_c
    )
    assert '#define II42_INITIAL_FOLD_TARGET_BYTES' in segment_pages_h
    assert 'ii42_segment_pages_write_streamed_initial_folds(' in (
        segment_pages_c
    )


def test_query_term_plan_capacity_tracks_extent_cap() -> None:
    segment_pages_h = read_text('src/ii42_segment_pages.h')
    segment_pages_c = read_text('src/ii42_segment_pages.c')

    assert (
        '#define II42_SEGMENT_QUERY_TERM_FOLD_MAX_RUNS UINT32_C(4)'
        in segment_pages_h
    )
    assert (
        '(II42_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM + \\\n'
        '     II42_SEGMENT_QUERY_TERM_FOLD_MAX_RUNS)'
        in segment_pages_h
    )
    assert (
        'query plans must hold every raw term extent and fold run'
        in segment_pages_c
    )


def test_term_directory_objects_use_compact_variable_extent_stride() -> None:
    segments_c = read_text('src/ii42_segments.c')
    cow_c = read_text('src/ii42_term_cow.c')

    assert (
        'max_extents_per_term == 0 ||\n'
        '        max_extents_per_term >\n'
        '            II42_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM'
        in segments_c
    )
    assert 'stored_max_extents_per_term = extents_size / extent_stride;' in (
        cow_c
    )
    assert (
        'stored_max_extents_per_term >\n'
        '                II42_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM'
        in cow_c
    )
    assert (
        'extent_index < stored_max_extents_per_term;'
        in cow_c
    )
    assert 'ii42_term_cow_leaf_stored_extent_count(' in cow_c
    assert (
        '            stored_extent_count *\n'
        '            II42_TERM_COW_EXTENT_SIZE'
        in cow_c
    )
    assert (
        '                         stored_extent_count +\n'
        '                     extent_index) * II42_TERM_COW_EXTENT_SIZE'
        in cow_c
    )
