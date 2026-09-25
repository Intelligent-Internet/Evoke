from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests' / 'fixtures' / 'cq3_pubmed_full'
TRACE_SOURCE = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding='utf-8')


def test_run_fixture_records_complete_hits_and_memory() -> None:
    source = fixture('run.sql')
    normal_probe = fixture('normal_probe.sql')
    setup = fixture('setup.sql')

    assert "SET application_name = :'route';" in source
    assert 'generate_series(1, 10)' in source
    assert '\\gexec' in source
    assert 'cq3_full_capture_probe' in source
    assert 'cq3_full_capture_hits' in source
    assert 'CREATE OR REPLACE FUNCTION bench.cq3_full_capture_probe' in setup
    assert 'CREATE OR REPLACE FUNCTION bench.cq3_full_capture_hits' in setup
    assert 'END;\n$function$;' in setup
    assert 'SELECT count(*)::int4' in setup
    assert 'FROM pg_backend_memory_contexts' in setup
    assert "pg_read_file('/proc/self/status')" in setup
    assert "'VmRSS:[[:space:]]+([0-9]+) kB'" in setup
    assert 'evoke_query_trace_internal()' in setup
    assert (
        "trace->>'accelerator_forward_postings_examined'" in normal_probe
    )
    assert "VALUES ('unfiltered', NULL);" in setup
    assert 'CREATE UNLOGGED TABLE' not in setup
    assert "index_persistence <> 'p'" in setup
    assert "heap_persistence <> 'p'" in setup
    assert "generation_status->>'docs'" in setup
    assert "generation_status#>>'{primary,segment_count}'" in setup
    assert "generation_status#>>'{posting,record_count}'" in setup


def test_compare_uses_only_exported_trace_fields() -> None:
    source = fixture('compare.sql')
    trace_fields = (
        'memory_bytes',
        'filtered_bmp_allowed_blocks',
        'filtered_bmp_allowed_superblocks',
        'filtered_bmp_matching_ref_count',
        'filtered_bmp_matching_super_ref_count',
        'semantic_bmp_ref_reads',
        'semantic_bmp_record_reads',
        'blocks_considered',
        'blocks_scored',
        'blocks_skipped',
        'semantic_bmp_postings_examined',
        'postings_examined',
        'documents_examined',
        'semantic_bmp_query_bytes',
        'query_route',
        'accelerator_fallback',
        'semantic_bmp_direct_flat_filtered',
    )

    for field in trace_fields:
        assert f"trace->>'{field}'" in source
        assert f'\\"{field}\\"' in TRACE_SOURCE
    assert "trace->>'topk_complete'" not in source
    assert 'bool_and(hit_count = 50)' in source


def test_compare_requires_bit_identical_scores() -> None:
    source = fixture('compare.sql')

    assert 'baseline.doc_ord IS DISTINCT FROM candidate.doc_ord' in source
    assert 'float8send(baseline.score) IS DISTINCT FROM' in source
    assert 'float8send(candidate.score)' in source
    assert "baseline.route = :'baseline_route'" in source
    assert "candidate.route = :'candidate_route'" in source
    assert "RAISE EXCEPTION" in source
    assert '\\q' not in source


def test_compare_requires_selective_work_reduction_and_memory_plateau() -> None:
    source = fixture('compare.sql')

    assert "selective.filter_name = 'p006'" in source
    assert 'selective.bound_refs < unfiltered.bound_refs' in source
    assert 'selective.semantic_postings + selective.page_native_postings <' in (
        source
    )
    assert 'unfiltered.semantic_postings + unfiltered.page_native_postings' in (
        source
    )
    assert 'attempt >= 6' in source
    assert 'memory_range <= 16 * 1024 * 1024' in source
    assert 'rss_range <= 64 * 1024 * 1024' in source


def test_compare_requires_bounded_filtered_exact_work() -> None:
    source = fixture('compare.sql')

    assert "results.filter_name IN ('p006', 'p100', 'p500')" in source
    assert 'query_bytes <= 64 * 1024 * 1024' in source
    assert "query_route' = 'semantic_bmp'" in source
    assert 'NOT accelerator_fallback' in source
    assert 'direct_flat_filtered' in source
    assert 'blocks_considered <= allowed_blocks' in source
    assert 'blocks_scored <= allowed_blocks' in source
    assert 'documents_examined <= allowed_documents' in source
    assert 'filtered_work_bounded' in source


def test_normal_probe_is_bounded_and_diagnostic_only() -> None:
    source = fixture('normal_probe.sql')

    assert 'SET evoke.test_disable_semantic_accelerator = off;' in source
    assert 'SET evoke.test_force_semantic_bmp = off;' in source
    assert "'evoke.test_filtered_forward_route'" in source
    assert ":'forward_route'" in source
    assert 'FROM bench.cq3_full_filter_sets' in source
    assert 'WHERE name IN' not in source
    assert 'generate_series(1, 5)' in source
    assert '\\gexec' in source
    assert "trace->>'query_route'" in source
    for field in (
        'document_block_reads',
        'positive_document_count',
        'zero_score_documents_added',
        'zero_score_cow_objects_loaded',
        'zero_score_cow_records_examined',
        'zero_score_heap_peak',
    ):
        assert f"trace->>'{field}'" in source
        assert f'\\"{field}\\"' in TRACE_SOURCE
    assert "exact.route = :'candidate_route'" in source
    assert 'shared_hits' in source
    assert "'suite', 'cq3_normal_route_probe'" in source
    assert "'forward_route'" in source
    assert "'summaries'" in source
    assert "'overlap'" in source
    assert 'qualification failed' not in source
    setup = fixture('setup.sql')
    assert "RAISE EXCEPTION 'could not read backend VmRSS'" in setup


def test_postbuild_runtime_migration_is_bounded_and_repeatable() -> None:
    source = fixture('postbuild_equivalence.sql')

    assert "('old'::text, 1" in source
    assert "('old'::text, 2" in source
    assert "('new'::text, 1" in source
    assert "('new'::text, 2" in source
    assert 'membership_diff = 0' in source
    assert 'max_rank_displacement <= 1' in source
    assert 'max_relative_score_delta <= 0.002' in source
    assert 'old_replay_rank_diff = 0' in source
    assert 'old_replay_score_diff = 0' in source
    assert 'new_replay_rank_diff = 0' in source
    assert 'new_replay_score_diff = 0' in source
    assert 'runtime migration is unstable' in source
