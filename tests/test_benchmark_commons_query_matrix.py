from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / 'scripts/benchmark_commons_query_matrix.py'


def load_module():
    spec = importlib.util.spec_from_file_location(
        'benchmark_commons_query_matrix',
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def valid_generated_columns(module):
    return {
        column_name: {
            'type': expected['type'],
            'generated': 's',
            'expression': expected['expression'],
        }
        for column_name, expected in
        module.REQUIRED_GENERATED_COLUMNS.items()
    }


def test_matrix_contains_every_commons_query_shape() -> None:
    module = load_module()
    cases = module.build_cases(
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        module.CA_STRESS_DOCUMENT_ID,
    )
    names = {case.name for case in cases}

    assert names == {
        'arxiv_unfiltered',
        'arxiv_date_category',
        'arxiv_broad_date',
        'arxiv_organization_selective',
        'arxiv_organization_broad',
        'pubmed_unfiltered',
        'pubmed_date_journal',
        'pubmed_broad_date',
        'pubmed_partial_date',
        'pubmed_date_category',
        'ca_policy_union',
        'ca_chunk_scope_ordinary',
        'tx_policy_union',
        'tx_chunk_scope_ordinary',
        'wa_policy_union',
        'wa_chunk_scope_ordinary',
        'ca_chunk_scope_stress',
        'system_chunk_scope',
    }
    assert all("'hit_signature'" in case.sql for case in cases)
    chunk_scope = next(
        case for case in cases if case.name == 'ca_chunk_scope_ordinary'
    )
    assert 'IS DISTINCT FROM sample.document_id' in chunk_scope.sql


def test_pubmed_index_resolution_uses_only_the_stable_product_name(
    monkeypatch,
) -> None:
    module = load_module()

    def fake_query_json(_args, query):
        assert 'field_aware_bm25_idx' in query
        assert 'current_next_idx' not in query
        return {
            'stable': (
                'commons.'
                'data_pubmed__title_abstract__field_aware_bm25_idx'
            ),
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    resolved = module.resolve_pubmed_index(
        SimpleNamespace(pubmed_index=None),
    )

    assert resolved == (
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        'stable',
    )


def test_pubmed_index_resolution_rejects_a_missing_stable_root(
    monkeypatch,
) -> None:
    module = load_module()
    monkeypatch.setattr(
        module,
        'query_json',
        lambda _args, _query: {'stable': None},
    )

    with pytest.raises(RuntimeError, match='stable Commons PubMed'):
        module.resolve_pubmed_index(SimpleNamespace(pubmed_index=None))


def test_matrix_uses_exact_subset_filter_contract() -> None:
    module = load_module()
    cases = module.build_cases(
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        module.CA_STRESS_DOCUMENT_ID,
    )
    partial_date = next(
        case for case in cases if case.name == 'pubmed_partial_date'
    )
    category = next(
        case for case in cases if case.name == 'pubmed_date_category'
    )

    assert 'publish_date_end_bound' in partial_date.sql
    assert 'publish_date_start_bound' in partial_date.sql
    assert 'source.publish_date_end_bound < 20240101' in partial_date.sql
    assert "unnest(source.categories)" in category.sql
    assert "'violations'" in category.sql
    assert category.oracle_sql is not None
    assert 'count(*)::bigint' in category.oracle_sql
    assert 'NOT EXISTS' in category.oracle_sql
    assert module.PUBMED_INDEX_VISIBILITY_SQL in category.oracle_sql
    assert category.rank_oracle_sql is not None
    assert 'array_agg(source.ctid ORDER BY source.ctid)' in (
        category.rank_oracle_sql
    )
    assert 'allowed.tids' in category.rank_oracle_sql
    assert module.PUBMED_INDEX_VISIBILITY_SQL in category.rank_oracle_sql
    assert partial_date.planner_sql is not None
    assert 'EXPLAIN (FORMAT JSON, COSTS true, SETTINGS true)' in (
        partial_date.planner_sql
    )
    assert 'publish_date_end_bound' in partial_date.planner_sql
    assert module.PUBMED_INDEX_VISIBILITY_SQL in partial_date.planner_sql
    assert category.planner_sql is not None
    assert 'unnest(source.categories)' in category.planner_sql


def test_result_validation_treats_null_predicates_as_violations() -> None:
    module = load_module()
    case = module.timed_search_case(
        name='null_safe_filter',
        search_sql='evoke_query(1::regclass, \'query\', 10)',
        table_name='commons.data_pubmed',
        violation_sql='source.publish_date < 20240101',
        expected_max_ms=100.0,
    )

    assert '(source.publish_date < 20240101) IS NOT FALSE' in case.sql


def test_serial_matrix_covers_representative_rank_oracles() -> None:
    module = load_module()
    cases = module.build_cases(
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        module.CA_STRESS_DOCUMENT_ID,
    )
    observed = {
        case.name
        for case in cases
        if case.rank_oracle_sql is not None
    }

    assert observed == {
        'arxiv_date_category',
        'ca_chunk_scope_stress',
        'pubmed_date_category',
        'pubmed_date_journal',
        'system_chunk_scope',
    }


def test_rank_oracle_rejects_result_mismatch() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    signature = [['(0,1)', 1, '3f800000']]
    summary = module.summarize_attempts(
        case,
        [{
            'attempt': 1,
            'status': 'completed',
            'elapsed_ms': 80.0,
            'violations': 0,
            'hit_signature': signature,
            'rank_oracle': {
                'allowed_documents': 1,
                'hit_signature': [['(0,2)', 2, '3f800000']],
            },
        }],
    )

    assert summary['rank_oracle']['passed'] is False
    assert summary['rank_oracle']['attempts'] == [{
        'attempt': 1,
        'allowed_documents': 1,
        'passed': False,
    }]
    assert summary['passed'] is False


def test_requested_rank_oracle_cannot_disappear() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    summary = module.summarize_attempts(
        case,
        [
            {
                'attempt': 1,
                'status': 'completed',
                'elapsed_ms': 80.0,
                'violations': 0,
                'hit_signature': [],
            },
            {
                'attempt': 2,
                'status': 'completed',
                'elapsed_ms': 70.0,
                'violations': 0,
                'hit_signature': [],
            },
        ],
        rank_oracle_required=True,
    )

    assert summary['rank_oracle']['passed'] is False
    assert summary['passed'] is False


def test_filter_oracle_rejects_native_membership_mismatch() -> None:
    module = load_module()
    case = module.BenchmarkCase(
        'example',
        'SELECT 1',
        100.0,
        'SELECT 2',
    )
    summary = module.summarize_attempts(
        case,
        [{
            'attempt': 1,
            'status': 'completed',
            'elapsed_ms': 80.0,
            'violations': 0,
            'evoke_trace': {'allowed_documents': '1'},
            'filter_oracle': {'allowed_documents': 2},
        }],
    )

    assert summary['filter_oracle'] == {
        'attempts': [{
            'attempt': 1,
            'sql_allowed_documents': 2,
            'native_allowed_documents': 1,
            'passed': False,
        }],
        'passed': False,
    }
    assert summary['passed'] is False

    no_result = module.summarize_attempts(
        case,
        [{'attempt': 1, 'status': 'statement_timeout'}],
    )
    assert no_result['filter_oracle']['passed'] is False


def test_relation_names_are_restricted() -> None:
    module = load_module()

    assert (
        module.regclass_literal('commons.data_arxiv')
        == "'commons.data_arxiv'::regclass"
    )
    with pytest.raises(ValueError, match='unsafe'):
        module.regclass_literal('commons.data_arxiv; DROP TABLE x')


def test_index_status_accepts_serviceable_stale_scope_metadata() -> None:
    module = load_module()
    current = {
        'health': 'ok',
        'valid': True,
        'runtime_contract_matches': True,
        'accelerator_state': 'ready',
        'forward_complete': True,
        'scope_present': True,
        'scope_version': 6,
        'scope_current': True,
    }

    assert module.index_status_qualified(current) is True
    assert module.index_status_qualified({
        **current,
        'accelerator_state': 'ready_baseline_delta',
        'scope_current': False,
    }) is True
    assert module.index_status_qualified({
        key: value
        for key, value in current.items()
        if key != 'scope_version'
    }) is False
    assert module.index_status_qualified({
        **current,
        'accelerator_state': 'building',
    }) is False
    assert module.index_status_qualified({
        **current,
        'scope_present': False,
        'scope_version': 0,
    }) is True


def test_summary_rejects_timeouts_and_slo_regressions() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    passing = module.summarize_attempts(
        case,
        [
            {
                'status': 'completed',
                'elapsed_ms': 120.0,
                'violations': 0,
                'hit_signature': [],
            },
            {
                'status': 'completed',
                'elapsed_ms': 80.0,
                'violations': 0,
                'hit_signature': [],
            },
        ],
    )
    timeout = module.summarize_attempts(
        case,
        [
            {
                'status': 'statement_timeout',
                'wall_ms': 1_000.0,
            },
        ],
    )

    assert passing['passed'] is True
    assert timeout['passed'] is False


def test_summary_rejects_result_drift_or_missing_signatures() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    common = {
        'status': 'completed',
        'elapsed_ms': 50.0,
        'violations': 0,
    }

    drift = module.summarize_attempts(
        case,
        [
            {**common, 'hit_signature': [['(0,1)', 1, '3f800000']]},
            {**common, 'hit_signature': [['(0,2)', 2, '3f800000']]},
        ],
    )
    missing = module.summarize_attempts(case, [common, common])

    assert drift['result_stability']['passed'] is False
    assert drift['passed'] is False
    assert missing['result_stability']['passed'] is False
    assert missing['passed'] is False


def test_summary_rejects_unbounded_warm_p95() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    attempts = [
        {
            'attempt': attempt,
            'status': 'completed',
            'elapsed_ms': elapsed_ms,
            'violations': 0,
        }
        for attempt, elapsed_ms in enumerate(
            (50.0, 50.0, 60.0, 250.0),
            start=1,
        )
    ]

    summary = module.summarize_attempts(case, attempts)

    assert summary['warm']['p50_ms'] == 60.0
    assert summary['warm']['p95_ms'] == 250.0
    assert summary['expected_p95_max_ms'] == 200.0
    assert summary['passed'] is False


def test_summary_rejects_unbounded_first_query() -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1', 100.0)
    summary = module.summarize_attempts(
        case,
        [
            {
                'attempt': 1,
                'status': 'completed',
                'elapsed_ms': 2_100.0,
                'violations': 0,
            },
            {
                'attempt': 2,
                'status': 'completed',
                'elapsed_ms': 80.0,
                'violations': 0,
            },
        ],
    )

    assert summary['first_ms'] == 2_100.0
    assert summary['expected_first_max_ms'] == 2_000.0
    assert summary['warm']['p50_ms'] == 80.0
    assert summary['passed'] is False


def test_runner_records_backend_local_query_trace(monkeypatch) -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1;', 100.0)
    args = SimpleNamespace(timeout_ms=1_000, repeats=1)
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            '{"elapsed_ms":12,"violations":0}\n'
            '{"evoke_trace":{"query_route":"forward_rows",'
            '"memory_bytes":"1048576"}}\n'
        ),
        stderr='',
    )

    monkeypatch.setattr(module, 'run_psql', lambda *args, **kwargs: completed)
    result = module.run_case(args, case)

    assert result['attempts'][0]['evoke_trace'] == {
        'query_route': 'forward_rows',
        'memory_bytes': '1048576',
    }


def test_runner_groups_concurrent_sessions_into_waves(monkeypatch) -> None:
    module = load_module()
    case = module.BenchmarkCase('example', 'SELECT 1;', 100.0)
    args = SimpleNamespace(
        timeout_ms=1_000,
        repeats=2,
        concurrency=4,
    )
    completed = SimpleNamespace(
        returncode=0,
        stdout=(
            '{"elapsed_ms":12,"violations":0}\n'
            '{"evoke_trace":{"query_route":"forward_rows"}}\n'
        ),
        stderr='',
    )

    monkeypatch.setattr(module, 'run_psql', lambda *args, **kwargs: completed)
    result = module.run_case(args, case)

    assert result['concurrency'] == 4
    assert result['completed'] == 8
    assert result['first']['count'] == 4
    assert result['warm']['count'] == 4
    assert {
        (attempt['wave'], attempt['client'])
        for attempt in result['attempts']
    } == {
        (wave, client)
        for wave in (1, 2)
        for client in range(1, 5)
    }


def test_psql_sessions_use_a_distinct_application_name(monkeypatch) -> None:
    module = load_module()
    args = SimpleNamespace(psql='psql', dsn='dbname=example')
    observed = {}

    def fake_run(*command, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', fake_run)
    module.run_psql(args, 'SELECT 1')

    assert observed['env']['PGAPPNAME'] == module.APPLICATION_NAME


def test_rank_oracle_is_separate_from_concurrent_qualification() -> None:
    module = load_module()
    args = SimpleNamespace(
        repeats=2,
        concurrency=4,
        timeout_ms=1_000,
        verify_rank_oracle=True,
    )

    with pytest.raises(ValueError, match='requires --concurrency 1'):
        module.validate_args(args)

    args.concurrency = 1
    module.validate_args(args)


def test_qualification_requires_first_and_warm_waves() -> None:
    module = load_module()
    args = SimpleNamespace(
        repeats=1,
        concurrency=1,
        timeout_ms=1_000,
        verify_rank_oracle=False,
        require_restart_within_seconds=None,
    )

    with pytest.raises(ValueError, match='first and warm waves'):
        module.validate_args(args)


def test_restart_window_must_be_positive() -> None:
    module = load_module()
    args = SimpleNamespace(
        repeats=2,
        concurrency=1,
        timeout_ms=1_000,
        verify_rank_oracle=False,
        require_restart_within_seconds=0,
    )

    with pytest.raises(ValueError, match='restart-within-seconds'):
        module.validate_args(args)

    args.require_restart_within_seconds = 601
    with pytest.raises(ValueError, match='cannot exceed 600'):
        module.validate_args(args)


def test_environment_records_planner_health(monkeypatch) -> None:
    module = load_module()
    captured = {}

    def fake_query_json(args, sql_text):
        captured.setdefault('sql', []).append(sql_text)
        if "json_build_object('value'," in sql_text:
            return {'value': module.CATALOG_CONTRACT}
        if 'evoke_onnxruntime_build_info' in sql_text:
            return {
                'build_info': 'enabled:api=29',
                'probe': 'available:linked:version=1.29.0',
            }
        if 'WITH source AS (' in sql_text:
            return {
                'generation_id': '1/2/3',
                'health': 'ok',
                'valid': True,
                'runtime_contract_matches': True,
                'accelerator_state': 'ready',
                'forward_complete': True,
                'builder_policy_id': 7,
                'scope_present': True,
                'scope_version': 6,
                'scope_current': True,
                'warm_expected': True,
                'shared_preload_available': True,
                'shared_preload_loading': False,
                'resident_fold_current': False,
                'resident_fold_loading': False,
                'query_metadata_warm': True,
            }
        return {
            'ok': True,
            'captured_at_epoch': 1_000.0,
            'postmaster_start_epoch': 950.0,
            'postmaster_age_seconds': 50.0,
            'settings': {
                'autovacuum': 'on',
                'evoke.maintenance_worker_limit': '4',
                'evoke.test_disable_semantic_accelerator': 'off',
                'evoke.test_filtered_forward_route': 'auto',
                'track_counts': 'on',
            },
            'table_stats': {
                table_name: {
                    'reltuples': 100.0,
                    'last_analyze': None,
                    'last_autoanalyze': None,
                }
                for table_name in module.REQUIRED_FILTER_STATS
            },
            'filter_stats': {
                table_name: sorted(columns)
                for table_name, columns in
                module.REQUIRED_FILTER_STATS.items()
            },
            'generated_columns': valid_generated_columns(module),
            'catalog_contract': {
                'installed_version': '0.2.4',
                'available_version': '0.2.4',
                'extension_schema': 'evoke_ext',
                'versions_match': True,
                'c_module_paths': ['$libdir/evoke'],
                'c_module_path_current': True,
                'required_functions': {
                    signature: True
                    for signature in module.REQUIRED_CATALOG_SIGNATURES
                },
            },
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    result = module.collect_environment(
        SimpleNamespace(require_restart_within_seconds=300),
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        'explicit',
    )

    assert result['ok'] is True
    assert result['catalog_qualified'] is True
    assert result['onnxruntime_qualified'] is True
    assert result['schema_qualified'] is True
    assert result['generation_qualified'] is True
    assert result['query_metadata_qualified'] is True
    assert result['query_warm_qualified'] is False
    assert result['planner_qualified'] is True
    assert result['runtime_qualified'] is True
    assert result['restart_requirement_seconds'] == 300
    assert result['restart_qualified'] is True
    assert result['planner_health'] == {
        'qualified': True,
        'reasons': [],
    }
    assert result['runtime_health'] == {
        'qualified': True,
        'reasons': [],
    }
    assert result['qualified'] is True
    assert len(result['indexes']) == 9
    assert all(
        status['query_warm_state'] == 'metadata_only'
        for status in result['indexes'].values()
    )
    environment_sql = captured['sql'][0]
    assert "'autovacuum'" in environment_sql
    assert 'pg_postmaster_start_time()' in environment_sql
    assert "'default_statistics_target'" in environment_sql
    assert "'evoke.test_filtered_forward_route'" in environment_sql
    assert "'track_counts'" in environment_sql
    assert "'reltuples'" in environment_sql
    assert "'filter_stats'" in environment_sql
    assert "'$libdir/evoke'" in environment_sql
    assert 'installed_procedure.probin' in environment_sql
    assert "'generated_columns'" in environment_sql
    assert 'attribute.attgenerated' in environment_sql
    assert 'pg_get_expr(' in environment_sql
    assert "'publish_date_end_bound'" in environment_sql
    assert "'publish_date_has_day'" in environment_sql
    assert "'nlm_ta'" in environment_sql
    assert "'policy_ca_doc_id'" in environment_sql
    assert 'pg_available_extensions' in environment_sql
    assert 'evoke_index_runtime_state_json(regclass)' in environment_sql
    assert 'evoke_query_trace_internal()' in environment_sql
    runtime_sql = next(
        sql for sql in captured['sql']
        if 'evoke_onnxruntime_build_info' in sql
    )
    assert '"evoke_ext"."evoke_onnxruntime_probe"' in runtime_sql
    index_status_sql = next(
        sql for sql in captured['sql']
        if 'WITH source AS (' in sql
    )
    assert 'evoke_index_runtime_state_json' in index_status_sql
    assert "'{semantic_accelerator,scope_present}'" in index_status_sql
    assert "'{semantic_accelerator,scope_version}'" in index_status_sql
    assert "'{semantic_accelerator,scope_current}'" in index_status_sql
    assert "'{shared_preload,query_metadata_warm}'" in index_status_sql
    assert "'{shared_preload,query_warm_pages}'" in index_status_sql


def test_planner_health_rejects_disabled_maintenance_and_missing_stats() -> None:
    module = load_module()
    environment = {
        'settings': {
            'autovacuum': 'off',
            'track_counts': 'on',
        },
        'table_stats': {
            table_name: {
                'reltuples': 100.0,
                'last_analyze': '2026-08-22T00:00:00Z',
                'last_autoanalyze': None,
            }
            for table_name in module.REQUIRED_FILTER_STATS
        },
        'filter_stats': {
            table_name: sorted(columns)
            for table_name, columns in module.REQUIRED_FILTER_STATS.items()
        },
    }
    environment['table_stats']['data_pubmed'] = {
        'reltuples': -1.0,
        'last_analyze': None,
        'last_autoanalyze': None,
    }
    environment['filter_stats']['data_arxiv'].remove('categories')

    result = module.evaluate_planner_health(environment)

    assert result['qualified'] is False
    assert result['reasons'] == [
        'autovacuum is not on',
        'data_arxiv missing pg_stats columns: categories',
        'data_pubmed cardinality is unknown',
    ]


def test_runtime_health_rejects_diagnostic_or_stalled_routes() -> None:
    module = load_module()

    assert module.evaluate_runtime_health({
        'settings': {
            'evoke.maintenance_worker_limit': '0',
            'evoke.test_disable_semantic_accelerator': 'on',
            'evoke.test_filtered_forward_route': 'transpose',
        },
    }) == {
        'qualified': False,
        'reasons': [
            'Evoke maintenance workers are disabled',
            'semantic accelerator test override is active',
            'filtered forward-route test override is active',
        ],
    }

    assert module.evaluate_runtime_health({
        'settings': {
            'evoke.maintenance_worker_limit': '4',
            'evoke.test_disable_semantic_accelerator': 'off',
            'evoke.test_filtered_forward_route': 'auto',
        },
    }) == {'qualified': True, 'reasons': []}

    assert module.evaluate_runtime_health({
        'settings': {
            'evoke.maintenance_worker_limit': '4',
        },
    }) == {
        'qualified': False,
        'reasons': [
            'semantic accelerator test control is unavailable',
            'filtered forward-route control is unavailable',
        ],
    }


def test_schema_health_rejects_wrong_generated_column_contract() -> None:
    module = load_module()
    generated_columns = valid_generated_columns(module)
    generated_columns['publish_date_start_bound']['type'] = 'bigint'
    generated_columns['publish_date_end_bound']['generated'] = ''
    generated_columns['publish_date_has_day']['expression'] = 'true'

    result = module.evaluate_schema_health({
        'generated_columns': generated_columns,
    })

    assert result == {
        'qualified': False,
        'reasons': [
            'publish_date_start_bound has the wrong type',
            'publish_date_end_bound is not stored generated',
            'publish_date_has_day has the wrong expression',
        ],
    }


def test_schema_health_rejects_missing_contract() -> None:
    module = load_module()

    assert module.evaluate_schema_health({}) == {
        'qualified': False,
        'reasons': ['PubMed generated-column contract is missing'],
    }


def test_query_warm_state_is_separate_from_generation() -> None:
    module = load_module()

    assert module.index_query_warm_state({
        'warm_expected': False,
    }) == 'not_requested'
    assert module.index_query_warm_state({
        'warm_expected': True,
        'shared_preload_available': True,
        'query_metadata_warm': True,
    }) == 'metadata_only'
    assert module.index_query_warm_state({
        'warm_expected': True,
        'resident_fold_current': True,
    }) == 'resident'
    assert module.index_query_warm_state({
        'warm_expected': True,
        'resident_fold_loading': True,
    }) == 'loading'
    assert module.index_query_warm_state({
        'warm_expected': True,
        'shared_preload_available': True,
    }) == 'cold'
    assert module.index_query_warm_state({
        'error': 'invalid serialized format',
    }) == 'error'
    assert module.index_query_metadata_qualified({
        'warm_expected': True,
        'query_metadata_warm': True,
    }) is True
    assert module.index_query_metadata_qualified({
        'warm_expected': True,
        'shared_preload_available': True,
    }) is False


def test_environment_requires_query_metadata_readiness(monkeypatch) -> None:
    module = load_module()

    def fake_query_json(args, sql_text):
        if "json_build_object('value'," in sql_text:
            return {'value': module.CATALOG_CONTRACT}
        if 'evoke_onnxruntime_build_info' in sql_text:
            return {
                'build_info': 'enabled:api=29',
                'probe': 'available:linked:version=1.29.0',
            }
        if 'WITH source AS (' in sql_text:
            return {
                'generation_id': '1/2/3',
                'health': 'ok',
                'valid': True,
                'runtime_contract_matches': True,
                'accelerator_state': 'ready',
                'forward_complete': True,
                'builder_policy_id': 7,
                'scope_present': True,
                'scope_version': 6,
                'scope_current': True,
                'warm_expected': True,
                'shared_preload_available': True,
                'shared_preload_loading': False,
                'resident_fold_current': False,
                'resident_fold_loading': False,
                'query_metadata_warm': False,
            }
        return {
            'settings': {
                'autovacuum': 'on',
                'evoke.maintenance_worker_limit': '4',
                'evoke.test_disable_semantic_accelerator': 'off',
                'evoke.test_filtered_forward_route': 'auto',
                'track_counts': 'on',
            },
            'table_stats': {
                table_name: {
                    'reltuples': 100.0,
                    'last_analyze': '2026-08-22T00:00:00Z',
                    'last_autoanalyze': None,
                }
                for table_name in module.REQUIRED_FILTER_STATS
            },
            'filter_stats': {
                table_name: sorted(columns)
                for table_name, columns in
                module.REQUIRED_FILTER_STATS.items()
            },
            'generated_columns': valid_generated_columns(module),
            'catalog_contract': {
                'installed_version': '0.2.4',
                'available_version': '0.2.4',
                'extension_schema': 'evoke_ext',
                'versions_match': True,
                'c_module_paths': ['$libdir/evoke'],
                'c_module_path_current': True,
                'required_functions': {
                    signature: True
                    for signature in module.REQUIRED_CATALOG_SIGNATURES
                },
            },
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    result = module.collect_environment(
        SimpleNamespace(),
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        'explicit',
    )

    assert result['generation_qualified'] is True
    assert result['schema_qualified'] is True
    assert result['planner_qualified'] is True
    assert result['runtime_qualified'] is True
    assert result['query_metadata_qualified'] is False
    assert result['qualified'] is False


def test_planner_capture_records_physical_nodes(monkeypatch) -> None:
    module = load_module()
    document = [{
        'Plan': {
            'Node Type': 'Bitmap Heap Scan',
            'Relation Name': 'data_arxiv',
            'Plan Rows': 2911,
            'Total Cost': 42.5,
            'Plans': [{
                'Node Type': 'Bitmap Index Scan',
                'Index Name': 'data_arxiv_categories_idx',
                'Index Cond': "categories && '{cs.LG}'::text[]",
            }],
        },
    }]
    monkeypatch.setattr(
        module,
        'run_psql',
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(document),
            stderr='',
        ),
    )
    cases = [
        module.BenchmarkCase(
            'filtered',
            'SELECT 1',
            500.0,
            planner_sql='EXPLAIN SELECT 1',
        ),
        module.BenchmarkCase('unfiltered', 'SELECT 1', 300.0),
    ]

    result = module.collect_planner_plans(SimpleNamespace(), cases)

    assert result['captured'] is True
    assert list(result['cases']) == ['filtered']
    assert result['cases']['filtered']['nodes'] == [
        {
            'Node Type': 'Bitmap Heap Scan',
            'Relation Name': 'data_arxiv',
            'Plan Rows': 2911,
            'Total Cost': 42.5,
        },
        {
            'Node Type': 'Bitmap Index Scan',
            'Index Name': 'data_arxiv_categories_idx',
            'Index Cond': "categories && '{cs.LG}'::text[]",
        },
    ]
    assert result['cases']['filtered']['raw'] == document


def test_environment_rejects_incomplete_catalog_contract(monkeypatch) -> None:
    module = load_module()

    def fake_query_json(args, sql_text):
        if 'WITH source AS (' in sql_text:
            return {
                'generation_id': '1/2/3',
                'health': 'ok',
                'valid': True,
                'runtime_contract_matches': True,
                'accelerator_state': 'ready',
                'forward_complete': True,
                'builder_policy_id': 7,
                'scope_present': True,
                'scope_version': 6,
                'scope_current': True,
            }
        return {
            'catalog_contract': {
                'installed_version': '0.2.4',
                'available_version': '0.2.4',
                'extension_schema': 'evoke_ext',
                'versions_match': True,
                'c_module_paths': ['$libdir/evoke'],
                'c_module_path_current': True,
                'required_functions': {
                    'evoke_index_generation_status_internal(regclass)': True,
                    'evoke_index_runtime_state_json(regclass)': True,
                    'evoke_query_trace_internal()': False,
                },
            },
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    result = module.collect_environment(
        SimpleNamespace(),
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        'explicit',
    )

    assert result['qualified'] is False
    assert result['catalog_qualified'] is False


@pytest.mark.parametrize(
    ('paths', 'current'),
    [
        (['/opt/evoke-old/lib/evoke.so'], False),
        (['$libdir/evoke', '/opt/evoke-old/lib/evoke.so'], False),
        (['/opt/evoke-old/lib/evoke.so'], True),
    ],
)
def test_catalog_contract_rejects_stale_c_module_binding(
    paths,
    current,
) -> None:
    module = load_module()
    contract = {
        'installed_version': '0.2.4',
        'available_version': '0.2.4',
        'extension_schema': 'evoke_ext',
        'versions_match': True,
        'c_module_paths': paths,
        'c_module_path_current': current,
        'catalog_identity': module.CATALOG_CONTRACT,
        'required_functions': {
            signature: True
            for signature in module.REQUIRED_CATALOG_SIGNATURES
        },
    }

    assert module.catalog_contract_qualified(contract) is False


def test_catalog_contract_rejects_stale_sql_identity() -> None:
    module = load_module()
    contract = {
        'installed_version': '0.2.4',
        'available_version': '0.2.4',
        'extension_schema': 'evoke_ext',
        'versions_match': True,
        'c_module_paths': ['$libdir/evoke'],
        'c_module_path_current': True,
        'catalog_identity': 'evoke_catalog_stale',
        'required_functions': {
            signature: True
            for signature in module.REQUIRED_CATALOG_SIGNATURES
        },
    }

    assert module.catalog_contract_qualified(contract) is False


def test_environment_records_stale_index_without_hiding_failure(
    monkeypatch,
) -> None:
    module = load_module()
    status_calls = 0

    def fake_query_json(args, sql_text):
        nonlocal status_calls
        if "json_build_object('value'," in sql_text:
            return {'value': module.CATALOG_CONTRACT}
        if 'evoke_onnxruntime_build_info' in sql_text:
            return {
                'build_info': 'enabled:api=29',
                'probe': 'available:linked:version=1.29.0',
            }
        if 'WITH source AS (' not in sql_text:
            return {
                'catalog_contract': {
                    'installed_version': '0.2.4',
                    'available_version': '0.2.4',
                    'extension_schema': 'evoke_ext',
                    'versions_match': True,
                    'c_module_paths': ['$libdir/evoke'],
                    'c_module_path_current': True,
                    'required_functions': {
                        signature: True
                        for signature in module.REQUIRED_CATALOG_SIGNATURES
                    },
                },
            }
        status_calls += 1
        if status_calls == 1:
            raise RuntimeError('invalid serialized format')
        return {
            'generation_id': '1/2/3',
            'health': 'ok',
            'valid': True,
            'runtime_contract_matches': True,
            'accelerator_state': 'ready',
            'forward_complete': True,
            'builder_policy_id': 7,
            'scope_present': True,
            'scope_version': 6,
            'scope_current': True,
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    result = module.collect_environment(
        SimpleNamespace(),
        'commons.data_pubmed__title_abstract__field_aware_bm25_idx',
        'explicit',
    )

    assert result['qualified'] is False
    assert sum('error' in status for status in result['indexes'].values()) == 1


def test_onnxruntime_contract_rejects_unpinned_runtime(monkeypatch) -> None:
    module = load_module()
    captured = {}

    def fake_query_json(args, sql_text):
        captured['sql'] = sql_text
        return {
            'build_info': 'enabled:api=26',
            'probe': 'available:linked:version=1.26.0',
        }

    monkeypatch.setattr(module, 'query_json', fake_query_json)
    result = module.collect_onnxruntime_contract(
        SimpleNamespace(),
        {'extension_schema': 'evoke_ext'},
    )

    assert result['qualified'] is False
    assert result['expected_version'] == '1.29.0'
    assert result['expected_build_info'] == 'enabled:api=29'
    assert '"evoke_ext"."evoke_onnxruntime_build_info"' in captured['sql']


def test_main_skips_queries_when_catalog_contract_is_incomplete(
    monkeypatch,
    tmp_path,
) -> None:
    module = load_module()
    output_path = tmp_path / 'matrix.json'
    args = SimpleNamespace(
        ca_stress_document_id='example',
        concurrency=1,
        repeats=2,
        only=[],
        output=output_path,
    )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'validate_args', lambda _args: None)
    monkeypatch.setattr(
        module,
        'resolve_pubmed_index',
        lambda _args: ('commons.pubmed_idx', 'explicit'),
    )
    monkeypatch.setattr(
        module,
        'build_cases',
        lambda _index, _document_id: [
            module.BenchmarkCase('must_not_run', 'SELECT 1', 1.0),
        ],
    )
    monkeypatch.setattr(
        module,
        'collect_environment',
        lambda *_args: {
            'catalog_qualified': False,
            'qualified': False,
        },
    )
    monkeypatch.setattr(
        module,
        'collect_planner_plans',
        lambda *_args: {'captured': True, 'cases': {}},
    )
    monkeypatch.setattr(
        module,
        'run_case',
        lambda *_args: pytest.fail('query ran with an incomplete catalog'),
    )
    monkeypatch.setattr(module, 'active_sessions', lambda _args: [])

    assert module.main() == 1
    payload = json.loads(output_path.read_text())
    assert payload['cases'] == []
    assert payload['passed'] is False


def test_main_skips_queries_when_query_environment_is_not_ready(
    monkeypatch,
    tmp_path,
) -> None:
    module = load_module()
    output_path = tmp_path / 'matrix.json'
    args = SimpleNamespace(
        ca_stress_document_id='example',
        concurrency=1,
        repeats=2,
        only=[],
        output=output_path,
    )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'validate_args', lambda _args: None)
    monkeypatch.setattr(
        module,
        'resolve_pubmed_index',
        lambda _args: ('commons.pubmed_idx', 'explicit'),
    )
    monkeypatch.setattr(
        module,
        'build_cases',
        lambda _index, _document_id: [
            module.BenchmarkCase('must_not_run', 'SELECT 1', 1.0),
        ],
    )
    monkeypatch.setattr(
        module,
        'collect_environment',
        lambda *_args: {
            'catalog_qualified': True,
            'generation_qualified': True,
            'query_metadata_qualified': False,
            'planner_qualified': True,
            'qualified': False,
        },
    )
    monkeypatch.setattr(
        module,
        'collect_planner_plans',
        lambda *_args: {'captured': True, 'cases': {}},
    )
    monkeypatch.setattr(
        module,
        'run_case',
        lambda *_args: pytest.fail('query ran before metadata was ready'),
    )
    monkeypatch.setattr(module, 'active_sessions', lambda _args: [])

    assert module.main() == 1
    payload = json.loads(output_path.read_text())
    assert payload['cases'] == []
    assert payload['passed'] is False


def test_plans_only_captures_evidence_without_qualifying(
    monkeypatch,
    tmp_path,
) -> None:
    module = load_module()
    output_path = tmp_path / 'plans.json'
    args = SimpleNamespace(
        ca_stress_document_id='example',
        concurrency=1,
        repeats=2,
        only=[],
        output=output_path,
        plans_only=True,
    )

    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(module, 'validate_args', lambda _args: None)
    monkeypatch.setattr(
        module,
        'resolve_pubmed_index',
        lambda _args: ('commons.pubmed_idx', 'explicit'),
    )
    monkeypatch.setattr(
        module,
        'build_cases',
        lambda _index, _document_id: [
            module.BenchmarkCase(
                'filtered',
                'SELECT 1',
                1.0,
                planner_sql='EXPLAIN SELECT 1',
            ),
        ],
    )
    monkeypatch.setattr(
        module,
        'collect_environment',
        lambda *_args: {
            'catalog_qualified': True,
            'qualified': False,
        },
    )
    monkeypatch.setattr(
        module,
        'collect_planner_plans',
        lambda *_args: {
            'captured': True,
            'cases': {'filtered': {'nodes': []}},
        },
    )
    monkeypatch.setattr(
        module,
        'run_case',
        lambda *_args: pytest.fail('plans-only mode ran a query case'),
    )
    monkeypatch.setattr(module, 'active_sessions', lambda _args: [])

    assert module.main() == 0
    payload = json.loads(output_path.read_text())
    assert payload['mode'] == 'plans_only'
    assert payload['capture_passed'] is True
    assert payload['passed'] is False
    assert payload['cases'] == []
