import importlib.util
import inspect
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / 'scripts' / 'rebuild_evoke_indexes.py'


def load_runner():
    spec = importlib.util.spec_from_file_location(
        'rebuild_evoke_indexes_tested',
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def semantic_index():
    return {
        'schema_name': 'search',
        'index_name': 'documents_idx',
        'index_regclass': 'search.documents_idx',
        'table_schema_name': 'search',
        'table_name': 'documents',
        'table_regclass': 'search.documents',
        'access_method': 'evoke',
        'indexdef': (
            'CREATE INDEX documents_idx ON search.documents USING evoke '
            '(body) WITH (method=lucene, idf_method=lucene, k1=1.5, '
            'b=.75, delta=.5, field_aware=true, sae=true, '
            "consistency='realtime', runtime_precision='fp32', "
            "semantic_impact_precision='fp16', "
            "semantic_alpha_mass=0.5, model='milestone', "
            "scoring_profile='p2')"
        ),
    }


def legacy_beta_index(*, sae: bool = True):
    idx = semantic_index()
    reloptions = (
        'sae=true, consistency=\'eventual\', runtime_precision=\'fp16\''
        if sae
        else 'sae=false, consistency=\'eventual\''
    )
    idx.update(
        {
            'access_method': 'ii42',
            'indexdef': (
                'CREATE INDEX documents_idx ON search.documents USING ii42 '
                f'(body) WITH ({reloptions})'
            ),
        }
    )
    return idx


def test_semantic_rebuild_preserves_product_reloptions():
    runner = load_runner()

    result = runner.reloptions_sql(
        semantic_index(),
        semantic=True,
        runtime_precision='fp16',
    )

    for expected in (
        'method = lucene',
        'idf_method = lucene',
        'k1 = 1.5',
        'b = .75',
        'delta = .5',
        'field_aware = true',
        "semantic_impact_precision = 'fp16'",
        'semantic_alpha_mass = 0.5',
        "model = 'milestone'",
        "scoring_profile = 'p2'",
        'sae = true',
        "consistency = 'eventual'",
        "runtime_precision = 'fp16'",
    ):
        assert expected in result
    for forbidden in (
        "consistency = 'realtime'",
        "runtime_precision = 'fp32'",
    ):
        assert forbidden not in result


def test_semantic_rebuild_overrides_impact_precision_and_alpha_mass():
    runner = load_runner()

    result = runner.reloptions_sql(
        semantic_index(),
        semantic=True,
        runtime_precision='fp16',
        semantic_impact_precision='u8',
        semantic_alpha_mass=0.75,
    )

    assert result.count('semantic_impact_precision') == 1
    assert "semantic_impact_precision = 'u8'" in result
    assert result.count('semantic_alpha_mass') == 1
    assert 'semantic_alpha_mass = 0.75' in result
    assert "semantic_impact_precision = 'fp16'" not in result
    assert 'semantic_alpha_mass = 0.5' not in result


def test_create_index_sql_preserves_include_and_predicate():
    runner = load_runner()
    idx = semantic_index()
    idx.update(
        {
            'predicate': "btrim(COALESCE(body, ''::text)) <> ''::text",
            'include_defs': ['publish_date', 'categories'],
            'rebuild_plan': {
                'action': 'reindex_current_sae',
                'keydefs': ['body'],
                'include_defs': ['publish_date', 'categories'],
            },
        }
    )

    sql = runner.create_index_sql(
        idx,
        'documents_idx',
        'reindex_current_sae',
        'fp16',
    )

    assert 'USING evoke (body)' in sql
    assert 'INCLUDE (publish_date, categories)' in sql
    assert "WHERE btrim(COALESCE(body, ''::text)) <> ''::text;" in sql
    assert 'field_aware = true' in sql
    assert 'sae = true' in sql


def test_current_text_lexical_index_is_rebuildable():
    runner = load_runner()
    idx = semantic_index()
    idx['indexdef'] = (
        'CREATE INDEX documents_idx ON search.documents USING evoke '
        "(body) WITH (sae=false, consistency='eventual')"
    )
    idx['columns'] = [
        {
            'attnum': 1,
            'keydef': 'body',
            'type': 'text',
        }
    ]

    assert runner.classify(idx) == 'rebuild_current_evoke_text_lexical_only'


def test_legacy_beta_semantic_index_migrates_to_evoke():
    runner = load_runner()
    idx = legacy_beta_index(sae=True)
    idx['columns'] = [
        {
            'attnum': 1,
            'keydef': 'body',
            'type': 'text',
        }
    ]

    plan = runner.inventory_rebuild_plan(idx)
    idx['rebuild_plan'] = plan
    sql = runner.create_index_sql(
        idx,
        'documents_idx_rebuilt',
        plan['action'],
        'fp16',
    )

    assert plan['action'] == 'migrate_legacy_beta_sae'
    assert runner.classify(idx) == 'migrate_legacy_beta_sae'
    assert 'USING evoke (body)' in sql
    assert 'USING ii42' not in sql
    assert 'sae = true' in sql


def test_legacy_beta_lexical_index_migrates_to_evoke():
    runner = load_runner()
    idx = legacy_beta_index(sae=False)
    idx['columns'] = [
        {
            'attnum': 1,
            'keydef': 'body',
            'type': 'text',
        }
    ]

    plan = runner.inventory_rebuild_plan(idx)

    assert plan['action'] == 'migrate_legacy_beta_text_lexical_only'
    assert runner.classify({**idx, 'rebuild_plan': plan}) == (
        'migrate_legacy_beta_text_lexical_only'
    )


def test_inventory_includes_legacy_beta_access_method():
    runner = load_runner()

    assert "'ii42'" in inspect.getsource(runner.inventory_database)


def test_inventory_plan_classifies_current_semantic_topology():
    runner = load_runner()
    idx = semantic_index()
    idx.update(
        {
            'columns': [
                {
                    'attnum': 1,
                    'keydef': 'body',
                    'type': 'text',
                }
            ],
            'include_defs': ['publish_date'],
        }
    )

    plan = runner.inventory_rebuild_plan(idx)

    assert plan == {
        'action': 'reindex_current_sae',
        'key_count': 1,
        'keydefs': ['body'],
        'include_defs': ['publish_date'],
        'reason': '',
    }


def test_refresh_plan_rejects_missing_live_index(monkeypatch):
    runner = load_runner()
    planned = semantic_index()
    planned.update(
        {
            'columns': [{'attnum': 1, 'keydef': 'body', 'type': 'text'}],
            'rebuild_plan': {
                'action': 'reindex_current_sae',
                'keydefs': ['body'],
                'include_defs': [],
            },
        }
    )
    live = dict(planned)
    live.update(
        {
            'schema_name': 'search',
            'index_name': 'unplanned_idx',
            'index_regclass': 'search.unplanned_idx',
        }
    )
    monkeypatch.setattr(
        runner,
        'inventory_database',
        lambda _db: {'indexes': [planned, live]},
    )

    with pytest.raises(RuntimeError, match='absent from the runnable plan'):
        runner.require_refresh_plan_complete(
            'postgres',
            [planned],
            [(planned, 'reindex_current_sae')],
            runner.REBUILD_PLAN_CONTRACT,
        )


def test_refresh_plan_rejects_stale_scope_topology(monkeypatch):
    runner = load_runner()
    planned = semantic_index()
    planned.update(
        {
            'columns': [{'attnum': 1, 'keydef': 'body', 'type': 'text'}],
            'include_defs': [],
            'rebuild_plan': {
                'action': 'reindex_current_sae',
                'keydefs': ['body'],
                'include_defs': [],
            },
        }
    )
    live = dict(planned)
    live['include_defs'] = ['publish_date']
    monkeypatch.setattr(
        runner,
        'inventory_database',
        lambda _db: {'indexes': [live]},
    )

    with pytest.raises(RuntimeError, match='no longer matches'):
        runner.require_refresh_plan_complete(
            'postgres',
            [planned],
            [(planned, 'reindex_current_sae')],
            runner.REBUILD_PLAN_CONTRACT,
        )


def test_extension_refresh_is_atomic_and_restrictive(monkeypatch, tmp_path):
    runner = load_runner()
    schemas = iter(['evoke_private', 'evoke_private'])
    statements = []

    monkeypatch.setattr(runner, 'extension_schema', lambda _db: next(schemas))
    monkeypatch.setattr(
        runner,
        'psql',
        lambda _db, sql, timeout=None: (statements.append(sql) or '', 0.0),
    )
    monkeypatch.setattr(runner, 'log_event', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        'reconcile_extension_catalog',
        lambda *_args, **_kwargs: None,
    )

    idx = semantic_index()
    actual = runner.refresh_extension_catalog(
        'postgres',
        [idx],
        tmp_path / 'rebuild.jsonl',
    )

    assert actual == 'evoke_private'
    transition = statements[0]
    assert transition.startswith('BEGIN;')
    assert 'DROP INDEX IF EXISTS "search"."documents_idx";' in transition
    assert 'DROP EXTENSION evoke;' in transition
    assert 'DROP EXTENSION evoke CASCADE;' not in transition
    assert 'CREATE EXTENSION evoke WITH SCHEMA "evoke_private";' in transition
    assert 'evoke_catalog_contract_internal' in transition
    assert 'evoke_catalog_v1' in transition
    assert transition.endswith('COMMIT;')


def test_extension_catalog_rejects_package_version_mismatch(
    monkeypatch,
):
    runner = load_runner()
    state = {'installed': '0.2.0', 'default': '0.2.1'}
    statements = []
    contracts = []

    monkeypatch.setattr(
        runner,
        'extension_version_state',
        lambda _db: state,
    )
    monkeypatch.setattr(
        runner,
        'psql',
        lambda _db, sql, timeout=None: (statements.append(sql) or '', 0.0),
    )
    monkeypatch.setattr(
        runner,
        'require_extension_catalog',
        lambda db, schema: contracts.append((db, schema)),
    )
    try:
        runner.reconcile_extension_catalog(
            'postgres',
            'evoke_ext',
        )
    except RuntimeError as exc:
        assert 'current-only package' in str(exc)
        assert '--refresh-extension' in str(exc)
    else:
        raise AssertionError('mismatched extension catalog was accepted')

    assert statements == []
    assert contracts == []


def test_extension_catalog_contract_requires_owned_precision_overloads(
    monkeypatch,
):
    runner = load_runner()
    statements = []

    def fake_psql(_db, sql, timeout=None):
        statements.append(sql)
        if 'evoke_catalog_contract_internal' in sql and sql.startswith(
            'SELECT '
        ):
            return runner.CATALOG_CONTRACT, 0.0
        return '', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    runner.require_extension_catalog('postgres', 'evoke_ext')

    sql = statements[0]
    assert ('evoke_query_trace_internal', '') in (
        runner.REQUIRED_EXTENSION_FUNCTIONS
    )
    assert ('evoke_catalog_contract_internal', '') in (
        runner.REQUIRED_EXTENSION_FUNCTIONS
    )
    assert ('evoke_index_generation_status_internal', 'regclass') in (
        runner.REQUIRED_EXTENSION_FUNCTIONS
    )
    for signature in runner.required_extension_signatures('evoke_ext'):
        assert signature in sql
    assert 'pg_catalog.pg_depend' in sql
    assert "dependency.deptype = 'e'" in sql
    assert len(statements) == 2
    assert 'evoke_catalog_contract_internal' in statements[1]


def test_extension_catalog_contract_rejects_stale_identity(monkeypatch):
    runner = load_runner()

    def fake_psql(_db, sql, timeout=None):
        if sql.startswith('SELECT '):
            return 'evoke_catalog_stale', 0.0
        return '', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    with pytest.raises(RuntimeError, match='catalog identity is stale'):
        runner.require_extension_catalog('postgres', 'evoke_ext')


def test_extension_catalog_contract_fails_closed(monkeypatch):
    runner = load_runner()

    monkeypatch.setattr(
        runner,
        'psql',
        lambda *_args, **_kwargs: (
            '"evoke_ext".evoke_runtime_service_query_atoms(text,text,text)',
            0.0,
        ),
    )

    try:
        runner.require_extension_catalog('postgres', 'evoke_ext')
    except RuntimeError as exc:
        assert 'catalog contract is incomplete' in str(exc)
        assert 'recreate the current extension catalog' in str(exc)
    else:
        raise AssertionError('incomplete extension catalog passed preflight')


def test_extension_module_authority_requires_current_libdir(monkeypatch):
    runner = load_runner()
    statements = []

    monkeypatch.setattr(
        runner,
        'psql',
        lambda _db, sql, timeout=None: (
            statements.append(sql) or '["$libdir/evoke"]',
            0.0,
        ),
    )

    runner.require_extension_module_authority('postgres')

    assert 'procedure.probin' in statements[0]
    assert "language.lanname = 'c'" in statements[0]
    assert "dependency.deptype = 'e'" in statements[0]


@pytest.mark.parametrize(
    'paths',
    [
        '[]',
        '["/opt/evoke-old/lib/evoke"]',
        '["$libdir/evoke", "/opt/evoke-old/lib/evoke"]',
    ],
)
def test_extension_module_authority_rejects_stale_paths(
    monkeypatch,
    paths,
):
    runner = load_runner()
    monkeypatch.setattr(
        runner,
        'psql',
        lambda *_args, **_kwargs: (paths, 0.0),
    )

    with pytest.raises(RuntimeError, match='C-module authority'):
        runner.require_extension_module_authority('postgres')


def test_ready_check_requires_catalog_and_current_generation(monkeypatch):
    runner = load_runner()
    statements = []

    def fake_psql(_db, sql, timeout=None):
        statements.append(sql)
        if 'FROM pg_extension e' in sql:
            return 'evoke_ext', 0.0
        return 't', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    assert runner.target_index_ready(
        'postgres',
        semantic_index(),
        'reindex_current_sae',
        'fp16',
    )
    assert len(statements) == 3
    assert all('evoke_index_status' not in sql for sql in statements)
    assert 'indisvalid' in statements[0]
    assert 'indisready' in statements[0]
    assert 'runtime_precision=%' in statements[0]
    assert 'evoke_index_runtime_state_json' in statements[2]
    assert "'convergent_segments'" in statements[2]
    assert "'payload_health' = 'ok'" in statements[2]
    assert "'rebuild_required'" in statements[2]


def test_ready_check_rebuilds_unsupported_physical_generation(monkeypatch):
    runner = load_runner()

    def fake_psql(_db, sql, timeout=None):
        if 'FROM pg_extension e' in sql:
            return 'evoke_ext', 0.0
        if 'evoke_index_runtime_state_json' in sql:
            raise RuntimeError(
                'psql failed in postgres: ERROR: unsupported evoke index '
                'metapage version 2'
            )
        return 't', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    assert not runner.target_index_ready(
        'postgres',
        semantic_index(),
        'reindex_current_sae',
        'fp16',
    )


def test_ready_check_propagates_unrelated_generation_error(monkeypatch):
    runner = load_runner()

    def fake_psql(_db, sql, timeout=None):
        if 'FROM pg_extension e' in sql:
            return 'evoke_ext', 0.0
        if 'evoke_index_runtime_state_json' in sql:
            raise RuntimeError('psql failed in postgres: connection lost')
        return 't', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    try:
        runner.target_index_ready(
            'postgres',
            semantic_index(),
            'reindex_current_sae',
            'fp16',
        )
    except RuntimeError as exc:
        assert 'connection lost' in str(exc)
    else:
        raise AssertionError('unrelated generation error was ignored')


def test_semantic_runtime_preflight_requires_local_query_runtime(monkeypatch):
    runner = load_runner()
    statements = []
    results = iter(
        [
            ('enabled:api=29', 0.0),
            ('available:linked:version=1.29.0', 0.0),
        ]
    )

    def fake_psql(_db, sql, timeout=None):
        statements.append((sql, timeout))
        return next(results)

    monkeypatch.setattr(runner, 'psql', fake_psql)

    actual = runner.require_semantic_query_runtime('postgres', 'evoke_ext')

    assert actual == {
        'expected_version': '1.29.0',
        'build_info': 'enabled:api=29',
        'probe': 'available:linked:version=1.29.0',
    }
    assert statements == [
        ('SELECT "evoke_ext"."evoke_onnxruntime_build_info"();', 60),
        ('SELECT "evoke_ext"."evoke_onnxruntime_probe"();', 60),
    ]


@pytest.mark.parametrize(
    ('build_info', 'probe'),
    [
        ('enabled:api=26', 'available:linked:version=1.26.0'),
        ('enabled:api=29', 'available:linked:version=1.26.0'),
    ],
)
def test_semantic_runtime_preflight_rejects_unpinned_ort(
    monkeypatch,
    build_info,
    probe,
):
    runner = load_runner()
    results = iter([(build_info, 0.0), (probe, 0.0)])

    monkeypatch.setattr(
        runner,
        'psql',
        lambda *_args, **_kwargs: next(results),
    )

    with pytest.raises(RuntimeError, match='pinned'):
        runner.require_semantic_query_runtime('postgres', 'evoke_ext')


def test_semantic_runtime_preflight_rejects_empty_version_override(
    monkeypatch,
) -> None:
    runner = load_runner()
    monkeypatch.setenv('EVOKE_EXPECTED_ONNXRUNTIME_VERSION', '   ')

    with pytest.raises(RuntimeError, match='non-empty pinned'):
        runner.require_semantic_query_runtime('postgres', 'evoke_ext')


def test_semantic_runtime_preflight_rejects_no_ort_binary(monkeypatch):
    runner = load_runner()
    statements = []

    def fake_psql(_db, sql, timeout=None):
        statements.append((sql, timeout))
        return 'disabled', 0.0

    monkeypatch.setattr(runner, 'psql', fake_psql)

    try:
        runner.require_semantic_query_runtime('postgres', 'public')
    except RuntimeError as exc:
        assert 'query-capable evoke binary' in str(exc)
    else:
        raise AssertionError('no-ORT binary passed semantic preflight')
    assert statements == [
        ('SELECT "public"."evoke_onnxruntime_build_info"();', 60),
    ]


def test_rebuild_swaps_indexes_in_one_transaction(monkeypatch, tmp_path):
    runner = load_runner()
    idx = semantic_index()
    statements = []
    validations = []

    monkeypatch.setattr(runner, 'table_exists', lambda *_args: True)
    monkeypatch.setattr(runner, 'target_index_ready', lambda *_args: False)
    monkeypatch.setattr(runner, 'index_exists', lambda *_args: True)
    monkeypatch.setattr(runner, 'temp_index_name', lambda *_args: 'tmp_idx')
    monkeypatch.setattr(runner, 'drop_leftover_temp', lambda *_args: None)
    monkeypatch.setattr(runner, 'create_index_sql', lambda *_args: 'CREATE TEMP')
    monkeypatch.setattr(
        runner,
        'validate_index',
        lambda _db, _idx, name: validations.append(name),
    )
    monkeypatch.setattr(runner, 'log_event', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runner,
        'psql',
        lambda _db, sql, timeout=None: (statements.append(sql) or '', 0.0),
    )

    result = runner.rebuild_index(
        'postgres',
        idx,
        'reindex_current_sae',
        False,
        tmp_path / 'rebuild.jsonl',
        'fp16',
    )

    assert result == 'done'
    assert validations == ['tmp_idx', 'documents_idx']
    swap = next(sql for sql in statements if sql.startswith('BEGIN;'))
    assert 'DROP INDEX "search"."documents_idx";' in swap
    assert 'ALTER INDEX "search"."tmp_idx"' in swap
    assert swap.endswith('COMMIT;')
    assert not any(
        sql.startswith('DROP INDEX "search"."documents_idx"')
        for sql in statements
    )
