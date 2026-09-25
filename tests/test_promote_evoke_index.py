import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'promote_evoke_index.py'
MATRIX_SCRIPT = ROOT / 'scripts' / 'benchmark_commons_query_matrix.py'
SERIAL_RANK_CASES = {
    'arxiv_date_category',
    'ca_chunk_scope_stress',
    'pubmed_date_category',
    'pubmed_date_journal',
    'system_chunk_scope',
}
COMMONS_CASES = {
    'arxiv_broad_date',
    'arxiv_date_category',
    'arxiv_organization_broad',
    'arxiv_organization_selective',
    'arxiv_unfiltered',
    'ca_chunk_scope_ordinary',
    'ca_chunk_scope_stress',
    'ca_policy_union',
    'pubmed_broad_date',
    'pubmed_date_category',
    'pubmed_date_journal',
    'pubmed_partial_date',
    'pubmed_unfiltered',
    'system_chunk_scope',
    'tx_chunk_scope_ordinary',
    'tx_policy_union',
    'wa_chunk_scope_ordinary',
    'wa_policy_union',
}
COMMONS_FIXED_INDEXES = {
    'commons.data_arxiv__title_abstract__field_aware_bm25_idx',
    'commons.data_policy_ca__title_description__field_aware_bm25_idx',
    'commons.data_policy_ca_chunks__content__bm25_idx',
    'commons.data_policy_tx__title_description__field_aware_bm25_idx',
    'commons.data_policy_tx_chunks__content__bm25_idx',
    'commons.data_policy_wa__title_description__field_aware_bm25_idx',
    'commons.data_policy_wa_chunks__content__bm25_idx',
    'commons.sys_chunks__content__bm25_idx',
}


def load_module():
    spec = importlib.util.spec_from_file_location('promote_evoke_index', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_matrix_module():
    name = 'benchmark_commons_query_matrix_for_promotion_test'
    spec = importlib.util.spec_from_file_location(name, MATRIX_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def matrix(path: Path, concurrency: int, index_name: str) -> Path:
    cases = []
    for name in sorted(COMMONS_CASES):
        case = {
            'name': name,
            'passed': True,
            'first': {'count': concurrency},
            'warm': {'count': concurrency},
            'result_stability': {
                'attempts': concurrency * 2,
                'passed': True,
            },
        }
        if concurrency == 1 and name in SERIAL_RANK_CASES:
            case['rank_oracle'] = {'passed': True}
        cases.append(case)
    indexes = {
        name: {'generation_id': f'fixed/{offset}'}
        for offset, name in enumerate(sorted(COMMONS_FIXED_INDEXES), start=1)
    }
    indexes[index_name] = {'generation_id': '1/2/3'}
    path.write_text(json.dumps({
        'suite': 'commons_query_matrix',
        'contract_version': 10,
        'mode': 'qualification',
        'passed': True,
        'concurrency': concurrency,
        'repeats': 2,
        'environment': {
            'qualified': True,
            'catalog_qualified': True,
            'onnxruntime_qualified': True,
            'schema_qualified': True,
            'generation_qualified': True,
            'planner_qualified': True,
            'runtime_qualified': True,
            'planner_capture_qualified': True,
            'query_metadata_qualified': True,
            'restart_qualified': True,
            'restart_requirement_seconds': 300,
            'postmaster_start_epoch': 900.0,
            'postmaster_age_seconds': 100.0,
            'extension_version': '0.2.4',
            'pubmed_index': index_name,
            'pubmed_index_source': 'explicit',
            'indexes': indexes,
        },
        'cases': cases,
    }))
    return path


def test_promotion_contract_matches_matrix_producer() -> None:
    promotion = load_module()
    matrix_producer = load_matrix_module()

    assert promotion.COMMONS_MATRIX_CONTRACT_VERSION == (
        matrix_producer.MATRIX_CONTRACT_VERSION
    )


def test_qualification_requires_same_generation_at_1_4_8(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    evidence = module.validate_qualification_evidence(paths, index_name)

    assert {item['concurrency'] for item in evidence} == {1, 4, 8}
    assert {item['generation_id'] for item in evidence} == {'1/2/3'}


def test_qualification_rejects_single_wave_evidence(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[0].read_text())
    document['repeats'] = 1
    paths[0].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_missing_warm_panel(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[1].read_text())
    document['cases'][0].pop('warm')
    paths[1].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='separate cold and warm panels'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_missing_result_stability(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[1].read_text())
    document['cases'][0].pop('result_stability')
    paths[1].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='stable repeated results'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_wrong_stability_attempt_count(
    tmp_path,
) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[2].read_text())
    document['cases'][0]['result_stability']['attempts'] = 1
    paths[2].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='stable repeated results'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_missing_concurrency(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4)
    ]

    with pytest.raises(ValueError, match='concurrency 1, 4, and 8'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_duplicate_concurrency(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    paths.append(matrix(tmp_path / 'c8-copy.json', 8, index_name))

    with pytest.raises(ValueError, match='exactly one matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_mixed_commons_generation_set(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[1].read_text())
    arxiv = 'commons.data_arxiv__title_abstract__field_aware_bm25_idx'
    document['environment']['indexes'][arxiv]['generation_id'] = 'changed'
    paths[1].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='complete Commons generation set'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_incomplete_commons_index_set(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[2].read_text())
    missing = 'commons.data_policy_tx_chunks__content__bm25_idx'
    document['environment']['indexes'].pop(missing)
    paths[2].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='complete index set'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_older_matrix_contract(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[0].read_text())
    document.pop('contract_version')
    paths[0].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)

    document = json.loads(matrix(
        paths[2],
        8,
        index_name,
    ).read_text())
    document['environment']['restart_requirement_seconds'] = 601
    paths[2].write_text(json.dumps(document))
    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_failed_schema_contract(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[0].read_text())
    document['environment']['schema_qualified'] = False
    paths[0].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_failed_runtime_contract(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[0].read_text())
    document['environment']['runtime_qualified'] = False
    paths[0].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_missing_restart_evidence(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[2].read_text())
    document['environment']['restart_qualified'] = False
    paths[2].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)

    document = json.loads(matrix(
        paths[2],
        8,
        index_name,
    ).read_text())
    document['environment']['postmaster_age_seconds'] = 301.0
    paths[2].write_text(json.dumps(document))
    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_incomplete_case_surface(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[1].read_text())
    document['cases'] = [
        case for case in document['cases']
        if case['name'] != 'pubmed_broad_date'
    ]
    paths[1].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='current contract case set'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_duplicate_case_surface(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[1].read_text())
    document['cases'].append(document['cases'][0])
    paths[1].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='current contract case set'):
        module.validate_qualification_evidence(paths, index_name)


def test_qualification_rejects_release_version_mismatch(tmp_path) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]

    with pytest.raises(ValueError, match='unqualified Commons matrix'):
        module.validate_qualification_evidence(
            paths,
            index_name,
            '0.2.5',
        )


def test_release_evidence_revalidates_installed_artifacts(tmp_path) -> None:
    module = load_module()
    package_root = tmp_path / 'package'
    package_root.mkdir()
    build_info = package_root / 'BUILD-INFO.txt'
    staged = package_root / 'evoke.so'
    installed = tmp_path / 'installed-evoke.so'
    staged_model = package_root / 'manifest.json'
    installed_model = tmp_path / 'installed-manifest.json'
    build_info.write_text('qualified\n')
    for path in (staged, installed):
        path.write_bytes(b'binary')
    for path in (staged_model, installed_model):
        path.write_bytes(b'model')

    def identity(path: Path) -> dict[str, str]:
        return module.current_artifact_identity(path)

    artifact = {
        'name': 'evoke.so',
        'staged_path': str(staged),
        'installed_path': str(installed),
        'staged': identity(staged),
        'installed': identity(installed),
    }
    model_artifact = {
        'name': 'manifest.json',
        'staged_path': str(staged_model),
        'installed_path': str(installed_model),
        'staged': identity(staged_model),
        'installed': identity(installed_model),
    }
    binding = {
        'passed': True,
        'fingerprint': 'release-fingerprint',
        'package_root': str(package_root),
        'build_info_sha256': identity(build_info)['sha256'],
        'build_info': {
            'Git tree': 'clean',
            'Git commit': 'abc123',
            'Version': '0.2.4',
            'ONNX Runtime': '1.29.0',
            'ONNX Runtime linkage': 'bundled in PostgreSQL pkglibdir',
        },
        'artifacts': [artifact],
        'model_binding': {
            'passed': True,
            'model_id': 'model-v1',
            'manifest_sha256': identity(staged_model)['sha256'],
            'artifacts': [model_artifact],
        },
    }
    evidence_path = tmp_path / 'maturity.json'
    evidence_path.write_text(json.dumps({
        'suite': 'evoke_product_maturity',
        'passed': True,
        'package_binding': {
            'stable': True,
            'preflight': binding,
            'postflight': binding,
        },
    }))

    result = module.validate_release_evidence(evidence_path)
    assert result['fingerprint'] == 'release-fingerprint'
    assert result['model_id'] == 'model-v1'
    assert result['onnxruntime_version'] == '1.29.0'

    stale = json.loads(evidence_path.read_text())
    stale['package_binding']['preflight']['build_info'][
        'ONNX Runtime'
    ] = '1.26.0'
    stale['package_binding']['postflight']['build_info'][
        'ONNX Runtime'
    ] = '1.26.0'
    evidence_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match='source or model identity'):
        module.validate_release_evidence(evidence_path)

    evidence_path.write_text(json.dumps({
        'suite': 'evoke_product_maturity',
        'passed': True,
        'package_binding': {
            'stable': True,
            'preflight': binding,
            'postflight': binding,
        },
    }))

    installed.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed after qualification'):
        module.validate_release_evidence(evidence_path)


def test_qualification_rejects_serial_run_without_rank_oracle(
    tmp_path,
) -> None:
    module = load_module()
    index_name = 'commons.pubmed_candidate'
    paths = [
        matrix(tmp_path / f'c{concurrency}.json', concurrency, index_name)
        for concurrency in (1, 4, 8)
    ]
    document = json.loads(paths[0].read_text())
    target = next(
        case for case in document['cases']
        if case['name'] == 'pubmed_date_category'
    )
    target.pop('rank_oracle')
    paths[0].write_text(json.dumps(document))

    with pytest.raises(ValueError, match='serial rank oracle set'):
        module.validate_qualification_evidence(paths, index_name)


def test_post_swap_evidence_requires_stable_name(tmp_path) -> None:
    module = load_module()
    path = matrix(tmp_path / 'stable.json', 1, 'commons.stable_idx')
    document = json.loads(path.read_text())
    document['environment']['pubmed_index_source'] = 'stable'
    path.write_text(json.dumps(document))

    identity = module.validate_post_swap_evidence(
        path,
        'commons.stable_idx',
        '0.2.4',
    )

    assert len(identity['sha256']) == 64
    assert identity['generation_id'] == '1/2/3'
    assert len(identity['index_generations']) == 9

    document = json.loads(path.read_text())
    document['cases'] = [
        case for case in document['cases']
        if case['name'] != 'arxiv_broad_date'
    ]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='current contract case set'):
        module.validate_post_swap_evidence(
            path,
            'commons.stable_idx',
            '0.2.4',
        )

    document = json.loads(matrix(
        path,
        1,
        'commons.stable_idx',
    ).read_text())
    document['environment']['pubmed_index_source'] = 'stable'
    document['cases'].append(document['cases'][0])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='current contract case set'):
        module.validate_post_swap_evidence(
            path,
            'commons.stable_idx',
            '0.2.4',
        )


def test_transaction_renames_stable_before_candidate() -> None:
    module = load_module()
    stable = ('commons', 'stable_idx')
    candidate = ('commons', 'candidate_idx')
    rollback = ('commons', 'rollback_idx')

    sql = module.transaction_sql([
        module.identity_guard_sql({
            'stable': ('commons.stable_idx', 10),
            'candidate': ('commons.candidate_idx', 20),
            'rollback': ('commons.rollback_idx', None),
        }),
        module.rename_sql(stable, rollback),
        module.rename_sql(candidate, stable),
    ], 'commons.stable_idx')

    assert sql.startswith('BEGIN;')
    assert 'pg_advisory_xact_lock' in sql
    assert "to_regclass('commons.stable_idx')::oid = 10::oid" in sql
    assert "to_regclass('commons.rollback_idx')::oid IS NULL" in sql
    assert sql.index('Evoke promotion identity changed') < sql.index(
        'ALTER INDEX',
    )
    assert sql.index('"stable_idx"') < sql.index('"candidate_idx"')
    assert sql.endswith('COMMIT;')


def test_names_must_share_one_schema() -> None:
    module = load_module()

    with pytest.raises(ValueError, match='share one schema'):
        module.validate_names(
            ('commons', 'stable'),
            ('other', 'candidate'),
            ('commons', 'rollback'),
        )


def test_generation_gate_requires_complete_accelerator() -> None:
    module = load_module()
    qualified = {
        'generation_id': '1/2/3',
        'valid': True,
        'health': 'ok',
        'runtime_contract_matches': True,
        'semantic_accelerator': {
            'state': 'ready',
            'forward_complete': True,
            'scope_current': True,
        },
    }

    assert module.generation_qualified(qualified) is True
    qualified['semantic_accelerator']['forward_complete'] = False
    assert module.generation_qualified(qualified) is False

    qualified['semantic_accelerator']['forward_complete'] = True
    qualified['semantic_accelerator']['scope_current'] = False
    assert module.generation_qualified(qualified) is False


@pytest.mark.parametrize('bootstrap', [False, True])
def test_promote_dry_run_retains_or_bootstraps_stable_index(
    monkeypatch,
    tmp_path,
    bootstrap,
) -> None:
    module = load_module()
    stable = {
        'oid': 10,
        'table_oid': 1,
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
    }
    candidate = {**stable, 'oid': 20}
    snapshots = {
        ('commons', 'stable'): None if bootstrap else stable,
        ('commons', 'candidate'): candidate,
        ('commons', 'rollback'): None,
    }
    paths = [
        matrix(
            tmp_path / f'c{concurrency}.json',
            concurrency,
            'commons.candidate',
        )
        for concurrency in (1, 4, 8)
    ]
    expected_generations = json.loads(paths[0].read_text())[
        'environment'
    ]['indexes']
    expected_generations = {
        name: str(status['generation_id'])
        for name, status in expected_generations.items()
    }
    args = SimpleNamespace(
        action='promote',
        dsn='',
        psql='psql',
        stable='commons.stable',
        candidate='commons.candidate',
        rollback='commons.rollback',
        qualification_evidence=paths,
        release_evidence=tmp_path / 'release.json',
        post_swap_evidence=None,
        output=tmp_path / 'promotion.json',
        dry_run=True,
        bootstrap=bootstrap,
    )
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(
        module,
        'catalog_snapshot',
        lambda _args, name: snapshots[name],
    )
    monkeypatch.setattr(
        module,
        'generation_snapshot',
        lambda *_args: {
            'generation_id': '1/2/3',
            'valid': True,
            'health': 'ok',
            'runtime_contract_matches': True,
            'semantic_accelerator': {
                'state': 'ready',
                'forward_complete': True,
                'scope_current': True,
            },
        },
    )
    monkeypatch.setattr(
        module,
        'run_psql',
        lambda *_args: pytest.fail('dry run executed SQL'),
    )
    monkeypatch.setattr(
        module,
        'validate_release_evidence',
        lambda _path: {'extension_version': '0.2.4'},
    )
    monkeypatch.setattr(
        module,
        'live_commons_index_generations',
        lambda *_args: expected_generations,
    )
    transaction_sql = module.transaction_sql
    captured = {}

    def capture_transaction(statements, stable_name):
        captured['statements'] = statements
        return transaction_sql(statements, stable_name)

    monkeypatch.setattr(module, 'transaction_sql', capture_transaction)

    assert module.main() == 0
    evidence = json.loads(args.output.read_text())
    assert evidence['expected'] == {
        'candidate_oid': None,
        'rollback_oid': None if bootstrap else 10,
        'stable_oid': 20,
    }
    assert evidence['bootstrap'] is bootstrap
    if bootstrap:
        assert evidence['before']['stable'] is None
    else:
        assert evidence['before']['stable']['oid'] == 10
    assert evidence['before']['candidate']['oid'] == 20
    assert evidence['live_commons_generations'] == expected_generations
    renames = [
        statement
        for statement in captured['statements']
        if statement.startswith('ALTER INDEX')
    ]
    assert len(renames) == (1 if bootstrap else 2)
    assert renames[-1] == (
        'ALTER INDEX "commons"."candidate" RENAME TO "stable";'
    )


def test_bootstrap_finalize_records_stable_without_rollback(
    monkeypatch,
    tmp_path,
) -> None:
    module = load_module()
    stable = {
        'oid': 20,
        'table_oid': 1,
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
    }
    snapshots = {
        ('commons', 'stable'): stable,
        ('commons', 'candidate'): None,
        ('commons', 'rollback'): None,
    }
    generations = {
        name: f'fixed/{offset}'
        for offset, name in enumerate(sorted(COMMONS_FIXED_INDEXES), start=1)
    }
    generations['commons.stable'] = '1/2/3'
    args = SimpleNamespace(
        action='finalize',
        dsn='',
        psql='psql',
        stable='commons.stable',
        candidate='commons.candidate',
        rollback='commons.rollback',
        qualification_evidence=[],
        release_evidence=tmp_path / 'release.json',
        post_swap_evidence=tmp_path / 'post-swap.json',
        output=tmp_path / 'finalize.json',
        dry_run=True,
        bootstrap=True,
    )
    monkeypatch.setattr(module, 'parse_args', lambda: args)
    monkeypatch.setattr(
        module,
        'catalog_snapshot',
        lambda _args, name: snapshots[name],
    )
    monkeypatch.setattr(
        module,
        'generation_snapshot',
        lambda *_args: {
            'generation_id': '1/2/3',
            'valid': True,
            'health': 'ok',
            'runtime_contract_matches': True,
            'semantic_accelerator': {
                'state': 'ready',
                'forward_complete': True,
                'scope_current': True,
            },
        },
    )
    monkeypatch.setattr(
        module,
        'validate_release_evidence',
        lambda _path: {'extension_version': '0.2.4'},
    )
    monkeypatch.setattr(
        module,
        'validate_post_swap_evidence',
        lambda *_args: {
            'generation_id': '1/2/3',
            'index_generations': generations,
        },
    )
    monkeypatch.setattr(
        module,
        'live_commons_index_generations',
        lambda *_args: generations,
    )
    monkeypatch.setattr(
        module,
        'run_psql',
        lambda *_args: pytest.fail('dry run executed SQL'),
    )
    transaction_sql = module.transaction_sql
    captured = {}

    def capture_transaction(statements, stable_name):
        captured['statements'] = statements
        return transaction_sql(statements, stable_name)

    monkeypatch.setattr(module, 'transaction_sql', capture_transaction)

    assert module.main() == 0
    evidence = json.loads(args.output.read_text())
    assert evidence['bootstrap'] is True
    assert evidence['expected'] == {
        'candidate_oid': None,
        'rollback_oid': None,
        'stable_oid': 20,
    }
    assert not any(
        statement.startswith('DROP INDEX')
        for statement in captured['statements']
    )


def test_bootstrap_is_invalid_for_rollback(monkeypatch) -> None:
    module = load_module()
    args = SimpleNamespace(
        action='rollback',
        stable='commons.stable',
        candidate='commons.candidate',
        rollback='commons.rollback',
        bootstrap=True,
    )
    monkeypatch.setattr(module, 'parse_args', lambda: args)

    with pytest.raises(ValueError, match='invalid for rollback'):
        module.main()
