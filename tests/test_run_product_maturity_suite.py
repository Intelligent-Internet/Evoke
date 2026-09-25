from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_product_maturity_suite


def mock_pg_config_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = {
        '--pkglibdir': tmp_path / 'pg-lib',
        '--sharedir': tmp_path / 'pg-share',
    }
    monkeypatch.setattr(
        run_product_maturity_suite,
        'pg_config_value',
        lambda pg_bin, option: str(paths[option]),
    )


def test_maturity_preflight_compiles_all_product_python() -> None:
    source = (
        run_product_maturity_suite.REPO_ROOT
        / 'scripts/run_product_maturity_suite.py'
    ).read_text(encoding='utf-8')

    assert "'compileall'" in source
    assert "'scripts'," in source
    assert "'tests'," in source
    assert "'py_compile'" not in source


def test_failed_step_is_persisted_before_suite_stops(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'maturity.json'
    args = SimpleNamespace(
        pg_bin=tmp_path / 'pg-bin',
        package_root=tmp_path / 'package',
        source_package_root=tmp_path / 'source-package',
        output=output,
        lifecycle_docs=10,
        benchmark_docs=10,
        benchmark_repeats=1,
        benchmark_concurrency=1,
        model_path=None,
        skip_benchmark=True,
        skip_restart_smoke=False,
        restart_wait_seconds=1.0,
    )
    monkeypatch.setattr(
        run_product_maturity_suite,
        'parse_args',
        lambda: args,
    )
    mock_pg_config_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_product_maturity_suite,
        'run_step',
        lambda name, cmd: {
            'name': name,
            'command': [str(item) for item in cmd],
            'returncode': 7,
            'elapsed_ms': 1.0,
            'stdout_tail': '',
            'stderr_tail': 'expected failure',
        },
    )
    monkeypatch.setattr(
        run_product_maturity_suite,
        'inspect_package_binding',
        lambda package_root, pg_bin: {
            'passed': True,
            'fingerprint': 'stable-package',
        },
    )

    assert run_product_maturity_suite.main() == 1

    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['passed'] is False
    assert result['failed_step'] == 'product convergence inventory'
    assert result['package_binding']['stable'] is True
    assert len(result['steps']) == 1
    assert result['steps'][0]['name'] == 'product convergence inventory'
    assert result['steps'][0]['returncode'] == 7
    assert result['steps'][0]['stderr_tail'] == 'expected failure'


def test_package_mismatch_stops_before_product_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'maturity.json'
    args = SimpleNamespace(
        pg_bin=tmp_path / 'pg-bin',
        package_root=tmp_path / 'package',
        source_package_root=tmp_path / 'source-package',
        output=output,
        lifecycle_docs=10,
        benchmark_docs=10,
        benchmark_repeats=1,
        benchmark_concurrency=1,
        model_path=None,
        skip_benchmark=True,
        skip_restart_smoke=False,
        restart_wait_seconds=1.0,
    )
    monkeypatch.setattr(
        run_product_maturity_suite,
        'parse_args',
        lambda: args,
    )
    mock_pg_config_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_product_maturity_suite,
        'inspect_package_binding',
        lambda package_root, pg_bin: {
            'passed': False,
            'errors': ['package binding mismatch: evoke.so'],
        },
    )
    called = False

    def unexpected_run(name: str, cmd: list[str | Path]) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError(f'unexpected step: {name}')

    monkeypatch.setattr(
        run_product_maturity_suite,
        'run_step',
        unexpected_run,
    )

    assert run_product_maturity_suite.main() == 1
    assert called is False
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['passed'] is False
    assert result['failed_step'] == 'staged package binding preflight'
    assert result['steps'] == []


def test_compare_package_artifact_detects_content_and_symlink_changes(
    tmp_path: Path,
) -> None:
    staged_file = tmp_path / 'stage-file'
    installed_file = tmp_path / 'installed-file'
    staged_file.write_bytes(b'same')
    installed_file.write_bytes(b'same')

    matching = run_product_maturity_suite.compare_package_artifact(
        staged_file,
        installed_file,
    )
    assert matching['matched'] is True

    installed_file.write_bytes(b'different')
    mismatch = run_product_maturity_suite.compare_package_artifact(
        staged_file,
        installed_file,
    )
    assert mismatch['matched'] is False

    staged_link = tmp_path / 'stage-link'
    installed_link = tmp_path / 'installed-link'
    staged_link.symlink_to('libonnxruntime.so.1')
    installed_link.symlink_to('libonnxruntime.so.1')
    assert run_product_maturity_suite.compare_package_artifact(
        staged_link,
        installed_link,
    )['matched'] is True

    installed_link.unlink()
    installed_link.symlink_to('libonnxruntime.so.2')
    assert run_product_maturity_suite.compare_package_artifact(
        staged_link,
        installed_link,
    )['matched'] is False


def test_validate_build_info_rejects_stale_or_dirty_package() -> None:
    current = {
        'commit': 'current-commit',
        'tree': 'clean',
    }
    build_info = {
        'Git commit': 'stale-commit',
        'Git tree': 'dirty',
        'ONNX Runtime': 'unexpected',
        'ONNX Runtime linkage': 'system',
    }

    assert run_product_maturity_suite.validate_build_info(
        build_info,
        current,
    ) == [
        'BUILD-INFO Git tree must be clean',
        'BUILD-INFO Git commit does not match current source',
        'BUILD-INFO ONNX Runtime does not match the pinned version',
        'BUILD-INFO ONNX Runtime linkage is not bundled',
    ]


def test_model_binding_covers_staged_and_installed_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sharedir = tmp_path / 'pg-share'
    package = tmp_path / 'package'
    staged_model = (
        package / sharedir.relative_to(sharedir.anchor)
        / 'evoke/models/default'
    )
    installed_model = sharedir / 'evoke/models/default'
    staged_model.mkdir(parents=True)
    installed_model.mkdir(parents=True)
    artifact_identity = {
        'path': 'encoder/model.onnx',
        'sha256': run_product_maturity_suite.hashlib.sha256(
            b'model'
        ).hexdigest(),
    }
    manifest = {
        'model_id': 'model-v1',
        'artifacts': {'encoder': artifact_identity},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_sha256 = run_product_maturity_suite.hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    lock = {
        'manifest_sha256': manifest_sha256,
        'artifacts': {'encoder': artifact_identity},
    }
    lock_path = tmp_path / 'milestone-model.json'
    lock_path.write_text(json.dumps(lock), encoding='utf-8')
    monkeypatch.setattr(
        run_product_maturity_suite,
        'MILESTONE_MODEL_LOCK',
        lock_path,
    )
    for root in (staged_model, installed_model):
        (root / 'encoder').mkdir()
        (root / 'encoder/model.onnx').write_bytes(b'model')
        (root / 'manifest.json').write_bytes(manifest_bytes)

    result = run_product_maturity_suite.inspect_model_binding(
        package,
        sharedir,
        {
            'Milestone model location': str(installed_model),
            'Milestone model manifest SHA-256': manifest_sha256,
            'Milestone model': 'model-v1',
        },
    )

    assert result['passed'] is True
    assert result['model_id'] == 'model-v1'
    assert len(result['artifacts']) == 2

    (installed_model / 'encoder/model.onnx').write_bytes(b'changed')
    result = run_product_maturity_suite.inspect_model_binding(
        package,
        sharedir,
        {
            'Milestone model location': str(installed_model),
            'Milestone model manifest SHA-256': manifest_sha256,
            'Milestone model': 'model-v1',
        },
    )
    assert result['passed'] is False
    assert any(
        'installed model artifact mismatch' in error
        for error in result['errors']
    )


def test_inspect_package_metadata_rejects_missing_or_changed_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        run_product_maturity_suite,
        'REPO_ROOT',
        tmp_path / 'source',
    )
    source = run_product_maturity_suite.REPO_ROOT
    source.mkdir()
    (source / 'README.md').write_text('current readme', encoding='utf-8')
    (source / 'LICENSE').write_text('current license', encoding='utf-8')

    package = tmp_path / 'package'
    (package / 'LICENSES').mkdir(parents=True)
    (package / 'README.md').write_text('stale readme', encoding='utf-8')
    (package / 'LICENSES/EVOKE-LICENSE').write_text(
        'current license',
        encoding='utf-8',
    )

    artifacts, errors = (
        run_product_maturity_suite.inspect_package_metadata(package)
    )

    assert [artifact['name'] for artifact in artifacts] == [
        'README.md',
        'EVOKE-LICENSE',
        'ONNXRUNTIME-LICENSE',
    ]
    assert errors == [
        'package metadata is missing or empty: ONNXRUNTIME-LICENSE',
        'package metadata does not match source: README.md',
    ]


def test_extension_sql_artifacts_include_only_current_install(
    tmp_path: Path,
) -> None:
    install = tmp_path / 'evoke--0.2.1.sql'
    install.touch()

    assert run_product_maturity_suite.extension_sql_artifacts(
        tmp_path,
        '0.2.1',
    ) == [install]


def test_extension_sql_artifacts_reject_extra_paths(
    tmp_path: Path,
) -> None:
    install = tmp_path / 'evoke--0.2.1.sql'
    upgrade = tmp_path / 'evoke--0.2.0--0.2.1.sql'
    extra_a = tmp_path / 'evoke--unsupported-a.sql'
    extra_b = tmp_path / 'evoke--unsupported-b.sql'
    unrelated = tmp_path / 'other--unsupported.sql'
    for path in (install, upgrade, extra_a, extra_b, unrelated):
        path.touch()

    with pytest.raises(ValueError, match='exactly one current install SQL'):
        run_product_maturity_suite.extension_sql_artifacts(
            tmp_path,
            '0.2.1',
        )


def test_maturity_suite_uses_only_current_catalog_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / 'maturity.json'
    args = SimpleNamespace(
        pg_bin=tmp_path / 'pg-bin',
        package_root=tmp_path / 'package',
        source_package_root=tmp_path / 'source-package',
        output=output,
        lifecycle_docs=10,
        benchmark_docs=10,
        benchmark_repeats=1,
        benchmark_concurrency=1,
        model_path=tmp_path / 'model',
        skip_benchmark=True,
        skip_restart_smoke=False,
        restart_wait_seconds=1.0,
    )
    monkeypatch.setattr(
        run_product_maturity_suite,
        'parse_args',
        lambda: args,
    )
    mock_pg_config_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        run_product_maturity_suite,
        'inspect_package_binding',
        lambda package_root, pg_bin: {
            'passed': True,
            'fingerprint': 'stable-package',
        },
    )
    commands: dict[str, list[str]] = {}

    def record_step(
        name: str,
        cmd: list[str | Path],
    ) -> dict[str, object]:
        commands[name] = [str(item) for item in cmd]
        return {
            'name': name,
            'command': commands[name],
            'returncode': 0,
            'elapsed_ms': 1.0,
            'stdout_tail': '',
            'stderr_tail': '',
        }

    monkeypatch.setattr(
        run_product_maturity_suite,
        'run_step',
        record_step,
    )

    assert run_product_maturity_suite.main() == 0
    upgrade_steps = [
        name for name in commands
        if 'upgrade' in name.lower()
    ]
    assert upgrade_steps == []
    assert commands['product Python unit tests'] == [
        run_product_maturity_suite.sys.executable,
        '-m',
        'pytest',
        '-q',
    ]
    assert commands['product Python compileall'] == [
        run_product_maturity_suite.sys.executable,
        '-m',
        'compileall',
        '-q',
        'scripts',
        'tests',
    ]
    assert commands['runtime server build'] == [
        'make',
        'runtime-server',
        f'PG_CONFIG={args.pg_bin / "pg_config"}',
    ]
    assert 'independent page-native golden' in commands
    assert 'convergent SAE lexical-first lifecycle smoke' in commands
    assert 'backend memory ownership smoke' in commands
    assert 'eventual semantic maintenance fairness smoke' in commands
    assert 'compact maintenance builder smoke' in commands
    assert 'spill maintenance builder smoke' in commands
    assert 'shared preload lifecycle closure' in commands
    assert 'shared preload automatic warmup smoke' in commands
    assert 'standby exact-root automatic preload smoke' in commands
    assert 'runtime service restart smoke' in commands
    staged_extension_args = [
        '--extension-libdir',
        str(run_product_maturity_suite.staged_install_path(
            args.package_root,
            tmp_path / 'pg-lib',
        )),
        '--extension-control-dir',
        str(run_product_maturity_suite.staged_install_path(
            args.package_root,
            tmp_path / 'pg-share',
        )),
    ]
    staged_steps = (
        'isolated product regression',
        'isolated extension schema placement smoke',
        'retired storage fail-closed and REINDEX recovery smoke',
        'isolated ONNX Runtime ABI smoke',
        'empty and UNLOGGED lifecycle smoke',
        'same-index writer concurrency smoke',
        'SAE two-phase transaction lifecycle smoke',
        'SAE page-native transaction memory and RSS smoke',
        'eventual semantic quarantine smoke',
        'cache publication and allocation failure-safety smoke',
        'runtime service required smoke',
        'runtime service privilege smoke',
        'runtime service temp pg smoke',
        'runtime failover backpressure smoke',
        'physical replication lifecycle smoke',
        'ONNX Runtime resource soak',
        'production model unified lifecycle smoke',
    )
    for step in staged_steps:
        command = commands[step]
        start = command.index('--extension-libdir')
        assert command[start:start + 4] == staged_extension_args
    runtime_temp_pg = commands['runtime service temp pg smoke']
    assert runtime_temp_pg[-2:] == [
        '--runtime-server-binary',
        str(run_product_maturity_suite.REPO_ROOT / 'build/evoke-runtime-server'),
    ]
    failover_smoke = commands['runtime failover backpressure smoke']
    assert failover_smoke[-5:] == [
        '--runtime-server-binary',
        str(run_product_maturity_suite.REPO_ROOT / 'build/evoke-runtime-server'),
        '--runtime-liveness-timeout-ms',
        '3000',
        '--failover-backpressure-only',
    ]
    lifecycle = commands['production model unified lifecycle smoke']
    assert lifecycle[1] == (
        'scripts/test_unified_index_lifecycle_smoke.py'
    )
    source_migration = commands['source-table BM25S migration smoke']
    assert source_migration == [
        run_product_maturity_suite.sys.executable,
        'scripts/test_psql_bm25s_source_migration_smoke.py',
        '--pg-bin',
        str(args.pg_bin),
        '--source-package-root',
        str(args.source_package_root),
        '--output',
        str(output.with_name('maturity.source-migration.json')),
    ]
