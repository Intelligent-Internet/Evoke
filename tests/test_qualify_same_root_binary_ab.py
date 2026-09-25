from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pwd
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / 'qualify_same_root_binary_ab.py'
SPEC = importlib.util.spec_from_file_location('same_root_ab', MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FIXTURE_ROOT = ROOT / 'tests' / 'fixtures' / 'cq3_pubmed_full'


def test_namespace_start_command_keeps_paths_as_arguments() -> None:
    args = Namespace(
        postgres_user='postgres',
        pg_ctl='/usr/lib/postgresql/18/bin/pg_ctl',
        data_dir=Path('/data/root with spaces'),
        socket_dir=Path('/tmp/socket;not-shell'),
        port=56544,
        server_options=[
            '-c',
            'shared_preload_libraries=evoke',
        ],
    )
    binary = Path('/data/candidate;not-shell.so')
    target = Path('/usr/lib/postgresql/18/lib/evoke.so')

    command = MODULE.namespace_start_command(
        args,
        binary,
        [target],
        freeze_maintenance=True,
    )

    assert command[:8] == [
        'sudo',
        '-n',
        'unshare',
        '--mount',
        '--propagation',
        'private',
        '--fork',
        'sh',
    ]
    assert str(binary) in command
    assert str(args.data_dir) in command
    assert str(args.socket_dir) in command[-3]
    assert command[-2] == 'freeze'
    assert 'shared_preload_libraries=evoke' in command[-3]
    shell_program = command[9]
    assert str(binary) not in shell_program
    assert str(args.data_dir) not in shell_program
    assert str(args.socket_dir) not in shell_program
    assert 'evoke.maintenance_worker_limit=0' in shell_program
    assert '-l "$data_dir/evoke-same-root-ab-postgres.log"' in shell_program


def test_namespace_restore_command_does_not_freeze_maintenance() -> None:
    args = Namespace(
        postgres_user='postgres',
        pg_ctl='/usr/lib/postgresql/18/bin/pg_ctl',
        data_dir=Path('/data/isolated'),
        socket_dir=Path('/tmp/isolated'),
        port=56544,
        server_options=[],
    )

    command = MODULE.namespace_start_command(
        args,
        Path('/data/baseline.so'),
        [Path('/usr/lib/postgresql/18/lib/evoke.so')],
        freeze_maintenance=False,
    )

    assert command[-2] == 'normal'
    assert 'if [ "$mode" = "freeze" ]' in command[9]


def test_final_runtime_binary_selects_requested_variant() -> None:
    args = Namespace(
        leave_running='candidate',
        baseline_binary=Path('/data/baseline.so'),
        candidate_binary=Path('/data/candidate.so'),
    )

    assert MODULE.final_runtime_binary(
        args,
        'baseline-hash',
        'candidate-hash',
    ) == (Path('/data/candidate.so'), 'candidate-hash')

    args.leave_running = 'baseline'
    assert MODULE.final_runtime_binary(
        args,
        'baseline-hash',
        'candidate-hash',
    ) == (Path('/data/baseline.so'), 'baseline-hash')

    args.leave_running = 'stopped'
    assert MODULE.final_runtime_binary(
        args,
        'baseline-hash',
        'candidate-hash',
    ) is None


def test_failed_run_cannot_start_requested_final_variant() -> None:
    source = MODULE_PATH.read_text(encoding='utf-8')

    assert 'if completed and final_runtime is not None:' in source
    assert 'freeze_maintenance=args.freeze_final_maintenance' in source
    assert "'guard_pid_file': (" in source


def test_qualified_index_requires_persistent_nonempty_root(monkeypatch) -> None:
    identity = {
        'index_name': 'bench.docs_idx',
        'index_persistence': 'p',
        'heap_persistence': 'p',
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
        'status': {
            'valid': True,
            'docs': 3,
            'primary': {'segment_count': 1},
            'posting': {'record_count': 7},
        },
    }
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: json.dumps(identity),
    )

    assert MODULE.assert_qualified_index(
        Namespace(qualified_index='bench.docs_idx')
    ) == identity


def test_qualified_index_accepts_current_folded_root(monkeypatch) -> None:
    identity = {
        'index_name': 'bench.docs_idx',
        'index_persistence': 'p',
        'heap_persistence': 'p',
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
        'status': {
            'valid': True,
            'docs': 7,
            'primary': {
                'segment_count': 0,
                'fragmented': True,
                'manifest_start_block': 42,
                'manifest_pages': 1,
                'physical_blocks': 128,
                'published_block_high_watermark': 128,
            },
            'posting': {'record_count': 7},
        },
    }
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: json.dumps(identity),
    )

    assert MODULE.assert_qualified_index(
        Namespace(qualified_index='bench.docs_idx')
    ) == identity


def test_qualified_index_rejects_root_without_persistent_authority(
    monkeypatch,
) -> None:
    identity = {
        'index_name': 'bench.docs_idx',
        'index_persistence': 'p',
        'heap_persistence': 'p',
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
        'status': {
            'valid': True,
            'docs': 7,
            'primary': {
                'segment_count': 0,
                'fragmented': True,
                'manifest_start_block': 0,
                'manifest_pages': 0,
                'physical_blocks': 1,
                'published_block_high_watermark': 1,
            },
            'posting': {'record_count': 7},
        },
    }
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: json.dumps(identity),
    )

    try:
        MODULE.assert_qualified_index(
            Namespace(qualified_index='bench.docs_idx')
        )
    except RuntimeError as exc:
        assert 'generation has no persistent posting authority' in str(exc)
    else:
        raise AssertionError('root without persistent authority was accepted')


def test_qualified_index_rejects_unlogged_or_empty_root(monkeypatch) -> None:
    identity = {
        'index_name': 'bench.docs_idx',
        'index_persistence': 'u',
        'heap_persistence': 'u',
        'access_method': 'evoke',
        'valid': True,
        'ready': True,
        'live': True,
        'status': {
            'valid': True,
            'docs': 0,
            'primary': {'segment_count': 0},
            'posting': {'record_count': 0},
        },
    }
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: json.dumps(identity),
    )

    try:
        MODULE.assert_qualified_index(
            Namespace(qualified_index='bench.docs_idx')
        )
    except RuntimeError as exc:
        assert 'index is not permanent' in str(exc)
        assert 'heap is not permanent' in str(exc)
        assert 'generation has no documents' in str(exc)
    else:
        raise AssertionError('unlogged empty root was accepted')


def test_start_server_stops_endpoint_rejected_after_start(monkeypatch) -> None:
    args = Namespace(command_timeout_seconds=60)
    stopped: list[bool] = []

    monkeypatch.setattr(
        MODULE,
        'namespace_start_command',
        lambda *unused_args, **unused_kwargs: ['start'],
    )
    monkeypatch.setattr(
        MODULE,
        'run_command',
        lambda *unused_args, **unused_kwargs: Namespace(returncode=0),
    )
    monkeypatch.setattr(MODULE, 'isolated_postmaster_pid', lambda unused: 42)
    monkeypatch.setattr(
        MODULE,
        'namespace_binary_hash',
        lambda *unused_args: 'expected',
    )
    monkeypatch.setattr(
        MODULE,
        'assert_endpoint_identity',
        lambda unused: (_ for _ in ()).throw(RuntimeError('wrong endpoint')),
    )
    monkeypatch.setattr(
        MODULE,
        'stop_server',
        lambda unused: stopped.append(True),
    )

    try:
        MODULE.start_server(
            args,
            Path('/data/candidate.so'),
            'expected',
            [Path('/usr/lib/postgresql/18/lib/evoke.so')],
            freeze_maintenance=False,
        )
    except RuntimeError as exc:
        assert str(exc) == 'wrong endpoint'
    else:
        raise AssertionError('rejected endpoint was accepted')

    assert stopped == [True]


def test_start_server_verifies_every_library_target(monkeypatch) -> None:
    args = Namespace(command_timeout_seconds=60)
    targets = [
        Path('/usr/lib/postgresql/18/lib/evoke.so'),
        Path('/data/staged/lib/evoke'),
    ]
    observed: list[Path] = []

    monkeypatch.setattr(
        MODULE,
        'namespace_start_command',
        lambda *unused_args, **unused_kwargs: ['start'],
    )
    monkeypatch.setattr(
        MODULE,
        'run_command',
        lambda *unused_args, **unused_kwargs: Namespace(returncode=0),
    )
    monkeypatch.setattr(MODULE, 'isolated_postmaster_pid', lambda unused: 42)

    def fake_hash(unused_args, unused_pid, target):
        observed.append(target)
        return 'expected'

    monkeypatch.setattr(MODULE, 'namespace_binary_hash', fake_hash)
    monkeypatch.setattr(MODULE, 'assert_endpoint_identity', lambda unused: None)

    MODULE.start_server(
        args,
        Path('/data/candidate.so'),
        'expected',
        targets,
        freeze_maintenance=False,
    )

    assert observed == targets


def test_psql_command_is_bound_to_isolated_endpoint() -> None:
    args = Namespace(
        psql='/usr/lib/postgresql/18/bin/psql',
        socket_dir=Path('/tmp/isolated'),
        port=56544,
        database='postgres',
    )

    command = MODULE.psql_command(args, '-qAt', '-c', 'SELECT 1')

    assert command == [
        '/usr/lib/postgresql/18/bin/psql',
        '-X',
        '-v',
        'ON_ERROR_STOP=1',
        '-h',
        '/tmp/isolated',
        '-p',
        '56544',
        '-d',
        'postgres',
        '-qAt',
        '-c',
        'SELECT 1',
    ]


def test_run_sql_file_passes_named_variables(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = Namespace(
        psql='/usr/lib/postgresql/18/bin/psql',
        socket_dir=Path('/tmp/isolated'),
        port=56544,
        database='postgres',
        command_timeout_seconds=60,
    )
    sql_path = tmp_path / 'compare.sql'
    output_path = tmp_path / 'compare.log'
    commands: list[list[str]] = []
    sql_path.write_text('SELECT 1;\n', encoding='utf-8')

    def fake_run_command(command, **unused_kwargs):
        commands.append(list(command))
        return Namespace(returncode=0, stderr='')

    monkeypatch.setattr(MODULE, 'run_command', fake_run_command)

    MODULE.run_sql_file(
        args,
        sql_path,
        output_path,
        variables={
            'baseline_route': 'baseline-custom',
            'candidate_route': 'candidate-custom',
        },
    )

    assert '--set' in commands[0]
    assert 'baseline_route=baseline-custom' in commands[0]
    assert 'candidate_route=candidate-custom' in commands[0]


def test_sha256_file_reads_complete_payload(tmp_path: Path) -> None:
    payload = tmp_path / 'payload.bin'
    payload.write_bytes(b'a' * (1024 * 1024 + 17))

    assert MODULE.sha256_file(payload) == hashlib.sha256(
        payload.read_bytes()
    ).hexdigest()


def test_sql_input_manifest_binds_optional_normal_probe(
    tmp_path: Path,
) -> None:
    paths = {
        name: tmp_path / f'{name}.sql'
        for name in ('setup', 'run', 'compare', 'normal_probe')
    }
    for name, path in paths.items():
        path.write_text(f'SELECT {name!r};\n', encoding='utf-8')
    args = Namespace(
        setup_sql=paths['setup'],
        run_sql=paths['run'],
        compare_sql=paths['compare'],
        normal_probe_sql=paths['normal_probe'],
    )

    inputs = MODULE.sql_input_manifest(args)

    assert set(inputs) == {'setup', 'run', 'compare', 'normal_probe'}
    for name, path in paths.items():
        assert inputs[name] == {
            'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_validate_args_rejects_missing_normal_probe(
    tmp_path: Path,
) -> None:
    binary = tmp_path / 'evoke.so'
    setup_sql = tmp_path / 'setup.sql'
    run_sql = tmp_path / 'run.sql'
    compare_sql = tmp_path / 'compare.sql'
    for path in (binary, setup_sql, run_sql, compare_sql):
        path.write_bytes(b'input')
    socket_dir = tmp_path / 'socket'
    socket_dir.mkdir()
    args = Namespace(
        port=56544,
        command_timeout_seconds=60,
        qualified_index='bench.docs_idx',
        server_options=[],
        data_dir=tmp_path,
        postgres_user=pwd.getpwuid(os.getuid()).pw_name,
        socket_dir=socket_dir,
        baseline_binary=binary,
        candidate_binary=binary,
        library_targets=[],
        setup_sql=setup_sql,
        run_sql=run_sql,
        compare_sql=compare_sql,
        normal_probe_sql=tmp_path / 'missing.sql',
        guard_pid_file=None,
    )

    try:
        MODULE.validate_args(args)
    except FileNotFoundError as exc:
        assert exc.filename == str(tmp_path / 'missing.sql')
    else:
        raise AssertionError('missing normal probe SQL was accepted')


def test_prepare_output_directory_requires_clean_run_directory(
    tmp_path: Path,
) -> None:
    new_output = tmp_path / 'new' / 'run'

    assert MODULE.prepare_output_directory(new_output) == new_output
    assert new_output.is_dir()

    (new_output / 'stale.log').write_text('old run\n', encoding='utf-8')
    try:
        MODULE.prepare_output_directory(new_output)
    except ValueError as exc:
        assert 'output directory is not empty' in str(exc)
    else:
        raise AssertionError('non-empty output directory was accepted')


def test_data_directory_owner_accepts_matching_os_user(tmp_path: Path) -> None:
    account = pwd.getpwuid(os.getuid())
    args = Namespace(
        data_dir=tmp_path,
        postgres_user=account.pw_name,
    )

    MODULE.assert_data_directory_owner(args)


def test_data_directory_owner_rejects_mismatched_os_user(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = Namespace(
        data_dir=tmp_path,
        postgres_user='wrong-owner',
    )
    monkeypatch.setattr(
        MODULE.pwd,
        'getpwnam',
        lambda unused_name: SimpleNamespace(pw_uid=os.getuid() + 1),
    )
    monkeypatch.setattr(
        MODULE.pwd,
        'getpwuid',
        lambda unused_uid: SimpleNamespace(pw_name='actual-owner'),
    )

    try:
        MODULE.assert_data_directory_owner(args)
    except ValueError as exc:
        assert 'owner=actual-owner' in str(exc)
        assert 'postgres_user=wrong-owner' in str(exc)
    else:
        raise AssertionError('mismatched PostgreSQL OS user was accepted')


def test_endpoint_identity_accepts_only_requested_cluster(monkeypatch) -> None:
    args = Namespace(
        data_dir=Path('/data/isolated'),
        port=56544,
    )
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: '/data/isolated|56544',
    )

    MODULE.assert_endpoint_identity(args)


def test_endpoint_identity_rejects_other_cluster(monkeypatch) -> None:
    args = Namespace(
        data_dir=Path('/data/isolated'),
        port=56544,
    )
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: '/data/main|5432',
    )

    try:
        MODULE.assert_endpoint_identity(args)
    except RuntimeError as exc:
        assert 'unexpected isolated endpoint identity' in str(exc)
    else:
        raise AssertionError('wrong PostgreSQL endpoint was accepted')


def test_catalog_library_targets_require_every_absolute_probin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    installed = tmp_path / 'installed-evoke.so'
    staged = tmp_path / 'staged-evoke.so'
    catalog_staged = staged.with_suffix('')
    installed.write_bytes(b'binary')
    staged.write_bytes(b'binary')
    monkeypatch.setattr(
        MODULE,
        'query_scalar',
        lambda unused_args, unused_sql: (
            f'["$libdir/evoke", "{catalog_staged}"]'
        ),
    )

    try:
        MODULE.assert_catalog_library_targets(
            Namespace(),
            [installed],
        )
    except RuntimeError as exc:
        assert str(catalog_staged) in str(exc)
    else:
        raise AssertionError('unisolated absolute catalog target was accepted')

    MODULE.assert_catalog_library_targets(
        Namespace(),
        [installed, staged],
    )


def test_full_root_fixture_guards_generation_identity() -> None:
    setup_sql = (FIXTURE_ROOT / 'setup.sql').read_text(encoding='utf-8')
    run_sql = (FIXTURE_ROOT / 'run.sql').read_text(encoding='utf-8')
    normal_sql = (FIXTURE_ROOT / 'normal_probe.sql').read_text(
        encoding='utf-8'
    )
    compare_sql = (FIXTURE_ROOT / 'compare.sql').read_text(
        encoding='utf-8'
    )

    assert 'cq3_full_root_identity' in setup_sql
    assert "'{primary,manifest_start_block}'" in setup_sql
    assert "'{primary,published_block_high_watermark}'" in setup_sql
    assert "'before'" in run_sql
    assert "'after'" in run_sql
    assert 'evoke_index_generation_status_internal' in run_sql
    assert 'count(DISTINCT identity) = 1' in compare_sql
    assert 'root identity changed during A/B' in compare_sql
    assert "WHERE name IN ('p006', 'unfiltered')" not in normal_sql
    for telemetry in (
        'ranked_prefix_probe_postings_examined',
        'accelerator_forward_postings_examined',
        'accelerator_owned_index_bytes',
        'accelerator_candidate_scratch_bytes',
        'accelerator_residual_scratch_bytes',
        'semantic_bmp_query_bytes',
    ):
        assert telemetry in normal_sql


def test_runner_executes_all_same_root_forward_diagnostics() -> None:
    source = MODULE_PATH.read_text(encoding='utf-8')

    assert "parser.add_argument('--normal-probe-sql', type=Path)" in source
    assert "for forward_route in ('auto', 'direct', 'transpose'):" in source
    assert "'candidate_route': args.candidate_route" in source
    assert "'forward_route': forward_route" in source
