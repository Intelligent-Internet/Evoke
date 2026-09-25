#!/usr/bin/env python3
"""Run an Evoke same-root binary A/B in a private mount namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--socket-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--database', default='postgres')
    parser.add_argument('--postgres-user', default='postgres')
    parser.add_argument('--pg-config', default='pg_config')
    parser.add_argument('--pg-ctl', default='pg_ctl')
    parser.add_argument('--psql', default='psql')
    parser.add_argument('--baseline-binary', type=Path, required=True)
    parser.add_argument('--candidate-binary', type=Path, required=True)
    parser.add_argument(
        '--qualified-index',
        required=True,
        help=(
            'Permanent, non-empty Evoke index whose current persistent root '
            'is being qualified.'
        ),
    )
    parser.add_argument(
        '--library-target',
        dest='library_targets',
        action='append',
        type=Path,
        default=[],
        help=(
            'Additional absolute catalog library path to bind to the tested '
            'binary; repeat for every distinct probin target.'
        ),
    )
    parser.add_argument('--setup-sql', type=Path, required=True)
    parser.add_argument('--run-sql', type=Path, required=True)
    parser.add_argument('--compare-sql', type=Path, required=True)
    parser.add_argument('--normal-probe-sql', type=Path)
    parser.add_argument('--baseline-route', default='full_baseline')
    parser.add_argument('--candidate-route', default='full_candidate')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--guard-pid-file',
        type=Path,
        help='Postmaster PID file that must remain unchanged during the A/B.',
    )
    parser.add_argument(
        '--leave-running',
        choices=('baseline', 'candidate', 'stopped'),
        default='baseline',
    )
    parser.add_argument(
        '--freeze-final-maintenance',
        action='store_true',
        help=(
            'Keep Evoke maintenance frozen on the successful final variant '
            'for a follow-up same-root probe.'
        ),
    )
    parser.add_argument(
        '--command-timeout-seconds',
        type=int,
        default=3600,
    )
    parser.add_argument(
        '--server-option',
        dest='server_options',
        action='append',
        default=[],
        help=(
            'One PostgreSQL server option token to preserve across both '
            'variants; repeat for each token.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def run_command(
    command: Sequence[str],
    *,
    timeout: int | None = None,
    check: bool = True,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        text=True,
        stdout=stdout,
        stderr=stderr,
        timeout=timeout,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = ' '.join(command)
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f'command failed ({rendered}): {message}')
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_guard_pid(path: Path) -> int:
    result = run_command(
        ['sudo', '-n', 'head', '-n', '1', str(path)]
    )
    pid = int(result.stdout.strip())
    if pid <= 1:
        raise RuntimeError(f'invalid guarded postmaster PID: {pid}')
    return pid


def postgres_admin_command(
    args: argparse.Namespace,
    *command: str,
) -> list[str]:
    return [
        'sudo',
        '-n',
        '-u',
        args.postgres_user,
        '--',
        *command,
    ]


def pg_ctl_command(
    args: argparse.Namespace,
    *command: str,
) -> list[str]:
    return postgres_admin_command(
        args,
        args.pg_ctl,
        '-D',
        str(args.data_dir),
        *command,
    )


def psql_command(args: argparse.Namespace, *command: str) -> list[str]:
    return [
        args.psql,
        '-X',
        '-v',
        'ON_ERROR_STOP=1',
        '-h',
        str(args.socket_dir),
        '-p',
        str(args.port),
        '-d',
        args.database,
        *command,
    ]


def namespace_start_command(
    args: argparse.Namespace,
    binary: Path,
    library_targets: Sequence[Path],
    *,
    freeze_maintenance: bool,
) -> list[str]:
    if not library_targets:
        raise ValueError('at least one Evoke library target is required')
    script = """
set -eu
binary="$1"
postgres_user="$2"
pg_ctl="$3"
data_dir="$4"
options="$5"
mode="$6"
shift 6
for library_target in "$@"; do
    mount --bind "$binary" "$library_target"
done
if [ "$mode" = "freeze" ]; then
    options="$options -c evoke.maintenance_worker_limit=0"
    exec runuser -u "$postgres_user" -- "$pg_ctl" -D "$data_dir" \
        -l "$data_dir/evoke-same-root-ab-postgres.log" \
        -o "$options" -w start
fi
exec runuser -u "$postgres_user" -- "$pg_ctl" -D "$data_dir" \
    -l "$data_dir/evoke-same-root-ab-postgres.log" \
    -o "$options" -w start
""".strip()
    start_options = shlex.join([
        '-k',
        str(args.socket_dir),
        '-p',
        str(args.port),
        *getattr(args, 'server_options', []),
    ])
    return [
        'sudo',
        '-n',
        'unshare',
        '--mount',
        '--propagation',
        'private',
        '--fork',
        'sh',
        '-ceu',
        script,
        'sh',
        str(binary),
        args.postgres_user,
        args.pg_ctl,
        str(args.data_dir),
        start_options,
        'freeze' if freeze_maintenance else 'normal',
        *(str(target) for target in library_targets),
    ]


def server_running(args: argparse.Namespace) -> bool:
    result = run_command(
        pg_ctl_command(args, 'status'),
        check=False,
    )
    return result.returncode == 0


def stop_server(args: argparse.Namespace) -> None:
    if not server_running(args):
        return
    run_command(
        pg_ctl_command(args, '-m', 'fast', '-w', 'stop'),
        timeout=args.command_timeout_seconds,
    )


def query_scalar(args: argparse.Namespace, sql: str) -> str:
    result = run_command(
        psql_command(args, '-qAt', '-c', sql),
        timeout=args.command_timeout_seconds,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f'expected one scalar row, got {lines!r}')
    return lines[0]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def assert_endpoint_identity(args: argparse.Namespace) -> None:
    identity = query_scalar(
        args,
        "SELECT current_setting('data_directory') || '|' || "
        "current_setting('port')",
    )
    data_dir, separator, port = identity.rpartition('|')
    if separator != '|' or int(port) != args.port:
        raise RuntimeError(f'unexpected isolated endpoint identity: {identity}')
    if Path(data_dir).resolve() != args.data_dir:
        raise RuntimeError(
            'isolated endpoint uses the wrong data directory: '
            f'expected={args.data_dir}, actual={data_dir}'
        )


def assert_no_active_build(args: argparse.Namespace) -> None:
    active = int(
        query_scalar(
            args,
            'SELECT count(*) FROM pg_stat_progress_create_index',
        )
    )
    if active != 0:
        raise RuntimeError(
            f'refusing binary A/B while {active} index build(s) are active'
        )


def assert_catalog_library_targets(
    args: argparse.Namespace,
    library_targets: Sequence[Path],
) -> None:
    raw_targets = query_scalar(
        args,
        "SELECT coalesce(json_agg(probin ORDER BY probin)::text, '[]') "
        "FROM (SELECT DISTINCT probin FROM pg_proc "
        "WHERE probin LIKE '%evoke%') AS targets",
    )
    catalog_targets = json.loads(raw_targets)
    declared_targets = {
        target.resolve(strict=True)
        for target in library_targets
    }
    declared_names = {str(target) for target in declared_targets}
    for target in declared_targets:
        if target.suffix:
            declared_names.add(str(target.with_suffix('')))
    for catalog_target in catalog_targets:
        if catalog_target in ('$libdir/evoke', '$libdir/evoke.so'):
            continue
        if not catalog_target.startswith('/'):
            raise RuntimeError(
                f'unsupported Evoke catalog library target: {catalog_target}'
            )
        if catalog_target not in declared_names:
            raise RuntimeError(
                'Evoke catalog library target is not isolated by the A/B: '
                f'{catalog_target}'
            )


def assert_qualified_index(args: argparse.Namespace) -> dict[str, Any]:
    index_literal = sql_literal(args.qualified_index)
    raw_identity = query_scalar(
        args,
        'WITH target AS ('
        f' SELECT {index_literal}::regclass AS oid'
        '), identity AS ('
        ' SELECT index_relation.oid::regclass::text AS index_name,'
        ' index_relation.relpersistence AS index_persistence,'
        ' heap_relation.relpersistence AS heap_persistence,'
        ' access_method.amname AS access_method,'
        ' index_catalog.indisvalid AS valid,'
        ' index_catalog.indisready AS ready,'
        ' index_catalog.indislive AS live,'
        ' evoke_index_generation_status_internal('
        ' index_relation.oid)::jsonb AS status'
        ' FROM target'
        ' JOIN pg_class AS index_relation'
        ' ON index_relation.oid = target.oid'
        ' JOIN pg_index AS index_catalog'
        ' ON index_catalog.indexrelid = index_relation.oid'
        ' JOIN pg_class AS heap_relation'
        ' ON heap_relation.oid = index_catalog.indrelid'
        ' JOIN pg_am AS access_method'
        ' ON access_method.oid = index_relation.relam'
        ')'
        ' SELECT jsonb_build_object('
        " 'index_name', index_name,"
        " 'index_persistence', index_persistence,"
        " 'heap_persistence', heap_persistence,"
        " 'access_method', access_method,"
        " 'valid', valid, 'ready', ready, 'live', live,"
        " 'status', status)::text"
        ' FROM identity',
    )
    identity = json.loads(raw_identity)
    status = identity.get('status') or {}
    primary = status.get('primary') or {}
    posting = status.get('posting') or {}
    failures: list[str] = []

    segmented_root = int(primary.get('segment_count') or 0) > 0
    folded_root = (
        primary.get('fragmented') is True
        and int(primary.get('manifest_start_block') or 0) > 0
        and int(primary.get('manifest_pages') or 0) > 0
        and int(primary.get('physical_blocks') or 0) > 1
        and int(primary.get('published_block_high_watermark') or 0) > 1
    )

    if identity.get('index_persistence') != 'p':
        failures.append('index is not permanent')
    if identity.get('heap_persistence') != 'p':
        failures.append('heap is not permanent')
    if identity.get('access_method') != 'evoke':
        failures.append('access method is not evoke')
    for field in ('valid', 'ready', 'live'):
        if identity.get(field) is not True:
            failures.append(f'index is not {field}')
    if status.get('valid') is not True:
        failures.append('generation is not valid')
    if int(status.get('docs') or 0) <= 0:
        failures.append('generation has no documents')
    if not (segmented_root or folded_root):
        failures.append('generation has no persistent posting authority')
    if int(posting.get('record_count') or 0) <= 0:
        failures.append('generation has no posting records')

    if failures:
        detail = ', '.join(failures)
        raise RuntimeError(
            'qualified index is not a persistent non-empty Evoke root: '
            f'{detail}; identity={raw_identity}'
        )
    return identity


def isolated_postmaster_pid(args: argparse.Namespace) -> int:
    result = run_command(
        postgres_admin_command(
            args,
            'head',
            '-n',
            '1',
            str(args.data_dir / 'postmaster.pid'),
        )
    )
    pid = int(result.stdout.strip())
    if pid <= 1:
        raise RuntimeError(f'invalid isolated postmaster PID: {pid}')
    return pid


def namespace_binary_hash(
    args: argparse.Namespace,
    pid: int,
    library_target: Path,
) -> str:
    namespaced_path = Path(f'/proc/{pid}/root') / library_target.relative_to('/')
    result = run_command(
        ['sudo', '-n', 'sha256sum', str(namespaced_path)]
    )
    return result.stdout.split()[0]


def start_server(
    args: argparse.Namespace,
    binary: Path,
    expected_hash: str,
    library_targets: Sequence[Path],
    *,
    freeze_maintenance: bool,
) -> int:
    run_command(
        namespace_start_command(
            args,
            binary,
            library_targets,
            freeze_maintenance=freeze_maintenance,
        ),
        timeout=args.command_timeout_seconds,
    )
    try:
        pid = isolated_postmaster_pid(args)
        for library_target in library_targets:
            actual_hash = namespace_binary_hash(
                args,
                pid,
                library_target,
            )
            if actual_hash != expected_hash:
                raise RuntimeError(
                    'isolated PostgreSQL loaded the wrong Evoke binary: '
                    f'target={library_target}, expected={expected_hash}, '
                    f'actual={actual_hash}'
                )
        assert_endpoint_identity(args)
        if freeze_maintenance:
            limit = int(
                query_scalar(args, 'SHOW evoke.maintenance_worker_limit')
            )
            if limit != 0:
                raise RuntimeError(
                    'isolated measurement server did not freeze maintenance: '
                    f'actual={limit}'
                )
        return pid
    except BaseException as exc:
        try:
            stop_server(args)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as stop_exc:
            raise RuntimeError(
                f'{exc}; failed to stop rejected isolated server: {stop_exc}'
            ) from exc
        raise


def run_sql_file(
    args: argparse.Namespace,
    path: Path,
    output_path: Path,
    *,
    variables: dict[str, str] | None = None,
) -> None:
    command = psql_command(args)
    if variables is not None:
        for name, value in variables.items():
            command.extend(['--set', f'{name}={value}'])
    command.extend(['--file', str(path)])
    with output_path.open('w', encoding='utf-8') as output:
        result = run_command(
            command,
            timeout=args.command_timeout_seconds,
            check=False,
            stdout=output,
        )
    if result.stderr:
        error_path = output_path.with_suffix(output_path.suffix + '.stderr')
        error_path.write_text(result.stderr, encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError(
            f'SQL file failed ({path}); see {output_path} and its stderr log'
        )


def validate_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f'{label} is not a regular file: {resolved}')
    return resolved


def sql_input_manifest(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    inputs = {
        'setup': args.setup_sql,
        'run': args.run_sql,
        'compare': args.compare_sql,
    }
    if args.normal_probe_sql is not None:
        inputs['normal_probe'] = args.normal_probe_sql
    return {
        name: {
            'path': str(path),
            'sha256': sha256_file(path),
        }
        for name, path in inputs.items()
    }


def prepare_output_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError(f'output path is not a directory: {resolved}')
    if next(resolved.iterdir(), None) is not None:
        raise ValueError(f'output directory is not empty: {resolved}')
    return resolved


def assert_data_directory_owner(args: argparse.Namespace) -> None:
    try:
        postgres_account = pwd.getpwnam(args.postgres_user)
    except KeyError as exc:
        raise ValueError(
            f'PostgreSQL OS user does not exist: {args.postgres_user}'
        ) from exc
    actual_uid = args.data_dir.stat().st_uid
    if actual_uid != postgres_account.pw_uid:
        actual_owner = pwd.getpwuid(actual_uid).pw_name
        raise ValueError(
            'PostgreSQL data directory owner does not match '
            '--postgres-user: '
            f'data_dir={args.data_dir}, owner={actual_owner}, '
            f'postgres_user={args.postgres_user}'
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.port < 1 or args.port > 65535:
        raise ValueError('--port must be between 1 and 65535')
    if args.command_timeout_seconds < 1:
        raise ValueError('--command-timeout-seconds must be positive')
    if not args.qualified_index.strip():
        raise ValueError('--qualified-index cannot be empty')
    for option in getattr(args, 'server_options', []):
        if '\x00' in option or '\n' in option or '\r' in option:
            raise ValueError('--server-option cannot contain control bytes')
    args.data_dir = args.data_dir.expanduser().resolve(strict=True)
    assert_data_directory_owner(args)
    args.socket_dir = args.socket_dir.expanduser().resolve(strict=True)
    args.baseline_binary = validate_file(
        args.baseline_binary,
        'baseline binary',
    )
    args.candidate_binary = validate_file(
        args.candidate_binary,
        'candidate binary',
    )
    args.library_targets = [
        validate_file(target, 'library target')
        for target in getattr(args, 'library_targets', [])
    ]
    args.setup_sql = validate_file(args.setup_sql, 'setup SQL')
    args.run_sql = validate_file(args.run_sql, 'run SQL')
    args.compare_sql = validate_file(args.compare_sql, 'compare SQL')
    if args.normal_probe_sql is not None:
        args.normal_probe_sql = validate_file(
            args.normal_probe_sql,
            'normal probe SQL',
        )
    if args.guard_pid_file is not None:
        args.guard_pid_file = Path(
            os.path.abspath(args.guard_pid_file.expanduser())
        )
        run_command(
            ['sudo', '-n', 'test', '-f', str(args.guard_pid_file)]
        )
        if args.guard_pid_file == args.data_dir / 'postmaster.pid':
            raise ValueError(
                '--guard-pid-file must identify a different PostgreSQL cluster'
            )


def final_runtime_binary(
    args: argparse.Namespace,
    baseline_hash: str,
    candidate_hash: str,
) -> tuple[Path, str] | None:
    if args.leave_running == 'baseline':
        return args.baseline_binary, baseline_hash
    if args.leave_running == 'candidate':
        return args.candidate_binary, candidate_hash
    return None


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir = prepare_output_directory(args.output_dir)

    pkglibdir = run_command([args.pg_config, '--pkglibdir']).stdout.strip()
    library_target = Path(pkglibdir) / 'evoke.so'
    if not library_target.is_file():
        raise RuntimeError(f'installed Evoke library is absent: {library_target}')
    library_targets = list(dict.fromkeys([
        library_target.resolve(strict=True),
        *args.library_targets,
    ]))

    host_hashes_before = {
        str(target): sha256_file(target)
        for target in library_targets
    }
    host_hash_before = host_hashes_before[str(library_targets[0])]
    baseline_hash = sha256_file(args.baseline_binary)
    candidate_hash = sha256_file(args.candidate_binary)
    if baseline_hash == candidate_hash:
        raise RuntimeError('baseline and candidate binaries are identical')
    final_runtime = final_runtime_binary(
        args,
        baseline_hash,
        candidate_hash,
    )

    guard_pid_before = (
        read_guard_pid(args.guard_pid_file)
        if args.guard_pid_file is not None
        else None
    )
    manifest: dict[str, Any] = {
        'started_at_epoch': time.time(),
        'runner': {
            'path': str(Path(__file__).resolve()),
            'sha256': sha256_file(Path(__file__).resolve()),
        },
        'sql_inputs': sql_input_manifest(args),
        'data_dir': str(args.data_dir),
        'socket_dir': str(args.socket_dir),
        'port': args.port,
        'database': args.database,
        'library_target': str(library_targets[0]),
        'library_targets': [str(target) for target in library_targets],
        'host_hashes_before': host_hashes_before,
        'host_hash_before': host_hash_before,
        'baseline_hash': baseline_hash,
        'candidate_hash': candidate_hash,
        'qualified_index': args.qualified_index,
        'guard_pid_file': (
            str(args.guard_pid_file)
            if args.guard_pid_file is not None
            else None
        ),
        'guard_pid_before': guard_pid_before,
        'leave_running': args.leave_running,
        'freeze_final_maintenance': args.freeze_final_maintenance,
        'server_options': args.server_options,
        'dry_run': args.dry_run,
    }
    manifest_path = args.output_dir / 'manifest.json'

    if args.dry_run:
        manifest['baseline_start_command'] = namespace_start_command(
            args,
            args.baseline_binary,
            library_targets,
            freeze_maintenance=True,
        )
        manifest['candidate_start_command'] = namespace_start_command(
            args,
            args.candidate_binary,
            library_targets,
            freeze_maintenance=True,
        )
        if final_runtime is not None:
            final_binary = final_runtime[0]
            manifest['restore_start_command'] = namespace_start_command(
                args,
                final_binary,
                library_targets,
                freeze_maintenance=args.freeze_final_maintenance,
            )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        return 0

    original_running = server_running(args)
    completed = False
    failure: BaseException | None = None
    cleanup_errors: list[str] = []
    final_postmaster_pid: int | None = None
    baseline_index_identity: dict[str, Any] | None = None
    candidate_index_identity: dict[str, Any] | None = None
    try:
        if original_running:
            assert_no_active_build(args)
        stop_server(args)

        start_server(
            args,
            args.baseline_binary,
            baseline_hash,
            library_targets,
            freeze_maintenance=True,
        )
        assert_catalog_library_targets(args, library_targets)
        baseline_index_identity = assert_qualified_index(args)
        run_sql_file(
            args,
            args.setup_sql,
            args.output_dir / 'setup.log',
        )
        run_sql_file(
            args,
            args.run_sql,
            args.output_dir / 'baseline.log',
            variables={'route': args.baseline_route},
        )
        stop_server(args)

        start_server(
            args,
            args.candidate_binary,
            candidate_hash,
            library_targets,
            freeze_maintenance=True,
        )
        assert_catalog_library_targets(args, library_targets)
        candidate_index_identity = assert_qualified_index(args)
        run_sql_file(
            args,
            args.run_sql,
            args.output_dir / 'candidate.log',
            variables={'route': args.candidate_route},
        )
        run_sql_file(
            args,
            args.compare_sql,
            args.output_dir / 'comparison.log',
            variables={
                'baseline_route': args.baseline_route,
                'candidate_route': args.candidate_route,
            },
        )
        if getattr(args, 'normal_probe_sql', None) is not None:
            for forward_route in ('auto', 'direct', 'transpose'):
                route = (
                    f'{args.candidate_route}_normal_{forward_route}'
                )
                run_sql_file(
                    args,
                    args.normal_probe_sql,
                    args.output_dir / f'normal-{forward_route}.log',
                    variables={
                        'route': route,
                        'candidate_route': args.candidate_route,
                        'forward_route': forward_route,
                    },
                )
        completed = True
    except BaseException as exc:
        failure = exc
    finally:
        try:
            stop_server(args)
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            cleanup_errors.append(f'could not stop isolated PostgreSQL: {exc}')
        if completed and final_runtime is not None:
            final_binary, final_hash = final_runtime
            try:
                final_postmaster_pid = start_server(
                    args,
                    final_binary,
                    final_hash,
                    library_targets,
                    freeze_maintenance=args.freeze_final_maintenance,
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(
                    'could not start requested final isolated PostgreSQL '
                    f'variant ({args.leave_running}): {exc}'
                )

        host_hashes_after = {
            str(target): sha256_file(target)
            for target in library_targets
        }
        host_hash_after = host_hashes_after[str(library_targets[0])]
        try:
            guard_pid_after = (
                read_guard_pid(args.guard_pid_file)
                if args.guard_pid_file is not None
                else None
            )
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            guard_pid_after = None
            cleanup_errors.append(f'could not verify guarded postmaster: {exc}')
        output_hashes = {
            path.name: sha256_file(path)
            for path in args.output_dir.iterdir()
            if path.is_file() and path != manifest_path
        }
        manifest.update(
            {
                'completed': completed,
                'finished_at_epoch': time.time(),
                'host_hash_after': host_hash_after,
                'host_hashes_after': host_hashes_after,
                'guard_pid_after': guard_pid_after,
                'final_postmaster_pid': final_postmaster_pid,
                'final_running_variant': (
                    args.leave_running
                    if final_postmaster_pid is not None
                    else 'stopped'
                ),
                'cleanup_errors': cleanup_errors,
                'failure': str(failure) if failure is not None else None,
                'baseline_index_identity': baseline_index_identity,
                'candidate_index_identity': candidate_index_identity,
                'output_hashes': output_hashes,
                'host_binary_unchanged': (
                    host_hashes_after == host_hashes_before
                ),
                'guard_pid_unchanged': (
                    guard_pid_before is None or
                    guard_pid_after == guard_pid_before
                ),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        if host_hashes_after != host_hashes_before:
            raise RuntimeError(
                'host Evoke library targets changed during isolated A/B'
            )
        if (
            guard_pid_before is not None and
            guard_pid_after != guard_pid_before
        ):
            raise RuntimeError('guarded PostgreSQL postmaster changed during A/B')

    if cleanup_errors:
        raise RuntimeError('; '.join(cleanup_errors))
    if failure is not None:
        raise failure

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f'error: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
