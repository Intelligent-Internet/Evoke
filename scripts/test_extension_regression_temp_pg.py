#!/usr/bin/env python3

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PG_BIN = Path('/opt/homebrew/opt/postgresql@18/bin')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the PostgreSQL extension regression in an isolated, '
            'preloaded temporary cluster.'
        ),
    )
    parser.add_argument('--pg-bin', type=Path, default=DEFAULT_PG_BIN)
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Keep pg_regress output in this directory.',
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    return parser.parse_args()


def find_pg_regress(pg_bin: Path) -> Path:
    pg_config = pg_bin / 'pg_config'
    result = subprocess.run(
        [str(pg_config), '--pgxs'],
        text=True,
        capture_output=True,
        check=True,
    )
    pgxs = Path(result.stdout.strip()).resolve()
    pg_regress = pgxs.parents[1] / 'test/regress/pg_regress'
    if not pg_regress.is_file():
        raise FileNotFoundError(
            f'pg_regress was not found relative to {pgxs}: {pg_regress}'
        )
    return pg_regress


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        return int(listener.getsockname()[1])


def print_failure_artifacts(output_dir: Path) -> None:
    for name in ('regression.diffs', 'regression.out'):
        path = output_dir / name
        if not path.is_file():
            continue
        print(f'--- {name} ---', file=sys.stderr)
        print(path.read_text(encoding='utf-8'), file=sys.stderr)


def run_regression(
    pg_bin: Path,
    work_root: Path,
    output_dir: Path,
    *,
    regression_name: str,
    extension_libdir: Path | None,
    extension_control_dir: Path | None,
) -> int:
    pg_regress = find_pg_regress(pg_bin)
    instance_dir = work_root / 'instance'
    config_path = work_root / 'postgresql.conf'
    config_lines = [
        "shared_preload_libraries = 'evoke'",
        'max_worker_processes = 16',
        # Keep golden SQL output independent of background convergence timing.
        # Dedicated lifecycle tests exercise the maintenance workers.
        'evoke.maintenance_worker_limit = 0',
    ]
    if extension_libdir is not None:
        libdir = str(extension_libdir).replace("'", "''")
        config_lines.append(
            f"dynamic_library_path = '{libdir}:$libdir'"
        )
    if extension_control_dir is not None:
        control_dir = str(extension_control_dir).replace("'", "''")
        config_lines.append(
            f"extension_control_path = '{control_dir}:$system'"
        )
    config_path.write_text(
        '\n'.join(config_lines) + '\n',
        encoding='utf-8',
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(pg_regress),
        f'--inputdir={REPO_ROOT}',
        f'--expecteddir={REPO_ROOT}',
        f'--outputdir={output_dir}',
        f'--bindir={pg_bin}',
        f'--temp-instance={instance_dir}',
        f'--temp-config={config_path}',
        f'--port={reserve_port()}',
        '--dbname=contrib_regression',
        regression_name,
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end='')
    if result.stderr:
        print(result.stderr, end='', file=sys.stderr)
    if result.returncode != 0:
        print_failure_artifacts(output_dir)
    return result.returncode


def main() -> int:
    args = parse_args()
    for executable in ('pg_config', 'initdb', 'postgres', 'psql'):
        path = args.pg_bin / executable
        if not path.is_file():
            raise FileNotFoundError(f'missing PostgreSQL executable: {path}')
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
        extension_libraries = [
            args.extension_libdir / name
            for name in ('evoke.so', 'evoke.dylib')
        ]
        if not any(path.is_file() for path in extension_libraries):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )

    with tempfile.TemporaryDirectory(
        prefix='evoke-extension-regression-'
    ) as temp_dir:
        work_root = Path(temp_dir)
        output_dir = args.output_dir or work_root / 'output'
        result = run_regression(
            args.pg_bin,
            work_root,
            output_dir,
            regression_name='evoke_integration',
            extension_libdir=args.extension_libdir,
            extension_control_dir=args.extension_control_dir,
        )
        if result != 0:
            return result
        print('Isolated extension regression passed')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
