#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_ROLE = 'postgres'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate the current ONNX Runtime loader and worker pool in '
            'isolated PostgreSQL.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        type=Path,
        default=Path('/opt/homebrew/opt/postgresql@18/bin'),
    )
    parser.add_argument('--port', type=int, default=55444)
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share directory containing extension/evoke.control, '
            'or the extension directory itself.'
        ),
    )
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def run(
    command: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def smoke_sql() -> str:
    return """
        CREATE EXTENSION evoke;

        DO $$
        DECLARE
            build_info text;
            probe text;
            service_status jsonb;
        BEGIN
            build_info := evoke_onnxruntime_build_info();
            IF build_info NOT LIKE 'enabled:%' THEN
                RAISE EXCEPTION
                    'ONNX Runtime build is disabled: %', build_info;
            END IF;

            probe := evoke_onnxruntime_probe();
            IF probe NOT LIKE 'available:%' THEN
                RAISE EXCEPTION
                    'ONNX Runtime probe failed: %', probe;
            END IF;

            service_status := evoke_runtime_service_status();
            IF (service_status->>'shared_memory_available')::boolean
                    IS DISTINCT FROM true
                OR (service_status->>'worker_ready')::boolean
                    IS DISTINCT FROM true
                OR (service_status->>'worker_count_configured')::int <> 2
                OR (service_status->>'worker_count_ready')::int <> 2
                OR jsonb_array_length(service_status->'workers') <> 2
                OR service_status->>'queue_policy'
                    <> 'bounded_affinity_worker_pool'
                OR service_status->>'queue_timeout_policy' <> 'none'
                OR (service_status->>'queue_capacity')::int <= 0
                OR (service_status->>'document_queue_limit')::int <> 1 THEN
                RAISE EXCEPTION
                    'shared runtime worker pool is not ready: %',
                    service_status;
            END IF;
            IF EXISTS (
                SELECT 1
                FROM jsonb_array_elements(service_status->'workers') AS worker
                WHERE NOT (
                    worker ? 'starts'
                    AND worker ? 'recoveries'
                    AND worker ? 'successes'
                    AND worker ? 'failures'
                    AND worker ? 'runtime_total_us'
                    AND worker ? 'runtime_max_us'
                )
            ) THEN
                RAISE EXCEPTION
                    'runtime worker health counters are incomplete: %',
                    service_status;
            END IF;
        END;
        $$;
    """


def initdb_command(initdb: Path, data_dir: Path) -> list[str]:
    return [
        str(initdb),
        '-D',
        str(data_dir),
        '-A',
        'trust',
        '-U',
        CLUSTER_ROLE,
    ]


def psql_command(
    psql: Path,
    socket_dir: Path,
    port: int,
) -> list[str]:
    return [
        str(psql),
        '-X',
        '-v',
        'ON_ERROR_STOP=1',
        '-U',
        CLUSTER_ROLE,
        '-h',
        str(socket_dir),
        '-p',
        str(port),
        '-d',
        'postgres',
    ]


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    psql = args.pg_bin / 'psql'

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

    with tempfile.TemporaryDirectory(prefix='evoke_onnxruntime_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()

        run(initdb_command(initdb, data_dir))
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as config:
            config.write("\nshared_preload_libraries = 'evoke'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                config.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(args.extension_control_dir).replace(
                    "'",
                    "''",
                )
                config.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            config.write("listen_addresses = ''\n")
            config.write(
                f"unix_socket_directories = '{socket_dir}'\n"
            )
            config.write(f'port = {args.port}\n')

        started = False
        try:
            run(
                [
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    '-l',
                    str(log_path),
                    'start',
                    '-w',
                ]
            )
            started = True
            run(
                psql_command(psql, socket_dir, args.port),
                input_text=smoke_sql(),
            )
        except Exception:
            if log_path.exists():
                print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
            raise
        finally:
            if started:
                run(
                    [
                        str(pg_ctl),
                        '-D',
                        str(data_dir),
                        'stop',
                        '-m',
                        'fast',
                    ],
                    check=False,
                )

    return {
        'api_version': 'evoke_index_v1',
        'route': 'shared ONNX Runtime worker',
        'status': 'passed',
    }


def main() -> None:
    args = parse_args()
    report = run_smoke(args)
    rendered = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
