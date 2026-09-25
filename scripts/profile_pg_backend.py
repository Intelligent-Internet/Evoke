#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import psycopg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a benchmark command and sample its PostgreSQL '
        'backend with macOS sample.'
    )
    parser.add_argument(
        '--dsn',
        default=os.environ.get(
            'EVOKE_BENCH_DSN',
            'dbname=postgres user=leask',
        ),
        help='Connection string used to inspect pg_stat_activity.',
    )
    parser.add_argument(
        '--application-name',
        required=True,
        help='Exact pg_stat_activity application_name to watch.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory where stdout/stderr/sample metadata will be stored.',
    )
    parser.add_argument(
        '--sample-seconds',
        type=int,
        default=5,
        help='Sampling duration in seconds.',
    )
    parser.add_argument(
        '--sample-interval-ms',
        type=int,
        default=1,
        help='Sampling interval in milliseconds.',
    )
    parser.add_argument(
        '--wait-timeout',
        type=float,
        default=30.0,
        help='How long to wait for the backend to appear.',
    )
    parser.add_argument(
        '--env',
        action='append',
        default=[],
        help='Extra environment variable in KEY=VALUE form.',
    )
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Command to run after "--".',
    )
    args = parser.parse_args()
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command:
        parser.error('missing benchmark command after "--"')
    return args


def find_backend_pid(dsn: str, application_name: str) -> int | None:
    sql_text = """
        SELECT pid
        FROM pg_stat_activity
        WHERE application_name = %s
          AND pid <> pg_backend_pid()
        ORDER BY
            CASE WHEN state = 'active' THEN 0 ELSE 1 END,
            query_start NULLS LAST
        LIMIT 1
    """
    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_text, (application_name,))
            row = cur.fetchone()
    if row is None:
        return None
    return int(row[0])


def wait_for_backend_pid(
    dsn: str,
    application_name: str,
    timeout: float,
    proc: subprocess.Popen[str],
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = find_backend_pid(dsn, application_name)
        if pid is not None:
            return pid
        if proc.poll() is not None:
            raise RuntimeError(
                'benchmark command exited before backend appeared'
            )
        time.sleep(0.1)
    raise RuntimeError('timed out waiting for PostgreSQL backend')


def run_sample(
    pid: int,
    output_file: Path,
    seconds: int,
    interval_ms: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        '/usr/bin/sample',
        str(pid),
        str(seconds),
        str(interval_ms),
        '-mayDie',
        '-file',
        str(output_file),
    ]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stdout_file = output_dir / 'benchmark.stdout.log'
    stderr_file = output_dir / 'benchmark.stderr.log'
    sample_file = output_dir / 'backend.sample.txt'
    metadata_file = output_dir / 'metadata.json'
    child_env = os.environ.copy()
    for item in args.env:
        if '=' not in item:
            raise SystemExit(f'invalid --env entry: {item!r}')
        key, value = item.split('=', 1)
        child_env[key] = value

    started = time.time()
    with stdout_file.open('w', encoding='utf-8') as stdout_fp, \
            stderr_file.open('w', encoding='utf-8') as stderr_fp:
        proc = subprocess.Popen(
            args.command,
            stdout=stdout_fp,
            stderr=stderr_fp,
            text=True,
            env=child_env,
        )
        backend_pid = wait_for_backend_pid(
            args.dsn,
            args.application_name,
            args.wait_timeout,
            proc,
        )
        sample_result = run_sample(
            backend_pid,
            sample_file,
            args.sample_seconds,
            args.sample_interval_ms,
        )
        command_rc = proc.wait()

    finished = time.time()
    metadata = {
        'application_name': args.application_name,
        'command': args.command,
        'command_pretty': ' '.join(shlex.quote(part) for part in args.command),
        'backend_pid': backend_pid,
        'command_returncode': command_rc,
        'env': {
            key: child_env[key]
            for key in sorted(set(item.split('=', 1)[0] for item in args.env))
        },
        'sample_returncode': sample_result.returncode,
        'sample_stderr': sample_result.stderr,
        'sample_stdout': sample_result.stdout,
        'started_at_epoch_s': started,
        'finished_at_epoch_s': finished,
        'duration_s': finished - started,
        'artifacts': {
            'stdout': str(stdout_file),
            'stderr': str(stderr_file),
            'sample': str(sample_file),
        },
    }
    metadata_file.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    if command_rc != 0:
        print(
            f'benchmark command failed with rc={command_rc}; '
            f'see {stdout_file} and {stderr_file}',
            file=sys.stderr,
        )
        return command_rc
    if sample_result.returncode != 0:
        print(
            f'sample failed with rc={sample_result.returncode}; '
            f'see {metadata_file}',
            file=sys.stderr,
        )
        return sample_result.returncode

    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
