#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Validate the configured and active ONNX Runtime providers '
            'through the shared runtime and native unified index.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--providers',
        nargs='+',
        default=['auto'],
        choices=['auto'],
        help='Provider policy to validate. Model checkouts always use auto.',
    )
    parser.add_argument('--iterations', default=5, type=int)
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Current model checkout.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help=(
            'Fail when --providers excludes the auto provider policy '
            'instead of reporting the unrelated provider rows as skipped.'
        ),
    )
    return parser.parse_args()


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    check = bool(kwargs.pop('check', True))
    result = subprocess.run(
        cmd,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        **kwargs,
    )
    if check and result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def psql(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
) -> str:
    result = run(
        [
            str(psql_bin),
            '-X',
            '-q',
            '-t',
            '-A',
            '-h',
            str(socket_dir),
            '-p',
            str(port),
            '-d',
            'postgres',
            '-v',
            'ON_ERROR_STOP=1',
        ],
        input=sql,
    )
    return result.stdout.strip()


def validate_model_contract(model_path: Path) -> None:
    manifest_path = model_path / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'model manifest is missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (
        manifest.get('schema_version') != 1
        or manifest.get('api_version') != 'evoke_model_v1'
        or manifest.get('runtime_abi') != 'evoke_p2_unified_text_atoms_v2'
    ):
        raise ValueError(
            '--model-path must use the current Evoke model contract'
        )
    if 'runtime_parameters' in manifest:
        raise ValueError('runtime_parameters must not be in model manifests')


def setup_sql(model_path: Path) -> str:
    escaped_path = str(model_path).replace("'", "''")
    return f'''
    CREATE EXTENSION evoke;
    CREATE TABLE docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO docs (id, body) VALUES
        (1, 'alpha cuda'),
        (2, 'semantic gpu'),
        (3, 'optimization graph');

    CREATE INDEX docs_body_idx
    ON docs
    USING evoke (body)
    WITH (
        sae = true,
        model_path = '{escaped_path}'
    );
    '''


def run_provider_case(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    model_path: Path,
    requested_provider: str,
    iterations: int,
) -> dict[str, object]:
    escaped_path = str(model_path).replace("'", "''")
    encodings: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    available: list[str] = []
    active_provider = ''

    for iteration in range(iterations):
        output = psql(
            psql_bin,
            socket_dir,
            port,
            f'''
            SELECT evoke_runtime_service_query_atoms(
                '{escaped_path}',
                'provider matrix iteration {iteration}'
            )::text;
            SELECT count(*)
            FROM evoke_query(
                'docs_body_idx'::regclass,
                'alpha cuda provider matrix {iteration}',
                3
            );
            SELECT evoke_runtime_service_status()::text;
            ''',
        ).splitlines()
        if len(output) != 3:
            raise AssertionError(f'provider matrix output mismatch: {output}')
        encoding = json.loads(output[0])
        if not encoding.get('atoms'):
            raise AssertionError(
                f'shared runtime returned no atoms: {encoding}'
            )
        if int(output[1]) < 1:
            raise AssertionError(f'native query returned no rows: {output}')
        provider = encoding.get('provider')
        if not isinstance(provider, dict):
            raise AssertionError(f'provider metadata is missing: {encoding}')
        if provider.get('requested') != requested_provider:
            raise AssertionError(
                f'requested provider mismatch: {encoding}'
            )
        available = [str(item) for item in provider.get('available', [])]
        active_provider = str(provider.get('active', ''))
        if active_provider not in {'cpu', 'cuda', 'coreml', 'tensorrt'}:
            raise AssertionError(
                f'unsupported active provider: {encoding}'
            )
        status = json.loads(output[2])
        if status.get('provider') != requested_provider:
            raise AssertionError(
                f'service requested provider mismatch: {status}'
            )
        if status.get('active_provider') != active_provider:
            raise AssertionError(
                f'service active provider mismatch: {status}'
            )
        encodings.append(encoding)
        statuses.append(status)

    return {
        'provider': requested_provider,
        'active_provider': active_provider,
        'available_providers': available,
        'iterations': iterations,
        'native_index': True,
        'encoding_last': encodings[-1],
        'service_last': statuses[-1],
        'status': 'passed',
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError('--iterations must be positive')

    model_path = args.model_path.expanduser().resolve()
    validate_model_contract(model_path)
    provider_policy = 'auto'
    if provider_policy not in args.providers:
        message = (
            f'provider policy {provider_policy!r} is not included in '
            f'--providers={args.providers!r}'
        )
        if args.strict:
            raise ValueError(message)
        print(json.dumps({
            'status': 'skipped',
            'reason': message,
            'results': [],
        }, indent=2, sort_keys=True))
        return

    pg_bin = Path(args.pg_bin)
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    psql_bin = pg_bin / 'psql'

    with tempfile.TemporaryDirectory(prefix='evoke_ort_matrix_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        port = 55433
        socket_dir.mkdir()

        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open(
            'a',
            encoding='utf-8',
        ) as config:
            config.write("\nshared_preload_libraries = 'evoke'\n")
            config.write("evoke.shared_runtime_size = '64MB'\n")
            config.write("listen_addresses = ''\n")
            config.write('max_worker_processes = 16\n')

        started = False
        try:
            run([
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-l',
                str(log_path),
                '-o',
                f'-k {socket_dir} -p {port}',
                'start',
                '-w',
            ])
            started = True
            psql(
                psql_bin,
                socket_dir,
                port,
                setup_sql(model_path),
            )
            result = run_provider_case(
                psql_bin,
                socket_dir,
                port,
                model_path,
                provider_policy,
                args.iterations,
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

    print(json.dumps({
        'status': 'passed',
        'provider_policy': provider_policy,
        'results': [result],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
