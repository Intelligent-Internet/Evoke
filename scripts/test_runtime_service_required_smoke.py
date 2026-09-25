#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root
from test_onnxruntime_smoke import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Verify ONNX-backed product queries require the shared '
            'evoke runtime service.'
        ),
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Production model checkout using the current runtime contract.',
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


def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        **kwargs,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result


def psql(psql_bin: Path, socket_dir: Path, port: int, sql: str) -> str:
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


def runtime_required_sql(model_path: Path) -> str:
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

    DO $$
    DECLARE
        service_status jsonb;
    BEGIN
        SELECT evoke_runtime_service_status() INTO service_status;
        IF (
            service_status->>'shared_memory_available'
        )::boolean IS DISTINCT FROM false THEN
            RAISE EXCEPTION
                'runtime service shared memory should be unavailable: %',
                service_status;
        END IF;
        IF (
            service_status->>'backend_model_loading_allowed'
        )::boolean IS DISTINCT FROM false THEN
            RAISE EXCEPTION
                'backend model loading must stay disabled: %',
                service_status;
        END IF;
        IF service_status->>'model_owner' <> 'runtime_worker_pool' THEN
            RAISE EXCEPTION
                'runtime model owner contract changed: %',
                service_status;
        END IF;
        IF service_status->>'queue_policy'
            <> 'bounded_affinity_worker_pool'
            OR service_status->>'queue_timeout_policy' <> 'none'
            OR (service_status->>'document_pipeline_depth')::int <> 0
            OR (service_status->>'document_queue_limit')::int <> 0
            OR (service_status->>'worker_count_configured')::int <> 0 THEN
            RAISE EXCEPTION
                'unavailable runtime capacity contract changed: %',
                service_status;
        END IF;
        IF NOT (service_status ? 'canceled_requests')
            OR NOT (service_status ? 'orphan_responses')
            OR NOT (service_status ? 'worker_recoveries') THEN
            RAISE EXCEPTION
                'runtime recovery counters missing: %',
                service_status;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        EXECUTE format(
            'CREATE INDEX docs_body_idx ON docs USING evoke (body) '
            || 'WITH (sae = true, model_path = %L)',
            '{escaped_path}'
        );
        RAISE EXCEPTION
            'SAE index build should require the shared runtime service';
    EXCEPTION WHEN others THEN
        IF position(
            'evoke SAE runtime service is not available'
            IN SQLERRM
        ) = 0 THEN
            RAISE EXCEPTION
                'unexpected product query failure: %',
                SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        IF to_regclass('docs_body_idx') IS NOT NULL THEN
            RAISE EXCEPTION
                'failed SAE build left a partial index relation';
        END IF;
    END;
    $$;

    CREATE INDEX docs_body_idx ON docs USING evoke (body)
    WITH (sae = false);

    DO $$
    DECLARE
        hit_count int;
        status jsonb;
    BEGIN
        SELECT count(*)
        INTO hit_count
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'semantic gpu',
            3
        );
        SELECT evoke_index_status('docs_body_idx'::regclass)
        INTO status;
        IF hit_count = 0 OR status->>'index_type' <> 'bm25'
            OR (status->>'query_ready')::boolean IS DISTINCT FROM true THEN
            RAISE EXCEPTION
                'BM25-only route should remain available: %', status;
        END IF;
    END;
    $$;
    '''


def arena_required_sql(model_path: Path) -> str:
    escaped_path = str(model_path).replace("'", "''")
    return f'''
    DROP INDEX docs_body_idx;

    DO $$
    BEGIN
        EXECUTE format(
            'CREATE INDEX docs_body_idx ON docs USING evoke (body) '
            || 'WITH (sae = true, model_path = %L)',
            '{escaped_path}'
        );
        RAISE EXCEPTION
            'SAE index build should require the shared arena';
    EXCEPTION WHEN others THEN
        IF position(
            'evoke SAE shared arena is not available'
            IN SQLERRM
        ) = 0 THEN
            RAISE EXCEPTION
                'unexpected shared-arena failure: %',
                SQLERRM;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        IF to_regclass('docs_body_idx') IS NOT NULL THEN
            RAISE EXCEPTION
                'failed SAE arena check left a partial index relation';
        END IF;
    END;
    $$;

    CREATE INDEX docs_body_idx ON docs USING evoke (body)
    WITH (sae = false);

    SELECT count(*)
    FROM evoke_query(
        'docs_body_idx'::regclass,
        'semantic gpu',
        3
    );
    '''


def runtime_enabled_sql(model_path: Path) -> str:
    escaped_path = str(model_path).replace("'", "''")
    return f'''
    DROP INDEX docs_body_idx;

    CREATE INDEX docs_body_idx
    ON docs USING evoke (body)
    WITH (
        sae = true,
        model_path = '{escaped_path}',
        consistency = eventual
    );

    INSERT INTO docs (id, body)
    SELECT
        1000 + ordinal,
        'runtime ownership semantic gpu sentinel ' || ordinal::text
    FROM generate_series(1, 256) AS ordinal;

    DO $$
    DECLARE
        backend_pid int := pg_backend_pid();
        cache_state jsonb;
        hit_count int;
        index_status jsonb;
        service_status jsonb;
    BEGIN
        SELECT count(*)
        INTO hit_count
        FROM evoke_query(
            'docs_body_idx'::regclass,
            'runtime ownership semantic gpu sentinel 256',
            10
        ) AS hit
        JOIN docs AS source ON source.ctid = hit.ctid
        WHERE source.id = 1256;
        SELECT evoke_index_runtime_state_json(
            'docs_body_idx'::regclass
        )
        INTO cache_state;
        SELECT evoke_index_status('docs_body_idx'::regclass)
        INTO index_status;
        SELECT evoke_runtime_service_status()
        INTO service_status;

        IF hit_count = 0 THEN
            RAISE EXCEPTION
                'relation-owned v3 posting query failed: hits=% state=%',
                hit_count,
                cache_state;
        END IF;
        IF index_status->>'index_type' <> 'semantic'
            OR (index_status->>'query_ready')::boolean IS DISTINCT FROM true
            OR index_status#>>'{{generation,primary,storage}}'
                <> 'convergent_segments'
            OR index_status#>>'{{generation,primary,role}}'
                <> 'unified_posting' THEN
            RAISE EXCEPTION
                'SAE v3 relation-page contract changed: %',
                index_status;
        END IF;
        IF (
            service_status->>'shared_memory_available'
        )::boolean IS DISTINCT FROM true
            OR (
                service_status->>'backend_model_loading_allowed'
            )::boolean IS DISTINCT FROM false
            OR service_status->>'model_owner' <> 'runtime_worker_pool'
            OR (service_status->>'worker_count_configured')::int <= 0
            OR (service_status->>'document_pipeline_depth')::int <= 0
            OR (service_status->>'document_queue_limit')::int <> 1
            OR (service_status->>'session_cache_loads')::int8 <= 0
            OR EXISTS (
                SELECT 1
                FROM jsonb_array_elements(
                    service_status->'workers'
                ) AS worker
                WHERE (worker->>'pid')::int = backend_pid
            ) THEN
            RAISE EXCEPTION
                'runtime worker ownership contract changed: backend=% status=%',
                backend_pid,
                service_status;
        END IF;
        IF (
            cache_state#>>'{{shared_preload,available}}'
        )::boolean IS DISTINCT FROM true
            OR (cache_state#>>'{{shared_preload,arena_size}}')::int8 <= 0 THEN
            RAISE EXCEPTION
                'SAE shared arena is not active: %',
                cache_state;
        END IF;
    END;
    $$;
    '''


def main() -> None:
    args = parse_args()
    pg_bin = Path(args.pg_bin)
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    psql_bin = pg_bin / 'psql'

    with tempfile.TemporaryDirectory(prefix='evoke_runtime_required_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        port = 55434
        socket_dir.mkdir()
        model_path = args.model_path.expanduser().resolve()
        if not (model_path / 'manifest.json').is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {model_path}'
            )
        manifest = json.loads(
            (model_path / 'manifest.json').read_text(encoding='utf-8')
        )
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                '--model-path must use the current Evoke model contract'
            )

        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as f:
            if args.extension_libdir is not None:
                libdir = str(
                    args.extension_libdir.expanduser().resolve()
                ).replace("'", "''")
                f.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_root = extension_control_root(
                    args.extension_control_dir
                )
                control_dir = str(control_root).replace("'", "''")
                f.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            f.write("listen_addresses = ''\n")
            f.write("max_worker_processes = 8\n")

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
            try:
                psql(psql_bin, socket_dir, port, runtime_required_sql(
                    model_path,
                ))
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    'stop',
                    '-m',
                    'fast',
                    '-w',
                ])
                started = False
                with (
                    data_dir / 'postgresql.conf'
                ).open('a', encoding='utf-8') as f:
                    f.write("\nshared_preload_libraries = 'evoke'\n")
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
                    arena_required_sql(model_path),
                )
                run([
                    str(pg_ctl),
                    '-D',
                    str(data_dir),
                    'stop',
                    '-m',
                    'fast',
                    '-w',
                ])
                started = False
                with (
                    data_dir / 'postgresql.conf'
                ).open('a', encoding='utf-8') as f:
                    f.write(
                        "evoke.shared_runtime_size = '1MB'\n"
                    )
                    f.write(
                        "evoke.maintenance_timer_interval_ms = '100ms'\n"
                    )
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
                    runtime_enabled_sql(model_path),
                )
            except Exception:
                if log_path.exists():
                    print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
                raise
        finally:
            if started:
                subprocess.run(
                    [str(pg_ctl), '-D', str(data_dir), 'stop', '-m', 'fast'],
                    text=True,
                    cwd=REPO_ROOT,
                    check=False,
                    capture_output=True,
                )

    print('Runtime service required smoke passed')


if __name__ == '__main__':
    main()
