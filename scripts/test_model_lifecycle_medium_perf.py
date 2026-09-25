#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parents[1]
MAINTENANCE_TIMEOUT_SECONDS = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a medium native BM25+SAE lifecycle/perf smoke.',
    )
    parser.add_argument(
        '--pg-bin',
        default='/opt/homebrew/opt/postgresql@18/bin',
        help='Directory containing initdb, pg_ctl, and psql.',
    )
    parser.add_argument(
        '--docs',
        type=int,
        default=50_000,
        help='Number of synthetic medium-corpus documents.',
    )
    parser.add_argument(
        '--model-path',
        type=Path,
        required=True,
        help='Current production model checkout.',
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
    parser.add_argument(
        '--output',
        type=Path,
        help='Optional JSON output path in addition to stdout.',
    )
    parser.add_argument('--port', type=int, default=55433)
    parser.add_argument(
        '--observe-build-rss',
        action='store_true',
        help='Sample build-backend RSS by pg_stat_activity phase.',
    )
    parser.add_argument(
        '--require-native-timeout',
        action='store_true',
        help=(
            'Require the fixed-atoms native query to exceed its deadline; '
            'use only for corpora large enough to make the scorer run longer '
            'than 10ms.'
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


def timed(
    metrics: dict[str, Any],
    label: str,
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
) -> str:
    started = time.perf_counter()
    output = psql(psql_bin, socket_dir, port, sql)
    metrics[f'{label}_ms'] = round((time.perf_counter() - started) * 1000, 3)
    return output


def timed_statement_timeout_response(
    metrics: dict[str, Any],
    label: str,
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
    timeout_ms: int,
    response_limit_ms: int,
    require_timeout: bool,
) -> None:
    started = time.perf_counter()
    result = subprocess.run(
        psql_command(psql_bin, socket_dir, port),
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        input=(
            f"SET statement_timeout = '{timeout_ms}ms';\n"
            f'{sql}\n'
        ),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    diagnostics = f'{result.stdout}\n{result.stderr}'.lower()
    if result.returncode == 0:
        if require_timeout:
            raise AssertionError(
                'native query completed before the required statement '
                f'timeout: {elapsed_ms}ms'
            )
        outcome = 'completed_before_timeout'
    elif 'statement timeout' in diagnostics:
        outcome = 'cancelled'
    else:
        raise AssertionError(
            'native query failed for a reason other than statement timeout: '
            f'returncode={result.returncode}, output={diagnostics.strip()}'
        )
    if outcome == 'cancelled' and elapsed_ms > response_limit_ms:
        raise AssertionError(
            'native query did not observe cancellation promptly: '
            f'{elapsed_ms}ms > {response_limit_ms}ms'
        )
    metrics[f'{label}_ms'] = elapsed_ms
    metrics[f'{label}_outcome'] = outcome
    metrics[f'{label}_timeout_ms'] = timeout_ms
    metrics[f'{label}_response_limit_ms'] = response_limit_ms


def restart_postgres(
    pg_ctl: Path,
    data_dir: Path,
    log_path: Path,
    socket_dir: Path,
    port: int,
) -> float:
    started = time.perf_counter()
    run([
        str(pg_ctl),
        '-D',
        str(data_dir),
        '-l',
        str(log_path),
        '-o',
        f'-k {socket_dir} -p {port}',
        'restart',
        '-m',
        'fast',
        '-w',
    ])
    return round((time.perf_counter() - started) * 1000, 3)


def psql_command(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
) -> list[str]:
    return [
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
    ]


def backend_phase_and_rss(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    application_name: str,
) -> tuple[str, int] | None:
    escaped_name = application_name.replace("'", "''")
    activity = subprocess.run(
        psql_command(psql_bin, socket_dir, port) + [
            '-c',
            (
                'SELECT pid, query FROM pg_stat_activity '
                f"WHERE application_name = '{escaped_name}' "
                "AND backend_type = 'client backend' LIMIT 1"
            ),
        ],
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if activity.returncode != 0 or not activity.stdout.strip():
        return None
    pid_text, _, phase = activity.stdout.strip().partition('|')
    if not pid_text.isdigit():
        return None
    rss = subprocess.run(
        ['ps', '-o', 'rss=', '-p', pid_text],
        text=True,
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    rss_text = rss.stdout.strip()
    if rss.returncode != 0 or not rss_text.isdigit():
        return None
    if not phase.startswith('evoke build:'):
        phase = 'sql'
    return phase, int(rss_text) * 1024


def timed_with_backend_rss(
    metrics: dict[str, Any],
    label: str,
    psql_bin: Path,
    socket_dir: Path,
    port: int,
    sql: str,
) -> str:
    application_name = f'evoke_{label}'
    environment = dict(os.environ)
    environment['PGAPPNAME'] = application_name
    started = time.perf_counter()
    process = subprocess.Popen(
        psql_command(psql_bin, socket_dir, port),
        text=True,
        cwd=REPO_ROOT,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None:
        process.kill()
        raise RuntimeError('failed to open psql input pipe')
    process.stdin.write(sql)
    process.stdin.close()

    phase_peaks: dict[str, int] = {}
    peak_rss = 0
    sample_count = 0
    while process.poll() is None:
        sample = backend_phase_and_rss(
            psql_bin,
            socket_dir,
            port,
            application_name,
        )
        if sample is not None:
            phase, rss_bytes = sample
            phase_peaks[phase] = max(phase_peaks.get(phase, 0), rss_bytes)
            peak_rss = max(peak_rss, rss_bytes)
            sample_count += 1
        time.sleep(0.02)

    stdout = process.stdout.read() if process.stdout is not None else ''
    stderr = process.stderr.read() if process.stderr is not None else ''
    metrics[f'{label}_ms'] = round(
        (time.perf_counter() - started) * 1000,
        3,
    )
    metrics[f'{label}_rss'] = {
        'peak_bytes': peak_rss,
        'phase_peak_bytes': phase_peaks,
        'samples': sample_count,
    }
    if process.returncode != 0:
        if stdout:
            print(stdout, file=sys.stderr)
        if stderr:
            print(stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(
            process.returncode,
            process.args,
            output=stdout,
            stderr=stderr,
        )
    return stdout.strip()


def setup_sql(model_path: Path, docs: int) -> str:
    escaped_path = str(model_path).replace("'", "''")
    return f'''
    CREATE EXTENSION evoke;
    CREATE SCHEMA medium;
    CREATE TABLE medium.docs (
        id int PRIMARY KEY,
        body text NOT NULL
    );
    INSERT INTO medium.docs (id, body)
    SELECT
        i,
        concat_ws(
            ' ',
            'document', i,
            CASE i % 6
                WHEN 0 THEN 'cuda graph neural network optimization'
                WHEN 1 THEN 'semantic sparse atoms evidence retrieval'
                WHEN 2 THEN 'postgres maintenance eventual consistency'
                WHEN 3 THEN 'bm25 lexical exact token ranking'
                WHEN 4 THEN 'policy search memory extraction agent rag'
                ELSE 'scientific article latent concept index'
            END,
            'topic', i % 997,
            'bucket', i % 127
        )
    FROM generate_series(1, {docs}) AS i;

    CREATE INDEX docs_body_idx
    ON medium.docs
    USING evoke (body)
    WITH (
        sae = true,
        model_path = '{escaped_path}',
        consistency = eventual
    );
    '''


def query_sql(query: str = 'cuda graph neural network optimization') -> str:
    escaped_query = query.replace("'", "''")
    return f'''
    WITH hits AS MATERIALIZED (
        SELECT *
        FROM evoke_query(
            'medium.docs_body_idx'::regclass,
            '{escaped_query}',
            20
        )
    )
    SELECT jsonb_build_object(
        'count', count(*),
        'max_score', max(score)
    )
    FROM hits;
    '''


def encode_query_sql(
    query: str = 'cuda graph neural network optimization',
) -> str:
    escaped_query = query.replace("'", "''")
    return (
        "SELECT evoke_encode_text_internal("
        "'medium.docs_body_idx'::regclass, "
        f"'{escaped_query}');"
    )


def native_query_sql(encoded: dict[str, Any], limit: int = 20) -> str:
    dims = ','.join(str(int(value)) for value in encoded['atoms'])
    weights = ','.join(str(float(value)) for value in encoded['weights'])
    signature = str(encoded['runtime_signature']).replace("'", "''")
    return f'''
    SELECT count(*)
    FROM evoke_index_semantic_query_native_internal(
        'medium.docs_body_idx'::regclass,
        ARRAY[{dims}]::int4[],
        ARRAY[{weights}]::real[],
        NULL::text[],
        NULL::real[],
        {limit},
        '{signature}'
    );
    '''


def query_diagnostics_sql(
    query: str = 'cuda graph neural network optimization',
) -> str:
    escaped_query = query.replace("'", "''")
    return f'''
    WITH encoded AS MATERIALIZED (
        SELECT evoke_encode_text_internal(
            'medium.docs_body_idx'::regclass,
            '{escaped_query}'
        ) AS value
    ),
    atoms AS MATERIALIZED (
        SELECT
            ARRAY(
                SELECT atom.value::int4
                FROM jsonb_array_elements_text(encoded.value->'atoms')
                    AS atom(value)
            ) AS dims,
            ARRAY(
                SELECT weight.value::real
                FROM jsonb_array_elements_text(encoded.value->'weights')
                    AS weight(value)
            ) AS weights,
            encoded.value->>'runtime_signature' AS runtime_signature
        FROM encoded
    ),
    raw AS MATERIALIZED (
        SELECT candidate.*
        FROM atoms,
             evoke_index_semantic_query_native_internal(
                 'medium.docs_body_idx'::regclass,
                 atoms.dims,
                 atoms.weights,
                 NULL::text[],
                 NULL::real[],
                 20,
                 atoms.runtime_signature
             ) AS candidate
    )
    SELECT jsonb_build_object(
        'raw_count', (SELECT count(*) FROM raw),
        'joined_count', (
            SELECT count(*)
            FROM raw
            JOIN medium.docs AS source ON source.ctid = raw.ctid
        ),
        'raw_tids', COALESCE(
            (
                SELECT jsonb_agg(sample.ctid::text ORDER BY sample.rank)
                FROM (SELECT rank, ctid FROM raw ORDER BY rank LIMIT 5) AS sample
            ),
            '[]'::jsonb
        ),
        'status', evoke_index_status('medium.docs_body_idx'::regclass)
    );
    '''


def mutation_sql(docs: int) -> str:
    inserted_id = docs + 1
    return f'''
    INSERT INTO medium.docs (id, body)
    VALUES (
        {inserted_id},
        'new cuda graph optimization semantic atom document'
    );
    UPDATE medium.docs
    SET body = body || ' updated semantic replacement'
    WHERE id = 1;
    DELETE FROM medium.docs WHERE id = 2;
    VACUUM (INDEX_CLEANUP ON) medium.docs;
    '''


def assert_health_sql(docs: int) -> str:
    return f'''
    DO $$
    DECLARE
        options jsonb;
        status jsonb;
        generation jsonb;
        generation_docs int8;
        document_slot_high_watermark int8;
        posting_record_count int8;
        row_count int8;
        sidecar_count int8;
    BEGIN
        SELECT count(*) INTO row_count FROM medium.docs;
        IF row_count <> {docs} THEN
            RAISE EXCEPTION 'unexpected row count after CRUD: %', row_count;
        END IF;

        options := evoke_index_options('medium.docs_body_idx'::regclass);
        status := evoke_index_status('medium.docs_body_idx'::regclass);
        generation := status->'generation';
        generation_docs := (generation->>'docs')::int8;
        document_slot_high_watermark := (
            generation->>'document_slot_high_watermark'
        )::int8;
        posting_record_count := (
            generation#>>'{{posting,record_count}}'
        )::int8;
        IF options->>'index_type' <> 'semantic'
            OR options->>'payload_owner' <> 'index_relation'
            OR options->>'lifecycle' <> 'postgresql_index'
            OR (status->>'query_ready')::boolean IS DISTINCT FROM true
            OR status->>'blocker' <> 'none'
            OR (generation->>'atomic')::boolean IS DISTINCT FROM true
            OR (generation->>'valid')::boolean IS DISTINCT FROM true
            OR generation#>>'{{layout,storage}}'
                <> 'convergent_segments'
            OR generation#>>'{{primary,role}}' <> 'unified_posting'
            OR generation#>>'{{primary,storage}}'
                <> 'convergent_segments'
            OR (generation#>>'{{primary,physical_blocks}}')::int <= 0
            OR (generation#>>'{{primary,reachable_blocks}}')::int <= 0
            OR generation#>>'{{posting,role}}' <> 'unified_posting'
            OR (generation#>>'{{posting,active}}')::boolean
                IS DISTINCT FROM true
            OR generation->>'docs_scope' <> 'sealed_generation'
            OR generation_docs < row_count
            OR generation_docs > document_slot_high_watermark
            OR posting_record_count <> generation_docs
            OR (generation#>>'{{delta,records}}')::int <> 0
            OR (generation#>>'{{delta,active,records}}')::int <> 0
            OR (generation#>>'{{delta,pending,records}}')::int <> 0
            OR (
                generation#>>'{{delta,semantic_completion,pending}}'
            )::int <> 0
            OR (
                generation#>>'{{delta,semantic_completion,converged}}'
            )::boolean IS DISTINCT FROM true THEN
            RAISE EXCEPTION 'native lifecycle is not converged: %', status;
        END IF;

        SELECT count(*)
        INTO sidecar_count
        FROM pg_catalog.pg_class AS relation
        JOIN pg_catalog.pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'medium'
          AND relation.relkind IN ('r', 'p')
          AND relation.relname <> 'docs';
        IF sidecar_count <> 0 THEN
            RAISE EXCEPTION 'native lifecycle created sidecar tables: %',
                sidecar_count;
        END IF;
    END;
    $$;
    '''


def scalar_json(output: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def index_status(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    return scalar_json(psql(
        psql_bin,
        socket_dir,
        port,
        "SELECT evoke_index_status('medium.docs_body_idx'::regclass);",
    ))


def page_native_converged(status: dict[str, Any]) -> bool:
    generation = status.get('generation') or {}
    delta = generation.get('delta') or {}
    active = delta.get('active') or {}
    pending = delta.get('pending') or {}
    semantic = delta.get('semantic_completion') or {}
    return (
        status.get('query_ready') is True
        and status.get('blocker') == 'none'
        and generation.get('valid') is True
        and generation.get('atomic') is True
        and int(delta.get('records') or 0) == 0
        and int(active.get('records') or 0) == 0
        and int(pending.get('records') or 0) == 0
        and int(semantic.get('pending') or 0) == 0
        and semantic.get('converged') is True
    )


def maintain_until_converged(
    psql_bin: Path,
    socket_dir: Path,
    port: int,
) -> dict[str, Any]:
    actions: list[str] = []
    started = time.monotonic()
    deadline = started + MAINTENANCE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = index_status(psql_bin, socket_dir, port)
        if page_native_converged(status):
            return {
                'actions': actions,
                'elapsed_ms': round(
                    (time.monotonic() - started) * 1000,
                    3,
                ),
                'rounds': len(actions),
                'status': status,
            }
        action = psql(
            psql_bin,
            socket_dir,
            port,
            "SET evoke.test_convergent_l0_rotation_records = '1';\n"
            "SELECT evoke_index_maintain("
            "'medium.docs_body_idx'::regclass);\n"
            'RESET evoke.test_convergent_l0_rotation_records;',
        )
        actions.append(action)
        if (
            'reason=lock_busy' in action
            or 'reason=xid_horizon' in action
            or 'reason=accelerator_build_busy' in action
            or 'reason=accelerator_root_checkpoint' in action
        ):
            time.sleep(0.05)
        else:
            time.sleep(0.01)

    status = index_status(psql_bin, socket_dir, port)
    raise AssertionError(
        'bounded page-native maintenance did not converge within '
        f'{MAINTENANCE_TIMEOUT_SECONDS:.0f} seconds: '
        f'actions={json.dumps(actions)}, '
        f'status={json.dumps(status, sort_keys=True)}'
    )


def require_query_rows(
    label: str,
    query_result: dict[str, Any],
    diagnostics: dict[str, Any],
    expected: int = 20,
) -> None:
    if query_result.get('count') == expected:
        return
    raise AssertionError(
        f'{label} returned {query_result.get("count")} rows; '
        f'expected {expected}; diagnostics={json.dumps(diagnostics, sort_keys=True)}'
    )


def main() -> None:
    args = parse_args()
    if args.docs < 1_000:
        raise ValueError('--docs should be at least 1000 for medium smoke')
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.resolve()
    if args.extension_control_dir is not None:
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )

    pg_bin = Path(args.pg_bin)
    initdb = pg_bin / 'initdb'
    pg_ctl = pg_bin / 'pg_ctl'
    psql_bin = pg_bin / 'psql'
    metrics: dict[str, Any] = {'docs': args.docs}

    with tempfile.TemporaryDirectory(prefix='evoke_native_medium_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()
        model_path = args.model_path.resolve()
        manifest_path = model_path / 'manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f'model manifest was not found: {model_path}'
            )
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if (
            manifest.get('schema_version') != 1
            or manifest.get('api_version') != 'evoke_model_v1'
            or manifest.get('runtime_abi')
            != 'evoke_p2_unified_text_atoms_v2'
        ):
            raise ValueError(
                '--model-path must use the current Evoke model contract'
            )
        metrics['model_source'] = 'production_checkout'
        metrics['model_path'] = str(model_path)

        run([str(initdb), '-D', str(data_dir), '-A', 'trust'])
        with (data_dir / 'postgresql.conf').open('a', encoding='utf-8') as f:
            f.write("\nshared_preload_libraries = 'evoke'\n")
            if args.extension_libdir is not None:
                libdir = str(args.extension_libdir).replace("'", "''")
                f.write(
                    "dynamic_library_path = '"
                    f'{libdir}:$libdir'
                    "'\n"
                )
            if args.extension_control_dir is not None:
                control_dir = str(
                    args.extension_control_dir
                ).replace("'", "''")
                f.write(
                    "extension_control_path = '"
                    f'{control_dir}:$system'
                    "'\n"
                )
            f.write("evoke.shared_runtime_size = '256MB'\n")
            f.write("listen_addresses = ''\n")
            f.write('max_worker_processes = 16\n')
            f.write("evoke.maintenance_timer_interval_ms = '60000ms'\n")

        started = False
        try:
            run([
                str(pg_ctl),
                '-D',
                str(data_dir),
                '-l',
                str(log_path),
                '-o',
                f'-k {socket_dir} -p {args.port}',
                'start',
                '-w',
            ])
            started = True

            build_timer = (
                timed_with_backend_rss
                if args.observe_build_rss
                else timed
            )
            build_timer(
                metrics,
                'setup_and_native_index_build',
                psql_bin,
                socket_dir,
                args.port,
                setup_sql(model_path, args.docs),
            )
            metrics['index_bytes'] = int(psql(
                psql_bin,
                socket_dir,
                args.port,
                "SELECT pg_relation_size("
                "'medium.docs_body_idx'::regclass);",
            ))
            metrics['cold_restart_ms'] = restart_postgres(
                pg_ctl,
                data_dir,
                log_path,
                socket_dir,
                args.port,
            )
            first_query = timed(
                metrics,
                'query_cold_after_restart',
                psql_bin,
                socket_dir,
                args.port,
                query_sql(),
            )
            second_query = timed(
                metrics,
                'query_warm_repeat',
                psql_bin,
                socket_dir,
                args.port,
                query_sql(),
            )
            metrics['prewarm_restart_ms'] = restart_postgres(
                pg_ctl,
                data_dir,
                log_path,
                socket_dir,
                args.port,
            )
            timed(
                metrics,
                'model_warmup_before_prewarm',
                psql_bin,
                socket_dir,
                args.port,
                "SELECT evoke_encode_text_internal("
                "'medium.docs_body_idx'::regclass, "
                "'cuda graph neural network optimization');",
            )
            metrics['prewarm_result'] = timed(
                metrics,
                'index_prewarm',
                psql_bin,
                socket_dir,
                args.port,
                "SELECT evoke_index_preload("
                "'medium.docs_body_idx'::regclass);",
            )
            prewarmed_query = timed(
                metrics,
                'query_after_prewarm',
                psql_bin,
                socket_dir,
                args.port,
                query_sql(),
            )
            encoded_query = scalar_json(timed(
                metrics,
                'query_encoding',
                psql_bin,
                socket_dir,
                args.port,
                encode_query_sql(),
            ))
            timed_statement_timeout_response(
                metrics,
                'native_timeout_response',
                psql_bin,
                socket_dir,
                args.port,
                native_query_sql(encoded_query),
                timeout_ms=10,
                response_limit_ms=1000,
                require_timeout=args.require_native_timeout,
            )
            initial_diagnostics = scalar_json(timed(
                metrics,
                'initial_query_diagnostics',
                psql_bin,
                socket_dir,
                args.port,
                query_diagnostics_sql(),
            ))
            first_query_result = scalar_json(first_query)
            second_query_result = scalar_json(second_query)
            prewarmed_query_result = scalar_json(prewarmed_query)
            require_query_rows(
                'first query',
                first_query_result,
                initial_diagnostics,
            )
            require_query_rows(
                'second query',
                second_query_result,
                initial_diagnostics,
            )
            require_query_rows(
                'prewarmed query',
                prewarmed_query_result,
                initial_diagnostics,
            )
            timed(
                metrics,
                'batch_crud',
                psql_bin,
                socket_dir,
                args.port,
                mutation_sql(args.docs),
            )
            maintain_started = time.perf_counter()
            maintain_result = maintain_until_converged(
                psql_bin,
                socket_dir,
                args.port,
            )
            metrics['unified_maintain_ms'] = round(
                (time.perf_counter() - maintain_started) * 1000,
                3,
            )
            post_mutation_query = timed(
                metrics,
                'post_mutation_query',
                psql_bin,
                socket_dir,
                args.port,
                query_sql(),
            )
            post_mutation_diagnostics = scalar_json(timed(
                metrics,
                'post_mutation_query_diagnostics',
                psql_bin,
                socket_dir,
                args.port,
                query_diagnostics_sql(),
            ))
            post_mutation_query_result = scalar_json(post_mutation_query)
            require_query_rows(
                'post-mutation query',
                post_mutation_query_result,
                post_mutation_diagnostics,
            )
            timed(
                metrics,
                'health_after_maintain',
                psql_bin,
                socket_dir,
                args.port,
                assert_health_sql(args.docs),
            )
            build_timer(
                metrics,
                'reindex',
                psql_bin,
                socket_dir,
                args.port,
                'REINDEX INDEX medium.docs_body_idx;',
            )
            timed(
                metrics,
                'health_after_reindex',
                psql_bin,
                socket_dir,
                args.port,
                assert_health_sql(args.docs),
            )
            timed(
                metrics,
                'drop_index',
                psql_bin,
                socket_dir,
                args.port,
                'DROP INDEX medium.docs_body_idx;',
            )
            if psql(
                psql_bin,
                socket_dir,
                args.port,
                "SELECT to_regclass('medium.docs_body_idx') IS NULL;",
            ) != 't':
                raise AssertionError('DROP INDEX left the index relation')

            metrics['first_query'] = first_query_result
            metrics['second_query'] = second_query_result
            metrics['prewarmed_query'] = prewarmed_query_result
            metrics['initial_query_diagnostics'] = initial_diagnostics
            metrics['post_mutation_query'] = post_mutation_query_result
            metrics['post_mutation_query_diagnostics'] = (
                post_mutation_diagnostics
            )
            metrics['maintain_result'] = maintain_result
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

    payload = json.dumps(metrics, indent=2, sort_keys=True) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding='utf-8')
    print(payload, end='')


if __name__ == '__main__':
    main()
