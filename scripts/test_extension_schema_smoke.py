from __future__ import annotations

import argparse
import re
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from evoke_test_support import extension_control_root


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_NAME_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
VERSION_RE = re.compile(r"^default_version = '([^']+)'$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run non-public extension schema wrapper smoke tests.'
    )
    parser.add_argument('--psql', default='psql')
    parser.add_argument('--admin-db', default='postgres')
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--user')
    parser.add_argument(
        '--pg-bin',
        type=Path,
        help=(
            'Start an isolated PostgreSQL cluster with binaries from this '
            'directory instead of using an existing server.'
        ),
    )
    parser.add_argument('--extension-libdir', type=Path)
    parser.add_argument(
        '--extension-control-dir',
        type=Path,
        help=(
            'PostgreSQL share or extension directory containing the '
            'staged evoke.control.'
        ),
    )
    parser.add_argument('--fresh-db', default='evoke_schema_fresh_smoke')
    parser.add_argument(
        '--relocate-db',
        default='evoke_schema_relocate_smoke',
    )
    parser.add_argument(
        '--pinned-version-db',
        default='evoke_schema_pinned_version_smoke',
    )
    return parser.parse_args()


def psql_cmd(args: argparse.Namespace, dbname: str) -> list[str]:
    command = [
        args.psql,
        '-X',
        '-d',
        dbname,
        '-v',
        'ON_ERROR_STOP=1',
    ]
    if args.host is not None:
        command.extend(['-h', args.host])
    if args.port is not None:
        command.extend(['-p', str(args.port)])
    if args.user is not None:
        command.extend(['-U', args.user])
    return command


def run_psql(args: argparse.Namespace, dbname: str, sql: str) -> str:
    result = subprocess.run(
        psql_cmd(args, dbname),
        input=sql,
        text=True,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()
    return result.stdout.strip()


def read_extension_version() -> str:
    for line in (REPO_ROOT / 'evoke.control').read_text().splitlines():
        match = VERSION_RE.match(line)
        if match is not None:
            return match.group(1)
    raise RuntimeError('could not read default_version from evoke.control')


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def run_command(command: list[str | Path]) -> None:
    result = subprocess.run(
        [str(item) for item in command],
        cwd=REPO_ROOT,
        text=True,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        result.check_returncode()


def validate_database_name(dbname: str) -> None:
    if DB_NAME_RE.fullmatch(dbname) is None:
        raise ValueError(f'unsafe database name for smoke test: {dbname}')


def reset_database(args: argparse.Namespace, dbname: str) -> None:
    validate_database_name(dbname)
    run_psql(
        args,
        args.admin_db,
        f'''
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{dbname}'
          AND pid <> pg_backend_pid();
        DROP DATABASE IF EXISTS {dbname};
        CREATE DATABASE {dbname} TEMPLATE template0;
        ''',
    )


def drop_database(args: argparse.Namespace, dbname: str) -> None:
    validate_database_name(dbname)
    run_psql(
        args,
        args.admin_db,
        f'''
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = '{dbname}'
          AND pid <> pg_backend_pid();
        DROP DATABASE IF EXISTS {dbname};
        ''',
    )


def wrapper_smoke_sql(version_sql: str) -> str:
    return f'''
    CREATE SCHEMA ext;
    CREATE EXTENSION evoke WITH SCHEMA ext {version_sql};

    CREATE TABLE public.docs (
        id serial PRIMARY KEY,
        body text,
        summary text,
        title varchar,
        tokens text[]
    );

    INSERT INTO public.docs (body, summary, title, tokens) VALUES
        ('alpha beta gamma', 'alpha summary', 'alpha title',
            ARRAY['alpha', 'beta']),
        ('beta gamma delta', 'beta summary', 'beta title',
            ARRAY['beta', 'gamma']),
        ('theta lambda', 'theta summary', 'theta title', ARRAY['theta']);

    SET search_path = ext, public;
    CREATE INDEX docs_body_idx ON public.docs USING evoke (body);
    CREATE INDEX docs_title_idx ON public.docs USING evoke (title);
    CREATE INDEX docs_tokens_idx ON public.docs USING evoke (tokens);
    CREATE INDEX docs_fields_idx ON public.docs USING evoke (body, summary)
        WITH (field_aware = true);

    DO $$
    DECLARE
        runtime_state jsonb;
    BEGIN
        runtime_state := ext.evoke_index_runtime_state_json(
            'public.docs_body_idx'::regclass
        );
        IF jsonb_typeof(runtime_state) <> 'object'
            OR NOT (runtime_state ?& ARRAY[
                'raw_state',
                'cache_epoch',
                'sae_enabled',
                'generation',
                'shared_preload',
                'maintenance',
                'debt'
            ])
            OR runtime_state->'generation'->>'storage'
                <> 'convergent_segments'
            OR (runtime_state->'generation'->>'docs')::integer <> 3
            OR NOT ((runtime_state->'shared_preload') ? 'registry')
            OR NOT ((runtime_state->'debt') ? 'counter_only')
            OR runtime_state->>'raw_state'
                NOT LIKE 'evoke_index_runtime_state(%'
        THEN
            RAISE EXCEPTION
                'runtime state JSON contract mismatch: %', runtime_state;
        END IF;
    END;
    $$;

    DO $$
    BEGIN
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_opclass AS opclass
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = opclass.opcnamespace
            JOIN pg_catalog.pg_amop AS amop
              ON amop.amopfamily = opclass.opcfamily
            JOIN pg_catalog.pg_operator AS operator
              ON operator.oid = amop.amopopr
            WHERE namespace.nspname = 'ext'
              AND opclass.opcname IN (
                  'evoke_text_ops',
                  'evoke_varchar_ops'
              )
              AND operator.oprname = '@@'
        ) THEN
            RAISE EXCEPTION
                'scalar evoke opclasses must not bind raw @@ operators';
        END IF;
    END;
    $$;

    DO $$
    DECLARE
        exposed_diagnostics text[];
        exposed_result_routes text[];
        function_oid oid;
        function_signature text;
        public_can_execute boolean;
        removed_signature text;
    BEGIN
        FOREACH removed_signature IN ARRAY ARRAY[
            'ext.evoke_index_drop(regclass)',
            'ext.evoke_fast_path_advice(regclass)',
            'ext.evoke_fast_path_plan(regclass,jsonb)',
            'ext.evoke_fast_path_explain(regclass,text)',
            'ext.evoke_ranked_query('
                || 'ext.evoke_result_prepared_query,integer,real[])',
            'ext.evoke_ranked_query('
                || 'regclass,text,integer,real[],boolean,text[],boolean,'
                || 'boolean)',
            'ext.evoke_query_prepared('
                || 'ext.evoke_result_prepared_query,integer,real[])',
            'ext.evoke_filter_query(ext.evoke_result_ranked_query)',
            'ext.evoke_order_tokens(ext.evoke_result_ranked_query)',
            'ext.evoke_fusion(ext.evoke_result_hit[],integer)'
        ]
        LOOP
            IF to_regprocedure(removed_signature) IS NOT NULL THEN
                RAISE EXCEPTION
                    'removed SQL API is installed: %', removed_signature;
            END IF;
        END LOOP;
        IF to_regtype('ext.evoke_result_ranked_query') IS NOT NULL THEN
            RAISE EXCEPTION
                'removed evoke_result_ranked_query type is installed';
        END IF;

        function_oid := to_regprocedure(
            'ext.evoke_query('
            || 'regclass,text,integer,real[],boolean,text[],boolean,boolean'
            || ')'
        );
        IF function_oid IS NULL OR NOT (
            SELECT prosecdef
            FROM pg_proc
            WHERE oid = function_oid
        ) THEN
            RAISE EXCEPTION
                'evoke_query must be the SECURITY DEFINER product boundary';
        END IF;
        SELECT EXISTS (
            SELECT 1
            FROM aclexplode(
                COALESCE(
                    procedure.proacl,
                    acldefault('f', procedure.proowner)
                )
            ) AS privilege
            WHERE privilege.grantee = 0
              AND privilege.privilege_type = 'EXECUTE'
        )
        INTO public_can_execute
        FROM pg_proc AS procedure
        WHERE procedure.oid = function_oid;
        IF NOT public_can_execute THEN
            RAISE EXCEPTION 'PUBLIC cannot execute evoke_query';
        END IF;

        FOREACH function_signature IN ARRAY ARRAY[
            'ext.evoke_encode_text_internal(regclass,text)',
            'ext.evoke_encode_document_batch_internal(regclass,text[])',
            'ext.evoke_index_semantic_query_native_internal('
                || 'regclass,integer[],real[],text[],real[],integer,text,'
                || 'integer[],tid[],jsonb)',
            'ext.evoke_query_semantic_internal('
                || 'regclass,text,integer,text[],real[],integer[],tid[],'
                || 'jsonb)',
            'ext.evoke_query_internal('
                || 'regclass,text,text[],real[],integer,real[],boolean,'
                || 'text[],boolean,boolean,integer[],tid[],jsonb)',
            'ext.evoke_query_bm25_internal('
                || 'regclass,text,integer,real[],boolean,text[],boolean,'
                || 'boolean)'
        ]
        LOOP
            function_oid := to_regprocedure(function_signature);
            IF function_oid IS NULL THEN
                RAISE EXCEPTION
                    'required internal function is missing: %',
                    function_signature;
            END IF;
            SELECT EXISTS (
                SELECT 1
                FROM aclexplode(
                    COALESCE(
                        procedure.proacl,
                        acldefault('f', procedure.proowner)
                    )
                ) AS privilege
                WHERE privilege.grantee = 0
                  AND privilege.privilege_type = 'EXECUTE'
            )
            INTO public_can_execute
            FROM pg_proc AS procedure
            WHERE procedure.oid = function_oid;
            IF public_can_execute THEN
                RAISE EXCEPTION
                    'PUBLIC can execute internal function %',
                    function_signature;
            END IF;
        END LOOP;

        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO exposed_diagnostics
        FROM pg_proc AS procedure
        JOIN pg_namespace AS namespace
          ON namespace.oid = procedure.pronamespace
        WHERE namespace.nspname = 'ext'
          AND procedure.proname = ANY (ARRAY[
              'evoke_query_ids',
              'evoke_query_tokens',
              'evoke_field_aware_query_tokens',
              'evoke_field_aware_query',
              'evoke_prepared_query',
              'evoke_order_tokens',
              'evoke_op_match_prepared_query',
              'evoke_op_match_prepared_query_scalar',
              'evoke_match_prepared_query',
              'evoke_match_query',
              'evoke_score_prepared_query',
              'evoke_score_query'
          ])
          AND EXISTS (
              SELECT 1
              FROM aclexplode(
                  COALESCE(
                      procedure.proacl,
                      acldefault('f', procedure.proowner)
                  )
              ) AS privilege
              WHERE privilege.grantee = 0
                AND privilege.privilege_type = 'EXECUTE'
          );
        IF exposed_diagnostics IS NOT NULL THEN
            RAISE EXCEPTION
                'PUBLIC can execute diagnostic query functions: %',
                exposed_diagnostics;
        END IF;

        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO exposed_diagnostics
        FROM pg_proc AS procedure
        JOIN pg_namespace AS namespace
          ON namespace.oid = procedure.pronamespace
        WHERE namespace.nspname = 'ext'
          AND procedure.proname = ANY (ARRAY[
              'evoke_fusion_weighted_query',
              'evoke_fusion_weighted_queries',
              'evoke_fusion_field_query',
              'evoke_fusion_field_queries',
              'evoke_fusion',
              'evoke_fusion_query',
              'evoke_fusion_query_fields',
              'evoke_fusion_query_weighted',
              'evoke_hybrid_candidate',
              'evoke_hybrid_bm25_candidate',
              'evoke_hybrid_vector_candidate',
              'evoke_hybrid_bm25_candidates',
              'evoke_hybrid_fuse_candidates'
          ])
          AND NOT EXISTS (
              SELECT 1
              FROM aclexplode(
                  COALESCE(
                      procedure.proacl,
                      acldefault('f', procedure.proowner)
                  )
              ) AS privilege
              WHERE privilege.grantee = 0
                AND privilege.privilege_type = 'EXECUTE'
          );
        IF exposed_diagnostics IS NOT NULL THEN
            RAISE EXCEPTION
                'PUBLIC cannot execute product composition functions: %',
                exposed_diagnostics;
        END IF;

        SELECT array_agg(
            procedure.oid::regprocedure::text
            ORDER BY procedure.oid::regprocedure::text
        )
        INTO exposed_result_routes
        FROM pg_proc AS procedure
        JOIN pg_namespace AS namespace
          ON namespace.oid = procedure.pronamespace
        JOIN pg_type AS result_type
          ON result_type.oid = procedure.prorettype
        WHERE namespace.nspname = 'ext'
          AND procedure.proretset
          AND result_type.typnamespace = namespace.oid
          AND result_type.typname = ANY (ARRAY[
              'evoke_result_hit',
              'evoke_result_hybrid_candidate',
              'evoke_result_hybrid_hit'
          ])
          AND procedure.proname <> ALL (ARRAY[
              'evoke_query',
              'evoke_fusion',
              'evoke_fusion_query',
              'evoke_fusion_query_fields',
              'evoke_fusion_query_weighted',
              'evoke_hybrid_bm25_candidates',
              'evoke_hybrid_fuse_candidates'
          ])
          AND EXISTS (
              SELECT 1
              FROM aclexplode(
                  COALESCE(
                      procedure.proacl,
                      acldefault('f', procedure.proowner)
                  )
              ) AS privilege
              WHERE privilege.grantee = 0
                AND privilege.privilege_type = 'EXECUTE'
          );
        IF exposed_result_routes IS NOT NULL THEN
            RAISE EXCEPTION
                'PUBLIC can execute alternate top-k result routes: %',
                exposed_result_routes;
        END IF;
    END;
    $$;

    CREATE ROLE evoke_schema_app;
    GRANT USAGE ON SCHEMA ext TO evoke_schema_app;
    GRANT SELECT ON public.docs TO evoke_schema_app;
    SET ROLE evoke_schema_app;
    SELECT count(*)
    FROM ext.evoke_query(
        'public.docs_body_idx'::regclass,
        'alpha',
        2
    );
    SELECT count(*)
    FROM ext.evoke_query(
        'public.docs_fields_idx'::regclass,
        'alpha',
        ARRAY['body', 'summary']::text[],
        ARRAY[2.0, 0.5]::real[],
        2
    );
    SELECT count(*)
    FROM ext.evoke_fusion_query(
        ARRAY[
            'public.docs_body_idx'::regclass,
            'public.docs_title_idx'::regclass
        ],
        'alpha',
        ARRAY[1.0, 0.5]::real[],
        2
    );
    WITH candidates AS (
        SELECT array_agg(candidate) AS items
        FROM ext.evoke_hybrid_bm25_candidates(
            'body',
            'public.docs_body_idx'::regclass,
            'alpha',
            1.0,
            2
        ) AS candidate
    )
    SELECT count(*)
    FROM candidates
    CROSS JOIN LATERAL ext.evoke_hybrid_fuse_candidates(
        candidates.items,
        2
    );
    SELECT count(*)
    FROM public.docs
    WHERE tokens OPERATOR(ext.@@) 'alpha';
    RESET ROLE;
    DROP OWNED BY evoke_schema_app;
    DROP ROLE evoke_schema_app;

    SET search_path = pg_catalog;

    SELECT count(*)
    FROM ext.evoke_fusion_query_weighted(
        ARRAY[
            ext.evoke_fusion_weighted_query(
                'public.docs_body_idx'::regclass,
                'alpha',
                2.0
            ),
            ext.evoke_fusion_weighted_query(
                'public.docs_title_idx'::regclass,
                'alpha',
                1.0
            )
        ]::ext.evoke_result_fusion_weighted_query[],
        2,
        10,
        NULL
    ) r;

    SELECT count(*)
    FROM ext.evoke_query(
        'public.docs_body_idx'::regclass,
        'alpha',
        2
    ) r;

    SELECT count(*)
    FROM ext.evoke_hybrid_fuse_candidates(
        ARRAY[
            ext.evoke_hybrid_bm25_candidate(
                'body',
                '(0,1)'::tid,
                1.0,
                1,
                2.0
            ),
            ext.evoke_hybrid_vector_candidate(
                'embedding',
                '(0,1)'::tid,
                0.2,
                1,
                1.0,
                'minmax'
            ),
            ext.evoke_hybrid_vector_candidate(
                'embedding',
                '(0,2)'::tid,
                0.4,
                2,
                1.0,
                'minmax'
            )
        ]::ext.evoke_result_hybrid_candidate[],
        2,
        'score'
    ) h;

    SELECT count(*)
    FROM public.docs
    WHERE tokens OPERATOR(ext.@@) 'alpha';

    DROP INDEX public.docs_tokens_idx;

    DO $$
    BEGIN
        IF to_regclass('public.docs_tokens_idx') IS NOT NULL THEN
            RAISE EXCEPTION 'DROP INDEX left the index relation';
        END IF;
    END;
    $$;
    '''


def run_case(
    args: argparse.Namespace,
    dbname: str,
    version_sql: str,
) -> None:
    reset_database(args, dbname)
    try:
        run_psql(args, dbname, wrapper_smoke_sql(version_sql))
    finally:
        drop_database(args, dbname)


def run_relocation_guard(args: argparse.Namespace) -> None:
    reset_database(args, args.relocate_db)
    try:
        run_psql(
            args,
            args.relocate_db,
            '''
            CREATE SCHEMA ext;
            CREATE SCHEMA ext2;
            CREATE EXTENSION evoke WITH SCHEMA ext;
            ''',
        )
        result = subprocess.run(
            psql_cmd(args, args.relocate_db),
            input='ALTER EXTENSION evoke SET SCHEMA ext2;',
            text=True,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            raise AssertionError(
                'evoke unexpectedly allowed ALTER EXTENSION SET SCHEMA'
            )
        expected = 'extension "evoke" does not support SET SCHEMA'
        if expected not in result.stderr:
            raise AssertionError(
                'unexpected relocation error: '
                f'{result.stderr.strip()}'
            )
    finally:
        drop_database(args, args.relocate_db)


def run_all_cases(args: argparse.Namespace) -> None:
    current_version = read_extension_version()
    run_case(args, args.fresh_db, '')
    run_case(
        args,
        args.pinned_version_db,
        f"VERSION '{current_version}'",
    )
    run_relocation_guard(args)


def run_isolated(args: argparse.Namespace) -> None:
    if args.pg_bin is None:
        raise AssertionError('isolated schema smoke requires --pg-bin')

    initdb = args.pg_bin / 'initdb'
    pg_ctl = args.pg_bin / 'pg_ctl'
    psql = args.pg_bin / 'psql'
    for binary in (initdb, pg_ctl, psql):
        if not binary.is_file():
            raise FileNotFoundError(f'PostgreSQL binary is missing: {binary}')

    with tempfile.TemporaryDirectory(prefix='evoke_schema_') as tmp:
        root = Path(tmp)
        data_dir = root / 'data'
        socket_dir = root / 'socket'
        log_path = root / 'postgres.log'
        socket_dir.mkdir()
        port = free_port()
        started = False
        try:
            run_command(
                [
                    initdb,
                    '-D',
                    data_dir,
                    '-A',
                    'trust',
                    '-U',
                    'postgres',
                ]
            )
            with (data_dir / 'postgresql.conf').open(
                'a',
                encoding='utf-8',
            ) as handle:
                handle.write("\nlisten_addresses = ''\n")
                handle.write(
                    f"unix_socket_directories = '{socket_dir}'\n"
                )
                handle.write(f'port = {port}\n')
                if args.extension_libdir is not None:
                    libdir = str(args.extension_libdir).replace("'", "''")
                    handle.write(
                        "dynamic_library_path = '"
                        f"{libdir}:$libdir'\n"
                    )
                if args.extension_control_dir is not None:
                    control_dir = str(
                        args.extension_control_dir
                    ).replace("'", "''")
                    handle.write(
                        "extension_control_path = '"
                        f"{control_dir}:$system'\n"
                    )
            run_command(
                [pg_ctl, '-D', data_dir, '-l', log_path, 'start', '-w']
            )
            started = True
            args.psql = str(psql)
            args.host = str(socket_dir)
            args.port = port
            args.user = 'postgres'
            run_all_cases(args)
        except Exception:
            if log_path.exists():
                print(log_path.read_text(encoding='utf-8'), file=sys.stderr)
            raise
        finally:
            if started:
                run_command([pg_ctl, '-D', data_dir, 'stop', '-m', 'fast'])


def main() -> None:
    args = parse_args()
    if (args.extension_libdir is None) != (
        args.extension_control_dir is None
    ):
        raise ValueError(
            '--extension-libdir and --extension-control-dir must be '
            'provided together'
        )
    if args.extension_libdir is not None:
        args.extension_libdir = args.extension_libdir.expanduser().resolve()
        if not any(
            (args.extension_libdir / name).is_file()
            for name in ('evoke.so', 'evoke.dylib')
        ):
            raise FileNotFoundError(
                'evoke extension library is missing from '
                f'{args.extension_libdir}'
            )
        args.extension_control_dir = extension_control_root(
            args.extension_control_dir
        )
    if args.pg_bin is None:
        if args.extension_libdir is not None:
            raise ValueError(
                'staged extension paths require isolated --pg-bin mode'
            )
        run_all_cases(args)
    else:
        run_isolated(args)
    print('extension schema smoke passed')


if __name__ == '__main__':
    main()
