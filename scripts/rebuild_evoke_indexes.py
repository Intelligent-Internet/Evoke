#!/usr/bin/env python3
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


PSQL = (
    os.environ.get('PSQL')
    or shutil.which('psql')
    or '/opt/homebrew/opt/postgresql@18/bin/psql'
)
REPO_ROOT = Path(__file__).resolve().parents[1]
ONNXRUNTIME_VERSION_PATH = REPO_ROOT / 'packaging/onnxruntime.version'
CATALOG_CONTRACT = 'evoke_catalog_v1'
REBUILD_PLAN_CONTRACT = 1
SIZE_RE = re.compile(r'^(\d+(?:\.\d+)?)([kmgt]i?b?|b)?$', re.IGNORECASE)
TEXT_LIKE_TYPES = {
    'text',
    'character varying',
    'text[]',
    'character varying[]',
}
INT_ARRAY_TYPES = {'integer[]'}
SEMANTIC_ACTIONS = {
    'migrate_psql_bm25s_text_like',
    'migrate_pg_search_text_columns',
    'migrate_legacy_beta_sae',
    'convert_current_evoke_to_sae',
    'reindex_current_sae',
}
LEXICAL_ONLY_ACTIONS = {
    'migrate_psql_bm25s_int4_lexical_only',
    'migrate_legacy_beta_int4_lexical_only',
    'migrate_legacy_beta_text_lexical_only',
    'rebuild_current_evoke_lexical_only',
    'rebuild_current_evoke_expression_lexical_only',
    'rebuild_current_evoke_text_lexical_only',
}
RESET_LEXICAL_ONLY_KEYS = {
    'auto_maintain',
}
LEXICAL_ONLY_KEEP_KEYS = {
    'method',
    'idf_method',
    'k1',
    'b',
    'delta',
    'create_empty_token',
    'consistency',
    'auto_preload',
    'text_lowercase',
    'text_stopwords',
    'text_stem_english',
    'text_fold_diacritics',
    'field_aware',
}
SEMANTIC_KEEP_KEYS = {
    'auto_preload',
    'method',
    'idf_method',
    'k1',
    'b',
    'delta',
    'create_empty_token',
    'text_lowercase',
    'text_stopwords',
    'text_stem_english',
    'text_fold_diacritics',
    'field_aware',
    'semantic_impact_precision',
    'semantic_alpha_mass',
    'model',
    'model_path',
    'atom_space',
    'scoring_profile',
}
SEMANTIC_FORCED_KEYS = {
    'sae',
    'consistency',
    'runtime_precision',
}
REQUIRED_EXTENSION_FUNCTIONS = (
    ('evoke_catalog_contract_internal', ''),
    ('evoke_index_generation_status_internal', 'regclass'),
    ('evoke_query_trace_internal', ''),
    ('evoke_runtime_service_query_atoms', 'text,text,text'),
    ('evoke_runtime_service_query_atoms_batch', 'text,text,text[]'),
    ('evoke_runtime_service_document_atoms_batch', 'text,text,text[]'),
    ('evoke_runtime_service_atoms_batch_internal', 'text,text,text[]'),
    (
        'evoke_runtime_service_atoms_batch_internal',
        'text,text,text,text[]',
    ),
)


log_lock = threading.Lock()


def expected_onnxruntime_version():
    override = os.environ.get('EVOKE_EXPECTED_ONNXRUNTIME_VERSION')
    if override is not None:
        version = override.strip()
    else:
        try:
            version = ONNXRUNTIME_VERSION_PATH.read_text(
                encoding='utf-8',
            ).strip()
        except OSError as exc:
            raise RuntimeError(
                'semantic rebuild requires the pinned ONNX Runtime version; '
                f'could not read {ONNXRUNTIME_VERSION_PATH}: {exc}'
            ) from exc
    if not version:
        raise RuntimeError(
            'semantic rebuild requires a non-empty pinned ONNX Runtime '
            f'version in {ONNXRUNTIME_VERSION_PATH}'
        )
    return version


def expected_onnxruntime_api(version):
    parts = version.split('.')
    if len(parts) < 2 or parts[0] != '1' or not parts[1].isdigit():
        raise RuntimeError(
            f'unsupported pinned ONNX Runtime version: {version!r}'
        )
    return int(parts[1])


def quote_ident(value):
    return '"' + value.replace('"', '""') + '"'


def qname(schema, name):
    return f'{quote_ident(schema)}.{quote_ident(name)}'


def parse_size(value):
    match = SIZE_RE.match(str(value).strip())
    if not match:
        raise argparse.ArgumentTypeError(f'invalid byte size: {value}')

    number = float(match.group(1))
    unit = (match.group(2) or 'b').lower()
    multipliers = {
        'b': 1,
        'k': 1024,
        'kb': 1024,
        'kib': 1024,
        'm': 1024 ** 2,
        'mb': 1024 ** 2,
        'mib': 1024 ** 2,
        'g': 1024 ** 3,
        'gb': 1024 ** 3,
        'gib': 1024 ** 3,
        't': 1024 ** 4,
        'tb': 1024 ** 4,
        'tib': 1024 ** 4,
    }
    return int(number * multipliers[unit])


def compile_patterns(patterns):
    return [re.compile(pattern) for pattern in patterns or []]


def any_pattern(patterns, value):
    return any(pattern.search(value or '') for pattern in patterns)


def psql(db, sql, timeout=None):
    env = os.environ.copy()
    env.setdefault('PGAPPNAME', 'evoke-rebuild-runner')
    start = time.monotonic()
    proc = subprocess.run(
        [
            PSQL,
            '-X',
            '-v',
            'ON_ERROR_STOP=1',
            '-Atq',
            '-d',
            db,
            '-c',
            sql,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    elapsed = time.monotonic() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f'psql failed in {db}: {proc.stderr.strip() or proc.stdout.strip()}'
        )
    return proc.stdout.strip(), elapsed


def inventory_database(db):
    sql = r"""
WITH index_catalog AS (
    SELECT
        current_database() AS database,
        index_namespace.nspname AS schema_name,
        index_relation.relname AS index_name,
        table_namespace.nspname AS table_schema_name,
        table_relation.relname AS table_name,
        index_relation.oid AS index_oid,
        table_relation.oid AS table_oid,
        index_state.indkey,
        index_state.indnkeyatts,
        index_state.indnatts,
        index_state.indisvalid,
        index_state.indisready,
        index_state.indislive,
        access_method.amname AS access_method,
        pg_catalog.pg_get_indexdef(index_relation.oid) AS indexdef,
        COALESCE(
            pg_catalog.pg_get_expr(
                index_state.indpred,
                index_state.indrelid,
                true
            ),
            ''
        ) AS predicate,
        pg_catalog.pg_relation_size(index_relation.oid) AS index_bytes,
        index_relation.reltuples::bigint AS index_tuples_est,
        pg_catalog.pg_relation_size(table_relation.oid) AS table_bytes,
        pg_catalog.pg_total_relation_size(table_relation.oid)
            AS table_total_bytes,
        table_relation.reltuples::bigint AS table_tuples_est
    FROM pg_catalog.pg_index AS index_state
    JOIN pg_catalog.pg_class AS index_relation
      ON index_relation.oid = index_state.indexrelid
    JOIN pg_catalog.pg_namespace AS index_namespace
      ON index_namespace.oid = index_relation.relnamespace
    JOIN pg_catalog.pg_class AS table_relation
      ON table_relation.oid = index_state.indrelid
    JOIN pg_catalog.pg_namespace AS table_namespace
      ON table_namespace.oid = table_relation.relnamespace
    JOIN pg_catalog.pg_am AS access_method
      ON access_method.oid = index_relation.relam
    WHERE access_method.amname IN ('evoke', 'psql_bm25s', 'ii42')
), index_inventory AS (
    SELECT
        index_catalog.*,
        COALESCE(
            (
                SELECT pg_catalog.json_agg(
                    pg_catalog.json_build_object(
                        'position', position,
                        'attnum', index_catalog.indkey[position - 1],
                        'attname', source_attribute.attname,
                        'keydef', pg_catalog.pg_get_indexdef(
                            index_catalog.index_oid,
                            position,
                            true
                        ),
                        'type', pg_catalog.format_type(
                            index_attribute.atttypid,
                            index_attribute.atttypmod
                        )
                    )
                    ORDER BY position
                )
                FROM pg_catalog.generate_series(
                    1,
                    index_catalog.indnkeyatts
                ) AS position
                JOIN pg_catalog.pg_attribute AS index_attribute
                  ON index_attribute.attrelid = index_catalog.index_oid
                 AND index_attribute.attnum = position
                LEFT JOIN pg_catalog.pg_attribute AS source_attribute
                  ON source_attribute.attrelid = index_catalog.table_oid
                 AND source_attribute.attnum =
                     index_catalog.indkey[position - 1]
            ),
            '[]'::json
        ) AS columns,
        COALESCE(
            (
                SELECT pg_catalog.json_agg(
                    pg_catalog.pg_get_indexdef(
                        index_catalog.index_oid,
                        position,
                        true
                    )
                    ORDER BY position
                )
                FROM pg_catalog.generate_series(
                    index_catalog.indnkeyatts + 1,
                    index_catalog.indnatts
                ) AS position
            ),
            '[]'::json
        ) AS include_defs
    FROM index_catalog
)
SELECT pg_catalog.json_build_object(
    'database', current_database(),
    'server_version_num',
        current_setting('server_version_num')::integer,
    'indexes', COALESCE(
        pg_catalog.json_agg(
            pg_catalog.json_build_object(
                'database', index_inventory.database,
                'schema_name', index_inventory.schema_name,
                'index_name', index_inventory.index_name,
                'index_regclass', pg_catalog.format(
                    '%I.%I',
                    index_inventory.schema_name,
                    index_inventory.index_name
                ),
                'table_schema_name', index_inventory.table_schema_name,
                'table_name', index_inventory.table_name,
                'table_regclass', pg_catalog.format(
                    '%I.%I',
                    index_inventory.table_schema_name,
                    index_inventory.table_name
                ),
                'access_method', index_inventory.access_method,
                'indexdef', index_inventory.indexdef,
                'predicate', index_inventory.predicate,
                'columns', index_inventory.columns,
                'include_defs', index_inventory.include_defs,
                'index_bytes', index_inventory.index_bytes,
                'index_tuples_est', index_inventory.index_tuples_est,
                'table_bytes', index_inventory.table_bytes,
                'table_total_bytes', index_inventory.table_total_bytes,
                'table_tuples_est', index_inventory.table_tuples_est,
                'indisvalid', index_inventory.indisvalid,
                'indisready', index_inventory.indisready,
                'indislive', index_inventory.indislive
            )
            ORDER BY index_inventory.schema_name, index_inventory.index_name
        ),
        '[]'::json
    )
)::text
FROM index_inventory;
"""
    value, _ = psql(db, sql)
    if not value:
        raise RuntimeError(f'could not inventory indexes in {db}')
    inventory = json.loads(value)
    for idx in inventory['indexes']:
        idx['rebuild_plan'] = inventory_rebuild_plan(idx)
    return inventory


def inventory_rebuild_plan(idx):
    columns = idx.get('columns') or []
    types = {col.get('type') for col in columns}
    keydefs = [col['keydef'] for col in columns]
    include_defs = list(idx.get('include_defs') or [])
    access_method = idx.get('access_method')
    options = index_options(idx.get('indexdef') or '')
    sae_enabled = options.get('sae', '').strip(" '\"").lower() in {
        'true',
        'on',
        'yes',
        '1',
    }

    if access_method == 'evoke' and sae_enabled:
        action = 'reindex_current_sae'
    elif access_method == 'evoke' and any(
        col.get('attnum') == 0 for col in columns
    ):
        action = 'rebuild_current_evoke_expression_lexical_only'
    elif access_method == 'evoke' and types and types <= INT_ARRAY_TYPES:
        action = 'rebuild_current_evoke_lexical_only'
    elif access_method == 'evoke' and types and types <= TEXT_LIKE_TYPES:
        action = 'rebuild_current_evoke_text_lexical_only'
    elif access_method == 'ii42' and sae_enabled:
        action = 'migrate_legacy_beta_sae'
    elif access_method == 'ii42' and types and types <= INT_ARRAY_TYPES:
        action = 'migrate_legacy_beta_int4_lexical_only'
    elif access_method == 'ii42' and types and types <= TEXT_LIKE_TYPES:
        action = 'migrate_legacy_beta_text_lexical_only'
    elif access_method == 'psql_bm25s' and types and types <= INT_ARRAY_TYPES:
        action = 'migrate_psql_bm25s_int4_lexical_only'
    elif access_method == 'psql_bm25s' and types and types <= TEXT_LIKE_TYPES:
        action = 'migrate_psql_bm25s_text_like'
    else:
        action = 'skip'

    return {
        'action': action,
        'key_count': len(keydefs),
        'keydefs': keydefs,
        'include_defs': include_defs,
        'reason': '' if action != 'skip' else 'unsupported index topology',
    }


def generate_inventory_plan(databases):
    inventories = [inventory_database(db) for db in databases]
    return {
        'contract': REBUILD_PLAN_CONTRACT,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'databases': [
            {
                'database': item['database'],
                'server_version_num': item['server_version_num'],
                'index_count': len(item['indexes']),
            }
            for item in inventories
        ],
        'indexes': [
            idx
            for item in inventories
            for idx in item['indexes']
        ],
    }


def log_event(log_path, event):
    event = dict(event)
    event.setdefault('at', time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    with log_lock:
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, sort_keys=True) + '\n')
        print(json.dumps(event, sort_keys=True), flush=True)


def option_span(indexdef):
    upper = indexdef.upper()
    i = 0
    in_quote = False
    parens = 0
    while i < len(indexdef):
        ch = indexdef[i]
        if ch == "'":
            if in_quote and i + 1 < len(indexdef) and indexdef[i + 1] == "'":
                i += 2
                continue
            in_quote = not in_quote
        elif not in_quote:
            if ch == '(':
                parens += 1
            elif ch == ')':
                parens -= 1
            elif (
                parens == 0
                and upper.startswith('WITH', i)
                and (i == 0 or not upper[i - 1].isalnum())
            ):
                j = i + len('WITH')
                while j < len(indexdef) and indexdef[j].isspace():
                    j += 1
                if j >= len(indexdef) or indexdef[j] != '(':
                    i += 1
                    continue
                start = j + 1
                depth = 1
                j += 1
                opt_quote = False
                while j < len(indexdef):
                    c = indexdef[j]
                    if c == "'":
                        if (
                            opt_quote
                            and j + 1 < len(indexdef)
                            and indexdef[j + 1] == "'"
                        ):
                            j += 2
                            continue
                        opt_quote = not opt_quote
                    elif not opt_quote:
                        if c == '(':
                            depth += 1
                        elif c == ')':
                            depth -= 1
                            if depth == 0:
                                return start, j
                    j += 1
        i += 1
    return None


def split_options(raw):
    if not raw:
        return []
    parts = []
    start = 0
    in_quote = False
    depth = 0
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch == "'":
            if in_quote and i + 1 < len(raw) and raw[i + 1] == "'":
                i += 2
                continue
            in_quote = not in_quote
        elif not in_quote:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(raw[start:i].strip())
                start = i + 1
        i += 1
    tail = raw[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def index_options(indexdef):
    span = option_span(indexdef)
    if span is None:
        return {}
    raw = indexdef[span[0]:span[1]]
    options = {}
    for part in split_options(raw):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        options[key.strip().strip('"').lower()] = value.strip()
    return options


def reloptions_sql(
    idx,
    semantic,
    runtime_precision,
    semantic_impact_precision=None,
    semantic_alpha_mass=None,
):
    original = index_options(idx.get('indexdef') or '')
    if semantic:
        options = []
        for key, value in original.items():
            if key in RESET_LEXICAL_ONLY_KEYS or key in SEMANTIC_FORCED_KEYS:
                continue
            if semantic_impact_precision is not None and (
                key == 'semantic_impact_precision'
            ):
                continue
            if semantic_alpha_mass is not None and (
                key == 'semantic_alpha_mass'
            ):
                continue
            if key in SEMANTIC_KEEP_KEYS:
                options.append(f'{key} = {value}')
        options.extend(
            [
                'sae = true',
                "consistency = 'eventual'",
                f"runtime_precision = '{runtime_precision}'",
            ]
        )
        if semantic_impact_precision is not None:
            options.append(
                'semantic_impact_precision = '
                f"'{semantic_impact_precision}'"
            )
        if semantic_alpha_mass is not None:
            options.append(
                f'semantic_alpha_mass = {semantic_alpha_mass:.17g}'
            )
        return 'WITH (' + ', '.join(options) + ')'

    kept = []
    seen = set()
    for key, value in original.items():
        if key in RESET_LEXICAL_ONLY_KEYS:
            continue
        if key in LEXICAL_ONLY_KEEP_KEYS:
            kept.append(f'{key} = {value}')
            seen.add(key)
    if 'consistency' not in seen:
        kept.append("consistency = 'eventual'")
    kept.append('sae = false')
    return 'WITH (' + ', '.join(kept) + ')'


def classify(idx):
    plan = idx.get('rebuild_plan') or {}
    action = plan.get('action')
    if action in SEMANTIC_ACTIONS | LEXICAL_ONLY_ACTIONS:
        return action

    cols = idx.get('columns') or []
    types = {col.get('type') for col in cols}
    am = idx.get('access_method')
    options = index_options(idx.get('indexdef') or '')
    sae_enabled = options.get('sae', '').strip(" '\"").lower() in {
        'true',
        'on',
        'yes',
        '1',
    }
    if am == 'evoke' and sae_enabled:
        return 'reindex_current_sae'
    if am == 'ii42' and sae_enabled:
        return 'migrate_legacy_beta_sae'
    if am == 'ii42' and types and types <= INT_ARRAY_TYPES:
        return 'migrate_legacy_beta_int4_lexical_only'
    if am == 'ii42' and types and types <= TEXT_LIKE_TYPES:
        return 'migrate_legacy_beta_text_lexical_only'
    if am == 'psql_bm25s' and types and types <= INT_ARRAY_TYPES:
        return 'migrate_psql_bm25s_int4_lexical_only'
    if am == 'evoke' and types and types <= INT_ARRAY_TYPES:
        return 'rebuild_current_evoke_lexical_only'
    if am == 'evoke' and any((col.get('attnum') == 0 for col in cols)):
        return 'rebuild_current_evoke_expression_lexical_only'
    if am == 'evoke' and types and types <= TEXT_LIKE_TYPES:
        return 'rebuild_current_evoke_text_lexical_only'
    return 'skip'


def action_allowed(action, mode):
    if action == 'skip':
        return False
    if mode == 'semantic':
        return action in SEMANTIC_ACTIONS
    if mode == 'lexical-only':
        return action in LEXICAL_ONLY_ACTIONS
    return True


def index_filter_allowed(db, idx, args):
    identity = f"{db}:{idx['index_regclass']}"
    if any_pattern(args.skip_db_patterns, db):
        return False
    if any_pattern(args.skip_index_patterns, idx['index_regclass']):
        return False
    if any_pattern(args.skip_index_patterns, identity):
        return False
    if args.max_index_bytes is not None:
        index_bytes = int(idx.get('index_bytes') or 0)
        if index_bytes > args.max_index_bytes:
            return False
    if args.max_table_bytes is not None:
        table_bytes = int(idx.get('table_bytes') or 0)
        if table_bytes > args.max_table_bytes:
            return False
    if args.max_table_tuples is not None:
        table_tuples = int(idx.get('table_tuples_est') or 0)
        if table_tuples > args.max_table_tuples:
            return False
    return True


def index_order_key(idx, args):
    if args.order_by == 'table-tuples-asc':
        return (
            int(idx.get('table_tuples_est') or 0),
            int(idx.get('table_bytes') or 0),
            int(idx.get('index_bytes') or 0),
        )
    if args.order_by == 'table-bytes-asc':
        return (
            int(idx.get('table_bytes') or 0),
            int(idx.get('table_tuples_est') or 0),
            int(idx.get('index_bytes') or 0),
        )
    return (int(idx.get('index_bytes') or 0),)


def index_order_reverse(args):
    return args.order_by == 'index-bytes-desc'


def sort_runnable(runnable, args):
    return sorted(
        runnable,
        key=lambda item: index_order_key(item[0], args),
        reverse=index_order_reverse(args),
    )


def keydefs_for(idx, action):
    plan = idx.get('rebuild_plan') or {}
    keydefs = list(plan.get('keydefs') or [])
    if action == 'migrate_pg_search_text_columns':
        keydefs = [
            col['keydef']
            for col in idx.get('columns') or []
            if col.get('type') in TEXT_LIKE_TYPES
        ]
    return keydefs


def create_index_sql(
    idx,
    name,
    action,
    runtime_precision,
    semantic_impact_precision=None,
    semantic_alpha_mass=None,
):
    semantic = action in SEMANTIC_ACTIONS
    keydefs = keydefs_for(idx, action)
    if not keydefs:
        raise RuntimeError('no key definitions for rebuild')
    options = reloptions_sql(
        idx,
        semantic=semantic,
        runtime_precision=runtime_precision,
        semantic_impact_precision=semantic_impact_precision,
        semantic_alpha_mass=semantic_alpha_mass,
    )
    table_name = qname(idx['table_schema_name'], idx['table_name'])
    index_name = quote_ident(name)
    keys = ', '.join(keydefs)
    include_defs = list(
        (idx.get('rebuild_plan') or {}).get('include_defs')
        or idx.get('include_defs')
        or []
    )
    predicate = (idx.get('predicate') or '').strip()
    include_sql = ''
    predicate_sql = ''
    if include_defs:
        include_sql = f"INCLUDE ({', '.join(include_defs)})\n"
    if predicate:
        predicate_sql = f'WHERE {predicate};'
    else:
        predicate_sql = ';'
    return (
        f'CREATE INDEX {index_name}\n'
        f'ON {table_name} USING evoke ({keys})\n'
        f'{include_sql}'
        f'{options}\n'
        f'{predicate_sql}'
    )


def temp_index_name(idx):
    base = idx['index_name']
    digest = hashlib.sha1(
        f"{idx['database']}:{idx['index_regclass']}".encode('utf-8')
    ).hexdigest()[:12]
    prefix = base[: max(1, 50 - len(digest))]
    return f'{prefix}_{digest}'


def extension_schema(db):
    sql = """
SELECT n.nspname
FROM pg_extension e
JOIN pg_namespace n ON n.oid = e.extnamespace
WHERE e.extname = 'evoke';
"""
    out, _ = psql(db, sql)
    return out.splitlines()[0] if out else None


def extension_exists(db, name):
    out, _ = psql(
        db,
        (
            "SELECT EXISTS ("
            "SELECT 1 FROM pg_extension WHERE extname = "
            + sql_literal(name)
            + ");"
        ),
    )
    return out == 't'


def sql_literal(value):
    return "'" + value.replace("'", "''") + "'"


def extension_version_state(db):
    sql = """
SELECT pg_catalog.json_build_object(
    'installed', installed.extversion,
    'default', available.default_version
)::text
FROM pg_catalog.pg_extension AS installed
JOIN pg_catalog.pg_available_extensions AS available
  ON available.name = installed.extname
WHERE installed.extname = 'evoke';
"""
    out, _ = psql(db, sql)
    if not out:
        raise RuntimeError('evoke extension version state is unavailable')
    state = json.loads(out)
    if not state.get('installed') or not state.get('default'):
        raise RuntimeError(f'invalid evoke extension version state: {state}')
    return state


def required_extension_signatures(extension_schema_name):
    return [
        f'{qname(extension_schema_name, name)}({arguments})'
        for name, arguments in REQUIRED_EXTENSION_FUNCTIONS
    ]


def require_extension_catalog(db, extension_schema_name):
    values = ', '.join(
        f'({sql_literal(signature)})'
        for signature in required_extension_signatures(
            extension_schema_name
        )
    )
    sql = f"""
WITH required(signature) AS (
    VALUES {values}
), extension_object AS (
    SELECT oid
    FROM pg_catalog.pg_extension
    WHERE extname = 'evoke'
), resolved AS (
    SELECT
        required.signature,
        pg_catalog.to_regprocedure(required.signature) AS procedure_oid
    FROM required
)
SELECT COALESCE(
    pg_catalog.string_agg(resolved.signature, ', ' ORDER BY resolved.signature)
        FILTER (
            WHERE resolved.procedure_oid IS NULL
               OR NOT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_depend AS dependency
                    CROSS JOIN extension_object
                    WHERE dependency.classid = 'pg_catalog.pg_proc'::regclass
                      AND dependency.objid = resolved.procedure_oid
                      AND dependency.refclassid =
                          'pg_catalog.pg_extension'::regclass
                      AND dependency.refobjid = extension_object.oid
                      AND dependency.deptype = 'e'
               )
        ),
    ''
)
FROM resolved;
"""
    missing, _ = psql(db, sql)
    if missing:
        raise RuntimeError(
            'evoke extension catalog contract is incomplete; missing or '
            f'unowned functions: {missing}. Install one coherent package '
            'and recreate the current extension catalog before rebuilding '
            'indexes.'
        )
    identity_function = qname(
        extension_schema_name,
        'evoke_catalog_contract_internal',
    )
    identity, _ = psql(db, f'SELECT {identity_function}();')
    if identity != CATALOG_CONTRACT:
        raise RuntimeError(
            'evoke extension catalog identity is stale or unknown; '
            f'expected {CATALOG_CONTRACT}, got '
            f'{identity or "<empty>"}. Install one coherent package and '
            'recreate the current extension catalog before rebuilding '
            'indexes.'
        )


def require_extension_module_authority(db):
    sql = """
WITH extension_object AS (
    SELECT oid
    FROM pg_catalog.pg_extension
    WHERE extname = 'evoke'
), c_modules AS (
    SELECT DISTINCT procedure.probin
    FROM extension_object
    JOIN pg_catalog.pg_depend AS dependency
      ON dependency.refclassid = 'pg_catalog.pg_extension'::regclass
     AND dependency.refobjid = extension_object.oid
     AND dependency.classid = 'pg_catalog.pg_proc'::regclass
     AND dependency.deptype = 'e'
    JOIN pg_catalog.pg_proc AS procedure
      ON procedure.oid = dependency.objid
    JOIN pg_catalog.pg_language AS language
      ON language.oid = procedure.prolang
    WHERE language.lanname = 'c'
)
SELECT COALESCE(
    pg_catalog.json_agg(c_modules.probin ORDER BY c_modules.probin)::text,
    '[]'
)
FROM c_modules;
"""
    value, _ = psql(db, sql)
    paths = json.loads(value)
    if paths != ['$libdir/evoke']:
        raise RuntimeError(
            'evoke extension catalog has stale or mixed C-module authority; '
            f'expected ["$libdir/evoke"], got {paths}. Install one coherent '
            'package and recreate the current extension catalog before '
            'rebuilding indexes.'
        )


def reconcile_extension_catalog(db, extension_schema_name):
    state = extension_version_state(db)
    if state['installed'] != state['default']:
        raise RuntimeError(
            'installed evoke catalog does not match the current-only package: '
            f'{state}; recreate the extension explicitly or rerun with '
            '--refresh-extension after preserving the source tables'
        )
    require_extension_catalog(db, extension_schema_name)
    require_extension_module_authority(db)
    return state


def ensure_extension(db, log_path):
    old_schema = extension_schema(db)
    preferred = old_schema or 'public'
    if old_schema is not None:
        reconcile_extension_catalog(db, old_schema)
        return old_schema

    for schema in dict.fromkeys([preferred, 'evoke_ext']):
        try:
            psql(db, f'CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)};')
            psql(
                db,
                (
                    'CREATE EXTENSION IF NOT EXISTS evoke '
                    f'WITH SCHEMA {quote_ident(schema)};'
                ),
            )
            actual = extension_schema(db)
            reconcile_extension_catalog(db, actual)
            log_event(
                log_path,
                {'event': 'ensure_extension', 'database': db, 'schema': actual},
            )
            return actual
        except Exception as exc:
            if schema == 'evoke_ext':
                raise
            log_event(
                log_path,
                {
                    'event': 'extension_schema_fallback',
                    'database': db,
                    'schema': schema,
                    'error': str(exc),
                },
            )
    raise RuntimeError('could not create evoke extension')


def refresh_extension_catalog(db, indexes, log_path):
    old_schema = extension_schema(db)
    if old_schema is None:
        return ensure_extension(db, log_path)

    drop_indexes = [
        qname(idx['schema_name'], idx['index_name'])
        for idx in indexes
        if idx.get('access_method') == 'evoke'
    ]
    statements = ['BEGIN;']
    statements.extend(
        f'DROP INDEX IF EXISTS {index_name};'
        for index_name in drop_indexes
    )
    statements.extend(
        [
            'DROP EXTENSION evoke;',
            (
                'CREATE EXTENSION evoke '
                f'WITH SCHEMA {quote_ident(old_schema)};'
            ),
            (
                'DO $evoke_catalog$\n'
                'BEGIN\n'
                f'    IF {qname(old_schema, "evoke_catalog_contract_internal")}() '
                f'IS DISTINCT FROM {sql_literal(CATALOG_CONTRACT)} THEN\n'
                "        RAISE EXCEPTION 'unexpected evoke catalog contract';\n"
                '    END IF;\n'
                'END\n'
                '$evoke_catalog$;'
            ),
            'COMMIT;',
        ]
    )
    psql(db, '\n'.join(statements))
    actual = extension_schema(db)
    if actual != old_schema:
        raise RuntimeError(
            'refreshed evoke extension changed schema unexpectedly; '
            f'expected {old_schema}, got {actual or "<missing>"}'
        )
    reconcile_extension_catalog(db, actual)
    log_event(
        log_path,
        {
            'event': 'refresh_extension',
            'database': db,
            'schema': actual,
            'dropped_indexes': len(drop_indexes),
        },
    )
    return actual


def require_semantic_query_runtime(db, extension_schema_name):
    expected_version = expected_onnxruntime_version()
    expected_api = expected_onnxruntime_api(expected_version)
    build_info_fn = qname(
        extension_schema_name,
        'evoke_onnxruntime_build_info',
    )
    probe_fn = qname(extension_schema_name, 'evoke_onnxruntime_probe')
    build_info, _ = psql(db, f'SELECT {build_info_fn}();', timeout=60)
    expected_build_info = f'enabled:api={expected_api}'
    if build_info != expected_build_info:
        raise RuntimeError(
            'semantic rebuild requires the pinned query-capable evoke '
            f'binary; expected {expected_build_info}, got '
            f'{build_info or "<empty>"}'
        )
    probe, _ = psql(db, f'SELECT {probe_fn}();', timeout=60)
    expected_probe_suffix = f':version={expected_version}'
    if (
        not probe.startswith('available:')
        or not probe.endswith(expected_probe_suffix)
    ):
        raise RuntimeError(
            'semantic rebuild requires the pinned local ONNX Runtime; '
            f'expected version {expected_version}, got '
            f'{probe or "<empty>"}'
        )
    return {
        'expected_version': expected_version,
        'build_info': build_info,
        'probe': probe,
    }


def index_exists(db, idx):
    sql = (
        "SELECT EXISTS ("
        "SELECT 1 FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = "
        + sql_literal(idx['schema_name'])
        + " AND c.relname = "
        + sql_literal(idx['index_name'])
        + ");"
    )
    out, _ = psql(db, sql)
    return out == 't'


def table_exists(db, idx):
    sql = (
        "SELECT to_regclass("
        + sql_literal(qname(idx['table_schema_name'], idx['table_name']))
        + ") IS NOT NULL;"
    )
    out, _ = psql(db, sql)
    return out == 't'


def validate_catalog_index(db, idx, index_name):
    index_regclass = qname(idx['schema_name'], index_name)
    sql = (
        "SELECT COALESCE(bool_and(i.indisvalid AND i.indisready), false) "
        "FROM pg_index i "
        "WHERE i.indexrelid = "
        + sql_literal(index_regclass)
        + "::regclass;"
    )
    out, _ = psql(db, sql)
    if out != 't':
        raise RuntimeError(f'index is not valid and ready: {index_regclass}')


def validate_index(db, idx, index_name):
    validate_catalog_index(db, idx, index_name)


def current_generation_ready(db, idx):
    schema = extension_schema(db)
    if not schema:
        raise RuntimeError(f'evoke extension schema is unavailable in {db}')

    index_regclass = qname(idx['schema_name'], idx['index_name'])
    state_function = qname(schema, 'evoke_index_runtime_state_json')
    sql = (
        "SELECT COALESCE("
        "state->'generation'->>'storage' = 'convergent_segments' "
        "AND state->'generation'->>'payload_health' = 'ok' "
        "AND NOT (state->'generation'->>'rebuild_required')::boolean, "
        "false) "
        "FROM (SELECT "
        f'{state_function}({sql_literal(index_regclass)}::regclass) AS state'
        ") generation_state;"
    )
    try:
        out, _ = psql(db, sql)
    except RuntimeError as exc:
        message = str(exc)
        unsupported_markers = (
            'unsupported evoke index metapage version',
            'unsupported evoke index storage layout',
        )
        if any(marker in message for marker in unsupported_markers):
            return False
        raise
    return out == 't'


def target_index_ready(
    db,
    idx,
    action,
    runtime_precision,
    semantic_impact_precision=None,
    semantic_alpha_mass=None,
):
    semantic = action in SEMANTIC_ACTIONS
    if not semantic:
        return False

    schema_literal = sql_literal(idx['schema_name'])
    index_literal = sql_literal(idx['index_name'])
    precision_literal = sql_literal(runtime_precision)
    impact_condition = ''
    alpha_condition = ''
    if semantic_impact_precision is not None:
        impact_literal = sql_literal(semantic_impact_precision)
        impact_condition = (
            " AND COALESCE((SELECT split_part(option, '=', 2) "
            "FROM unnest(COALESCE(c.reloptions, ARRAY[]::text[])) option "
            "WHERE option LIKE 'semantic_impact_precision=%' LIMIT 1), "
            f"'f32') = {impact_literal}"
        )
    if semantic_alpha_mass is not None:
        alpha_condition = (
            " AND COALESCE((SELECT split_part(option, '=', 2)::float8 "
            "FROM unnest(COALESCE(c.reloptions, ARRAY[]::text[])) option "
            "WHERE option LIKE 'semantic_alpha_mass=%' LIMIT 1), 1.0) "
            f"= {semantic_alpha_mass:.17g}::float8"
        )
    sql = (
        "SELECT COALESCE(bool_or("
        "am.amname = 'evoke' AND i.indisvalid AND i.indisready "
        "AND 'sae=true' = ANY(COALESCE(c.reloptions, ARRAY[]::text[])) "
        "AND COALESCE(("
        "SELECT split_part(option, '=', 2) "
        "FROM unnest(COALESCE(c.reloptions, ARRAY[]::text[])) option "
        "WHERE option LIKE 'runtime_precision=%' LIMIT 1"
        f"), 'fp16') = {precision_literal}"
        f"{impact_condition}{alpha_condition}"
        "), false) "
        "FROM pg_class c "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "JOIN pg_am am ON am.oid = c.relam "
        "JOIN pg_index i ON i.indexrelid = c.oid "
        f"WHERE n.nspname = {schema_literal} "
        f"AND c.relname = {index_literal};"
    )
    out, _ = psql(db, sql)
    if out != 't':
        return False
    return current_generation_ready(db, idx)


def drop_leftover_temp(db, idx, temp_name):
    psql(db, f'DROP INDEX IF EXISTS {qname(idx["schema_name"], temp_name)};')


def rebuild_index(
    db,
    idx,
    action,
    had_evoke_refresh,
    log_path,
    runtime_precision,
    semantic_impact_precision=None,
    semantic_alpha_mass=None,
):
    if not table_exists(db, idx):
        log_event(
            log_path,
            {
                'event': 'skip_missing_table',
                'database': db,
                'index': idx['index_regclass'],
                'table': idx['table_regclass'],
            },
        )
        return 'skipped'

    if target_index_ready(
        db,
        idx,
        action,
        runtime_precision,
        semantic_impact_precision,
        semantic_alpha_mass,
    ):
        log_event(
            log_path,
            {
                'event': 'index_skip_ready',
                'database': db,
                'index': idx['index_regclass'],
                'action': action,
            },
        )
        return 'skipped'

    old_exists = index_exists(db, idx)
    direct_create = (
        (had_evoke_refresh and idx.get('access_method') == 'evoke')
        or not old_exists
    )
    create_name = idx['index_name'] if direct_create else temp_index_name(idx)
    if not direct_create:
        drop_leftover_temp(db, idx, create_name)

    log_event(
        log_path,
        {
            'event': 'index_start',
            'database': db,
            'index': idx['index_regclass'],
            'action': action,
            'create_name': create_name,
            'old_exists': old_exists,
            'index_bytes_before': idx.get('index_bytes'),
            'table_bytes': idx.get('table_bytes'),
        },
    )
    start = time.monotonic()
    psql(
        db,
        create_index_sql(
            idx,
            create_name,
            action,
            runtime_precision,
            semantic_impact_precision,
            semantic_alpha_mass,
        ),
    )
    validate_index(db, idx, create_name)
    if not direct_create and old_exists:
        psql(
            db,
            (
                'BEGIN;\n'
                f'DROP INDEX {qname(idx["schema_name"], idx["index_name"])};\n'
                f'ALTER INDEX {qname(idx["schema_name"], create_name)} '
                f'RENAME TO {quote_ident(idx["index_name"])};\n'
                'COMMIT;'
            ),
        )
        validate_index(db, idx, idx['index_name'])
    elapsed = time.monotonic() - start
    log_event(
        log_path,
        {
            'event': 'index_done',
            'database': db,
            'index': idx['index_regclass'],
            'action': action,
            'elapsed_sec': round(elapsed, 3),
        },
    )
    return 'done'


def maybe_drop_retired_extension(db, name, amname, log_path):
    if not extension_exists(db, name):
        return
    out, _ = psql(
        db,
        (
            "SELECT count(*) FROM pg_class c "
            "JOIN pg_am am ON am.oid = c.relam "
            "WHERE am.amname = "
            + sql_literal(amname)
            + ';'
        ),
    )
    if out != '0':
        return
    try:
        psql(db, f'DROP EXTENSION {quote_ident(name)};')
        log_event(
            log_path,
            {'event': 'drop_retired_extension', 'database': db, 'extension': name},
        )
    except Exception as exc:
        log_event(
            log_path,
            {
                'event': 'drop_retired_extension_failed',
                'database': db,
                'extension': name,
                'error': str(exc),
            },
        )


def require_refresh_plan_complete(db, indexes, runnable, plan_contract):
    if plan_contract != REBUILD_PLAN_CONTRACT:
        raise RuntimeError(
            'extension refresh requires a current catalog inventory plan; '
            f'expected contract {REBUILD_PLAN_CONTRACT}, got '
            f'{plan_contract or "<missing>"}'
        )

    live_indexes = {
        (idx['schema_name'], idx['index_name']): idx
        for idx in inventory_database(db)['indexes']
        if idx.get('access_method') == 'evoke'
    }
    planned_indexes = {
        (idx['schema_name'], idx['index_name']): (idx, action)
        for idx, action in runnable
        if idx.get('access_method') == 'evoke'
    }
    missing = sorted(set(live_indexes) - set(planned_indexes))
    if missing:
        formatted = ', '.join(qname(*identity) for identity in missing)
        raise RuntimeError(
            'extension refresh would drop live Evoke indexes absent from the '
            f'runnable plan: {formatted}'
        )

    stale = []
    for identity, live in live_indexes.items():
        planned, action = planned_indexes[identity]
        planned_keydefs = keydefs_for(planned, action)
        live_keydefs = [
            column['keydef'] for column in live.get('columns') or []
        ]
        planned_include = list(
            (planned.get('rebuild_plan') or {}).get('include_defs')
            or planned.get('include_defs')
            or []
        )
        live_include = list(live.get('include_defs') or [])
        if (
            planned.get('indexdef') != live.get('indexdef')
            or planned_keydefs != live_keydefs
            or planned_include != live_include
            or (planned.get('predicate') or '').strip()
            != (live.get('predicate') or '').strip()
        ):
            stale.append(identity)
    if stale:
        formatted = ', '.join(qname(*identity) for identity in stale)
        raise RuntimeError(
            'extension refresh plan no longer matches the live index DDL; '
            f'regenerate it before destructive work: {formatted}'
        )


def process_database(db, indexes, args, log_path):
    started = time.monotonic()
    actions = [(idx, classify(idx)) for idx in indexes]
    runnable = [
        (idx, action)
        for idx, action in actions
        if action_allowed(action, args.mode)
        and index_filter_allowed(db, idx, args)
    ]
    skipped = len(actions) - len(runnable)
    if not runnable:
        log_event(
            log_path,
            {'event': 'database_skip', 'database': db, 'skipped': skipped},
        )
        return {'database': db, 'done': 0, 'skipped': skipped, 'failed': 0}

    try:
        if args.refresh_extension:
            require_refresh_plan_complete(
                db,
                indexes,
                runnable,
                args.plan_contract,
            )
            log_event(
                log_path,
                {
                    'event': 'refresh_plan_preflight',
                    'database': db,
                    'live_evoke_indexes': sum(
                        idx.get('access_method') == 'evoke'
                        for idx in indexes
                    ),
                },
            )
        if args.dry_run:
            for idx, action in sort_runnable(runnable, args):
                log_event(
                    log_path,
                    {
                        'event': 'dry_run_index',
                        'database': db,
                        'index': idx['index_regclass'],
                        'action': action,
                    },
                )
            return {
                'database': db,
                'done': 0,
                'skipped': skipped,
                'failed': 0,
            }

        semantic_rebuild = any(
            action in SEMANTIC_ACTIONS for _, action in runnable
        )
        if args.refresh_extension:
            old_schema = extension_schema(db)
            if semantic_rebuild and old_schema is not None:
                runtime = require_semantic_query_runtime(db, old_schema)
                log_event(
                    log_path,
                    {
                        'event': 'semantic_runtime_preflight_before_refresh',
                        'database': db,
                        **runtime,
                    },
                )
            status_schema = refresh_extension_catalog(
                db,
                [idx for idx, _action in runnable],
                log_path,
            )
        else:
            status_schema = ensure_extension(db, log_path)
        if semantic_rebuild:
            runtime = require_semantic_query_runtime(db, status_schema)
            log_event(
                log_path,
                {
                    'event': 'semantic_runtime_preflight',
                    'database': db,
                    **runtime,
                },
            )
        done = 0
        failed = 0
        for idx, action in sort_runnable(runnable, args):
            try:
                result = rebuild_index(
                    db,
                    idx,
                    action,
                    args.refresh_extension,
                    log_path,
                    args.runtime_precision,
                    args.semantic_impact_precision,
                    args.semantic_alpha_mass,
                )
                done += int(result == 'done')
                skipped += int(result == 'skipped')
            except Exception as exc:
                failed += 1
                log_event(
                    log_path,
                    {
                        'event': 'index_failed',
                        'database': db,
                        'index': idx['index_regclass'],
                        'action': action,
                        'error': str(exc),
                    },
                )
                if args.stop_on_error:
                    raise
        if not args.dry_run:
            maybe_drop_retired_extension(db, 'psql_bm25s', 'psql_bm25s', log_path)
            maybe_drop_retired_extension(db, 'ii42', 'ii42', log_path)
        elapsed = time.monotonic() - started
        log_event(
            log_path,
            {
                'event': 'database_done',
                'database': db,
                'done': done,
                'skipped': skipped,
                'failed': failed,
                'elapsed_sec': round(elapsed, 3),
            },
        )
        return {'database': db, 'done': done, 'skipped': skipped, 'failed': failed}
    except Exception as exc:
        elapsed = time.monotonic() - started
        log_event(
            log_path,
            {
                'event': 'database_failed',
                'database': db,
                'error': str(exc),
                'elapsed_sec': round(elapsed, 3),
            },
        )
        return {'database': db, 'done': 0, 'skipped': skipped, 'failed': len(runnable)}


def load_plan(path):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    by_db = {}
    for idx in manifest['indexes']:
        by_db.setdefault(idx['database'], []).append(idx)
    return manifest, by_db


def filtered_jobs(by_db, args):
    jobs = []
    include = re.compile(args.include_db) if args.include_db else None
    exclude = re.compile(args.exclude_db) if args.exclude_db else None
    for db, indexes in by_db.items():
        if include and not include.search(db):
            continue
        if exclude and exclude.search(db):
            continue
        if any_pattern(args.skip_db_patterns, db):
            continue
        runnable = [
            idx
            for idx in indexes
            if action_allowed(classify(idx), args.mode)
            and index_filter_allowed(db, idx, args)
        ]
        if not runnable:
            continue
        bytes_total = sum(int(idx.get('index_bytes') or 0) for idx in runnable)
        jobs.append((db, indexes, bytes_total))
    if args.order_by == 'table-tuples-asc':
        jobs.sort(
            key=lambda item: min(
                int(idx.get('table_tuples_est') or 0)
                for idx in item[1]
                if action_allowed(classify(idx), args.mode)
                and index_filter_allowed(item[0], idx, args)
            )
        )
    elif args.order_by == 'table-bytes-asc':
        jobs.sort(
            key=lambda item: min(
                int(idx.get('table_bytes') or 0)
                for idx in item[1]
                if action_allowed(classify(idx), args.mode)
                and index_filter_allowed(item[0], idx, args)
            )
        )
    else:
        jobs.sort(key=lambda item: item[2], reverse=True)
    if args.limit_databases:
        jobs = jobs[: args.limit_databases]
    return jobs


def main():
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--plan', type=Path)
    input_group.add_argument('--inventory-output', type=Path)
    parser.add_argument(
        '--database',
        action='append',
        default=[],
        help='Database to inventory; repeat for multiple databases.',
    )
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--include-db')
    parser.add_argument('--exclude-db')
    parser.add_argument('--skip-db', action='append', default=[])
    parser.add_argument('--skip-index', action='append', default=[])
    parser.add_argument('--max-index-bytes', type=parse_size)
    parser.add_argument('--max-table-bytes', type=parse_size)
    parser.add_argument('--max-table-tuples', type=int)
    parser.add_argument('--limit-databases', type=int)
    parser.add_argument(
        '--order-by',
        choices=('index-bytes-desc', 'table-bytes-asc', 'table-tuples-asc'),
        default='index-bytes-desc',
    )
    parser.add_argument(
        '--mode',
        choices=('all', 'semantic', 'lexical-only'),
        default='all',
    )
    parser.add_argument(
        '--runtime-precision',
        choices=('fp16', 'fp32'),
        default='fp16',
        help='Runtime precision for rebuilt SSR indexes.',
    )
    parser.add_argument(
        '--semantic-impact-precision',
        choices=('f32', 'fp16', 'u8'),
        help=(
            'Override stored semantic impact precision for rebuilt SSR '
            'indexes; omit to preserve each index setting.'
        ),
    )
    parser.add_argument(
        '--semantic-alpha-mass',
        type=float,
        help=(
            'Override retained semantic impact mass for rebuilt SSR '
            'indexes; omit to preserve each index setting.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--refresh-extension', action='store_true')
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    if args.semantic_alpha_mass is not None and not (
        0.01 <= args.semantic_alpha_mass <= 1.0
    ):
        parser.error('--semantic-alpha-mass must be between 0.01 and 1.0')
    args.skip_db_patterns = compile_patterns(args.skip_db)
    args.skip_index_patterns = compile_patterns(args.skip_index)

    if args.inventory_output is not None:
        if not args.database:
            parser.error('--inventory-output requires at least one --database')
        manifest = generate_inventory_plan(args.database)
        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(
            json.dumps(
                {
                    'contract': manifest['contract'],
                    'databases': len(manifest['databases']),
                    'indexes': len(manifest['indexes']),
                    'output': str(args.inventory_output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.log is None:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        args.log = args.plan.with_name(f'evoke-reindex-run-{stamp}.jsonl')

    manifest, by_db = load_plan(args.plan)
    args.plan_contract = manifest.get('contract')
    jobs = filtered_jobs(by_db, args)
    log_event(
        args.log,
        {
            'event': 'run_start',
            'plan': str(args.plan),
            'jobs': args.jobs,
            'database_jobs': len(jobs),
            'dry_run': args.dry_run,
            'mode': args.mode,
            'max_index_bytes': args.max_index_bytes,
            'max_table_bytes': args.max_table_bytes,
            'max_table_tuples': args.max_table_tuples,
            'refresh_extension': args.refresh_extension,
            'runtime_precision': args.runtime_precision,
            'order_by': args.order_by,
            'skip_db': args.skip_db,
            'skip_index': args.skip_index,
        },
    )
    totals = {'done': 0, 'skipped': 0, 'failed': 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(process_database, db, indexes, args, args.log)
            for db, indexes, _ in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for key in totals:
                totals[key] += result.get(key, 0)
    log_event(args.log, {'event': 'run_done', **totals})
    return 1 if totals['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
