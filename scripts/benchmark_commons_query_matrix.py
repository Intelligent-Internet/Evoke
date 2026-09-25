#!/usr/bin/env python3
"""Benchmark the read-only Evoke query contract used by Commons."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path('/tmp/evoke-commons-query-matrix.json')
APPLICATION_NAME = 'evoke_commons_query_matrix'
REPO_ROOT = Path(__file__).resolve().parents[1]
ONNXRUNTIME_VERSION_PATH = REPO_ROOT / 'packaging/onnxruntime.version'
MATRIX_CONTRACT_VERSION = 10
CATALOG_CONTRACT = 'evoke_catalog_v1'
FIRST_QUERY_MAX_MS = 2_000.0
MAX_RESTART_EVIDENCE_AGE_SECONDS = 600
REQUIRED_CATALOG_SIGNATURES = (
    'evoke_catalog_contract_internal()',
    'evoke_index_generation_status_internal(regclass)',
    'evoke_index_runtime_state_json(regclass)',
    'evoke_query_trace_internal()',
)
REQUIRED_FILTER_STATS = {
    'data_arxiv': {
        'categories',
        'organizations',
        'publish_date',
    },
    'data_pubmed': {
        'categories',
        'journal_title',
        'nlm_ta',
        'publish_date',
        'publish_date_end_bound',
        'publish_date_has_day',
        'publish_date_start_bound',
    },
    'data_policy_ca_chunks': {'policy_ca_doc_id'},
    'data_policy_tx_chunks': {'policy_tx_doc_id'},
    'data_policy_wa_chunks': {'policy_wa_doc_id'},
    'sys_chunks': {'document_id'},
}
REQUIRED_GENERATED_COLUMNS = {
    'publish_date_start_bound': {
        'type': 'integer',
        'expression': (
            'CASE WHEN (((publish_date / 100) % 100) = 0) '
            'THEN (((publish_date / 10000) * 10000) + 101) '
            'WHEN ((publish_date % 100) = 0) '
            'THEN (((publish_date / 100) * 100) + 1) '
            'ELSE publish_date END'
        ),
    },
    'publish_date_end_bound': {
        'type': 'integer',
        'expression': (
            'CASE WHEN (((publish_date / 100) % 100) = 0) '
            'THEN (((publish_date / 10000) * 10000) + 1231) '
            'WHEN ((publish_date % 100) = 0) '
            'THEN (((publish_date / 100) * 100) + 31) '
            'ELSE publish_date END'
        ),
    },
    'publish_date_has_day': {
        'type': 'boolean',
        'expression': '((publish_date % 100) <> 0)',
    },
}
DEFAULT_QUERY = 'consumer privacy data deletion rights'
ARXIV_QUERY = 'graph neural network optimization'
PUBMED_QUERY = 'cancer immunotherapy survival'
PUBMED_INDEX_VISIBILITY_SQL = (
    "btrim(coalesce(source.title, '')) <> '' OR "
    "btrim(coalesce(source.abstract, '')) <> ''"
)
CA_STRESS_DOCUMENT_ID = '5fff0827-f141-456c-bdae-315b8f8cf9fb'
IDENTIFIER_RE = re.compile(
    r'^[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$'
)
SIMPLE_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    sql: str
    expected_max_ms: float
    oracle_sql: str | None = None
    rank_oracle_sql: str | None = None
    planner_sql: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the exact Commons Evoke query matrix against one PostgreSQL '
            'database without mutating tables or indexes.'
        ),
    )
    parser.add_argument(
        '--dsn',
        default=os.environ.get('EVOKE_COMMONS_DSN', 'dbname=ii_dev'),
    )
    parser.add_argument('--psql', default='psql')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument(
        '--concurrency',
        type=int,
        default=1,
        help='independent PostgreSQL sessions per benchmark wave',
    )
    parser.add_argument('--timeout-ms', type=int, default=30_000)
    parser.add_argument(
        '--require-restart-within-seconds',
        type=int,
        help=(
            'fail qualification unless PostgreSQL started within this many '
            'seconds before environment capture'
        ),
    )
    parser.add_argument('--only', action='append', default=[])
    parser.add_argument('--pubmed-index')
    parser.add_argument(
        '--verify-rank-oracle',
        action='store_true',
        help=(
            'compare selected bounded JSON filters with the exact tid[] '
            'ranking route'
        ),
    )
    parser.add_argument(
        '--plans-only',
        action='store_true',
        help=(
            'capture environment and EXPLAIN evidence without running ranked '
            'queries; output is never qualification evidence'
        ),
    )
    parser.add_argument(
        '--ca-stress-document-id',
        default=CA_STRESS_DOCUMENT_ID,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 2:
        raise ValueError(
            '--repeats must include separate first and warm waves'
        )
    if args.concurrency < 1:
        raise ValueError('--concurrency must be positive')
    if args.timeout_ms < 1:
        raise ValueError('--timeout-ms must be positive')
    restart_limit = getattr(args, 'require_restart_within_seconds', None)
    if restart_limit is not None and restart_limit < 1:
        raise ValueError('--require-restart-within-seconds must be positive')
    if (
        restart_limit is not None
        and restart_limit > MAX_RESTART_EVIDENCE_AGE_SECONDS
    ):
        raise ValueError(
            '--require-restart-within-seconds cannot exceed '
            f'{MAX_RESTART_EVIDENCE_AGE_SECONDS}'
        )
    if args.verify_rank_oracle and args.concurrency != 1:
        raise ValueError(
            '--verify-rank-oracle requires --concurrency 1; run exactness '
            'and concurrent latency as separate qualification passes'
        )


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote_identifier(value: str) -> str:
    if not SIMPLE_IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f'unsafe SQL identifier: {value!r}')
    return '"' + value.replace('"', '""') + '"'


def pinned_onnxruntime_version() -> str:
    try:
        version = ONNXRUNTIME_VERSION_PATH.read_text(
            encoding='utf-8',
        ).strip()
    except OSError as exc:
        raise RuntimeError(
            'Commons qualification requires the pinned ONNX Runtime '
            f'version; could not read {ONNXRUNTIME_VERSION_PATH}: {exc}'
        ) from exc
    if not version:
        raise RuntimeError(
            'Commons qualification requires a non-empty pinned ONNX '
            'Runtime version'
        )
    return version


def expected_onnxruntime_api(version: str) -> int:
    parts = version.split('.')
    if len(parts) < 2 or parts[0] != '1' or not parts[1].isdigit():
        raise RuntimeError(
            f'unsupported pinned ONNX Runtime version: {version!r}'
        )
    return int(parts[1])


def regclass_literal(value: str) -> str:
    if not IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f'unsafe schema-qualified relation name: {value!r}')
    return f'{sql_literal(value)}::regclass'


def json_literal(value: dict[str, Any]) -> str:
    return f'{sql_literal(json.dumps(value, separators=(",", ":")))}::jsonb'


def psql_command(args: argparse.Namespace) -> list[str]:
    return [
        args.psql,
        '--dbname',
        args.dsn,
        '-X',
        '-qAt',
        '-v',
        'ON_ERROR_STOP=1',
    ]


def run_psql(
    args: argparse.Namespace,
    sql_text: str,
    *,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        psql_command(args),
        input=sql_text,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env={**os.environ, 'PGAPPNAME': APPLICATION_NAME},
    )


def query_json(args: argparse.Namespace, sql_text: str) -> Any:
    result = run_psql(args, sql_text, timeout_seconds=15.0)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f'expected one JSON row, got {lines!r}')
    return json.loads(lines[0])


def filter_oracle_sql(
    table_name: str,
    violation_sql: str,
    index_visibility_sql: str,
) -> str:
    regclass_literal(table_name)
    return f"""
    SELECT json_build_object(
        'allowed_documents', count(*)::bigint
    )
    FROM ONLY {table_name} AS source
    WHERE ({index_visibility_sql})
      AND NOT ({violation_sql});
    """


def hit_signature_sql(hit_alias: str) -> str:
    return f"""
        COALESCE(
            jsonb_agg(
                jsonb_build_array(
                    {hit_alias}.ctid::text,
                    {hit_alias}.doc_id,
                    encode(float4send({hit_alias}.score), 'hex')
                )
                ORDER BY {hit_alias}.score DESC, {hit_alias}.doc_id
            ),
            '[]'::jsonb
        )
    """


def rank_oracle_sql(
    table_name: str,
    violation_sql: str,
    index_visibility_sql: str,
    allowed_search_sql: str,
) -> str:
    regclass_literal(table_name)
    return f"""
    WITH allowed AS MATERIALIZED (
        SELECT COALESCE(
            array_agg(source.ctid ORDER BY source.ctid),
            ARRAY[]::tid[]
        ) AS tids
        FROM ONLY {table_name} AS source
        WHERE ({index_visibility_sql})
          AND NOT ({violation_sql})
    ),
    hits AS MATERIALIZED (
        SELECT hit.*
        FROM allowed
        CROSS JOIN LATERAL {allowed_search_sql} AS hit
    )
    SELECT json_build_object(
        'allowed_documents', (SELECT cardinality(tids) FROM allowed),
        'hit_signature', (
            SELECT {hit_signature_sql('hits')}
            FROM hits
        )
    );
    """


def resolve_pubmed_index(args: argparse.Namespace) -> tuple[str, str]:
    if args.pubmed_index:
        regclass_literal(args.pubmed_index)
        return args.pubmed_index, 'explicit'
    result = query_json(
        args,
        """
        SELECT json_build_object(
            'stable', to_regclass(
                'commons.data_pubmed__title_abstract__field_aware_bm25_idx'
            )::text
        );
        """,
    )
    if result['stable']:
        return result['stable'], 'stable'
    raise RuntimeError('the stable Commons PubMed Evoke index is not present')


def timed_search_case(
    *,
    name: str,
    search_sql: str,
    table_name: str | None = None,
    violation_sql: str = 'FALSE',
    index_visibility_sql: str = 'TRUE',
    allowed_search_sql: str | None = None,
    expected_max_ms: float,
) -> BenchmarkCase:
    oracle_sql = None
    ranked_oracle_sql = None
    planner_sql = None
    if table_name is not None:
        regclass_literal(table_name)
        oracle_sql = filter_oracle_sql(
            table_name,
            violation_sql,
            index_visibility_sql,
        )
        planner_sql = f"""
        EXPLAIN (FORMAT JSON, COSTS true, SETTINGS true)
        SELECT source.ctid
        FROM ONLY {table_name} AS source
        WHERE ({index_visibility_sql})
          AND NOT ({violation_sql});
        """
        if allowed_search_sql is not None:
            ranked_oracle_sql = rank_oracle_sql(
                table_name,
                violation_sql,
                index_visibility_sql,
                allowed_search_sql,
            )
        summary_sql = f"""
        SELECT
            count(*)::bigint AS hits,
            count(*) FILTER (
                WHERE ({violation_sql}) IS NOT FALSE
            )::bigint AS violations,
            {hit_signature_sql('h')} AS hit_signature
        FROM hits AS h
        JOIN {table_name} AS source ON source.ctid = h.ctid
        """
    else:
        summary_sql = f"""
        SELECT
            count(*)::bigint AS hits,
            0::bigint AS violations,
            {hit_signature_sql('hits')} AS hit_signature
        FROM hits
        """
    sql_text = f"""
    WITH started AS MATERIALIZED (
        SELECT clock_timestamp() AS started_at
    ),
    hits AS MATERIALIZED (
        SELECT hit.*
        FROM started
        CROSS JOIN LATERAL {search_sql} AS hit
    ),
    summary AS MATERIALIZED (
        {summary_sql}
    )
    SELECT json_build_object(
        'name', {sql_literal(name)},
        'hits', summary.hits,
        'violations', summary.violations,
        'hit_signature', summary.hit_signature,
        'elapsed_ms', extract(
            epoch FROM clock_timestamp() - started.started_at
        ) * 1000.0
    )
    FROM started CROSS JOIN summary;
    """
    return BenchmarkCase(
        name,
        sql_text,
        expected_max_ms,
        oracle_sql,
        ranked_oracle_sql,
        planner_sql,
    )


def field_search(
    index_name: str,
    query_text: str,
    fields: list[str],
    weights: list[float],
    filters: dict[str, Any] | None = None,
    k: int = 50,
) -> str:
    field_sql = 'ARRAY[' + ','.join(sql_literal(item) for item in fields) + ']'
    weight_sql = 'ARRAY[' + ','.join(str(item) for item in weights) + ']'
    arguments = [
        regclass_literal(index_name),
        sql_literal(query_text),
        f'{field_sql}::text[]',
        f'{weight_sql}::real[]',
    ]
    if filters is not None:
        arguments.append(json_literal(filters))
    arguments.append(str(k))
    return f'evoke_query({", ".join(arguments)})'


def field_search_allowed(
    index_name: str,
    query_text: str,
    fields: list[str],
    weights: list[float],
    allowed_tids_sql: str,
    k: int = 50,
) -> str:
    field_sql = 'ARRAY[' + ','.join(sql_literal(item) for item in fields) + ']'
    weight_sql = 'ARRAY[' + ','.join(str(item) for item in weights) + ']'
    return (
        'evoke_query('
        f'{regclass_literal(index_name)}, {sql_literal(query_text)}, '
        f'{field_sql}::text[], {weight_sql}::real[], '
        f'{allowed_tids_sql}, {k})'
    )


def single_search(
    index_name: str,
    query_text: str,
    filters: dict[str, Any] | None = None,
    k: int = 50,
) -> str:
    arguments = [regclass_literal(index_name), sql_literal(query_text)]
    if filters is not None:
        arguments.append(json_literal(filters))
    arguments.append(str(k))
    return f'evoke_query({", ".join(arguments)})'


def single_search_allowed(
    index_name: str,
    query_text: str,
    allowed_tids_sql: str,
    k: int = 50,
) -> str:
    return (
        'evoke_query('
        f'{regclass_literal(index_name)}, {sql_literal(query_text)}, '
        f'{allowed_tids_sql}, {k})'
    )


def policy_union_case(
    jurisdiction: str,
    doc_index: str,
    chunk_index: str,
) -> BenchmarkCase:
    doc_table = f'commons.data_policy_{jurisdiction}'
    chunk_table = f'{doc_table}_chunks'
    chunk_doc_column = f'policy_{jurisdiction}_doc_id'
    regclass_literal(doc_table)
    regclass_literal(chunk_table)
    doc_search = field_search(
        doc_index,
        DEFAULT_QUERY,
        ['title', 'description'],
        [2.0, 1.0],
    )
    chunk_search = single_search(chunk_index, DEFAULT_QUERY)
    name = f'{jurisdiction}_policy_union'
    sql_text = f"""
    WITH started AS MATERIALIZED (
        SELECT clock_timestamp() AS started_at
    ),
    doc_hits AS MATERIALIZED (
        SELECT source.id
        FROM started
        CROSS JOIN LATERAL {doc_search} AS hit
        JOIN {doc_table} AS source ON source.ctid = hit.ctid
    ),
    chunk_hits AS MATERIALIZED (
        SELECT source.{chunk_doc_column} AS id
        FROM started
        CROSS JOIN LATERAL {chunk_search} AS hit
        JOIN {chunk_table} AS source ON source.ctid = hit.ctid
    ),
    candidates AS MATERIALIZED (
        SELECT id FROM doc_hits UNION SELECT id FROM chunk_hits
    )
    SELECT json_build_object(
        'name', {sql_literal(name)},
        'hits', count(*),
        'violations', 0,
        'hit_signature', COALESCE(
            jsonb_agg(
                candidates.id::text
                ORDER BY candidates.id::text
            ),
            '[]'::jsonb
        ),
        'elapsed_ms', extract(
            epoch FROM clock_timestamp() - started.started_at
        ) * 1000.0
    )
    FROM started CROSS JOIN candidates
    GROUP BY started.started_at;
    """
    return BenchmarkCase(name, sql_text, 300.0)


def chunk_scope_case(
    jurisdiction: str,
    index_name: str,
) -> BenchmarkCase:
    table_name = f'commons.data_policy_{jurisdiction}_chunks'
    column_name = f'policy_{jurisdiction}_doc_id'
    regclass_literal(table_name)
    name = f'{jurisdiction}_chunk_scope_ordinary'
    sql_text = f"""
    WITH started AS MATERIALIZED (
        SELECT clock_timestamp() AS started_at
    ),
    sample AS MATERIALIZED (
        SELECT {column_name} AS document_id
        FROM {table_name}
        ORDER BY {column_name}
        LIMIT 1
    ),
    hits AS MATERIALIZED (
        SELECT hit.*
        FROM started CROSS JOIN sample
        CROSS JOIN LATERAL evoke_query(
            {regclass_literal(index_name)},
            {sql_literal(DEFAULT_QUERY)},
            jsonb_build_object(
                {sql_literal(column_name)},
                jsonb_build_object('eq', sample.document_id::text)
            ),
            50
        ) AS hit
    ),
    summary AS MATERIALIZED (
        SELECT
            count(*)::bigint AS hits,
            count(*) FILTER (
                WHERE source.{column_name}
                    IS DISTINCT FROM sample.document_id
            )::bigint AS violations,
            {hit_signature_sql('hits')} AS hit_signature
        FROM sample CROSS JOIN hits
        JOIN {table_name} AS source ON source.ctid = hits.ctid
    )
    SELECT json_build_object(
        'name', {sql_literal(name)},
        'hits', summary.hits,
        'violations', summary.violations,
        'hit_signature', summary.hit_signature,
        'elapsed_ms', extract(
            epoch FROM clock_timestamp() - started.started_at
        ) * 1000.0
    )
    FROM started CROSS JOIN summary;
    """
    return BenchmarkCase(name, sql_text, 500.0)


def ca_stress_scope_case(index_name: str, document_id: str) -> BenchmarkCase:
    name = 'ca_chunk_scope_stress'
    filters = {'policy_ca_doc_id': {'eq': document_id}}
    search_sql = single_search(index_name, DEFAULT_QUERY, filters)
    return timed_search_case(
        name=name,
        search_sql=search_sql,
        table_name='commons.data_policy_ca_chunks',
        violation_sql=(
            'source.policy_ca_doc_id <> '
            f'{sql_literal(document_id)}::uuid'
        ),
        allowed_search_sql=single_search_allowed(
            index_name,
            DEFAULT_QUERY,
            'allowed.tids',
        ),
        expected_max_ms=2_000.0,
    )


def build_cases(
    pubmed_index: str,
    ca_stress_document_id: str,
) -> list[BenchmarkCase]:
    arxiv_index = (
        'commons.data_arxiv__title_abstract__field_aware_bm25_idx'
    )
    arxiv_fields = ['title', 'abstract']
    arxiv_weights = [2.0, 1.0]
    pubmed_fields = ['title', 'abstract']
    pubmed_weights = [2.0, 1.0]
    pubmed_category_filters = {
        'publish_date': {
            'range': {'gte': 20240101, 'lte': 20241231},
        },
        'categories': {'ilike_any': ['%cancer%']},
    }
    pubmed_category_violation = (
        'source.publish_date < 20240101 OR '
        'source.publish_date > 20241231 OR NOT EXISTS ('
        'SELECT 1 FROM unnest(source.categories) AS item(value) '
        "WHERE item.value ILIKE '%cancer%')"
    )
    cases = [
        timed_search_case(
            name='arxiv_unfiltered',
            search_sql=field_search(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
            ),
            expected_max_ms=300.0,
        ),
        timed_search_case(
            name='arxiv_date_category',
            search_sql=field_search(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
                {
                    'publish_date': {
                        'range': {'gte': 20260501, 'lte': 20261231},
                    },
                    'categories': {'overlap': ['cs.LG']},
                },
            ),
            table_name='commons.data_arxiv',
            violation_sql=(
                'source.publish_date < 20260501 OR '
                'source.publish_date > 20261231 OR '
                "NOT source.categories && ARRAY['cs.LG']::text[]"
            ),
            allowed_search_sql=field_search_allowed(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
                'allowed.tids',
            ),
            expected_max_ms=500.0,
        ),
        timed_search_case(
            name='arxiv_broad_date',
            search_sql=field_search(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
                {'publish_date': {'range': {'gte': 20240101}}},
            ),
            table_name='commons.data_arxiv',
            violation_sql='source.publish_date < 20240101',
            expected_max_ms=1_000.0,
        ),
        timed_search_case(
            name='arxiv_organization_selective',
            search_sql=field_search(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
                {
                    'organizations': {
                        'ilike_any': ['%stanford university%'],
                    },
                },
            ),
            table_name='commons.data_arxiv',
            violation_sql=(
                "source.organizations NOT ILIKE '%stanford university%'"
            ),
            expected_max_ms=500.0,
        ),
        timed_search_case(
            name='arxiv_organization_broad',
            search_sql=field_search(
                arxiv_index,
                ARXIV_QUERY,
                arxiv_fields,
                arxiv_weights,
                {'organizations': {'ilike_any': ['%university%']}},
            ),
            table_name='commons.data_arxiv',
            violation_sql="source.organizations NOT ILIKE '%university%'",
            expected_max_ms=1_000.0,
        ),
        timed_search_case(
            name='pubmed_unfiltered',
            search_sql=field_search(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
            ),
            expected_max_ms=300.0,
        ),
        timed_search_case(
            name='pubmed_date_journal',
            search_sql=field_search(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                {
                    'publish_date': {'range': {'gte': 20240101}},
                    'journal_title': {'ilike_any': ['%nature%']},
                },
            ),
            table_name='commons.data_pubmed',
            index_visibility_sql=PUBMED_INDEX_VISIBILITY_SQL,
            violation_sql=(
                'source.publish_date < 20240101 OR '
                "source.journal_title::text NOT ILIKE '%nature%'"
            ),
            allowed_search_sql=field_search_allowed(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                'allowed.tids',
            ),
            expected_max_ms=500.0,
        ),
        timed_search_case(
            name='pubmed_broad_date',
            search_sql=field_search(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                {'publish_date': {'range': {'gte': 20240101}}},
            ),
            table_name='commons.data_pubmed',
            index_visibility_sql=PUBMED_INDEX_VISIBILITY_SQL,
            violation_sql='source.publish_date < 20240101',
            expected_max_ms=1_000.0,
        ),
        timed_search_case(
            name='pubmed_partial_date',
            search_sql=field_search(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                {
                    'publish_date_end_bound': {
                        'range': {'gte': 20240101},
                    },
                    'publish_date_start_bound': {
                        'range': {'lte': 20241231},
                    },
                },
            ),
            table_name='commons.data_pubmed',
            index_visibility_sql=PUBMED_INDEX_VISIBILITY_SQL,
            violation_sql=(
                'source.publish_date_end_bound < 20240101 OR '
                'source.publish_date_start_bound > 20241231'
            ),
            expected_max_ms=1_000.0,
        ),
        timed_search_case(
            name='pubmed_date_category',
            search_sql=field_search(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                pubmed_category_filters,
            ),
            table_name='commons.data_pubmed',
            index_visibility_sql=PUBMED_INDEX_VISIBILITY_SQL,
            violation_sql=pubmed_category_violation,
            allowed_search_sql=field_search_allowed(
                pubmed_index,
                PUBMED_QUERY,
                pubmed_fields,
                pubmed_weights,
                'allowed.tids',
            ),
            expected_max_ms=500.0,
        ),
    ]
    for jurisdiction in ('ca', 'tx', 'wa'):
        doc_index = (
            f'commons.data_policy_{jurisdiction}'
            '__title_description__field_aware_bm25_idx'
        )
        chunk_index = (
            f'commons.data_policy_{jurisdiction}_chunks'
            '__content__bm25_idx'
        )
        cases.append(
            policy_union_case(jurisdiction, doc_index, chunk_index)
        )
        cases.append(chunk_scope_case(jurisdiction, chunk_index))
    cases.append(
        ca_stress_scope_case(
            'commons.data_policy_ca_chunks__content__bm25_idx',
            ca_stress_document_id,
        )
    )
    cases.append(
        timed_search_case(
            name='system_chunk_scope',
            search_sql=single_search(
                'commons.sys_chunks__content__bm25_idx',
                DEFAULT_QUERY,
                {'document_id': {'eq': 'arXiv:2604.10074'}},
            ),
            table_name='commons.sys_chunks',
            violation_sql="source.document_id <> 'arXiv:2604.10074'",
            allowed_search_sql=single_search_allowed(
                'commons.sys_chunks__content__bm25_idx',
                DEFAULT_QUERY,
                'allowed.tids',
            ),
            expected_max_ms=500.0,
        )
    )
    return cases


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    offset = max(0, min(len(ordered) - 1, int(len(ordered) * fraction)))
    return ordered[offset]


def summarize_attempts(
    case: BenchmarkCase,
    attempts: list[dict[str, Any]],
    *,
    rank_oracle_required: bool = False,
) -> dict[str, Any]:
    completed = [
        item for item in attempts if item.get('status') == 'completed'
    ]
    elapsed = [float(item['elapsed_ms']) for item in completed]
    concurrent_run = any('wave' in item for item in attempts)
    if concurrent_run:
        first = [
            float(item['elapsed_ms'])
            for item in completed
            if item.get('wave') == 1
        ]
        warm = [
            float(item['elapsed_ms'])
            for item in completed
            if int(item.get('wave', 1)) > 1
        ]
        if not first:
            first = elapsed[:1]
    else:
        first = elapsed[:1]
        warm = elapsed[1:]
    violations = sum(int(item.get('violations', 0)) for item in completed)
    result_stability_passed = bool(
        completed
        and all('hit_signature' in item for item in completed)
        and all(
            item['hit_signature'] == completed[0]['hit_signature']
            for item in completed[1:]
        )
    )
    summary: dict[str, Any] = {
        'name': case.name,
        'expected_max_ms': case.expected_max_ms,
        'expected_p95_max_ms': case.expected_max_ms * 2.0,
        'expected_first_max_ms': FIRST_QUERY_MAX_MS,
        'concurrency': max(
            (int(item.get('client', 1)) for item in attempts),
            default=1,
        ),
        'attempts': attempts,
        'completed': len(completed),
        'violations': violations,
        'result_stability': {
            'attempts': len(completed),
            'passed': result_stability_passed,
        },
        'passed': False,
    }
    filter_oracle_passed = True
    if case.oracle_sql is not None:
        oracle_attempts = []
        for item in completed:
            trace = item.get('evoke_trace')
            oracle = item.get('filter_oracle')
            expected = (
                int(oracle['allowed_documents'])
                if isinstance(oracle, dict)
                and oracle.get('allowed_documents') is not None
                else None
            )
            observed = (
                int(trace['allowed_documents'])
                if isinstance(trace, dict)
                and trace.get('allowed_documents') is not None
                else None
            )
            passed = expected is not None and observed == expected
            filter_oracle_passed = filter_oracle_passed and passed
            oracle_attempts.append({
                'attempt': item['attempt'],
                **(
                    {
                        'wave': item['wave'],
                        'client': item['client'],
                    }
                    if 'wave' in item
                    else {}
                ),
                'sql_allowed_documents': expected,
                'native_allowed_documents': observed,
                'passed': passed,
            })
        filter_oracle_passed = bool(
            completed
            and len(oracle_attempts) == len(completed)
            and filter_oracle_passed
        )
        summary['filter_oracle'] = {
            'attempts': oracle_attempts,
            'passed': filter_oracle_passed,
        }
    rank_oracle_required = bool(
        rank_oracle_required
        or any(item.get('rank_oracle') is not None for item in completed)
    )
    rank_oracle_passed = True
    if rank_oracle_required:
        rank_attempts = []
        for item in completed:
            oracle = item.get('rank_oracle')
            expected = (
                oracle.get('hit_signature')
                if isinstance(oracle, dict)
                else None
            )
            observed = item.get('hit_signature')
            passed = expected is not None and observed == expected
            rank_oracle_passed = rank_oracle_passed and passed
            rank_attempts.append({
                'attempt': item['attempt'],
                **(
                    {
                        'wave': item['wave'],
                        'client': item['client'],
                    }
                    if 'wave' in item
                    else {}
                ),
                'allowed_documents': (
                    oracle.get('allowed_documents')
                    if isinstance(oracle, dict)
                    else None
                ),
                'passed': passed,
            })
        rank_oracle_passed = bool(
            len(rank_attempts) == len(completed)
            and rank_oracle_passed
        )
        summary['rank_oracle'] = {
            'attempts': rank_attempts,
            'passed': rank_oracle_passed,
        }
    if elapsed:
        summary['first_ms'] = max(first)
        summary['first'] = {
            'count': len(first),
            'min_ms': min(first),
            'p50_ms': statistics.median(first),
            'p95_ms': percentile(first, 0.95),
            'max_ms': max(first),
            'mean_ms': statistics.fmean(first),
        }
    if warm:
        summary['warm'] = {
            'count': len(warm),
            'min_ms': min(warm),
            'p50_ms': statistics.median(warm),
            'p95_ms': percentile(warm, 0.95),
            'max_ms': max(warm),
            'mean_ms': statistics.fmean(warm),
        }
    summary['passed'] = bool(
        len(completed) == len(attempts)
        and violations == 0
        and result_stability_passed
        and first
        and max(first) <= FIRST_QUERY_MAX_MS
        and warm
        and statistics.median(warm) <= case.expected_max_ms
        and percentile(warm, 0.95) <= case.expected_max_ms * 2.0
        and filter_oracle_passed
        and rank_oracle_passed
    )
    return summary


def run_case_attempt(
    args: argparse.Namespace,
    case: BenchmarkCase,
    sql_text: str,
    *,
    attempt: int,
    verify_rank_oracle: bool,
    wave: int | None = None,
    client: int | None = None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {'attempt': attempt}

    if wave is not None:
        identity['wave'] = wave
    if client is not None:
        identity['client'] = client

    started = time.monotonic()
    try:
        result = run_psql(
            args,
            sql_text,
            timeout_seconds=max(5.0, args.timeout_ms / 1000.0 + 5.0),
        )
    except subprocess.TimeoutExpired:
        return {
            **identity,
            'status': 'client_timeout',
            'wall_ms': (time.monotonic() - started) * 1000.0,
        }
    wall_ms = (time.monotonic() - started) * 1000.0
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        status = (
            'statement_timeout'
            if 'statement timeout' in error
            else 'error'
        )
        return {
            **identity,
            'status': status,
            'wall_ms': wall_ms,
            'error': error,
        }
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[0])
        trace_payload = json.loads(lines[1])
        next_line = 2
        oracle_payload = None
        rank_oracle_payload = None
        if case.oracle_sql is not None:
            oracle_payload = json.loads(lines[next_line])
            next_line += 1
        if verify_rank_oracle:
            rank_oracle_payload = json.loads(lines[next_line])
            next_line += 1
        if next_line != len(lines):
            raise ValueError(
                f'expected {next_line} JSON rows, got {len(lines)}'
            )
    except (IndexError, json.JSONDecodeError, ValueError) as exc:
        return {
            **identity,
            'status': 'parse_error',
            'wall_ms': wall_ms,
            'error': str(exc),
            'stdout': result.stdout,
        }
    payload['evoke_trace'] = trace_payload.get('evoke_trace')
    if oracle_payload is not None:
        payload['filter_oracle'] = oracle_payload
    if rank_oracle_payload is not None:
        payload['rank_oracle'] = rank_oracle_payload
    payload.update({
        **identity,
        'status': 'completed',
        'wall_ms': wall_ms,
    })
    return payload


def run_case(
    args: argparse.Namespace,
    case: BenchmarkCase,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    concurrency = int(getattr(args, 'concurrency', 1))
    verify_rank_oracle = bool(
        getattr(args, 'verify_rank_oracle', False)
        and case.rank_oracle_sql is not None
    )
    sql_parts = [
        'BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;',
        f"SET LOCAL statement_timeout = '{args.timeout_ms}ms';",
        case.sql,
        (
            "SELECT json_build_object("
            "'evoke_trace', evoke_query_trace_internal());"
        ),
    ]
    if case.oracle_sql is not None:
        sql_parts.append(case.oracle_sql)
    if verify_rank_oracle:
        sql_parts.append(case.rank_oracle_sql)
    sql_parts.append('COMMIT;')
    sql_text = '\n'.join(sql_parts)

    if concurrency == 1:
        for attempt in range(1, args.repeats + 1):
            attempts.append(
                run_case_attempt(
                    args,
                    case,
                    sql_text,
                    attempt=attempt,
                    verify_rank_oracle=verify_rank_oracle,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency,
        ) as executor:
            for wave in range(1, args.repeats + 1):
                futures = [
                    executor.submit(
                        run_case_attempt,
                        args,
                        case,
                        sql_text,
                        attempt=(wave - 1) * concurrency + client,
                        verify_rank_oracle=verify_rank_oracle,
                        wave=wave,
                        client=client,
                    )
                    for client in range(1, concurrency + 1)
                ]
                attempts.extend(future.result() for future in futures)
    return summarize_attempts(
        case,
        attempts,
        rank_oracle_required=verify_rank_oracle,
    )


def collect_environment(
    args: argparse.Namespace,
    pubmed_index: str,
    pubmed_source: str,
) -> dict[str, Any]:
    indexes = [
        'commons.data_arxiv__title_abstract__field_aware_bm25_idx',
        pubmed_index,
        'commons.data_policy_ca__title_description__field_aware_bm25_idx',
        'commons.data_policy_ca_chunks__content__bm25_idx',
        'commons.data_policy_tx__title_description__field_aware_bm25_idx',
        'commons.data_policy_tx_chunks__content__bm25_idx',
        'commons.data_policy_wa__title_description__field_aware_bm25_idx',
        'commons.data_policy_wa_chunks__content__bm25_idx',
        'commons.sys_chunks__content__bm25_idx',
    ]
    required_catalog_values = ','.join(
        f'({sql_literal(signature)})'
        for signature in REQUIRED_CATALOG_SIGNATURES
    )
    sql_text = f"""
    WITH extension_catalog AS (
        SELECT
            installed.oid AS extension_oid,
            installed.extversion AS installed_version,
            available.default_version AS available_version,
            namespace.nspname AS extension_schema
        FROM pg_extension AS installed
        JOIN pg_namespace AS namespace
          ON namespace.oid = installed.extnamespace
        JOIN pg_available_extensions AS available
          ON available.name = installed.extname
        WHERE installed.extname = 'evoke'
    ),
    required_catalog_function(signature) AS (
        VALUES {required_catalog_values}
    ),
    resolved_catalog_function AS (
        SELECT
            required.signature,
            to_regprocedure(
                format(
                    '%I.%s',
                    extension_catalog.extension_schema,
                    required.signature
                )
            ) AS procedure_oid,
            extension_catalog.extension_oid
        FROM required_catalog_function AS required
        CROSS JOIN extension_catalog
    ),
    extension_c_modules AS (
        SELECT
            coalesce(
                jsonb_agg(
                    DISTINCT installed_procedure.probin
                    ORDER BY installed_procedure.probin
                ),
                '[]'::jsonb
            ) AS paths,
            count(*) > 0
            AND bool_and(
                installed_procedure.probin = '$libdir/evoke'
            ) AS current
        FROM extension_catalog
        JOIN pg_depend AS dependency
          ON dependency.refclassid = 'pg_extension'::regclass
         AND dependency.refobjid = extension_catalog.extension_oid
         AND dependency.classid = 'pg_proc'::regclass
         AND dependency.deptype = 'e'
        JOIN pg_proc AS installed_procedure
          ON installed_procedure.oid = dependency.objid
        JOIN pg_language AS language
          ON language.oid = installed_procedure.prolang
        WHERE language.lanname = 'c'
    ),
    catalog_contract AS (
        SELECT jsonb_build_object(
            'installed_version', extension_catalog.installed_version,
            'available_version', extension_catalog.available_version,
            'extension_schema', extension_catalog.extension_schema,
            'versions_match',
                extension_catalog.installed_version =
                    extension_catalog.available_version,
            'c_module_paths', extension_c_modules.paths,
            'c_module_path_current', extension_c_modules.current,
            'required_functions', (
                SELECT jsonb_object_agg(
                    resolved.signature,
                    resolved.procedure_oid IS NOT NULL
                    AND EXISTS (
                        SELECT 1
                        FROM pg_depend AS dependency
                        WHERE dependency.classid = 'pg_proc'::regclass
                          AND dependency.objid = resolved.procedure_oid
                          AND dependency.refclassid =
                              'pg_extension'::regclass
                          AND dependency.refobjid =
                              resolved.extension_oid
                          AND dependency.deptype = 'e'
                    )
                    ORDER BY resolved.signature
                )
                FROM resolved_catalog_function AS resolved
            )
        ) AS value
        FROM extension_catalog
        CROSS JOIN extension_c_modules
    ),
    settings AS (
        SELECT jsonb_object_agg(name, setting ORDER BY name) AS value
        FROM pg_settings
        WHERE name IN (
            'autovacuum',
            'default_statistics_target',
            'evoke.shared_runtime_size',
            'evoke.maintenance_worker_limit',
            'evoke.test_disable_semantic_accelerator',
            'evoke.test_filtered_forward_route',
            'shared_preload_libraries',
            'track_counts'
        )
    ),
    table_stats AS (
        SELECT jsonb_object_agg(
            stats.relname,
            jsonb_build_object(
                'reltuples', relation.reltuples,
                'n_live_tup', stats.n_live_tup,
                'last_analyze', stats.last_analyze,
                'last_autoanalyze', stats.last_autoanalyze
            )
            ORDER BY stats.relname
        ) AS value
        FROM pg_stat_user_tables AS stats
        JOIN pg_class AS relation ON relation.oid = stats.relid
        WHERE stats.schemaname = 'commons'
          AND stats.relname IN (
              'data_arxiv',
              'data_pubmed',
              'data_policy_ca',
              'data_policy_ca_chunks',
              'data_policy_tx',
              'data_policy_tx_chunks',
              'data_policy_wa',
              'data_policy_wa_chunks',
              'sys_chunks'
          )
    ),
    filter_stats AS (
        SELECT jsonb_object_agg(
            filter_columns.tablename,
            filter_columns.columns
            ORDER BY filter_columns.tablename
        ) AS value
        FROM (
            SELECT
                tablename,
                jsonb_agg(attname ORDER BY attname) AS columns
            FROM pg_stats
            WHERE schemaname = 'commons'
              AND tablename IN (
                  'data_arxiv',
                  'data_pubmed',
                  'data_policy_ca',
                  'data_policy_ca_chunks',
                  'data_policy_tx',
                  'data_policy_tx_chunks',
                  'data_policy_wa',
                  'data_policy_wa_chunks',
                  'sys_chunks'
              )
              AND attname IN (
                  'categories',
                  'document_id',
                  'journal_title',
                  'nlm_ta',
                  'organizations',
                  'policy_ca_doc_id',
                  'policy_tx_doc_id',
                  'policy_wa_doc_id',
                  'publish_date',
                  'publish_date_end_bound',
                  'publish_date_has_day',
                  'publish_date_start_bound'
              )
            GROUP BY tablename
        ) AS filter_columns
    ),
    generated_columns AS (
        SELECT jsonb_object_agg(
            attribute.attname,
            jsonb_build_object(
                'type', format_type(
                    attribute.atttypid,
                    attribute.atttypmod
                ),
                'generated', attribute.attgenerated,
                'expression', pg_get_expr(
                    definition.adbin,
                    definition.adrelid
                )
            )
            ORDER BY attribute.attname
        ) AS value
        FROM pg_attribute AS attribute
        JOIN pg_class AS relation
          ON relation.oid = attribute.attrelid
        JOIN pg_namespace AS namespace
          ON namespace.oid = relation.relnamespace
        LEFT JOIN pg_attrdef AS definition
          ON definition.adrelid = attribute.attrelid
         AND definition.adnum = attribute.attnum
        WHERE namespace.nspname = 'commons'
          AND relation.relname = 'data_pubmed'
          AND attribute.attname IN (
              'publish_date_start_bound',
              'publish_date_end_bound',
              'publish_date_has_day'
          )
          AND attribute.attnum > 0
          AND NOT attribute.attisdropped
    )
    SELECT json_build_object(
        'database', current_database(),
        'server_version', current_setting('server_version'),
        'captured_at_epoch', extract(epoch FROM clock_timestamp()),
        'postmaster_start_epoch',
            extract(epoch FROM pg_postmaster_start_time()),
        'postmaster_age_seconds',
            extract(epoch FROM clock_timestamp() - pg_postmaster_start_time()),
        'extension_version', (
            SELECT extversion FROM pg_extension WHERE extname = 'evoke'
        ),
        'catalog_contract', (SELECT value FROM catalog_contract),
        'pubmed_index', {sql_literal(pubmed_index)},
        'pubmed_index_source', {sql_literal(pubmed_source)},
        'settings', (SELECT value FROM settings),
        'table_stats', (SELECT value FROM table_stats),
        'filter_stats', (SELECT value FROM filter_stats),
        'generated_columns', (SELECT value FROM generated_columns)
    );
    """
    environment = query_json(args, sql_text)
    contract = environment.get('catalog_contract')
    if isinstance(contract, dict):
        contract['catalog_identity'] = collect_catalog_identity(
            args,
            contract,
        )
    environment['catalog_qualified'] = catalog_contract_qualified(
        contract
    )
    if environment['catalog_qualified']:
        environment['onnxruntime_contract'] = collect_onnxruntime_contract(
            args,
            environment['catalog_contract'],
        )
    else:
        environment['onnxruntime_contract'] = {
            'qualified': False,
            'error': 'catalog contract is not qualified',
        }
    environment['onnxruntime_qualified'] = environment[
        'onnxruntime_contract'
    ].get('qualified') is True
    environment['indexes'] = {}
    for index_name in indexes:
        try:
            environment['indexes'][index_name] = collect_index_status(
                args,
                index_name,
            )
        except RuntimeError as exc:
            environment['indexes'][index_name] = {'error': str(exc)}
    environment['schema_health'] = evaluate_schema_health(environment)
    environment['schema_qualified'] = environment[
        'schema_health'
    ]['qualified']
    environment['generation_qualified'] = (
        environment['catalog_qualified']
        and all(
            index_status_qualified(status)
            for status in environment['indexes'].values()
        )
    )
    environment['query_metadata_qualified'] = (
        environment['catalog_qualified']
        and all(
            index_query_metadata_qualified(status)
            for status in environment['indexes'].values()
        )
    )
    environment['query_warm_qualified'] = (
        environment['catalog_qualified']
        and all(
            index_query_warm_state(status) in ('resident', 'not_requested')
            for status in environment['indexes'].values()
        )
    )
    environment['planner_health'] = evaluate_planner_health(environment)
    environment['planner_qualified'] = environment[
        'planner_health'
    ]['qualified']
    environment['runtime_health'] = evaluate_runtime_health(environment)
    environment['runtime_qualified'] = environment[
        'runtime_health'
    ]['qualified']
    restart_limit = getattr(args, 'require_restart_within_seconds', None)
    postmaster_age = environment.get('postmaster_age_seconds')
    environment['restart_requirement_seconds'] = restart_limit
    environment['restart_qualified'] = bool(
        restart_limit is not None
        and isinstance(postmaster_age, (int, float))
        and 0.0 <= float(postmaster_age) <= float(restart_limit)
    )
    environment['qualified'] = (
        environment['schema_qualified']
        and environment['onnxruntime_qualified']
        and environment['generation_qualified']
        and environment['query_metadata_qualified']
        and environment['planner_qualified']
        and environment['runtime_qualified']
        and (
            restart_limit is None
            or environment['restart_qualified']
        )
    )
    return environment


def catalog_contract_qualified(contract: Any) -> bool:
    if not isinstance(contract, dict):
        return False
    required_functions = contract.get('required_functions')
    return (
        contract.get('versions_match') is True
        and isinstance(contract.get('extension_schema'), str)
        and SIMPLE_IDENTIFIER_RE.fullmatch(
            contract['extension_schema']
        ) is not None
        and contract.get('c_module_paths') == ['$libdir/evoke']
        and contract.get('c_module_path_current') is True
        and contract.get('catalog_identity') == CATALOG_CONTRACT
        and isinstance(required_functions, dict)
        and all(
            required_functions.get(signature) is True
            for signature in REQUIRED_CATALOG_SIGNATURES
        )
    )


def collect_catalog_identity(
    args: argparse.Namespace,
    catalog_contract: dict[str, Any],
) -> str | None:
    schema = catalog_contract.get('extension_schema')
    if not isinstance(schema, str) or (
        SIMPLE_IDENTIFIER_RE.fullmatch(schema) is None
    ):
        return None
    function = (
        f'{quote_identifier(schema)}.'
        '"evoke_catalog_contract_internal"'
    )
    try:
        value = query_json(
            args,
            f'SELECT json_build_object(\'value\', {function}());',
        )
    except RuntimeError:
        return None
    if not isinstance(value, dict):
        return None
    result = value.get('value')
    return result if isinstance(result, str) else None


def collect_onnxruntime_contract(
    args: argparse.Namespace,
    catalog_contract: dict[str, Any],
) -> dict[str, Any]:
    schema = catalog_contract.get('extension_schema')
    if not isinstance(schema, str):
        return {
            'qualified': False,
            'error': 'extension schema is unavailable',
        }
    qualified_schema = quote_identifier(schema)
    build_info_fn = f'{qualified_schema}."evoke_onnxruntime_build_info"'
    probe_fn = f'{qualified_schema}."evoke_onnxruntime_probe"'
    try:
        observed = query_json(
            args,
            f"""
            SELECT json_build_object(
                'build_info', {build_info_fn}(),
                'probe', {probe_fn}()
            );
            """,
        )
    except RuntimeError as exc:
        return {
            'qualified': False,
            'error': str(exc),
        }
    expected_version = pinned_onnxruntime_version()
    expected_api = expected_onnxruntime_api(expected_version)
    expected_build_info = f'enabled:api={expected_api}'
    expected_probe_suffix = f':version={expected_version}'
    build_info = observed.get('build_info')
    probe = observed.get('probe')
    observed.update({
        'expected_version': expected_version,
        'expected_build_info': expected_build_info,
        'qualified': (
            build_info == expected_build_info
            and isinstance(probe, str)
            and probe.startswith('available:')
            and probe.endswith(expected_probe_suffix)
        ),
    })
    return observed


def normalized_expression(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r'\s+', '', value)


def evaluate_schema_health(environment: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    observed = environment.get('generated_columns')

    if not isinstance(observed, dict):
        reasons.append('PubMed generated-column contract is missing')
        return {'qualified': False, 'reasons': reasons}

    for column_name, expected in REQUIRED_GENERATED_COLUMNS.items():
        column = observed.get(column_name)
        if not isinstance(column, dict):
            reasons.append(f'{column_name} is missing')
            continue
        if column.get('type') != expected['type']:
            reasons.append(f'{column_name} has the wrong type')
        if column.get('generated') != 's':
            reasons.append(f'{column_name} is not stored generated')
        if normalized_expression(column.get('expression')) != (
            normalized_expression(expected['expression'])
        ):
            reasons.append(f'{column_name} has the wrong expression')

    return {'qualified': not reasons, 'reasons': reasons}


def evaluate_planner_health(environment: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    settings = environment.get('settings')
    table_stats = environment.get('table_stats')
    filter_stats = environment.get('filter_stats')

    if not isinstance(settings, dict):
        reasons.append('settings are missing')
    else:
        if settings.get('autovacuum') != 'on':
            reasons.append('autovacuum is not on')
        if settings.get('track_counts') != 'on':
            reasons.append('track_counts is not on')

    for table_name, required_columns in REQUIRED_FILTER_STATS.items():
        stats = (
            table_stats.get(table_name)
            if isinstance(table_stats, dict)
            else None
        )
        if not isinstance(stats, dict):
            reasons.append(f'{table_name} table statistics are missing')
        else:
            reltuples = stats.get('reltuples')
            if not isinstance(reltuples, (int, float)) or reltuples < 0:
                reasons.append(f'{table_name} cardinality is unknown')

        observed_columns = (
            set(filter_stats.get(table_name, []))
            if isinstance(filter_stats, dict)
            and isinstance(filter_stats.get(table_name), list)
            else set()
        )
        missing_columns = sorted(required_columns - observed_columns)
        if missing_columns:
            reasons.append(
                f'{table_name} missing pg_stats columns: '
                f'{", ".join(missing_columns)}'
            )

    return {
        'qualified': not reasons,
        'reasons': reasons,
    }


def evaluate_runtime_health(environment: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    settings = environment.get('settings')

    if not isinstance(settings, dict):
        reasons.append('settings are missing')
        return {'qualified': False, 'reasons': reasons}

    worker_limit = settings.get('evoke.maintenance_worker_limit')
    try:
        workers_enabled = int(worker_limit) > 0
    except (TypeError, ValueError):
        workers_enabled = False
    if not workers_enabled:
        reasons.append('Evoke maintenance workers are disabled')

    accelerator_override = settings.get(
        'evoke.test_disable_semantic_accelerator'
    )
    if accelerator_override is None:
        reasons.append('semantic accelerator test control is unavailable')
    elif accelerator_override != 'off':
        reasons.append('semantic accelerator test override is active')
    forward_route = settings.get('evoke.test_filtered_forward_route')
    if forward_route is None:
        reasons.append('filtered forward-route control is unavailable')
    elif forward_route != 'auto':
        reasons.append('filtered forward-route test override is active')

    return {
        'qualified': not reasons,
        'reasons': reasons,
    }


def summarize_plan_nodes(plan: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        summary = {
            key: node[key]
            for key in (
                'Node Type',
                'Relation Name',
                'Alias',
                'Index Name',
                'Plan Rows',
                'Total Cost',
                'Index Cond',
                'Recheck Cond',
                'Filter',
            )
            if key in node
        }
        nodes.append(summary)
        for child in node.get('Plans', []):
            visit(child)

    visit(plan)
    return nodes


def collect_planner_plans(
    args: argparse.Namespace,
    cases: list[BenchmarkCase],
) -> dict[str, Any]:
    plans: dict[str, Any] = {}
    for case in cases:
        if case.planner_sql is None:
            continue
        result = run_psql(
            args,
            case.planner_sql,
            timeout_seconds=15.0,
        )
        if result.returncode != 0:
            plans[case.name] = {
                'error': result.stderr.strip() or result.stdout.strip(),
            }
            continue
        try:
            document = json.loads(result.stdout.strip())
            root = document[0]['Plan']
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            plans[case.name] = {'error': f'invalid EXPLAIN JSON: {exc}'}
            continue
        plans[case.name] = {
            'nodes': summarize_plan_nodes(root),
            'raw': document,
        }
    return {
        'captured': all('error' not in plan for plan in plans.values()),
        'cases': plans,
    }


def collect_index_status(
    args: argparse.Namespace,
    index_name: str,
) -> dict[str, Any]:
    sql_text = f"""
    WITH source AS (
        SELECT evoke_index_generation_status_internal(
            {regclass_literal(index_name)}
        )::jsonb AS status
    ),
    runtime_source AS (
        SELECT evoke_index_runtime_state_json(
            {regclass_literal(index_name)}
        )::jsonb AS runtime
    )
    SELECT json_build_object(
        'generation_id', status ->> 'generation_id',
        'health', status ->> 'health',
        'valid', status -> 'valid',
        'runtime_contract_matches', status -> 'runtime_contract_matches',
        'accelerator_state', status #>> '{{semantic_accelerator,state}}',
        'forward_complete',
            status #> '{{semantic_accelerator,forward_complete}}',
        'builder_policy_id',
            status #> '{{semantic_accelerator,builder_policy_id}}',
        'scope_present',
            status #> '{{semantic_accelerator,scope_present}}',
        'scope_version',
            status #> '{{semantic_accelerator,scope_version}}',
        'scope_current',
            status #> '{{semantic_accelerator,scope_current}}',
        'auto_preload_priority',
            runtime #> '{{generation,auto_preload_priority}}',
        'shared_preload_available',
            runtime #> '{{shared_preload,available}}',
        'shared_preload_resident',
            runtime #> '{{shared_preload,resident}}',
        'shared_preload_loading',
            runtime #> '{{shared_preload,loading}}',
        'resident_fold_current',
            runtime #> '{{shared_preload,resident_fold_current}}',
        'resident_fold_loading',
            runtime #> '{{shared_preload,resident_fold_loading}}',
        'document_tid_lookup_entries',
            runtime #> '{{shared_preload,document_tid_lookup_entries}}',
        'query_warm_marker_valid',
            runtime #> '{{shared_preload,query_warm_marker_valid}}',
        'query_metadata_warm',
            runtime #> '{{shared_preload,query_metadata_warm}}',
        'query_warm_pages',
            runtime #> '{{shared_preload,query_warm_pages}}',
        'warm_expected',
            COALESCE(
                (runtime #>>
                    '{{generation,auto_preload_priority}}')::int4,
                0
            ) > 0,
        'query_warm',
            COALESCE(
                (runtime #>>
                    '{{shared_preload,resident_fold_current}}')::boolean,
                false
            ) OR COALESCE(
                (runtime #>>
                    '{{shared_preload,query_metadata_warm}}')::boolean,
                false
            )
    )
    FROM source
    CROSS JOIN runtime_source;
    """
    result = query_json(args, sql_text)
    if not isinstance(result, dict):
        raise RuntimeError(f'invalid status result for {index_name}')
    result['query_warm_state'] = index_query_warm_state(result)
    return result


def index_status_qualified(status: dict[str, Any]) -> bool:
    accelerator_state = status.get('accelerator_state')
    scope_present = status.get('scope_present')
    scope_version = int(status.get('scope_version', 0) or 0)

    return (
        'error' not in status
        and status.get('valid') is True
        and status.get('health') == 'ok'
        and status.get('runtime_contract_matches') is True
        and accelerator_state in {'ready', 'ready_baseline_delta'}
        and status.get('forward_complete') is True
        and (scope_present is False or scope_version > 0)
    )


def index_query_warm_state(status: dict[str, Any]) -> str:
    if 'error' in status:
        return 'error'
    if status.get('warm_expected') is not True:
        return 'not_requested'
    if status.get('resident_fold_current') is True:
        return 'resident'
    if (status.get('resident_fold_loading') is True or
            status.get('shared_preload_loading') is True):
        return 'loading'
    if status.get('query_metadata_warm') is True:
        return 'metadata_only'
    if status.get('shared_preload_available') is not True:
        return 'unavailable'
    return 'cold'


def index_query_metadata_qualified(status: dict[str, Any]) -> bool:
    return index_query_warm_state(status) in (
        'resident',
        'metadata_only',
        'not_requested',
    )


def active_sessions(args: argparse.Namespace) -> list[dict[str, Any]]:
    return query_json(
        args,
        f"""
        SELECT COALESCE(json_agg(row_to_json(active)), '[]'::json)
        FROM (
            SELECT
                pid,
                application_name,
                state,
                wait_event_type,
                wait_event,
                left(query, 160) AS query
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND pid <> pg_backend_pid()
              AND state <> 'idle'
              AND application_name = {sql_literal(APPLICATION_NAME)}
            ORDER BY query_start
        ) AS active;
        """,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    plans_only = bool(getattr(args, 'plans_only', False))

    pubmed_index, pubmed_source = resolve_pubmed_index(args)
    cases = build_cases(pubmed_index, args.ca_stress_document_id)
    if args.only:
        selected = set(args.only)
        known = {case.name for case in cases}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f'unknown --only cases: {unknown}')
        cases = [case for case in cases if case.name in selected]

    environment = collect_environment(
        args,
        pubmed_index,
        pubmed_source,
    )
    planner_plans = collect_planner_plans(args, cases)
    environment['planner_plans'] = planner_plans['cases']
    environment['planner_capture_qualified'] = planner_plans['captured']
    environment['qualified'] = (
        environment.get('qualified') is True and planner_plans['captured']
    )

    output: dict[str, Any] = {
        'suite': 'commons_query_matrix',
        'contract_version': MATRIX_CONTRACT_VERSION,
        'mode': 'plans_only' if plans_only else 'qualification',
        'started_at_epoch': time.time(),
        'concurrency': args.concurrency,
        'repeats': args.repeats,
        'environment': environment,
        'cases': [],
    }
    if (
        not plans_only
        and output['environment'].get('qualified') is True
    ):
        for case in cases:
            summary = run_case(args, case)
            output['cases'].append(summary)
            warm = summary.get('warm', {})
            print(
                f"{case.name}: passed={summary['passed']} "
                f"p50_ms={warm.get('p50_ms')} "
                f"violations={summary['violations']}"
            )
    elif not plans_only:
        print('qualification environment is incomplete; skipped query cases')
    output['active_sessions_after'] = active_sessions(args)
    output['finished_at_epoch'] = time.time()
    output['capture_passed'] = bool(
        output['environment'].get('catalog_qualified') is True
        and output['environment'].get('planner_capture_qualified') is True
        and not output['active_sessions_after']
    )
    output['passed'] = bool(
        not plans_only
        and output['cases']
        and all(case['passed'] for case in output['cases'])
        and output['environment'].get('qualified') is True
        and not output['active_sessions_after']
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(args.output)
    print(f'wrote {args.output}')
    if plans_only:
        return 0 if output['capture_passed'] else 1
    return 0 if output['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
