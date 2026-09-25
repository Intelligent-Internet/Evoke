#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import statistics
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import psycopg
from psycopg import conninfo, sql

DEFAULT_DATASETS_DIR = (
    Path(tempfile.gettempdir()) / 'evoke_dataset_cache/beir_official'
)
TOKEN_RE = re.compile(r'[A-Za-z0-9]+')
STOPWORDS = frozenset({
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'in',
    'is',
    'it',
    'of',
    'on',
    'or',
    'that',
    'the',
    'their',
    'this',
    'to',
    'was',
    'were',
    'with',
})


@dataclass
class TokenizedCorpus:
    ids: list[list[int]]
    vocab: dict[str, int]


def benchmark_admin_dsn() -> str:
    base_dsn = os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres')
    params = conninfo.conninfo_to_dict(base_dsn)
    if not params.get('user'):
        params['user'] = getpass.getuser()
    if not params.get('dbname'):
        params['dbname'] = 'postgres'
    return conninfo.make_conninfo(**params)


PG_DSN = benchmark_admin_dsn()


def drop_database_if_exists(db_name: str) -> None:
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT EXISTS ('
                'SELECT 1 FROM pg_database WHERE datname = %s'
                ')',
                (db_name,),
            )
            if not cur.fetchone()[0]:
                return
            cur.execute(
                sql.SQL('DROP DATABASE {} WITH (FORCE)').format(
                    sql.Identifier(db_name)
                )
            )


def combine_doc_text(row: dict[str, Any]) -> str:
    title = (row.get('title') or '').strip()
    text = (row.get('text') or '').strip()
    if title and text:
        return f'{title}\n{text}'
    return title or text


def read_jsonl_map(
    path: Path,
    value_fn: Callable[[dict[str, Any]], str],
) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get('_id') or row.get('id')
            if key is None:
                raise SystemExit(f'missing id field in {path}')
            data[str(key)] = value_fn(row)
    return data


def load_dataset(
    dataset: str,
    datasets_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    data_path = datasets_dir / dataset
    corpus_path = data_path / 'corpus.jsonl'
    queries_path = data_path / 'queries.jsonl'
    if not corpus_path.exists() or not queries_path.exists():
        raise SystemExit(
            f'dataset {dataset!r} is missing corpus.jsonl or queries.jsonl '
            f'under {data_path}'
        )

    corpus = read_jsonl_map(corpus_path, combine_doc_text)
    queries = read_jsonl_map(queries_path, lambda row: str(row['text']))
    return list(corpus.keys()), list(corpus.values()), list(queries.values())


def normalize_tokens(text: str) -> list[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def tokenize_dataset(
    corpus_texts: list[str],
    query_texts: list[str],
) -> tuple[TokenizedCorpus, list[list[int]], list[str]]:
    vocab: dict[str, int] = {}
    corpus_ids: list[list[int]] = []
    for text in corpus_texts:
        doc_ids = []
        for token in normalize_tokens(text):
            token_id = vocab.setdefault(token, len(vocab))
            doc_ids.append(token_id)
        corpus_ids.append(doc_ids)

    query_ids: list[list[int]] = []
    for text in query_texts:
        ids = []
        for token in normalize_tokens(text):
            token_id = vocab.get(token)
            if token_id is not None:
                ids.append(token_id)
        query_ids.append(ids)

    vocab_by_id = [''] * len(vocab)
    for token, token_id in vocab.items():
        vocab_by_id[token_id] = token
    return TokenizedCorpus(ids=corpus_ids, vocab=vocab), query_ids, vocab_by_id


def benchmark_db_name(dataset: str) -> str:
    suffix = dataset.replace('-', '_')
    return f'evoke_profile_filtered_ordered_must_{suffix}'


def decode_ids(vocab_by_id: list[str], token_ids: list[int]) -> list[str]:
    return [vocab_by_id[token_id] for token_id in token_ids]


def summarize(latencies_ms: list[float]) -> dict[str, float]:
    total_ms = sum(latencies_ms)
    count = len(latencies_ms)
    qps = (1000.0 * count / total_ms) if total_ms > 0 else 0.0
    return {
        'count': count,
        'total_ms': total_ms,
        'avg_ms': statistics.mean(latencies_ms),
        'median_ms': statistics.median(latencies_ms),
        'p95_ms': (
            statistics.quantiles(latencies_ms, n=20)[18]
            if count >= 20 else max(latencies_ms)
        ),
        'qps': qps,
    }


def pick_cases(
    corpus_ids: list[list[int]],
    query_ids: list[list[int]],
    vocab_by_id: list[str],
    max_cases: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    df = Counter()
    for doc_ids in corpus_ids:
        for token_id in set(doc_ids):
            df[token_id] += 1

    num_docs = len(corpus_ids)
    max_candidate = max(1, num_docs // 4)
    cases = []
    candidate_counts = []
    for token_query_ids in query_ids:
        unique_ids = []
        seen = set()
        for token_id in token_query_ids:
            if token_id in seen:
                continue
            seen.add(token_id)
            unique_ids.append(token_id)
        if len(unique_ids) < 2:
            continue
        eligible = [
            token_id
            for token_id in unique_ids
            if 0 < df[token_id] <= max_candidate
        ]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda token_id: (df[token_id], token_id))
        optional = next(
            token_id for token_id in unique_ids if token_id != chosen
        )
        filter_query = f'+{vocab_by_id[chosen]} {vocab_by_id[optional]}'
        cases.append({
            'filter_query': filter_query,
            'order_query_tokens': decode_ids(vocab_by_id, token_query_ids),
            'candidate_count': df[chosen],
        })
        candidate_counts.append(df[chosen])
        if len(cases) >= max_cases:
            break

    return cases, {
        'eligible_case_count': len(cases),
        'candidate_count_min': min(candidate_counts) if candidate_counts else 0,
        'candidate_count_median': (
            statistics.median(candidate_counts) if candidate_counts else 0
        ),
        'candidate_count_max': max(candidate_counts) if candidate_counts else 0,
    }


def fetch_plan(
    cur: psycopg.Cursor,
    filter_query: str,
    order_query_tokens: list[str],
    top_k: int,
) -> list[str]:
    sql_text = """
        SELECT id
        FROM bench.docs_tokens
        WHERE tokens @@ %s
        ORDER BY tokens <=> %s::text[] ASC
        LIMIT %s
    """
    cur.execute(
        f'EXPLAIN (COSTS OFF) {sql_text}',
        (filter_query, order_query_tokens, top_k),
    )
    return [row[0] for row in cur.fetchall()]


def write_state(state_file: Path, payload: dict[str, object]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def read_state(state_file: Path) -> dict[str, object]:
    return json.loads(state_file.read_text(encoding='utf-8'))


def wait_for_index_ready(
    db_name: str,
    expected_bytes: int,
    timeout_s: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='evoke_filtered_ordered_must_ready',
    )

    while time.monotonic() < deadline:
        try:
            with psycopg.connect(db_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_relation_size("
                        "'bench.docs_tokens_bm25_idx'::regclass)"
                    )
                    relation_bytes = int(cur.fetchone()[0])
                    cur.execute(
                        "SELECT 1 FROM evoke_index_details("
                        "'bench.docs_tokens_bm25_idx'::regclass)"
                    )
                    cur.fetchone()
            if relation_bytes == expected_bytes and relation_bytes > 0:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)

    detail = f'last error: {last_error}' if last_error is not None else ''
    raise RuntimeError(
        'evoke benchmark index was not visible to a fresh backend '
        f'within {timeout_s:.1f}s; expected {expected_bytes} bytes. {detail}'
    )


def prepare_state(
    repo_root: Path,
    dataset: str,
    datasets_dir: Path | None,
    top_k: int,
    max_cases: int,
    state_file: Path,
) -> dict[str, object]:
    if datasets_dir is None:
        datasets_dir = DEFAULT_DATASETS_DIR

    _, corpus_texts, query_texts = load_dataset(dataset, datasets_dir)
    corpus_tokenized, query_ids, vocab_by_id = tokenize_dataset(
        corpus_texts,
        query_texts,
    )
    cases, case_stats = pick_cases(
        corpus_tokenized.ids,
        query_ids,
        vocab_by_id,
        max_cases,
    )
    if not cases:
        raise SystemExit(
            f'No eligible must+should filtered ordered cases found for '
            f'{dataset}'
        )

    db_name = benchmark_db_name(dataset)
    drop_database_if_exists(db_name)
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL('CREATE DATABASE {}').format(sql.Identifier(db_name))
            )

    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='evoke_filtered_ordered_must_prepare',
    )
    try:
        with psycopg.connect(db_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION evoke')
                cur.execute('CREATE SCHEMA bench')
                cur.execute(
                    'CREATE TABLE bench.docs_tokens ('
                    'id integer PRIMARY KEY, '
                    'tokens text[] NOT NULL)'
                )
                with cur.copy(
                    'COPY bench.docs_tokens (id, tokens) FROM STDIN'
                ) as copy:
                    for doc_id, token_ids in enumerate(
                        corpus_tokenized.ids,
                        start=1,
                    ):
                        copy.write_row((
                            doc_id,
                            decode_ids(vocab_by_id, token_ids),
                        ))

                started = time.perf_counter()
                cur.execute(
                    """
                    CREATE INDEX docs_tokens_bm25_idx
                    ON bench.docs_tokens USING evoke (tokens)
                    WITH (
                        method = 'lucene',
                        idf_method = 'lucene',
                        k1 = 1.5,
                        b = 0.75,
                        delta = 0.5
                    )
                    """
                )
                build_ms = (time.perf_counter() - started) * 1000.0
                cur.execute(
                    "SELECT pg_relation_size('bench.docs_tokens_bm25_idx'"
                    '::regclass)'
                )
                build_bytes = cur.fetchone()[0]
                wait_for_index_ready(db_name, int(build_bytes))
                cur.execute('SET enable_seqscan = off')
                cur.execute('SET enable_bitmapscan = off')
                plan = fetch_plan(
                    cur,
                    str(cases[0]['filter_query']),
                    list(cases[0]['order_query_tokens']),
                    top_k,
                )
                cur.execute('RESET enable_bitmapscan')
                cur.execute('RESET enable_seqscan')
    except Exception:
        drop_database_if_exists(db_name)
        raise

    state = {
        'dataset': dataset,
        'db_name': db_name,
        'top_k': top_k,
        'stats': {
            'documents': len(corpus_tokenized.ids),
            'queries': len(query_ids),
        },
        'build_ms': build_ms,
        'build_bytes': build_bytes,
        'case_stats': case_stats,
        'sample_plan': plan,
        'cases': cases,
    }
    write_state(state_file, state)
    return state


def run_query_loop(
    repo_root: Path,
    state_file: Path,
    application_name: str,
    repeats: int,
) -> dict[str, object]:
    state = read_state(state_file)
    db_name = str(state['db_name'])
    cases = list(state['cases'])
    top_k = int(state['top_k'])
    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name=application_name,
    )

    sql_text = """
        SELECT count(*), coalesce(min(id), 0)
        FROM (
            SELECT id
            FROM bench.docs_tokens
            WHERE tokens @@ %s
            ORDER BY tokens <=> %s::text[] ASC
            LIMIT %s
        ) hits
    """
    latencies_ms = []
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('SET enable_seqscan = off')
            cur.execute('SET enable_bitmapscan = off')
            for _ in range(repeats):
                for case in cases:
                    t0 = time.perf_counter()
                    cur.execute(
                        sql_text,
                        (
                            case['filter_query'],
                            case['order_query_tokens'],
                            top_k,
                        ),
                    )
                    cur.fetchone()
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            cur.execute('RESET enable_bitmapscan')
            cur.execute('RESET enable_seqscan')

    return {
        'dataset': state['dataset'],
        'stats': state['stats'],
        'build_ms': state['build_ms'],
        'build_bytes': state['build_bytes'],
        'case_stats': state['case_stats'],
        'sample_plan': state['sample_plan'],
        'query': summarize(latencies_ms),
    }


def cleanup_state(repo_root: Path, state_file: Path) -> None:
    state = read_state(state_file)
    drop_database_if_exists(str(state['db_name']))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--repo-root',
        type=Path,
        required=True,
    )
    parser.add_argument('--datasets-dir', type=Path)
    parser.add_argument('--dataset')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--max-cases', type=int, default=250)
    parser.add_argument(
        '--state-file',
        type=Path,
        required=True,
    )
    parser.add_argument(
        '--application-name',
        default='evoke_filtered_ordered_must_query_only',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--mode',
        choices=('prepare', 'query', 'cleanup'),
        required=True,
    )
    args = parser.parse_args()
    if args.mode == 'prepare' and not args.dataset:
        parser.error('--dataset is required in prepare mode')
    return args


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if args.mode == 'prepare':
        result = prepare_state(
            repo_root,
            args.dataset,
            args.datasets_dir,
            args.top_k,
            args.max_cases,
            args.state_file.resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.mode == 'query':
        result = run_query_loop(
            repo_root,
            args.state_file.resolve(),
            args.application_name,
            args.repeats,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    cleanup_state(repo_root, args.state_file.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
