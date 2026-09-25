#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import bm25s
import psycopg
from psycopg import conninfo, sql

try:
    import Stemmer
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        'This script requires optional benchmark dependencies. '
        'Install them in the benchmark environment with:\n'
        '  pip install PyStemmer tqdm\n'
        f'Import error: {exc}'
    ) from exc


BASE_URL = (
    'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'
)
OFFICIAL_QPS = {
    'arguana': 573.91,
    'climate-fever': 13.09,
    'cqadupstack': 170.91,
    'dbpedia-entity': 13.44,
    'fever': 20.19,
    'fiqa': 717.78,
    'hotpotqa': 20.88,
    'msmarco': 12.20,
    'nfcorpus': 1196.16,
    'nq': 41.85,
    'quora': 272.04,
    'scidocs': 767.05,
    'scifact': 1317.12,
    'trec-covid': 85.64,
    'webis-touche2020': 60.59,
}
OFFICIAL_ORDER = [
    'arguana',
    'climate-fever',
    'cqadupstack',
    'dbpedia-entity',
    'fever',
    'fiqa',
    'hotpotqa',
    'msmarco',
    'nfcorpus',
    'nq',
    'quora',
    'scidocs',
    'scifact',
    'trec-covid',
    'webis-touche2020',
]
TOP_K = 1000
DEFAULT_RESULTS_DIR = Path('benchmarks')
DEFAULT_DATASETS_DIR = Path('/tmp/evoke_beir')
DB_PREFIX = 'evoke_official_'
ARIA2C = shutil.which('aria2c')
BOOTSTRAP_SQL = os.environ.get('EVOKE_BENCH_BOOTSTRAP_SQL')
MODULE_PATH = os.environ.get('EVOKE_BENCH_MODULE_PATH')


@dataclass
class QueryStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    qps: float
    count: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_latencies(latencies_ms: list[float]) -> QueryStats:
    ordered = sorted(latencies_ms)
    avg_ms = statistics.fmean(latencies_ms)
    return QueryStats(
        avg_ms=avg_ms,
        p50_ms=percentile(ordered, 0.50),
        p95_ms=percentile(ordered, 0.95),
        qps=1000.0 / avg_ms if avg_ms > 0 else math.nan,
        count=len(latencies_ms),
    )


def benchmark_admin_dsn() -> str:
    base_dsn = os.environ.get('EVOKE_BENCH_DSN', 'dbname=postgres')
    params = conninfo.conninfo_to_dict(base_dsn)
    if not params.get('user'):
        params['user'] = 'postgres'
    if not params.get('dbname'):
        params['dbname'] = 'postgres'
    return conninfo.make_conninfo(**params)


PG_DSN = benchmark_admin_dsn()


def combine_doc_text(row: dict[str, Any]) -> str:
    title = (row.get('title') or '').strip()
    text = (row.get('text') or '').strip()
    if title and text:
        return f'{title}\n{text}'
    return title or text


def ensure_database(db_name: str) -> None:
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                'SELECT EXISTS ('
                'SELECT 1 FROM pg_database WHERE datname = %s'
                ')',
                (db_name,),
            )
            if cur.fetchone()[0]:
                return
            cur.execute(
                sql.SQL('CREATE DATABASE {}').format(
                    sql.Identifier(db_name)
                )
            )


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


def benchmark_db_names(dataset: str, path_mode: str) -> list[str]:
    base_name = f'{DB_PREFIX}{dataset.replace("-", "_")}'
    names: list[str] = []
    if path_mode in ('ids', 'both'):
        names.append(base_name)
    if path_mode in ('text', 'both'):
        names.append(f'{base_name}_text')
    return names


def reset_benchmark_databases(dataset: str, path_mode: str) -> None:
    for db_name in benchmark_db_names(dataset, path_mode):
        drop_database_if_exists(db_name)


def prepare_benchmark_schema(cur: psycopg.Cursor[Any], reuse_db: bool) -> None:
    if not reuse_db:
        cur.execute('DROP SCHEMA IF EXISTS bench CASCADE')
        cur.execute('CREATE SCHEMA bench')
        return
    cur.execute('CREATE SCHEMA IF NOT EXISTS bench')


def extension_is_current(cur: psycopg.Cursor[Any]) -> bool:
    cur.execute(
        """
        SELECT
            EXISTS (
                SELECT 1
                FROM pg_proc
                WHERE proname = 'evoke_query_ids'
            )
            AND EXISTS (
                SELECT 1
                FROM pg_proc
                WHERE proname = 'evoke_query_tokens'
            )
        """
    )
    return bool(cur.fetchone()[0])


def install_extension_via_sql(cur: psycopg.Cursor[Any]) -> None:
    if not BOOTSTRAP_SQL or not MODULE_PATH:
        raise RuntimeError(
            'Manual extension bootstrap requires both '
            'EVOKE_BENCH_BOOTSTRAP_SQL and '
            'EVOKE_BENCH_MODULE_PATH'
        )
    sql_path = pathlib.Path(BOOTSTRAP_SQL)
    sql_text = sql_path.read_text(encoding='utf-8')
    sql_text = sql_text.replace('MODULE_PATHNAME', MODULE_PATH)
    started_tx = False
    if cur.connection.autocommit:
        cur.execute('BEGIN')
        started_tx = True
    try:
        # The extension SQL can define SQL functions that reference helper
        # functions created later in the same file. In manual bootstrap mode,
        # disable function-body validation for this transaction only.
        cur.execute('SET LOCAL check_function_bodies = off')
        cur.execute(sql_text)
        if started_tx:
            cur.execute('COMMIT')
    except Exception:
        if started_tx:
            cur.execute('ROLLBACK')
        raise


def ensure_extension(cur: psycopg.Cursor[Any], reuse_db: bool) -> None:
    if BOOTSTRAP_SQL or MODULE_PATH:
        if not reuse_db:
            install_extension_via_sql(cur)
            return
        if extension_is_current(cur):
            return
        install_extension_via_sql(cur)
        return
    if not reuse_db:
        cur.execute('DROP EXTENSION IF EXISTS evoke CASCADE')
        cur.execute('CREATE EXTENSION evoke')
        return
    cur.execute('CREATE EXTENSION IF NOT EXISTS evoke')
    if extension_is_current(cur):
        return
    cur.execute('DROP EXTENSION evoke CASCADE')
    cur.execute('CREATE EXTENSION evoke')


def count_lines(path: Path) -> int:
    with path.open('r', encoding='utf-8') as handle:
        return sum(1 for _ in handle)


def merge_cqadupstack(data_path: Path) -> None:
    def clean_metadata(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        tags = value.get('tags')
        if tags is None:
            return {}
        return {'tags': tags}

    def needs_query_rebuild(path: Path, expected_lines: int) -> bool:
        if not path.exists():
            return True
        if count_lines(path) != expected_lines:
            return True
        with path.open('r', encoding='utf-8') as handle:
            first = handle.readline()
        if not first:
            return True
        row = json.loads(first)
        metadata = row.get('metadata')
        return isinstance(metadata, dict) and any(
            key != 'tags' for key in metadata
        )

    def needs_corpus_rebuild(path: Path, expected_lines: int) -> bool:
        if not path.exists():
            return True
        if count_lines(path) != expected_lines:
            return True
        with path.open('r', encoding='utf-8') as handle:
            first = handle.readline()
        if not first:
            return True
        row = json.loads(first)
        metadata = row.get('metadata')
        return isinstance(metadata, dict) and any(
            key != 'tags' for key in metadata
        )

    corpus_path = data_path / 'corpus.jsonl'
    queries_path = data_path / 'queries.jsonl'
    qrels_path = data_path / 'qrels' / 'test.tsv'
    corpus_files = sorted(data_path.glob('*/corpus.jsonl'))
    query_files = sorted(data_path.glob('*/queries.jsonl'))
    qrel_files = sorted(data_path.glob('*/qrels/test.tsv'))

    # Some environments stage CQADupStack as an already flattened bundle with
    # top-level corpus/queries/qrels files only. In that layout there are no
    # per-forum subdirectories to merge, so rebuilding from an empty glob would
    # truncate the dataset to zero rows.
    if (
        not corpus_files
        and not query_files
        and not qrel_files
        and corpus_path.exists()
        and queries_path.exists()
        and qrels_path.exists()
    ):
        return

    expected_corpus_lines = sum(count_lines(path) for path in corpus_files)
    if needs_corpus_rebuild(corpus_path, expected_corpus_lines):
        with corpus_path.open('w', encoding='utf-8') as handle:
            for file in corpus_files:
                corpus_name = file.parent.name
                with file.open('r', encoding='utf-8') as src:
                    for line in src:
                        row = json.loads(line)
                        row = {
                            '_id': f'{corpus_name}_{row["_id"]}',
                            'title': row.get('title', ''),
                            'text': row.get('text', ''),
                            'metadata': clean_metadata(row.get('metadata')),
                        }
                        handle.write(json.dumps(row))
                        handle.write('\n')
    expected_query_lines = sum(count_lines(path) for path in query_files)
    if needs_query_rebuild(queries_path, expected_query_lines):
        with queries_path.open('w', encoding='utf-8') as handle:
            for file in query_files:
                corpus_name = file.parent.name
                with file.open('r', encoding='utf-8') as src:
                    for line in src:
                        row = json.loads(line)
                        row = {
                            '_id': f'{corpus_name}_{row["_id"]}',
                            'text': row.get('text', ''),
                            'metadata': clean_metadata(row.get('metadata')),
                        }
                        handle.write(json.dumps(row))
                        handle.write('\n')
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    expected_qrel_rows = sum(count_lines(path) - 1 for path in qrel_files)
    if (
        not qrels_path.exists()
        or count_lines(qrels_path) - 1 != expected_qrel_rows
    ):
        with qrels_path.open('w', encoding='utf-8') as handle:
            handle.write('query-id\tcorpus-id\tscore\n')
            for file in qrel_files:
                corpus_name = file.parent.parent.name
                with file.open('r', encoding='utf-8') as src:
                    next(src)
                    for line in src:
                        qid, cid, score = line.rstrip('\n').split('\t')
                        handle.write(
                            f'{corpus_name}_{qid}\t{corpus_name}_{cid}\t{score}\n'
                        )


def read_jsonl_map(
    path: Path,
    value_fn: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            result[row['_id']] = value_fn(row)
    return result


def make_output_path(results_dir: Path) -> Path:
    stamp = time.strftime('%Y%m%d-%H%M%S')
    return results_dir / f'beir-official-{stamp}.json'


def load_existing_results(output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        return json.loads(output_path.read_text(encoding='utf-8'))

    return {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'official_source': (
            'https://github.com/xhluca/bm25-benchmarks '
            '(BM25S single-threaded table)'
        ),
        'top_k': TOP_K,
        'paths': [],
        'results': {},
    }


def persist_results(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )


def result_is_success(result: dict[str, Any]) -> bool:
    return 'error' not in result


def benchmark_upstream(
    corpus_ids: list[str],
    corpus_tokenized: Any,
    query_ids: list[list[int]],
    top_k: int,
) -> dict[str, Any]:
    retriever = bm25s.BM25(method='lucene', idf_method='lucene')

    started = time.perf_counter()
    retriever.index(corpus_tokenized, leave_progress=False)
    build_ms = (time.perf_counter() - started) * 1000.0

    query_count = len(query_ids)
    if query_count == 0:
        return {
            'build_ms': build_ms,
            'query': {
                'count': 0,
                'total_ms': 0.0,
                'qps': 0.0,
            },
        }

    batch_size = 512
    started = time.perf_counter()
    for start_idx in range(0, len(query_ids), batch_size):
        batch = query_ids[start_idx:start_idx + batch_size]
        retriever.retrieve(
            batch,
            corpus=corpus_ids,
            k=top_k,
            return_as='tuple',
            n_threads=1,
        )
    query_ms = (time.perf_counter() - started) * 1000.0
    query_ms = max(query_ms, 1e-9)
    qps = query_count / (query_ms / 1000.0)

    return {
        'build_ms': build_ms,
        'query': {
            'count': query_count,
            'total_ms': query_ms,
            'qps': qps,
        },
    }


def benchmark_postgres_ids(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    top_k: int,
    reuse_db: bool,
) -> dict[str, Any]:
    db_name = f'{DB_PREFIX}{dataset.replace("-", "_")}'
    ensure_database(db_name)
    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='evoke_bench',
    )
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_extension(cur, reuse_db)
            prepare_benchmark_schema(cur, reuse_db)
            cur.execute(
                'CREATE TABLE IF NOT EXISTS bench.docs_ids ('
                'id integer PRIMARY KEY, '
                'token_ids int4[] NOT NULL'
                ')'
            )

            cur.execute('SELECT count(*) FROM bench.docs_ids')
            loaded_rows = cur.fetchone()[0]
            if loaded_rows != len(corpus_id_tokens):
                cur.execute('TRUNCATE bench.docs_ids')
                with cur.copy(
                    'COPY bench.docs_ids (id, token_ids) FROM STDIN'
                ) as copy:
                    for doc_id, token_ids in enumerate(
                        corpus_id_tokens,
                        start=1,
                    ):
                        copy.write_row((doc_id, token_ids))

            cur.execute('DROP INDEX IF EXISTS bench.docs_ids_bm25_idx')

            started = time.perf_counter()
            cur.execute(
                """
                CREATE INDEX docs_ids_bm25_idx
                ON bench.docs_ids USING evoke (token_ids)
                WITH (
                    method = 'lucene',
                    idf_method = 'lucene',
                    k1 = 1.5,
                    b = 0.75,
                    delta = 0.5,
                    create_empty_token = true,
                    consistency = 'manual'
                )
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_ids_bm25_idx'::regclass)"
            )
            build_bytes = cur.fetchone()[0]

            latencies_ms: list[float] = []
            query_sql = """
                SELECT
                    count(*),
                    coalesce(min(doc_id), 0),
                    coalesce(max(score), 0::real)
                FROM public.evoke_query_ids(
                    'bench.docs_ids_bm25_idx'::regclass,
                    %s::int4[],
                    %s::int4,
                    NULL
                )
            """
            for query_ids in query_id_tokens:
                started = time.perf_counter()
                cur.execute(query_sql, (query_ids, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

            query = asdict(summarize_latencies(latencies_ms))
            return {
                'build_ms': build_ms,
                'build_bytes': build_bytes,
                'query': query,
            }


def maybe_benchmark_postgres_text(
    dataset: str,
    corpus_id_tokens: list[list[int]],
    query_id_tokens: list[list[int]],
    vocab_by_id: list[str],
    top_k: int,
    reuse_db: bool,
) -> dict[str, Any]:
    db_name = f'{DB_PREFIX}{dataset.replace("-", "_")}_text'

    def decode(ids: list[int]) -> list[str]:
        return [vocab_by_id[token_id] for token_id in ids]

    ensure_database(db_name)
    db_dsn = conninfo.make_conninfo(
        PG_DSN,
        dbname=db_name,
        application_name='evoke_bench',
    )
    with psycopg.connect(db_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            ensure_extension(cur, reuse_db)
            prepare_benchmark_schema(cur, reuse_db)
            cur.execute(
                'CREATE TABLE IF NOT EXISTS bench.docs_tokens ('
                'id integer PRIMARY KEY, '
                'tokens text[] NOT NULL'
                ')'
            )

            cur.execute('SELECT count(*) FROM bench.docs_tokens')
            loaded_rows = cur.fetchone()[0]
            if loaded_rows != len(corpus_id_tokens):
                cur.execute('TRUNCATE bench.docs_tokens')
                with cur.copy(
                    'COPY bench.docs_tokens (id, tokens) FROM STDIN'
                ) as copy:
                    for doc_id, token_ids in enumerate(
                        corpus_id_tokens,
                        start=1,
                    ):
                        copy.write_row((doc_id, decode(token_ids)))

            cur.execute('DROP INDEX IF EXISTS bench.docs_tokens_bm25_idx')

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
                    delta = 0.5,
                    consistency = 'manual'
                )
                """
            )
            build_ms = (time.perf_counter() - started) * 1000.0
            cur.execute(
                "SELECT pg_relation_size('bench.docs_tokens_bm25_idx'::regclass)"
            )
            build_bytes = cur.fetchone()[0]

            latencies_ms: list[float] = []
            query_sql = """
                SELECT
                    count(*),
                    coalesce(min(doc_id), 0),
                    coalesce(max(score), 0::real)
                FROM public.evoke_query_tokens(
                    'bench.docs_tokens_bm25_idx'::regclass,
                    %s::text[],
                    %s::int4,
                    NULL
                )
            """
            for query_ids in query_id_tokens:
                query_tokens = decode(query_ids)
                started = time.perf_counter()
                cur.execute(query_sql, (query_tokens, top_k))
                cur.fetchone()
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

            query = asdict(summarize_latencies(latencies_ms))
            return {
                'build_ms': build_ms,
                'build_bytes': build_bytes,
                'query': query,
            }


def load_dataset(
    dataset: str,
    datasets_dir: Path,
) -> tuple[list[str], list[str]]:
    data_path = download_and_unpack_dataset(dataset, datasets_dir)
    if dataset == 'cqadupstack':
        merge_cqadupstack(data_path)

    corpus = read_jsonl_map(
        data_path / 'corpus.jsonl',
        lambda row: combine_doc_text(row),
    )
    queries = read_jsonl_map(
        data_path / 'queries.jsonl',
        lambda row: row['text'],
    )
    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())
    query_texts = list(queries.values())
    return corpus_ids, corpus_texts, query_texts


def unzip_dataset(zip_path: Path, datasets_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(datasets_dir)


def download_dataset_with_stdlib(url: str, zip_path: Path) -> None:
    with urllib.request.urlopen(url) as response:
        with zip_path.open('wb') as handle:
            shutil.copyfileobj(response, handle)


def download_and_unpack_dataset(dataset: str, datasets_dir: Path) -> Path:
    url = BASE_URL.format(dataset)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    zip_path = datasets_dir / f'{dataset}.zip'
    data_path = datasets_dir / dataset

    if not zip_path.exists() and ARIA2C is None:
        download_dataset_with_stdlib(url, zip_path)
        unzip_dataset(zip_path, datasets_dir)
        return data_path

    if ARIA2C is not None and not data_path.exists():
        subprocess.run(
            [
                ARIA2C,
                '--allow-overwrite=false',
                '--auto-file-renaming=false',
                '--continue=true',
                '--file-allocation=none',
                '--max-connection-per-server=16',
                '--min-split-size=16M',
                '--split=16',
                '--summary-interval=5',
                '--dir',
                str(datasets_dir),
                '--out',
                zip_path.name,
                url,
            ],
            check=True,
        )
        unzip_dataset(zip_path, datasets_dir)
        return data_path

    if not data_path.exists():
        unzip_dataset(zip_path, datasets_dir)

    return data_path


def tokenize_dataset(
    corpus_texts: list[str],
    query_texts: list[str],
) -> tuple[Any, list[list[int]], list[str]]:
    tokenizer = bm25s.tokenization.Tokenizer(
        stopwords='en',
        stemmer=Stemmer.Stemmer('english'),
    )
    corpus_tokenized = tokenizer.tokenize(
        corpus_texts,
        update_vocab=True,
        return_as='tuple',
    )
    query_ids = tokenizer.tokenize(
        query_texts,
        update_vocab=False,
        return_as='ids',
    )

    vocab_by_id = [''] * len(corpus_tokenized.vocab)
    for token, token_id in corpus_tokenized.vocab.items():
        vocab_by_id[token_id] = token

    return corpus_tokenized, query_ids, vocab_by_id


def dataset_stats(
    corpus_tokenized: Any,
    query_ids: list[list[int]],
) -> dict[str, Any]:
    num_docs = len(corpus_tokenized.ids)
    num_tokens = sum(len(doc_ids) for doc_ids in corpus_tokenized.ids)
    num_queries = len(query_ids)
    num_query_tokens = sum(len(ids) for ids in query_ids)
    return {
        'documents': num_docs,
        'queries': num_queries,
        'tokens': num_tokens,
        'query_tokens': num_query_tokens,
        'vocab_size': len(corpus_tokenized.vocab),
    }


def run_dataset(
    dataset: str,
    path_mode: str,
    datasets_dir: Path,
    top_k: int,
    reuse_db: bool,
    skip_upstream: bool,
) -> dict[str, Any]:
    corpus_ids, corpus_texts, query_texts = load_dataset(dataset, datasets_dir)
    corpus_tokenized, query_ids, vocab_by_id = tokenize_dataset(
        corpus_texts,
        query_texts,
    )

    result = {
        'dataset': dataset,
        'official_qps': OFFICIAL_QPS[dataset],
        'stats': dataset_stats(corpus_tokenized, query_ids),
    }
    if path_mode in ('ids', 'both'):
        result['evoke_ids'] = benchmark_postgres_ids(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            top_k,
            reuse_db,
        )
    if not skip_upstream:
        result['upstream_bm25s'] = benchmark_upstream(
            corpus_ids,
            corpus_tokenized,
            query_ids,
            top_k,
        )
    if path_mode in ('text', 'both'):
        result['evoke_text'] = maybe_benchmark_postgres_text(
            dataset,
            corpus_tokenized.ids,
            query_ids,
            vocab_by_id,
            top_k,
            reuse_db,
        )

    del corpus_ids
    del corpus_texts
    del query_texts
    del corpus_tokenized
    del query_ids
    del vocab_by_id
    gc.collect()

    return result


def run_dataset_with_retries(
    dataset: str,
    path_mode: str,
    datasets_dir: Path,
    top_k: int,
    reuse_db: bool,
    skip_upstream: bool,
    retry_failures: int,
) -> dict[str, Any]:
    max_attempts = retry_failures + 1
    for attempt in range(1, max_attempts + 1):
        try:
            result = run_dataset(
                dataset,
                path_mode,
                datasets_dir,
                top_k,
                reuse_db,
                skip_upstream,
            )
            if attempt > 1:
                result['retry_attempts'] = attempt - 1
            return result
        except Exception:
            if attempt >= max_attempts:
                raise
            reset_benchmark_databases(dataset, path_mode)
            gc.collect()
            print(
                f'Retrying {dataset} after benchmark failure '
                f'(attempt {attempt + 1}/{max_attempts})',
                file=sys.stderr,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark official upstream BEIR datasets.'
    )
    parser.add_argument(
        '--datasets',
        nargs='*',
        default=OFFICIAL_ORDER,
        choices=OFFICIAL_ORDER,
        help='Datasets to benchmark. Defaults to the full official list.',
    )
    parser.add_argument(
        '--path-mode',
        default='ids',
        choices=['ids', 'text', 'both', 'none'],
        help=(
            'Benchmark ids only, text only, both ids and text, '
            'or none for upstream-only.'
        ),
    )
    parser.add_argument(
        '--datasets-dir',
        type=Path,
        default=DEFAULT_DATASETS_DIR,
        help='Directory used to cache BEIR dataset downloads.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='JSON file for progressive benchmark results.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=TOP_K,
        help='Top-k used for retrieval benchmarking.',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip datasets already present in the output file.',
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop immediately when a dataset benchmark fails.',
    )
    parser.add_argument(
        '--reuse-db',
        action='store_true',
        help='Reuse loaded benchmark databases when they already exist.',
    )
    parser.add_argument(
        '--skip-upstream',
        action='store_true',
        help='Skip the local upstream bm25s rerun and compare only to official QPS.',
    )
    parser.add_argument(
        '--retry-failures',
        type=int,
        default=0,
        help='Retry a failed dataset this many times after resetting its benchmark DBs.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output or make_output_path(DEFAULT_RESULTS_DIR)
    payload = load_existing_results(output_path)
    if args.path_mode == 'ids':
        payload['paths'] = ['ids']
    elif args.path_mode == 'text':
        payload['paths'] = ['text']
    elif args.path_mode == 'none':
        payload['paths'] = []
    else:
        payload['paths'] = ['ids', 'text']
    payload['top_k'] = args.top_k

    for dataset in args.datasets:
        if args.resume and dataset in payload['results']:
            if result_is_success(payload['results'][dataset]):
                print(
                    f'Skipping completed dataset: {dataset}',
                    file=sys.stderr,
                )
                continue
            print(
                f'Rerunning failed dataset: {dataset}',
                file=sys.stderr,
            )

        print(f'=== {dataset} ===', file=sys.stderr)
        started = time.perf_counter()
        try:
            payload['results'][dataset] = run_dataset_with_retries(
                dataset,
                args.path_mode,
                args.datasets_dir,
                args.top_k,
                args.reuse_db,
                args.skip_upstream,
                args.retry_failures,
            )
            payload['results'][dataset]['wall_time_s'] = (
                time.perf_counter() - started
            )
        except Exception as exc:  # pragma: no cover - benchmark failure path
            payload['results'][dataset] = {
                'dataset': dataset,
                'official_qps': OFFICIAL_QPS[dataset],
                'error': str(exc),
                'traceback': traceback.format_exc(),
                'wall_time_s': time.perf_counter() - started,
            }
            persist_results(output_path, payload)
            print(
                f'Benchmark failed for {dataset}: {exc}',
                file=sys.stderr,
            )
            if args.stop_on_error:
                raise

        persist_results(output_path, payload)
        print(
            json.dumps(payload['results'][dataset], indent=2, sort_keys=True),
            file=sys.stderr,
        )

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
