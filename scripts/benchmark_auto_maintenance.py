from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass

import psycopg


DB_NAME = 'evoke_auto_maintenance_bench'
DOC_COUNT = 5000
DOC_LEN = 24
VOCAB_SIZE = 4000
QUERY_COUNT = 200
TOP_K = 20
INSERT_COUNT = 100
UPDATE_COUNT = 100
DELETE_COUNT = 100
SEED = 20260323


@dataclass
class QueryStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    qps: float


def random_doc(rng: random.Random, doc_len: int) -> list[int]:
    return [rng.randrange(VOCAB_SIZE) for _ in range(doc_len)]


def make_docs(count: int, rng: random.Random) -> list[tuple[int, list[int]]]:
    return [(i + 1, random_doc(rng, DOC_LEN)) for i in range(count)]


def make_queries(count: int, rng: random.Random) -> list[list[int]]:
    queries: list[list[int]] = []
    for _ in range(count):
        query_len = rng.randint(2, 5)
        queries.append([rng.randrange(VOCAB_SIZE) for _ in range(query_len)])
    return queries


def measure_query(
    cur: psycopg.Cursor,
    index_name: str,
    queries: list[list[int]],
) -> QueryStats:
    latencies_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        cur.execute(
            '''
            SELECT *
            FROM public.evoke_query_ids(
                %s::regclass,
                %s::int4[],
                %s,
                NULL
            )
            ''',
            (index_name, query, TOP_K),
        )
        cur.fetchone()
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    total_ms = sum(latencies_ms)
    return QueryStats(
        avg_ms=total_ms / len(latencies_ms),
        p50_ms=statistics.median(latencies_ms),
        p95_ms=sorted(latencies_ms)[int(len(latencies_ms) * 0.95) - 1],
        qps=(len(latencies_ms) * 1000.0) / total_ms,
    )


def maintenance_state(cur: psycopg.Cursor, index_name: str) -> str:
    cur.execute(
        '''
        SELECT format(
            'maintenance_state(rebuilds=%s, '
            'pending_writes=%s, pending_deletes=%s, delta_records=%s, '
            'delta_bytes=%s, stale=%s)',
            rebuilds,
            pending_writes,
            pending_deletes,
            delta_records,
            delta_bytes,
            stale
        )
        FROM public.evoke_index_details(%s::regclass)
        ''',
        (index_name,),
    )
    return str(cur.fetchone()[0])


def insert_rows(cur: psycopg.Cursor, table_name: str, rows: list[tuple[int, list[int]]]) -> float:
    start = time.perf_counter()
    cur.executemany(
        f'INSERT INTO {table_name} (id, token_ids) VALUES (%s, %s)',
        rows,
    )
    return (time.perf_counter() - start) * 1000.0


def update_rows(
    cur: psycopg.Cursor,
    table_name: str,
    rows: list[tuple[list[int], int]]
) -> float:
    start = time.perf_counter()
    cur.executemany(
        f'UPDATE {table_name} SET token_ids = %s WHERE id = %s',
        rows,
    )
    return (time.perf_counter() - start) * 1000.0


def delete_rows(
    cur: psycopg.Cursor,
    table_name: str,
    ids: list[int]
) -> float:
    start = time.perf_counter()
    cur.executemany(
        f'DELETE FROM {table_name} WHERE id = %s',
        [(doc_id,) for doc_id in ids],
    )
    return (time.perf_counter() - start) * 1000.0


def vacuum_table(db_name: str, table_name: str) -> float:
    start = time.perf_counter()
    with psycopg.connect(f'dbname={db_name}', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'VACUUM {table_name}')
    return (time.perf_counter() - start) * 1000.0


def setup_table(
    cur: psycopg.Cursor,
    table_name: str,
    docs: list[tuple[int, list[int]]],
    consistency: str = 'realtime',
) -> None:
    reloptions = [
        "method = 'lucene'",
        "idf_method = 'lucene'",
        f"consistency = '{consistency}'",
    ]
    reloptions_sql = ',\n                '.join(reloptions)

    cur.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
    cur.execute(
        f'''
        CREATE TABLE {table_name} (
            id int PRIMARY KEY,
            token_ids int4[] NOT NULL
        )
        '''
    )
    cur.executemany(
        f'INSERT INTO {table_name} (id, token_ids) VALUES (%s, %s)',
        docs,
    )
    cur.execute(
        f'''
        CREATE INDEX {table_name}_bm25_idx
            ON {table_name} USING evoke (token_ids)
            WITH (
                {reloptions_sql}
            )
        '''
    )


def main() -> None:
    rng = random.Random(SEED)
    docs = make_docs(DOC_COUNT, rng)
    queries = make_queries(QUERY_COUNT, rng)
    insert_rows_data = [
        (DOC_COUNT + i + 1, random_doc(rng, DOC_LEN))
        for i in range(INSERT_COUNT)
    ]
    update_rows_data = [
        (
            random_doc(rng, DOC_LEN),
            rng.randint(1, DOC_COUNT + INSERT_COUNT),
        )
        for _ in range(UPDATE_COUNT)
    ]
    delete_ids = rng.sample(
        list(range(1, DOC_COUNT + INSERT_COUNT + 1)),
        DELETE_COUNT,
    )

    with psycopg.connect('dbname=postgres', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS {DB_NAME}')
            cur.execute(f'CREATE DATABASE {DB_NAME}')

    with psycopg.connect(f'dbname={DB_NAME}') as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION evoke')
            setup_table(cur, 'docs_manual', docs, consistency='manual')
            setup_table(cur, 'docs_auto', docs)
            setup_table(cur, 'docs_auto_txn', docs)
            setup_table(
                cur,
                'docs_auto_threshold_update',
                docs,
            )
            setup_table(
                cur,
                'docs_auto_threshold',
                docs,
            )
            setup_table(
                cur,
                'docs_auto_threshold_delete',
                docs,
            )
            conn.commit()

            manual_before = measure_query(cur, 'docs_manual_bm25_idx', queries)
            auto_before = measure_query(cur, 'docs_auto_bm25_idx', queries)

            start = time.perf_counter()
            insert_rows(cur, 'docs_manual', insert_rows_data)
            conn.commit()
            manual_insert_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            insert_rows(cur, 'docs_auto', insert_rows_data)
            conn.commit()
            auto_insert_ms = (time.perf_counter() - start) * 1000.0

            manual_insert_error = None
            try:
                cur.execute(
                    '''
                    SELECT *
                    FROM public.evoke_query_ids(
                        'docs_manual_bm25_idx'::regclass,
                        %s::int4[],
                        %s,
                        NULL
                    )
                    ''',
                    (queries[0], TOP_K),
                )
                cur.fetchone()
            except Exception as exc:  # pragma: no cover
                manual_insert_error = str(exc).splitlines()[0]
                conn.rollback()
            manual_state_after_insert_query = maintenance_state(
                cur,
                'docs_manual_bm25_idx',
            )

            cur.execute(
                "SELECT public.evoke_index_refresh('docs_manual_bm25_idx'::regclass)"
            )
            manual_after_insert = measure_query(cur, 'docs_manual_bm25_idx', queries)
            auto_after_insert = measure_query(cur, 'docs_auto_bm25_idx', queries)

            start = time.perf_counter()
            update_rows(cur, 'docs_manual', update_rows_data)
            conn.commit()
            manual_update_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            update_rows(cur, 'docs_auto', update_rows_data)
            conn.commit()
            auto_update_ms = (time.perf_counter() - start) * 1000.0

            manual_update_error = None
            try:
                cur.execute(
                    '''
                    SELECT *
                    FROM public.evoke_query_ids(
                        'docs_manual_bm25_idx'::regclass,
                        %s::int4[],
                        %s,
                        NULL
                    )
                    ''',
                    (queries[0], TOP_K),
                )
                cur.fetchone()
            except Exception as exc:  # pragma: no cover
                manual_update_error = str(exc).splitlines()[0]
                conn.rollback()
            manual_state_after_update_query = maintenance_state(
                cur,
                'docs_manual_bm25_idx',
            )

            cur.execute(
                "SELECT public.evoke_index_refresh('docs_manual_bm25_idx'::regclass)"
            )
            manual_after_update = measure_query(cur, 'docs_manual_bm25_idx', queries)
            auto_after_update = measure_query(cur, 'docs_auto_bm25_idx', queries)

            start = time.perf_counter()
            delete_rows(cur, 'docs_manual', delete_ids)
            conn.commit()
            manual_delete_ms = (time.perf_counter() - start) * 1000.0

            start = time.perf_counter()
            delete_rows(cur, 'docs_auto', delete_ids)
            conn.commit()
            auto_delete_ms = (time.perf_counter() - start) * 1000.0

            manual_after_delete = measure_query(cur, 'docs_manual_bm25_idx', queries)
            auto_after_delete = measure_query(cur, 'docs_auto_bm25_idx', queries)

            manual_vacuum_ms = vacuum_table(DB_NAME, 'docs_manual')
            auto_vacuum_ms = vacuum_table(DB_NAME, 'docs_auto')

            manual_delete_vacuum_error = None
            try:
                cur.execute(
                    '''
                    SELECT *
                    FROM public.evoke_query_ids(
                        'docs_manual_bm25_idx'::regclass,
                        %s::int4[],
                        %s,
                        NULL
                    )
                    ''',
                    (queries[0], TOP_K),
                )
                cur.fetchone()
            except Exception as exc:  # pragma: no cover
                manual_delete_vacuum_error = str(exc).splitlines()[0]
                conn.rollback()
            manual_state_after_delete_vacuum_query = maintenance_state(
                cur,
                'docs_manual_bm25_idx',
            )

            cur.execute(
                "SELECT public.evoke_index_refresh('docs_manual_bm25_idx'::regclass)"
            )
            manual_after_delete_refresh = measure_query(
                cur,
                'docs_manual_bm25_idx',
                queries,
            )
            auto_after_delete_vacuum = measure_query(
                cur,
                'docs_auto_bm25_idx',
                queries,
            )

            start = time.perf_counter()
            insert_rows(cur, 'docs_auto_txn', insert_rows_data)
            update_rows(cur, 'docs_auto_txn', update_rows_data)
            conn.commit()
            auto_txn_insert_update_ms = (time.perf_counter() - start) * 1000.0
            auto_txn_maintenance_state = maintenance_state(
                cur,
                'docs_auto_txn_bm25_idx',
            )
            auto_after_txn_batch = measure_query(
                cur,
                'docs_auto_txn_bm25_idx',
                queries,
            )

            start = time.perf_counter()
            insert_rows(cur, 'docs_auto_threshold', insert_rows_data)
            conn.commit()
            auto_threshold_insert_ms = (time.perf_counter() - start) * 1000.0
            auto_threshold_state_before_query = maintenance_state(
                cur,
                'docs_auto_threshold_bm25_idx',
            )
            start = time.perf_counter()
            cur.execute(
                '''
                SELECT *
                FROM public.evoke_query_ids(
                    'docs_auto_threshold_bm25_idx'::regclass,
                    %s::int4[],
                    %s,
                    NULL
                )
                ''',
                (queries[0], TOP_K),
            )
            cur.fetchone()
            auto_threshold_first_query_ms = (
                time.perf_counter() - start
            ) * 1000.0
            auto_threshold_state_after_query = maintenance_state(
                cur,
                'docs_auto_threshold_bm25_idx',
            )
            auto_threshold_after_insert = measure_query(
                cur,
                'docs_auto_threshold_bm25_idx',
                queries,
            )

            start = time.perf_counter()
            update_rows(cur, 'docs_auto_threshold_update', update_rows_data)
            conn.commit()
            auto_threshold_update_ms = (time.perf_counter() - start) * 1000.0
            auto_threshold_update_state_before_query = maintenance_state(
                cur,
                'docs_auto_threshold_update_bm25_idx',
            )
            start = time.perf_counter()
            cur.execute(
                '''
                SELECT *
                FROM public.evoke_query_ids(
                    'docs_auto_threshold_update_bm25_idx'::regclass,
                    %s::int4[],
                    %s,
                    NULL
                )
                ''',
                (queries[0], TOP_K),
            )
            cur.fetchone()
            auto_threshold_update_first_query_ms = (
                time.perf_counter() - start
            ) * 1000.0
            auto_threshold_update_state_after_query = maintenance_state(
                cur,
                'docs_auto_threshold_update_bm25_idx',
            )
            auto_threshold_after_update = measure_query(
                cur,
                'docs_auto_threshold_update_bm25_idx',
                queries,
            )

            start = time.perf_counter()
            delete_rows(cur, 'docs_auto_threshold_delete', delete_ids)
            conn.commit()
            auto_threshold_delete_ms = (time.perf_counter() - start) * 1000.0
            auto_threshold_delete_vacuum_ms = vacuum_table(
                DB_NAME,
                'docs_auto_threshold_delete',
            )
            auto_threshold_delete_state_before_query = maintenance_state(
                cur,
                'docs_auto_threshold_delete_bm25_idx',
            )
            start = time.perf_counter()
            cur.execute(
                '''
                SELECT *
                FROM public.evoke_query_ids(
                    'docs_auto_threshold_delete_bm25_idx'::regclass,
                    %s::int4[],
                    %s,
                    NULL
                )
                ''',
                (queries[0], TOP_K),
            )
            cur.fetchone()
            auto_threshold_delete_first_query_ms = (
                time.perf_counter() - start
            ) * 1000.0
            auto_threshold_delete_state_after_query = maintenance_state(
                cur,
                'docs_auto_threshold_delete_bm25_idx',
            )
            auto_threshold_after_delete = measure_query(
                cur,
                'docs_auto_threshold_delete_bm25_idx',
                queries,
            )

    with psycopg.connect('dbname=postgres', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS {DB_NAME}')

    result = {
        'doc_count': DOC_COUNT,
        'insert_count': INSERT_COUNT,
        'update_count': UPDATE_COUNT,
        'delete_count': DELETE_COUNT,
        'query_count': QUERY_COUNT,
        'top_k': TOP_K,
        'manual_before': manual_before.__dict__,
        'auto_before': auto_before.__dict__,
        'manual_insert_ms': manual_insert_ms,
        'auto_insert_ms': auto_insert_ms,
        'manual_query_after_insert': manual_insert_error,
        'manual_state_after_insert_query': manual_state_after_insert_query,
        'manual_after_insert_refresh': manual_after_insert.__dict__,
        'auto_after_insert': auto_after_insert.__dict__,
        'manual_update_ms': manual_update_ms,
        'auto_update_ms': auto_update_ms,
        'manual_query_after_update': manual_update_error,
        'manual_state_after_update_query': manual_state_after_update_query,
        'manual_after_update_refresh': manual_after_update.__dict__,
        'auto_after_update': auto_after_update.__dict__,
        'manual_delete_ms': manual_delete_ms,
        'auto_delete_ms': auto_delete_ms,
        'manual_after_delete': manual_after_delete.__dict__,
        'auto_after_delete': auto_after_delete.__dict__,
        'manual_vacuum_ms': manual_vacuum_ms,
        'auto_vacuum_ms': auto_vacuum_ms,
        'manual_query_after_delete_vacuum': manual_delete_vacuum_error,
        'manual_state_after_delete_vacuum_query':
            manual_state_after_delete_vacuum_query,
        'manual_after_delete_refresh': manual_after_delete_refresh.__dict__,
        'auto_after_delete_vacuum': auto_after_delete_vacuum.__dict__,
        'auto_txn_insert_update_ms': auto_txn_insert_update_ms,
        'auto_txn_maintenance_state': auto_txn_maintenance_state,
        'auto_after_txn_batch': auto_after_txn_batch.__dict__,
        'auto_threshold_insert_ms': auto_threshold_insert_ms,
        'auto_threshold_state_before_query': auto_threshold_state_before_query,
        'auto_threshold_first_query_ms': auto_threshold_first_query_ms,
        'auto_threshold_state_after_query': auto_threshold_state_after_query,
        'auto_threshold_after_insert': auto_threshold_after_insert.__dict__,
        'auto_threshold_update_ms': auto_threshold_update_ms,
        'auto_threshold_update_state_before_query':
            auto_threshold_update_state_before_query,
        'auto_threshold_update_first_query_ms':
            auto_threshold_update_first_query_ms,
        'auto_threshold_update_state_after_query':
            auto_threshold_update_state_after_query,
        'auto_threshold_after_update': auto_threshold_after_update.__dict__,
        'auto_threshold_delete_ms': auto_threshold_delete_ms,
        'auto_threshold_delete_vacuum_ms': auto_threshold_delete_vacuum_ms,
        'auto_threshold_delete_state_before_query':
            auto_threshold_delete_state_before_query,
        'auto_threshold_delete_first_query_ms':
            auto_threshold_delete_first_query_ms,
        'auto_threshold_delete_state_after_query':
            auto_threshold_delete_state_after_query,
        'auto_threshold_after_delete': auto_threshold_after_delete.__dict__,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
