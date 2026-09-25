from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from dataclasses import asdict
from dataclasses import dataclass

import psycopg


DEFAULT_DB_NAME = 'evoke_maintenance_churn_bench'
DEFAULT_DOC_COUNT = 5000
DEFAULT_DOC_LEN = 24
DEFAULT_VOCAB_SIZE = 4000
DEFAULT_QUERY_COUNT = 100
DEFAULT_TOP_K = 20
DEFAULT_CYCLE_COUNT = 6
DEFAULT_INSERT_PER_CYCLE = 50
DEFAULT_UPDATE_PER_CYCLE = 50
DEFAULT_DELETE_PER_CYCLE = 50
DEFAULT_SEED = 20260324


@dataclass
class QueryStats:
    avg_ms: float
    p50_ms: float
    p95_ms: float
    qps: float


@dataclass
class ModeConfig:
    table_name: str
    reloptions: str


@dataclass
class BenchmarkConfig:
    db_name: str
    doc_count: int
    doc_len: int
    vocab_size: int
    query_count: int
    top_k: int
    cycle_count: int
    insert_per_cycle: int
    update_per_cycle: int
    delete_per_cycle: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the mixed-churn index maintenance benchmark.'
    )
    parser.add_argument('--db-name', default=DEFAULT_DB_NAME)
    parser.add_argument('--doc-count', type=int, default=DEFAULT_DOC_COUNT)
    parser.add_argument('--doc-len', type=int, default=DEFAULT_DOC_LEN)
    parser.add_argument('--vocab-size', type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument('--query-count', type=int, default=DEFAULT_QUERY_COUNT)
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--cycle-count', type=int, default=DEFAULT_CYCLE_COUNT)
    parser.add_argument(
        '--insert-per-cycle',
        type=int,
        default=DEFAULT_INSERT_PER_CYCLE,
    )
    parser.add_argument(
        '--update-per-cycle',
        type=int,
        default=DEFAULT_UPDATE_PER_CYCLE,
    )
    parser.add_argument(
        '--delete-per-cycle',
        type=int,
        default=DEFAULT_DELETE_PER_CYCLE,
    )
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument(
        '--mode',
        action='append',
        default=[],
        help=(
            'Override the default maintenance mode list with one or more '
            'entries formatted as table_name:reloptions'
        ),
    )
    parser.add_argument('--output')
    return parser.parse_args()


def random_doc(rng: random.Random, cfg: BenchmarkConfig) -> list[int]:
    return [rng.randrange(cfg.vocab_size) for _ in range(cfg.doc_len)]


def make_docs(
    count: int,
    rng: random.Random,
    cfg: BenchmarkConfig,
) -> list[tuple[int, list[int]]]:
    return [(i + 1, random_doc(rng, cfg)) for i in range(count)]


def make_queries(
    count: int,
    rng: random.Random,
    cfg: BenchmarkConfig,
) -> list[list[int]]:
    queries: list[list[int]] = []
    for _ in range(count):
        query_len = rng.randint(2, 5)
        queries.append([rng.randrange(cfg.vocab_size) for _ in range(query_len)])
    return queries


def measure_query(
    cur: psycopg.Cursor,
    index_name: str,
    queries: list[list[int]],
    top_k: int,
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
            (index_name, query, top_k),
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
            'maintenance_state(rebuilds=%%s, '
            'pending_writes=%%s, pending_deletes=%%s, delta_records=%%s, '
            'delta_bytes=%%s, stale=%%s)',
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


def maintenance_policy(cur: psycopg.Cursor, index_name: str) -> str:
    cur.execute(
        '''
        SELECT format(
            'maintenance_policy(consistency=%%s)',
            consistency
        )
        FROM public.evoke_index_details(%s::regclass)
        ''',
        (index_name,),
    )
    return str(cur.fetchone()[0])


def parse_rebuild_count(state: str) -> int:
    match = re.search(r'rebuilds=(\d+)', state)
    if match is None:
        raise ValueError(f'could not parse rebuild count from: {state}')
    return int(match.group(1))


def setup_table(
    cur: psycopg.Cursor,
    docs: list[tuple[int, list[int]]],
    mode: ModeConfig,
) -> None:
    cur.execute(f'DROP TABLE IF EXISTS {mode.table_name} CASCADE')
    cur.execute(
        f'''
        CREATE TABLE {mode.table_name} (
            id int PRIMARY KEY,
            token_ids int4[] NOT NULL
        )
        '''
    )
    cur.executemany(
        f'INSERT INTO {mode.table_name} (id, token_ids) VALUES (%s, %s)',
        docs,
    )
    cur.execute(
        f'''
        CREATE INDEX {mode.table_name}_bm25_idx
            ON {mode.table_name} USING evoke (token_ids)
            WITH ({mode.reloptions})
        '''
    )


def vacuum_table(db_name: str, table_name: str) -> float:
    start = time.perf_counter()
    with psycopg.connect(f'dbname={db_name}', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'VACUUM {table_name}')
    return (time.perf_counter() - start) * 1000.0


def apply_cycle(
    cur: psycopg.Cursor,
    cfg: BenchmarkConfig,
    mode: ModeConfig,
    rng: random.Random,
    next_id: int,
    active_ids: list[int],
) -> tuple[int, list[int], dict[str, object]]:
    insert_rows: list[tuple[int, list[int]]] = []
    update_rows: list[tuple[list[int], int]] = []
    delete_ids = rng.sample(active_ids, cfg.delete_per_cycle)
    surviving_ids = [doc_id for doc_id in active_ids if doc_id not in delete_ids]

    for _ in range(cfg.insert_per_cycle):
        insert_rows.append((next_id, random_doc(rng, cfg)))
        surviving_ids.append(next_id)
        next_id += 1

    update_targets = rng.sample(surviving_ids, cfg.update_per_cycle)
    for doc_id in update_targets:
        update_rows.append((random_doc(rng, cfg), doc_id))

    start = time.perf_counter()
    try:
        cur.executemany(
            f'INSERT INTO {mode.table_name} (id, token_ids) VALUES (%s, %s)',
            insert_rows,
        )
        cur.executemany(
            f'UPDATE {mode.table_name} SET token_ids = %s WHERE id = %s',
            update_rows,
        )
        cur.executemany(
            f'DELETE FROM {mode.table_name} WHERE id = %s',
            [(doc_id,) for doc_id in delete_ids],
        )
        cur.connection.commit()
    except Exception as exc:
        cur.connection.rollback()
        raise RuntimeError(
            'cycle maintenance failed '
            f'(db={cfg.db_name}, table={mode.table_name}, '
            f'inserts={cfg.insert_per_cycle}, '
            f'updates={cfg.update_per_cycle}, '
            f'deletes={cfg.delete_per_cycle})'
        ) from exc
    commit_ms = (time.perf_counter() - start) * 1000.0

    vacuum_ms = vacuum_table(cfg.db_name, mode.table_name)
    index_name = f'{mode.table_name}_bm25_idx'
    state = maintenance_state(cur, index_name)
    policy = maintenance_policy(cur, index_name)

    return next_id, surviving_ids, {
        'commit_ms': commit_ms,
        'vacuum_ms': vacuum_ms,
        'maintenance_state': state,
        'maintenance_policy': policy,
        'rebuild_count': parse_rebuild_count(state),
    }


def summarize_mode(cycles: list[dict[str, object]]) -> dict[str, float]:
    qps_values = [float(cycle['query']['qps']) for cycle in cycles]
    commit_values = [float(cycle['commit_ms']) for cycle in cycles]
    vacuum_values = [float(cycle['vacuum_ms']) for cycle in cycles]
    rebuild_increments = [
        int(cycles[i]['rebuild_count']) - int(cycles[i - 1]['rebuild_count'])
        for i in range(1, len(cycles))
    ]

    return {
        'median_qps': statistics.median(qps_values),
        'min_qps': min(qps_values),
        'max_qps': max(qps_values),
        'avg_commit_ms': statistics.mean(commit_values),
        'avg_vacuum_ms': statistics.mean(vacuum_values),
        'final_rebuild_count': int(cycles[-1]['rebuild_count']),
        'max_rebuild_increment': max(rebuild_increments, default=0),
    }


def default_modes() -> list[ModeConfig]:
    return [
        ModeConfig(
            table_name='docs_realtime',
            reloptions=(
                "method = 'lucene', idf_method = 'lucene', "
                "consistency = 'realtime'"
            ),
        ),
        ModeConfig(
            table_name='docs_eventual',
            reloptions=(
                "method = 'lucene', idf_method = 'lucene', "
                "consistency = 'eventual'"
            ),
        ),
    ]


def parse_modes(mode_args: list[str]) -> list[ModeConfig]:
    modes: list[ModeConfig] = []

    if not mode_args:
        return default_modes()

    for mode_arg in mode_args:
        if ':' not in mode_arg:
            raise ValueError(
                'mode must be formatted as table_name:reloptions'
            )
        table_name, reloptions = mode_arg.split(':', 1)
        table_name = table_name.strip()
        reloptions = reloptions.strip()
        if not table_name or not reloptions:
            raise ValueError(
                'mode must include both table_name and reloptions'
            )
        modes.append(ModeConfig(table_name=table_name, reloptions=reloptions))

    return modes


def main() -> None:
    args = parse_args()
    cfg = BenchmarkConfig(
        db_name=args.db_name,
        doc_count=args.doc_count,
        doc_len=args.doc_len,
        vocab_size=args.vocab_size,
        query_count=args.query_count,
        top_k=args.top_k,
        cycle_count=args.cycle_count,
        insert_per_cycle=args.insert_per_cycle,
        update_per_cycle=args.update_per_cycle,
        delete_per_cycle=args.delete_per_cycle,
        seed=args.seed,
    )
    rng = random.Random(cfg.seed)
    docs = make_docs(cfg.doc_count, rng, cfg)
    queries = make_queries(cfg.query_count, rng, cfg)
    modes = parse_modes(args.mode)

    with psycopg.connect('dbname=postgres', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS {cfg.db_name}')
            cur.execute(f'CREATE DATABASE {cfg.db_name} TEMPLATE template0')

    output: dict[str, object] = {
        'config': asdict(cfg),
        'modes': {},
    }

    with psycopg.connect(f'dbname={cfg.db_name}') as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION evoke')
            for mode in modes:
                setup_table(cur, docs, mode)
            conn.commit()

            for mode in modes:
                active_ids = [doc_id for doc_id, _ in docs]
                next_id = cfg.doc_count + 1
                baseline = measure_query(
                    cur,
                    f'{mode.table_name}_bm25_idx',
                    queries,
                    cfg.top_k,
                )
                index_name = f'{mode.table_name}_bm25_idx'
                baseline_state = maintenance_state(cur, index_name)
                mode_cycles: list[dict[str, object]] = [{
                    'cycle': 0,
                    'query': asdict(baseline),
                    'commit_ms': 0.0,
                    'vacuum_ms': 0.0,
                    'maintenance_state': baseline_state,
                    'maintenance_policy': maintenance_policy(cur, index_name),
                    'rebuild_count': parse_rebuild_count(baseline_state),
                }]

                for cycle in range(1, cfg.cycle_count + 1):
                    try:
                        next_id, active_ids, cycle_info = apply_cycle(
                            cur,
                            cfg,
                            mode,
                            rng,
                            next_id,
                            active_ids,
                        )
                        cycle_info['cycle'] = cycle
                        cycle_info['query'] = asdict(
                            measure_query(cur, index_name, queries, cfg.top_k)
                        )
                        mode_cycles.append(cycle_info)
                    except Exception as exc:
                        raise RuntimeError(
                            'benchmark mode failed '
                            f'(db={cfg.db_name}, table={mode.table_name}, '
                            f'cycle={cycle}, seed={cfg.seed})'
                        ) from exc

                summary = summarize_mode(mode_cycles)
                summary['maintenance_policy'] = maintenance_policy(
                    cur,
                    index_name,
                )
                output['modes'][mode.table_name] = {
                    'config': asdict(mode),
                    'cycles': mode_cycles,
                    'summary': summary,
                }

    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(rendered)
            f.write('\n')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
