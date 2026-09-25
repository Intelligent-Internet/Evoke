from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import asdict
from dataclasses import dataclass

import psycopg

from benchmark_maintenance_churn import BenchmarkConfig
from benchmark_maintenance_churn import ModeConfig
from benchmark_maintenance_churn import maintenance_policy
from benchmark_maintenance_churn import maintenance_state
from benchmark_maintenance_churn import make_docs
from benchmark_maintenance_churn import make_queries
from benchmark_maintenance_churn import measure_query
from benchmark_maintenance_churn import parse_rebuild_count
from benchmark_maintenance_churn import random_doc
from benchmark_maintenance_churn import setup_table
from benchmark_maintenance_churn import summarize_mode
from benchmark_maintenance_churn import vacuum_table
from benchmark_maintenance_policy_matrix import aggregate_repeats
from benchmark_maintenance_policy_matrix import database_name
from benchmark_maintenance_policy_matrix import normalized_run_id
from benchmark_maintenance_policy_matrix import summarize_matrix
from benchmark_maintenance_policy_sweep import POLICIES
from benchmark_maintenance_policy_sweep import ScenarioConfig
from benchmark_maintenance_policy_sweep import summarize_scenario


@dataclass(frozen=True)
class TraceStep:
    name: str
    inserts: int
    updates: int
    deletes: int
    update_selector: str = 'uniform'
    delete_selector: str = 'uniform'


@dataclass(frozen=True)
class TraceProfile:
    name: str
    db_name: str
    doc_count: int
    query_count: int
    steps: tuple[TraceStep, ...]


TRACE_PROFILES = [
    TraceProfile(
        name='read_heavy_microbatch',
        db_name='evoke_maintenance_trace_read_heavy',
        doc_count=12000,
        query_count=180,
        steps=(
            TraceStep(
                name='microbatch',
                inserts=12,
                updates=18,
                deletes=6,
                update_selector='recent_hotset',
                delete_selector='oldest_coldset',
            ),
        ) * 12,
    ),
    TraceProfile(
        name='bursty_ingest',
        db_name='evoke_maintenance_trace_bursty_ingest',
        doc_count=15000,
        query_count=160,
        steps=(
            TraceStep(
                name='ingest_burst',
                inserts=260,
                updates=30,
                deletes=20,
                update_selector='recent_hotset',
                delete_selector='oldest_coldset',
            ),
            TraceStep(
                name='settle',
                inserts=60,
                updates=70,
                deletes=30,
                update_selector='recent_hotset',
                delete_selector='uniform',
            ),
            TraceStep(
                name='cleanup',
                inserts=20,
                updates=40,
                deletes=90,
                update_selector='uniform',
                delete_selector='oldest_coldset',
            ),
        ) * 4,
    ),
    TraceProfile(
        name='hotset_mutation',
        db_name='evoke_maintenance_trace_hotset',
        doc_count=10000,
        query_count=140,
        steps=(
            TraceStep(
                name='hot_mutation',
                inserts=35,
                updates=220,
                deletes=20,
                update_selector='recent_hotset',
                delete_selector='oldest_coldset',
            ),
        ) * 10,
    ),
    TraceProfile(
        name='retention_cleanup',
        db_name='evoke_maintenance_trace_retention',
        doc_count=14000,
        query_count=150,
        steps=(
            TraceStep(
                name='cleanup_wave',
                inserts=30,
                updates=50,
                deletes=220,
                update_selector='uniform',
                delete_selector='oldest_coldset',
            ),
            TraceStep(
                name='recovery',
                inserts=170,
                updates=70,
                deletes=40,
                update_selector='recent_hotset',
                delete_selector='uniform',
            ),
        ) * 5,
    ),
]


DEFAULT_DOC_LEN = 24
DEFAULT_VOCAB_SIZE = 4000
DEFAULT_TOP_K = 20
DEFAULT_REPEAT_COUNT = 3
DEFAULT_BASE_SEED = 20260324
DEFAULT_SEED_STEP = 97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run production-shaped maintenance churn profiles.'
    )
    parser.add_argument(
        '--profile',
        action='append',
        default=[],
        help='Optional trace profile name filter.',
    )
    parser.add_argument('--repeat-count', type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument('--seed-step', type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument('--run-id')
    parser.add_argument('--output')
    return parser.parse_args()


def selected_profiles(names: list[str]) -> list[TraceProfile]:
    if not names:
        return TRACE_PROFILES

    selected: list[TraceProfile] = []
    wanted = set(names)
    for profile in TRACE_PROFILES:
        if profile.name in wanted:
            selected.append(profile)

    if len(selected) != len(wanted):
        known = ', '.join(profile.name for profile in TRACE_PROFILES)
        missing = sorted(wanted - {profile.name for profile in selected})
        raise ValueError(
            f'unknown profile(s): {", ".join(missing)}; known: {known}'
        )

    return selected


def sample_ids(
    rng: random.Random,
    active_ids: list[int],
    count: int,
    selector: str,
) -> list[int]:
    if count <= 0 or not active_ids:
        return []

    if selector == 'recent_hotset':
        hot_len = max(count, len(active_ids) // 5)
        pool = active_ids[-hot_len:]
    elif selector == 'oldest_coldset':
        cold_len = max(count, len(active_ids) // 2)
        pool = active_ids[:cold_len]
    else:
        pool = active_ids

    if len(pool) < count:
        pool = active_ids

    return rng.sample(pool, count)


def apply_trace_step(
    cur: psycopg.Cursor,
    cfg: BenchmarkConfig,
    table_name: str,
    step: TraceStep,
    rng: random.Random,
    next_id: int,
    active_ids: list[int],
) -> tuple[int, list[int], dict[str, object]]:
    delete_ids = sample_ids(
        rng,
        active_ids,
        min(step.deletes, len(active_ids)),
        step.delete_selector,
    )
    surviving_ids = [doc_id for doc_id in active_ids if doc_id not in delete_ids]

    insert_rows: list[tuple[int, list[int]]] = []
    for _ in range(step.inserts):
        insert_rows.append((next_id, random_doc(rng, cfg)))
        surviving_ids.append(next_id)
        next_id += 1

    update_targets = sample_ids(
        rng,
        surviving_ids,
        min(step.updates, len(surviving_ids)),
        step.update_selector,
    )
    update_rows = [
        (random_doc(rng, cfg), doc_id)
        for doc_id in update_targets
    ]

    started = time.perf_counter()
    try:
        if insert_rows:
            cur.executemany(
                f'INSERT INTO {table_name} (id, token_ids) VALUES (%s, %s)',
                insert_rows,
            )
        if update_rows:
            cur.executemany(
                f'UPDATE {table_name} SET token_ids = %s WHERE id = %s',
                update_rows,
            )
        if delete_ids:
            cur.executemany(
                f'DELETE FROM {table_name} WHERE id = %s',
                [(doc_id,) for doc_id in delete_ids],
            )
        cur.connection.commit()
    except Exception as exc:
        cur.connection.rollback()
        raise RuntimeError(
            'trace step failed '
            f'(db={cfg.db_name}, table={table_name}, step={step.name}, '
            f'inserts={step.inserts}, updates={step.updates}, '
            f'deletes={step.deletes})'
        ) from exc
    commit_ms = (time.perf_counter() - started) * 1000.0

    vacuum_ms = vacuum_table(cfg.db_name, table_name)
    index_name = f'{table_name}_bm25_idx'
    state = maintenance_state(cur, index_name)
    policy = maintenance_policy(cur, index_name)

    return next_id, surviving_ids, {
        'commit_ms': commit_ms,
        'vacuum_ms': vacuum_ms,
        'maintenance_state': state,
        'maintenance_policy': policy,
        'rebuild_count': parse_rebuild_count(state),
        'trace_step': step.name,
        'step_config': asdict(step),
    }


def profile_benchmark_config(
    profile: TraceProfile,
    db_name: str,
    seed: int,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        db_name=db_name,
        doc_count=profile.doc_count,
        doc_len=DEFAULT_DOC_LEN,
        vocab_size=DEFAULT_VOCAB_SIZE,
        query_count=profile.query_count,
        top_k=DEFAULT_TOP_K,
        cycle_count=len(profile.steps),
        insert_per_cycle=0,
        update_per_cycle=0,
        delete_per_cycle=0,
        seed=seed,
    )


def run_profile_once(
    profile: TraceProfile,
    db_name: str,
    seed: int,
) -> dict[str, object]:
    cfg = profile_benchmark_config(profile, db_name, seed)
    rng = random.Random(cfg.seed)
    docs = make_docs(cfg.doc_count, rng, cfg)
    queries = make_queries(cfg.query_count, rng, cfg)

    with psycopg.connect('dbname=postgres', autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS {cfg.db_name}')
            cur.execute(f'CREATE DATABASE {cfg.db_name} TEMPLATE template0')

    output: dict[str, object] = {
        'config': asdict(cfg),
        'profile': {
            'name': profile.name,
            'db_name': profile.db_name,
            'steps': [asdict(step) for step in profile.steps],
        },
        'modes': {},
    }

    with psycopg.connect(f'dbname={cfg.db_name}') as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION evoke')
            for policy in POLICIES:
                setup_table(
                    cur,
                    docs,
                    ModeConfig(
                        table_name=f'docs_{policy.name}',
                        reloptions=policy.reloptions,
                    ),
                )
            conn.commit()

            for policy in POLICIES:
                table_name = f'docs_{policy.name}'
                index_name = f'{table_name}_bm25_idx'
                active_ids = [doc_id for doc_id, _ in docs]
                next_id = cfg.doc_count + 1
                baseline = measure_query(cur, index_name, queries, cfg.top_k)
                baseline_state = maintenance_state(cur, index_name)
                cycles: list[dict[str, object]] = [{
                    'cycle': 0,
                    'query': asdict(baseline),
                    'commit_ms': 0.0,
                    'vacuum_ms': 0.0,
                    'maintenance_state': baseline_state,
                    'maintenance_policy': maintenance_policy(cur, index_name),
                    'rebuild_count': parse_rebuild_count(baseline_state),
                    'trace_step': 'baseline',
                    'step_config': None,
                }]

                for cycle_no, step in enumerate(profile.steps, start=1):
                    next_id, active_ids, cycle_info = apply_trace_step(
                        cur,
                        cfg,
                        table_name,
                        step,
                        rng,
                        next_id,
                        active_ids,
                    )
                    cycle_info['cycle'] = cycle_no
                    cycle_info['query'] = asdict(
                        measure_query(cur, index_name, queries, cfg.top_k)
                    )
                    cycles.append(cycle_info)

                summary = summarize_mode(cycles)
                summary['maintenance_policy'] = maintenance_policy(
                    cur,
                    index_name,
                )
                output['modes'][table_name] = {
                    'config': {
                        'table_name': table_name,
                        'reloptions': policy.reloptions,
                    },
                    'cycles': cycles,
                    'summary': summary,
                }

    return output


def summarize_profile_repeat(
    profile: TraceProfile,
    raw: dict[str, object],
) -> dict[str, object]:
    scenario = ScenarioConfig(
        name=profile.name,
        db_name=profile.db_name,
        doc_count=profile.doc_count,
        query_count=profile.query_count,
        cycle_count=len(profile.steps),
        insert_per_cycle=max(step.inserts for step in profile.steps),
        update_per_cycle=max(step.updates for step in profile.steps),
        delete_per_cycle=max(step.deletes for step in profile.steps),
    )
    summary = summarize_scenario(scenario, raw)
    return summarize_matrix(scenario, summary)


def main() -> None:
    args = parse_args()
    profiles = selected_profiles(args.profile)
    run_id = normalized_run_id(args.run_id)
    output: dict[str, object] = {
        'profiles': {},
        'policies': [asdict(policy) for policy in POLICIES],
        'repeat_count': args.repeat_count,
        'base_seed': args.base_seed,
        'seed_step': args.seed_step,
        'run_id': run_id,
    }

    with tempfile.TemporaryDirectory(
        prefix='evoke_maintenance_trace_profiles_'
    ):
        for profile in profiles:
            repeats: list[dict[str, object]] = []
            for repeat_idx in range(args.repeat_count):
                seed = args.base_seed + (repeat_idx * args.seed_step)
                db_name = database_name(
                    profile.db_name,
                    run_id,
                    repeat_idx + 1,
                )
                raw = run_profile_once(profile, db_name, seed)
                repeats.append(
                    {
                        'repeat': repeat_idx + 1,
                        'seed': seed,
                        **summarize_profile_repeat(profile, raw),
                    }
                )

            output['profiles'][profile.name] = {
                'profile': {
                    'name': profile.name,
                    'db_name': profile.db_name,
                    'doc_count': profile.doc_count,
                    'query_count': profile.query_count,
                    'steps': [asdict(step) for step in profile.steps],
                },
                'repeats': repeats,
                'aggregate': aggregate_repeats(repeats),
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
