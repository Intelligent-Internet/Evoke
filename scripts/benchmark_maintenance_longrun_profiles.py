from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from benchmark_maintenance_policy_matrix import aggregate_repeats
from benchmark_maintenance_policy_matrix import database_name
from benchmark_maintenance_policy_matrix import normalized_run_id
from benchmark_maintenance_policy_matrix import summarize_matrix
from benchmark_maintenance_policy_sweep import POLICIES
from benchmark_maintenance_policy_sweep import ScenarioConfig
from benchmark_maintenance_policy_sweep import run_scenario
from benchmark_maintenance_policy_sweep import summarize_scenario


DEFAULT_DB_NAME = 'evoke_maintenance_longrun_mixed'
DEFAULT_DOC_COUNT = 15000
DEFAULT_QUERY_COUNT = 150
DEFAULT_CYCLE_COUNT = 12
DEFAULT_INSERT_PER_CYCLE = 150
DEFAULT_UPDATE_PER_CYCLE = 150
DEFAULT_DELETE_PER_CYCLE = 150
DEFAULT_REPEAT_COUNT = 3
DEFAULT_BASE_SEED = 20260324
DEFAULT_SEED_STEP = 97


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a repeatable long-run maintenance policy comparison.'
    )
    parser.add_argument('--db-name', default=DEFAULT_DB_NAME)
    parser.add_argument('--doc-count', type=int, default=DEFAULT_DOC_COUNT)
    parser.add_argument('--query-count', type=int, default=DEFAULT_QUERY_COUNT)
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
    parser.add_argument('--repeat-count', type=int, default=DEFAULT_REPEAT_COUNT)
    parser.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument('--seed-step', type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument('--run-id')
    parser.add_argument('--output')
    return parser.parse_args()


def build_scenario(args: argparse.Namespace) -> ScenarioConfig:
    return ScenarioConfig(
        name='longrun_mixed',
        db_name=args.db_name,
        doc_count=args.doc_count,
        query_count=args.query_count,
        cycle_count=args.cycle_count,
        insert_per_cycle=args.insert_per_cycle,
        update_per_cycle=args.update_per_cycle,
        delete_per_cycle=args.delete_per_cycle,
    )


def main() -> None:
    args = parse_args()
    scenario = build_scenario(args)
    run_id = normalized_run_id(args.run_id)
    repeats: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(
        prefix='evoke_maintenance_longrun_profiles_'
    ) as td:
        tmpdir = Path(td)
        for repeat_idx in range(args.repeat_count):
            seed = args.base_seed + (repeat_idx * args.seed_step)
            run_cfg = ScenarioConfig(
                name=scenario.name,
                db_name=database_name(
                    scenario.db_name,
                    run_id,
                    repeat_idx + 1,
                ),
                doc_count=scenario.doc_count,
                query_count=scenario.query_count,
                cycle_count=scenario.cycle_count,
                insert_per_cycle=scenario.insert_per_cycle,
                update_per_cycle=scenario.update_per_cycle,
                delete_per_cycle=scenario.delete_per_cycle,
            )
            raw = run_scenario(
                run_cfg,
                tmpdir / f'longrun_mixed_r{repeat_idx + 1}.json',
                seed=seed,
            )
            summary = summarize_scenario(run_cfg, raw)
            repeats.append(
                {
                    'repeat': repeat_idx + 1,
                    'seed': seed,
                    **summarize_matrix(scenario, summary),
                }
            )

    output = {
        'scenario': asdict(scenario),
        'policies': [asdict(policy) for policy in POLICIES],
        'repeat_count': args.repeat_count,
        'base_seed': args.base_seed,
        'seed_step': args.seed_step,
        'run_id': run_id,
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
