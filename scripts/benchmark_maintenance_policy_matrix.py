from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from benchmark_maintenance_policy_sweep import POLICIES
from benchmark_maintenance_policy_sweep import ScenarioConfig
from benchmark_maintenance_policy_sweep import run_scenario
from benchmark_maintenance_policy_sweep import summarize_scenario


MATRIX_SCENARIOS = [
    ScenarioConfig(
        name='small_mixed',
        db_name='evoke_maintenance_policy_matrix_small_mixed',
        doc_count=5000,
        query_count=100,
        cycle_count=6,
        insert_per_cycle=50,
        update_per_cycle=50,
        delete_per_cycle=50,
    ),
    ScenarioConfig(
        name='heavy_mixed',
        db_name='evoke_maintenance_policy_matrix_heavy_mixed',
        doc_count=10000,
        query_count=150,
        cycle_count=8,
        insert_per_cycle=100,
        update_per_cycle=100,
        delete_per_cycle=100,
    ),
    ScenarioConfig(
        name='heavy_insert_skew',
        db_name='evoke_maintenance_policy_matrix_heavy_insert_skew',
        doc_count=10000,
        query_count=120,
        cycle_count=8,
        insert_per_cycle=150,
        update_per_cycle=50,
        delete_per_cycle=50,
    ),
    ScenarioConfig(
        name='heavy_update_skew',
        db_name='evoke_maintenance_policy_matrix_heavy_update_skew',
        doc_count=10000,
        query_count=120,
        cycle_count=8,
        insert_per_cycle=50,
        update_per_cycle=150,
        delete_per_cycle=50,
    ),
    ScenarioConfig(
        name='heavy_delete_skew',
        db_name='evoke_maintenance_policy_matrix_heavy_delete_skew',
        doc_count=10000,
        query_count=120,
        cycle_count=8,
        insert_per_cycle=50,
        update_per_cycle=50,
        delete_per_cycle=150,
    ),
]


DEFAULT_BASE_SEED = 20260324
DEFAULT_SEED_STEP = 97
MAX_DB_NAME_LEN = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run a repeatable index maintenance policy matrix.'
    )
    parser.add_argument('--repeat-count', type=int, default=1)
    parser.add_argument('--base-seed', type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument('--seed-step', type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument('--run-id')
    parser.add_argument(
        '--scenario',
        action='append',
        default=[],
        help='Optional scenario name filter.',
    )
    parser.add_argument('--output')
    return parser.parse_args()


def normalized_run_id(raw: str | None) -> str:
    if raw is None or raw.strip() == '':
        return f'p{os.getpid()}'

    chars: list[str] = []
    for ch in raw.strip():
        if ch.isalnum():
            chars.append(ch.lower())
        else:
            chars.append('_')

    value = ''.join(chars).strip('_')
    if value == '':
        return f'p{os.getpid()}'
    return value[:16]


def database_name(base: str, run_id: str, repeat_no: int) -> str:
    suffix = f'_{run_id}_r{repeat_no}'
    if len(base) + len(suffix) <= MAX_DB_NAME_LEN:
        return f'{base}{suffix}'

    keep = MAX_DB_NAME_LEN - len(suffix)
    return f'{base[:keep]}{suffix}'


def selected_scenarios(names: list[str]) -> list[ScenarioConfig]:
    if not names:
        return MATRIX_SCENARIOS

    selected: list[ScenarioConfig] = []
    wanted = set(names)
    for scenario in MATRIX_SCENARIOS:
        if scenario.name in wanted:
            selected.append(scenario)

    if len(selected) != len(wanted):
        known = ', '.join(scenario.name for scenario in MATRIX_SCENARIOS)
        missing = sorted(wanted - {scenario.name for scenario in selected})
        raise ValueError(
            f'unknown scenario(s): {", ".join(missing)}; known: {known}'
        )

    return selected


def best_policy_name(modes: dict[str, object]) -> str:
    best_name = ''
    best_qps = None

    for name, mode in modes.items():
        median_qps = mode['median_qps']
        if best_qps is None or median_qps > best_qps:
            best_qps = median_qps
            best_name = name

    return best_name


def best_deferred_policy_name(modes: dict[str, object]) -> str:
    filtered: dict[str, object] = {
        name: mode for name, mode in modes.items() if name != 'eager'
    }
    return best_policy_name(filtered)


def summarize_matrix(
    scenario: ScenarioConfig,
    summary: dict[str, object],
) -> dict[str, object]:
    modes = summary['modes']
    return {
        'scenario': asdict(scenario),
        'best_policy': best_policy_name(modes),
        'best_deferred_policy': best_deferred_policy_name(modes),
        'modes': modes,
    }


def aggregate_repeats(repeats: list[dict[str, object]]) -> dict[str, object]:
    best_policy_wins: dict[str, int] = {}
    best_deferred_wins: dict[str, int] = {}
    modes: dict[str, dict[str, object]] = {}

    for repeat in repeats:
        best_policy = repeat['best_policy']
        best_deferred = repeat['best_deferred_policy']
        best_policy_wins[best_policy] = best_policy_wins.get(best_policy, 0) + 1
        best_deferred_wins[best_deferred] = (
            best_deferred_wins.get(best_deferred, 0) + 1
        )
        for mode_name, mode in repeat['modes'].items():
            bucket = modes.setdefault(
                mode_name,
                {
                    'policy': mode['policy'],
                    'median_qps_values': [],
                    'avg_commit_ms_values': [],
                    'avg_vacuum_ms_values': [],
                    'final_rebuild_counts': [],
                },
            )
            bucket['median_qps_values'].append(mode['median_qps'])
            bucket['avg_commit_ms_values'].append(mode['avg_commit_ms'])
            bucket['avg_vacuum_ms_values'].append(mode['avg_vacuum_ms'])
            bucket['final_rebuild_counts'].append(mode['final_rebuild_count'])

    aggregated_modes: dict[str, object] = {}
    for mode_name, bucket in modes.items():
        aggregated_modes[mode_name] = {
            'policy': bucket['policy'],
            'median_qps_min': min(bucket['median_qps_values']),
            'median_qps_max': max(bucket['median_qps_values']),
            'median_qps_avg': (
                sum(bucket['median_qps_values']) /
                len(bucket['median_qps_values'])
            ),
            'avg_commit_ms_avg': (
                sum(bucket['avg_commit_ms_values']) /
                len(bucket['avg_commit_ms_values'])
            ),
            'avg_vacuum_ms_avg': (
                sum(bucket['avg_vacuum_ms_values']) /
                len(bucket['avg_vacuum_ms_values'])
            ),
            'final_rebuild_count_values': bucket['final_rebuild_counts'],
        }

    return {
        'repeat_count': len(repeats),
        'best_policy_wins': best_policy_wins,
        'best_deferred_policy_wins': best_deferred_wins,
        'modes': aggregated_modes,
    }


def aggregate_wins(scenarios: dict[str, object]) -> dict[str, object]:
    overall: dict[str, int] = {}
    deferred: dict[str, int] = {}

    for scenario in scenarios.values():
        best = max(
            scenario['aggregate']['best_policy_wins'].items(),
            key=lambda item: item[1],
        )[0]
        best_deferred = max(
            scenario['aggregate']['best_deferred_policy_wins'].items(),
            key=lambda item: item[1],
        )[0]
        overall[best] = overall.get(best, 0) + 1
        deferred[best_deferred] = deferred.get(best_deferred, 0) + 1

    return {
        'overall_best_policy_wins': overall,
        'deferred_best_policy_wins': deferred,
    }


def main() -> None:
    args = parse_args()
    scenarios = selected_scenarios(args.scenario)
    run_id = normalized_run_id(args.run_id)
    output: dict[str, object] = {
        'policies': [asdict(policy) for policy in POLICIES],
        'scenarios': {},
        'repeat_count': args.repeat_count,
        'base_seed': args.base_seed,
        'seed_step': args.seed_step,
        'run_id': run_id,
    }

    with tempfile.TemporaryDirectory(
        prefix='evoke_maintenance_policy_matrix_'
    ) as td:
        tmpdir = Path(td)
        for scenario in scenarios:
            repeat_results: list[dict[str, object]] = []
            for repeat_idx in range(args.repeat_count):
                seed = args.base_seed + (repeat_idx * args.seed_step)
                scenario_run = ScenarioConfig(
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
                    scenario_run,
                    tmpdir / f'{scenario.name}_r{repeat_idx + 1}.json',
                    seed=seed,
                )
                summary = summarize_scenario(scenario_run, raw)
                repeat_results.append(
                    {
                        'repeat': repeat_idx + 1,
                        'seed': seed,
                        **summarize_matrix(scenario, summary),
                    }
                )
            output['scenarios'][scenario.name] = {
                'scenario': asdict(scenario),
                'repeats': repeat_results,
                'aggregate': aggregate_repeats(repeat_results),
            }

    output['summary'] = aggregate_wins(output['scenarios'])
    rendered = json.dumps(output, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(rendered)
            f.write('\n')
    else:
        print(rendered)


if __name__ == '__main__':
    main()
