from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
CHURN_SCRIPT = REPO_ROOT / 'scripts' / 'benchmark_maintenance_churn.py'


@dataclass
class PolicyConfig:
    name: str
    reloptions: str


@dataclass
class ScenarioConfig:
    name: str
    db_name: str
    doc_count: int
    query_count: int
    cycle_count: int
    insert_per_cycle: int
    update_per_cycle: int
    delete_per_cycle: int


POLICIES = [
    PolicyConfig(
        name='realtime',
        reloptions=(
            "method = 'lucene', idf_method = 'lucene', "
            "consistency = 'realtime'"
        ),
    ),
    PolicyConfig(
        name='eventual',
        reloptions=(
            "method = 'lucene', idf_method = 'lucene', "
            "consistency = 'eventual'"
        ),
    ),
]


SCENARIOS = [
    ScenarioConfig(
        name='small',
        db_name='evoke_maintenance_policy_sweep_small',
        doc_count=5000,
        query_count=100,
        cycle_count=6,
        insert_per_cycle=50,
        update_per_cycle=50,
        delete_per_cycle=50,
    ),
    ScenarioConfig(
        name='heavy',
        db_name='evoke_maintenance_policy_sweep_heavy',
        doc_count=10000,
        query_count=150,
        cycle_count=8,
        insert_per_cycle=100,
        update_per_cycle=100,
        delete_per_cycle=100,
    ),
]


def run_scenario(
    scenario: ScenarioConfig,
    output_path: Path,
    seed: int | None = None,
    policies: list[PolicyConfig] | None = None,
) -> dict[str, object]:
    active_policies = POLICIES if policies is None else policies
    cmd = [
        sys.executable,
        str(CHURN_SCRIPT),
        '--db-name',
        scenario.db_name,
        '--doc-count',
        str(scenario.doc_count),
        '--query-count',
        str(scenario.query_count),
        '--cycle-count',
        str(scenario.cycle_count),
        '--insert-per-cycle',
        str(scenario.insert_per_cycle),
        '--update-per-cycle',
        str(scenario.update_per_cycle),
        '--delete-per-cycle',
        str(scenario.delete_per_cycle),
        '--output',
        str(output_path),
    ]
    if seed is not None:
        cmd.extend(['--seed', str(seed)])

    for policy in active_policies:
        cmd.extend(
            [
                '--mode',
                f'docs_{policy.name}:{policy.reloptions}',
            ]
        )

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return json.loads(output_path.read_text(encoding='utf-8'))


def summarize_scenario(
    scenario: ScenarioConfig,
    raw: dict[str, object],
) -> dict[str, object]:
    modes = raw['modes']
    summary_modes: dict[str, object] = {}
    best_policy = None
    best_median_qps = None

    for policy in POLICIES:
        mode = modes[f'docs_{policy.name}']
        summary = mode['summary']
        last_cycle = mode['cycles'][-1]
        summary_modes[policy.name] = {
            'policy': summary['maintenance_policy'],
            'median_qps': summary['median_qps'],
            'min_qps': summary['min_qps'],
            'max_qps': summary['max_qps'],
            'avg_commit_ms': summary['avg_commit_ms'],
            'avg_vacuum_ms': summary['avg_vacuum_ms'],
            'final_rebuild_count': summary['final_rebuild_count'],
            'final_state': last_cycle['maintenance_state'],
        }
        if best_median_qps is None or summary['median_qps'] > best_median_qps:
            best_median_qps = summary['median_qps']
            best_policy = policy.name

    return {
        'scenario': asdict(scenario),
        'best_median_qps_policy': best_policy,
        'modes': summary_modes,
    }


def main() -> None:
    output: dict[str, object] = {
        'policies': [asdict(policy) for policy in POLICIES],
        'scenarios': {},
    }

    with tempfile.TemporaryDirectory(
        prefix='evoke_maintenance_policy_sweep_'
    ) as td:
        tmpdir = Path(td)
        for scenario in SCENARIOS:
            raw = run_scenario(
                scenario,
                tmpdir / f'{scenario.name}.json',
            )
            output['scenarios'][scenario.name] = summarize_scenario(
                scenario,
                raw,
            )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
