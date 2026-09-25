"""Replay frozen NG71 TRAIN score derivatives against NG72 pair outcomes.

This is descriptive score-space evidence, not parameter-update causality.
No encoder, teacher, DEV or LOCKED_TEST inference is performed.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import psutil

import ng71_diagnostics as diagnostics


PARENT = 'NG-0071/pilot-v1'
DIAG = 'NG-0072/cross-codes-v1'
PARENT_SHA = '5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a'
DIAG_SHA = '74f69b38d84456976810a89748b7ea7b2a9f50819471abb8dc527ef86c264671'
DIAG_INPUT_SHA = '8ca39510880622470e7423f389a34d5dd5b06c68b504d0c47b6a7f5ec467b794'
EPSILON = 1e-12
GROUPS = ('lost', 'retained', 'gained', 'stayed_behind')
STATES = ('absent', 'ineligible', 'expand', 'shrink', 'stationary')


def require(test, message):
    if not test:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def unique(records):
    result = {r['query_id']: r for r in records}
    require(len(result) == len(records), 'duplicate query identity')
    return result


def outcome(before, after):
    return ('retained' if after else 'lost') if before else (
        'gained' if after else 'stayed_behind')


def direction(gradient):
    require(np.isfinite(gradient), 'non-finite derivative')
    return 'shrink' if gradient > EPSILON else (
        'expand' if gradient < -EPSILON else 'stationary')


def replay_example(record, example, config, batch_size=4):
    require(record['split'] == 'TRAIN', 'only TRAIN gradients allowed')
    require(record['query_id'] == example['query_id'], 'query order changed')
    require(example['objective'] == 'balanced_soft_pair'
            and example['vjp_replay_exact'] is True, 'frozen objective changed')
    require(example['documents'] == len(record['pool']), 'candidate count changed')
    require(len(set(record['pool'])) == len(record['pool']), 'duplicate candidates')
    result = diagnostics.objective(
        dict(record, hybrid_scores=example['scores']), config, 'balanced_soft_pair')
    expected = np.asarray(result['score_gradient']) / batch_size
    np.testing.assert_allclose(example['score_gradient'], expected,
                               rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(example['loss'], result['loss'], rtol=2e-6, atol=2e-7)
    require(example['eligible_pairs'] == result['eligible_pairs']
            and example['supervised_positives'] == result['supervised_positives'],
            'recorded supervision changed')
    return result


def pair_rows(record, example, replay, margins, head, step, reference):
    positions = {doc: i for i, doc in enumerate(record['pool'])}
    gold = {doc for doc, positive in zip(record['pool'], record['positive_mask'], strict=True)
            if positive}
    require({m['positive_id'] for m in margins} == gold, 'positive coverage changed')
    require(set(head['gold_ids']) == gold, 'baseline gold differs')
    pairs = {tuple(pair): (target, coefficient, derivative)
             for pair, target, coefficient, derivative in zip(
                 replay['pair_indices'], replay['pair_targets'],
                 replay['pair_coefficients'], replay['pair_score_derivatives'], strict=True)}
    rank = {doc: i + 1 for i, doc in enumerate(head['top100'])}
    rank.update(zip(head['gold_ids'], head['gold_ranks'], strict=True))
    gradient = replay['score_gradient']
    covered = set()
    result = []
    for m in margins:
        p = positions[m['positive_id']]
        ids = m['competitor_ids']
        require(len(ids) == len(set(ids)) and not gold.intersection(ids), 'invalid rivals')
        require(m['query_id'] == record['query_id']
                and m['surface'] == 'TRAIN_PILOT', 'sentinel/foreign margin forbidden')
        data = dict(query_id=record['query_id'], domain=record['domain'],
                    positive_id=m['positive_id'], origin=m['origin'], step=step,
                    reference_update=reference, competitor_ids=ids,
                    before_margin=m['before_margin'], after_margin=m['after_margin'])
        fields = ('outcome', 'state', 'target', 'coefficient', 'own_derivative',
                  'current_margin', 'aggregate_margin_pressure',
                  'cross_top10', 'cross_top100', 'baseline_head_pair')
        data.update({key: [] for key in fields})
        for j, doc in enumerate(ids):
            n = positions.get(doc)
            pair = pairs.get((p, n))
            target, coefficient, derivative = pair if pair else (None, None, None)
            state = direction(derivative) if pair else ('absent' if n is None else 'ineligible')
            if pair:
                covered.add((p, n))
            values = dict(
                outcome=outcome(m['baseline_ahead'][j], m['final_ahead'][j]),
                state=state, target=target, coefficient=coefficient,
                own_derivative=derivative,
                current_margin=example['scores'][p] - example['scores'][n] if n is not None else None,
                aggregate_margin_pressure=gradient[p] - gradient[n] if n is not None else None,
                cross_top10=(rank[m['positive_id']] <= 10) != (rank.get(doc, 101) <= 10),
                cross_top100=(rank[m['positive_id']] <= 100) != (rank.get(doc, 101) <= 100),
                baseline_head_pair=min(rank[m['positive_id']], rank.get(doc, 101)) <= 10)
            for key, value in values.items():
                data[key].append(value)
        result.append(data)
    require(covered == set(pairs), 'NG72 rival union misses an actual supervised pair')
    return result


def summarize(records):
    """Full rival denominator, then positive/exposure/query balance per domain."""
    result = {}
    for domain in ('fever', 'hotpotqa', 'nq'):
        domain_rows = [r for r in records if r['domain'] == domain]
        result[domain] = {}
        for origin in ('all', 'original_positive', 'added_vs_original'):
            selected = [r for r in domain_rows if origin == 'all' or r['origin'] == origin]
            report = {}
            for group in ('all', *GROUPS):
                counts = Counter({s: 0 for s in STATES})
                masses = Counter(expand=0., shrink=0.)
                balanced = defaultdict(list)
                aggregate, head = Counter(), Counter()
                for row in selected:
                    mask = np.asarray([group == 'all' or value == group for value in row['outcome']])
                    local = Counter(s for s, keep in zip(row['state'], mask, strict=True) if keep)
                    counts.update(local)
                    own = {s: sum(abs(d) for d, state, keep in zip(
                        row['own_derivative'], row['state'], mask, strict=True)
                        if keep and state == s) for s in ('expand', 'shrink')}
                    masses.update(own)
                    for state, g, is_head, keep in zip(
                            row['state'], row['aggregate_margin_pressure'],
                            row['baseline_head_pair'], mask, strict=True):
                        if keep:
                            aggregate.update(['unobserved' if g is None else direction(g)])
                            if is_head:
                                head.update([state])
                    denom = len(row['competitor_ids'])
                    require(denom > 0, 'empty full rival denominator')
                    balanced[row['query_id']].append({**{s: local[s] / denom for s in STATES},
                                                     **{s + '_mass': v / denom for s, v in own.items()}})
                means = {key: float(np.mean([np.mean([r[key] for r in parts])
                                             for parts in balanced.values()]))
                         for key in (*STATES, 'expand_mass', 'shrink_mass')} if balanced else {}
                report[group] = dict(
                    pair_exposures=sum(counts.values()), state_counts=dict(counts),
                    absolute_pair_derivative_mass=dict(masses),
                    aggregate_margin_pressure_counts=dict(aggregate),
                    baseline_head_state_counts=dict(head),
                    full_rival_query_balanced=means)
            result[domain][origin] = dict(queries=len({r['query_id'] for r in selected}),
                                         positive_exposures=len(selected), groups=report)
    return result


def verify(base, run):
    frozen = read(run / 'inputs.json')
    require(frozen['protocol'] == 'NG73_actual_TRAIN_derivatives_v1', 'protocol changed')
    for root, key in ((base, 'dependencies'), (run, 'source')):
        for name, digest in frozen[key].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()) and sha(path) == digest,
                    'frozen input changed: ' + name)
    return frozen


def freeze(base, run):
    require(sha(base / PARENT / 'inputs.json') == PARENT_SHA, 'NG71 anchor changed')
    require(sha(base / DIAG / 'diagnosis/complete.json') == DIAG_SHA, 'NG72 anchor changed')
    require(sha(base / DIAG / 'inputs.json') == DIAG_INPUT_SHA, 'NG72 input anchor changed')
    parent = read(base / PARENT / 'inputs.json')
    diag_inputs = read(base / DIAG / 'inputs.json')
    dependencies = {PARENT + '/inputs.json': PARENT_SHA, DIAG + '/inputs.json': DIAG_INPUT_SHA}
    trained_path = base / PARENT / 'train-D-192'
    require(sha(trained_path / 'complete.json') == diag_inputs['dependencies'][
        PARENT + '/train-D-192/complete.json'], 'D192 anchor changed')
    require(sha(trained_path / 'results.json') == read(trained_path / 'complete.json')[
        'files']['results.json'], 'D192 predecessor receipt changed')
    require(sha(base / PARENT / 'train-D-96/complete.json')
            == read(trained_path / 'results.json')['previous_complete_sha256'],
            'D96 optimizer predecessor changed')
    for name in ('config.json', 'selection.json', 'training-order.json'):
        require(name in parent['source'], 'NG71 source input missing')
        dependencies[PARENT + '/' + name] = parent['source'][name]
    witness = 'NG-0071/pilot-step0-v1/witnesses.jsonl'
    dependencies[witness] = parent['dependencies'][witness]
    for root, phase, files in (
            (PARENT, 'train-D-96', ['progress.jsonl']),
            (PARENT, 'train-D-192', ['progress.jsonl']),
            (PARENT, 'snapshot-96', ['witnesses.jsonl']),
            (DIAG, 'diagnosis', ['margins.jsonl', 'rankings.jsonl'])):
        prefix = root + '/' + phase + '/'
        complete = read(base / prefix / 'complete.json')
        require(complete['passed'] is True, 'unsealed predecessor')
        dependencies[prefix + 'complete.json'] = sha(base / prefix / 'complete.json')
        if prefix + 'complete.json' in diag_inputs['dependencies']:
            require(dependencies[prefix + 'complete.json']
                    == diag_inputs['dependencies'][prefix + 'complete.json'], 'NG72 lineage changed')
        for name in (*files, 'exit.json', 'clearml.json', 'results.json'):
            dependencies[prefix + name] = complete['files'][name]
            require(sha(base / prefix / name) == complete['files'][name], 'predecessor file changed')
        exit_receipt = read(base / prefix / 'exit.json')
        tracking = read(base / prefix / 'clearml.json')
        require(exit_receipt['exit_code'] == 0 and exit_receipt['error'] is None
                and exit_receipt['owned_group_closed'] is True
                and tracking['actual_start'] is True and tracking['closed'] is True,
                'incomplete predecessor lifecycle')
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'tests/test_ng73_gradient_replay.py',
               root / 'docs/research-sae/reports/ng0001-ng0099/ng0073-gradient-replay-plan.zh.md']
    for name in ('ng71_diagnostics.py', 'ng71_training.py', 'ng71_ranking.py'):
        path = root / 'scripts' / name
        require(sha(path) == parent['source'][name], 'NG71 derivative implementation changed')
        sources.append(path)
    run.mkdir(parents=True, exist_ok=False)
    for path in sources:
        shutil.copy2(path, run / path.name)
    write(run / 'inputs.json', dict(protocol='NG73_actual_TRAIN_derivatives_v1',
                                    dependencies=dependencies,
                                    source={p.name: sha(p) for p in sources},
                                    training_updates=0, new_model_inference=False,
                                    dev_scored=False, locked_test_scored=False))
    verify(base, run)


def audit(base, run, output, task):
    parent = base / PARENT
    config = read(parent / 'config.json')
    order = read(parent / 'training-order.json')
    selection = read(parent / 'selection.json')
    require(len(order) == 768 and Counter(order) == Counter({q: 2 for q in selection['pilot']}),
            'TRAIN exposure order changed')
    require(not set(order).intersection(selection['sentinel']), 'sentinel gradient leakage')
    margins = defaultdict(list)
    for row in rows(base / DIAG / 'diagnosis/margins.jsonl'):
        if row['surface'] == 'TRAIN_PILOT':
            margins[row['query_id']].append(row)
    heads = unique([r for r in rows(base / DIAG / 'diagnosis/rankings.jsonl')
                    if r['surface'] == 'TRAIN_PILOT' and r['mode'] == '00'])
    require(set(margins) == set(heads) == set(order), 'pilot diagnostic identities differ')
    sources = {0: unique(rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')),
               96: unique(rows(parent / 'snapshot-96/witnesses.jsonl'))}
    progress = rows(parent / 'train-D-96/progress.jsonl') + rows(parent / 'train-D-192/progress.jsonl')
    require([r['step'] for r in progress] == list(range(1, 193)), 'noncontiguous actual updates')
    result, observed, max_error = [], [], 0.
    for update in progress:
        reference = 0 if update['step'] <= 96 else 96
        require(update['reference_update'] == reference and len(update['examples']) == 4,
                'snapshot/batch changed')
        for example in update['examples']:
            identity = example['query_id']
            record = sources[reference][identity]
            require(identity == order[len(observed)] and record['reference_update'] == reference,
                    'actual exposure/snapshot order changed')
            replay = replay_example(record, example, config['ranking'])
            max_error = max(max_error, float(np.max(np.abs(
                np.asarray(example['score_gradient']) - np.asarray(replay['score_gradient']) / 4))))
            result.extend(pair_rows(record, example, replay, margins[identity], heads[identity],
                                    update['step'], reference))
            observed.append(identity)
        if update['step'] % 24 == 0:
            print('NG73_REPLAY', update['step'], 192, flush=True)
            task.get_logger().report_scalar('progress', 'updates_replayed', update['step'], update['step'])
    with (output / 'pairs.jsonl').open('x') as stream:
        for row in result:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    return dict(passed=True, summary=summarize(result), updates_replayed=192,
                query_exposures=len(observed), positive_exposures=len(result),
                queries=len(set(observed)), max_recorded_gradient_error=max_error,
                training_updates=0, new_model_inference=False, dev_scored=False,
                locked_test_scored=False, parameter_update_causality_established=False,
                overall_goal_qualified=False)


def worker(base, run):
    verify(base, run)
    require(sha(Path(__file__)) == read(run / 'inputs.json')['source'][Path(__file__).name],
            'executing source differs from frozen source')
    from clearml import Task
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0073/actual-TRAIN-derivatives-v1',
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     auto_connect_frameworks=False, auto_connect_arg_parser=False,
                     auto_connect_streams=False, auto_resource_monitoring=False)
    output = run / 'audit'
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=sha(run / 'inputs.json'), new_training_updates=0))
        result = audit(base, run, output, task)
        verify(base, run)
        write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException:
        receipt['outcome'] = 'failed'
        raise
    finally:
        task.close()
        receipt['closed'] = True
        write(output / 'clearml.json', receipt)


def supervise(base, run):
    verify(base, run)
    output = run / 'audit'
    output.mkdir(exist_ok=False)
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CLEARML_TASK_', 'TRAINS_TASK_', 'CLEARML_PROC_', 'TRAINS_PROC_'))}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CLEARML_OFFLINE_MODE='1', HF_HUB_OFFLINE='1',
               CLEARML_CACHE_DIR=str(output / 'tracking-cache'))
    stopped, error, peak = [], None, 0
    old = {sig: signal.signal(sig, lambda number, frame: stopped.append(number))
           for sig in (signal.SIGINT, signal.SIGTERM)}
    start = time.monotonic()
    process = None
    try:
        with (output / 'stdout.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(run / Path(__file__).name),
                                        'worker', '--research-root', str(base), '--run', str(run)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                write(output / 'started.json', dict(worker_pid=process.pid, controller_pid=os.getpid(),
                                                    started_unix=time.time(), limit_seconds=900, gpu=None))
                while process.poll() is None:
                    try:
                        root = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(not stopped and rss < 4 * 1024 ** 3
                            and psutil.virtual_memory().available > 8 * 1024 ** 3
                            and shutil.disk_usage(run).free > 10 * 1024 ** 3
                            and time.monotonic() - start < 900, 'bounded replay stopped')
                    time.sleep(.5)
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        break
                    time.sleep(.2)
                process.wait(timeout=10)
        try:
            os.killpg(process.pid, 0)
            closed = False
        except ProcessLookupError:
            closed = True
        write(output / 'exit.json', dict(exit_code=process.returncode, error=error,
                                         owned_group_closed=closed, peak_tree_rss_bytes=peak,
                                         elapsed_seconds=time.monotonic() - start, finished_unix=time.time()))
        tracking = read(output / 'clearml.json') if (output / 'clearml.json').exists() else {}
        passed = (process.returncode == 0 and error is None and closed and not stopped
                  and tracking.get('closed') is True and tracking.get('actual_start') is True
                  and tracking.get('outcome') == 'passed')
        verify(base, run)
        write(output / 'complete.json', dict(passed=passed, files={str(p.relative_to(output)): sha(p)
                              for p in sorted(output.rglob('*')) if p.is_file()}))
        require(passed, 'replay failed; preserve without automatic retry')
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    base, run = args.research_root.resolve(strict=True), args.run.resolve()
    require(run.parent == base / 'NG-0073', 'fresh NG73 unit required')
    globals()[args.mode](base, run)


if __name__ == '__main__':
    main()
