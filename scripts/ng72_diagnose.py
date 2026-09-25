"""Bounded TRAIN-only cross-code and margin-retention diagnosis, no inference."""

import argparse
from collections import Counter
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
from scipy import sparse


PARENT = 'NG-0071/pilot-v1'
INITIAL = 'NG-0069/evaluation-v2/encode-initial'
INPUT_SHA = '5c44c86e13d688972100f1eb1a85b562ba8a631bdf31a2b2e4073ef9b304fb2a'
REVIEW_SHA = 'f99d5a33670d3a9cb018eba86ec5780256f4c51d14fcfeb14345ff51c7e6fda3'
DOMAINS = ('fever', 'hotpotqa', 'nq')
MODES = ('00', '10', '01', '11')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(test, message):
    if not test:
        raise ValueError(message)


def parent_modules(base):
    parent = base / PARENT
    require(sha(parent / 'inputs.json') == INPUT_SHA, 'NG71 input anchor changed')
    for name, digest in read(parent / 'inputs.json')['source'].items():
        require(sha(parent / name) == digest, 'NG71 source changed: ' + name)
    sys.path.insert(0, str(parent))
    import ng71_pilot as pipeline
    import ng71_observation as observation
    import ng71_execution as execution
    return pipeline, observation, execution


def train_surface(rows, selection, per_domain=128):
    lookup = {row['query_id']: (i, row) for i, row in enumerate(rows)}
    require(len(lookup) == len(rows), 'duplicate lexical query identity')
    result, seen = [], set()
    for surface in ('pilot', 'sentinel'):
        selected = selection[surface]
        require(len(selected) == per_domain * 3, 'TRAIN count changed')
        for identity in selected:
            require(identity not in seen, 'overlapping TRAIN surfaces')
            seen.add(identity)
            index, row = lookup[identity]
            require(row['split'] == 'TRAIN', 'DEV/LOCKED access forbidden')
            result.append(dict(row, lexical_index=index,
                               surface='TRAIN_' + surface.upper()))
        require(Counter(lookup[q][1]['subset'] for q in selected)
                == Counter({d: per_domain for d in DOMAINS}), 'domain balance changed')
    return result


def frozen_rank_match(actual, expected):
    for key in ('gold_ids', 'gold_ranks', 'top100'):
        require(actual[key] == expected[key], 'frozen TRAIN rank mismatch: ' + key)
    for key in ('ndcg10', 'recall100', 'top100_scores'):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-12)


def decomposition(q0, q1, d0, d1):
    """Exact score terms plus effective matching-support changes per document."""
    dq, dd = q1 - q0, d1 - d0
    terms = dict(query=np.asarray(d0 @ dq).ravel(),
                 document=np.asarray(dd @ q0).ravel(),
                 interaction=np.asarray(dd @ dq).ravel())
    z0, z1 = d0.multiply(q0).tocsr(), d1.multiply(q1).tocsr()
    z0.eliminate_zeros()
    z1.eliminate_zeros()
    total0, total1 = [np.asarray(z.sum(1)).ravel() for z in (z0, z1)]
    common0 = np.asarray(z0.multiply(z1.sign()).sum(1)).ravel()
    common1 = np.asarray(z1.multiply(z0.sign()).sum(1)).ravel()
    terms.update(support_lost=common0 - total0, support_gained=total1 - common1,
                 common_support_weight=common1 - common0)
    delta = total1 - total0
    np.testing.assert_allclose(sum(terms[k] for k in MODAL_TERMS), delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(sum(terms[k] for k in SUPPORT_TERMS), delta, rtol=0, atol=1e-12)
    return terms, delta


MODAL_TERMS = ('query', 'document', 'interaction')
SUPPORT_TERMS = ('support_lost', 'support_gained', 'common_support_weight')


def teacher_pair(record, positive, competitor):
    if record is None:
        return dict(status='unobserved', margin=None, confidence=None)
    positions = {doc: i for i, doc in enumerate(record['pool'])}
    if positive not in positions or competitor not in positions:
        return dict(status='unobserved', margin=None, confidence=None)
    values = record.get('teacher_scores', record.get('scores'))
    margin = values[positions[positive]] - values[positions[competitor]]
    return dict(status='agrees' if margin > 0 else 'tie' if margin == 0 else 'opposes',
                margin=margin, confidence=float(np.tanh(margin / .08)))


def collect_inputs(base):
    pipeline, _, execution = parent_modules(base)
    frozen = pipeline.verify(base, base / PARENT)
    require(sha(base / PARENT / 'review/complete.json') == REVIEW_SHA,
            'NG71 terminal review anchor changed')
    review = execution.sealed(base / PARENT / 'review')
    require(review['rows_audited'] == 17664 and not review['locked_test_scored'],
            'NG71 review contract changed')
    config = read(base / PARENT / 'config.json')
    require(config['base']['lexical_weight'] == .1
            and config['base']['semantic_weight'] == .9
            and config['base']['refit_calibration'] is False
            and config['base']['output_posting_cap'] is None, 'scoring basis changed')
    initial = read(base / INITIAL / 'results.json')
    current = execution.sealed(base / PARENT / 'encode-D-192')
    trained = execution.sealed(base / PARENT / 'train-D-192')
    require(initial['model_sha256'] == config['base']['state_sha256']
            and current['model_sha256'] == trained['model_sha256']
            and current['checkpoint_complete_sha256']
            == sha(base / PARENT / 'train-D-192/complete.json'), 'encoded model chain changed')
    files = dict(frozen['dependencies'])
    files.update({PARENT + '/' + n: h for n, h in frozen['source'].items()})
    files[PARENT + '/inputs.json'] = INPUT_SHA
    for phase in ('review', 'train-D-192', 'encode-D-192', 'rank-initial', 'rank-D-192', 'snapshot-96'):
        folder = base / PARENT / phase
        execution.sealed(folder)
        files.update({PARENT + '/' + phase + '/' + n: h
                      for n, h in read(folder / 'complete.json')['files'].items()})
        files[PARENT + '/' + phase + '/complete.json'] = sha(folder / 'complete.json')
    return files


def verify(base, run):
    frozen = read(run / 'inputs.json')
    require(frozen['protocol'] == 'NG72_TRAIN_cross_codes_v1'
            and frozen['training_updates'] == 0 and frozen['new_model_inference'] is False,
            'diagnostic contract changed')
    for root, field in ((base, 'dependencies'), (run, 'source')):
        for name, expected in frozen[field].items():
            path = root / name
            require(path.resolve().is_relative_to(root.resolve()) and sha(path) == expected,
                    'frozen input changed: ' + name)
    return frozen


def freeze(base, run):
    require(run.parent == base / 'NG-0072', 'fresh NG72 unit required')
    dependencies = collect_inputs(base)
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root / 'tests/test_ng72_diagnose.py',
               root / 'docs/research-sae/reports/ng0001-ng0099/ng0072-retention-diagnosis-plan.zh.md']
    run.mkdir(parents=True, exist_ok=False)
    for path in sources:
        shutil.copy2(path, run / path.name)
    write(run / 'inputs.json', dict(
        protocol='NG72_TRAIN_cross_codes_v1', dependencies=dependencies,
        source={p.name: sha(p) for p in sources}, parent_review_sha256=REVIEW_SHA,
        training_updates=0, new_model_inference=False, locked_test_scored=False))
    verify(base, run)


def summarize(rankings, margin_rows, audit):
    quality, harm, effects = {}, {}, {}
    for surface in ('TRAIN_PILOT', 'TRAIN_SENTINEL'):
        for domain in DOMAINS:
            key = surface + '/' + domain
            selected = {mode: [r for r in rows if r['surface'] == surface
                               and r['domain'] == domain] for mode, rows in rankings.items()}
            # Do not call the three-domain macro helper on a single domain.
            quality[key] = {mode: {m: float(np.mean([r[m] for r in rows]))
                                   for m in ('ndcg10', 'recall100')}
                            for mode, rows in selected.items()}
            harm[key] = {mode: audit.harm(selected['00'], selected[mode])
                         for mode in ('10', '01', '11')}
            group = [r for r in margin_rows if r['surface'] == surface and r['domain'] == domain]
            effects[key] = {}
            for origin in ('all', 'original_positive', 'added_vs_original'):
                subset = [r for r in group if origin == 'all' or r['origin'] == origin]
                # First average competitors, then positives, then queries.
                by_query = {}
                for r in subset:
                    by_query.setdefault(r['query_id'], []).append(r['means'])
                effects[key][origin] = dict(
                    queries=len(by_query), positives=len(subset),
                    pair_count=sum(r['competitors'] for r in subset),
                    query_balanced_mean={term: float(np.mean([
                        np.mean([p[term] for p in pieces]) for pieces in by_query.values()]))
                        for term in (*MODAL_TERMS, *SUPPORT_TERMS, 'total')}
                    if by_query else {})
            effects[key]['retention'] = {
                state: dict(pairs=sum(r['retention'][state]['pairs'] for r in group),
                            query_balanced_contribution={term: float(np.mean([
                                np.mean([r['retention'][state][term] for r in group
                                         if r['query_id'] == identity])
                                for identity in sorted({r['query_id'] for r in group})]))
                                for term in (*MODAL_TERMS, *SUPPORT_TERMS, 'total')})
                for state in ('lost', 'retained', 'gained', 'stayed_behind')}
            effects[key]['teacher_status_pair_counts'] = dict(sum(
                (Counter(r['teacher_counts']) for r in group), Counter()))
    return dict(quality=quality, harm=harm, margin_effects=effects)


def diagnose(base, run, output, task):
    _, observation, _ = parent_modules(base)
    import analyze_ng66_learning_curves as audit
    parent, lexical = base / PARENT, base / 'NG-0069/lexical-v1'
    queries = train_surface(read(lexical / 'queries.json'), read(parent / 'selection.json'))
    ids = {q['query_id'] for q in queries}
    labels = read(lexical / 'labels.json')
    gold = [labels[q['lexical_index']] for q in queries]
    encoded_ids = read(parent / 'encode-D-192/query-ids.json')
    require(len(encoded_ids) == len(set(encoded_ids)) == 2304, 'D192 query identity changed')
    positions = {q: i for i, q in enumerate(encoded_ids)}
    indexes = [q['lexical_index'] for q in queries]
    q0 = sparse.load_npz(base / INITIAL / 'query.npz')[indexes].astype(np.float64)
    q1 = sparse.load_npz(parent / 'encode-D-192/query.npz')[[positions[q['query_id']]
                                                          for q in queries]].astype(np.float64)
    d0 = sparse.load_npz(base / INITIAL / 'document.npz').astype(np.float64)
    d1 = sparse.load_npz(parent / 'encode-D-192/document.npz').astype(np.float64)
    lq = sparse.load_npz(lexical / 'lexical-query.npz')[indexes].astype(np.float64)
    ld = sparse.load_npz(lexical / 'lexical-document.npz').astype(np.float64)
    require(q0.shape == q1.shape == (768, d0.shape[1])
            and d0.shape == d1.shape and d0.shape[0] == ld.shape[0] == 233009,
            'incompatible encoded basis or background')
    for matrix in (q0, q1, d0, d1, lq, ld):
        require(matrix.has_canonical_format and np.isfinite(matrix.data).all()
                and (matrix.data > 0).all(), 'invalid canonical positive codes')
    di0, di1, li = d0.T.tocsr(), d1.T.tocsr(), ld.T.tocsr()
    expected = {}
    for mode, name in (('00', 'initial'), ('11', 'D-192')):
        with (parent / f'rank-{name}/rankings.jsonl').open() as stream:
            expected[mode] = {r['query_id']: r for line in stream
                              if (r := json.loads(line))['query_id'] in ids}
        require(set(expected[mode]) == ids, 'frozen TRAIN reference incomplete')
    history = {r['query_id']: r for r in read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    wanted = {doc for values in gold for doc in values}
    keys = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                keys[i] = json.loads(line)['key']
    require(i + 1 == 233009 and set(keys) == wanted, 'positive document mapping incomplete')
    targets = {r['query_id']: r for r in read(base / 'NG-0069/teacher-v1/targets.json')}
    snapshots = {}
    for step, folder in ((0, base / 'NG-0071/pilot-step0-v1'), (96, parent / 'snapshot-96')):
        with (folder / 'witnesses.jsonl').open() as stream:
            snapshots[step] = {r['query_id']: r for line in stream if (r := json.loads(line))}
    rankings, margins, max_error = {m: [] for m in MODES}, [], 0.
    start = time.monotonic()
    with (output / 'rankings.jsonl').open('x') as rank_stream, (output / 'margins.jsonl').open('x') as margin_stream:
        for i, query in enumerate(queries):
            require(query['split'] == 'TRAIN', 'scoring forbidden split')
            identity = query['query_id']
            lexical_score = (lq.getrow(i) @ li).toarray().ravel()
            lexical_alt = ld @ lq.getrow(i).toarray().ravel()
            scores, records = {}, {}
            for mode, q, d, index in (('00', q0, d0, di0), ('10', q1, d0, di0),
                                      ('01', q0, d1, di1), ('11', q1, d1, di1)):
                score = lexical_score + (q.getrow(i) @ index).toarray().ravel()
                alternate = lexical_alt + d @ q.getrow(i).toarray().ravel()
                row, _, error = observation.rank_record(score, alternate, gold[i])
                max_error = max(max_error, error)
                row.update(query_id=identity, domain=query['subset'], split='TRAIN', surface=query['surface'])
                if mode in expected:
                    frozen_rank_match(row, expected[mode][identity])
                records[mode], scores[mode] = row, score
                rankings[mode].append(row)
                rank_stream.write(json.dumps(dict(mode=mode, **row), allow_nan=False) + '\n')
            rivals = set()
            for mode in ('00', '11'):
                rivals.update(records[mode]['top100'][:10])
                rivals.update(records[mode]['top100'][79:100])
            for reference in (targets.get(identity), snapshots[0].get(identity), snapshots[96].get(identity)):
                if reference:
                    rivals.update(reference['pool'])
            rivals.difference_update(gold[i])
            rivals = sorted(rivals)
            selected = sorted(set(rivals) | set(gold[i]))
            loc = {doc: j for j, doc in enumerate(selected)}
            terms, delta = decomposition(q0.getrow(i).toarray().ravel(), q1.getrow(i).toarray().ravel(),
                                         d0[selected], d1[selected])
            np.testing.assert_allclose(delta, scores['11'][selected] - scores['00'][selected],
                                       rtol=0, atol=1e-12)
            lineage = history[identity]
            require(lineage['status'] == 'resolved', 'unresolved TRAIN provenance')
            origins = {p['document_key']: p['provenance'] for p in lineage['pairs']}
            require(set(origins) == {keys[g] for g in gold[i]}, 'all-positive lineage mismatch')
            for positive in gold[i]:
                a, indexes = loc[positive], [loc[d] for d in rivals]
                effects = {term: (values[a] - values[indexes]).tolist() for term, values in terms.items()}
                effects['total'] = (delta[a] - delta[indexes]).tolist()
                before = scores['00'][positive] - scores['00'][rivals]
                after = scores['11'][positive] - scores['11'][rivals]
                ahead0 = (before > 0) | ((before == 0) & (positive < np.asarray(rivals)))
                ahead1 = (after > 0) | ((after == 0) & (positive < np.asarray(rivals)))
                groups = dict(lost=ahead0 & ~ahead1, retained=ahead0 & ahead1,
                              gained=~ahead0 & ahead1, stayed_behind=~ahead0 & ~ahead1)
                visibility = next((v for v in snapshots[0].get(identity, {}).get('positive_visibility', [])
                                   if v['document_id'] == positive), None)
                record = dict(query_id=identity, surface=query['surface'], domain=query['subset'],
                              positive_id=positive, origin=origins[keys[positive]],
                              positive_ranks={mode: row['gold_ranks'][gold[i].index(positive)]
                                              for mode, row in records.items()},
                              competitor_ids=rivals, effects=effects,
                              before_margin=before.tolist(), after_margin=after.tolist(),
                              baseline_ahead=ahead0.tolist(), final_ahead=ahead1.tolist(),
                              teacher_original=[teacher_pair(targets[identity], positive, d) for d in rivals],
                              teacher_step0=[teacher_pair(snapshots[0].get(identity), positive, d) for d in rivals],
                              teacher_step96=[teacher_pair(snapshots[96].get(identity), positive, d) for d in rivals],
                              visibility=visibility, evidence_span_visible='unknown_no_span_judgments',
                              competitors_are_judged_negatives=False)
                margin_stream.write(json.dumps(record, allow_nan=False) + '\n')
                margins.append(dict(query_id=identity, surface=query['surface'], domain=query['subset'],
                                    origin=origins[keys[positive]], competitors=len(rivals),
                                    teacher_counts=dict(Counter(v['status'] for v in record[
                                        'teacher_step96' if identity in snapshots[96] else 'teacher_original'])),
                                    retention={state: dict(pairs=int(mask.sum()), **{
                                        term: float(np.mean(np.asarray(value) * mask))
                                        for term, value in effects.items()}) for state, mask in groups.items()},
                                    means={k: float(np.mean(v)) for k, v in effects.items()}))
            if i == 15:
                prediction = (time.monotonic() - start) / 16 * len(queries) * 1.5 + 180
                write(output / 'canary.json', dict(queries=16, predicted_seconds=prediction,
                                                    passed=prediction < 5100))
                require(prediction < 5100, 'TRAIN diagnosis exceeds fixed cost bound')
            if (i + 1) % 32 == 0:
                rank_stream.flush()
                margin_stream.flush()
                print('NG72_TRAIN', i + 1, len(queries), time.monotonic() - start, flush=True)
                task.get_logger().report_scalar('progress', 'queries', i + 1, i + 1)
    return dict(passed=True, **summarize(rankings, margins, audit), queries=768,
                positive_rows=len(margins), ranked_rows=3072, documents=233009,
                max_independent_score_error=max_error, frozen_00_11_rank_parity=True,
                loop_seconds=time.monotonic() - start, training_updates=0,
                new_model_inference=False, dev_scored=False, locked_test_scored=False,
                native_cost_evaluated=False, overall_goal_qualified=False)


def worker(base, run):
    verify(base, run)
    require(sha(Path(__file__)) == read(run / 'inputs.json')['source']['ng72_diagnose.py'],
            'executing source differs from frozen diagnostic')
    parent_modules(base)
    from clearml import Task
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0072/TRAIN-cross-codes-v1',
                     task_type=Task.TaskTypes.testing, reuse_last_task_id=False,
                     auto_connect_frameworks=False, auto_connect_arg_parser=False,
                     auto_connect_streams=False, auto_resource_monitoring=False)
    output = run / 'diagnosis'
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(input_sha256=sha(run / 'inputs.json'), training_updates=0, train_queries=768))
        result = diagnose(base, run, output, task)
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


def resources_ok(run):
    return (psutil.virtual_memory().available > 24 * 1024 ** 3
            and shutil.disk_usage(run).free > 40 * 1024 ** 3)


def supervise(base, run):
    verify(base, run)
    pipeline, _, _ = parent_modules(base)
    require(resources_ok(run), 'host memory or disk insufficient')
    output = run / 'diagnosis'
    output.mkdir(exist_ok=False)
    stopped, peak, error = [], 0, None
    old = {sig: signal.signal(sig, lambda number, frame: stopped.append(number))
           for sig in (signal.SIGINT, signal.SIGTERM)}
    env = {k: v for k, v in os.environ.items() if not k.startswith(
        ('CLEARML_TASK_', 'TRAINS_TASK_', 'CLEARML_PROC_', 'TRAINS_PROC_'))}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               MKL_NUM_THREADS='4', VECLIB_MAXIMUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
               PYTHONDONTWRITEBYTECODE='1', CLEARML_OFFLINE_MODE='1', HF_HUB_OFFLINE='1',
               CLEARML_CACHE_DIR=str(output / 'tracking-cache'))
    start = time.monotonic()
    try:
        with (output / 'stdout.log').open('x') as log:
            process = subprocess.Popen([sys.executable, '-B', str(run / 'ng72_diagnose.py'),
                                        'worker', '--research-root', str(base), '--run', str(run)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            try:
                write(output / 'started.json', dict(worker_pid=process.pid, controller_pid=os.getpid(),
                                                     started_unix=time.time(), gpu=None, limit_seconds=5400))
                while process.poll() is None:
                    try:
                        root = psutil.Process(process.pid)
                        rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
                    except psutil.NoSuchProcess:
                        rss = 0
                    peak = max(peak, rss)
                    require(not stopped and resources_ok(run) and rss < 16 * 1024 ** 3
                            and time.monotonic() - start < 5400, 'bounded diagnosis stopped')
                    time.sleep(.5)
            except BaseException as exc:
                error = f'{type(exc).__name__}: {exc}'
            finally:
                closed = pipeline.terminate_group(process)
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
        require(passed, 'diagnosis failed; preserve without automatic retry')
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
    require(run.parent == base / 'NG-0072', 'run must remain in the new NG72 unit')
    globals()[args.mode](base, run)


if __name__ == '__main__':
    main()
