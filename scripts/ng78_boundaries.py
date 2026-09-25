"""TRAIN-only, cached-code audit of NG77 boundary loss and coverage.

This is posthoc diagnosis, not a new quality trial, training or causal estimate.
All candidate sets contain the complete three endpoint heads and all positives.
"""

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
from scipy import sparse

import analyze_ng66_learning_curves as metrics
import ng71_execution as execution
import ng71_pilot as pipeline
import ng71_preflight as io
import ng77_endpoints as endpoint
import review_ng78_boundaries as independent
from review_ng75_displacement import compare, read, require, sha, verify as verify_files
from review_ng77_training import rows


PARENT = 'NG-0077/train-endpoints-v1'
MODELS = ('initial', 'Z-96', 'K-96')
SURFACES = ('TRAIN_PILOT', 'TRAIN_SENTINEL')
INVENTORY = 'dd8e23dc862f57ef0e54779c4e89df170973a928525ec7c258469b08c2f19454'
LOCAL_REVIEW = 'ffc57b0da6b74230a445cb2571976d78ebc384309c546607d237f5d775d2229e'
LINEAGE = '8ca39510880622470e7423f389a34d5dd5b06c68b504d0c47b6a7f5ec467b794'
METHOD = dict(protocol='NG78_TRAIN_boundary_coverage_v1', models=list(MODELS),
              queries=768, documents=233009, cutoffs=[10, 100],
              training_updates=0, new_model_inference=False, dev_access=False,
              locked_test_access=False, stage_limit_seconds=1800)


def discount(rank):
    require(type(rank) is int and rank > 0, 'rank must be positive integer')
    return 1 / math.log2(rank + 1) if rank <= 10 else 0.


def teacher_status(positive, rival, scores, judged=False):
    if judged:
        return 'judged_negative'
    if scores.get(positive) is None or scores.get(rival) is None:
        return 'unobserved'
    margin = scores[positive] - scores[rival]
    require(math.isfinite(margin), 'nonfinite teacher margin')
    return 'agrees' if margin > 0 else 'opposes' if margin < 0 else 'tie'


def validate_scope(queries):
    require(len(queries) == 768 and len({q['query_id'] for q in queries}) == 768,
            'exact unique TRAIN identities required')
    require(all(q['split'] == 'TRAIN' and q['surface'] in SURFACES for q in queries),
            'non-TRAIN query forbidden')
    require(Counter((q['surface'], q['subset']) for q in queries)
            == Counter({(s, d): 128 for s in SURFACES for d in metrics.DOMAINS}),
            'TRAIN domain balance changed')


def universe(heads, gold, pool):
    require(gold and len(set(gold)) == len(gold), 'all positives required')
    require(set(heads) == set(MODELS), 'all three endpoint heads required')
    require(all(len(h) == len(set(h)) == 100 for h in heads.values()), 'incomplete endpoint heads')
    return sorted(set(gold).union(pool or (), *heads.values()))


def check_scores(query, docs, lexical_query, lexical_docs, ids, head, expected):
    primary = (query @ docs.T + lexical_query @ lexical_docs.T).toarray().ravel()
    alternate = np.asarray(docs.multiply(query).sum(1)
                           + lexical_docs.multiply(lexical_query).sum(1)).ravel()
    require(np.isfinite(primary).all() and np.isfinite(alternate).all(), 'nonfinite score')
    np.testing.assert_allclose(primary, alternate, rtol=0, atol=1e-12)
    order = np.lexsort((ids, -primary))
    require([ids[j] for j in order[:100]] == head, 'complete head parity failed')
    np.testing.assert_allclose(primary[order[:100]], expected, rtol=0, atol=1e-12)
    return primary, float(np.max(abs(primary - alternate)))


def reductions(raw):
    """Use actual full-corpus gold ranks, never ranks within the audit union."""
    ids, gold = raw['document_ids'], raw['gold_ids']
    require(ids == sorted(set(ids)) and set(gold).issubset(ids), 'invalid audit universe')
    require(gold and len(set(gold)) == len(gold), 'all positives required')
    require(all(len(raw['gold_ranks'][m]) == len(gold) for m in MODELS), 'gold rank count differs')
    score = {m: dict(zip(ids, raw['scores'][m], strict=True)) for m in MODELS}
    require(all(math.isfinite(v) for table in score.values() for v in table.values()), 'nonfinite score')
    ahead = lambda m, p, n: (score[m][p] > score[m][n]
                              or score[m][p] == score[m][n] and p < n)
    pool = None if raw['pool'] is None else set(raw['pool'])
    anchors = {tuple(a) for a in raw['anchors']}
    teacher = dict(zip(ids, raw['teacher_scores'], strict=True))
    judged = set(raw['judged_negative_ids'])
    require(not judged.intersection(gold), 'positive mislabeled as negative')
    norm = sum(discount(r) for r in range(1, min(10, len(gold)) + 1))
    results = []
    for j, p in enumerate(gold):
        rank = {m: raw['gold_ranks'][m][j] for m in MODELS}
        dcg = {m: discount(rank[m]) / norm for m in MODELS}
        recall = {m: float(rank[m] <= 100) / len(gold) for m in MODELS}
        boundary, counts, categories = {}, {}, {}
        for cutoff in (10, 100):
            key = str(cutoff)
            sets = {m: set() for m in ('Z-96', 'K-96')}
            for m in sets:
                sets[m] = {n for n in raw['heads'][m][:cutoff] if n not in gold
                           and ahead('initial', p, n) and not ahead(m, p, n)}
            boundary[key] = {m: sorted(v) for m, v in sets.items()}
            counts[key] = {}
            categories[key] = {}
            for state, ns in (('persistent', sets['Z-96'] & sets['K-96']),
                              ('repaired', sets['Z-96'] - sets['K-96']),
                              ('new', sets['K-96'] - sets['Z-96'])):
                counts[key][state] = len(ns)
                cells = Counter()
                for n in ns:
                    coverage = ('not_trained' if pool is None else 'trusted_anchor' if (p, n) in anchors
                                else 'in_pool_not_anchor' if n in pool else 'outside_pool')
                    status = teacher_status(p, n, teacher, n in judged)
                    cells[coverage + '/' + status] += 1
                categories[key][state] = dict(cells)
        strict = {m: [n for n in ids if (p, n) in anchors and score[m][p] <= score[m][n]]
                  for m in ('Z-96', 'K-96')}
        z, k = map(set, strict.values())
        # Strict margin loss and ID-tie-broken ranking loss are different facts.
        strict_transition = dict(persistent=len(z & k), repaired=len(z - k), new=len(k - z))
        loss_buckets = {}
        for m in ('Z-96', 'K-96'):
            ns = set(boundary['10'][m])
            if pool is None:
                bucket = 'not_trained'
            elif not ns:
                bucket = 'no_nongold_boundary_crossing'
            elif not ns.issubset(pool):
                bucket = 'some_outside_pool'
            elif all((p, n) in anchors for n in ns):
                bucket = 'all_trusted'
            else:
                bucket = 'all_in_pool_some_untrusted'
            loss_buckets[m] = bucket
        results.append(dict(query_id=raw['query_id'], domain=raw['domain'], surface=raw['surface'],
            positive_id=p, origin=raw['origins'][j], ranks=rank, dcg_contribution=dcg,
            recall_contribution=recall, boundary_losses=boundary, transition_counts=counts,
            coverage_teacher_counts=categories, strict_anchor_loss=strict_transition,
            ndcg_loss_buckets=loss_buckets))
    return results


def summarize(positives):
    result = {}
    for surface in SURFACES:
        for domain in metrics.DOMAINS:
            group = [r for r in positives if r['surface'] == surface and r['domain'] == domain]
            require(len({r['query_id'] for r in group}) == 128, 'summary query denominator changed')
            comparisons = {}
            for model, reference in (('Z-96', 'initial'), ('K-96', 'initial'), ('K-96', 'Z-96')):
                differences = {field: sum(r[field][model] - r[field][reference] for r in group) / 128
                               for field in ('dcg_contribution', 'recall_contribution')}
                losses = defaultdict(lambda: dict(positives=0, dcg_loss=0.))
                for r in group if reference == 'initial' else ():
                    loss = r['dcg_contribution'][reference] - r['dcg_contribution'][model]
                    if loss > 0:
                        key = r['ndcg_loss_buckets'][model] + '/' + r['origin']
                        losses[key]['positives'] += 1
                        losses[key]['dcg_loss'] += loss / 128
                comparisons[model + '-minus-' + reference] = dict(
                    ndcg10=differences['dcg_contribution'], recall100=differences['recall_contribution'],
                    gross_positive_dcg_loss_partition=dict(losses))
            transitions = {}
            for cutoff in ('10', '100'):
                transitions[cutoff] = {}
                for state in ('persistent', 'repaired', 'new'):
                    cells, harmed = Counter(), Counter()
                    for r in group:
                        cells.update(r['coverage_teacher_counts'][cutoff][state])
                        models = ('Z-96',) if state == 'repaired' else ('K-96',)
                        field = 'dcg_contribution' if cutoff == '10' else 'recall_contribution'
                        if any(r[field][m] < r[field]['initial'] for m in models):
                            harmed.update(r['coverage_teacher_counts'][cutoff][state])
                    transitions[cutoff][state] = dict(pairs=sum(cells.values()),
                        coverage_teacher=dict(cells), on_harmed_positives=dict(harmed),
                        harm_reference='initial', harm_endpoint='Z-96' if state == 'repaired' else 'K-96')
            strict = {s: sum(r['strict_anchor_loss'][s] for r in group)
                      for s in ('persistent', 'repaired', 'new')}
            result[surface + '/' + domain] = dict(queries=128, positives=len(group),
                origins=dict(Counter(r['origin'] for r in group)), comparisons=comparisons,
                boundary_transitions=transitions, strict_anchor_transitions=strict)
    return result


def verify(base, run):
    manifest = read(run / 'inputs.json')
    require(manifest['method'] == METHOD, 'diagnostic contract changed')
    verify_files(base, manifest['dependencies'])
    verify_files(run, manifest['source'])
    validate_scope(read(base / PARENT / 'endpoint-queries.json'))
    return manifest


def freeze(args):
    base, run = args.research_root, args.run
    require(run.parent == base / 'NG-0078', 'fresh NG78 unit required')
    parent = base / PARENT
    inv, receipt = [parent.with_name(parent.name + suffix) for suffix in
                    ('-remote-inventory.json', '-local-review.json')]
    require(sha(inv) == INVENTORY and sha(receipt) == LOCAL_REVIEW, 'verified NG77 parent changed')
    require(read(receipt)['passed'] and not read(receipt)['decision']['exploratory_gate_passed'],
            'unexpected parent scientific decision')
    manifest = endpoint.verify(base, parent)
    verify_files(parent, read(inv)['files'], exact=True)
    for name in endpoint.PHASES:
        execution.sealed(parent / name)
    dependencies = dict(manifest['dependencies'])
    dependencies.update({PARENT + '/' + n: h for n, h in read(inv)['files'].items()})
    lineage_path = base / 'NG-0072/cross-codes-v1/inputs.json'
    require(sha(lineage_path) == LINEAGE, 'historical lineage manifest changed')
    lineage = read(lineage_path)['dependencies']
    for name in ('NG-0070/provenance-v1/pair-provenance.json',
                 'NG-0069/teacher-v1/targets.json', 'NG-0067/data/documents.jsonl'):
        require(sha(base / name) == lineage[name], 'historical lineage input changed')
        require(name not in dependencies or dependencies[name] == lineage[name], 'parent hash conflict')
        dependencies[name] = lineage[name]
    for p in (inv, receipt, lineage_path):
        dependencies[str(p.relative_to(base))] = sha(p)
    root = Path(__file__).resolve().parent
    sources = [root / n for n in manifest['source'] if n.endswith('.py') and not n.startswith('test_')]
    sources += [Path(__file__), root / 'review_ng78_boundaries.py', root.parent / 'tests/test_ng78_boundaries.py',
                root.parent / 'docs/research-sae/reports/ng0001-ng0099/ng0078-boundary-coverage-plan.zh.md']
    run.mkdir(parents=True, exist_ok=False)
    for p in sources:
        shutil.copy2(p, run / p.name)
    io.write(run / 'inputs.json', dict(method=METHOD, dependencies=dependencies,
                                      source={p.name: sha(p) for p in sources}))
    verify(base, run)
    print('NG78_FROZEN', sha(run / 'inputs.json'), flush=True)


def audit(base, run, task):
    parent, output = base / PARENT, run / 'audit'
    queries = read(parent / 'endpoint-queries.json')
    validate_scope(queries)
    ranks = {m: rows(parent / ('rank-' + m) / 'rankings.jsonl') for m in MODELS}
    require(all(len(r) == len(queries) for r in ranks.values()), 'rank query count differs')
    labels = read(base / 'NG-0069/lexical-v1/labels.json')
    witnesses = {r['query_id']: r for r in rows(base / 'NG-0071/pilot-step0-v1/witnesses.jsonl')}
    anchors = read(base / 'NG-0077/preparation-v1/audit/anchors.json')
    teachers = {r['query_id']: r for r in read(base / 'NG-0069/teacher-v1/targets.json')}
    provenance = {r['query_id']: r for r in read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    wanted = {p for q in queries for p in labels[q['lexical_index']]}
    document_keys = {}
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        for i, line in enumerate(stream):
            if i in wanted:
                document_keys[i] = json.loads(line)['key']
    require(i + 1 == METHOD['documents'] and set(document_keys) == wanted, 'document universe differs')
    lexical = base / 'NG-0069/lexical-v1'
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz').astype(np.float64) for r in ('query', 'document')]
    codes = {}
    for model in MODELS:
        folder = (base / 'NG-0069/evaluation-v2/encode-initial' if model == 'initial'
                  else parent / ('encode-' + model))
        codes[model] = tuple(sparse.load_npz(folder / f'{r}.npz') for r in ('query', 'document'))
        cq, cd = codes[model]
        require(cq.shape[0] == (7872 if model == 'initial' else 768)
                and cd.shape[0] == METHOD['documents'] and cq.shape[1] == cd.shape[1], 'code shape differs')
        require(all(c.has_canonical_format and np.isfinite(c.data).all() for c in (cq, cd)),
                'codes must be finite canonical CSR')
        if model != 'initial':
            require(read(folder / 'query-ids.json') == [q['query_id'] for q in queries], 'code identity differs')
    started, max_error, positive_rows = time.monotonic(), 0., []
    with (output / 'queries.jsonl').open('x') as stream:
        for i, q in enumerate(queries):
            identity, index = q['query_id'], q['lexical_index']
            gold = labels[index]
            records = {m: ranks[m][i] for m in MODELS}
            for record in records.values():
                require(record['query_id'] == identity and record['split'] == 'TRAIN'
                        and record['domain'] == q['subset'] and record['surface'] == q['surface'],
                        'rank identity differs')
                metrics.audit_row(record, gold, 233009)
            w = witnesses.get(identity)
            require((w is not None) == (q['surface'] == 'TRAIN_PILOT'), 'training pool surface differs')
            pool = w['pool'] if w else None
            heads = {m: r['top100'] for m, r in records.items()}
            ids = universe(heads, gold, pool)
            scores = {}
            for model in MODELS:
                sq, sd = codes[model]
                query = sq.getrow(index if model == 'initial' else i).astype(np.float64)
                docs = sd[ids].astype(np.float64)
                primary, error = check_scores(query, docs, lq.getrow(index), ld[ids], ids,
                                              heads[model], records[model]['top100_scores'])
                max_error = max(max_error, error)
                scores[model] = primary.tolist()
            if w:
                locations = {d: j for j, d in enumerate(ids)}
                for a in anchors[identity]['anchors']:
                    margin = (scores['initial'][locations[a['positive_id']]]
                              - scores['initial'][locations[a['rival_id']]])
                    compare(margin, a['baseline_margin'])
                    require(margin > 0, 'anchor was not initially strictly correct')
            source = provenance[identity]
            require(source['status'] == 'resolved', 'positive provenance unresolved')
            origins = {p['document_key']: p['provenance'] for p in source['pairs']}
            require(set(origins) == {document_keys[p] for p in gold}, 'positive lineage differs')
            teacher_record = w or teachers[identity]
            t = dict(zip(teacher_record['pool'], teacher_record.get('teacher_scores', teacher_record.get('scores')), strict=True))
            original = teachers[identity]
            original_teacher = dict(zip(original['pool'], original['scores'], strict=True))
            raw = dict(query_id=identity, domain=q['subset'], surface=q['surface'], split='TRAIN',
                document_ids=ids, gold_ids=gold, origins=[origins[document_keys[p]] for p in gold],
                gold_ranks={m: r['gold_ranks'] for m, r in records.items()}, heads=heads, scores=scores,
                pool=pool, anchors=[(a['positive_id'], a['rival_id']) for a in anchors[identity]['anchors']] if w else [],
                teacher_scores=[t.get(d) for d in ids], teacher_source='step0' if w else 'original',
                original_teacher_scores=[original_teacher.get(d) for d in ids],
                positive_visibility=w['positive_visibility'] if w else None,
                judged_negative_ids=[d for d, flag in zip(w['pool'], w['judged_negative_mask'], strict=True) if flag] if w else [],
                supporting_evidence_visible='unknown_no_span_judgments')
            derived = reductions(raw)
            for model in MODELS:
                compare(sum(r['dcg_contribution'][model] for r in derived), records[model]['ndcg10'])
                compare(sum(r['recall_contribution'][model] for r in derived), records[model]['recall100'])
            positive_rows.extend(derived)
            stream.write(json.dumps(raw, allow_nan=False) + '\n')
            if i == 15:
                predicted = (time.monotonic() - started) * 768 / 16 * 1.5
                io.write(output / 'canary.json', dict(predicted_seconds=predicted, passed=predicted < 1200))
                require(predicted < 1200, 'bounded analysis canary failed')
            if i % 64 == 0:
                stream.flush()
                print('NG78_QUERY', i + 1, 768, flush=True)
                task.get_logger().report_scalar('progress', 'queries', i + 1, i)
    with (output / 'positives.jsonl').open('x') as stream:
        for r in positive_rows:
            stream.write(json.dumps(r, allow_nan=False) + '\n')
    summary = summarize(positive_rows)
    require(len(positive_rows) == 1371, 'all-positive count differs')
    parent_review = read(parent / 'review/results.json')
    for s in SURFACES:
        for d in metrics.DOMAINS:
            for c, values in summary[s + '/' + d]['comparisons'].items():
                compare({k: values[k] for k in ('ndcg10', 'recall100')},
                        parent_review['comparisons'][s][c]['domains'][d])
    for d in metrics.DOMAINS:
        counts = summary['TRAIN_PILOT/' + d]['strict_anchor_transitions']
        for model, state in (('Z-96', 'repaired'), ('K-96', 'new')):
            compare(counts['persistent'] + counts[state],
                    parent_review['trusted_transfer']['domains'][model][d]['previously_correct_order_lost'])
    return dict(passed=True, summary=summary, queries=768, positives=len(positive_rows),
        max_dual_score_error=max_error, all_endpoint_head_parity=True,
        full_corpus_reranked=False, new_model_inference=False, training_updates=0,
        dev_access=False, locked_test_access=False, native_cost_evaluated=False,
        independent_holdout=False, parameter_causality_proven=False,
        automatic_training_authorized=False)


def worker(args):
    verify(args.research_root, args.run)
    require(sha(Path(__file__)) == read(args.run / 'inputs.json')['source'][Path(__file__).name],
            'worker differs from frozen source')
    from clearml import Task
    Task.set_offline(True)
    task = Task.init(project_name='Evoke-NG', task_name='NG-0078/boundary-coverage-v1',
                     reuse_last_task_id=False, auto_connect_frameworks=False,
                     auto_connect_arg_parser=False, auto_connect_streams=False,
                     auto_resource_monitoring=False)
    output = args.run / 'audit'
    receipt = dict(task_id=task.id, actual_start=True, offline=True, remote_synced=False, closed=False)
    io.write(output / 'clearml-start.json', receipt)
    try:
        task.connect(dict(method=METHOD, input_sha256=sha(args.run / 'inputs.json')))
        result = audit(args.research_root, args.run, task)
        result['independent_scalar_review'] = independent.review(args.run)
        verify(args.research_root, args.run)
        io.write(output / 'results.json', result)
        receipt['outcome'] = 'passed'
    except BaseException:
        receipt['outcome'] = 'failed'
        raise
    finally:
        task.close()
        receipt['closed'] = True
        io.write(output / 'clearml.json', receipt)


def supervise(args):
    verify(args.research_root, args.run)
    require(sha(Path(__file__)) == read(args.run / 'inputs.json')['source'][Path(__file__).name],
            'controller differs from frozen source')
    args.worker_script, args.gpu = Path(__file__).name, None
    stopped = []
    previous = {s: signal.signal(s, lambda signum, frame: stopped.append(signum))
                for s in (signal.SIGINT, signal.SIGTERM)}
    io.write(args.run / 'controller-start.json', dict(pid=os.getpid(), gpu=None, started_unix=time.time()))
    try:
        pipeline.phase(args, 'audit', stopped, cuda=False, limit_seconds=1800)
        verify(args.research_root, args.run)
        execution.sealed(args.run / 'audit')
        io.write(args.run / 'controller-exit.json', dict(status='all_phases_complete', finished_unix=time.time()))
    except BaseException as exc:
        io.write(args.run / 'controller-exit.json', dict(status='failed_preserve_attempt',
                 error=f'{type(exc).__name__}: {exc}', finished_unix=time.time()))
        raise
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'verify', 'supervise', 'worker'))
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--phase', choices=('audit',))
    args = parser.parse_args()
    if args.mode == 'verify':
        print('NG78_VERIFIED', len(verify(args.research_root, args.run)['dependencies']))
    else:
        globals()[args.mode](args)
