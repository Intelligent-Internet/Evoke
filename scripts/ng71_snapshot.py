"""TRAIN-only step-zero witnesses from verified full-corpus NG69 artifacts.

No model updates, new DEV inference, or candidate-local ranks are permitted.
This is the P0 diagnostic, not the four-arm scientific training controller.
"""

import hashlib
import importlib.util
import json
import time

import numpy as np
from scipy import sparse

import ng71_data as data
import ng71_ranking as ranking
import ng71_witness as witness


REVIEW_SHA = 'ae0604ec64a0abd517f8f392f7effc71587500e9d2d23ab14d7e88cb469d8ae2'
RUN = 'NG-0069/evaluation-v2'


def read(path):
    return json.loads(path.read_text())


def dependencies(base, config):
    """Bind inherited inputs, completion receipts and actual encoded payloads."""
    review_path = base / 'NG-0069/paired-review-v1.json'
    if data.provenance.sha(review_path) != REVIEW_SHA:
        raise ValueError('NG69 independent review changed')
    review = read(review_path)
    if (not review['breadth_gate']['passed'] or review['locked_test_scored']
            or review['all_query_records_audited'] != 70848):
        raise ValueError('NG69 review is not eligible for this next-stage diagnostic')
    run = base / RUN
    spec = importlib.util.spec_from_file_location('ng71_frozen_ng69', run / 'ng69_pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    frozen = pipeline.verify(base, run)
    if (pipeline.sha(run / 'inputs.json') != review['input_sha256']
            or frozen['source_sha256'] != review['pipeline_source_sha256']
            or read(run / 'continuation-state.json')['status'] != 'all_phases_complete'):
        raise ValueError('review does not bind completed NG69 run')
    files = dict(frozen['files'])
    files.update(data.ANCHORS)
    controls = ['inputs.json', 'ng69_pipeline.py', 'continuation-state.json',
                'protocol.md', 'research-protocol.md', 'continuation-protocol.md']
    for name in controls:
        files[RUN + '/' + name] = pipeline.sha(run / name)
    for phase in ('encode-initial', 'encode-dense', 'rank-initial', 'rank-dense', 'rank-bm25'):
        result = pipeline.completed(run, phase)
        if phase == 'encode-initial' and result['model_sha256'] != config['base']['state_sha256']:
            raise ValueError('snapshot does not use common NG3 initialization')
        names = set(read(run / phase / 'complete.json')['files'])
        names.update(('complete.json', 'exit.json', 'clearml.json'))
        for name in names:
            files[RUN + '/' + phase + '/' + name] = pipeline.sha(run / phase / name)
    files['NG-0069/paired-review-v1.json'] = REVIEW_SHA
    return files


def verify_scores(primary, alternate, count, selected=()):
    """Check every score, independently selected head and count-based ranks."""
    a, b = np.asarray(primary), np.asarray(alternate)
    if (a.shape != (count,) or b.shape != (count,)
            or a.dtype != np.float64 or b.dtype != np.float64
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError('complete finite float64 corpus vectors required')
    error = float(np.max(np.abs(a - b)))
    if error > 1e-12:
        raise ValueError('independent full-corpus score error exceeds 1e-12')
    ids = np.arange(count)
    order = np.lexsort((ids, -a))
    size = min(120, count)
    threshold = np.partition(b, count - size)[count - size]
    eligible = np.flatnonzero(b >= threshold)
    other_head = eligible[np.lexsort((eligible, -b[eligible]))][:size]
    if not np.array_equal(order[:size], other_head):
        raise ValueError('independent head/boundary ordering changed')
    ranks = np.empty(count, dtype=np.int64)
    ranks[order] = np.arange(1, count + 1)
    for doc in selected:
        if type(doc) is not int or not 0 <= doc < count:
            raise ValueError('selected ID outside corpus')
        independent = 1 + np.count_nonzero(b > b[doc])
        independent += np.count_nonzero((b == b[doc]) & (ids < doc))
        if ranks[doc] != independent:
            raise ValueError('independent selected/positive rank changed')
    return error


def supervision(record, config):
    """Measure uncertain supervision against actual threats, not just easy pairs."""
    pool = record['pool']
    positions = {doc: i for i, doc in enumerate(pool)}
    positives = [i for i, flag in enumerate(record['positive_mask']) if flag]
    scores, ranks = record['teacher_scores'], record['global_ranks']
    sources = {'original_pool': [d for d in record['original_pool']
                                 if not record['positive_mask'][positions[d]]],
               **record['sources']}
    result = {}
    params = {k: config[k] for k in (
        'ndcg_cutoff', 'recall_cutoff', 'recall_weight', 'pair_floor')}
    for name, docs in sources.items():
        targets, supervised = [], set()
        totals = dict(pairs=0, eligible_pairs=0, masked_pairs=0,
                      priority_mass=0., masked_priority_mass=0.,
                      cross_top10_mass=0., masked_cross_top10_mass=0.,
                      cross_top100_mass=0., masked_cross_top100_mass=0.)
        for p in positives:
            for doc in docs:
                n = positions[doc]
                if record['positive_mask'][n]:
                    raise ValueError('positive used as negative witness')
                margin = (scores[p] - scores[n]) / config['teacher_temperature']
                target = float(np.exp(-np.logaddexp(0., -margin)))
                weight = ranking.pair_weight(ranks[p], ranks[n], len(positives), **params)
                eligible = target > .5
                totals['pairs'] += 1
                totals['eligible_pairs' if eligible else 'masked_pairs'] += 1
                totals['priority_mass'] += weight
                if eligible:
                    supervised.add(p)
                    targets.append(target)
                else:
                    totals['masked_priority_mass'] += weight
                for cutoff in (10, 100):
                    if (ranks[p] <= cutoff) != (ranks[n] <= cutoff):
                        totals[f'cross_top{cutoff}_mass'] += weight
                        if not eligible:
                            totals[f'masked_cross_top{cutoff}_mass'] += weight
        result[name] = {
            **totals, 'documents': len(docs),
            'supervised_positive_positions': sorted(supervised),
            'eligible_targets': targets,
            'confidence_sum': float(sum(2 * t - 1 for t in targets)),
            'target_above_099': sum(t > .99 for t in targets),
        }
    return result


def summarize(records, domains):
    result = {}
    for domain in domains:
        rows = [r for r in records if r['domain'] == domain]
        if not rows:
            raise ValueError('missing TRAIN domain')
        positives = sum(r['total_positives'] for r in rows)
        supervised = sum(len(set().union(*(
            set(s['supervised_positive_positions']) for s in r['supervision'].values())))
                         for r in rows)
        sources = {}
        for name in rows[0]['supervision']:
            pieces = [r['supervision'][name] for r in rows]
            keys = [k for k in pieces[0]
                    if k not in ('supervised_positive_positions', 'eligible_targets')]
            summary = {k: sum(s[k] for s in pieces) for k in keys}
            targets = [t for s in pieces for t in s['eligible_targets']]
            summary['target_quantiles'] = (
                np.quantile(targets, [0, .25, .5, .75, 1]).tolist() if targets else [])
            summary['supervised_positives'] = sum(
                len(s['supervised_positive_positions']) for s in pieces)
            sources[name] = summary
        result[domain] = dict(queries=len(rows), positives=positives,
                              supervised_positives=supervised,
                              eligible_positive_fraction=supervised / positives,
                              sources=sources)
    return result


def reference_inputs(reference, selected):
    from ng71_execution import sealed

    result = sealed(reference)
    identities = read(reference / 'query-ids.json')
    if (result['reference_update'] != 96 or result['reference_arm'] != 'A'
            or len(identities) != len(selected)
            or len(set(identities)) != len(identities)
            or set(identities) != set(selected)):
        raise ValueError('only the complete shared A96 TRAIN reference is permitted')
    return result, {query_id: i for i, query_id in enumerate(identities)}


def build(base, config, output, *, surface='canary', reference=None):
    prepared = data.audit(base, config)
    return build_prepared(base, config, output, prepared, surface=surface, reference=reference)


def build_prepared(base, config, output, prepared, *, surface='pilot', reference=None):
    """Reuse scoring for a caller-verified frozen TRAIN selection.

    The caller binds selection/provenance hashes. Corpus/teacher identity,
    TRAIN bounds, all positives and independent scores remain checked here.
    """
    if not prepared['eligibility_gate_passed']:
        raise ValueError('existing TRAIN eligibility gate failed')
    if surface not in ('canary', 'pilot'):
        raise ValueError('only TRAIN canary or pilot snapshots are permitted')
    selected = prepared['selection'][surface]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('unique nonempty TRAIN selection required')
    from transformers import AutoTokenizer

    full_pilot = surface == 'pilot'
    if reference is not None and not full_pilot:
        raise ValueError('A96 references require the full TRAIN pilot')
    lexical = base / 'NG-0069/lexical-v1'
    queries, labels = read(lexical / 'queries.json'), read(lexical / 'labels.json')
    pools = read(lexical / 'train-pools.json')
    indices = {q['query_id']: i for i, q in enumerate(queries)}
    expected_count = config['pilot']['train_queries' if full_pilot else 'preflight_train_queries']
    if (len(selected) != expected_count
            or any(indices[q] >= 6144 or queries[indices[q]]['split'] != 'TRAIN' for q in selected)):
        raise ValueError('only the fixed TRAIN selection may be mined')
    count = config['pilot']['corpus_documents']
    stable_ids = [f'{i:09d}' for i in range(count)]
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz').astype(np.float64)
              for r in ('query', 'document')]
    semantic_path = reference if reference is not None else base / RUN / 'encode-initial'
    reference_result, semantic_indices = (reference_inputs(reference, selected)
                                           if reference is not None else (None, indices))
    reference_update = 96 if reference_result else 0
    state_sha = reference_result['model_sha256'] if reference_result else config['base']['state_sha256']
    sq, sd = [sparse.load_npz(semantic_path / f'{r}.npz').astype(np.float64)
              for r in ('query', 'document')]
    dq, dd = [np.load(base / RUN / 'encode-dense' / f'{r}.npy', mmap_mode='r')
              for r in ('query', 'document')]
    if (ld.shape[0] != count or sd.shape[0] != count or dd.shape != (count, 1024)
            or any(q.shape[0] != 7872 for q in (lq, dq))
            or sq.shape[0] != (len(selected) if reference is not None else 7872)):
        raise ValueError('full corpus/query encoding inventory changed')
    li, si = ld.T.tocsr(), sd.T.tocsr()
    dd = dd.astype(np.float64)
    with (base / 'NG-0067/data/documents.jsonl').open() as stream:
        documents = [json.loads(line) for line in stream]
    if len(documents) != count:
        raise ValueError('canonical document inventory changed')
    targets = read(base / 'NG-0069/teacher-v1/targets.json')
    lineage = {r['query_id']: r for r in read(base / 'NG-0070/provenance-v1/pair-provenance.json')}
    tokenizer = AutoTokenizer.from_pretrained(base / config['base']['tokenizer'], local_files_only=True)
    records, errors = [], []
    started = time.monotonic()
    with (output / 'witnesses.jsonl').open('x') as stream:
        for number, query_id in enumerate(selected, 1):
            i = indices[query_id]
            si_query = semantic_indices[query_id]
            bm25 = (lq.getrow(i) @ li).toarray().ravel()
            bm25_other = ld @ lq.getrow(i).toarray().ravel()
            hybrid = bm25 + (sq.getrow(si_query) @ si).toarray().ravel()
            hybrid_other = bm25_other + sd @ sq.getrow(si_query).toarray().ravel()
            vector = dq[i].astype(np.float64)
            dense = dd @ vector
            dense_other = np.einsum('ij,j->i', dd, vector, optimize=False)
            target = targets[i]
            if target['query_id'] != query_id or target['pool'] != pools[i]:
                raise ValueError('teacher TRAIN identity or pool changed')
            if np.max(np.abs(dense[pools[i]] - target['scores'])) > 1e-12:
                raise ValueError('full-corpus teacher differs from frozen TRAIN targets')
            scores = dict(hybrid=hybrid, pplx=dense, bm25=bm25)
            alternate = dict(hybrid=hybrid_other, pplx=dense_other, bm25=bm25_other)
            record = witness.mine(query_id, pools[i], labels[i], stable_ids,
                                  scores, config['witness'], corpus_size=count)
            for name in scores:
                errors.append(verify_scores(scores[name], alternate[name], count, record['pool']))
            record.update(
                domain=queries[i]['subset'], split='TRAIN', reference_update=reference_update,
                checkpoint_state_sha256=state_sha,
                query_sha256=hashlib.sha256(queries[i]['query'].encode()).hexdigest(),
                document_text_sha256=[hashlib.sha256(documents[d]['text'].encode()).hexdigest()
                                      for d in record['pool']],
                lexical_scores=[float(bm25[d]) for d in record['pool']],
                hybrid_scores=[float(hybrid[d]) for d in record['pool']],
                independent_score_verification_required=False,
                independent_score_and_rank_verification_passed=True)
            origins = {p['document_key']: p['provenance'] for p in lineage[query_id]['pairs']}
            visibility = []
            for doc in labels[i]:
                text = documents[doc]['text']
                full = tokenizer(text, truncation=False)['input_ids']
                visible = tokenizer(text, truncation=True, max_length=256, return_offsets_mapping=True)
                visibility.append(dict(
                    document_id=doc, origin=origins[documents[doc]['key']],
                    total_tokens=len(full), input_tokens=len(visible['input_ids']),
                    truncated=len(full) > 256,
                    visible_character_end=max(end for _, end in visible['offset_mapping']),
                    supporting_evidence_visible='unknown_no_span_judgments'))
            record['positive_visibility'] = visibility
            record['supervision'] = supervision(record, config['ranking'])
            if full_pilot:
                import ng71_diagnostics as diagnostics

                record['arm_score_diagnostics'] = diagnostics.diagnose(record, config)
                if 'A' in record['arm_score_diagnostics']:
                    diagnostics.assert_close(
                        record['arm_score_diagnostics']['A']['target'], target['target'])
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            records.append(record)
            elapsed = time.monotonic() - started
            print('NG71_P0_WITNESSES', number, len(selected), elapsed, flush=True)
            if number == 16:
                prediction = elapsed * len(selected) / number * 1.5
                if prediction > 5100:
                    raise ValueError('TRAIN witness canary predicts more than 5100 seconds')
    summary = summarize(records, config['pilot']['domains'])
    eligible = all(r['eligible_positive_fraction'] >=
                   config['gates']['eligible_positive_fraction_min_per_domain']
                   for r in summary.values())
    full = {}
    if full_pilot:
        full['fixed_batch_audit'] = diagnostics.batch_audit(
            records, prepared['query_order'], config)
        full['score_space_diagnostics'] = diagnostics.summarize(records, config)
        full['score_space_only_not_parameter_gradients'] = True
        full['snapshot_scores_not_fresh_training_forwards'] = True
        with (output / 'training-order.json').open('x') as stream:
            json.dump(prepared['query_order'], stream)
            stream.write('\n')
    return dict(
        passed=True, stage=f'NG71_TRAIN_step{reference_update}_witness_diagnostic',
        queries=len(records),
        selection_surface=surface, **full,
        corpus_documents=count, summaries=summary, eligibility_gate_passed=eligible,
        max_independent_score_error=max(errors), global_rank_verification=True,
        tie_break='frozen_corpus_integer_document_id',
        reused_encodings=('verified_A96_and_PPLX_no_new_model_inference'
                          if reference_result else 'verified_NG69_initial_and_PPLX_no_new_model_inference'),
        reference_complete_sha256=data.provenance.sha(reference / 'complete.json') if reference_result else None,
        checkpoint_state_sha256=state_sha,
        actual_optimizer_updates=0, reference_update=reference_update,
        positive_visibility_unknown=sum(len(r['positive_visibility']) for r in records),
        positive_documents_truncated=sum(v['truncated'] for r in records
                                         for v in r['positive_visibility']),
        visibility_does_not_establish_evidence_absence=True,
        cuda_model_preflight_passed=False, full_pilot_manifest_prepared=full_pilot,
        scientific_training_enabled=False, locked_test_access=False,
        seconds=time.monotonic() - started)
