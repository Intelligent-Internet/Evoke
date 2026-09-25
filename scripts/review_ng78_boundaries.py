"""Independent scalar reduction of sealed-input NG78 query witnesses.

Does not import the primary reducer, infer a model, or recalculate corpus ranks.
"""

from collections import Counter, defaultdict
import json
import math

from review_ng75_displacement import compare, require


def reference(raw):
    ids, gold = raw['document_ids'], raw['gold_ids']
    models = ('initial', 'Z-96', 'K-96')
    lookup = {d: i for i, d in enumerate(ids)}
    values = {m: {d: raw['scores'][m][lookup[d]] for d in ids} for m in models}
    teacher = {d: raw['teacher_scores'][lookup[d]] for d in ids}
    pool, anchors = raw['pool'], [tuple(a) for a in raw['anchors']]
    ideal = math.fsum(1 / math.log2(r + 1) for r in range(1, min(10, len(gold)) + 1))
    result = []
    for j, positive in enumerate(gold):
        ranks = {m: raw['gold_ranks'][m][j] for m in models}
        losses = {c: {m: [] for m in models[1:]} for c in ('10', '100')}
        counts, cells = {}, {}
        strict = Counter(persistent=0, repaired=0, new=0)
        for p, rival in anchors:
            if p != positive:
                continue
            bad_z = values['Z-96'][p] - values['Z-96'][rival] <= 0
            bad_k = values['K-96'][p] - values['K-96'][rival] <= 0
            if bad_z or bad_k:
                strict['persistent' if bad_z and bad_k else 'repaired' if bad_z else 'new'] += 1
        for cutoff in ('10', '100'):
            counts[cutoff] = dict(persistent=0, repaired=0, new=0)
            cells[cutoff] = {s: Counter() for s in counts[cutoff]}
            for rival in ids:
                if rival in gold:
                    continue
                before = (-values['initial'][positive], positive) < (-values['initial'][rival], rival)
                bad = []
                for model in models[1:]:
                    after = (-values[model][rival], rival) < (-values[model][positive], positive)
                    lost = before and after and rival in raw['heads'][model][:int(cutoff)]
                    bad.append(lost)
                    if lost:
                        losses[cutoff][model].append(rival)
                if not any(bad):
                    continue
                state = 'persistent' if all(bad) else 'repaired' if bad[0] else 'new'
                counts[cutoff][state] += 1
                if rival in raw['judged_negative_ids']:
                    status = 'judged_negative'
                elif teacher[positive] is None or teacher[rival] is None:
                    status = 'unobserved'
                else:
                    delta = teacher[positive] - teacher[rival]
                    status = 'agrees' if delta > 0 else 'opposes' if delta < 0 else 'tie'
                coverage = ('not_trained' if pool is None else 'trusted_anchor'
                            if (positive, rival) in anchors else 'in_pool_not_anchor'
                            if rival in pool else 'outside_pool')
                cells[cutoff][state][coverage + '/' + status] += 1
        buckets = {}
        for model in models[1:]:
            rivals = losses['10'][model]
            buckets[model] = ('not_trained' if pool is None else 'no_nongold_boundary_crossing'
                             if not rivals else 'some_outside_pool' if any(n not in pool for n in rivals)
                             else 'all_trusted' if all((positive, n) in anchors for n in rivals)
                             else 'all_in_pool_some_untrusted')
        result.append(dict(query_id=raw['query_id'], domain=raw['domain'], surface=raw['surface'],
            positive_id=positive, origin=raw['origins'][j], ranks=ranks,
            dcg_contribution={m: 1 / math.log2(r + 1) / ideal if r <= 10 else 0. for m, r in ranks.items()},
            recall_contribution={m: 1 / len(gold) if r <= 100 else 0. for m, r in ranks.items()},
            boundary_losses=losses, transition_counts=counts, coverage_teacher_counts=cells,
            strict_anchor_loss=dict(strict), ndcg_loss_buckets=buckets))
    return result


def review(run):
    expected, observed = [], []
    with (run / 'audit/queries.jsonl').open() as stream:
        for line in stream:
            raw = json.loads(line)
            require(raw['split'] == 'TRAIN', 'non-TRAIN witness forbidden')
            expected.extend(reference(raw))
    with (run / 'audit/positives.jsonl').open() as stream:
        observed = [json.loads(line) for line in stream]
    require(len(expected) == len(observed) == 1371, 'incomplete positive reduction')
    compare(expected, observed)
    groups = defaultdict(lambda: defaultdict(list))
    for row in expected:
        for model in ('initial', 'Z-96', 'K-96'):
            groups[row['surface'] + '/' + row['domain']][model].append(
                (row['dcg_contribution'][model], row['recall_contribution'][model]))
    return dict(passed=True, independent_scalar_positive_reduction=True, positives=len(expected),
                quality={s: {m: dict(ndcg10=math.fsum(x[0] for x in v) / 128,
                                     recall100=math.fsum(x[1] for x in v) / 128)
                             for m, v in g.items()} for s, g in groups.items()},
                new_model_inference=False, full_corpus_reranked=False,
                independent_rank_or_teacher_validation=False)
