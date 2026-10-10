"""Review sealed candidate diagnostics without promoting partial checkpoints."""

import argparse
from pathlib import Path
import shutil
import sys

import numpy as np


def paired_interval(values, domains):
    rng = np.random.default_rng(85017)
    groups = [np.asarray([v for v, d in zip(values, domains, strict=True) if d == domain])
              for domain in sorted(set(domains))]
    bootstrap = np.mean([g[rng.integers(len(g), size=(10000, len(g)))].mean(1)
                         for g in groups], axis=0)
    return np.quantile(bootstrap, [.025, .975]).tolist()


def summarize(records):
    result = {}
    for split in ('fit', 'validation'):
        reference = {r['query_index']: r for r in records
                     if r['checkpoint'] == 'base' and r['split'] == split}
        if len(reference) != 64:
            raise ValueError('incomplete base panel')
        result[split] = {}
        for label in ('base', 'A-1024', 'A-2048', 'B-1024', 'B-2048'):
            items = {r['query_index']: r for r in records
                     if r['checkpoint'] == label and r['split'] == split}
            if set(items) != set(reference):
                raise ValueError('unmatched panel')
            rows = [items[i] for i in sorted(items)]
            valid = [r for r in rows if r['valid']]
            if not valid:
                raise ValueError('all panel queries invalid')
            ndcg = [r.get('candidate_ndcg10', 0.) for r in rows]
            delta = [r.get('candidate_ndcg10', 0.)
                     - reference[r['query_index']].get('candidate_ndcg10', 0.) for r in rows]
            stats = dict(queries=len(rows), invalid=len(rows) - len(valid),
                candidate_ndcg10=float(np.mean(ndcg)),
                candidate_recall10=float(np.mean([r.get('candidate_recall10', 0.) for r in rows])),
                ndcg_delta_vs_base=float(np.mean(delta)),
                descriptive_paired_95=paired_interval(delta, [r['subset'] for r in rows]),
                document_nnz_valid_mean=float(np.mean([r['document_nnz_mean'] for r in valid])),
                query_nnz_valid_mean=float(np.mean([r['query_nnz'] for r in valid])),
                score_std_valid_median=float(np.median([r['score_std'] for r in valid])),
                denominator_valid_median=float(np.median([r['query_raw'][0]['denominator'] for r in valid])),
                unsupported_query_mass_valid_mean=float(np.mean([
                    r['query_raw'][0]['unsupported_mass'] / r['query_raw'][0]['raw_sum'] for r in valid])))
            stats['domains'] = {d: float(np.mean([r.get('candidate_ndcg10', 0.)
                for r in rows if r['subset'] == d])) for d in sorted({r['subset'] for r in rows})}
            if split == 'fit':
                stats['scale_pressure'] = {arm: dict(
                    mean=float(np.mean([r['geometry_' + arm]['detached_scale_derivative'] for r in valid])),
                    fraction_negative=float(np.mean([
                        r['geometry_' + arm]['detached_scale_derivative'] < -1e-9 for r in valid])))
                    for arm in ('A', 'B')}
            result[split][label] = stats
    return result


def teacher_reference(parent, manifest):
    from ng85_data import require, rows, sha
    from ng85_diagnose import candidate_metrics
    from ng85_run import closed

    require(sha(parent / 'inputs.json') == manifest['parent_artifacts']['inputs.json'],
            'teacher reference parent mismatch')
    for phase in ('prepare', 'bank', *(f'rankt5-{i}' for i in range(4))):
        require(sha(parent / phase / 'complete.json')
                == manifest['parent_artifacts'][phase + '/complete.json'], 'phase mismatch')
        closed(parent, phase)
    queries = rows(parent / 'prepare/queries.jsonl')
    with np.load(parent / 'bank/bank.npz') as archive:
        bank = {key: archive[key] for key in archive.files}
    selected = set(manifest['panel'])
    rankt5 = {}
    for shard in range(4):
        with np.load(parent / f'rankt5-{shard}/scores.npz') as values:
            for j, raw in enumerate(values['query_indices']):
                i = int(raw)
                if i in selected:
                    left, right = values['offsets'][j:j + 2]
                    rankt5[i] = values['scores'][left:right].copy()
    records = []
    for i in manifest['panel']:
        q = queries[i]
        left, right = bank['offsets'][i:i + 2]
        ids = bank['document_ids'][left:right]
        scores = dict(PPLX=bank['pplx'][left:right], BM25=bank['lexical'][left:right])
        if i in rankt5:
            scores['RankT5'] = rankt5[i]
        for label, vector in scores.items():
            records.append(dict(query_index=i, subset=q['subset'], split=q['split'],
                teacher=label, **candidate_metrics(ids, vector, q['positives'])))
    summary = {}
    for split in ('fit', 'validation'):
        summary[split] = {}
        for teacher in ('PPLX', 'BM25', 'RankT5'):
            subset = [r for r in records if r['split'] == split and r['teacher'] == teacher]
            if subset:
                require(len(subset) == 64, 'incomplete teacher panel')
                summary[split][teacher] = dict(queries=64,
                    candidate_ndcg10=float(np.mean([r['candidate_ndcg10'] for r in subset])),
                    candidate_recall10=float(np.mean([r['candidate_recall10'] for r in subset])))
    return dict(summary=summary, records=records, additional_model_inference=False,
                full_corpus=False, RankT5_validation='not_scored')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--teacher-parent', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run))
    from ng85_data import read, require, rows, sha, write
    from ng85_run import closed

    require(not args.out.exists(), 'immutable review output required')
    closed(args.run, 'profile')
    records = rows(args.run / 'profile/queries.jsonl')
    manifest = read(args.run / 'inputs.json')
    require(len(records) == 640 and not manifest['locked_test_access'], 'panel contract')
    result = dict(profile=summarize(records), candidate_only=True,
                  validation_independent=False, invalid_queries_score_zero=True,
                  full_corpus_evaluation=False, production_qualified=False)
    if args.teacher_parent is not None:
        result['teacher_reference'] = teacher_reference(args.teacher_parent, manifest)
    if (args.run / 'replay/complete.json').exists():
        closed(args.run, 'replay')
        result['replay'] = read(args.run / 'replay/results.json')
        trace = rows(args.run / 'replay/steps.jsonl')
        result['replay_exact_trace'] = all(
            all(v == 0. for v in r['trace_max_abs_errors'].values()) for r in trace)
        result['replayed_updates'] = len(trace)
    else:
        result['replay'] = 'pending'
    args.out.mkdir()
    shutil.copy2(__file__, args.out / Path(__file__).name)
    write(args.out / 'inputs.json', dict(
        source_sha256=sha(Path(__file__)), manifest_sha256=sha(args.run / 'inputs.json'),
        profile_complete_sha256=sha(args.run / 'profile/complete.json'),
        replay_complete_sha256=sha(args.run / 'replay/complete.json')
            if (args.run / 'replay/complete.json').exists() else None))
    write(args.out / 'results.json', result)
    print(result)


if __name__ == '__main__':
    main()
