"""Full-background NG71 observations with explicit TRAIN/terminal surfaces."""

import json
import time

import numpy as np
from scipy import sparse

import ng71_execution as execution
import ng71_preflight as io
import ng71_snapshot as snapshot


TRAIN_PHASES = [f'train-{arm}-{step}' for arm in 'ABCD' for step in (96, 192)]


def training_closed(run, phases=None):
    results = {}
    selected = TRAIN_PHASES if phases is None else phases
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('training barrier must contain unique nonempty phases')
    for name in selected:
        result = execution.sealed(run / name)
        _, arm, step = name.split('-')
        if (result['arm'] != arm or result['cumulative_steps'] != int(step)
                or result['steps'] != 96 or result['query_exposures'] != 384):
            raise ValueError('terminal barrier has an incorrect training chunk')
        results[name] = result
    return results


def surface_queries(base, selection, terminal, *, include_dev=True):
    if type(terminal) is not bool or type(include_dev) is not bool:
        raise ValueError('observation boundary flags must be explicit booleans')
    rows = execution.read(base / 'NG-0069/lexical-v1/queries.json')
    if len(rows) != 7872 or len({q['query_id'] for q in rows}) != len(rows):
        raise ValueError('frozen lexical query universe changed')
    by_id = {q['query_id']: (i, q) for i, q in enumerate(rows)}
    wanted = [(q, 'TRAIN_PILOT') for q in selection['pilot']]
    if terminal:
        wanted += [(q, 'TRAIN_SENTINEL') for q in selection['sentinel']]
        if include_dev:
            wanted += [(q['query_id'], 'DEV_NEW') for q in rows if q['split'] == 'DEV_NEW']
    expected = (2304 if include_dev else 768) if terminal else 384
    if len(wanted) != expected or len({q for q, _ in wanted}) != expected:
        raise ValueError('observation surface count or identity overlap changed')
    result = []
    for identity, surface in wanted:
        index, row = by_id[identity]
        if row['split'] != ('DEV_NEW' if surface == 'DEV_NEW' else 'TRAIN'):
            raise ValueError('forbidden query split in observation surface')
        result.append(dict(row, surface=surface, lexical_index=index))
    return result


def rank_record(scores, alternate, gold):
    count = len(scores)
    if (not gold or len(set(gold)) != len(gold)
            or any(type(g) is not int or not 0 <= g < count for g in gold)):
        raise ValueError('incomplete or invalid positive IDs')
    error = snapshot.verify_scores(scores, alternate, count, gold)
    order = np.lexsort((np.arange(count), -scores))
    inverse = np.empty(count, dtype=np.int64)
    inverse[order] = np.arange(1, count + 1)
    ranks = inverse[gold]
    ideal = np.sum(1 / np.log2(np.arange(1, min(10, len(gold)) + 1) + 1))
    return dict(
        gold_ids=gold, gold_ranks=ranks.tolist(), top100=order[:100].tolist(),
        top100_scores=scores[order[:100]].tolist(),
        ndcg10=float(np.sum(1 / np.log2(ranks[ranks <= 10] + 1)) / ideal),
        recall100=float(np.mean(ranks <= 100))), inverse, error


def sparse_counts(query, document, queries):
    for matrix in (query, document):
        if (not matrix.has_canonical_format or not np.isfinite(matrix.data).all()
                or not (matrix.data > 0).all()):
            raise ValueError('invalid sparse observation payload')
    df = np.bincount(document.indices, minlength=document.shape[1])
    work = np.asarray(query.sign().astype(np.float64) @ df).ravel()
    fractions = df / document.shape[0]
    bins = [0., .0001, .001, .01, .1, 1.000001]
    result = dict(
        document_nnz=int(document.nnz), query_nnz=int(query.nnz),
        document_csr_bytes=sum(a.nbytes for a in
                               (document.data, document.indices, document.indptr)),
        df_fraction_bins=bins, df_coordinate_histogram=np.histogram(fractions, bins)[0].tolist(),
        high_df_over_01_postings=int(df[fractions > .1].sum()),
        query_df_by_surface={}, native_cost_evaluated=False)
    surfaces = np.array([q['surface'] for q in queries])
    for surface in sorted(set(surfaces)):
        values = work[surfaces == surface]
        result['query_df_by_surface'][str(surface)] = dict(
            mean=float(values.mean()), p95=float(np.quantile(values, .95)))
    return result


def rank(base, config, run, output, model, *, training_phases=None,
         witness_source=None, train_only=False, training_run=None,
         stage_limit_seconds=5400):
    # Cached DEV also waits for the caller's fixed terminal barrier (NG71: four arms).
    training_closed(run if training_run is None else training_run, training_phases)
    if type(stage_limit_seconds) is not int or not 300 < stage_limit_seconds <= 5400:
        raise ValueError('ranking stage budget must not relax existing bounds')
    if type(train_only) is not bool:
        raise ValueError('TRAIN-only observation flag must be boolean')
    terminal = train_only or not model.endswith('-96')
    queries = surface_queries(base, execution.read(run / 'selection.json'), terminal,
                              include_dev=not train_only)
    return _rank_surface(base, config, run, output, model, queries,
                         witness_source, stage_limit_seconds)


def rank_fixed_surface(base, config, run, output, model, queries, training_run,
                       *, witness_source, stage_limit_seconds=1800):
    """The separately frozen NG79 endpoint, never a generic split override."""
    phases = [f'train-{arm}-{step}' for step in (96, 192, 288, 384) for arm in ('R', 'B')]
    training_closed(training_run, phases)
    execution.validate_fixed_terminal_surface(queries)
    if (model not in ('initial', 'dense', 'bm25', 'R-384', 'B-384')
            or stage_limit_seconds != 1800 or type(stage_limit_seconds) is not int
            or config['pilot']['updates'] != 384 or set(config['arms']) != {'R', 'B'}
            or config['pilot']['corpus_documents'] != 233009):
        raise ValueError('unapproved fixed terminal ranking')
    return _rank_surface(base, config, run, output, model, queries,
                         witness_source, stage_limit_seconds)


def _rank_surface(base, config, run, output, model, queries, witness_source, stage_limit_seconds):
    """Shared unchanged dual-score full-background ranking and cost counters."""
    lexical = base / 'NG-0069/lexical-v1'
    labels = execution.read(lexical / 'labels.json')
    lq, ld = [sparse.load_npz(lexical / f'lexical-{r}.npz').astype(np.float64)
              for r in ('query', 'document')]
    indices = [q['lexical_index'] for q in queries]
    lq = lq[indices]
    li = ld.T.tocsr()
    count = config['pilot']['corpus_documents']
    if ld.shape[0] != count:
        raise ValueError('full corpus is required for every observation')
    counts, sq, sd, si = {}, None, None, None
    if model == 'dense':
        folder = base / snapshot.RUN / 'encode-dense'
        dq = np.load(folder / 'query.npy', mmap_mode='r')[indices].astype(np.float64)
        dd = np.load(folder / 'document.npy', mmap_mode='r').astype(np.float64)
        if dd.shape != (count, 1024):
            raise ValueError('dense background changed')
    elif model != 'bm25':
        folder = base / snapshot.RUN / 'encode-initial' if model == 'initial' else run / ('encode-' + model)
        if model != 'initial':
            execution.sealed(folder)
            if execution.read(folder / 'query-ids.json') != [q['query_id'] for q in queries]:
                raise ValueError('encoded query identity/order changed')
        sq, sd = [sparse.load_npz(folder / f'{r}.npz') for r in ('query', 'document')]
        if model == 'initial':
            sq = sq[indices]
        if sq.shape[0] != len(queries) or sd.shape[0] != count or sq.shape[1] != sd.shape[1]:
            raise ValueError('sparse corpus or query surface changed')
        counts = sparse_counts(sq, sd, queries)
        sq, sd = sq.astype(np.float64), sd.astype(np.float64)
        si = sd.T.tocsr()
    witnesses = {}
    if model not in ('initial', 'dense', 'bm25'):
        source = run / 'snapshot-96' if witness_source is None else witness_source
        with (source / 'witnesses.jsonl').open() as stream:
            witnesses = {r['query_id']: r for line in stream if (r := json.loads(line))}
    started, errors = time.monotonic(), []
    with (output / 'rankings.jsonl').open('x') as stream:
        for number, query in enumerate(queries):
            if model == 'dense':
                score = dd @ dq[number]
                other = np.einsum('ij,j->i', dd, dq[number], optimize=False)
            else:
                score = (lq.getrow(number) @ li).toarray().ravel()
                other = ld @ lq.getrow(number).toarray().ravel()
                if si is not None:
                    score += (sq.getrow(number) @ si).toarray().ravel()
                    other += sd @ sq.getrow(number).toarray().ravel()
            row, ranks, error = rank_record(score, other, labels[query['lexical_index']])
            errors.append(error)
            row.update(query_id=query['query_id'], domain=query['subset'],
                       split=query['split'], surface=query['surface'])
            witness = witnesses.get(query['query_id'])
            if witness:
                # These are observation ranks, never fed back into frozen training.
                snapshot.verify_scores(score, other, count, witness['pool'])
                pool = set(witness['pool'])
                row['reference_staleness'] = dict(
                    pool=witness['pool'], reference_ranks=witness['global_ranks'],
                    actual_ranks=ranks[witness['pool']].tolist(),
                    outside_reference_pool_top100=[d for d in row['top100'] if d not in pool])
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            if number % 32 == 0:
                stream.flush()
                print('NG71_RANK', model, number + 1, len(queries),
                      time.monotonic() - started, flush=True)
            if number == 15:
                prediction = (time.monotonic() - started) * len(queries) / 16 * 1.5
                io.write(output / 'canary.json', dict(
                    passed=prediction < stage_limit_seconds - 300, predicted_seconds=prediction))
                if prediction >= stage_limit_seconds - 300:
                    raise ValueError('full-background ranking exceeds stage budget')
    return dict(model=model, queries=len(queries), counts=counts,
                max_independent_score_error=max(errors),
                all_query_score_and_rank_checks_passed=True, locked_test_scored=False,
                native_cost_evaluated=False)
