"""Independent NG80 scalar/exposure and serialized optimizer audit.

No inference or parameter VJP is recomputed. Worker reload/VJP evidence is
checked separately; serialized moment contents are independently hashed.
"""

import hashlib
import json

import numpy as np

import ng80_campaign as campaign
import ng80_fit as training
from ng80_data import require


INPUTS = {
    'F': 'c6f6e0bd612dde796bb34cabb8aade176b3dffd2b9998fa68058676783f0a942',
    'D': 'b8fd35a599bbad9080317907b9131333d44473e778c1e3f5f43abcc88940c656',
}


def sealed(path):
    from ng71_execution import sealed as original

    result = original(path)
    start = campaign.read(path / 'started.json')
    end = campaign.read(path / 'exit.json')
    track = campaign.read(path / 'clearml.json')
    require(track['outcome'] == 'passed' and start['limit_seconds'] == 5400
            and 0 < end['elapsed_seconds'] < 5400
            and end['peak_tree_rss_bytes'] <= 16 * 2 ** 30
            and end['finished_unix'] > start['started_unix'], 'unsafe closure')
    return result


def scalar(record, teacher, positive):
    scores = np.asarray(record['scores'], dtype=np.float64)
    target = np.asarray(teacher, dtype=np.float64)
    require(scores.shape == target.shape == (record['documents'],)
            and len(scores) > 1 and np.isfinite(scores).all()
            and record['vjp_replay_exact'] is True
            and record['positive'] == positive, 'changed training example')
    np.testing.assert_allclose(record['teacher'], target, rtol=0, atol=1e-12)
    error = scores - target.astype(np.float32).astype(np.float64)
    error -= error.mean()
    loss = float(error @ error / len(error))
    gradient = error / (2 * len(error))  # 2 / N, accumulated over four queries.
    np.testing.assert_allclose(record['loss'], loss, rtol=3e-5, atol=2e-8)
    np.testing.assert_allclose(record['score_gradient'], gradient,
                               rtol=3e-5, atol=2e-8)
    return float(np.max(np.abs(record['score_gradient'] - gradient)))


def optimizer_audit(path, step, expected):
    import torch

    saved = torch.load(path, weights_only=True, map_location='cpu')
    names, state = saved['names'], saved['state']
    require(saved['step'] == step and saved['fingerprint'] == expected
            and len(names) == len(state['param_groups']), 'optimizer header')
    slots = {}
    for group_names, group in zip(names, state['param_groups'], strict=True):
        require(len(group_names) == len(group['params']), 'optimizer layout')
        for name, index in zip(group_names, group['params'], strict=True):
            require(name not in slots, 'duplicate optimizer parameter')
            slots[name] = index
    require(len(set(slots.values())) == len(slots)
            and set(slots.values()) == set(state['state']), 'optimizer coverage')
    digest = hashlib.sha256(json.dumps(
        dict(names=names, groups=state['param_groups']), sort_keys=True).encode())
    # ng80_fit.holder uses a ParameterDict initialized from a plain dict.
    for name in sorted(slots):
        values = state['state'][slots[name]]
        require(set(values) == {'step', 'exp_avg', 'exp_avg_sq'}
                and float(values['step']) == step, 'moment step/inventory')
        require(values['exp_avg'].shape == values['exp_avg_sq'].shape
                and values['exp_avg'].dtype == values['exp_avg_sq'].dtype
                == torch.float32 and (values['exp_avg_sq'] >= 0).all(),
                'moment shape/precision')
        digest.update(name.encode())
        for key, raw in sorted(values.items()):
            value = raw.detach().cpu().contiguous()
            require(torch.isfinite(value).all(), 'nonfinite moment')
            digest.update(key.encode())
            digest.update(value.numpy().tobytes())
    require(digest.hexdigest() == expected, 'serialized moment fingerprint')
    return len(slots)


def review(base):
    parent = base / training.PARENT
    queries = campaign.read(parent / 'prepare/queries.json')
    selection = campaign.read(parent / 'prepare/selection.json')
    panels = campaign.read(parent / 'prepare/panels.json')
    labels = campaign.read(parent / 'prepare/labels.json')
    query_codes = np.load(parent / 'encode-teacher/query.npy', mmap_mode='r')
    doc_codes = np.load(base / campaign.DENSE, mmap_mode='r')
    lengths = {role: np.minimum(np.load(parent / 'encode-hidden' /
               f'{role}-token-lengths.npy'), limit)
               for role, limit in (('query', 64), ('document', 256))}
    order = training.order(queries, selection)
    output, starts = {}, []
    for arm, expected in INPUTS.items():
        run = base / f'NG-0080/dense-{arm}-v1'
        require(campaign.io.sha(run / 'inputs.json') == expected, 'training input')
        training.verify(base, run)
        controller = campaign.read(run / 'controller-exit.json')
        require(controller['status'] == 'all_phases_complete'
                and controller['completed'] == list(training.PHASES)
                and campaign.read(run / 'order.json') == order, 'training graph')
        canary = sealed(run / 'train-canary')
        require(canary['vjp_relative_l2'] <= 1e-5 and canary['real_reload_exact']
                and canary['canary_updates_discarded'], 'canary replay')
        starts.append((canary['initial_model_sha256'],
                       canary['initial_projection_sha256']))
        previous, total, peak_error, moments = None, 0, 0., {}
        tokens, candidates = dict(query=0, document=0), 0
        for end in (1024, 2048, 3072):
            root = run / f'train-{end}'
            result = sealed(root)
            require(result['arm'] == arm and result['updates'] == end
                    and result['start_update'] == end - 1024
                    and result['query_exposures'] == 4096
                    and result['real_reload_exact']
                    and not any(result[k] for k in ('canary_updates_discarded',
                        'locked_test_access', 'quality_evaluated',
                        'native_cost_evaluated')), 'chunk contract')
            initial = (result['initial_model_sha256'],
                       result['initial_projection_sha256'])
            expected_initial = ((previous['model_sha256'],
                                 previous['projection_sha256'])
                                if previous else starts[-1])
            require(initial == expected_initial, 'checkpoint continuation')
            require((result['model_sha256'] == initial[0]) == (arm == 'F')
                    and result['projection_sha256'] != initial[1], 'trainable scope')
            moments[str(end)] = optimizer_audit(root / 'optimizer.pt', end,
                                                result['optimizer_sha256'])
            observed, chunk_tokens, chunk_candidates, losses = [], dict(query=0, document=0), 0, []
            for offset, line in enumerate((root / 'updates.jsonl').open()):
                row = json.loads(line)
                step = end - 1024 + offset
                indices = order[4 * step:4 * (step + 1)]
                require(offset < 1024 and row['update'] == step + 1
                        and [r['query_id'] for r in row['queries']]
                        == [queries[i]['query_id'] for i in indices]
                        and len(indices) == 4 and np.isfinite(row['gradient_norm'])
                        and row['gradient_norm'] > 0, 'update identity/count')
                for index, record in zip(indices, row['queries'], strict=True):
                    pool = panels[index]
                    teacher = doc_codes[pool].astype(np.float64) @ query_codes[index].astype(np.float64)
                    peak_error = max(peak_error, scalar(record, teacher,
                        [i in labels[index] for i in pool]))
                    require(record['documents'] == len(pool), 'panel cardinality')
                    chunk_tokens['query'] += int(lengths['query'][index])
                    chunk_tokens['document'] += int(lengths['document'][pool].sum())
                    chunk_candidates += len(pool)
                np.testing.assert_allclose(row['loss'], np.mean([
                    r['loss'] for r in row['queries']]), rtol=0, atol=1e-12)
                losses.append(row['loss'])
                observed.extend(indices)
            require(observed == order[4 * (end - 1024):4 * end]
                    and chunk_tokens == result['token_exposures']
                    and chunk_candidates == result['candidate_exposures'], 'exposure totals')
            np.testing.assert_allclose(result['mean_loss'], np.mean(losses), rtol=0, atol=1e-12)
            for role in tokens:
                tokens[role] += chunk_tokens[role]
            candidates += chunk_candidates
            total += len(observed)
            previous = result
        output[arm] = dict(updates=3072, query_exposures=total,
            unique_queries=len(set(order)), candidate_exposures=candidates,
            token_exposures=tokens, max_scalar_gradient_error=peak_error,
            serialized_optimizer_parameters=moments,
            terminal_model_sha256=previous['model_sha256'],
            terminal_projection_sha256=previous['projection_sha256'])
    require(starts[0] == starts[1]
            and output['F']['token_exposures'] == output['D']['token_exposures']
            and output['F']['candidate_exposures'] == output['D']['candidate_exposures'],
            'matched initialization/exposure failed')
    return dict(passed=True, arms=output, optimizer_tensors_independently_hashed=True,
        checkpoint_continuation_worker_reload_verified=True,
        actual_parameter_vjp_recomputed=False, independent_optimizer_replay=False,
        inference_performed=False, independent_holdout=False, locked_test_access=False)
