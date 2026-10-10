"""NG87 fixed panels, full-std dispatch and terminal-only evidence contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng87_run import CAPS, PHASES, panels  # noqa: E402
from ng87_train import corrected_objective  # noqa: E402
from review_ng87 import paired, summarize  # noqa: E402


def queries():
    return [dict(query_id=f'{split}/{domain}/{i}', split=split, subset=domain)
        for split in ('fit', 'validation', 'locked_test')
        for domain in ('fever', 'hotpotqa', 'nq', 'fiqa') for i in range(512)]


def test_endpoint_is_all_validation_with_fixed_nested_monitor():
    data = queries()
    monitor, endpoint = panels(data)
    assert len(monitor) == 640 and len(endpoint) == 2176
    assert set(monitor) <= set(endpoint)
    assert {i for i in endpoint if data[i]['split'] == 'validation'} == {
        i for i, q in enumerate(data) if q['split'] == 'validation'}
    assert not any(data[i]['split'] == 'locked_test' for i in endpoint)
    with pytest.raises(ValueError, match='validation inventory'):
        panels([q for q in data if q['query_id'] != 'validation/fever/0'])
    data[-1]['query_id'] = data[0]['query_id']
    with pytest.raises(ValueError, match='duplicate query'):
        panels(data)


def test_common_loss_selection_restores_even_after_failure():
    import ng85_train
    from ng85_diagnose import full_std_loss

    old = ng85_train.calibrated_loss
    with pytest.raises(RuntimeError):
        with corrected_objective():
            assert ng85_train.calibrated_loss is full_std_loss
            raise RuntimeError('fixture')
    assert ng85_train.calibrated_loss is old
    assert CAPS == {'preflight': 1800, 'train': 43200, 'profile': 3600}
    assert PHASES[:3] == ('preflight-A', 'preflight-B', 'profile-base')
    assert PHASES[-1] == 'profile-B'


def records():
    data = queries()
    monitor, endpoint = panels(data)
    output = []
    for label in ('base', 'A-2048', 'A-4096', 'A-8192', 'B-2048', 'B-4096', 'B-8192'):
        arm, step = ('base', 'base') if label == 'base' else label.split('-')
        for i in endpoint if step in ('base', '8192') else monitor:
            q = data[i]
            value = .5 if arm == 'base' else .52 if arm == 'A' else .54
            output.append(dict(arm=arm, checkpoint=step, query_index=i,
                subset=q['subset'], split=q['split'], valid=True, document_ids=[1, 2],
                candidate_ndcg10=value, candidate_recall10=.8,
                document_nnz_mean=100., query_nnz=10,
                query_raw=[dict(unsupported_mass=1., raw_sum=2.)],
                pplx=dict(candidate_ndcg10=.7, candidate_recall10=.9),
                bm25=dict(candidate_ndcg10=.3, candidate_recall10=.5)))
    return output


def test_fixed_endpoint_not_best_checkpoint_and_not_product_qualification():
    sample = records()
    for row in sample:
        if row['checkpoint'] == '2048':
            row['candidate_ndcg10'] = .99
    result = summarize(sample)
    assert result['metrics']['validation']['B-8192']['ndcg10'] == pytest.approx(.54)
    assert result['comparisons']['B-8192_minus_A-8192']['candidate_ndcg10']['delta'] == pytest.approx(.02)
    assert result['decisions']['B']['candidate_quality_signal']
    assert not result['production_qualified']
    assert not result['full_corpus_evaluation']


def test_invalid_queries_stay_in_denominator_and_cost_is_not_ignored():
    sample = records()
    sample[-1].update(valid=False, candidate_ndcg10=0., candidate_recall10=0.)
    for row in sample:
        if row['arm'] == 'B':
            row['document_nnz_mean'] = 300.
    result = summarize(sample)
    assert result['metrics']['validation']['B-8192']['invalid'] == 1
    assert result['metrics']['validation']['B-8192']['ndcg10'] == pytest.approx(.54 * 2047 / 2048)
    assert not result['decisions']['B']['candidate_quality_signal']
    assert not result['decisions']['B']['candidate_cost_screen_pass']
    sample[-1]['candidate_ndcg10'] = .54
    with pytest.raises(ValueError, match='denominator'):
        summarize(sample)


def test_review_rejects_missing_duplicate_or_misaligned_rows():
    sample = records()
    with pytest.raises(ValueError):
        summarize(sample[:-1])
    sample[-1]['query_index'] = sample[-2]['query_index']
    with pytest.raises(ValueError):
        summarize(sample)
    a = {i: row for i, row in enumerate(records()[:2])}
    with pytest.raises(ValueError, match='identity mismatch'):
        paired(a, {}, 'candidate_ndcg10', 1)
    sample = records()
    sample[-1]['document_ids'] = [2, 1]
    with pytest.raises(ValueError, match='candidate mismatch'):
        summarize(sample)


@pytest.mark.skipif(sys.platform == 'darwin', reason='Torch execution on Lambda2 only')
def test_update_dispatch_has_scale_tangent_gradient_not_detached_loss():
    import torch
    import ng85_train

    scores = torch.tensor([.2, .5, 2., 3.], requires_grad=True)
    target = np.array([.01, .04, .15, .8])
    before = ng85_train.calibrated_loss(scores, target)
    with corrected_objective():
        after = ng85_train.calibrated_loss(scores, target)
        assert torch.equal(before, after)
        gradient, = torch.autograd.grad(after, scores)
    assert float(gradient @ (scores.detach() - scores.detach().mean())) == pytest.approx(0., abs=2e-6)
