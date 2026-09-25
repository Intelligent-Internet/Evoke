"""Synthetic retention algebra, unknown-label boundaries and shared VJP checks."""

from decimal import Decimal, localcontext
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_training as original
import ng77_retention as keep


def decimal_gap(a, a0):
    with localcontext() as ctx:
        ctx.prec = 100
        a, a0 = Decimal(a), Decimal(a0)
        if a >= a0:
            return 0., 0.
        def sp(x):
            return max(x, 0) + (1 + (-abs(x)).exp()).ln()
        r = 1 / (1 + a0.exp())
        gap = sp(-a) - sp(-a0) - r * (a0 - a)
        derivative = r - 1 / (1 + a.exp())
        return float(gap), float(derivative)


@pytest.mark.parametrize('a0', [1e-8, .1, 1., 10., 20., 100., 1000.])
@pytest.mark.parametrize('delta', [-1., 0., 1e-12, 1e-7, .0009999, .0010001, .1, 2., 2000.])
def test_gap_and_derivative_against_independent_decimal(a0, delta):
    a = a0 - delta
    x = torch.tensor(a, dtype=torch.float64, requires_grad=True)
    reference = torch.tensor(a0, dtype=torch.float64, requires_grad=True)
    actual = keep.bernoulli_gap(x, reference)
    actual.backward()
    expected, derivative = decimal_gap(a, a0)
    assert actual >= 0 and reference.grad is None
    assert float(actual.detach()) == pytest.approx(expected, rel=2e-8, abs=1e-18)
    assert float(x.grad) == pytest.approx(derivative, rel=2e-8, abs=1e-14)


def fixture():
    record = dict(pool=[4, 5, 6, 7, 8], positive_ids=[4, 5],
        positive_mask=[True, True, False, False, False],
        judged_negative_mask=[False] * 5, teacher_scores=[.9, .05, .2, .1, .3],
        global_ranks=[1, 120, 10, 90, 130], hybrid_scores=[5., 1., 4., 3., 0.],
        total_positives=2, corpus_size=200, rank_scope='full_corpus_reference_snapshot')
    config = dict(ndcg_cutoff=10, recall_cutoff=100, recall_weight=1.,
                  pair_floor=.05, teacher_temperature=.04, student_temperature=1.)
    return record, config


def test_all_positive_denominator_not_only_covered_positives():
    r, c = fixture()
    a = keep.prepare_anchors(r, c)
    assert a['covered_positives'] == 1 and a['total_positives'] == 2
    assert len(a['anchors']) == 3
    assert sum(x['coefficient'] for x in a['anchors']) <= .5
    assert all(x['positive_id'] == 4 for x in a['anchors'])


@pytest.mark.parametrize('teacher,status', [(None, 'unobserved'), (.9, 'tie'), (1., 'opposes')])
def test_unknown_tied_opposed_teacher_not_implicit_negative(teacher, status):
    assert keep.preference(.9, teacher, False, .04) == (status, 0.)
    assert keep.preference(.9, teacher, True, .04) == ('judged_negative', 1.)


@pytest.mark.parametrize('bad', ['positive', 'mask', 'ranks', 'pool', 'local', 'nonfinite'])
def test_corrupt_input_cannot_make_anchors(bad):
    r, c = fixture()
    if bad == 'positive':
        r['total_positives'] = 1
    elif bad == 'mask':
        r['judged_negative_mask'][0] = True
    elif bad == 'ranks':
        r['global_ranks'][0] = 0
    elif bad == 'pool':
        r['pool'][0] = r['pool'][1]
    elif bad == 'local':
        r['rank_scope'] = 'candidate_local'
    else:
        r['hybrid_scores'][0] = float('nan')
    with pytest.raises(ValueError):
        keep.prepare_anchors(r, c)


def test_improved_zero_boundary_zero_and_no_false_retention_of_wrong_order():
    r, c = fixture()
    r['hybrid_scores'][0] = 3.5
    a = keep.prepare_anchors(r, c)
    assert {x['rival_id'] for x in a['anchors']} == {7, 8}
    for delta in (0., .1):
        scores = torch.tensor(r['hybrid_scores'], dtype=torch.float64, requires_grad=True)
        scores = scores + torch.tensor([delta, 0., 0., 0., 0.])
        scores.retain_grad()
        loss = keep.retention_loss(scores, a)
        loss.backward()
        assert loss == 0 and torch.count_nonzero(scores.grad) == 0


def test_zero_coefficient_bit_exact_original_fp32_loss_and_score_gradient():
    r, c = fixture()
    anchors = keep.prepare_anchors(r, c)
    pairs = original.prepare_pairs(r, c)
    scores = torch.tensor([2., 3., 4., 1., 5.], requires_grad=True)
    baseline = original.pair_loss(scores, pairs)
    expected = torch.autograd.grad(baseline, scores, retain_graph=True)[0]
    total, penalty = keep.combined_loss(baseline, scores, anchors, 0.)
    actual = torch.autograd.grad(total, scores)[0]
    assert penalty > 0 and total is baseline and total.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_empty_anchors_keep_zero_connected_gradient():
    r, c = fixture()
    r['teacher_scores'] = [None] * 5
    a = keep.prepare_anchors(r, c)
    scores = torch.tensor(r['hybrid_scores'], requires_grad=True)
    keep.retention_loss(scores, a).backward()
    assert not a['anchors'] and torch.equal(scores.grad, torch.zeros_like(scores))


def test_shared_document_vjp_and_four_query_accumulation():
    r, c = fixture()
    a = keep.prepare_anchors(r, c)
    pairs = original.prepare_pairs(r, c)
    generator = torch.Generator().manual_seed(77)
    inputs = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    queries = torch.randn(4, 3, generator=generator, dtype=torch.float64)
    w0 = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    gradients = []
    for replay in (False, True):
        w = w0.clone().requires_grad_()
        for query in queries:
            documents = inputs @ w
            leaf = documents.detach().requires_grad_() if replay else documents
            scores = leaf @ (query @ w)
            loss, _ = keep.combined_loss(original.pair_loss(scores, pairs), scores, a, 1.)
            (loss / len(queries)).backward()
            if replay:
                (inputs @ w).backward(leaf.grad)
        gradients.append(w.grad)
    torch.testing.assert_close(*gradients, rtol=1e-13, atol=1e-13)
    # Detaching without replay is not a frozen shared document encoder.
    w = w0.clone().requires_grad_()
    for query in queries:
        scores = (inputs @ w).detach() @ (query @ w)
        loss, _ = keep.combined_loss(original.pair_loss(scores, pairs), scores, a, 1.)
        (loss / len(queries)).backward()
    assert not torch.allclose(w.grad, gradients[0])


@pytest.mark.parametrize('coefficient', [True, -1., float('nan')])
def test_invalid_coefficient(coefficient):
    r, c = fixture()
    scores = torch.tensor(r['hybrid_scores'], requires_grad=True)
    with pytest.raises(ValueError):
        keep.combined_loss(scores.sum(), scores, keep.prepare_anchors(r, c), coefficient)


def test_fp32_cast_has_finite_promoting_gradient():
    r, c = fixture()
    scores = torch.tensor([2., 1., 4., 3., 0.], requires_grad=True)
    loss = keep.retention_loss(scores, keep.prepare_anchors(r, c))
    loss.backward()
    assert loss.dtype == torch.float64 and scores.grad.dtype == torch.float32
    assert scores.grad[0] < 0 and torch.isfinite(scores.grad).all()
    assert abs(float(scores.grad.sum())) < 1e-7
