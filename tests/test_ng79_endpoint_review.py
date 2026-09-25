from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng79_endpoints as review


def comparisons():
    item = dict(macro=dict(ndcg10=.01, recall100=0.), ci95=dict(ndcg10=[.005, .015], recall100=[0., 0.]),
                domains={d: dict(ndcg10=.01, recall100=0.) for d in ('fever', 'hotpotqa', 'nq')})
    return {'B-384-minus-R-384': deepcopy(item), 'B-384-minus-initial': deepcopy(item)}


def test_pass_is_not_holdout_or_native_cost_qualification():
    result = review.advancement(comparisons(), 1., 1., review.protocol.GATES)
    assert result['exploratory_gate_passed']
    assert not any(result[k] for k in ('automatic_scale_authorized', 'independent_holdout',
                                      'overall_goal_qualified', 'native_cost_evaluated'))


@pytest.mark.parametrize('bad', ['point', 'ndcg_ci', 'recall_ci', 'nq_ndcg', 'hotpot_recall',
                               'nnz', 'df', 'relaxed'])
def test_each_original_quality_and_cost_gate_is_binding(bad):
    c, gates, nnz, df = comparisons(), deepcopy(review.protocol.GATES), 1., 1.
    br, bi = c['B-384-minus-R-384'], c['B-384-minus-initial']
    if bad == 'point': br['macro']['ndcg10'] = .00499
    elif bad == 'ndcg_ci': br['ci95']['ndcg10'][0] = 0.
    elif bad == 'recall_ci': br['ci95']['recall100'][0] = -.00201
    elif bad == 'nq_ndcg': bi['domains']['nq']['ndcg10'] = -.00501
    elif bad == 'hotpot_recall': br['domains']['hotpotqa']['recall100'] = -.00501
    elif bad == 'nnz': nnz = 1.250001
    elif bad == 'df': df = 1.250001
    else:
        gates['ndcg_point_min'] = 0.
        with pytest.raises(ValueError): review.advancement(c, nnz, df, gates)
        return
    assert not review.advancement(c, nnz, df, gates)['exploratory_gate_passed']


def test_bootstrap_is_domain_macro_and_uses_frozen79079_seed():
    domains = ['fever'] * 2 + ['hotpotqa'] * 3 + ['nq'] * 4
    delta = [[.01, 0.]] * 2 + [[.02, -.001]] * 3 + [[.03, .001]] * 4
    result = review.paired(delta, domains)
    assert result['macro']['ndcg10'] == pytest.approx(.02)
    assert result['ci95']['ndcg10'] == pytest.approx([.02, .02])
    assert result['bootstrap_seed'] == 79079 and result['bootstrap_replicates'] == 10000
    assert result == review.paired(delta, domains)
    with pytest.raises(ValueError): review.paired(delta[:5], domains[:5])


def test_coordinate_head_reduction_agrees_with_sparse_dot():
    q = sparse.csr_matrix([[.2, 0., .6]])
    d = sparse.csr_matrix([[.3, .4, .1], [.9, .1, 0.], [0., .3, .8]])
    ids = [2, 0]
    np.testing.assert_allclose(review.coordinates(q, d, ids), (q @ d[ids].T).toarray().ravel(),
                               rtol=0, atol=1e-12)
