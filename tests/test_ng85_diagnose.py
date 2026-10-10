"""CPU diagnostics contracts; Torch comparison is run on Lambda2 only."""

from pathlib import Path
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).parents[1] / 'scripts'
if not SOURCE.is_dir():
    SOURCE = Path(__file__).parent
sys.path.insert(0, str(SOURCE))
from ng85_diagnose import candidate_metrics, geometry, panel_indices  # noqa: E402


def test_full_derivative_removes_artificial_radial_component():
    s = np.array([.2, .5, 2., 3.])
    t = np.array([.01, .04, .15, .8])
    result = geometry(s, t)
    assert abs(result['detached_scale_derivative']) > .01
    assert abs(result['full_scale_derivative']) < 1e-12
    assert geometry(s * 100, t)['ce'] == pytest.approx(result['ce'])
    epsilon = 1e-5
    numerical = (geometry(s * (1 + epsilon), t)['ce']
                 - geometry(s * (1 - epsilon), t)['ce']) / (2 * epsilon)
    assert numerical == pytest.approx(result['full_scale_derivative'], abs=1e-8)


def test_equal_score_floor_and_bad_targets():
    result = geometry([1., 1.], [.2, .8])
    assert result['full_gradient_norm'] == result['detached_gradient_norm']
    assert result['ce'] == pytest.approx(np.log(2))
    with pytest.raises(ValueError):
        geometry([1., np.nan], [.2, .8])


def test_zero_rms_support_allows_unbounded_nonempty_query():
    rms = np.array([1., 0.])
    outputs = []
    for epsilon in (1e-2, 1e-6):
        raw = np.array([epsilon, 1.])
        outputs.append(.9 * raw / (raw @ rms))
    assert outputs[1][0] == pytest.approx(outputs[0][0])
    assert outputs[1][1] / outputs[0][1] == pytest.approx(10000.)
    assert np.array([0., 1.]) @ rms == 0.


def test_panel_is_balanced_deterministic_and_excludes_locked():
    queries = [dict(query_id=f'{split}/{domain}/{i}', split=split, subset=domain)
               for split in ('fit', 'validation', 'locked_test')
               for domain in ('fever', 'hotpotqa', 'nq', 'fiqa') for i in range(20)]
    chosen = panel_indices(queries)
    assert len(chosen) == len(set(chosen)) == 128
    assert chosen == panel_indices(queries)
    assert all(queries[i]['split'] != 'locked_test' for i in chosen)


def test_candidate_metrics_use_all_positive_denominator_and_id_ties():
    result = candidate_metrics([3, 2, 1], [1., 1., 1.], [1])
    assert result == dict(candidate_ndcg10=1., candidate_recall10=1.)
    with pytest.raises(ValueError):
        candidate_metrics([1, 2], [2., 1.], [3])


def test_drift_reference_preserves_failed_receipt_and_checks_payload(tmp_path):
    import hashlib
    import json
    from ng85_diagnose import PARENT_SHA, drift_reference

    def write(path, value):
        path.write_text(json.dumps(value))

    replay = tmp_path / 'replay'
    replay.mkdir()
    write(tmp_path / 'inputs.json', dict(scientific_changes=False,
        source={}, parent_artifacts={'inputs.json': PARENT_SHA}))
    write(replay / 'exit.json', dict(owned_group_closed=True, exit_code=1))
    trace = replay / 'steps.jsonl'
    trace.write_text(json.dumps(dict(step=2789, trace_matches=False)) + '\n')
    receipt = dict(passed=False, files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in replay.iterdir()})
    write(replay / 'complete.json', receipt)
    assert drift_reference(tmp_path)['first_mismatch_step'] == 2789
    assert not json.loads((replay / 'complete.json').read_text())['passed']
    trace.write_text('{}\n')
    with pytest.raises(ValueError, match='changed drift reference'):
        drift_reference(tmp_path)


@pytest.mark.skipif(sys.platform == 'darwin', reason='Torch checks run on isolated Linux only')
def test_geometry_matches_actual_autograd():
    import torch

    scores = torch.tensor([.2, .5, 2., 3.], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([.01, .04, .15, .8], dtype=torch.float64)
    expected = geometry(scores.detach().numpy(), target.numpy())
    for detached, key in ((True, 'detached'), (False, 'full')):
        scale = scores.std(correction=0)
        if detached:
            scale = scale.detach()
        loss = -(target * torch.log_softmax(scores / scale, 0)).sum()
        grad, = torch.autograd.grad(loss, scores)
        centered = scores - scores.mean()
        assert float((grad @ centered).detach()) == pytest.approx(
            expected[key + '_scale_derivative'], abs=1e-12)
        assert float(grad.norm()) == pytest.approx(expected[key + '_gradient_norm'], abs=1e-12)


@pytest.mark.skipif(sys.platform == 'darwin', reason='Torch checks run on isolated Linux only')
def test_observer_preserves_forward_gradient_and_records_zero_support():
    import torch
    from types import SimpleNamespace
    from ng85_diagnose import QueryObserver

    class Base(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            return input_ids.float()[..., None].expand(-1, -1, 4)

    class Encoder:
        def __init__(self):
            self.model = SimpleNamespace(base_model=Base(), lm_head=torch.nn.Linear(4, 6))
            with torch.no_grad():
                self.model.lm_head.weight.fill_(.1)
                self.model.lm_head.bias.fill_(.1)
            self.rms = torch.tensor([0., 0., 1., 1., 1., 1.])

        def encode(self, texts, role):
            tokens = dict(input_ids=torch.tensor([[1, 2, 0]]),
                          attention_mask=torch.tensor([[1, 1, 0]]))
            values = self.model.lm_head(self.model.base_model(**tokens)).relu().log1p()
            values = values.masked_fill(~tokens['attention_mask'].bool()[..., None], 0).amax(1)
            denominator = values @ self.rms
            if (denominator <= 0).any():
                raise ValueError('invalid fixed RMS normalization')
            return .9 * values / denominator[:, None]

    encoder = Encoder()
    before = encoder.encode(['test'], 'query')
    gradient, = torch.autograd.grad(before.square().sum(), encoder.model.lm_head.weight)
    observer = QueryObserver(encoder)
    after = encoder.encode(['test'], 'query')
    actual, = torch.autograd.grad(after.square().sum(), encoder.model.lm_head.weight)
    assert torch.equal(before, after) and torch.equal(gradient, actual)
    assert observer.events[-1]['supported_nnz'] == 4
    encoder.rms.zero_()
    with pytest.raises(ValueError, match='invalid fixed RMS'):
        encoder.encode(['test'], 'query')
    assert observer.events[-1]['supported_nnz'] == 0
    assert observer.events[-1]['unsupported_nnz'] == 6
    assert not encoder.model.lm_head._forward_hooks
    assert not encoder.model.base_model._forward_pre_hooks


@pytest.mark.skipif(sys.platform == 'darwin', reason='Torch checks run on isolated Linux only')
def test_intervention_changes_only_std_gradient_not_forward():
    import torch
    from ng85_train import calibrated_loss
    from ng85_diagnose import full_std_loss

    for values in ([.2, .5, 2., 3.], [1., 1., 1., 1.]):
        scores = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        target = np.array([.01, .04, .15, .8])
        original = calibrated_loss(scores, target)
        corrected = full_std_loss(scores, target)
        assert torch.equal(original, corrected)
        gradient, = torch.autograd.grad(corrected, scores)
        assert torch.isfinite(gradient).all()
        assert float(gradient @ (scores.detach() - scores.detach().mean())) == pytest.approx(0., abs=1e-12)


def test_review_preserves_invalid_queries_in_quality_denominator():
    from review_ng85_diagnose import summarize

    rows = []
    for split in ('fit', 'validation'):
        for label in ('base', 'A-1024', 'A-2048', 'B-1024', 'B-2048'):
            for i in range(64):
                rows.append(dict(checkpoint=label, split=split, query_index=i,
                    subset=('fever', 'hotpotqa', 'nq', 'fiqa')[i // 16],
                    valid=True, candidate_ndcg10=1., candidate_recall10=1.,
                    document_nnz_mean=2., query_nnz=1, score_std=1.,
                    query_raw=[dict(denominator=1., unsupported_mass=0., raw_sum=1.)],
                    geometry_A=dict(detached_scale_derivative=0.),
                    geometry_B=dict(detached_scale_derivative=-1.)))
    bad = next(r for r in rows if r['checkpoint'] == 'B-2048' and r['split'] == 'validation')
    bad.update(valid=False)
    del bad['candidate_ndcg10'], bad['candidate_recall10']
    report = summarize(rows)
    actual = report['validation']['B-2048']
    assert actual['invalid'] == 1
    assert actual['candidate_ndcg10'] == 63 / 64
    assert actual['ndcg_delta_vs_base'] == -1 / 64
