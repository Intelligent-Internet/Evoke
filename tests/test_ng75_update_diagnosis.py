import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ng71_pilot as pipeline
import ng75_update_diagnosis as diagnosis


def fixture():
    sentinels = [f'{d}-{i}' for d in diagnosis.math_probe.DOMAINS for i in range(8)]
    batch = [f'fever-batch{i}' for i in range(4)]
    queries = {q: dict(query_id=q, query=q, split='TRAIN', subset=q.split('-')[0],
                       text_sha256=hashlib.sha256(q.encode()).hexdigest()) for q in sentinels + batch}
    cases = {}
    for name in diagnosis.CASES:
        _, arm, step = name.split('-')
        rows = [dict(query_id=q, split='TRAIN', domain=queries[q]['subset'],
                     surface='TRAIN_SENTINEL' if q in sentinels else 'TRAIN_CURRENT_BATCH',
                     positive_id=0, rival_id=1, origin='original_positive',
                     lexical_margin=0., rivals_are_judged_negative=False) for q in sentinels + batch]
        cases[name] = dict(arm=arm, start=int(step) - 1, batch_ids=batch,
            witnesses=[dict(query_id=q, split='TRAIN', pool=[0, 1]) for q in batch], probes=rows,
            expected=dict(step=int(step), reference_update=int(step) - 1,
                          examples=[dict(query_id=q) for q in batch]))
    return dict(cases=cases, queries=queries, documents={'0': {'text': 'p'}, '1': {'text': 'n'}},
                document_text_sha256={str(i): hashlib.sha256(s.encode()).hexdigest()
                                      for i, s in enumerate(('p', 'n'))},
                sentinel_ids=sentinels, sentinel_pairs=24, locked_test_access=False, new_inference=False)


class DiagnosisTests(unittest.TestCase):
    def test_only_four_disposable_steps_then_review(self):
        self.assertEqual(diagnosis.PHASES, ('probe-D-1', 'probe-U-1', 'probe-D-97', 'probe-U-97', 'review'))
        diagnosis.validate_data(fixture())

    def test_identity_split_hash_and_coverage_changes_are_rejected(self):
        def case(data):
            return data['cases']['probe-D-1']
        mutations = [
            lambda d: d.update(locked_test_access=True),
            lambda d: d.update(new_inference=True),
            lambda d: d['queries']['nq-0'].update(split='LOCKED_TEST'),
            lambda d: d['queries']['nq-0'].update(query='changed'),
            lambda d: d['queries']['nq-0'].update(subset='fever'),
            lambda d: d['documents']['0'].update(text='changed'),
            lambda d: case(d).update(start=96),
            lambda d: case(d)['batch_ids'].__setitem__(0, 'nq-0'),
            lambda d: case(d)['probes'].pop(),
            lambda d: case(d)['probes'][0].update(positive_id=2),
            lambda d: case(d)['probes'][0].update(rival_id=0),
            lambda d: case(d)['probes'][0].update(origin='ambiguous'),
            lambda d: case(d)['probes'][0].update(rivals_are_judged_negative=True),
            lambda d: case(d)['witnesses'][0].update(split='DEV_NEW'),
        ]
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.assertRaises(ValueError):
                diagnosis.validate_data(data)

    def test_unknown_teacher_and_non_gold_are_not_negative_labels(self):
        record = dict(pool=[1, 2, 3], teacher_scores=[.2, .3, .2])
        self.assertEqual(diagnosis.teacher_pair(record, 1, 2)['status'], 'opposes')
        self.assertEqual(diagnosis.teacher_pair(record, 2, 1)['status'], 'agrees')
        self.assertEqual(diagnosis.teacher_pair(record, 1, 3)['status'], 'tie')
        self.assertEqual(diagnosis.teacher_pair(record, 1, 4), {'status': 'unobserved', 'margin': None})

    def test_same_batch_pressure_is_total_score_gradient_not_parameter_effect(self):
        case = fixture()['cases']['probe-D-1']
        actual = dict(examples=[dict(score_gradient=[-.2, .1])] * 4)
        own = case['probes'][-1]
        self.assertAlmostEqual(diagnosis.score_pressure(own, case, actual), .3)
        self.assertIsNone(diagnosis.score_pressure(case['probes'][0], case, actual))
        self.assertIsNone(diagnosis.score_pressure(dict(own, rival_id=99), case, actual))

    def test_actual_replay_checks_gradients_and_scientific_trace(self):
        row = dict(scores=[1., 0.], score_gradient=[-.1, .1], loss=.2, documents=2,
                   objective='uniform_soft_pair', eligible_pairs=1, supervised_positives=1,
                   vjp_replay_exact=True)
        expected = dict(gradient_before_clip=1., update_l2=.03, examples=[row] * 4)
        diagnosis.compare_replay(copy.deepcopy(expected), expected)
        for key, value in (('scores', [1., .01]), ('score_gradient', [.1, -.1]),
                           ('objective', 'balanced_soft_pair'), ('vjp_replay_exact', False)):
            actual = copy.deepcopy(expected)
            actual['examples'][0][key] = value
            with self.assertRaises((ValueError, AssertionError)):
                diagnosis.compare_replay(actual, expected)

    def test_frozen_sources_immutable_and_no_path_escape(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'inside').mkdir()
            pipeline.io.write(root / 'outside.json', {'a': 1})
            digest = pipeline.io.sha(root / 'outside.json')
            diagnosis.verify_files(root, {'outside.json': digest})
            with self.assertRaisesRegex(ValueError, 'frozen input'):
                diagnosis.verify_files(root / 'inside', {'../outside.json': digest})
            (root / 'outside.json').write_text('{"a": 2}\n')
            with self.assertRaisesRegex(ValueError, 'frozen input'):
                diagnosis.verify_files(root, {'outside.json': digest})

    def test_timeout_cannot_relax_existing_limit_or_use_nonboolean_gpu_flag(self):
        args = SimpleNamespace()
        for limit in (0, 5401, True, 1.5):
            with self.assertRaisesRegex(ValueError, 'timeout'):
                pipeline.phase(args, 'probe-D-1', [], cuda=True, limit_seconds=limit)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            pipeline.phase(args, 'probe-D-1', [], cuda='yes', limit_seconds=1800)

    def test_worker_tracking_receipt_failure_closes_task(self):
        from clearml import Task
        args = SimpleNamespace(research_root=Path('/none'), run=Path('/none'), phase='probe-D-1')
        with patch.object(diagnosis, 'verify'), patch.object(Task, 'init') as init, \
                patch.object(diagnosis.io, 'write', side_effect=[OSError('receipt'), None]), \
                patch('torch.set_num_interop_threads'):
            with self.assertRaisesRegex(OSError, 'receipt'):
                diagnosis.worker(args)
            init.return_value.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
