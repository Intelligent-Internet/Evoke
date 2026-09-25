import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import review_ng75_displacement as review


def row(query='q', change=-.02):
    return dict(query_id=query, split='TRAIN', domain='nq', surface='TRAIN_SENTINEL',
        case='probe-D-1', positive_id=0, rival_id=1, origin='original_positive',
        rivals_are_judged_negative=False, before_margin=.1, after_margin=.1 + change,
        semantic_margin=.09, lexical_margin=.01, no_update_error=0.,
        predicted=change, predicted_change=change, actual_change=change,
        numerical_floor=1e-7, taylor_residual=0., actual_direction=review.sign(change, 1e-7),
        predicted_direction=review.sign(change, 1e-7), same_batch_score_pressure=None,
        contributions={'query': {'trunk': .5 * change, 'head': .1 * change},
                       'document': {'trunk': .3 * change, 'head': .1 * change}},
        gradient_chain_max_abs=0., gradient_chain_relative_l2=0.,
        teacher_original={'status': 'agrees', 'margin': .1},
        teacher_reference={'status': 'unobserved', 'margin': None})


class IndependentDisplacementReviewTests(unittest.TestCase):
    def test_raw_arithmetic_and_separate_same_batch_pressure(self):
        r = row()
        case = dict(batch_ids=[], witnesses=[])
        self.assertLess(review.audit_row(r, {'query_id': 'q'}, case, {}), 1e-12)
        r.update(surface='TRAIN_CURRENT_BATCH', same_batch_score_pressure=.3)
        case = dict(batch_ids=['q'], witnesses=[{'pool': [0, 1]}])
        trace = {'examples': [{'score_gradient': [-.2, .1]}]}
        review.audit_row(r, {'query_id': 'q'}, case, trace)
        self.assertEqual(review.characterize([r])['score_pressure']['expand']['directions']['shrink'], 1)

    def test_corrupted_arithmetic_scope_and_projection_fail(self):
        for key, value in (('actual_change', 1.), ('taylor_residual', .1), ('split', 'LOCKED_TEST'),
                           ('no_update_error', .01), ('predicted_direction', 'expand'),
                           ('rivals_are_judged_negative', True), ('predicted', float('nan'))):
            changed = dict(row(), **{key: value})
            with self.assertRaises(ValueError):
                review.audit_row(changed, {}, {'batch_ids': []}, {})
        changed = row()
        changed['contributions']['query']['trunk'] = 10.
        with self.assertRaises(ValueError):
            review.audit_row(changed, {}, {'batch_ids': []}, {})

    def test_query_balancing_does_not_drop_multi_gold_pairs(self):
        rows = [row('q1', .1), row('q1', .1), row('q2', -.1)]
        rows[-1]['origin'] = 'added_vs_original'
        values = review.reduce_summary(rows)['TRAIN_SENTINEL/nq']
        self.assertEqual(values['all']['pairs'], 3)
        self.assertEqual(values['all']['queries'], 2)
        self.assertEqual(values['all']['query_balanced_mean_change'], 0.)
        self.assertAlmostEqual(values['all']['mean_actual_change'], .1 / 3)
        self.assertEqual(values['added_vs_original']['directions']['shrink'], 1)

    def test_teacher_unknown_cannot_be_relabelled(self):
        changed = row()
        changed['teacher_original'] = {'status': 'unobserved', 'margin': .1}
        with self.assertRaises(ValueError):
            review.audit_row(changed, {}, {'batch_ids': []}, {})
        changed['teacher_original'] = {'status': 'opposes', 'margin': .1}
        with self.assertRaises(ValueError):
            review.audit_row(changed, {}, {'batch_ids': []}, {})

    def test_recursive_comparison_rejects_missing_keys_nan_and_bool(self):
        original = {'x': [1., 2.], 'flag': True, 'empty': None}
        self.assertEqual(review.compare(copy.deepcopy(original), original), 0.)
        for altered in ({'x': [1., 2.]}, dict(original, x=[1., float('nan')]),
                        dict(original, flag=1), dict(original, x=[1.])):
            with self.assertRaises(ValueError):
                review.compare(altered, original)

    def test_empty_strata_and_zero_crossing_are_explicit(self):
        empty = review.characterize([])
        self.assertIsNone(empty['residual_l1_over_change_l1'])
        self.assertEqual(empty['summary']['pairs'], 0)
        result = review.characterize([row(change=-.2)])
        self.assertEqual(result['zero_crossings']['positive_to_negative'], 1)
        self.assertEqual(result['shrinking_queries'], 1)


if __name__ == '__main__':
    unittest.main()
