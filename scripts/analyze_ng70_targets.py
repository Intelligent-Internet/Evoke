#!/usr/bin/env python3
"""Read saved TRAIN targets to quantify provenance-dependent supervision mass."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import audit_ng70_provenance as provenance


def verify(folder):
    complete = json.loads((folder / 'complete.json').read_text())
    assert complete['passed']
    for name, digest in complete['files'].items():
        assert provenance.sha(folder / name) == digest, name


def target_summary(target, kinds):
    scores = np.asarray(target['scores'], dtype=np.float64)
    positive = np.asarray(target['positive_mask'], dtype=bool)
    values = np.asarray(target['target'], dtype=np.float64)
    assert len(scores) == len(positive) == len(values) == len(kinds)
    assert positive.any() and (~positive).any()
    assert all((k != 'source_negative') == p for k, p in zip(kinds, positive))
    teacher = np.exp((scores - scores.max()) / .04)
    teacher /= teacher.sum()
    gold = positive.astype(np.float64) / positive.sum()
    np.testing.assert_allclose(values, .5 * gold + .5 * teacher, rtol=0, atol=1e-12)
    kinds = np.asarray(kinds)
    original, added = kinds == 'original_positive', kinds == 'added_vs_original'
    assert original.any() and not (kinds == 'unresolved').any()
    return {
        'positive_count': int(positive.sum()), 'added_count': int(added.sum()),
        'has_added': bool(added.any()),
        'original_label_floor': float(.5 * gold[original].sum()),
        'original_teacher_mass': float(teacher[original].sum()),
        'original_target_mass': float(values[original].sum()),
        'added_target_mass': float(values[added].sum()),
        'teacher_negative_above_all_original': bool(scores[~positive].max() > scores[original].max()),
        'target_negative_above_any_original': bool(values[~positive].max() > values[original].min()),
        'teacher_added_above_all_original': bool(added.any() and scores[added].max() > scores[original].max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    verify(args.provenance)
    teacher = args.research_root / 'NG-0069/teacher-v1'
    verify(teacher)
    targets = json.loads((teacher / 'targets.json').read_text())
    records = json.loads((args.provenance / 'pair-provenance.json').read_text())
    assert len(records) == len(targets) == 6144
    needed = {i for t in targets for i, positive in zip(t['pool'], t['positive_mask']) if positive}
    documents = {}
    path = args.research_root / 'NG-0067/data/documents.jsonl'
    lexical = args.research_root / 'NG-0069/lexical-v1'
    expected = json.loads((lexical / 'inputs.json').read_text())['files'][
        'NG-0067/data/documents.jsonl']
    assert provenance.sha(path) == expected, 'Document identity map changed'
    with path.open() as stream:
        for index, line in enumerate(stream):
            if index in needed:
                documents[index] = json.loads(line)['key']
    assert set(documents) == needed
    analyzed, skipped = [], Counter()
    for record, target in zip(records, targets):
        assert record['query_id'] == target['query_id']
        if record['status'] != 'resolved':
            skipped[record['domain']] += 1
            continue
        lookup = {p['document_key']: p['provenance'] for p in record['pairs']}
        assert len(lookup) == len(record['pairs'])
        assert set(lookup) == {documents[i] for i, positive in
                               zip(target['pool'], target['positive_mask']) if positive}
        kinds = [lookup[documents[i]] if positive else 'source_negative'
                 for i, positive in zip(target['pool'], target['positive_mask'])]
        row = target_summary(target, kinds)
        analyzed.append(dict(row, query_id=target['query_id'], domain=record['domain']))
    domains = {}
    for domain in provenance.DOMAINS:
        rows = [r for r in analyzed if r['domain'] == domain]
        domains[domain] = {'resolved_queries': len(rows), 'unresolved_skipped': skipped[domain],
            'queries_with_added': sum(r['has_added'] for r in rows)}
        for key in ('original_label_floor', 'original_teacher_mass',
                    'original_target_mass', 'added_target_mass'):
            domains[domain][key + '_mean'] = float(np.mean([r[key] for r in rows]))
        for key in ('teacher_negative_above_all_original', 'target_negative_above_any_original',
                    'teacher_added_above_all_original'):
            domains[domain][key] = sum(r[key] for r in rows)
    provenance.save(args.output, {'domains': domains, 'per_query': analyzed,
        'inputs': {'provenance_complete_sha256': provenance.sha(args.provenance / 'complete.json'),
                   'teacher_complete_sha256': provenance.sha(teacher / 'complete.json'),
                   'document_source_sha256': provenance.sha(path)},
        'source_sha256': provenance.sha(Path(__file__)),
        'helper_sha256': provenance.sha(Path(provenance.__file__)),
        'new_inference': False, 'new_human_labels': 0, 'qrels_modified': False,
        'locked_test_scored': False, 'loss_mass_is_not_a_label_error_rate': True})
    print(json.dumps(domains, indent=2))


if __name__ == '__main__':
    main()
