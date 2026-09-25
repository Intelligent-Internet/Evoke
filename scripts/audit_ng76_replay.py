"""CPU-only raw trace comparison; differences are evidence, not waived gates."""

import argparse
import hashlib
import json
import math
from pathlib import Path


RTOL, ATOL = 2e-5, 2e-7
IDENTITY = ('query_id', 'query_tokens', 'document_tokens', 'documents',
            'objective', 'eligible_pairs', 'supervised_positives',
            'vjp_replay_exact')
READOUT = ('query_nnz', 'document_nnz')
NUMERIC = ('scores', 'score_gradient', 'loss')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(actual, historical):
    require(0 < len(actual) <= len(historical), 'empty or excess trace')
    first = historical[0]['step']
    require([r['step'] for r in historical] == list(range(first, first + len(historical))),
            'historical step coverage changed')
    require([r['step'] for r in actual] == list(range(first, first + len(actual))),
            'actual step coverage changed')
    fields = ('gradient_before_clip', 'update_l2', *NUMERIC)
    summary = {k: dict(first_nonexact_step=None, changed_values=0,
                      max_abs_difference=0., outside_tolerance=0) for k in fields}
    identity, readout, differences = [], [], []

    def numeric(key, a, b, where):
        a = a if isinstance(a, list) else [a]
        b = b if isinstance(b, list) else [b]
        require(len(a) == len(b) > 0, 'numeric shape changed: ' + key)
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            require(type(x) in (int, float) and type(y) in (int, float)
                    and math.isfinite(x) and math.isfinite(y), 'non-finite/non-numeric trace')
            delta = abs(x - y)
            outside = delta > ATOL + RTOL * abs(y)
            row = summary[key]
            row['max_abs_difference'] = max(row['max_abs_difference'], delta)
            row['outside_tolerance'] += outside
            if x != y:
                row['changed_values'] += 1
                if row['first_nonexact_step'] is None:
                    row['first_nonexact_step'] = where['step']
                differences.append(dict(where, field=key, index=i, actual=x,
                                        historical=y, outside_tolerance=outside))

    for a, b in zip(actual, historical):
        require(a['reference_update'] == b['reference_update'], 'reference changed')
        require(len(a['examples']) == len(b['examples']) == 4, 'exposure count changed')
        for key in fields[:2]:
            numeric(key, a[key], b[key], dict(step=a['step']))
        for j, (x, y) in enumerate(zip(a['examples'], b['examples'], strict=True)):
            where = dict(step=a['step'], example=j, query_id=y['query_id'])
            for keys, target in ((IDENTITY, identity), (READOUT, readout)):
                for key in keys:
                    if x[key] != y[key]:
                        target.append(dict(where, field=key, actual=x[key], historical=y[key]))
            for key in NUMERIC:
                numeric(key, x[key], y[key], where)
    return dict(steps=len(actual), exposures=len(actual) * 4,
                historical_steps=len(historical), numeric=summary,
                identity_differences=identity, readout_differences=readout,
                numeric_differences=differences,
                numeric_tolerance_passed=all(r['outside_tolerance'] == 0 for r in summary.values()),
                complete_trace=len(actual) == len(historical),
                quality_evaluation=False, historical_endpoint_verified=False,
                causal_mechanism_identified=False, overall_goal_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hashes = {str(p): sha(p) for p in (args.actual, args.historical)}
    result = audit(rows(args.actual), rows(args.historical))
    require(hashes == {str(p): sha(p) for p in (args.actual, args.historical)}, 'trace changed during audit')
    with args.output.open('x') as stream:
        json.dump(dict(result, source_sha256=sha(Path(__file__)), inputs=hashes),
                  stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'numeric_differences'}, allow_nan=False))


if __name__ == '__main__':
    main()
