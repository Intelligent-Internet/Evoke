"""No-process tests of immutable prerequisite checking."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from ng85_data import read, sha, write
from ng85_run import closed, verify
import ng85_run


def test_failed_or_changed_phase_cannot_be_consumed(tmp_path):
    out = tmp_path / 'prepare'
    out.mkdir()
    write(out / 'results.json', dict(queries=5))
    write(out / 'exit.json', dict(exit_code=0))
    files = {p.name: sha(p) for p in out.iterdir()}
    write(out / 'complete.json', dict(passed=True, files=files))
    assert closed(tmp_path, 'prepare') == dict(queries=5)
    write(out / 'unexpected.json', dict(unbound=True))
    with pytest.raises(ValueError, match='inventory'):
        closed(tmp_path, 'prepare')
    (out / 'unexpected.json').unlink()
    write(out / 'results.json', dict(queries=6))
    with pytest.raises(ValueError):
        closed(tmp_path, 'prepare')


def test_frozen_code_change_fails(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'code.py').write_text('pass\n')
    write(run / 'inputs.json', dict(protocol='NG85_teacher_only_AB_v1',
        locked_test_access=False, source={'code.py': sha(run / 'code.py')}, dependencies={}))
    verify(tmp_path, run)
    (run / 'code.py').write_text('changed\n')
    with pytest.raises(ValueError):
        verify(tmp_path, run)


def test_continuation_copies_only_verified_phases_without_changing_science(tmp_path):
    import numpy as np

    root = tmp_path / 'NG-0085'
    parent = root / 'v2'
    parent.mkdir(parents=True)
    (parent / 'science.py').write_text('frozen scientific source\n')
    write(parent / 'inputs.json', dict(protocol='NG85_teacher_only_AB_v1',
        locked_test_access=False, epochs=2, seeds=[85001, 85002],
        source={'science.py': sha(parent / 'science.py')}, dependencies={}))
    original = sha(parent / 'inputs.json')
    for phase in ('prepare', 'dense', 'bank'):
        out = parent / phase
        out.mkdir()
        write(out / 'results.json', dict(test=True))
        write(out / 'exit.json', dict(exit_code=0))
        write(out / 'complete.json', dict(passed=True,
            files={p.name: sha(p) for p in out.iterdir()}))
    (parent / 'rankt5-0').mkdir()
    child = root / 'v3'
    ng85_run.continue_frozen(tmp_path, child, parent)
    assert sha(parent / 'inputs.json') == original
    assert not (child / 'rankt5-0').exists()
    assert (child / 'science.py').read_bytes() == (parent / 'science.py').read_bytes()
    manifest = verify(tmp_path, child)
    assert manifest['epochs'] == 2 and manifest['seeds'] == [85001, 85002]
    assert read(child / 'continuation.json')['parent_inputs_sha256'] == original
    for phase in ('prepare', 'dense', 'bank'):
        assert closed(child, phase) == dict(test=True)
        assert (child / phase / 'results.json').stat().st_ino != (
            parent / phase / 'results.json').stat().st_ino
    with pytest.raises(ValueError, match='new NG85 run'):
        ng85_run.continue_frozen(tmp_path, child, parent)
    probe = root / 'probe'
    probe.mkdir()
    write(probe / 'inputs.json', dict(inputs_sha256=sha(parent / 'inputs.json'),
        batches=[2, 8], queries=32, dtype='float32', length_sorted=True))
    write(probe / 'results.json', dict(passed=False, equivalent=True,
        metrics={'8': dict(guarded_seconds=30000)}))
    with pytest.raises(ValueError, match='probe did not pass'):
        ng85_run.continue_frozen(tmp_path, root / 'rejected', parent, probe)
    assert not (root / 'rejected').exists()
    source = probe / 'ng85_teacher_batch_probe.py'
    source.write_text('probe source\n')
    targets_source = Path(ng85_run.__file__).parent / 'ng85_targets.py'
    (probe / 'ng85_targets.py').write_bytes(targets_source.read_bytes())
    write(probe / 'inputs.json', dict(inputs_sha256=sha(parent / 'inputs.json'),
        batches=[2, 8], queries=32, dtype='float32', source_sha256=sha(source),
        length_sorted=True, targets_source_sha256=sha(targets_source)))
    write(probe / 'results.json', dict(passed=True, equivalent=True,
        metrics={'8': dict(guarded_seconds=20000)}))
    np.save(probe / 'batch-2.npy', np.array([1., 2.]))
    np.save(probe / 'batch-8.npy', np.array([1., 3.]))
    with pytest.raises(ValueError, match='payload mismatch'):
        ng85_run.continue_frozen(tmp_path, root / 'corrupt', parent, probe)
    np.save(probe / 'batch-8.npy', np.array([1., 2.]))
    child4 = root / 'v4'
    ng85_run.continue_frozen(tmp_path, child4, parent, probe)
    assert read(child4 / 'execution.json')['rankt5_batch_size'] == 8
    assert verify(tmp_path, child4)['epochs'] == 2
