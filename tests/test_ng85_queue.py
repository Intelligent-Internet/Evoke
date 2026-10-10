"""CPU-only queue tests: busy admission is not a failed computation retry."""

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import ng85_queue as queue
from ng85_data import read


def snapshot(gpu=3, memory=1, utilization=0, pids=()):
    return dict(gpu=gpu, memory_mib=memory, utilization=utilization,
                compute_pids=list(pids))


@pytest.mark.parametrize('state,expected', [
    (snapshot(), True),
    (snapshot(memory=100), False),
    (snapshot(utilization=1), False),
    (snapshot(pids=(42,)), False),
])
def test_idle_requires_memory_utilization_and_no_processes(state, expected):
    assert queue.idle(state) is expected


def test_busy_can_only_requeue_before_any_phase_output(tmp_path):
    pending = []
    queue.finish_phase(tmp_path, 'rankt5-0', 75, pending, None)
    assert pending == ['rankt5-0']
    (tmp_path / 'rankt5-0').mkdir()
    with pytest.raises(ValueError, match='computation output'):
        queue.finish_phase(tmp_path, 'rankt5-0', 75, pending, None)
    with pytest.raises(ValueError, match='failed phase'):
        queue.finish_phase(tmp_path, 'rankt5-1', 1, pending, None)


def harness(monkeypatch, tmp_path, states, codes):
    clock, commands, receipts, processes = [0.], [], [], []
    monkeypatch.setattr(queue.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(queue.time, 'sleep', lambda n: clock.__setitem__(0, clock[0] + n))
    monkeypatch.setattr(queue, 'gpu_snapshot', states)

    class Process:
        pid = 999

        def __init__(self, code):
            self.code, self.killed = code, False

        def poll(self):
            return self.code

        def terminate(self):
            self.killed = True

        def wait(self, timeout):
            return self.code

    def launch(command):
        commands.append(command)
        proc = Process(codes.pop(0))
        processes.append(proc)
        return proc

    monkeypatch.setattr(queue.subprocess, 'Popen', launch)
    return commands, receipts, processes, lambda r, p: receipts.append(p)


def test_four_shards_finish_on_one_free_gpu(monkeypatch, tmp_path):
    def states(g):
        return snapshot(g, memory=1 if g == 3 else 12000)
    commands, receipts, _, closed = harness(monkeypatch, tmp_path, states, [0] * 4)
    phases = tuple(f'rankt5-{i}' for i in range(4))
    queue.run_stage(tmp_path, tmp_path, phases, [], 10000, closed)
    assert len(commands) == 4
    assert all(c[c.index('--gpu') + 1] == '3' for c in commands)
    assert receipts == list(phases)


def test_busy_race_requeues_but_computation_failure_stops(monkeypatch, tmp_path):
    def states(g):
        return snapshot(g, memory=1 if g == 3 else 12000)
    commands, receipts, _, closed = harness(monkeypatch, tmp_path, states, [75, 0])
    queue.run_stage(tmp_path, tmp_path, ('rankt5-0',), [], 10000, closed)
    assert len(commands) == 2 and receipts == ['rankt5-0']


def test_failure_cleans_up_only_owned_active_peer(monkeypatch, tmp_path):
    commands, _, processes, closed = harness(
        monkeypatch, tmp_path, lambda g: snapshot(g), [1, None])
    with pytest.raises(ValueError, match='failed phase'):
        queue.run_stage(tmp_path, tmp_path, ('rankt5-0', 'rankt5-1'), [], 10000, closed)
    assert len(commands) == 2 and processes[1].killed


def test_wait_for_resources_is_bounded(monkeypatch, tmp_path):
    commands, _, _, closed = harness(
        monkeypatch, tmp_path, lambda g: snapshot(g, pids=(42,)), [])
    monkeypatch.setattr(queue, 'QUEUE_WAIT_SECONDS', 120)
    with pytest.raises(ValueError, match='queue wait limit'):
        queue.run_stage(tmp_path, tmp_path, ('rankt5-0',), [], 10000, closed)
    assert not commands
    assert read(tmp_path / 'campaign-progress.json')['state'] == 'waiting_for_resources'


def test_repeated_admission_races_consume_wait_budget(monkeypatch, tmp_path):
    _, receipts, _, closed = harness(
        monkeypatch, tmp_path, lambda g: snapshot(g), [75, 75, 75])
    monkeypatch.setattr(queue, 'QUEUE_WAIT_SECONDS', 120)
    with pytest.raises(ValueError, match='queue wait limit'):
        queue.run_stage(tmp_path, tmp_path, ('rankt5-0',), [], 10000, closed)
    assert not receipts


def test_completed_phase_is_validated_without_launch(monkeypatch, tmp_path):
    (tmp_path / 'prepare').mkdir()
    commands, receipts, _, closed = harness(monkeypatch, tmp_path, None, [])
    queue.run_stage(tmp_path, tmp_path, ('prepare',), [], 10000, closed)
    assert receipts == ['prepare'] and not commands


def test_gpu_snapshot_records_compute_pids(monkeypatch):
    outputs = iter(('GPU-xyz, 1, 0\n', '123\n456\n'))
    monkeypatch.setattr(queue.subprocess, 'check_output', lambda *a, **k: next(outputs))
    state = queue.gpu_snapshot(3)
    assert state['uuid'] == 'GPU-xyz' and state['compute_pids'] == [123, 456]
    assert not queue.idle(state)
