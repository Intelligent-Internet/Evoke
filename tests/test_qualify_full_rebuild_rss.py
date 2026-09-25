from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'qualify_full_rebuild_rss.py'
SPEC = importlib.util.spec_from_file_location(
    'qualify_full_rebuild_rss',
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qualification_accepts_ready_valid_index() -> None:
    catalog = {'present': True, 'valid': True, 'ready': True}

    assert MODULE.qualification_errors(0, 'bench.index', catalog) == []


def test_qualification_rejects_missing_index() -> None:
    errors = MODULE.qualification_errors(
        0,
        'bench.index',
        {'present': False},
    )

    assert errors == ['qualified index is absent: bench.index']


def test_qualification_rejects_invalid_or_unready_index() -> None:
    errors = MODULE.qualification_errors(
        0,
        'bench.index',
        {'present': True, 'valid': False, 'ready': False},
    )

    assert errors == [
        'qualified index is not valid: bench.index',
        'qualified index is not ready: bench.index',
    ]


def test_qualification_preserves_psql_failure() -> None:
    errors = MODULE.qualification_errors(3, None, None)

    assert errors == ['psql exited with status 3']


def test_binary_binding_accepts_exact_mapped_copy(tmp_path: Path) -> None:
    expected = tmp_path / 'expected' / 'evoke.so'
    mapped = tmp_path / 'loaded' / 'evoke.so'
    expected.parent.mkdir()
    mapped.parent.mkdir()
    expected.write_bytes(b'qualified artifact')
    mapped.write_bytes(expected.read_bytes())

    original = MODULE.process_mapped_files
    MODULE.process_mapped_files = lambda _pid: [mapped]
    try:
        binding = MODULE.binary_binding(123, expected)
    finally:
        MODULE.process_mapped_files = original

    assert binding['qualified'] is True
    assert binding['matching_paths'] == [str(mapped)]
    assert MODULE.binary_binding_errors(binding) == []


def test_binary_binding_rejects_different_mapped_binary(
    tmp_path: Path,
) -> None:
    expected = tmp_path / 'expected' / 'evoke.so'
    mapped = tmp_path / 'loaded' / 'evoke.so'
    expected.parent.mkdir()
    mapped.parent.mkdir()
    expected.write_bytes(b'qualified artifact')
    mapped.write_bytes(b'other artifact')

    original = MODULE.process_mapped_files
    MODULE.process_mapped_files = lambda _pid: [mapped]
    try:
        binding = MODULE.binary_binding(123, expected)
    finally:
        MODULE.process_mapped_files = original

    assert binding['qualified'] is False
    assert binding['matching_paths'] == []
    assert MODULE.binary_binding_errors(binding) == [
        'loaded extension binary differs from qualified artifact',
    ]


def test_binary_binding_rejects_missing_evidence() -> None:
    assert MODULE.binary_binding_errors(None) == [
        'loaded extension binary binding is missing',
    ]


def test_memory_qualification_accepts_bounded_rss_without_swap() -> None:
    observed, errors = MODULE.memory_qualification(
        {
            'evoke build: heap scan': {
                'status_VmHWM_bytes': 512 * 1024 * 1024,
                'status_VmSwap_bytes': 0,
            },
            'evoke build: segment publish': {
                'status_VmHWM_bytes': 768 * 1024 * 1024,
                'smaps_Swap_bytes': 0,
            },
        },
        1024 * 1024 * 1024,
        0,
    )

    assert observed == {
        'peak_backend_rss_bytes': 768 * 1024 * 1024,
        'peak_backend_swap_bytes': 0,
        'peak_family_hwm_bytes': None,
        'peak_family_private_bytes': None,
        'peak_family_pss_bytes': None,
        'peak_family_swap_bytes': None,
        'peak_rss_bytes': 768 * 1024 * 1024,
        'peak_runtime_worker_hwm_bytes': None,
        'peak_swap_bytes': 0,
    }
    assert errors == []


def test_memory_qualification_rejects_rss_or_swap_over_limit() -> None:
    observed, errors = MODULE.memory_qualification(
        {
            'evoke build: COW publish': {
                'status_VmHWM_bytes': 2 * 1024 * 1024 * 1024,
                'status_VmSwap_bytes': 4096,
            },
        },
        1024 * 1024 * 1024,
        0,
    )

    assert observed == {
        'peak_backend_rss_bytes': 2 * 1024 * 1024 * 1024,
        'peak_backend_swap_bytes': 4096,
        'peak_family_hwm_bytes': None,
        'peak_family_private_bytes': None,
        'peak_family_pss_bytes': None,
        'peak_family_swap_bytes': None,
        'peak_rss_bytes': 2 * 1024 * 1024 * 1024,
        'peak_runtime_worker_hwm_bytes': None,
        'peak_swap_bytes': 4096,
    }
    assert errors == [
        'full rebuild peak RSS exceeds limit: '
        '2147483648 > 1073741824 bytes',
        'full rebuild peak swap exceeds limit: 4096 > 0 bytes',
    ]


def test_memory_qualification_rejects_missing_telemetry() -> None:
    observed, errors = MODULE.memory_qualification({}, 1024, 0)

    assert observed == {
        'peak_backend_rss_bytes': None,
        'peak_backend_swap_bytes': None,
        'peak_family_hwm_bytes': None,
        'peak_family_private_bytes': None,
        'peak_family_pss_bytes': None,
        'peak_family_swap_bytes': None,
        'peak_rss_bytes': None,
        'peak_runtime_worker_hwm_bytes': None,
        'peak_swap_bytes': None,
    }
    assert errors == [
        'full rebuild RSS telemetry is missing',
        'full rebuild swap telemetry is missing',
    ]


def test_process_family_memory_accounts_runtime_worker_footprint() -> None:
    family = MODULE.aggregate_process_family_memory([
        (
            'postgres: postmaster',
            {
                'status_VmHWM_bytes': 120,
                'status_VmRSS_bytes': 100,
                'status_VmSwap_bytes': 0,
                'smaps_Pss_bytes': 80,
                'smaps_Private_Clean_bytes': 10,
                'smaps_Private_Dirty_bytes': 50,
            },
        ),
        (
            'postgres: evoke runtime service 2',
            {
                'status_VmHWM_bytes': 800,
                'status_VmRSS_bytes': 700,
                'status_VmSwap_bytes': 5,
                'smaps_Pss_bytes': 650,
                'smaps_Private_Clean_bytes': 20,
                'smaps_Private_Dirty_bytes': 600,
            },
        ),
    ])

    assert family == {
        'family_MaxProcessHwm_bytes': 800,
        'family_Private_bytes': 680,
        'family_Pss_bytes': 730,
        'family_Rss_bytes': 800,
        'family_Swap_bytes': 5,
        'family_process_count': 2,
        'runtime_worker_Hwm_bytes': 800,
        'runtime_worker_Pss_bytes': 650,
        'runtime_worker_Rss_bytes': 700,
        'runtime_worker_Swap_bytes': 5,
    }


def test_memory_qualification_uses_process_family_peak() -> None:
    observed, errors = MODULE.memory_qualification(
        {
            'evoke build: heap scan': {
                'status_VmHWM_bytes': 200,
                'status_VmSwap_bytes': 0,
            },
        },
        700,
        0,
        {
            'evoke build: heap scan': {
                'family_MaxProcessHwm_bytes': 800,
                'family_Private_bytes': 680,
                'family_Pss_bytes': 730,
                'family_Swap_bytes': 0,
                'runtime_worker_Hwm_bytes': 800,
            },
        },
    )

    assert observed['peak_backend_rss_bytes'] == 200
    assert observed['peak_family_pss_bytes'] == 730
    assert observed['peak_runtime_worker_hwm_bytes'] == 800
    assert observed['peak_rss_bytes'] == 800
    assert errors == [
        'full rebuild peak RSS exceeds limit: 800 > 700 bytes',
    ]


def test_memory_qualification_requires_family_pss_when_requested() -> None:
    _, errors = MODULE.memory_qualification(
        {
            'evoke build: heap scan': {
                'status_VmHWM_bytes': 200,
                'status_VmSwap_bytes': 0,
            },
        },
        700,
        0,
        {},
    )

    assert errors == [
        'full rebuild process-family PSS telemetry is missing',
    ]
