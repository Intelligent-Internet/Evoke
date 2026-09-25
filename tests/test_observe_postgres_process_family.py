from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'observe_postgres_process_family.py'
)
SPEC = importlib.util.spec_from_file_location(
    'observe_postgres_process_family',
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_qualification_report_accepts_bounded_family_memory() -> None:
    report = MODULE.qualification_report(
        {
            'evoke build: heap scan': {
                'family_MaxProcessHwm_bytes': 700,
                'family_Private_bytes': 650,
                'family_Pss_bytes': 680,
                'family_Swap_bytes': 0,
                'runtime_worker_Hwm_bytes': 700,
            },
        },
        800,
        0,
    )

    assert report['qualified'] is True
    assert report['qualification_errors'] == []
    assert report['memory_observed']['peak_rss_bytes'] == 700


def test_qualification_report_rejects_runtime_worker_hwm() -> None:
    report = MODULE.qualification_report(
        {
            'evoke build: heap scan': {
                'family_MaxProcessHwm_bytes': 900,
                'family_Private_bytes': 650,
                'family_Pss_bytes': 680,
                'family_Swap_bytes': 0,
                'runtime_worker_Hwm_bytes': 900,
            },
        },
        800,
        0,
    )

    assert report['qualified'] is False
    assert report['qualification_errors'] == [
        'full rebuild peak RSS exceeds limit: 900 > 800 bytes',
    ]
