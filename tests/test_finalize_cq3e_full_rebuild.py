from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'finalize_cq3e_full_rebuild.py'
)
SPEC = importlib.util.spec_from_file_location(
    'finalize_cq3e_full_rebuild',
    MODULE_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EQUIVALENCE_FIXTURE = (
    Path(__file__).resolve().parent
    / 'fixtures'
    / 'cq3_pubmed_full'
    / 'postbuild_equivalence.sql'
)


def backend_sample() -> dict[str, object]:
    return {
        'activity': {
            'pid': 123,
            'backend_started_unix': 100.0,
        },
        'elapsed_seconds': 99.0,
    }


def bounded_build_report() -> dict[str, object]:
    return {
        'application_name': 'evoke_cq3e',
        'completed_unix': 200.0,
        'database': 'postgres',
        'host': '/tmp/pg',
        'index': 'bench.candidate_idx',
        'port': 56544,
        'sample_count': 10,
        'samples': [backend_sample()],
        'started_unix': 100.0,
        'status': 'completed',
        'psql_returncode': 0,
        'memory_limits': {
            'max_rss_bytes': 1_000,
            'max_swap_bytes': 0,
        },
        'phase_peaks': {
            'evoke build: heap scan': {
                'status_VmHWM_bytes': 700,
                'status_VmSwap_bytes': 0,
            },
        },
        'memory_observed': {
            'peak_backend_rss_bytes': 700,
            'peak_backend_swap_bytes': 0,
            'peak_family_hwm_bytes': None,
            'peak_family_private_bytes': None,
            'peak_family_pss_bytes': None,
            'peak_family_swap_bytes': None,
            'peak_rss_bytes': 700,
            'peak_runtime_worker_hwm_bytes': None,
            'peak_swap_bytes': 0,
        },
        'qualification_errors': [],
        'user': 'admin',
        'binary': {
            'binding': {
                'qualified': True,
            },
        },
    }


def bounded_family_report() -> dict[str, object]:
    return {
        'application_name': 'evoke_cq3e',
        'backend_started_unix': 100.0,
        'completed_unix': 201.0,
        'coverage_started_seconds_after_backend': 1.0,
        'database': 'postgres',
        'host': '/tmp/pg',
        'status': 'completed',
        'qualified': True,
        'port': 56544,
        'sample_count': 10,
        'samples': [backend_sample()],
        'memory_limits': {
            'max_rss_bytes': 1_000,
            'max_swap_bytes': 0,
        },
        'family_phase_peaks': {
            'evoke build: heap scan': {
                'family_MaxProcessHwm_bytes': 900,
                'family_Private_bytes': 850,
                'family_Pss_bytes': 880,
                'family_Swap_bytes': 0,
                'runtime_worker_Hwm_bytes': 900,
            },
        },
        'memory_observed': {
            'peak_backend_rss_bytes': None,
            'peak_backend_swap_bytes': None,
            'peak_family_hwm_bytes': 900,
            'peak_family_private_bytes': 850,
            'peak_family_pss_bytes': 880,
            'peak_family_swap_bytes': 0,
            'peak_rss_bytes': 900,
            'peak_runtime_worker_hwm_bytes': 900,
            'peak_swap_bytes': 0,
        },
        'qualification_errors': [],
        'user': 'admin',
    }


def test_reports_accept_complete_bounded_evidence() -> None:
    assert MODULE.equivalence_sql_errors(EQUIVALENCE_FIXTURE) == []
    assert MODULE.build_report_errors(bounded_build_report()) == []
    assert MODULE.family_report_errors(
        bounded_family_report(),
        'evoke_cq3e',
    ) == []
    assert MODULE.cross_report_errors(
        bounded_build_report(),
        bounded_family_report(),
        'bench.candidate_idx',
    ) == []


def test_build_report_fails_closed_without_completion_or_limits() -> None:
    report = bounded_build_report()
    report['status'] = 'running'
    del report['memory_limits']

    assert MODULE.build_report_errors(report) == [
        'full rebuild report has no complete memory limits',
        'full rebuild report is not completed',
    ]


def test_build_report_requires_loaded_binary_binding() -> None:
    report = bounded_build_report()
    del report['binary']

    assert MODULE.build_report_errors(report) == [
        'loaded extension binary binding is missing',
    ]


def test_build_report_recomputes_embedded_family_memory() -> None:
    report = bounded_build_report()
    report['family_phase_peaks'] = {
        'evoke build: heap scan': {
            'family_MaxProcessHwm_bytes': 900,
            'family_Private_bytes': 850,
            'family_Pss_bytes': 880,
            'family_Swap_bytes': 0,
            'runtime_worker_Hwm_bytes': 900,
        },
    }
    report['memory_observed'] = {
        'peak_backend_rss_bytes': 700,
        'peak_backend_swap_bytes': 0,
        'peak_family_hwm_bytes': 900,
        'peak_family_private_bytes': 850,
        'peak_family_pss_bytes': 880,
        'peak_family_swap_bytes': 0,
        'peak_rss_bytes': 900,
        'peak_runtime_worker_hwm_bytes': 900,
        'peak_swap_bytes': 0,
    }

    assert MODULE.build_report_errors(report) == []


def test_family_report_recomputes_memory_and_identity() -> None:
    report = bounded_family_report()
    report['application_name'] = 'other'
    report['family_phase_peaks']['evoke build: heap scan'][
        'runtime_worker_Hwm_bytes'
    ] = 1_100

    assert MODULE.family_report_errors(report, 'evoke_cq3e') == [
        'full rebuild peak RSS exceeds limit: 1100 > 1000 bytes',
        'process-family application identity differs',
        'process-family recorded memory summary is inconsistent',
    ]


def test_family_report_rejects_late_or_unstable_observation() -> None:
    report = bounded_family_report()
    report['coverage_started_seconds_after_backend'] = 5.1
    report['samples'] = [
        backend_sample(),
        {
            'activity': {
                'pid': 124,
                'backend_started_unix': 101.0,
            },
        },
    ]

    assert MODULE.family_report_errors(report, 'evoke_cq3e') == [
        'process-family backend identity is missing or unstable',
        'process-family observation began too late: 5.1 seconds',
    ]


def test_equivalence_sql_rejects_untracked_fixture(tmp_path) -> None:
    replacement = tmp_path / 'replacement.sql'
    replacement.write_text('SELECT 1;\n', encoding='utf-8')

    errors = MODULE.equivalence_sql_errors(replacement)

    assert len(errors) == 1
    assert errors[0].startswith(
        'postbuild equivalence SQL differs from the tracked fixture:'
    )


def test_catalog_requires_present_valid_ready_nonempty_index() -> None:
    assert MODULE.catalog_errors('idx', {'present': False}) == [
        'index is absent: idx',
    ]
    assert MODULE.catalog_errors(
        'idx',
        {
            'present': True,
            'valid': False,
            'ready': False,
            'bytes': 0,
        },
    ) == [
        'index is not valid: idx',
        'index is not ready: idx',
        'index has no physical bytes: idx',
    ]


def test_cross_report_requires_identity_phase_and_completion_coverage() -> None:
    build = bounded_build_report()
    family = bounded_family_report()
    family['family_phase_peaks'] = {'postgres: loading tuples': {}}
    family['completed_unix'] = 198.9
    family['samples'][0]['activity']['pid'] = 456

    assert MODULE.cross_report_errors(
        build,
        family,
        'bench.other_idx',
    ) == [
        'full rebuild candidate index identity differs',
        'process-family backend identity differs',
        'process-family observation ended before the rebuild',
        'process-family report does not cover every build phase',
    ]


def test_cross_report_accepts_observer_before_wrapper_completion() -> None:
    build = bounded_build_report()
    family = bounded_family_report()
    family['completed_unix'] = 199.5

    assert MODULE.cross_report_errors(
        build,
        family,
        'bench.candidate_idx',
    ) == []


def test_cross_report_requires_last_sample_timestamp() -> None:
    build = bounded_build_report()
    del build['samples'][0]['elapsed_seconds']

    assert MODULE.cross_report_errors(
        build,
        bounded_family_report(),
        'bench.candidate_idx',
    ) == ['full rebuild observation timestamps are missing']
