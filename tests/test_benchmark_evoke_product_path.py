from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import benchmark_evoke_product_path


def test_concurrency_summary_separates_query_and_connection_cost() -> None:
    rows = [
        {
            'ok': True,
            'query_ms': 4.0,
            'connection_ms': 10.0,
            'warmup_ms': 100.0,
            'barrier_wait_ms': 5.0,
            'total_ms': 15.0,
        },
        {
            'ok': True,
            'query_ms': 6.0,
            'connection_ms': 20.0,
            'warmup_ms': 120.0,
            'barrier_wait_ms': 7.0,
            'total_ms': 27.0,
        },
        {'ok': False, 'error': 'expected fixture failure'},
    ]

    summary = benchmark_evoke_product_path.summarize_concurrent_queries(rows)

    assert summary['workers'] == 3
    assert summary['successes'] == 2
    assert summary['failures'] == 1
    assert summary['query_ms']['mean_ms'] == 5.0
    assert summary['connection_ms']['mean_ms'] == 15.0
    assert summary['warmup_ms']['mean_ms'] == 110.0
    assert summary['barrier_wait_ms']['mean_ms'] == 6.0
    assert summary['total_ms']['mean_ms'] == 21.0


def test_concurrency_summary_preserves_all_failure_evidence() -> None:
    summary = benchmark_evoke_product_path.summarize_concurrent_queries(
        [{'ok': False, 'error': 'worker unavailable'}],
    )

    assert summary == {
        'workers': 1,
        'successes': 0,
        'failures': 1,
        'query_ms': None,
        'connection_ms': None,
        'warmup_ms': None,
        'barrier_wait_ms': None,
        'total_ms': None,
    }


def test_source_lookup_plan_distinguishes_tid_and_sequential_scans() -> None:
    summary = benchmark_evoke_product_path.summarize_source_lookup_plan(
        'Nested Loop\n  Tid Scan on bench.docs source'
    )

    assert summary == {
        'tid_scan': True,
        'sequential_scan': False,
    }


def test_benchmark_validation_accepts_healthy_semantic_run() -> None:
    results = {
        'bm25': {
            'docs': 75_000,
            'body_source_lookup': {
                'tid_scan': True,
                'sequential_scan': False,
            },
            'field_source_lookup': {
                'tid_scan': True,
                'sequential_scan': False,
            },
        },
        'semantic_enabled': {
            'docs': 75_000,
            'status': {
                'query_ready': True,
                'runtime_signature_matches': True,
            },
            'audit': {
                'passed': True,
                'model_artifacts_valid': True,
            },
            'index_runtime_phase': {
                'failures': 0,
                'encoded_texts': 75_000,
                'single_request_execution': False,
            },
            'batch_encoding_diagnostic': {'parity_mismatches': 0},
            'source_lookup': {
                'tid_scan': True,
                'sequential_scan': False,
            },
        },
        'semantic_concurrency_summary': {
            'workers': 96,
            'successes': 96,
            'failures': 0,
        },
        'runtime_after_concurrency': {
            'worker_ready': True,
            'queue_depth': 0,
            'failures': 0,
            'busy_rejections': 0,
            'orphan_responses': 0,
            'canceled_requests': 0,
        },
    }

    assert benchmark_evoke_product_path.validate_benchmark_results(
        results
    ) == {'passed': True, 'errors': []}


def test_benchmark_validation_rejects_false_green_results() -> None:
    results = {
        'bm25': {
            'docs': 75_000,
            'body_source_lookup': {
                'tid_scan': False,
                'sequential_scan': True,
            },
            'field_source_lookup': {
                'tid_scan': True,
                'sequential_scan': False,
            },
        },
        'semantic_enabled': {
            'docs': 75_000,
            'status': {
                'query_ready': False,
                'runtime_signature_matches': True,
            },
            'audit': {
                'passed': False,
                'model_artifacts_valid': True,
            },
            'index_runtime_phase': {
                'failures': 0,
                'encoded_texts': 75_000,
                'single_request_execution': False,
            },
            'batch_encoding_diagnostic': {'parity_mismatches': 1},
            'source_lookup': {
                'tid_scan': True,
                'sequential_scan': False,
            },
        },
        'semantic_concurrency_summary': {
            'workers': 96,
            'successes': 95,
            'failures': 1,
        },
        'runtime_after_concurrency': {
            'worker_ready': True,
            'queue_depth': 1,
            'failures': 0,
        },
    }

    validation = benchmark_evoke_product_path.validate_benchmark_results(
        results
    )

    assert validation['passed'] is False
    assert 'body_source_lookup did not use TID source lookup' in (
        validation['errors']
    )
    assert 'semantic index is not query ready' in validation['errors']
    assert 'batch encoding does not match single-text output' in (
        validation['errors']
    )
    assert 'semantic concurrent queries reported failures' in (
        validation['errors']
    )
    assert 'runtime queue did not drain after concurrency' in (
        validation['errors']
    )


def test_runtime_phase_summary_reports_batch_and_overhead() -> None:
    before = {
        'requests': 4,
        'successes': 4,
        'runtime_runs': 4,
        'runtime_total_us': 100_000,
        'encoded_texts': 4,
        'batch_successes': 0,
    }
    after = {
        'requests': 7,
        'successes': 7,
        'runtime_runs': 7,
        'runtime_total_us': 1_600_000,
        'runtime_max_us': 600_000,
        'encoded_texts': 84,
        'batch_successes': 3,
        'last_batch_size': 16,
        'max_observed_batch_size': 128,
        'max_supported_batch_size': 512,
    }

    summary = benchmark_evoke_product_path.runtime_phase_summary(
        before,
        after,
        2.0,
    )

    assert summary['runtime_runs'] == 3
    assert summary['encoded_texts'] == 80
    assert summary['batch_successes'] == 3
    assert summary['average_batch_size'] == 80 / 3
    assert summary['runtime_seconds'] == 1.5
    assert summary['non_runtime_seconds'] == 0.5
    assert summary['runtime_wall_share'] == 0.75
    assert summary['single_request_execution'] is False
    assert summary['last_batch_size'] == 16
    assert summary['max_observed_batch_size'] == 128


def test_runtime_phase_summary_identifies_single_query_execution() -> None:
    before = {'runtime_runs': 10, 'encoded_texts': 100}
    after = {'runtime_runs': 15, 'encoded_texts': 105}

    summary = benchmark_evoke_product_path.runtime_phase_summary(
        before,
        after,
        0.25,
    )

    assert summary['runtime_runs'] == 5
    assert summary['encoded_texts'] == 5
    assert summary['average_batch_size'] == 1.0
    assert summary['single_request_execution'] is True


def test_atom_outputs_match_allows_only_small_weight_roundoff() -> None:
    single = {'atoms': [1, 2], 'weights': [0.5, 1.25]}

    assert benchmark_evoke_product_path.atom_outputs_match(
        single,
        {'atoms': [1, 2], 'weights': [0.5000001, 1.250001]},
    )
    assert not benchmark_evoke_product_path.atom_outputs_match(
        single,
        {'atoms': [2, 1], 'weights': [0.5, 1.25]},
    )
    assert not benchmark_evoke_product_path.atom_outputs_match(
        single,
        {'atoms': [1, 2], 'weights': [0.5, 1.3]},
    )


def test_configure_postgres_records_runtime_thread_cap(tmp_path) -> None:
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    config = data_dir / 'postgresql.conf'
    config.write_text('', encoding='utf-8')

    benchmark_evoke_product_path.configure_postgres(
        data_dir,
        SimpleNamespace(
            shared_cache_mb=512,
            onnxruntime_intra_op_threads=4,
            extension_libdir=tmp_path / 'lib',
            extension_control_dir=tmp_path / 'extension',
        ),
    )

    text = config.read_text(encoding='utf-8')
    assert "evoke.shared_runtime_size = '512MB'" in text
    assert 'evoke.onnxruntime_intra_op_threads = 4' in text
    assert "dynamic_library_path = '" in text
    assert str(tmp_path / 'lib') in text
    assert "extension_control_path = '" in text
    assert str(tmp_path / 'extension') in text


def test_extension_control_root_accepts_share_or_extension_dir(
    tmp_path,
) -> None:
    extension_dir = tmp_path / 'extension'
    extension_dir.mkdir()
    (extension_dir / 'evoke.control').write_text('', encoding='utf-8')

    assert (
        benchmark_evoke_product_path.extension_control_root(tmp_path)
        == tmp_path.resolve()
    )
    assert (
        benchmark_evoke_product_path.extension_control_root(extension_dir)
        == tmp_path.resolve()
    )


def test_extension_control_root_rejects_missing_control(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match='evoke.control is missing'):
        benchmark_evoke_product_path.extension_control_root(tmp_path)


def test_cleanup_product_cluster_removes_ephemeral_workdir(tmp_path) -> None:
    root = tmp_path / 'product-cluster'
    data_dir = root / 'data'
    data_dir.mkdir(parents=True)

    benchmark_evoke_product_path.cleanup_product_cluster(
        root,
        tmp_path,
        data_dir,
        started=False,
        keep_pg=False,
    )

    assert not root.exists()


def test_cleanup_product_cluster_preserves_explicit_keep(tmp_path) -> None:
    root = tmp_path / 'product-cluster'
    data_dir = root / 'data'
    data_dir.mkdir(parents=True)

    benchmark_evoke_product_path.cleanup_product_cluster(
        root,
        tmp_path,
        data_dir,
        started=False,
        keep_pg=True,
    )

    assert root.is_dir()


def test_cleanup_product_cluster_preserves_failed_stop(
    monkeypatch,
    tmp_path,
) -> None:
    root = tmp_path / 'product-cluster'
    data_dir = root / 'data'
    data_dir.mkdir(parents=True)
    monkeypatch.setattr(
        benchmark_evoke_product_path,
        'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    benchmark_evoke_product_path.cleanup_product_cluster(
        root,
        tmp_path,
        data_dir,
        started=True,
        keep_pg=False,
    )

    assert root.is_dir()
