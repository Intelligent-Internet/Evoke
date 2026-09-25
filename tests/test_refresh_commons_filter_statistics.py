from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SQL = (
    ROOT / 'scripts' / 'refresh_commons_filter_statistics.sql'
).read_text(encoding='utf-8')


def test_statistics_refresh_requires_normal_production_policy() -> None:
    assert "current_setting('autovacuum') <> 'on'" in SQL
    assert "current_setting('track_counts') <> 'on'" in SQL
    assert 'ALTER SYSTEM' not in SQL
    assert 'SET enable_' not in SQL
    assert 'attribute.attgenerated' in SQL
    assert 'pg_get_expr(' in SQL
    assert "actual_generated IS DISTINCT FROM 's'" in SQL
    assert 'invalid generated-column contract' in SQL


def test_statistics_refresh_covers_every_filter_authority() -> None:
    required_columns = (
        'publish_date',
        'categories',
        'organizations',
        'publish_date_start_bound',
        'publish_date_end_bound',
        'publish_date_has_day',
        'journal_title',
        'nlm_ta',
        'policy_ca_doc_id',
        'policy_tx_doc_id',
        'policy_wa_doc_id',
        'document_id',
    )

    for column in required_columns:
        assert f"'{column}'" in SQL
    assert 'ANALYZE commons.data_arxiv' in SQL
    assert 'ANALYZE commons.data_pubmed' in SQL
    assert 'ANALYZE commons.data_policy_ca_chunks' in SQL
    assert 'ANALYZE commons.data_policy_tx_chunks' in SQL
    assert 'ANALYZE commons.data_policy_wa_chunks' in SQL
    assert 'ANALYZE commons.sys_chunks' in SQL


def test_statistics_refresh_does_not_touch_evoke_relations() -> None:
    analyze_lines = [
        line.strip()
        for line in SQL.splitlines()
        if line.lstrip().startswith('ANALYZE ')
    ]

    assert analyze_lines
    assert all(line.startswith('ANALYZE commons.') for line in analyze_lines)
    assert all('__' not in line for line in analyze_lines)
