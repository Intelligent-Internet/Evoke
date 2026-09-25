from scripts.evaluate_evoke_beir15_native_baselines import (
    filter_queries,
)
from scripts.evaluate_evoke_beir15_native_baselines import (
    read_query_ids as read_baseline_query_ids,
)


def test_query_filter_reads_comments_and_keeps_ids(tmp_path) -> None:
    query_ids_path = tmp_path / 'query_ids.txt'
    query_ids_path.write_text('# comment\nq2\n\nq3\n', encoding='utf-8')
    queries = [
        {'query_id': 'q1'},
        {'query_id': 'q2'},
        {'query_id': 'q3'},
    ]

    filtered = filter_queries(
        queries,
        query_ids=read_baseline_query_ids(query_ids_path),
    )

    assert [query['query_id'] for query in filtered] == ['q2', 'q3']
