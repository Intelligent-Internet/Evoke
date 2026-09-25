import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    REPO_ROOT / 'docs/technical-report-evoke-system.md',
    REPO_ROOT / 'docs/technical-report-evoke-system-zh.md',
)


def fenced_blocks(source: str, language: str) -> list[str]:
    return re.findall(
        rf'^```{language}\n(.*?)\n```$', source, re.M | re.S
    )


@pytest.mark.parametrize('language', ('math', 'text', 'sql'))
def test_system_report_bilingual_blocks_match(language: str) -> None:
    english, chinese = (
        fenced_blocks(path.read_text(encoding='utf-8'), language)
        for path in REPORTS
    )
    assert english
    assert english == chinese


@pytest.mark.parametrize('path', REPORTS, ids=lambda path: path.name)
def test_system_report_math_and_diagrams(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    assert r'\operatorname' not in source
    assert not re.search(r'^\$\$', source, re.M)
    assert len(fenced_blocks(source, 'math')) == source.count('```math\n')
    for diagram in fenced_blocks(source, 'text'):
        assert diagram.isascii()
    # Punctuation immediately before an opening dollar can defeat GitHub math.
    for match in re.finditer(r'\$([^$\n]+)\$', source):
        assert match.start() == 0 or source[match.start() - 1].isspace()


def test_system_report_bilingual_inline_math_matches() -> None:
    english, chinese = (
        Counter(re.findall(
            r'\$([^$\n]+)\$', path.read_text(encoding='utf-8')
        ))
        for path in REPORTS
    )
    assert english
    assert english == chinese


@pytest.mark.parametrize('path', REPORTS, ids=lambda path: path.name)
def test_system_report_local_links_exist(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', source):
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        assert (path.parent / unquote(url.path)).exists(), target


@pytest.mark.parametrize('path', REPORTS, ids=lambda path: path.name)
def test_system_report_package_identity(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    lock = json.loads(
        (REPO_ROOT / 'packaging/milestone-model.json').read_text(
            encoding='utf-8'
        )
    )
    identities = (
        lock['bundle_name'],
        lock['manifest_sha256'],
        lock['manifest_contract']['model_id'],
        lock['manifest_contract']['runtime_abi'],
        lock['upstream']['revision'],
        (REPO_ROOT / 'packaging/onnxruntime.version').read_text(
            encoding='utf-8'
        ).strip(),
    )
    for identity in identities:
        assert f'`{identity}`' in source


def numeric_table_cells(source: str) -> list[str]:
    return [
        cell.strip()
        for line in source.splitlines()
        if line.startswith('|')
        for cell in line.split('|')[1:-1]
        if re.fullmatch(r'[\d,. /]+', cell) and re.search(r'\d', cell)
    ]


def test_system_report_bilingual_measurements_match() -> None:
    english, chinese = (
        numeric_table_cells(path.read_text(encoding='utf-8'))
        for path in REPORTS
    )
    assert english
    assert english == chinese


@pytest.mark.parametrize('path', REPORTS, ids=lambda path: path.name)
def test_system_report_bm25_timings_match_record(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    record_path = REPO_ROOT / (
        'docs/performance/data/diagnostics/'
        'bm25-page-native-regression-2026-08-18.json'
    )
    record = json.loads(record_path.read_text(encoding='utf-8'))
    for key in (
        'historical_official_psql_bm25s',
        'ii42_before_repair',
        'ii42_after_repair',
    ):
        row = record[key]
        values = ' | '.join(
            f'{row[metric]:.3f}'
            for metric in ('mean_ms', 'p50_ms', 'p95_ms')
        )
        assert f'| {values} |' in source
