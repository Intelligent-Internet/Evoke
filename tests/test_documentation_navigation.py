from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
# Current guides, release reports, archive entrypoints, and relocated records.
# Raw historical experiment ledgers are not rewritten as current contracts.
DOCUMENTS = sorted({
    *ROOT.glob('*.md'),
    *DOCS.glob('*.md'),
    *(DOCS / 'examples').glob('*.md'),
    *(DOCS / 'performance').glob('*.md'),
    *DOCS.rglob('README.md'),
    DOCS / 'archive/engineering/psql-bm25s-project-narrative.md',
    DOCS / 'performance/reports/semantic-accelerator-bounded-execution.md',
})


def prose(source: str) -> str:
    return re.sub(
        r'^(`{3,}|~{3,})[^\n]*\n.*?^\1\s*$',
        '', source, flags=re.MULTILINE | re.DOTALL,
    )


def anchors(source: str) -> set[str]:
    source = prose(source)
    result = set(re.findall(r'<a\s+(?:id|name)=[\'"]([^\'"]+)', source))
    counts: dict[str, int] = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', source, re.MULTILINE):
        heading = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', heading)
        heading = html.unescape(re.sub(r'<[^>]+>', '', heading))
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f'{slug}-{count}' if count else slug)
    return result


def links(source: str) -> list[str]:
    source = prose(source)
    return [
        *re.findall(r'\[[^]\n]*\]\(([^\s)]+)\)', source),
        *re.findall(r'(?:src|href)=[\'"]([^\'"]+)', source),
        *re.findall(r'^\[[^]]+\]:\s*<?([^\s>]+)', source, re.MULTILINE),
    ]


@pytest.mark.parametrize('path', DOCUMENTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_main_document_links_and_fragments(path: Path) -> None:
    for target in links(path.read_text(encoding='utf-8')):
        url = urlsplit(target)
        if url.scheme or url.netloc:
            continue
        assert not url.path.startswith('/'), (path, target)
        destination = (path.parent / unquote(url.path)).resolve() if url.path else path
        assert destination.is_relative_to(ROOT), (path, target)
        assert destination.exists(), (path, target)
        if url.fragment and destination.suffix == '.md':
            assert unquote(url.fragment) in anchors(
                destination.read_text(encoding='utf-8')
            ), (path, target)


def test_heading_fragment_parser_handles_code_duplicates_and_unicode() -> None:
    assert anchors(
        '# `evoke_query` & SQL\n## 模型設計\n## Repeat\n## Repeat\n'
        '```text\n# Not A Heading\n```\n'
    ) == {'evoke_query--sql', '模型設計', 'repeat', 'repeat-1'}


def test_document_map_reaches_every_main_guide() -> None:
    index = DOCS / 'README.md'
    reachable = {index}
    pending = [index]
    while pending:
        path = pending.pop()
        for target in links(path.read_text(encoding='utf-8')):
            url = urlsplit(target)
            if url.scheme or not url.path:
                continue
            destination = (path.parent / unquote(url.path)).resolve()
            if (destination.is_file() and destination.suffix == '.md'
                    and destination in DOCUMENTS and destination not in reachable):
                reachable.add(destination)
                pending.append(destination)
    required = {*DOCS.glob('*.md'), *(DOCS / 'examples').glob('*.md')}
    assert not required - reachable


def test_multicolumn_example_joins_the_indexed_table() -> None:
    source = (DOCS / 'multicolumn-indexes.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```sql\n(.*?)\n```', source, re.DOTALL)
    query = next(block for block in blocks if "'docs_field_semantic_idx'" in block)
    assert 'JOIN docs_text AS d ON d.ctid = h.ctid' in query


def test_fusion_examples_only_select_declared_columns() -> None:
    source = (DOCS / 'multi-index-fusion.md').read_text(encoding='utf-8')
    ddl = re.search(r'CREATE TABLE docs \((.*?)\);', source, re.DOTALL)
    assert ddl is not None
    columns = set(re.findall(r'^\s*(\w+)\s+', ddl.group(1), re.MULTILINE))
    selected = set(re.findall(r'\bd\.(\w+)\b', source))
    assert selected <= columns | {'ctid'}


def test_scoring_profile_uses_the_current_sparse_contract() -> None:
    source = (DOCS / 'examples/scoring-profile-reference.md').read_text(
        encoding='utf-8'
    )
    assert 'p2_unified_sparse_dot_v1' in source
    assert '"score_inputs": ["lexical_postings", "semantic_postings"]' in source
    assert '"weights"' not in source
    assert 'abbreviated excerpt' in source


def test_cmake_instructions_configure_build_and_run_tests_in_order() -> None:
    source = (DOCS / 'testing-and-validation.md').read_text(encoding='utf-8')
    commands = ('cmake -S . -B build_tmp', 'cmake --build build_tmp',
                'ctest --test-dir build_tmp')
    offsets = [source.index(command) for command in commands]
    assert offsets == sorted(offsets)


@pytest.mark.parametrize('name', (
    'api-reference.md',
    'multi-index-fusion.md',
    'hybrid-search.md',
    'hybrid-fusion-engine.md',
))
def test_composition_documents_keep_same_table_tid_identity(name: str) -> None:
    source = (DOCS / name).read_text(encoding='utf-8')
    normalized = ' '.join(source.split())
    assert 'same base table' in normalized
    assert 'application SQL' in normalized
    assert 'document ID' in normalized
    assert 'hybrid candidates](hybrid-search.md) with explicit' not in source


def test_migration_guide_explains_operator_cutover_not_just_inventory() -> None:
    source = (DOCS / 'upgrading.md').read_text(encoding='utf-8')
    assert '`migrate_psql_bm25s_text_like`' in source
    assert '`--mode lexical-only` filters actions' in source
    assert 'drops the old index' in source
    assert 'rollback window' in source


def test_performance_guide_separates_frozen_evidence_from_public_api() -> None:
    source = (DOCS / 'performance/README.md').read_text(encoding='utf-8')
    assert '## Frozen Lexical Cross-Engine Reference' in source
    assert 'not recommended application entrypoints' in source
    assert 'does not rerun a benchmark against Evoke' in source
    assert 'per-dataset query win' in source
