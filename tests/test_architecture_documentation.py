import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = REPO_ROOT / 'docs/architecture-and-design.md'
CONVERGENT_DESIGN = REPO_ROOT / 'docs/convergent-segmented-index.md'
ARCHIVE = REPO_ROOT / 'docs/archive/engineering'
ARCHIVED_REPORTS = (
    'query-first-eventual-background-maintenance.md',
    'eventual-sae-release-readiness-history.md',
    'product-roadmap-through-v0.2.5.md',
)
DOCUMENTS = (
    REPO_ROOT / 'README.md',
    REPO_ROOT / 'CONTRIBUTING.md',
    *sorted((REPO_ROOT / 'docs').glob('*.md')),
    *sorted((REPO_ROOT / 'docs/examples').glob('*.md')),
    ARCHIVE / 'README.md',
    *(ARCHIVE / name for name in ARCHIVED_REPORTS),
)


@pytest.mark.parametrize('path', DOCUMENTS, ids=lambda path: path.name)
def test_architecture_documentation_local_links_exist(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    targets = re.findall(r'\[[^\]]+\]\(([^)]+)\)', source)
    for target in targets:
        url = urlsplit(target)
        if url.scheme or not url.path:
            continue
        assert (path.parent / unquote(url.path)).exists(), target


def test_architecture_source_map_references_existing_entry_points() -> None:
    source = ARCHITECTURE.read_text(encoding='utf-8')
    source_map = source.split('## Implementation Map\n', 1)[1]
    source_map = source_map.split('\n## ', 1)[0]
    rows = re.findall(
        r'^\|[^\n]+\| \[[^\]]+\]\(([^)]+)\) \| ([^\n]+) \|$',
        source_map,
        re.M,
    )
    assert rows
    for target, entries in rows:
        implementation = (ARCHITECTURE.parent / target).read_text(
            encoding='utf-8'
        )
        symbols = re.findall(r'`(evoke_[a-z0-9_]+)`', entries)
        assert symbols, target
        for symbol in symbols:
            # Match definitions, not a mention in a comment or a call site.
            prefix = r'^CREATE FUNCTION ' if target.endswith('.sql') else '^'
            assert re.search(
                prefix + re.escape(symbol) + r'\s*\(',
                implementation,
                re.M,
            ), (target, symbol)


@pytest.mark.parametrize('path', (ARCHITECTURE, CONVERGENT_DESIGN))
def test_architecture_diagrams_are_ascii_and_fences_are_balanced(
    path: Path,
) -> None:
    source = path.read_text(encoding='utf-8')
    diagrams = re.findall(r'^```text\n(.*?)\n```$', source, re.M | re.S)
    assert diagrams
    assert source.count('```') == 2 * len(diagrams)
    for diagram in diagrams:
        assert diagram.isascii()


@pytest.mark.parametrize(
    ('source_path', 'macro', 'description'),
    (
        (
            'src/evoke_am.c',
            'EVOKE_AM_FILTER_RESOLVE_MAX_ROWS',
            '{value:,}-match limit',
        ),
        (
            'src/evoke_scope_pg.c',
            'EVOKE_SCOPE_SNAPSHOT_BATCH_DOCUMENTS',
            'at most {value} document slots',
        ),
    ),
)
@pytest.mark.parametrize('path', (ARCHITECTURE, CONVERGENT_DESIGN))
def test_architecture_batch_limits_match_source(
    source_path: str,
    macro: str,
    description: str,
    path: Path,
) -> None:
    implementation = (REPO_ROOT / source_path).read_text(encoding='utf-8')
    match = re.search(
        rf'^#define {re.escape(macro)} UINT(?:32|64)_C\((\d+)\)$',
        implementation,
        re.M,
    )
    assert match is not None, macro
    source = path.read_text(encoding='utf-8')
    assert description.format(value=int(match.group(1))) in source


@pytest.mark.parametrize(
    ('source_path', 'macro', 'description'),
    (
        (
            'src/evoke_lexicon_cow.h',
            'EVOKE_LEXICON_COW_BUCKET_TARGET_BYTES',
            'bucket target is {value:,} bytes',
        ),
        (
            'src/evoke_prefix_cow.h',
            'EVOKE_PREFIX_COW_LEAF_MAX_ENTRIES',
            'Prefix leaves hold at most {value} entries',
        ),
        (
            'src/evoke_prefix_cow.h',
            'EVOKE_PREFIX_COW_NODE_MAX_CHILDREN',
            'internal nodes at most {value} child',
        ),
        (
            'src/evoke_segments.h',
            'EVOKE_SEGMENT_MANIFEST_VERSION',
            'current manifest writer emits version {value}',
        ),
        (
            'src/evoke_segments.h',
            'EVOKE_TERM_DIRECTORY_MAX_EXTENTS_PER_TERM',
            'At most {value} extents per term',
        ),
    ),
)
def test_convergent_storage_limits_match_source(
    source_path: str,
    macro: str,
    description: str,
) -> None:
    implementation = (REPO_ROOT / source_path).read_text(encoding='utf-8')
    match = re.search(
        rf'^#define {re.escape(macro)} UINT(?:16|32|64)_C\((\d+)\)$',
        implementation,
        re.M,
    )
    assert match is not None, macro
    source = CONVERGENT_DESIGN.read_text(encoding='utf-8')
    assert description.format(value=int(match.group(1))) in source


def test_maintenance_reclaim_threshold_matches_source() -> None:
    implementation = (REPO_ROOT / 'src/evoke_am.c').read_text(encoding='utf-8')
    match = re.search(
        r'^#define EVOKE_AM_RECLAIM_RETIRED_RANGE_THRESHOLD '
        r'UINT32_C\((\d+)\)$',
        implementation,
        re.M,
    )
    assert match is not None
    document = (REPO_ROOT / 'docs/maintenance-lifecycle.md').read_text(
        encoding='utf-8'
    )
    assert f'either {match.group(1)} retired ranges' in document


@pytest.mark.parametrize('name', ARCHIVED_REPORTS)
def test_engineering_archive_preserves_and_indexes_reports(name: str) -> None:
    index = (ARCHIVE / 'README.md').read_text(encoding='utf-8')
    assert f']({name})' in index
    source = (ARCHIVE / name).read_text(encoding='utf-8')
    assert '## Archive Guide' in source
    assert '](README.md)' in source
    headings = set()
    for heading in re.findall(r'^#{1,6} (.+)$', source, re.M):
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        headings.add(slug)
    for fragment in re.findall(r'\]\(#([^)]+)\)', source):
        assert fragment in headings, (name, fragment)


@pytest.mark.parametrize(
    'relative',
    (
        'docs/research-sae/reports/designs/evoke-rebrand-plan.md',
        'docs/research-sae/reports/designs/evoke-sae-productization-plan.md',
        'docs/performance/reports/maintenance-pre-merge-review.md',
        'docs/performance/reports/maintenance-final-status.md',
    ),
)
def test_retired_intermediate_plans_are_not_active_documents(
    relative: str,
) -> None:
    assert not (REPO_ROOT / relative).exists()
    name = Path(relative).name
    for path in DOCUMENTS:
        assert name not in path.read_text(encoding='utf-8'), path
