import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = (
    REPO_ROOT / 'docs/technical-report-evoke-model.md',
    REPO_ROOT / 'docs/technical-report-evoke-model-zh.md',
)


def display_formulas(source: str) -> list[str]:
    return re.findall(r'^```math\n(.*?)\n```$', source, re.M | re.S)


def test_bilingual_model_report_formulas_match() -> None:
    english, chinese = (
        display_formulas(path.read_text(encoding='utf-8'))
        for path in REPORTS
    )
    assert english
    assert english == chinese


@pytest.mark.parametrize('path', REPORTS, ids=lambda path: path.name)
def test_model_report_uses_github_compatible_math(path: Path) -> None:
    source = path.read_text(encoding='utf-8')
    # GitHub's page renderer rejects this macro even when its API accepts it.
    assert r'\operatorname' not in source
    # Fences protect LaTeX spacing escapes from Markdown punctuation unescaping.
    assert not re.search(r'^\$\$', source, re.M)
    assert len(display_formulas(source)) == source.count('```math\n')
