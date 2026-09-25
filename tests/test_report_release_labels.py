import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_HEADINGS = {
    'docs/technical-report-evoke-model.md':
        '# Evoke Model Technical Report (Beta 1)',
    'docs/technical-report-evoke-model-zh.md':
        '# Evoke 模型技術報告 (Beta 1)',
    'docs/technical-report-evoke-system.md':
        '## System Technical Report (Beta 1)',
    'docs/technical-report-evoke-system-zh.md':
        '## 系統技術報告 (Beta 1)',
}


@pytest.mark.parametrize('relative_path, heading', REPORT_HEADINGS.items())
def test_technical_report_beta1_identity(
    relative_path: str, heading: str,
) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
    assert heading in source.splitlines()
    assert not re.search(r'prerelease|first official|首個正式|預發布', source,
                         re.I)


@pytest.mark.parametrize('relative_path', (
    'README.md', 'docs/README.md', 'docs/model-planning.md',
))
def test_report_reference_uses_beta1(relative_path: str) -> None:
    source = (REPO_ROOT / relative_path).read_text(encoding='utf-8')
    assert 'Beta 1' in source
    assert not re.search(r'first[- ](?:official|release)', source, re.I)
