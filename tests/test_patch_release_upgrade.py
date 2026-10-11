import hashlib
from pathlib import Path

import pytest

from scripts.run_product_maturity_suite import extension_sql_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_patch_install_catalog_matches_published_025() -> None:
    # Reviewed against public v0.2.5, not an intermediate development tag.
    published_sha256 = (
        '9401cf1d7a106145ffd75f5c87b09edcb59737ee80cff478c2d459d703887da9'
    )
    catalog = (ROOT / 'sql/evoke--0.2.6.sql').read_bytes()
    assert hashlib.sha256(catalog).hexdigest() == published_sha256


def test_patch_upgrade_does_not_modify_sql_objects() -> None:
    path = ROOT / 'sql/evoke--0.2.5--0.2.6.sql'
    lines = path.read_text(encoding='utf-8').splitlines()
    assert lines[0] == (
        '\\echo Use "ALTER EXTENSION evoke UPDATE TO \'0.2.6\'" '
        'to load this file. \\quit'
    )
    assert all(not line.strip() or line.startswith('--') for line in lines[1:])
    assert path.relative_to(ROOT).as_posix() in (
        ROOT / 'Makefile'
    ).read_text(encoding='utf-8')


def test_package_requires_exact_reviewed_patch_scripts(tmp_path: Path) -> None:
    install = tmp_path / 'evoke--0.2.6.sql'
    upgrade = tmp_path / 'evoke--0.2.5--0.2.6.sql'
    install.touch()
    with pytest.raises(ValueError, match='reviewed patch upgrade'):
        extension_sql_artifacts(tmp_path, '0.2.6')
    upgrade.touch()
    assert extension_sql_artifacts(tmp_path, '0.2.6') == sorted([
        install, upgrade,
    ])
    (tmp_path / 'evoke--0.2.4--0.2.6.sql').touch()
    with pytest.raises(ValueError, match='reviewed patch upgrade'):
        extension_sql_artifacts(tmp_path, '0.2.6')
