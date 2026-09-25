from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SOURCE = (ROOT / 'evoke.control').read_text(encoding='utf-8')
VERSION = CONTROL_SOURCE.split("default_version = '", 1)[1].split("'", 1)[0]
AM_SOURCE = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
OPTIONS_SOURCE = (ROOT / 'src' / 'evoke_am_options.c').read_text(
    encoding='utf-8',
)
OPTIONS_HEADER = (ROOT / 'src' / 'evoke_am_options.h').read_text(
    encoding='utf-8',
)
SQL_SOURCE = (ROOT / 'sql' / f'evoke--{VERSION}.sql').read_text(
    encoding='utf-8',
)
REBUILD_SOURCE = (ROOT / 'scripts' / 'rebuild_evoke_indexes.py').read_text(
    encoding='utf-8',
)
LIFECYCLE_SOURCE = (
    ROOT / 'scripts' / 'test_convergent_sae_lifecycle_smoke.py'
).read_text(encoding='utf-8')


def test_alpha_mass_is_an_explicit_sae_reloption() -> None:
    assert 'double semantic_alpha_mass;' in OPTIONS_SOURCE
    assert '"semantic_alpha_mass",\n        RELOPT_TYPE_REAL' in OPTIONS_SOURCE
    sae_options = OPTIONS_SOURCE.split(
        'static const char *sae_option_names[] = {',
        1,
    )[1].split('};', 1)[0]
    assert '"semantic_alpha_mass"' in sae_options
    assert 'evoke_am_semantic_alpha_mass(Relation index_relation)' in (
        OPTIONS_SOURCE
    )
    assert 'double evoke_am_semantic_alpha_mass(' in OPTIONS_HEADER
    assert '0.01,\n        1.0,' in OPTIONS_SOURCE


def test_alpha_mass_covers_initial_and_eventual_publication() -> None:
    assert 'evoke_am_semantic_pair_alpha_mass_keep_count(' in AM_SOURCE
    assert 'evoke_am_convergent_semantic_alpha_mass_keep_count(' in AM_SOURCE
    assert AM_SOURCE.count('evoke_am_semantic_alpha_mass(') >= 3


def test_nondefault_alpha_mass_is_generation_bound() -> None:
    contract = AM_SOURCE.split(
        'evoke_am_runtime_signature_for_contract(',
        1,
    )[1].split('static void\nevoke_am_runtime_signature(', 1)[0]
    assert 'if (semantic_alpha_mass < 1.0)' in contract
    assert '"semantic_alpha_mass=%016llx;"' in contract
    assert 'evoke_am_semantic_alpha_mass(indexRelation)' in contract


def test_alpha_mass_is_reported_and_preserved_by_rebuilds() -> None:
    assert "'semantic_alpha_mass', semantic_alpha_mass" in SQL_SOURCE
    assert "'semantic_accuracy_profile', CASE" in SQL_SOURCE
    assert "'semantic_alpha_mass'," in REBUILD_SOURCE


def test_alpha_mass_has_a_lifecycle_gate() -> None:
    assert "'--semantic-alpha-mass'" in LIFECYCLE_SOURCE
    assert 'run_semantic_alpha_contract_audit(' in LIFECYCLE_SOURCE
    assert "'semantic_alpha_mass_is_generation_bound'" in LIFECYCLE_SOURCE
    assert 'does not match the configured index contract' in SQL_SOURCE
    assert (
        'runtime_precision, semantic_impact_precision, '
        in SQL_SOURCE
    )
    assert 'semantic_alpha_mass, the default' in SQL_SOURCE
