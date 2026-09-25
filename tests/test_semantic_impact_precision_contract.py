from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL_SOURCE = (ROOT / 'evoke.control').read_text(encoding='utf-8')
VERSION = CONTROL_SOURCE.split("default_version = '", 1)[1].split("'", 1)[0]
AM_SOURCE = (ROOT / 'src' / 'evoke_am.c').read_text(encoding='utf-8')
OPTIONS_SOURCE = (ROOT / 'src' / 'evoke_am_options.c').read_text(
    encoding='utf-8',
)
PAGES_SOURCE = (ROOT / 'src' / 'evoke_segment_pages.c').read_text(
    encoding='utf-8',
)
SEGMENTS_SOURCE = (ROOT / 'src' / 'evoke_segments.c').read_text(
    encoding='utf-8',
)
BMP_HEADER = (ROOT / 'src' / 'evoke_semantic_bmp.h').read_text(
    encoding='utf-8',
)
BMP_SOURCE = (ROOT / 'src' / 'evoke_semantic_bmp.c').read_text(
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


def test_impact_precision_is_an_explicit_sae_reloption() -> None:
    for value in ('f32', 'fp16', 'u8'):
        assert f'{{"{value}", EVOKE_SEMANTIC_IMPACT_PRECISION_' in (
            OPTIONS_SOURCE
        )
    assert '"semantic_impact_precision"' in OPTIONS_SOURCE
    sae_options = OPTIONS_SOURCE.split(
        'static const char *sae_option_names[] = {',
        1,
    )[1].split('};', 1)[0]
    assert '"semantic_impact_precision"' in sae_options


def test_precision_is_one_generation_bound_authority() -> None:
    contract = AM_SOURCE.split(
        'evoke_am_runtime_signature_for_contract(',
        1,
    )[1].split('static void\nevoke_am_runtime_signature(', 1)[0]
    assert 'semantic_impact_precision' in contract
    assert 'EVOKE_SEMANTIC_IMPACT_PRECISION_F32' in contract
    assert PAGES_SOURCE.count(
        'evoke_am_get_semantic_impact_precision(index_relation)'
    ) >= 3
    assert 'bundle->semantic_impact_precision' in SEGMENTS_SOURCE
    assert 'payload->semantic_impact_precision' in SEGMENTS_SOURCE


def test_packed_authority_supports_all_three_precisions() -> None:
    assert 'EVOKE_SEMANTIC_IMPACT_PRECISION_F32 = 32' in BMP_HEADER
    assert 'EVOKE_SEMANTIC_IMPACT_PRECISION_FP16 = 16' in BMP_HEADER
    assert 'EVOKE_SEMANTIC_IMPACT_PRECISION_U8 = 8' in BMP_HEADER
    assert 'evoke_semantic_bmp_packed_impact_decode(' in BMP_HEADER


def test_packed_decoder_uses_the_block64_impact_limit() -> None:
    decoder = BMP_SOURCE.split(
        'evoke_semantic_bmp_packed_impacts_decode(',
        1,
    )[1].split('\nevoke_status\nevoke_semantic_bmp_deserialize(', 1)[0]
    assert 'EVOKE_SEMANTIC_BMP_PACKED_BLOCK_DOCUMENTS' in decoder
    assert 'EVOKE_SEMANTIC_BMP_BLOCK_SHIFT' not in decoder


def test_precision_is_reported_rebuilt_and_lifecycle_gated() -> None:
    assert "'semantic_impact_precision', semantic_impact_precision" in (
        SQL_SOURCE
    )
    assert "'--semantic-impact-precision'" in REBUILD_SOURCE
    assert 'run_semantic_impact_precision_contract_audit(' in (
        LIFECYCLE_SOURCE
    )
    assert 'semantic_impact_precision_is_generation_bound' in (
        LIFECYCLE_SOURCE
    )
