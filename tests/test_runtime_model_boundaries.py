from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SOURCE = REPO_ROOT / 'src/evoke_semantic.c'


def function_source(name: str, next_name: str) -> str:
    source = SEMANTIC_SOURCE.read_text(encoding='utf-8')
    start = source.index(name)
    end = source.index(next_name, start)
    return source[start:end]


def test_manifest_and_text_artifact_limits_are_not_swapped() -> None:
    manifest_reader = function_source(
        'evoke_checkout_read_manifest_text',
        'evoke_checkout_artifact_path',
    )
    artifact_reader = function_source(
        'evoke_checkout_read_artifact_text',
        'evoke_checkout_is_hex_sha256',
    )

    assert 'EVOKE_CHECKOUT_MAX_MANIFEST_BYTES' in manifest_reader
    assert 'EVOKE_CHECKOUT_MAX_ARTIFACT_TEXT_BYTES' not in manifest_reader
    assert 'EVOKE_CHECKOUT_MAX_ARTIFACT_TEXT_BYTES' in artifact_reader
    assert 'EVOKE_CHECKOUT_MAX_MANIFEST_BYTES' not in artifact_reader
