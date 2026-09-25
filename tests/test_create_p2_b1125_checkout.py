import json
import struct
import subprocess
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def test_create_p2_b1125_checkout_artifacts(tmp_path: Path) -> None:
    vocabulary = tmp_path / 'vocabulary.json'
    queries = tmp_path / 'queries.jsonl'
    output = tmp_path / 'p2_checkout'
    vocabulary.write_text(
        json.dumps(['alpha', 'beta', 'gamma']) + '\n',
        encoding='utf-8',
    )
    queries.write_text(
        json.dumps(
            {
                'atom_ids': [0, 3, 6],
                'atom_impacts': [1.0, 0.5, 0.25],
                'id': 'query-a',
            },
        ) + '\n',
        encoding='utf-8',
    )

    subprocess.run(
        [
            sys.executable,
            'scripts/create_p2_b1125_checkout.py',
            '--output',
            str(output),
            '--dataset',
            'smoke',
            '--lexical-vocabulary',
            str(vocabulary),
            '--query-atoms',
            str(queries),
            '--semantic-dims',
            '4',
            '--model-id',
            'ii42_p2_test',
        ],
        check=True,
    )

    manifest = read_json(output / 'manifest.json')
    scoring = read_json(output / 'calibration/scoring_profile.json')
    assert scoring['document_compiler']['semantic_encoder'][
        'score_scale'
    ] == 1.0
    atom_space = read_json(output / 'sae/atom_space.json')
    contract = read_json(output / 'provenance/p2_contract.json')

    assert manifest['model_id'] == 'ii42_p2_test'
    assert manifest['latent_dims'] == 7
    assert manifest['runtime'] == 'offline_precomputed_atoms'
    assert 'runtime_parameters' not in manifest
    assert manifest['index_compatibility']['scoring'] == (
        'p2_unified_sparse_dot_v1'
    )

    assert scoring['profile'] == 'p2_unified_sparse_dot_v1'
    assert scoring['score_inputs'] == [
        'lexical_postings',
        'semantic_postings',
    ]
    assert scoring['document_compiler']['semantic_budget_ratio_to_lexical'] == (
        1.125
    )
    assert scoring['query_compiler']['mode'] == 'rms'
    assert scoring['query_compiler']['multiplier'] == 4.0
    assert scoring['query_compiler']['semantic_encoder']['active_dims'] == 50
    assert scoring['query_compiler']['semantic_encoder']['pooling'] == (
        'relu_log1p_masked_max'
    )

    assert atom_space['lexical']['start'] == 0
    assert atom_space['lexical']['end_exclusive'] == 3
    assert atom_space['semantic']['start'] == 3
    assert atom_space['semantic']['end_exclusive'] == 7
    assert contract['query_stats']['rows'] == 1
    assert contract['route'] == 'M1934-b1.125-rms_m4'
    assert contract['m1914_transform']['selected_branch'] == 'global_power'


def test_create_p2_b1125_native_runtime_checkout(tmp_path: Path) -> None:
    vocabulary = tmp_path / 'vocabulary.json'
    queries = tmp_path / 'queries.jsonl'
    runtime = tmp_path / 'runtime'
    calibration = tmp_path / 'query_calibration_rms.f32'
    output = tmp_path / 'p2_native_checkout'
    vocabulary.write_text(json.dumps(['alpha', 'beta']) + '\n')
    queries.write_text(json.dumps({
        'atom_ids': [0, 2],
        'atom_impacts': [1.0, 0.5],
        'id': 'query-a',
    }) + '\n')
    runtime.mkdir()
    (runtime / 'semantic_document_compiler.onnx').write_bytes(
        b'document-onnx-fixture',
    )
    (runtime / 'semantic_query_compiler.onnx').write_bytes(b'onnx-fixture')
    (runtime / 'semantic_runtime.json').write_text(json.dumps({
        'abi': 'evoke_p2_semantic_onnx_v2',
    }) + '\n')
    for name in (
        'merges.txt',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
    ):
        (runtime / name).write_text('{}\n')
    calibration.write_bytes(struct.pack(
        '<8sIIQ6f',
        b'II42P2R1',
        1,
        6,
        2,
        1.0,
        0.5,
        0.25,
        0.125,
        0.0,
        2.0,
    ))

    subprocess.run(
        [
            sys.executable,
            'scripts/create_p2_b1125_checkout.py',
            '--output',
            str(output),
            '--dataset',
            'smoke',
            '--lexical-vocabulary',
            str(vocabulary),
            '--query-atoms',
            str(queries),
            '--semantic-runtime-dir',
            str(runtime),
            '--query-calibration-runtime',
            str(calibration),
            '--semantic-dims',
            '4',
            '--model-id',
            'ii42_p2_native_test',
        ],
        check=True,
    )

    manifest = read_json(output / 'manifest.json')
    assert manifest['schema_version'] == 1
    assert manifest['api_version'] == 'evoke_model_v1'
    assert manifest['model_format'] == 'onnx'
    assert manifest['runtime'] == 'onnxruntime'
    assert manifest['runtime_abi'] == 'evoke_p2_unified_text_atoms_v2'
    assert 'runtime_parameters' not in manifest
    assert manifest['runtime_io'] == {
        'attention_mask': 'attention_mask',
        'input_ids': 'input_ids',
        'semantic_ids': 'semantic_ids',
        'semantic_weights': 'semantic_weights',
    }
    assert manifest['artifacts']['query_encoder']['path'] == (
        'encoder/semantic_query_compiler.onnx'
    )
    assert manifest['artifacts']['document_encoder']['path'] == (
        'encoder/semantic_document_compiler.onnx'
    )
    assert manifest['runtime_output']['query_semantic_max_atoms'] == 50
    assert manifest['runtime_output']['document_semantic_max_atoms'] == 192
    assert manifest['runtime_output']['lexical_dims'] == 2
    assert manifest['runtime_output']['semantic_dims'] == 4
    assert (
        manifest['runtime_output']['document_semantic_budget_ratio']
        == 1.125
    )
    assert manifest['artifacts']['tokenizer_vocabulary']['path'] == (
        'encoder/vocab.json'
    )
    assert manifest['artifacts']['tokenizer_merges']['path'] == (
        'encoder/merges.txt'
    )
    assert manifest['artifacts']['query_calibration_runtime']['path'] == (
        'calibration/query_calibration_rms.f32'
    )
    scoring = read_json(output / 'calibration/scoring_profile.json')
    assert scoring['query_compiler']['corpus_statistics'] == {
        'active_dimensions': 5,
        'document_count': 2,
        'method': 'column_rms_over_all_documents',
        'runtime_artifact': 'calibration/query_calibration_rms.f32',
    }


def test_create_p2_b1125_runtime_requires_calibration(tmp_path: Path) -> None:
    vocabulary = tmp_path / 'vocabulary.json'
    queries = tmp_path / 'queries.jsonl'
    runtime = tmp_path / 'runtime'
    output = tmp_path / 'p2_native_checkout'
    vocabulary.write_text(json.dumps(['alpha']) + '\n')
    queries.write_text(json.dumps({
        'atom_ids': [0],
        'atom_impacts': [1.0],
        'id': 'query-a',
    }) + '\n')
    runtime.mkdir()
    for name in (
        'semantic_document_compiler.onnx',
        'semantic_query_compiler.onnx',
        'semantic_runtime.json',
        'merges.txt',
        'tokenizer.json',
        'tokenizer_config.json',
        'vocab.json',
    ):
        payload = {'abi': 'evoke_p2_semantic_onnx_v2'} if (
            name == 'semantic_runtime.json'
        ) else {}
        (runtime / name).write_text(json.dumps(payload) + '\n')

    result = subprocess.run(
        [
            sys.executable,
            'scripts/create_p2_b1125_checkout.py',
            '--output',
            str(output),
            '--dataset',
            'smoke',
            '--lexical-vocabulary',
            str(vocabulary),
            '--query-atoms',
            str(queries),
            '--semantic-runtime-dir',
            str(runtime),
            '--semantic-dims',
            '1',
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert '--query-calibration-runtime is required' in result.stderr


def test_create_p2_b1125_rejects_batch_size_manifest_knob(
    tmp_path: Path,
) -> None:
    vocabulary = tmp_path / 'vocabulary.json'
    queries = tmp_path / 'queries.jsonl'
    output = tmp_path / 'p2_checkout'
    vocabulary.write_text(json.dumps(['alpha']) + '\n', encoding='utf-8')
    queries.write_text(json.dumps({
        'atom_ids': [0],
        'atom_impacts': [1.0],
        'id': 'query-a',
    }) + '\n', encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            'scripts/create_p2_b1125_checkout.py',
            '--output',
            str(output),
            '--dataset',
            'smoke',
            '--lexical-vocabulary',
            str(vocabulary),
            '--query-atoms',
            str(queries),
            '--semantic-dims',
            '1',
            '--max-batch-size',
            '33',
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert 'unrecognized arguments: --max-batch-size 33' in result.stderr
