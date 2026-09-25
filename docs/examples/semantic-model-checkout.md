# Semantic Model Checkout

A checkout is trusted, server-local, digest-validated executable configuration
for `text -> sparse atoms`.

Official packages ship one frozen milestone checkout. Custom checkouts use the
same contract and can be selected per index without replacing package files.

## Download The Default Model

The current checkout is published as
[Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
under the Intelligent Internet organization on Hugging Face. It is the same
P2.2 ABI-v2 content pinned by
[`packaging/milestone-model.json`](../../packaging/milestone-model.json), not a
new model revision. The model weights are intentionally excluded from Git.
Public downloads do not require an account, token, or model-access approval.

From the Evoke repository root, download the immutable archive and validate it:

```bash
model_repo='Intelligent-Internet/Evoke-Model-Beta-1'
model_revision='ee17e24a5f8eede1c9ac098b1f8bf637c32d9f07'
model_archive='evoke-p2.2-nfcorpus-v2.zip'
model_sha256='ab5cd4b2a1fec10c70113d2997c6ce070caa34cf7a74c114b7ab48d81643cd98'
model_url="https://huggingface.co/${model_repo}/resolve/${model_revision}/${model_archive}"

python3 scripts/fetch_milestone_model.py \
    --url "$model_url" \
    --archive-sha256 "$model_sha256" \
    --output .artifacts/evoke-milestone-model
python3 scripts/validate_milestone_model_checkout.py \
    --checkout .artifacts/evoke-milestone-model
```

The archive is 187,596,050 bytes (about 179 MiB); the extracted checkout is
400,908,113 bytes (about 382 MiB). Validation checks the ZIP SHA-256, manifest
SHA-256, all 13 artifact digests, model/runtime contract, and exact file
inventory before publishing the local directory. The fetcher refuses an
existing output by default: validate it instead, or use a new output path.
Do not overwrite a model checkout used by a running PostgreSQL instance.

This path is the default input to `scripts/build_release_zip.sh` and
`scripts/build_release_docker_image.sh`. An alternate validated location can
be selected using `--model-checkout` or `EVOKE_MILESTONE_MODEL_CHECKOUT`.
The legacy `II42_MILESTONE_MODEL_CHECKOUT` variable is still accepted during
the transition.
See [Contributing](../../CONTRIBUTING.md#release-automation) for release builds.
For GitHub Actions, use the value of `model_url` above for
`EVOKE_MILESTONE_MODEL_URL` and `model_sha256` for
`EVOKE_MILESTONE_MODEL_ARCHIVE_SHA256`; neither is a secret. The old `II42_*`
repository variables remain fallback inputs until all jobs are migrated.

The Hub also exposes individual files under `checkout/` and an `hf download`
example in its model card. Do not pass the entire Hub repository, or a local
download directory containing `.cache/huggingface`, as `--model-checkout`.
The card, license, and Hub/Git metadata must remain outside the exact checkout
inventory. Use the archive fetcher above to obtain a clean directory.

## Checkout Contract

The default checkout contains:

```text
evoke-milestone-model/
  manifest.json
  encoder/
    semantic_document_compiler.onnx
    semantic_query_compiler.onnx
    semantic_runtime.json
    tokenizer.json
    tokenizer_config.json
    merges.txt
    vocab.json
  lexical/vocabulary.json
  sae/atom_space.json
  calibration/
    query_calibration_rms.f32
    query_calibration_stats.npz
    scoring_profile.json
  provenance/p2_contract.json
```

Custom checkouts may use different artifact filenames; the manifest declares:

- schema, runtime ABI, model identity, and tensor contract;
- tokenizer and input/output limits;
- atom namespace and scoring profile;
- relative artifact paths and SHA-256 digests;
- index compatibility assertions.

Execution provider selection, provider-specific options, and batching limits are
runtime deployment policy. They are not part of model identity and are not
declared by the checkout.

Artifact paths must remain inside the checkout. Keep the directory writable
only by the PostgreSQL deployment administrator. Database application roles
must not choose or modify server-local paths.

## Source Installation

`make` compiles the extension without reading model weights, and `make install`
does not install this checkout automatically. Exact BM25 indexes do not need
model inference. A source-installed Sparse Semantic Retrieval (SSR) deployment
must install the validated checkout at the compiled default shared-data path,
normally `$(pg_config --sharedir)/ii42/models/default`, or configure an
administrator-owned absolute path before creating SSR indexes:

```conf
shared_preload_libraries = 'ii42'
ii42.shared_runtime_size = '64MB'
ii42.sae_model_path = '/absolute/path/to/evoke-milestone-model'
```

The directory must be readable by PostgreSQL and not writable by application
roles. Configure the same model content on primary and standby. Follow the
[runtime setup](semantic-runtime.md) and
[deployment boundary](../upgrading.md#deployment-boundary); source builds still
require the pinned ONNX Runtime SDK, independently of downloading the model.

## Bind To An Index

Use the deployment default:

```sql
CREATE INDEX docs_search_idx
ON docs USING ii42 (body)
WITH (sae = true);
```

Or set a per-index override:

```sql
CREATE INDEX docs_search_idx
ON docs USING ii42 (body)
WITH (
    sae = true,
    model_path = '/opt/ii42/models/search_model'
);
```

`model`, `atom_space`, and `scoring_profile` reloptions are optional assertions
against the manifest. They do not register independent components.

Resolution order is per-index `model_path`, server-wide
`ii42.sae_model_path`, then the package checkout. Because the path is
server-local, install an identical checkout on every primary and standby.

## Validate

```sql
SELECT ii42_index_options('docs_search_idx'::regclass);
SELECT ii42_index_status('docs_search_idx'::regclass);
SELECT ii42_index_audit('docs_search_idx'::regclass);
SELECT * FROM ii42_query('docs_search_idx'::regclass, 'query text', 20);
```

Status performs bounded readiness and runtime-identity checks after enforcing
the caller's table access boundary. It deliberately reports
`model_artifacts_valid = null` with
`model_artifact_validation = 'explicit_audit_required'`. Run
`ii42_index_audit(...)` explicitly for a complete generation walk and SHA-256
validation of every server-local model artifact; do not put that heavy audit in
readiness polling or request paths.

Changing any contract-bound artifact requires a new digest-valid checkout and
`REINDEX`. Atomic replacement at the same path cannot reuse an incompatible
worker session because session identity includes the checkout signature.

Remove the index with PostgreSQL `DROP INDEX`; the checkout remains deployment
configuration and is not owned by the index relation.
