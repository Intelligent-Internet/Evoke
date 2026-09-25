# Contributing

This file is the canonical contributor guide for build, testing, release, and
submission requirements.

Before contributing, review the [Code of Conduct](CODE_OF_CONDUCT.md),
[support policy](SUPPORT.md), and [security policy](SECURITY.md). Suspected
vulnerabilities must use private security reporting rather than a public
issue.

## Submitting Changes

- Keep each pull request focused on one coherent change and explain its
  user-visible behavior, compatibility boundary, and operational impact.
- Add or update tests for behavior changes. Include the exact validation
  commands and results in the pull request description.
- Update current documentation and `CHANGELOG.md` when a change affects the
  public API, defaults, storage contract, migration, packaging, or deployment.
- Do not commit local build trees, generated release packages, model
  checkouts, credentials, or machine-specific configuration.
- Use [SUPPORT.md](SUPPORT.md) for usage questions and reproducible bug-report
  requirements. Report security issues through [SECURITY.md](SECURITY.md).

## Build From Source

Install the development files for the target PostgreSQL major, a C compiler,
`make`, `pkg-config`, ICU, OpenSSL, and Jansson. The runtime-service build also
needs a C++17 compiler. Standalone C tests use CMake 3.20 or newer; Python
validation uses the dependencies of the selected test/evaluation harness.
Select the pinned ONNX Runtime SDK below **before** running the default build.

With those dependencies configured, build the extension:

```bash
make
make install
make installcheck
```

If `pg_config` is not on `PATH`, pass it explicitly:

```bash
make PG_CONFIG=/path/to/pg_config
make install PG_CONFIG=/path/to/pg_config
make installcheck PG_CONFIG=/path/to/pg_config
```

The default build is fail-closed: `pkg-config libonnxruntime` must resolve the
exact version pinned by `packaging/onnxruntime.version`, currently 1.29.0. A
system SDK with another version is rejected rather than silently linked. Use
the checksum-locked installer when the pinned SDK is not already available:

```bash
ort_version="$(tr -d '[:space:]' < packaging/onnxruntime.version)"
ort_prefix="$PWD/.artifacts/onnxruntime-${ort_version}"
./scripts/install_onnxruntime_c.sh \
    --prefix "$ort_prefix"
export PKG_CONFIG_PATH="$ort_prefix/lib/pkgconfig"
```

`scripts/build_release_zip.sh` is stricter: by default it installs and selects
the checksum-locked pinned SDK under `.artifacts`, so the host's system
`pkg-config` selection cannot determine a release artifact. For an existing or
offline SDK, pass `--onnxruntime-prefix`; the script validates that prefix
against the repository pin before compiling.

`EVOKE_ENABLE_ONNXRUNTIME=0` is reserved for explicit lexical-only diagnostics
and CI coverage; such a binary cannot serve SSR query encoding and must not be
deployed to a query-serving PostgreSQL instance.

For Homebrew PostgreSQL 18 on macOS, that often looks like:

```bash
make PG_CONFIG=/opt/homebrew/opt/postgresql@18/bin/pg_config
make install PG_CONFIG=/opt/homebrew/opt/postgresql@18/bin/pg_config
make installcheck PG_CONFIG=/opt/homebrew/opt/postgresql@18/bin/pg_config
```

After installation, create the extension in a target database:

```sql
CREATE EXTENSION evoke;
```

If you need a non-`public` extension schema, choose it when creating the
extension:

```sql
CREATE SCHEMA ext;
CREATE EXTENSION evoke WITH SCHEMA ext;
```

The extension is not relocatable after creation because SQL helper functions
capture the extension schema for safe wrapper resolution.

## Testing

The main local validation entry points are:

```bash
cmake -S . -B build_tmp -DCMAKE_BUILD_TYPE=Release
cmake --build build_tmp
ctest --test-dir build_tmp --output-on-failure
make installcheck
```

Current validation coverage includes:

- standalone unit tests
- PostgreSQL `pg_regress` integration tests
- maintenance-state and policy regression coverage
- restart smoke
- crash-recovery smoke
- physical-replication smoke
- package-bound `psql_bm25s` source-table migration smoke
- concurrent maintenance stability smoke and broader stress runs

For the full test matrix, smoke scripts, schema-placement checks, hybrid-search
coverage, and benchmark validation gates, see
[Testing and Validation](docs/testing-and-validation.md).

## Documentation Changes

- Keep [the documentation map](docs/README.md) as the entrypoint. Every current
  guide should be reachable from it, directly or through a focused guide.
- Keep BM25-first quickstarts short. Put complete signatures, defaults,
  lifecycle boundaries, and deployment procedures in their canonical guides.
- Keep `technical-report-evoke-model` and `technical-report-evoke-system` as the
  bilingual Beta 1 reports; `technical-report-psql_bm25s` is lexical history.
  Change both language editions together, including formulas and evidence.
- Separate current contracts, dated experimental evidence, and future plans.
  Preserve historical numbers and provenance; do not relabel an old run as a
  new package qualification. Closed engineering records belong in the archive.
- When moving a document, update incoming links and its own relative links,
  including heading fragments, diagrams, tests, and inventory checks.
- Run `python3 -m pytest -q` and
  `python3 scripts/test_product_convergence_inventory.py`. Documentation tests
  check local navigation/anchors, SQL example bindings, model/release identity,
  bilingual report formulas, and selected source-backed architecture constants.

These static checks do not execute every SQL snippet or establish performance.
For query/API changes, also run the relevant isolated PostgreSQL regressions.

## Release Automation

Release ZIPs and Docker images always carry the frozen milestone checkout.
Download [Evoke Model (Beta 1)](https://huggingface.co/Intelligent-Internet/Evoke-Model-Beta-1)
using the [pinned public archive instructions](docs/examples/semantic-model-checkout.md#download-the-default-model).
Public downloads require no Hugging Face login or token. The model is not
needed for C compilation, but it is required for these complete release
artifacts and for running SSR indexes after a source-only install.

Provide it with `--model-checkout`, `EVOKE_MILESTONE_MODEL_CHECKOUT`, or the
ignored local path `.artifacts/evoke-milestone-model`. The builder validates
`packaging/milestone-model.json`, every artifact digest, runtime ABI,
and exact file inventory before staging it under PostgreSQL's shared-data
directory. GitHub release workflows fetch the same archive from the protected
`EVOKE_MILESTONE_MODEL_URL` repository variable and optionally verify
`EVOKE_MILESTONE_MODEL_ARCHIVE_SHA256` before the content-level validation.

The 382 MiB checkout is deliberately not stored in ordinary Git history.
Changing the milestone requires a reviewed lock update, full native lifecycle
qualification, and a new immutable release artifact.

Validate the qualified checkout and build that deterministic release asset
with:

```bash
python3 scripts/validate_milestone_model_checkout.py \
    --checkout /path/to/milestone-checkout
python3 scripts/build_milestone_model_archive.py \
    --checkout /path/to/milestone-checkout \
    --output dist/evoke-p2.2-nfcorpus-v2.zip
```

For the current milestone, set `EVOKE_MILESTONE_MODEL_URL` to the commit-pinned
Hugging Face ZIP URL and `EVOKE_MILESTONE_MODEL_ARCHIVE_SHA256` to the archive
digest in the [download instructions](docs/examples/semantic-model-checkout.md#download-the-default-model).
These are public repository variables, not credentials. For a future
milestone, publish and validate its immutable ZIP before changing these
variables. Release jobs fail closed if the archive digest or its content-level
lock does not match. A fork must set its own release-workflow variables; the
source-build download command does not require GitHub configuration.

Current release line:

- extension version: read `default_version` from `evoke.control`
- supported release targets:
  - PostgreSQL `17`
  - PostgreSQL `18`
  - Docker image based on official `postgres:18`
- validated artifact formats:
  - `.zip`
  - a PostgreSQL 18 Docker archive and the container image
    `ghcr.io/intelligent-internet/evoke`

Automation model in this development repository:

- pushes and pull requests build and test on GitHub Actions
- a manual `CI` dispatch runs the same full matrix for a candidate commit
- release-branch pushes build PostgreSQL 17/18 ZIP packages and a PostgreSQL
  18 Docker image as a dry run
- tag pushes or manual dispatches repeat the same release dry run
- a successful release-branch dry run can sync the validated source tree to
  the public [Intelligent-Internet/Evoke](https://github.com/Intelligent-Internet/Evoke) repository
- this development repository does not publish release artifacts; the public
  repository builds and publishes them using its separately maintained workflows

Current workflows:

- `CI`
  - runs on push to `main`, push to `release`, pull requests, and manual
    dispatch
  - builds and tests PostgreSQL `17` and `18`
  - builds and smoke-tests the PostgreSQL `18` Docker image without publishing
    it
- `Release Dry Run`
  - runs on tag push matching `v*` or manual dispatch
  - rebuilds and validates packages without publishing them
- `Release Branch Dry Run`
  - runs on pushes to `release` or manual dispatch
  - validates the same package and Docker surfaces before public sync
- `Sync Public Repository`
  - runs only after a successful release-branch dry run and matching CI
  - targets `Intelligent-Internet/Evoke` and requires
    `PUBLIC_REPO_SYNC_TOKEN` with write access to that repository
  - remains disabled unless the repository variable
    `PUBLIC_REPO_SYNC_ENABLED` is exactly `true`
  - copies the source tree into public `main` as a normal new commit, keeping
    public history and its existing `.github/workflows/`; development history
    is not imported and the push is not forced

The current release practice is:

1. Merge validated changes to `main`.
2. Review the complete current install catalog, then bump its version with
   `scripts/bump_extension_version.py`. The package contains only that current
   install SQL; it does not carry transitions between Evoke beta versions.
3. Fast-forward or merge `main` into `release` and require CI plus the
   release-branch dry run to pass.
4. Validate public-repository access and the sync enable variable before source
   sync. Sync waits for successful CI and dry-run results for the same release
   commit, then appends a public snapshot commit without rewriting history.
5. Public `CI` validates the snapshot. `Prepare Release` tags the exact tested
   version from `evoke.control` and dispatches public `Release`; it does not
   automatically bump the catalog or rewrite existing tags. A changed release
   requires a newly reviewed version, not reuse of an already published tag.
   Documentation, tests, and workflow-only follow-ups can sync without a version
   bump: if the existing tag is an ancestor and its release is already published,
   `Prepare Release` leaves that tag and its assets unchanged. The exemption
   covers Markdown files, `docs/`, `tests/`, and `.github/`; changes to extension
   code, SQL, build scripts, model locks, or license/notice files still require
   a new version. Updated online documentation does not rewrite documentation
   inside previously published archives.
6. Public `Release` repeats CI for the tag, builds PostgreSQL 17/18 Linux ZIPs
   with the pinned ORT SDK and model, smoke-tests the PostgreSQL 18 Docker image,
   and publishes ZIPs, the Docker archive, and checksums as GitHub Release assets.
   The image is first pushed to `ghcr.io/intelligent-internet/evoke` with the
   versioned tag `pg18-v<version>`. Only after all assets are published does
   `Promote Release` verify the complete asset inventory, source commit, and
   image digest, then point `pg18` and `latest` at the same manifest and set
   GitHub Latest. The Beta designation describes product maturity; this default
   channel does not use GitHub's pre-release flag, which cannot be Latest.
   Existing releases can use the manual `Promote Release` workflow with an
   explicit release tag and reviewed image digest. Promotion changes only
   aliases and release metadata: it does not build, upload, replace assets, or
   move the source tag. It rejects promotion of an older published release.

The public workflow files are intentionally excluded from source sync. Changes
to those files must be reviewed and committed in the public repository as well;
copying the development dry-run workflows over them would disable publishing.

Detailed benchmark evidence and release-readiness checks should link to
[Performance and Benchmarks](docs/performance/README.md) and
[Testing and Validation](docs/testing-and-validation.md) rather than
duplicating their contents here.

Storage and lifecycle changes must preserve the current
[Convergent Segmented Index](docs/convergent-segmented-index.md) contract.
Historical implementation decisions and acceptance evidence are preserved in
the [development record](docs/archive/engineering/convergent-segmented-index-development-record.md);
current release qualification belongs in
[Testing and Validation](docs/testing-and-validation.md).

Release and deployment changes must preserve the
[BM25 migration contract](docs/upgrading.md). The mutable lifecycle suite must
continue to prove that a current `sae = false` relation fails closed during
the reloption transition and becomes a valid `sae = true` page-native index
only after `REINDEX` publishes the matching checked root.
