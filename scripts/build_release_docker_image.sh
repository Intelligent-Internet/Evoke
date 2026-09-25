#!/usr/bin/env bash

set -euo pipefail

version=''
image_tag=''
model_checkout=''
allow_dirty=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            version="$2"
            shift 2
            ;;
        --image-tag)
            image_tag="$2"
            shift 2
            ;;
        --model-checkout)
            model_checkout="$2"
            shift 2
            ;;
        --allow-dirty)
            allow_dirty=true
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$version" ]; then
    echo '--version is required' >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "$model_checkout" ]; then
    model_checkout="${EVOKE_MILESTONE_MODEL_CHECKOUT:-}"
fi
if [ -z "$model_checkout" ]; then
    model_checkout="${repo_root}/.artifacts/evoke-milestone-model"
fi
if [ ! -d "$model_checkout" ]; then
    echo "milestone model checkout not found: ${model_checkout}" >&2
    echo 'pass --model-checkout or set EVOKE_MILESTONE_MODEL_CHECKOUT' >&2
    exit 1
fi
model_checkout="$(cd "$model_checkout" && pwd)"
python3 "${repo_root}/scripts/validate_milestone_model_checkout.py" \
    --checkout "$model_checkout" \
    --quiet
model_id="$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' \
    "${model_checkout}/manifest.json")"
model_manifest_sha256="$(python3 -c \
    'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
    "${model_checkout}/manifest.json")"
if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_commit="$(git -C "$repo_root" rev-parse HEAD)"
    git_tree_state='clean'
    if [ -n "$(git -C "$repo_root" status \
        --porcelain --untracked-files=normal)" ]; then
        git_tree_state='dirty'
    fi
else
    git_commit="${EVOKE_GIT_COMMIT:-}"
    git_tree_state="${EVOKE_GIT_TREE_STATE:-source-archive}"
    if [ -z "$git_commit" ]; then
        echo 'EVOKE_GIT_COMMIT is required outside a Git worktree' >&2
        exit 1
    fi
fi
if [ "$git_tree_state" = 'dirty' ] && [ "$allow_dirty" != true ]; then
    echo 'refusing to build a release from a dirty Git worktree' >&2
    echo 'commit or stash changes, or pass --allow-dirty explicitly' >&2
    exit 1
fi
control_version="$(sed -n \
    "s/^default_version = '\(.*\)'$/\1/p" \
    "${repo_root}/evoke.control")"
if [ "$version" != "$control_version" ]; then
    echo "Docker version ${version} does not match evoke.control " \
        "version ${control_version}" >&2
    exit 1
fi
onnxruntime_version="$(tr -d '[:space:]' \
    <"${repo_root}/packaging/onnxruntime.version")"
arch="$(uname -m)"
os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [ -z "$image_tag" ]; then
    image_tag="evoke:pg18-v${version}"
fi

package_name="evoke-v${version}-docker-pg18-${os_name}-${arch}"
dist_dir="${repo_root}/dist"
archive_path="${dist_dir}/${package_name}.tar.gz"
checksum_path="${archive_path}.sha256"

rm -rf "$archive_path" "$checksum_path"
mkdir -p "$dist_dir"

docker build \
    --pull \
    --build-context "evoke_milestone_model=${model_checkout}" \
    --build-arg "EVOKE_VERSION=${version}" \
    --build-arg "ONNXRUNTIME_VERSION=${onnxruntime_version}" \
    --build-arg "EVOKE_GIT_COMMIT=${git_commit}" \
    --build-arg "EVOKE_GIT_TREE_STATE=${git_tree_state}" \
    --build-arg "EVOKE_MODEL_ID=${model_id}" \
    --build-arg "EVOKE_MODEL_MANIFEST_SHA256=${model_manifest_sha256}" \
    --file "${repo_root}/packaging/docker/postgres18/Dockerfile" \
    --tag "${image_tag}" \
    "${repo_root}"

docker save "${image_tag}" | gzip -c >"${archive_path}"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$archive_path" >"$checksum_path"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$archive_path" >"$checksum_path"
else
    echo 'no sha256 checksum tool available' >&2
    exit 1
fi

echo "$archive_path"
