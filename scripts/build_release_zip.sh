#!/usr/bin/env bash

set -euo pipefail

version=''
pg_config=''
model_checkout=''
onnxruntime_prefix=''
onnxruntime_prefix_explicit=false
allow_dirty=false

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            version="$2"
            shift 2
            ;;
        --pg-config)
            pg_config="$2"
            shift 2
            ;;
        --model-checkout)
            model_checkout="$2"
            shift 2
            ;;
        --onnxruntime-prefix)
            onnxruntime_prefix="$2"
            onnxruntime_prefix_explicit=true
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

if [ -z "$pg_config" ]; then
    pg_config="$(command -v pg_config || true)"
fi

if [ -z "$pg_config" ] || [ ! -x "$pg_config" ]; then
    echo 'pg_config not found' >&2
    exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
    echo 'pkg-config not found' >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo 'zip not found' >&2
    exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
    checksum_tool='sha256sum'
elif command -v shasum >/dev/null 2>&1; then
    checksum_tool='shasum'
else
    echo 'no sha256 checksum tool available' >&2
    exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "$model_checkout" ]; then
    model_checkout="$(
        printf '%s' \
            "${EVOKE_MILESTONE_MODEL_CHECKOUT:-${II42_MILESTONE_MODEL_CHECKOUT:-}}"
    )"
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
    git_commit="${II42_GIT_COMMIT:-}"
    git_tree_state="${II42_GIT_TREE_STATE:-source-archive}"
    if [ -z "$git_commit" ]; then
        echo 'II42_GIT_COMMIT is required outside a Git worktree' >&2
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
    "${repo_root}/ii42.control")"
if [ "$version" != "$control_version" ]; then
    echo "release version ${version} does not match ii42.control " \
        "version ${control_version}" >&2
    exit 1
fi
expected_ort_version="$(tr -d '[:space:]' \
    <"${repo_root}/packaging/onnxruntime.version")"
if [ -z "$expected_ort_version" ]; then
    echo 'packaging/onnxruntime.version is empty' >&2
    exit 1
fi
arch="$(uname -m)"
os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"
if [ -z "$onnxruntime_prefix" ]; then
    onnxruntime_prefix="${II42_ONNXRUNTIME_PREFIX:-}"
    if [ -n "$onnxruntime_prefix" ]; then
        onnxruntime_prefix_explicit=true
    fi
fi
if [ -z "$onnxruntime_prefix" ]; then
    onnxruntime_prefix="${repo_root}/.artifacts/onnxruntime"
    onnxruntime_prefix+="-${expected_ort_version}-${os_name}-${arch}"
fi
onnxruntime_pkg_config_path="${onnxruntime_prefix}/lib/pkgconfig"
onnxruntime_pc="${onnxruntime_pkg_config_path}/libonnxruntime.pc"
if [ ! -f "$onnxruntime_pc" ]; then
    if [ "$onnxruntime_prefix_explicit" = true ]; then
        echo "ONNX Runtime SDK is missing from ${onnxruntime_prefix}" >&2
        exit 1
    fi
    "${repo_root}/scripts/install_onnxruntime_c.sh" \
        --version "$expected_ort_version" \
        --prefix "$onnxruntime_prefix"
fi
onnxruntime_prefix="$(cd "$onnxruntime_prefix" && pwd)"
onnxruntime_pkg_config_path="${onnxruntime_prefix}/lib/pkgconfig"
existing_pkg_config_path="${PKG_CONFIG_PATH:-}"
export PKG_CONFIG_PATH="$onnxruntime_pkg_config_path"
if [ -n "$existing_pkg_config_path" ]; then
    export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${existing_pkg_config_path}"
fi
if ! pkg-config --exists libonnxruntime; then
    echo "ONNX Runtime SDK is unusable: ${onnxruntime_prefix}" >&2
    exit 1
fi
actual_ort_version="$(pkg-config --modversion libonnxruntime)"
if [ "$actual_ort_version" != "$expected_ort_version" ]; then
    printf 'ONNX Runtime SDK at %s reports %s; expected %s\n' \
        "$onnxruntime_prefix" \
        "$actual_ort_version" \
        "$expected_ort_version" >&2
    exit 1
fi
pg_version="$("$pg_config" --version | awk '{print $2}')"
pg_major="${pg_version%%.*}"
pg_version_text="$("$pg_config" --version)"
pg_sharedir="$("$pg_config" --sharedir)"
package_name="evoke-v${version}-${os_name}-${arch}-pg${pg_major}"
dist_dir="${repo_root}/dist"
stage_dir="${dist_dir}/${package_name}"
zip_path="${dist_dir}/${package_name}.zip"
checksum_path="${zip_path}.sha256"

rm -rf "$stage_dir" "$zip_path" "$checksum_path"
mkdir -p "$stage_dir"

make -C "$repo_root" clean PG_CONFIG="$pg_config"
make -C "$repo_root" II42_ENABLE_ONNXRUNTIME=1 PG_CONFIG="$pg_config"
make -C "$repo_root" install II42_ENABLE_ONNXRUNTIME=1 \
    DESTDIR="$stage_dir" PG_CONFIG="$pg_config"

runtime_libdir="$(pkg-config --variable=libdir libonnxruntime)"
pkglibdir="$("$pg_config" --pkglibdir)"
runtime_stage_dir="${stage_dir}${pkglibdir}"
case "$os_name" in
    darwin)
        runtime_pattern='libonnxruntime*.dylib*'
        ;;
    linux)
        runtime_pattern='libonnxruntime*.so*'
        ;;
    *)
        echo "unsupported release runtime platform: ${os_name}" >&2
        exit 1
        ;;
esac
shopt -s nullglob
runtime_candidates=("${runtime_libdir}/"${runtime_pattern})
runtime_libraries=()
for runtime_library in "${runtime_candidates[@]}"; do
    case "$(basename "$runtime_library")" in
        libonnxruntime.dylib|libonnxruntime.so)
            # The unversioned file is a build-time linker input. The loaded
            # extension requests the SONAME and does not need this duplicate.
            continue
            ;;
    esac
    if [ -f "$runtime_library" ] || [ -L "$runtime_library" ]; then
        runtime_libraries+=("$runtime_library")
    fi
done
if [ "${#runtime_libraries[@]}" -eq 0 ]; then
    echo 'ONNX Runtime shared library was not found' >&2
    exit 1
fi
project_license="${repo_root}/LICENSE"
if [ ! -f "$project_license" ]; then
    echo "project license was not found: ${project_license}" >&2
    exit 1
fi

mkdir -p "$runtime_stage_dir" "${stage_dir}/LICENSES"
cp -P "${runtime_libraries[@]}" "$runtime_stage_dir/"
if [ "$os_name" = 'darwin' ]; then
    extension_library="${runtime_stage_dir}/ii42.dylib"
    runtime_dependency="$(
        otool -L "$extension_library" |
            awk '/libonnxruntime\.[0-9]+\.dylib/ { print $1; exit }'
    )"
    if [ -z "$runtime_dependency" ]; then
        echo 'staged extension has no versioned ONNX Runtime dependency' >&2
        exit 1
    fi
    runtime_dependency_name="$(basename "$runtime_dependency")"
    bundled_runtime="${runtime_stage_dir}/${runtime_dependency_name}"
    if [ ! -e "$bundled_runtime" ]; then
        echo "staged ONNX Runtime is missing: ${bundled_runtime}" >&2
        exit 1
    fi
    packaged_dependency="@loader_path/${runtime_dependency_name}"
    if [ "$runtime_dependency" != "$packaged_dependency" ]; then
        install_name_tool -change \
            "$runtime_dependency" \
            "$packaged_dependency" \
            "$extension_library"
    fi
    runtime_dependency="$(
        otool -L "$extension_library" |
            awk '/libonnxruntime\.[0-9]+\.dylib/ { print $1; exit }'
    )"
    if [ "$runtime_dependency" != "$packaged_dependency" ]; then
        echo 'staged extension does not load its bundled ONNX Runtime' >&2
        exit 1
    fi
fi
model_stage_dir="${stage_dir}${pg_sharedir}/ii42/models/default"
mkdir -p "$model_stage_dir"
cp -R "${model_checkout}/." "$model_stage_dir/"
chmod -R a+rX "$model_stage_dir"
runtime_prefix="$(pkg-config --variable=prefix libonnxruntime)"
runtime_license=''
for candidate in \
    "${runtime_prefix}/share/licenses/onnxruntime/LICENSE" \
    "${runtime_prefix}/LICENSE" \
    "${runtime_prefix}/LICENSE.txt"
do
    if [ -f "$candidate" ]; then
        runtime_license="$candidate"
        break
    fi
done
if [ -z "$runtime_license" ]; then
    echo "ONNX Runtime license was not found under ${runtime_prefix}" >&2
    exit 1
fi
cp "$runtime_license" "${stage_dir}/LICENSES/ONNXRUNTIME-LICENSE"
cp "$project_license" "${stage_dir}/LICENSES/EVOKE-LICENSE"
cp "$project_license" "${stage_dir}/LICENSES/MODEL-LICENSE"
cp "${repo_root}/packaging/MILESTONE-MODEL-NOTICE" \
    "${stage_dir}/LICENSES/MILESTONE-MODEL-NOTICE"

cat >"${stage_dir}/BUILD-INFO.txt" <<EOF
Package: ${package_name}
Version: ${version}
Operating system: ${os_name}
Architecture: ${arch}
PostgreSQL major: ${pg_major}
PostgreSQL version: ${pg_version_text}
PostgreSQL pkglibdir: ${pkglibdir}
PostgreSQL sharedir: ${pg_sharedir}
pg_config: ${pg_config}
Git commit: ${git_commit}
Git tree: ${git_tree_state}
Built at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
ONNX Runtime: ${actual_ort_version}
ONNX Runtime SDK prefix: ${onnxruntime_prefix}
ONNX Runtime linkage: bundled in PostgreSQL pkglibdir
Milestone model: ${model_id}
Milestone model manifest SHA-256: ${model_manifest_sha256}
Milestone model location: ${pg_sharedir}/ii42/models/default
EOF

cp "${repo_root}/README.md" "${stage_dir}/README.md"

(
    cd "$dist_dir"
    zip -qry "$(basename "$zip_path")" "$(basename "$stage_dir")"
)
zip -T "$zip_path" >/dev/null

zip_basename="$(basename "$zip_path")"
checksum_basename="$(basename "$checksum_path")"
(
    cd "$dist_dir"
    if [ "$checksum_tool" = 'sha256sum' ]; then
        sha256sum "$zip_basename" >"$checksum_basename"
    else
        shasum -a 256 "$zip_basename" >"$checksum_basename"
    fi
)

echo "$zip_path"
