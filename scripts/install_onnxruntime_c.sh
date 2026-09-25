#!/usr/bin/env bash

set -euo pipefail

version=''
prefix=''
archive_override=''
asset_override=''
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            if [ "$#" -lt 2 ]; then
                echo '--version requires a value' >&2
                exit 1
            fi
            version="$2"
            shift 2
            ;;
        --archive)
            if [ "$#" -lt 2 ]; then
                echo '--archive requires a value' >&2
                exit 1
            fi
            archive_override="$2"
            shift 2
            ;;
        --asset)
            if [ "$#" -lt 2 ]; then
                echo '--asset requires a value' >&2
                exit 1
            fi
            asset_override="$2"
            shift 2
            ;;
        --prefix)
            if [ "$#" -lt 2 ]; then
                echo '--prefix requires a value' >&2
                exit 1
            fi
            prefix="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$prefix" ]; then
    echo '--prefix is required' >&2
    exit 1
fi

if [ -z "$version" ]; then
    version_file="${repo_root}/packaging/onnxruntime.version"
    if [ ! -s "$version_file" ]; then
        echo '--version is required outside the evoke source tree' >&2
        exit 1
    fi
    version="$(tr -d '[:space:]' <"$version_file")"
fi

case "$(uname -s):$(uname -m)" in
    Linux:x86_64)
        asset="onnxruntime-linux-x64-${version}"
        library_pattern='libonnxruntime*.so*'
        ;;
    Linux:aarch64|Linux:arm64)
        asset="onnxruntime-linux-aarch64-${version}"
        library_pattern='libonnxruntime*.so*'
        ;;
    Darwin:arm64)
        asset="onnxruntime-osx-arm64-${version}"
        library_pattern='libonnxruntime*.dylib*'
        ;;
    *)
        echo "unsupported ONNX Runtime platform: $(uname -s) $(uname -m)" >&2
        exit 1
        ;;
esac

if [ -n "$asset_override" ]; then
    asset="$asset_override"
fi

checksum_file="${EVOKE_ONNXRUNTIME_CHECKSUMS:-}"
if [ -z "$checksum_file" ]; then
    checksum_file="${repo_root}/packaging/onnxruntime.sha256"
fi
if [ ! -f "$checksum_file" ] \
    && [ -f /usr/local/share/evoke/onnxruntime.sha256 ]; then
    checksum_file='/usr/local/share/evoke/onnxruntime.sha256'
fi
if [ ! -f "$checksum_file" ]; then
    echo 'ONNX Runtime checksum manifest was not found' >&2
    exit 1
fi
if [ -n "$archive_override" ]; then
    archive_name="$(basename "$archive_override")"
else
    archive_name="${asset}.tgz"
fi
expected_sha256="$(awk -v archive="$archive_name" \
    '$2 == archive {print $1}' "$checksum_file")"
if [ -z "$expected_sha256" ]; then
    echo "ONNX Runtime checksum is not pinned: ${archive_name}" >&2
    exit 1
fi

temp_dir="$(mktemp -d "${TMPDIR:-/tmp}/evoke-onnxruntime.XXXXXX")"
trap 'rm -rf "$temp_dir"' EXIT
archive="${temp_dir}/${archive_name}"
url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive_name}"

if [ -n "$archive_override" ]; then
    cp "$archive_override" "$archive"
else
    curl -fsSL "$url" -o "$archive"
fi
if command -v sha256sum >/dev/null 2>&1; then
    actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
else
    echo 'no SHA-256 checksum tool is available' >&2
    exit 1
fi
if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "ONNX Runtime checksum mismatch for ${archive_name}" >&2
    exit 1
fi
tar -xzf "$archive" -C "$temp_dir"
if [ -z "$asset_override" ] && [ ! -d "${temp_dir}/${asset}" ]; then
    case "$archive_name" in
        *.tar.gz)
            asset="${archive_name%.tar.gz}"
            ;;
        *.tgz)
            asset="${archive_name%.tgz}"
            ;;
    esac
fi
if [ ! -d "${temp_dir}/${asset}" ]; then
    echo "ONNX Runtime archive root was not found: ${asset}" >&2
    exit 1
fi
mkdir -p \
    "$prefix/include/onnxruntime" \
    "$prefix/lib/pkgconfig" \
    "$prefix/share/licenses/onnxruntime"
include_root="${temp_dir}/${asset}/include"
if [ -d "${include_root}/onnxruntime/core/session" ]; then
    cp "${include_root}/onnxruntime/core/session"/onnxruntime*.h \
        "$prefix/include/onnxruntime/"
    if [ -f "${include_root}/onnxruntime/core/framework/provider_options.h" ]; then
        cp "${include_root}/onnxruntime/core/framework/provider_options.h" \
            "$prefix/include/onnxruntime/"
    fi
else
    cp -R "${include_root}/." "$prefix/include/onnxruntime/"
fi
shopt -s nullglob
# The release archives contain the SONAME link and the versioned library.
# Preserve both so build-time and runtime lookup behave identically.
library_candidates=("${temp_dir}/${asset}/lib/"${library_pattern})
libraries=()
for library in "${library_candidates[@]}"; do
    if [ -f "$library" ] || [ -L "$library" ]; then
        libraries+=("$library")
    fi
done
if [ "${#libraries[@]}" -eq 0 ]; then
    echo "ONNX Runtime archive has no matching shared library" >&2
    exit 1
fi
cp -P "${libraries[@]}" "$prefix/lib/"
cp "${temp_dir}/${asset}/LICENSE" \
    "$prefix/share/licenses/onnxruntime/LICENSE"

cat >"$prefix/lib/pkgconfig/libonnxruntime.pc" <<EOF
prefix=${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include/onnxruntime

Name: onnxruntime
Description: ONNX Runtime C API
URL: https://github.com/microsoft/onnxruntime
Version: ${version}
Libs: -L\${libdir} -lonnxruntime
Cflags: -I\${includedir}
EOF

echo "$prefix"
