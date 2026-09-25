#!/usr/bin/env bash

set -euo pipefail

output="${1:-}"
if [ -z "$output" ]; then
    echo 'usage: fetch_milestone_model_for_ci.sh OUTPUT' >&2
    exit 1
fi
model_url="${EVOKE_MILESTONE_MODEL_URL:-}"
model_sha256="${EVOKE_MILESTONE_MODEL_ARCHIVE_SHA256:-}"
if [ -z "$model_url" ]; then
    echo 'EVOKE_MILESTONE_MODEL_URL repository variable is required' >&2
    exit 1
fi

args=(
    --url "$model_url"
    --output "$output"
    --overwrite
)
if [ -n "$model_sha256" ]; then
    args+=(
        --archive-sha256 "$model_sha256"
    )
fi
python3 scripts/fetch_milestone_model.py "${args[@]}"

if [ -n "${GITHUB_ENV:-}" ]; then
    echo "EVOKE_MILESTONE_MODEL_CHECKOUT=${output}" >>"$GITHUB_ENV"
fi
