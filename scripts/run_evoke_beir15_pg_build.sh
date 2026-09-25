#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${EVOKE_BEIR15_PG_LOG_ROOT:-${TMPDIR:-/tmp}/evoke-beir15-pg-build/logs}"
mkdir -p "$LOG_ROOT"

if [[ -z "${EVOKE_BEIR15_SOURCE:-}" ]]; then
    echo 'EVOKE_BEIR15_SOURCE must name the rsync source root' >&2
    exit 2
fi

DATASETS=(
    nfcorpus
    scifact
    arguana
    scidocs
    fiqa
    trec-covid
    webis-touche2020
    cqadupstack
    quora
    nq
    dbpedia-entity
    hotpotqa
    fever
    climate-fever
    msmarco
)

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
fi

cd "$ROOT"

for dataset in "${DATASETS[@]}"; do
    log_file="$LOG_ROOT/${dataset}.log"
    done_file="$LOG_ROOT/${dataset}.done"
    if [[ -f "$done_file" ]]; then
        echo "skip ${dataset}: done marker exists"
        continue
    fi
    {
        echo "== ${dataset} start $(date '+%Y-%m-%dT%H:%M:%S%z') =="
        python3 scripts/build_evoke_beir15_pg.py sync --datasets "$dataset"
        python3 scripts/build_evoke_beir15_pg.py init --datasets "$dataset"
        python3 scripts/build_evoke_beir15_pg.py load --datasets "$dataset"
        python3 scripts/build_evoke_beir15_pg.py index --datasets "$dataset" --reindex
        python3 scripts/build_evoke_beir15_pg.py verify --datasets "$dataset"
        python3 scripts/build_evoke_beir15_pg.py manifest
        echo "== ${dataset} done $(date '+%Y-%m-%dT%H:%M:%S%z') =="
    } 2>&1 | tee "$log_file"
    touch "$done_file"
done
