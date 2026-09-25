#!/usr/bin/env bash
set -euo pipefail

require_env() {
    local name=$1

    if [[ -z ${!name:-} ]]; then
        printf 'missing required environment variable: %s\n' "$name" >&2
        exit 2
    fi
}

for name in \
    CONTAINER DSN PYTHON_BIN SCRIPT_DIR DOCUMENTS_JSONL QUERIES_JSONL \
    QRELS_JSON EXPECTED_ROWS EXPECTED_QUERIES DATASET SCHEMA TABLE \
    MODEL_PATH OUTPUT_DIR
do
    require_env "$name"
done

for identifier in "$SCHEMA" "$TABLE"; do
    if [[ ! $identifier =~ ^[a-z_][a-z0-9_]*$ ]]; then
        printf 'unsafe SQL identifier: %s\n' "$identifier" >&2
        exit 2
    fi
done

if [[ ! $EXPECTED_ROWS =~ ^[1-9][0-9]*$ ]]; then
    printf 'EXPECTED_ROWS must be a positive integer\n' >&2
    exit 2
fi
if [[ ! $EXPECTED_QUERIES =~ ^[1-9][0-9]*$ ]]; then
    printf 'EXPECTED_QUERIES must be a positive integer\n' >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

dataset_slug=${DATASET//[^a-zA-Z0-9_-]/_}
load_json="$OUTPUT_DIR/${dataset_slug}-official-load.json"
result_json="$OUTPUT_DIR/${dataset_slug}-official-native-qrels.json"
result_md="$OUTPUT_DIR/${dataset_slug}-official-native-qrels.md"
build_json="$OUTPUT_DIR/${dataset_slug}-official-build.json"
bm25_index="${SCHEMA}.${TABLE}_bm25_idx"
p2_index="${SCHEMA}.${TABLE}_p2_idx"

psql_exec() {
    docker exec "$CONTAINER" \
        psql -U postgres -d postgres -v ON_ERROR_STOP=1 "$@"
}

elapsed_seconds() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys

started = int(sys.argv[1])
finished = int(sys.argv[2])
print(f'{(finished - started) / 1_000_000_000:.6f}')
PY
}

psql_exec -c "DROP SCHEMA IF EXISTS $SCHEMA CASCADE; CHECKPOINT;"

"$PYTHON_BIN" "$SCRIPT_DIR/load_evoke_native_qrels_corpus.py" \
    --dsn "$DSN" \
    --input "$DOCUMENTS_JSONL" \
    --schema "$SCHEMA" \
    --table "$TABLE" \
    --create-schema \
    --expected-rows "$EXPECTED_ROWS" \
    --output-json "$load_json"

bm25_started=$(date +%s%N)
psql_exec -c "
    SET maintenance_work_mem='2GB';
    CREATE INDEX ${TABLE}_bm25_idx
    ON $SCHEMA.$TABLE USING evoke (text_content)
    WITH (
        method='lucene',
        idf_method='lucene',
        consistency='manual'
    );
"
bm25_finished=$(date +%s%N)

p2_started=$(date +%s%N)
psql_exec -c "
    SET maintenance_work_mem='2GB';
    CREATE INDEX ${TABLE}_p2_idx
    ON $SCHEMA.$TABLE USING evoke (text_content)
    WITH (
        sae=true,
        model_path='$MODEL_PATH',
        consistency='manual'
    );
"
p2_finished=$(date +%s%N)

psql_exec -c "ANALYZE $SCHEMA.$TABLE; CHECKPOINT;"

bm25_seconds=$(elapsed_seconds "$bm25_started" "$bm25_finished")
p2_seconds=$(elapsed_seconds "$p2_started" "$p2_finished")
"$PYTHON_BIN" - "$build_json" "$DATASET" "$EXPECTED_ROWS" \
    "$bm25_seconds" "$p2_seconds" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    'dataset': sys.argv[2],
    'document_count': int(sys.argv[3]),
    'build_seconds': {
        'BM25': float(sys.argv[4]),
        'P2.2': float(sys.argv[5]),
    },
}
path.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + '\n',
    encoding='utf-8',
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_evoke_native_qrels.py" \
    --dsn "$DSN" \
    --dataset "$DATASET" \
    --schema "$SCHEMA" \
    --table "$TABLE" \
    --index "BM25=$bm25_index" \
    --index "P2.2=$p2_index" \
    --queries-jsonl "$QUERIES_JSONL" \
    --qrels-json "$QRELS_JSON" \
    --k 1000 \
    --expected-documents "$EXPECTED_ROWS" \
    --expected-queries "$EXPECTED_QUERIES" \
    --progress-every 10 \
    --output-json "$result_json" \
    --output-md "$result_md"

printf 'EVOKE_NATIVE_QRELS_COMPLETE dataset=%s rows=%s\n' \
    "$DATASET" "$EXPECTED_ROWS"
