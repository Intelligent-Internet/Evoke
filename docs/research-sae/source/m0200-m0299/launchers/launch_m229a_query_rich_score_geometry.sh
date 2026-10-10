#!/usr/bin/env bash
set -euo pipefail

SESSION='ii42_m229a_query_rich_gpu0'
ROOT='/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1'
SOURCE='/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts'
OUT="${ROOT}/runs/m229a-query-rich-score-geometry-32d-int4-epoch16-gpu0-v1"
LOG="${ROOT}/logs/m229a-query-rich-score-geometry-32d-int4-epoch16-gpu0-v1.log"
SCRIPT='research_sae_m229_query_rich_score_geometry_bridge_head.py'

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'session_exists=%s\n' "${SESSION}" >&2
    exit 2
fi

if pgrep -af "[r]esearch_sae_m229_.*\\.py" >/tmp/m229a_pgrep.txt; then
    cat /tmp/m229a_pgrep.txt >&2
    exit 3
fi

if [ -e "${OUT}" ]; then
    printf 'output_exists=%s\n' "${OUT}" >&2
    exit 4
fi

GPU_LINE="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i 0
)"
GPU_MEM="$(printf '%s\n' "${GPU_LINE}" | awk -F, '{gsub(/ /, "", $2); print $2}')"
GPU_UTIL="$(printf '%s\n' "${GPU_LINE}" | awk -F, '{gsub(/ /, "", $3); print $3}')"
if [ "${GPU_MEM}" -gt 512 ] || [ "${GPU_UTIL}" -gt 10 ]; then
    printf 'gpu0_busy=memory_mib:%s,util_pct:%s\n' \
        "${GPU_MEM}" "${GPU_UTIL}" >&2
    nvidia-smi -i 0 >&2 || true
    exit 5
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

tmux new-session -d -s "${SESSION}" -c "${SOURCE}" bash -lc "
set -euo pipefail
exec > '${LOG}' 2>&1
printf '[%s] M229A query-rich score-geometry bridge start: %s\n' \
  \"\$(date -Is)\" '${OUT}'
export PYTHONPATH=/home/huoju/leask/research/NG-0084/runtime/lib/python3.12/site-packages:${SOURCE}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CLEARML_OFFLINE_MODE=1
/home/huoju/leask/research-runtime/ng9-py312/bin/python -B \
  '${SCRIPT}' \
  --run-root /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1 \
  --dense-root /home/huoju/leask/runs/m150-beir-full-pplx \
  --official-root /home/huoju/leask/runs/ii42-m310b-official-roots-v1 \
  --projection-basis /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/basis/shared5-1024.npz \
  --reference-summary /home/huoju/leask/runs/ii42-m203-dual-head-operator-ceiling-v1/runs/m203a-full7-source1000-operator-sweep-v1/summary.json \
  --m210-checkpoint-dir /home/huoju/leask/runs/ii42-m210-nonlinear-mini-vector-head-v1/runs/m210a-residual-mlp-score-shape-full7-32-64d-int4-gpu0-v1 \
  --teacher-checkpoint-paths /home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-301-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt,/home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-313-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt,/home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-337-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt \
  --output-dir '${OUT}' \
  --train-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid \
  --eval-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,webis-touche2020 \
  --query-architectures query_rich_512x2,query_rich_768x3 \
  --objective-policy meanz_pair_meanz_score \
  --dimension 32 \
  --quantization int4_row \
  --epochs 16 \
  --learning-rate 0.0004 \
  --weight-decay 0.0001 \
  --distill-weight 0.2 \
  --score-geometry-weight 0.2 \
  --positive-k 16 \
  --negative-start-rank 16 \
  --negative-k 128 \
  --min-positive-k 8 \
  --min-negative-k 32 \
  --fixed-weights 2.5,3.0,3.25,3.5,3.75,4.0 \
  --router-policies fixed,family_dense_half,m220c_joint \
  --rrf-k 60 \
  --protected-head 0 \
  --correction-window 256 \
  --dense-weight 1.25 \
  --source-depth 1000 \
  --top-k 100 \
  --seed 229 \
  --device cuda \
  --encode-batch-size 8192 \
  --overlap-k 24 \
  --joint-threshold 0.30
printf '[%s] M229A query-rich score-geometry bridge done\n' \
  \"\$(date -Is)\"
"

printf 'launched=%s\n' "${SESSION}"
