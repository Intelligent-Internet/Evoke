#!/usr/bin/env bash
set -euo pipefail

SESSION='ii42_m230b_richer_m229_distill_gpu0'
ROOT='/home/huoju/leask/runs/ii42-m230-unified-dual-output-encoder-v1'
SOURCE='/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts'
LOG="${ROOT}/logs/m230b-richer-m229-distill-gpu0-v1.log"
SCRIPT='research_sae_m230_unified_dual_output_encoder.py'
OUT="${ROOT}/runs/m230b-ng80-s-richer-m229-distill-32d-int4-gpu0-v1"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'session_exists=%s\n' "${SESSION}" >&2
    exit 2
fi

if pgrep -af "[r]esearch_sae_m230_.*\\.py" >/tmp/m230b_pgrep.txt; then
    cat /tmp/m230b_pgrep.txt >&2
    exit 3
fi

if [ -f "${OUT}/summary.json" ]; then
    printf 'summary_exists=%s\n' "${OUT}/summary.json" >&2
    exit 4
fi

if [ -e "${OUT}" ]; then
    printf 'partial_output_exists=%s\n' "${OUT}" >&2
    exit 6
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

mkdir -p "${ROOT}/runs" "${ROOT}/logs" "${ROOT}/feature-cache"

tmux new-session -d -s "${SESSION}" -c "${SOURCE}" bash -lc "
set -euo pipefail
exec > '${LOG}' 2>&1
printf '[%s] M230B richer frozen-SAE M229 distill start\n' \"\$(date -Is)\"
export PYTHONPATH=/home/huoju/leask/research/NG-0084/runtime/lib/python3.12/site-packages:/home/huoju/leask/research/NG-0080/sparse-endpoints-v2:${SOURCE}
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export CLEARML_OFFLINE_MODE=1

/home/huoju/leask/research-runtime/ng9-py312/bin/python -B \
  '${SCRIPT}' \
  --run-label M230B \
  --run-root /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1 \
  --dense-root /home/huoju/leask/runs/m150-beir-full-pplx \
  --official-root /home/huoju/leask/runs/ii42-m310b-official-roots-v1 \
  --projection-basis /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/basis/shared5-1024.npz \
  --reference-summary /home/huoju/leask/runs/ii42-m203-dual-head-operator-ceiling-v1/runs/m203a-full7-source1000-operator-sweep-v1/summary.json \
  --m210-checkpoint-dir /home/huoju/leask/runs/ii42-m210-nonlinear-mini-vector-head-v1/runs/m210a-residual-mlp-score-shape-full7-32-64d-int4-gpu0-v1 \
  --teacher-source m229_meanz \
  --m229-checkpoint-paths /home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-301-query-rich-score-geometry-32d-int4-gpu0-v1/query_rich_512x2_student.pt,/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-313-query-rich-score-geometry-32d-int4-gpu0-v1/query_rich_512x2_student.pt,/home/huoju/leask/runs/ii42-m229-query-rich-score-geometry-v1/runs/m229b-student-seed-337-query-rich-score-geometry-32d-int4-gpu0-v1/query_rich_512x2_student.pt \
  --output-dir '${OUT}' \
  --feature-cache-dir '${ROOT}/feature-cache/m230b-ng80-s-richer-hidden-v1' \
  --ng80-research-root /home/huoju/leask/research \
  --ng80-source-dir /home/huoju/leask/research/NG-0080/sparse-endpoints-v2 \
  --ng80-config /home/huoju/leask/research/NG-0080/sparse-endpoints-v2/config.json \
  --sae-checkpoint /home/huoju/leask/research/NG-0080/sparse-S-v1/train-3072/checkpoint \
  --train-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid \
  --eval-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,webis-touche2020 \
  --dimension 32 \
  --quantization int4_row \
  --epochs 16 \
  --learning-rate 0.0004 \
  --weight-decay 0.0001 \
  --score-geometry-weight 0.25 \
  --positive-k 16 \
  --negative-start-rank 16 \
  --negative-k 128 \
  --min-positive-k 8 \
  --min-negative-k 32 \
  --fixed-weights 1.5,2.0,2.5,3.0,3.5 \
  --router-policies fixed,family_dense_half,m220c_joint \
  --rrf-k 60 \
  --protected-head 0 \
  --correction-window 256 \
  --dense-weight 1.25 \
  --source-depth 1000 \
  --top-k 100 \
  --seed 231 \
  --device cuda \
  --feature-batch-size 16 \
  --head-batch-size 8192 \
  --head-hidden-dim 1024 \
  --head-depth 3 \
  --feature-taps mean,max,first,sparse_summary \
  --sparse-summary-top-k 32 \
  --overlap-k 24 \
  --joint-threshold 0.30

printf '[%s] M230B richer frozen-SAE M229 distill done\n' \"\$(date -Is)\"
"

printf 'launched=%s\n' "${SESSION}"
