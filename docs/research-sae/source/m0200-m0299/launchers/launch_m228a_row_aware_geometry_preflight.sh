#!/usr/bin/env bash
set -euo pipefail

SESSION='ii42_m228a_row_aware_geometry_gpu0'
ROOT='/home/huoju/leask/runs/ii42-m228-row-aware-geometry-preflight-v1'
SOURCE='/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts'
OUT="${ROOT}/runs/m228a-qrels-free-row-aware-geometry-teacher-preflight-gpu0-v1"
LOG="${ROOT}/logs/m228a-qrels-free-row-aware-geometry-teacher-preflight-gpu0-v1.log"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'session_exists=%s\n' "${SESSION}" >&2
    exit 2
fi

if pgrep -af '[r]esearch_sae_m228_row_aware_geometry_preflight.py' >/tmp/m228a_pgrep.txt; then
    cat /tmp/m228a_pgrep.txt >&2
    exit 3
fi

if [ -e "${OUT}" ]; then
    printf 'output_exists=%s\n' "${OUT}" >&2
    exit 4
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")"

tmux new-session -d -s "${SESSION}" -c "${SOURCE}" bash -lc "
set -euo pipefail
exec > '${LOG}' 2>&1
printf '[%s] M228A row-aware qrels-free geometry preflight start: %s\n' \
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
  research_sae_m228_row_aware_geometry_preflight.py \
  --run-root /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1 \
  --dense-root /home/huoju/leask/runs/m150-beir-full-pplx \
  --official-root /home/huoju/leask/runs/ii42-m310b-official-roots-v1 \
  --projection-basis /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/basis/shared5-1024.npz \
  --reference-summary /home/huoju/leask/runs/ii42-m203-dual-head-operator-ceiling-v1/runs/m203a-full7-source1000-operator-sweep-v1/summary.json \
  --m210-checkpoint-dir /home/huoju/leask/runs/ii42-m210-nonlinear-mini-vector-head-v1/runs/m210a-residual-mlp-score-shape-full7-32-64d-int4-gpu0-v1 \
  --teacher-score-cache-dir /home/huoju/leask/runs/ii42-m217-rankt5-pseudo-global-head-v1/teacher-cache/m217a-full7-top256-rankt5-v1 \
  --supervised-reference-checkpoint-paths /home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-301-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt,/home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-313-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt,/home/huoju/leask/runs/ii42-m214-row-expert-mini-vector-head-v1/runs/m214b-seed-337-row-expert-qrels-safe-32d-int4-epoch8-gpu0-v1/dim32_global.pt \
  --output-dir '${OUT}' \
  --eval-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,webis-touche2020 \
  --teacher-policies agreement_soft_mean_z,agreement_hard_mean_z,triad_consensus_mean_z,dense_anchor_residual_soft,dense_anchor_residual_hard,rankt5_dense_rrf_guarded,risk_dropout_mean_z,family_damped_mean_z,family_damped_residual \
  --fixed-weights 0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5 \
  --dimension 32 \
  --protected-head 0 \
  --correction-window 256 \
  --dense-weight 1.25 \
  --source-depth 1000 \
  --top-k 100 \
  --device cuda \
  --encode-batch-size 8192 \
  --rrf-k 60 \
  --similarity-ks 24,64,100 \
  --gate-k 24
printf '[%s] M228A row-aware qrels-free geometry preflight done\n' \
  \"\$(date -Is)\"
"

printf 'launched=%s\n' "${SESSION}"
