#!/usr/bin/env bash
set -euo pipefail

SESSION='ii42_m233a_embeddinggemma_teacher_gpu0'
ROOT='/home/huoju/leask/runs/ii42-m233-embeddinggemma-teacher-v1'
SOURCE='/home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1/source/scripts'
OUT="${ROOT}/runs/m233a-embeddinggemma2-teacher-preflight-gpu0-v1"
LOG="${ROOT}/logs/m233a-embeddinggemma2-teacher-preflight-gpu0-v1.log"
DEPS="${ROOT}/runtime/python"
CACHE="${ROOT}/embedding-cache/m233a-google-embeddinggemma-2-len512"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'session_exists=%s\n' "${SESSION}" >&2
    exit 2
fi

if pgrep -af '[r]esearch_sae_m233_embeddinggemma_teacher_preflight.py' \
    >/tmp/m233a_pgrep.txt; then
    cat /tmp/m233a_pgrep.txt >&2
    exit 3
fi

if [ -e "${OUT}" ]; then
    printf 'output_exists=%s\n' "${OUT}" >&2
    exit 4
fi

mkdir -p "$(dirname "${OUT}")" "$(dirname "${LOG}")" "${DEPS}" "${CACHE}"

tmux new-session -d -s "${SESSION}" -c "${SOURCE}" bash -lc "
set -euo pipefail
exec > '${LOG}' 2>&1
printf '[%s] M233A EmbeddingGemma 2 teacher preflight controller start\n' \
    \"\$(date -Is)\"

if [ ! -e '${DEPS}/.m233-eg2-deps-ok' ]; then
    printf '[%s] installing isolated EmbeddingGemma dependencies\n' \
        \"\$(date -Is)\"
    /home/huoju/leask/research-runtime/ng9-py312/bin/python -m ensurepip \
        --upgrade
    /home/huoju/leask/research-runtime/ng9-py312/bin/python -m pip install \
        --target '${DEPS}' \
        --upgrade \
        --no-cache-dir \
        --no-deps \
        'tokenizers>=0.23.2' \
        'git+https://github.com/huggingface/transformers.git'
    touch '${DEPS}/.m233-eg2-deps-ok'
fi

while true; do
    busy_count=\$(nvidia-smi --id=0 \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null | wc -l)
    gpu_util=\$(nvidia-smi --id=0 \
        --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    mem_used=\$(nvidia-smi --id=0 \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    printf '[%s] gpu0 check busy=%s util=%s mem=%sMiB\n' \
        \"\$(date -Is)\" \"\${busy_count}\" \"\${gpu_util}\" \"\${mem_used}\"
    if [ \"\${busy_count}\" -eq 0 ] && [ \"\${gpu_util}\" -le 5 ]; then
        break
    fi
    sleep 300
done

printf '[%s] M233A preflight start: %s\n' \"\$(date -Is)\" '${OUT}'
export PYTHONPATH='${DEPS}':/home/huoju/leask/research/NG-0084/runtime/lib/python3.12/site-packages:'${SOURCE}'
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export CLEARML_OFFLINE_MODE=1
export HF_HOME='${ROOT}/model-cache'
export HF_HUB_CACHE='${ROOT}/model-cache/hub'

/home/huoju/leask/research-runtime/ng9-py312/bin/python -B \
    research_sae_m233_embeddinggemma_teacher_preflight.py \
    --run-root /home/huoju/leask/runs/ii42-m1970-compact-dense-cascade-v1 \
    --dense-root /home/huoju/leask/runs/m150-beir-full-pplx \
    --official-root /home/huoju/leask/runs/ii42-m310b-official-roots-v1 \
    --reference-summary /home/huoju/leask/runs/ii42-m203-dual-head-operator-ceiling-v1/runs/m203a-full7-source1000-operator-sweep-v1/summary.json \
    --output-dir '${OUT}' \
    --cache-dir '${CACHE}' \
    --eval-datasets arguana,fiqa,nfcorpus,scidocs,scifact,trec-covid,webis-touche2020 \
    --model-name google/embeddinggemma-2 \
    --embedding-dimensions 128,256,512,768 \
    --fixed-weights 0.5,0.75,1.0,1.25,1.5,2.0,2.5,3.0,3.5,4.0 \
    --protected-head 0 \
    --correction-window 256 \
    --dense-weight 1.25 \
    --source-depth 1000 \
    --top-k 100 \
    --device cuda \
    --encode-batch-size 16 \
    --max-length 512 \
    --rrf-k 60
printf '[%s] M233A EmbeddingGemma 2 teacher preflight done\n' \
    \"\$(date -Is)\"
"

printf 'launched=%s\n' "${SESSION}"
