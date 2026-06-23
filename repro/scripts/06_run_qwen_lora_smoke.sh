#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DATA_DIR="artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data_smoke"
if [[ ! -s "${DATA_DIR}/raw_pool_gtr_rerank_train.jsonl" || ! -s "${DATA_DIR}/raw_pool_gtr_rerank_score_prompt_train.jsonl" ]]; then
  echo "Smoke SFT data missing; building ${DATA_DIR}"
  DEVICE="${DEVICE:-cpu}" \
  OUTPUT_DIR="${DATA_DIR}" \
  python build_cail_qwen_lora_sft_data.py \
    --train-path final_all_data/cail2018_statute_classification/train.jsonl \
    --valid-path final_all_data/cail2018_statute_classification/valid.jsonl \
    --test-path final_all_data/cail2018_statute_classification/test.jsonl \
    --checkpoint artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt \
    --embedding-cache artifacts/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz \
    --output-dir "${DATA_DIR}" \
    --max-train-rows "${SMOKE_MAX_TRAIN_ROWS:-2000}" \
    --max-valid-rows "${SMOKE_MAX_VALID_ROWS:-276}" \
    --max-test-rows 276 \
    --k 5 \
    --raw-pool-k 20 \
    --seed 42 \
    --sample-strategy random \
    --device "${DEVICE:-cpu}"
fi

GPU="${GPU:-0}" bash run_cail_qwen3_8b_lora_smoke.sh

