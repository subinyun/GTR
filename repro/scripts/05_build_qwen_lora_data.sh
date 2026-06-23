#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cpu}"
CHECKPOINT="${CHECKPOINT:-artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt}"
EMBED_CACHE="${EMBED_CACHE:-artifacts/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data_full}"

python build_cail_qwen_lora_sft_data.py \
  --train-path final_all_data/cail2018_statute_classification/train.jsonl \
  --valid-path final_all_data/cail2018_statute_classification/valid.jsonl \
  --test-path final_all_data/cail2018_statute_classification/test.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --embedding-cache "${EMBED_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-train-rows 0 \
  --max-valid-rows 0 \
  --max-test-rows 276 \
  --k 5 \
  --raw-pool-k 20 \
  --seed 42 \
  --sample-strategy random \
  --device "${DEVICE}"

echo "Qwen LoRA SFT data written under ${OUTPUT_DIR}"

