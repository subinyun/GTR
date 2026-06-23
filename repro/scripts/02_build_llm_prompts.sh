#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cpu}"
MAX_ROWS="${MAX_ROWS:-276}"
K="${K:-5}"
RAW_POOL_K="${RAW_POOL_K:-20}"
CHECKPOINT="${CHECKPOINT:-artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt}"
EMBED_CACHE="${EMBED_CACHE:-artifacts/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/cail2018_gtr_v2_only/full/llm_prompts}"

python build_cail_gtr_v2_llm_prompts.py \
  --test-path final_all_data/cail2018_statute_classification/test.jsonl \
  --checkpoint "${CHECKPOINT}" \
  --embedding-cache "${EMBED_CACHE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-rows "${MAX_ROWS}" \
  --k "${K}" \
  --raw-pool-k "${RAW_POOL_K}" \
  --seed 42 \
  --sample-strategy random \
  --device "${DEVICE}"

echo "Prompt files written under ${OUTPUT_DIR}"

