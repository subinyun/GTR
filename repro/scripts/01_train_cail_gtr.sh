#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEVICE="${DEVICE:-cuda}"
ENCODER_DEVICE="${ENCODER_DEVICE:-${DEVICE}}"
EPOCHS="${EPOCHS:-50}"
FIELD_PRETRAIN_EPOCHS="${FIELD_PRETRAIN_EPOCHS:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-output/cail2018_gtr_v2_only/full}"
EMBED_CACHE="${EMBED_CACHE:-${OUTPUT_DIR}/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz}"

mkdir -p "${OUTPUT_DIR}/cache"

python train_cail_hybrid_gtr_v2.py \
  --train-path final_all_data/cail2018_statute_classification/train.jsonl \
  --valid-path final_all_data/cail2018_statute_classification/valid.jsonl \
  --test-path final_all_data/cail2018_statute_classification/test.jsonl \
  --output-dir "${OUTPUT_DIR}" \
  --embed-cache "${EMBED_CACHE}" \
  --device "${DEVICE}" \
  --encoder-device "${ENCODER_DEVICE}" \
  --epochs "${EPOCHS}" \
  --field-pretrain-epochs "${FIELD_PRETRAIN_EPOCHS}" \
  --seed 42 \
  --sample-strategy random

echo "GTR checkpoint and report written under ${OUTPUT_DIR}"

