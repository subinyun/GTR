#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROMPTS="artifacts/cail2018_gtr_v2_only/full/llm_prompts/gpt/prompts_276.jsonl"
RESULTS="artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/results_gpt54_rerank_276.jsonl"
OUT_DIR="outputs/metrics/recomputed"

mkdir -p "${OUT_DIR}"

python evaluate_llm_candidate_routing_results.py \
  --prompts "${PROMPTS}" \
  --results "${RESULTS}" \
  --train-path final_all_data/cail2018_statute_classification/train.jsonl \
  --output-json "${OUT_DIR}/metrics_gpt54_rerank_276.json" \
  --output-csv "${OUT_DIR}/metrics_gpt54_rerank_276.csv" \
  --conditions raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt \
  --k-values 5

echo "Recomputed metrics written under ${OUT_DIR}"

