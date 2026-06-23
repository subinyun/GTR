#!/usr/bin/env bash
set -euo pipefail

# Run only the two GTR-reranked CAIL2018 GPT conditions:
# - raw_pool_gtr_rerank
# - raw_pool_gtr_rerank_score_prompt

export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

if [[ -z "${OPENAI_API_KEY}" ]]; then
  echo "ERROR: export OPENAI_API_KEY before running this script." >&2
  exit 1
fi

PROMPTS="artifacts/cail2018_gtr_v2_only/full/llm_prompts/gpt/prompts_276.jsonl"
OUT_DIR="artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276"
RESULTS="${OUT_DIR}/results_gpt54_rerank_276.jsonl"
METRICS_JSON="${OUT_DIR}/metrics_gpt54_rerank_276.json"
METRICS_CSV="${OUT_DIR}/metrics_gpt54_rerank_276.csv"
TRAIN_PATH="final_all_data/cail2018_statute_classification/train.jsonl"

mkdir -p "${OUT_DIR}"

echo "[1/2] Running GPT-5.4 on CAIL2018 GTR-reranked prompts"
echo "Prompts: ${PROMPTS}"
echo "Results: ${RESULTS}"
echo "Conditions: raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt"

python run_openai_prompt_jsonl.py \
  --prompts "${PROMPTS}" \
  --results "${RESULTS}" \
  --model gpt-5.4 \
  --conditions raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt

echo "[2/2] Evaluating GPT-5.4 CAIL2018 GTR-reranked results"

python evaluate_llm_candidate_routing_results.py \
  --prompts "${PROMPTS}" \
  --results "${RESULTS}" \
  --train-path "${TRAIN_PATH}" \
  --output-json "${METRICS_JSON}" \
  --output-csv "${METRICS_CSV}" \
  --conditions raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt \
  --k-values 5

echo "Done."
echo "Metrics JSON: ${METRICS_JSON}"
echo "Metrics CSV: ${METRICS_CSV}"
