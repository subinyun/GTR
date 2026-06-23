#!/usr/bin/env bash
set -euo pipefail

# Run all four CAIL2018 Gemini + GTR prompt conditions on the same 276 samples:
# - full_fact
# - raw_topk
# - raw_pool_gtr_rerank
# - raw_pool_gtr_rerank_score_prompt
#
# Override MODEL if your Gemini endpoint uses a different exact model slug:
#   MODEL=gemini-3.5-flash bash run_cail_gemini35_gtr_v2_rerank_276.sh

export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

if [[ -z "${GOOGLE_API_KEY}" ]]; then
  echo "ERROR: export GOOGLE_API_KEY before running this script." >&2
  exit 1
fi

MODEL="${MODEL:-gemini-3.5-flash}"
PROMPTS="artifacts/cail2018_gtr_v2_only/full/llm_prompts/gemini/prompts_276.jsonl"
OUT_DIR="artifacts/cail2018_gtr_v2_only/full/gemini35_gtr_v2_all4_276"
RESULTS="${OUT_DIR}/results_gemini35_all4_276.jsonl"
METRICS_JSON="${OUT_DIR}/metrics_gemini35_all4_276.json"
METRICS_CSV="${OUT_DIR}/metrics_gemini35_all4_276.csv"
TRAIN_PATH="final_all_data/cail2018_statute_classification/train.jsonl"

mkdir -p "${OUT_DIR}"

echo "[1/2] Running ${MODEL} on CAIL2018 all four prompt conditions"
echo "Prompts: ${PROMPTS}"
echo "Results: ${RESULTS}"
echo "Conditions: full_fact raw_topk raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt"

python run_gemini_prompt_jsonl.py \
  --prompts "${PROMPTS}" \
  --results "${RESULTS}" \
  --model "${MODEL}" \
  --conditions full_fact raw_topk raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt \
  --sleep 0 \
  --retry-sleep 1

echo "[2/2] Evaluating ${MODEL} CAIL2018 all four prompt conditions"

python evaluate_llm_candidate_routing_results.py \
  --prompts "${PROMPTS}" \
  --results "${RESULTS}" \
  --train-path "${TRAIN_PATH}" \
  --output-json "${METRICS_JSON}" \
  --output-csv "${METRICS_CSV}" \
  --conditions full_fact raw_topk raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt \
  --k-values 0 5

echo "Done."
echo "Metrics JSON: ${METRICS_JSON}"
echo "Metrics CSV: ${METRICS_CSV}"
