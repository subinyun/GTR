#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-7}"
DATA_DIR="artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data_smoke"
OUT_ROOT="artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/smoke"
TRAIN_PATH="final_all_data/cail2018_statute_classification/train.jsonl"
CONDITIONS_STR="${CONDITIONS_STR:-raw_pool_gtr_rerank raw_pool_gtr_rerank_score_prompt}"
read -r -a CONDITIONS <<< "${CONDITIONS_STR}"

mkdir -p "${OUT_ROOT}/logs"

echo "MODEL_NAME=${MODEL_NAME}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "GPU=${GPU}"
echo "CONDITIONS=${CONDITIONS[*]}"

for CONDITION in "${CONDITIONS[@]}"; do
  CKPT_DIR="${OUT_ROOT}/${CONDITION}_adapter"
  EVAL_DIR="${OUT_ROOT}/${CONDITION}_eval"
  LOG_PATH="${OUT_ROOT}/logs/${CONDITION}.log"
  for SPLIT in train valid test; do
    if [[ ! -s "${DATA_DIR}/${CONDITION}_${SPLIT}.jsonl" ]]; then
      echo "ERROR: missing data file ${DATA_DIR}/${CONDITION}_${SPLIT}.jsonl" >&2
      exit 1
    fi
  done
  echo "=== LoRA smoke: ${CONDITION} on GPU ${GPU} ===" | tee "${LOG_PATH}"

  CUDA_VISIBLE_DEVICES="${GPU}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" train_qwen_completion_lora.py \
      --model-name-or-path "${MODEL_NAME}" \
      --data-path "${DATA_DIR}/${CONDITION}_train.jsonl" \
      --valid-data-path "${DATA_DIR}/${CONDITION}_valid.jsonl" \
      --output-dir "${CKPT_DIR}" \
      --device cuda \
      --epochs 1 \
      --max-steps 200 \
      --lr 1e-4 \
      --batch-size 1 \
      --gradient-accumulation-steps 8 \
      --max-length 4096 \
      --log-every 10 \
      --save-every-steps 100 \
      --use-chat-template \
      2>&1 | tee -a "${LOG_PATH}"

  mkdir -p "${EVAL_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU}" TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON_BIN}" run_qwen_completion_eval.py \
      --prompt-path "${DATA_DIR}/${CONDITION}_test.jsonl" \
      --output-dir "${EVAL_DIR}" \
      --results-filename "predictions_test.jsonl" \
      --model "${MODEL_NAME}" \
      --adapter-path "${CKPT_DIR}" \
      --device cuda \
      --condition "${CONDITION}" \
      --max-input-tokens 4096 \
      --max-new-tokens 80 \
      --use-chat-template \
      2>&1 | tee -a "${LOG_PATH}"

  "${PYTHON_BIN}" evaluate_qwen_completion_results.py \
    --train-path "${TRAIN_PATH}" \
    --prompt-path "${DATA_DIR}/${CONDITION}_test.jsonl" \
    --results-path "${EVAL_DIR}/predictions_test.jsonl" \
    --output-json "${EVAL_DIR}/metrics.json" \
    --output-csv "${EVAL_DIR}/metrics.csv" \
    --raw-sample-output "${EVAL_DIR}/parse_failures.json" \
    2>&1 | tee -a "${LOG_PATH}"
done

"${PYTHON_BIN}" - <<'PY'
import csv
import json
from pathlib import Path

root = Path("artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/smoke")
rows = []
for metrics_path in sorted(root.glob("*_eval/metrics.json")):
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.extend(payload.get("rows", []))

summary_json = root / "summary_metrics.json"
summary_csv = root / "summary_metrics.csv"
summary_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
keys = []
for row in rows:
    for key in row:
        if key not in keys:
            keys.append(key)
with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"rows": rows, "summary_csv": str(summary_csv)}, ensure_ascii=False, indent=2))
PY
