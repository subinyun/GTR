#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "Repository: ${REPO_ROOT}"
echo "Python: $(python --version 2>&1 || true)"

missing=0

require_file() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    echo "OK   ${path}"
  else
    echo "MISS ${path}"
    missing=1
  fi
}

optional_file() {
  local path="$1"
  if [[ -s "${path}" ]]; then
    echo "OK   ${path}"
  else
    echo "WARN missing optional local file: ${path}"
  fi
}

echo
echo "[Data for full retraining]"
optional_file "final_all_data/cail2018_statute_classification/train.jsonl"
optional_file "final_all_data/cail2018_statute_classification/valid.jsonl"
optional_file "final_all_data/cail2018_statute_classification/test.jsonl"

echo
echo "[LBOX data included in this repository]"
require_file "LBOX/statute_classification/train.jsonl"
require_file "LBOX/statute_classification/valid.jsonl"
require_file "LBOX/statute_classification/test.jsonl"
require_file "LBOX/statute_classification/test2.jsonl"
require_file "LBOX/statute_classification/plus_train.jsonl"
require_file "LBOX/statute_classification/plus_valid.jsonl"
require_file "LBOX/statute_classification/plus_test.jsonl"

echo
echo "[Core source files]"
require_file "train_cail_hybrid_gtr_v2.py"
require_file "build_cail_gtr_v2_llm_prompts.py"
require_file "run_openai_prompt_jsonl.py"
require_file "run_gemini_prompt_jsonl.py"
require_file "evaluate_llm_candidate_routing_results.py"
require_file "build_cail_qwen_lora_sft_data.py"
require_file "train_qwen_completion_lora.py"
require_file "run_qwen_completion_eval.py"
require_file "evaluate_qwen_completion_results.py"

echo
echo "[Existing reusable artifacts]"
require_file "artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json"
if [[ -s "artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt" ]]; then
  echo "OK   artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt"
else
  echo "WARN missing checkpoint; run repro/scripts/01_train_cail_gtr.sh to rebuild it"
fi

echo
echo "[API keys]"
[[ -n "${OPENAI_API_KEY:-}" ]] && echo "OK   OPENAI_API_KEY is set" || echo "WARN OPENAI_API_KEY is not set"
[[ -n "${GOOGLE_API_KEY:-}" ]] && echo "OK   GOOGLE_API_KEY is set" || echo "WARN GOOGLE_API_KEY is not set"

echo
echo "[Python imports]"
python - <<'PY'
import importlib
required = [
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "sentence_transformers",
]
optional = [
    "openai",
    "google.genai",
    "transformers",
    "peft",
    "accelerate",
]
failed = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"OK   {name}")
    except Exception as exc:
        print(f"MISS {name}: {exc}")
        failed.append(name)
for name in optional:
    try:
        importlib.import_module(name)
        print(f"OK   optional {name}")
    except Exception as exc:
        print(f"WARN optional {name}: {exc}")
if failed:
    raise SystemExit(1)
PY

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "Required files are missing."
  exit 1
fi

echo
echo "Environment check completed."

