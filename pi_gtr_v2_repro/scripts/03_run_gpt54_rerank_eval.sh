#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: export OPENAI_API_KEY before running this script." >&2
  exit 1
fi

bash run_cail_gpt54_gtr_v2_rerank_276.sh

