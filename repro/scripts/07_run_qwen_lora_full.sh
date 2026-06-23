#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPU="${GPU:-0}" EPOCHS="${EPOCHS:-1}" LR="${LR:-1e-4}" bash run_cail_qwen3_8b_lora_full.sh

