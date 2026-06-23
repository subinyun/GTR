"""Training entrypoint vocabulary.

The included reproducibility package delegates actual training to the legacy
scripts at the repository root. This module records the paper-facing entrypoint
names used by configs and docs.
"""

from __future__ import annotations


TRAINING_ENTRYPOINTS = {
    "cail_gtr": "repro/scripts/01_train_cail_gtr.sh",
    "qwen_lora_smoke": "repro/scripts/06_run_qwen_lora_smoke.sh",
    "qwen_lora_full": "repro/scripts/07_run_qwen_lora_full.sh",
}
