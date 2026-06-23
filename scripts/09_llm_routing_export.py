#!/usr/bin/env python3
"""Export saved LLM-routing metrics to standard output paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gtr.provenance import build_provenance


metrics_src = Path("artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json")
metrics_dst = Path("outputs/metrics/llm_routing.json")
metrics_dst.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "provenance": build_provenance("configs/exp/llm_routing.yaml", seed=42),
    "source": str(metrics_src),
    "report": json.loads(metrics_src.read_text(encoding="utf-8")),
    "prediction_artifact": "artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/results_gpt54_rerank_276.jsonl",
}
metrics_dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {metrics_dst}")
