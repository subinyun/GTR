#!/usr/bin/env python3
"""Assemble result-table metrics from included artifacts.

This script is intentionally artifact-driven: it does not retrain models by
default. It normalizes the included reports into `outputs/metrics/main_table.json`
so the result tables can be regenerated deterministically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gtr.provenance import build_provenance


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp/main_table.yaml")
    parser.add_argument("--output", default="outputs/metrics/main_table.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config_copy = Path("outputs/configs") / config_path.name
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    cail_report = read_json("artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json")
    gpt_metrics = read_json(
        "artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json"
    )
    qwen_metrics = read_json("artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json")
    ablation = read_json("artifacts/supporting_claims/hybrid_gtr_v2_ablation_report.json")
    hard_negative = read_json("artifacts/supporting_claims/hard_negative_margin_report.json")
    mechanism = read_json("artifacts/supporting_claims/hybrid_gtr_v2_mechanism_report.json")
    threshold = read_json("artifacts/supporting_claims/threshold_vs_gtr_report.json")

    payload = {
        "provenance": build_provenance(args.config, seed=42),
        "config": args.config,
        "tables": {
            "main_results": {
                "cail_gtr_test": cail_report.get("test_metrics", {}),
                "gpt54_llm_routing": gpt_metrics.get("rows", []),
                "qwen_lora_full": qwen_metrics.get("rows", []),
            },
            "ablation": ablation.get("ablation_table", []),
            "threshold_vs_gtr": threshold.get("main_table", []),
            "suppression": mechanism.get("suppression", []),
            "hard_negative": hard_negative.get("aggregate", {}),
        },
        "sources": [
            "artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json",
            "artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json",
            "artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json",
            "artifacts/supporting_claims/hybrid_gtr_v2_ablation_report.json",
            "artifacts/supporting_claims/threshold_vs_gtr_report.json",
            "artifacts/supporting_claims/hybrid_gtr_v2_mechanism_report.json",
            "artifacts/supporting_claims/hard_negative_margin_report.json",
        ],
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
