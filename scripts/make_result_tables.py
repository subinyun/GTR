#!/usr/bin/env python3
"""Render compact markdown result tables from normalized metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/metrics")
    parser.add_argument("--output", default="outputs/metrics/result_tables.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.input) / "main_table.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    main = payload["tables"]["main_results"]

    lines: list[str] = ["# Result Tables", ""]
    lines.extend(
        [
            "## Table 1. Main Results",
            "",
            "| Condition | Exact Match | Micro-F1 | Macro-F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    cail = main["cail_gtr_test"]
    lines.append(
        f"| CAIL GTR checkpoint | {fmt(cail.get('exact_match'))} | {fmt(cail.get('micro_f1'))} | {fmt(cail.get('macro_f1'))} |"
    )
    for row in main["gpt54_llm_routing"]:
        lines.append(
            f"| GPT-5.4 `{row.get('condition')}` | {fmt(row.get('exact_match'))} | {fmt(row.get('micro_f1'))} | {fmt(row.get('macro_f1'))} |"
        )
    for row in main["qwen_lora_full"]:
        lines.append(
            f"| Qwen3-8B LoRA `{row.get('condition')}` | {fmt(row.get('exact_match'))} | {fmt(row.get('micro_f1'))} | {fmt(row.get('macro_f1'))} |"
        )

    threshold_rows = payload["tables"].get("threshold_vs_gtr", [])
    lines.extend(
        [
            "",
            "## Table 3. Threshold vs GTR",
            "",
            "| Method | Exact Match | Micro-F1 | Macro-F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in threshold_rows:
        test = row.get("test", {})
        lines.append(
            f"| {row.get('method')} | {fmt(test.get('exact_match'))} | {fmt(test.get('micro_f1'))} | {fmt(test.get('macro_f1'))} |"
        )

    hard_negative = payload["tables"]["hard_negative"]
    lines.extend(
        [
            "",
            "## Figure/Table Support. Hard-Negative Margin",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| raw mean margin | {fmt(hard_negative.get('raw_head_on_z_mean_margin'))} |",
            f"| full hybrid mean margin | {fmt(hard_negative.get('full_hybrid_mean_margin'))} |",
            f"| raw confuser FP rate | {fmt(hard_negative.get('raw_head_on_z_confuser_fp_rate'))} |",
            f"| full hybrid confuser FP rate | {fmt(hard_negative.get('full_hybrid_confuser_fp_rate'))} |",
            "",
        ]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
