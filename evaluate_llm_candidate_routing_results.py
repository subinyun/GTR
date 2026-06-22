#!/usr/bin/env python3
"""Evaluate actual LLM candidate-routing API results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from evaluate_gtr_llm_decoder import compute_metrics, load_train_vocab, normalize_statute_list


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def prompt_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (int(row["sample_index"]), str(row["condition"]), int(row.get("k", 0)))


def run(args: argparse.Namespace) -> Dict[str, Any]:
    classes = load_train_vocab(args.train_path)
    prompts = load_jsonl(args.prompts)
    results = load_jsonl(args.results)
    prompt_by_key = {prompt_key(row): row for row in prompts}
    result_by_key = {prompt_key(row): row for row in results}

    grouped: Dict[tuple[str, int], List[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for key, result in result_by_key.items():
        prompt = prompt_by_key.get(key)
        if prompt is None:
            continue
        condition = str(result.get("condition", prompt.get("condition")))
        k = int(result.get("k", prompt.get("k", 0)))
        if args.k_values and k not in set(args.k_values):
            continue
        if args.conditions and condition not in set(args.conditions):
            continue
        grouped[(condition, k)].append((prompt, result))

    rows: List[Dict[str, Any]] = []
    for (condition, k), pairs in sorted(grouped.items()):
        gold = [normalize_statute_list(prompt.get("true_statutes")) for prompt, _ in pairs]
        pred = [normalize_statute_list(result.get("predicted_statutes")) for _, result in pairs]
        metrics = compute_metrics(gold, pred, classes)
        metrics.update(
            {
                "condition": condition,
                "k": k,
                "parse_ok_rate": sum(bool(result.get("parse_ok", False)) for _, result in pairs) / len(pairs)
                if pairs
                else 0.0,
                "model": str(pairs[0][1].get("model", "")) if pairs else "",
            }
        )
        rows.append(metrics)

    report = {
        "meta": {
            "prompts": str(args.prompts),
            "results": str(args.results),
            "train_path": str(args.train_path),
            "n_result_rows": len(results),
        },
        "rows": rows,
    }
    write_json(args.output_json, report)
    write_csv(args.output_csv, rows)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=Path("output/hybrid_gtr_v2_improved/llm_candidate_routing/llm_candidate_routing_prompts.jsonl"))
    p.add_argument("--results", type=Path, default=Path("output/hybrid_gtr_v2_improved/llm_candidate_routing/results_openai_gpt54_k5.jsonl"))
    p.add_argument("--train-path", type=Path, default=Path("LBOX/statute_classification/train.jsonl"))
    p.add_argument("--output-json", type=Path, default=Path("output/hybrid_gtr_v2_improved/llm_candidate_routing/actual_llm_candidate_routing_metrics.json"))
    p.add_argument("--output-csv", type=Path, default=Path("output/hybrid_gtr_v2_improved/llm_candidate_routing/actual_llm_candidate_routing_metrics.csv"))
    p.add_argument("--k-values", type=int, nargs="*", default=None)
    p.add_argument("--conditions", nargs="*", default=None)
    return p.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["rows"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
