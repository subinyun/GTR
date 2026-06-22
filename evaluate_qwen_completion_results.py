#!/usr/bin/env python3
"""Evaluate completion-style Qwen statute outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def normalize(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def resolve_candidate_numbers(labels: list[str], row: dict[str, Any]) -> list[str]:
    candidates = [str(x) for x in row.get("candidates", [])]
    out: list[str] = []
    seen: set[str] = set()
    for label in labels:
        mapped = label
        if label.isdigit():
            idx = int(label) - 1
            if 0 <= idx < len(candidates):
                mapped = candidates[idx]
        if mapped and mapped not in seen:
            out.append(mapped)
            seen.add(mapped)
    return out


def train_vocab(path: Path) -> list[str]:
    labels: set[str] = set()
    for row in load_jsonl(path):
        labels.update(normalize(row.get("statutes")))
    return sorted(labels)


def compute_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], classes: Sequence[str]) -> dict[str, Any]:
    vocab = set(classes)
    gold_filtered = [[x for x in row if x in vocab] for row in gold]
    pred_filtered = [[x for x in row if x in vocab] for row in pred]
    mlb = MultiLabelBinarizer(classes=list(classes))
    mlb.fit([[]])
    y_true = mlb.transform(gold_filtered)
    y_pred = mlb.transform(pred_filtered)
    exact = [set(a) == set(b) for a, b in zip(gold_filtered, pred_filtered)]
    return {
        "n": len(gold_filtered),
        "exact_match": float(sum(exact) / len(exact)) if exact else 0.0,
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "avg_pred_labels": float(sum(len(row) for row in pred_filtered) / len(pred_filtered)) if pred_filtered else 0.0,
        "invalid_pred_labels": int(sum(len(set(row) - vocab) for row in pred)),
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    classes = train_vocab(args.train_path)
    prompt_by_key: dict[tuple[int, str, int], dict[str, Any]] = {}
    if args.prompt_path is not None:
        for row in load_jsonl(args.prompt_path):
            prompt_by_key[(int(row["sample_index"]), str(row["condition"]), int(row.get("k", 0)))] = row
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(args.results_path):
        prompt_row = prompt_by_key.get((int(row["sample_index"]), str(row["condition"]), int(row.get("k", 0))))
        if prompt_row:
            row = {**prompt_row, **row, "candidates": prompt_row.get("candidates", row.get("candidates", []))}
            row["gold_statutes"] = row.get("gold_statutes") or prompt_row.get("gold_statutes") or prompt_row.get("true_statutes")
            row["true_statutes"] = row.get("true_statutes") or prompt_row.get("true_statutes") or prompt_row.get("gold_statutes")
        by_condition[str(row.get("condition"))].append(row)

    metric_rows: list[dict[str, Any]] = []
    raw_samples: list[dict[str, Any]] = []
    for condition, rows in sorted(by_condition.items()):
        rows.sort(key=lambda row: int(row["sample_index"]))
        gold = [normalize(row.get("gold_statutes", row.get("true_statutes"))) for row in rows]
        pred = [resolve_candidate_numbers(normalize(row.get("parsed_prediction", row.get("predicted_statutes"))), row) for row in rows]
        metrics = compute_metrics(gold, pred, classes)
        metrics.update(
            {
                "condition": condition,
                "k": int(rows[0].get("k", 0)) if rows else 0,
                "parse_ok_rate": float(sum(bool(row.get("parse_ok")) for row in rows) / len(rows)) if rows else 0.0,
                "model": str(rows[0].get("model", "")) if rows else "",
            }
        )
        metric_rows.append(metrics)
        failures = [row for row in rows if not row.get("parse_ok")]
        raw_samples.extend(
            {
                "sample_index": row.get("sample_index"),
                "condition": condition,
                "gold_statutes": row.get("gold_statutes"),
                "raw_generation": row.get("raw_generation"),
                "prompt_tail": str(row.get("prompt", ""))[-1200:],
            }
            for row in failures[: args.raw_samples_per_condition]
        )

    payload = {"meta": {"results": str(args.results_path), "train_path": str(args.train_path)}, "rows": metric_rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv or args.output_json.with_suffix(".csv"), metric_rows)
    if args.raw_sample_output:
        args.raw_sample_output.parent.mkdir(parents=True, exist_ok=True)
        args.raw_sample_output.write_text(json.dumps(raw_samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", type=Path, default=Path("LBOX/statute_classification/train.jsonl"))
    p.add_argument("--prompt-path", type=Path, default=None)
    p.add_argument("--results-path", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--raw-sample-output", type=Path, default=None)
    p.add_argument("--raw-samples-per-condition", type=int, default=8)
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args())["rows"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
