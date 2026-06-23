#!/usr/bin/env python3
"""Evaluate LLM decoder outputs against LBOX statute labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


DEFAULT_TRAIN_PATH = Path("LBOX/statute_classification/train.jsonl")
DEFAULT_GOLD_PATH = Path("artifacts/full_gtr_fresh_20260528_1633/llm_decoder/gold.jsonl")
DEFAULT_RESULTS = Path("artifacts/full_gtr_fresh_20260528_1633/llm_decoder/results_dry_run.jsonl")
DEFAULT_OUTPUT = Path("artifacts/full_gtr_fresh_20260528_1633/llm_decoder/metrics_dry_run.json")


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


def normalize_statute_list(value: Any) -> list[str]:
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


def load_train_vocab(path: Path) -> list[str]:
    vocab: set[str] = set()
    for row in load_jsonl(path):
        vocab.update(normalize_statute_list(row.get("statutes")))
    return sorted(vocab)


def compute_metrics(gold: Sequence[Sequence[str]], pred: Sequence[Sequence[str]], classes: Sequence[str]) -> dict[str, Any]:
    vocab = set(classes)
    gold_filtered = [[label for label in labels if label in vocab] for labels in gold]
    pred_filtered = [[label for label in labels if label in vocab] for labels in pred]
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
        "avg_pred_labels": float(sum(len(labels) for labels in pred_filtered) / len(pred_filtered)) if pred_filtered else 0.0,
        "invalid_pred_labels": int(sum(len(set(labels) - vocab) for labels in pred)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    p.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD_PATH)
    p.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--output-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    classes = load_train_vocab(args.train_path)
    gold_by_index = {int(row["sample_index"]): normalize_statute_list(row.get("true_statutes")) for row in load_jsonl(args.gold_path)}
    pred_by_condition: dict[str, dict[int, list[str]]] = defaultdict(dict)
    parse_by_condition: dict[str, list[bool]] = defaultdict(list)
    for row in load_jsonl(args.results_path):
        condition = str(row.get("condition"))
        idx = int(row["sample_index"])
        pred_by_condition[condition][idx] = normalize_statute_list(row.get("predicted_statutes"))
        parse_by_condition[condition].append(bool(row.get("parse_ok", False)))
    rows: list[dict[str, Any]] = []
    for condition, pred_map in sorted(pred_by_condition.items()):
        indices = sorted(set(gold_by_index) & set(pred_map))
        metrics = compute_metrics([gold_by_index[i] for i in indices], [pred_map[i] for i in indices], classes)
        metrics["condition"] = condition
        metrics["parse_ok_rate"] = float(sum(parse_by_condition[condition]) / len(parse_by_condition[condition])) if parse_by_condition[condition] else 0.0
        rows.append(metrics)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.output_csv or args.output_json.with_suffix(".csv"), rows)
    print(json.dumps({"rows": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
