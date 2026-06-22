#!/usr/bin/env python3
"""Build weak legal-element axis labels for LBOX Full GTR experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from gtr_axis_schema import get_axis_ids, get_axis_schema, get_statute_to_axes_map, save_axis_schema


DEFAULT_TRAIN_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/train.jsonl")
DEFAULT_VALID_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/valid.jsonl")
DEFAULT_TEST_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/test.jsonl")
DEFAULT_OUTPUT_DIR = Path("output/full_gtr/axis_labels")


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
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_text(row: Dict[str, Any]) -> str:
    for key in ("facts", "text", "fact", "content"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword and keyword in text for keyword in keywords)


def make_axis_labels(
    text: str,
    statutes: Sequence[str],
    *,
    use_keyword_rules: bool,
    allow_unknown_axis: bool,
) -> tuple[Dict[str, int], Dict[str, str]]:
    schema = get_axis_schema()
    axis_ids = get_axis_ids()
    statute_to_axes = get_statute_to_axes_map()
    unknown_value = -1 if allow_unknown_axis else 0
    labels = {axis_id: unknown_value for axis_id in axis_ids}
    sources = {axis_id: "unknown" if allow_unknown_axis else "default_negative" for axis_id in axis_ids}

    for statute in statutes:
        for axis_id in statute_to_axes.get(str(statute), []):
            labels[axis_id] = 1
            sources[axis_id] = "statute_mapping"

    if use_keyword_rules:
        for item in schema:
            axis_id = str(item["axis_id"])
            negative_hit = contains_any(text, [str(x) for x in item.get("negative_keywords", [])])
            positive_hit = contains_any(text, [str(x) for x in item.get("positive_keywords", [])])
            if negative_hit:
                if labels[axis_id] != 1:
                    labels[axis_id] = 0
                    sources[axis_id] = "rule_negative_keyword"
                continue
            if positive_hit and labels[axis_id] != 1:
                labels[axis_id] = 1
                sources[axis_id] = "rule_keyword"

    return labels, sources


def build_split(
    split_name: str,
    rows: Sequence[Dict[str, Any]],
    output_dir: Path,
    *,
    use_keyword_rules: bool,
    allow_unknown_axis: bool,
) -> Dict[str, Any]:
    output_rows: List[Dict[str, Any]] = []
    axis_counts: Dict[str, Counter[int]] = {axis_id: Counter() for axis_id in get_axis_ids()}
    statute_axis_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for idx, row in enumerate(rows):
        text = extract_text(row)
        statutes = normalize_labels(row.get("statutes") or row.get("labels") or row.get("label"))
        labels, sources = make_axis_labels(
            text,
            statutes,
            use_keyword_rules=use_keyword_rules,
            allow_unknown_axis=allow_unknown_axis,
        )
        for axis_id, value in labels.items():
            axis_counts[axis_id][int(value)] += 1
        for statute in statutes:
            for axis_id, value in labels.items():
                if value == 1:
                    statute_axis_counts[statute][axis_id] += 1
        output_rows.append(
            {
                "sample_index": idx,
                "source_id": row.get("id", idx),
                "text": text,
                "true_statutes": statutes,
                "axis_labels": labels,
                "axis_label_source": sources,
            }
        )

    write_jsonl(output_dir / f"{split_name}_axis_labels.jsonl", output_rows)
    return {
        "num_samples": len(output_rows),
        "axis_counts": {
            axis_id: {
                "positive": int(counter.get(1, 0)),
                "negative": int(counter.get(0, 0)),
                "unknown": int(counter.get(-1, 0)),
            }
            for axis_id, counter in axis_counts.items()
        },
        "statute_axis_distribution": {
            statute: dict(counter) for statute, counter in sorted(statute_axis_counts.items())
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build weak axis pseudo-labels for Full GTR.")
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    p.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    p.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    p.add_argument("--use-keyword-rules", action="store_true")
    p.add_argument("--allow-unknown-axis", action="store_true")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--force", action="store_true", help="Rebuild even when output files already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    expected = [args.output_dir / f"{split}_axis_labels.jsonl" for split in ("train", "valid", "test")]
    if not args.force and all(path.exists() for path in expected):
        print(f"Axis labels already exist in {args.output_dir}; use --force to rebuild.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_axis_schema(args.output_dir.parent / "axis_schema.json")
    stats: Dict[str, Any] = {
        "use_keyword_rules": bool(args.use_keyword_rules),
        "allow_unknown_axis": bool(args.allow_unknown_axis),
        "axis_ids": get_axis_ids(),
        "splits": {},
    }
    for split_name, path in (
        ("train", args.train_path),
        ("valid", args.valid_path),
        ("test", args.test_path),
    ):
        stats["splits"][split_name] = build_split(
            split_name,
            load_jsonl(path),
            args.output_dir,
            use_keyword_rules=args.use_keyword_rules,
            allow_unknown_axis=args.allow_unknown_axis,
        )
    write_json(args.output_dir / "axis_label_stats.json", stats)
    print(f"Saved axis labels to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
