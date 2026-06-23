#!/usr/bin/env python3
"""Run OpenAI JSON prompts exported by routing experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def result_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (int(row["sample_index"]), str(row["condition"]), int(row.get("k", 0)))


def load_done(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    return {result_key(row) for row in load_jsonl(path)}


def normalize_statutes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def parse_statutes(raw_response: str) -> tuple[list[str], bool]:
    raw_response = (raw_response or "").strip()
    if raw_response.startswith("[API_ERROR]"):
        return [], False
    parsed: Any = None
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw_response)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if not isinstance(parsed, dict):
        return [], False
    if "statutes" in parsed:
        return normalize_statutes(parsed.get("statutes")), True
    # Backward-compatible parsing only; prompts still require the "statutes" key.
    if "predicted_statutes" in parsed:
        return normalize_statutes(parsed.get("predicted_statutes")), True
    return [], False


def call_openai(
    *,
    client: Any,
    model: str,
    prompt: str,
    max_retries: int,
    retry_sleep: float,
    max_tokens: int,
) -> tuple[list[str], bool, str]:
    last_raw = ""
    for attempt in range(max_retries):
        try:
            request = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            }
            try:
                response = client.chat.completions.create(**request, max_completion_tokens=max_tokens)
            except TypeError:
                response = client.chat.completions.create(**request, max_tokens=max_tokens)
            last_raw = response.choices[0].message.content or ""
            statutes, ok = parse_statutes(last_raw)
            if ok:
                return statutes, ok, last_raw
        except Exception as exc:  # pragma: no cover - external API
            last_raw = f"[API_ERROR] {type(exc).__name__}: {exc}"
        if attempt + 1 < max_retries:
            time.sleep(retry_sleep * (2**attempt))
    statutes, ok = parse_statutes(last_raw)
    return statutes, ok, last_raw


def selected_rows(rows: Sequence[Mapping[str, Any]], conditions: set[str] | None, max_rows: int | None) -> list[Mapping[str, Any]]:
    out = [row for row in rows if conditions is None or str(row.get("condition")) in conditions]
    return out[:max_rows] if max_rows is not None else out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=Path("artifacts/hybrid_gtr_v2_improved/gpt54_gtr_evidence_comparison/prompts_60.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/hybrid_gtr_v2_improved/gpt54_gtr_evidence_comparison/results_gpt54_60.jsonl"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum new prompt rows to call after resume skips.")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(f"ERROR: Set {args.api_key_env}.", file=sys.stderr)
        sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: Install the openai package.", file=sys.stderr)
        sys.exit(1)

    prompts = selected_rows(load_jsonl(args.prompts), set(args.conditions) if args.conditions else None, args.max_rows)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.results)
    pending = [row for row in prompts if result_key(row) not in done]
    total = len(prompts)
    print(
        f"Loaded {total} prompt rows; {len(done)} already complete; {len(pending)} pending.",
        flush=True,
    )
    client = OpenAI(api_key=api_key)
    new_calls = 0
    with args.results.open("a", encoding="utf-8") as handle:
        for ordinal, row in enumerate(prompts, 1):
            key = result_key(row)
            if key in done:
                print(
                    f"[{ordinal}/{total}] skip existing sample_index={row['sample_index']} "
                    f"condition={row['condition']} k={row.get('k', 0)}",
                    flush=True,
                )
                continue
            print(
                f"[{ordinal}/{total}] calling {args.model} sample_index={row['sample_index']} "
                f"condition={row['condition']} k={row.get('k', 0)}",
                flush=True,
            )
            statutes, parse_ok, raw_response = call_openai(
                client=client,
                model=args.model,
                prompt=str(row["prompt"]),
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                max_tokens=args.max_tokens,
            )
            out = {
                "split": row.get("split", "test"),
                "sample_index": int(row["sample_index"]),
                "condition": str(row["condition"]),
                "k": int(row.get("k", 0)),
                "raw_pool_k": row.get("raw_pool_k"),
                "true_statutes": row.get("true_statutes", []),
                "candidate_statutes": row.get("candidate_statutes", row.get("candidates", [])),
                "statutes": statutes,
                "predicted_statutes": statutes,
                "parse_ok": parse_ok,
                "model": args.model,
                "raw_response": raw_response[:50000],
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(key)
            new_calls += 1
            print(
                f"[{ordinal}/{total}] wrote parse_ok={parse_ok} predicted={len(statutes)} "
                f"-> {args.results}",
                flush=True,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
    print(f"Appended {new_calls} rows -> {args.results}")


if __name__ == "__main__":
    main()
