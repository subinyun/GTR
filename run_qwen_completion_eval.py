#!/usr/bin/env python3
"""Run local HF completion-style statute prompts without chat templates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


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
        text = str(item).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def parse_generation(raw: str) -> tuple[list[str], bool]:
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    candidates: list[str] = []
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = None
        for match in re.findall(r"\{[\s\S]*?\}", text):
            try:
                parsed = json.loads(match)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and ("statutes" in parsed or "predicted_statutes" in parsed):
                obj = parsed
                break
    if isinstance(obj, dict):
        candidates = normalize_statutes(obj.get("statutes", obj.get("predicted_statutes")))
        return candidates, True
    return [], False


def resolve_candidate_numbers(labels: list[str], candidates: list[str]) -> list[str]:
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


def key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (int(row["sample_index"]), str(row["condition"]), int(row.get("k", 0)))


def done_keys(path: Path) -> set[tuple[int, str, int]]:
    if not path.exists():
        return set()
    out: set[tuple[int, str, int]] = set()
    for row in load_jsonl(path):
        try:
            out.add(key(row))
        except Exception:
            continue
    return out


def load_model(model_name: str, device: str, adapter_path: Path | None = None) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto" if device == "auto" else (torch.float16 if device.startswith("cuda") else torch.float32),
    }
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    if device != "auto":
        model.to(device)
    model.eval()
    return tokenizer, model


def generate_completion(
    tokenizer: Any,
    model: Any,
    prompt: str,
    *,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
    use_chat_template: bool,
    system_message: str,
) -> tuple[str, int, int]:
    import torch

    if use_chat_template:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    target_device = next(model.parameters()).device if device == "auto" else torch.device(device)
    enc = enc.to(target_device)
    prompt_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        output = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][prompt_len:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return raw, prompt_len, int(generated.shape[0])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_jsonl(args.prompt_path)
    if args.condition != "all":
        prompts = [row for row in prompts if str(row.get("condition")) == args.condition]
    if args.max_calls > 0:
        prompts = prompts[: args.max_calls]
    result_path = args.output_dir / args.results_filename
    completed = done_keys(result_path)
    tokenizer, model = load_model(args.model, args.device, args.adapter_path)
    written = 0
    total = len(prompts)
    started = __import__("time").time()
    print(f"Loaded {total} prompts; {len(completed)} already complete; {total - len(completed)} pending.", flush=True)
    for ordinal, row in enumerate(prompts, 1):
        if key(row) in completed:
            print(
                f"[{ordinal}/{total}] skip sample_index={row['sample_index']} condition={row['condition']}",
                flush=True,
            )
            continue
        prompt = str(row["prompt"])
        print(
            f"[{ordinal}/{total}] generating sample_index={row['sample_index']} condition={row['condition']}",
            flush=True,
        )
        try:
            raw, prompt_tokens, generated_tokens = generate_completion(
                tokenizer,
                model,
                prompt,
                device=args.device,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                use_chat_template=args.use_chat_template,
                system_message=args.system_message,
            )
            pred, parse_ok = parse_generation(raw)
            pred = resolve_candidate_numbers(pred, [str(x) for x in row.get("candidates", [])])
            error = None
        except Exception as exc:
            raw = f"[ERROR] {type(exc).__name__}: {exc}"
            prompt_tokens = generated_tokens = 0
            pred, parse_ok = [], False
            error = raw
            error_type = type(exc).__name__
        else:
            error_type = None
        candidate_statutes = [str(x) for x in row.get("candidates", row.get("candidate_statutes", []))]
        append_jsonl(
            result_path,
            [
                {
                    "sample_id": row.get("sample_id", row.get("sample_index")),
                    "sample_index": int(row["sample_index"]),
                    "condition": str(row["condition"]),
                    "k": int(row.get("k", 0)),
                    "model": args.model,
                    "adapter_path": str(args.adapter_path) if args.adapter_path else None,
                    "prompt": prompt,
                    "prompt_token_count": prompt_tokens,
                    "generated_token_count": generated_tokens,
                    "raw_generation": raw,
                    "parsed_prediction": pred,
                    "predicted_statutes": pred,
                    "gold_statutes": normalize_statutes(row.get("gold_statutes", row.get("true_statutes"))),
                    "true_statutes": normalize_statutes(row.get("true_statutes", row.get("gold_statutes"))),
                    "candidates": candidate_statutes,
                    "candidate_statutes": candidate_statutes,
                    "parse_ok": parse_ok,
                    "error": error,
                    "error_type": error_type if error_type else ("parse_failed" if not parse_ok else None),
                }
            ],
        )
        completed.add(key(row))
        written += 1
        elapsed = max(__import__("time").time() - started, 1e-6)
        rate = ordinal / elapsed
        eta = (total - ordinal) / rate if rate > 0 else 0.0
        print(
            f"[{ordinal}/{total}] wrote parse_ok={parse_ok} predicted={len(pred)} rate={rate:.2f}/s eta={eta:.1f}s",
            flush=True,
        )
    return {"results": str(result_path), "written": written, "total_requested": len(prompts)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--results-filename", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter-path", type=Path, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--condition", default="all")
    p.add_argument("--max-calls", type=int, default=0)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--system-message", default="You are a legal assistant.")
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
