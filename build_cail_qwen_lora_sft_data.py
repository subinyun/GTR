#!/usr/bin/env python3
"""Build CAIL2018 Qwen LoRA/SFT data from full GTR v2 scores."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from build_cail_gtr_v2_llm_prompts import candidate_only_prompt, full_fact_prompt, gtr_score_prompt
from lbox_raw_lowrank_eval import l2_normalize
from train_cail_hybrid_gtr_v2 import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    HybridGTRv2,
    read_jsonl,
)


CONDITIONS = ("full_fact", "raw_topk", "raw_pool_gtr_rerank", "raw_pool_gtr_rerank_score_prompt")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_statutes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def select_indices(n_rows: int, limit: int | None, *, seed: int, strategy: str) -> List[int]:
    if limit is None or limit <= 0 or limit >= n_rows:
        return list(range(n_rows))
    if strategy == "head":
        return list(range(limit))
    if strategy != "random":
        raise ValueError(f"Unknown sample strategy: {strategy}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), limit))


def target_json(labels: Sequence[str]) -> str:
    return json.dumps({"statutes": list(labels)}, ensure_ascii=False)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[HybridGTRv2, List[str], Mapping[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"]
    args = checkpoint["args"]
    raw_weight = state["raw_head.weight"]
    axis_key = "axis_head.3.weight" if "axis_head.3.weight" in state else "axis_head.0.weight"
    model = HybridGTRv2(
        dim=int(raw_weight.shape[1]),
        num_axes=int(state[axis_key].shape[0]),
        num_statutes=int(raw_weight.shape[0]),
        axis_hidden=int(args.get("axis_hidden", 256)),
        field_hidden=int(args.get("field_hidden", 0)),
        residual_rank=int(args.get("residual_rank", 32)),
        residual_scale=float(args.get("residual_scale", 0.02)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, [str(item) for item in checkpoint["statutes"]], args


@torch.no_grad()
def score_selected(
    model: HybridGTRv2,
    z_split: np.ndarray,
    selected_indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_chunks: List[np.ndarray] = []
    gtr_chunks: List[np.ndarray] = []
    started = time.time()
    total = len(selected_indices)
    for start in range(0, total, batch_size):
        batch_indices = selected_indices[start : start + batch_size]
        batch = torch.from_numpy(z_split[np.asarray(batch_indices, dtype=np.int64)].astype(np.float32)).to(device)
        out = model(batch)
        raw_chunks.append(model.raw_head(batch).cpu().numpy())
        gtr_chunks.append(out["logits"].cpu().numpy())
        done = min(start + len(batch_indices), total)
        elapsed = max(time.time() - started, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"[score {split_name} {done}/{total}] rate={rate:.1f}/s eta={eta:.1f}s", flush=True)
    return np.concatenate(raw_chunks, axis=0), np.concatenate(gtr_chunks, axis=0)


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int32)
    return ranks


def inject_gold(candidates: Sequence[int], gold_indices: Sequence[int], *, max_candidates: int) -> tuple[list[int], dict[str, Any]]:
    out = list(dict.fromkeys(int(idx) for idx in candidates))
    injected: list[int] = []
    removed: list[int] = []
    for idx in gold_indices:
        if int(idx) not in out:
            out.append(int(idx))
            injected.append(int(idx))
    gold_set = {int(idx) for idx in gold_indices}
    while len(out) > max_candidates:
        remove_pos = None
        for pos in range(len(out) - 1, -1, -1):
            if out[pos] not in gold_set:
                remove_pos = pos
                break
        if remove_pos is None:
            remove_pos = len(out) - 1
        removed.append(out.pop(remove_pos))
    return out, {"injected_gold_indices": injected, "removed_candidate_indices": removed}


def row_gold_indices(gold: Sequence[str], statute_to_idx: Mapping[str, int]) -> list[int]:
    return [statute_to_idx[label] for label in gold if label in statute_to_idx]


def make_rows_for_split(
    *,
    split_name: str,
    source_rows: Sequence[Mapping[str, Any]],
    selected_indices: Sequence[int],
    raw_scores: np.ndarray,
    gtr_scores: np.ndarray,
    statutes: Sequence[str],
    k: int,
    raw_pool_k: int,
    inject_train_gold: bool,
    progress_every: int,
) -> dict[str, list[dict[str, Any]]]:
    statute_to_idx = {label: idx for idx, label in enumerate(statutes)}
    by_condition: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    started = time.time()
    total = len(selected_indices)
    for local_idx, original_idx in enumerate(selected_indices):
        row = source_rows[original_idx]
        facts = str(row.get("facts") or row.get("text") or "").strip()
        gold = normalize_statutes(row.get("statutes") or row.get("label") or row.get("labels"))
        gold_indices = row_gold_indices(gold, statute_to_idx)
        sample_id = row.get("id", original_idx)

        raw_row = raw_scores[local_idx]
        gtr_row = gtr_scores[local_idx]
        raw_order = np.argsort(-raw_row)
        gtr_ranks = descending_ranks(gtr_row)
        raw_top = raw_order[:k].astype(int).tolist()
        raw_pool = raw_order[:raw_pool_k]
        reranked = raw_pool[np.argsort(-gtr_row[raw_pool], kind="mergesort")[:k]].astype(int).tolist()
        raw_injection = {"injected_gold_indices": [], "removed_candidate_indices": []}
        rerank_injection = {"injected_gold_indices": [], "removed_candidate_indices": []}
        if inject_train_gold:
            raw_top, raw_injection = inject_gold(raw_top, gold_indices, max_candidates=k)
            reranked, rerank_injection = inject_gold(reranked, gold_indices, max_candidates=k)

        raw_candidates = [statutes[idx] for idx in raw_top]
        reranked_candidates = [statutes[idx] for idx in reranked]
        reranked_with_scores = [(statutes[idx], int(gtr_ranks[idx]), float(gtr_row[idx])) for idx in reranked]
        raw_target = [label for label in gold if label in set(raw_candidates)]
        rerank_target = [label for label in gold if label in set(reranked_candidates)]

        base = {
            "split": split_name,
            "sample_index": int(original_idx),
            "sample_id": sample_id,
            "true_statutes": gold,
            "gold_statutes": gold,
            "source": "cail2018_full_gtr_v2",
        }
        by_condition["full_fact"].append(
            {
                **base,
                "condition": "full_fact",
                "k": 0,
                "raw_pool_k": None,
                "candidate_statutes": [],
                "candidates": [],
                "candidate_gold_statutes": gold,
                "prompt": full_fact_prompt(facts),
                "completion": target_json(gold),
            }
        )
        by_condition["raw_topk"].append(
            {
                **base,
                "condition": "raw_topk",
                "condition_alias": "raw_k",
                "k": int(k),
                "raw_pool_k": None,
                "candidate_statutes": raw_candidates,
                "candidates": raw_candidates,
                "candidate_gold_statutes": raw_target,
                "gold_injection": raw_injection,
                "prompt": candidate_only_prompt(facts, raw_candidates),
                "completion": target_json(raw_target),
            }
        )
        by_condition["raw_pool_gtr_rerank"].append(
            {
                **base,
                "condition": "raw_pool_gtr_rerank",
                "k": int(k),
                "raw_pool_k": int(raw_pool_k),
                "candidate_statutes": reranked_candidates,
                "candidates": reranked_candidates,
                "candidate_gold_statutes": rerank_target,
                "gold_injection": rerank_injection,
                "prompt": candidate_only_prompt(facts, reranked_candidates),
                "completion": target_json(rerank_target),
            }
        )
        by_condition["raw_pool_gtr_rerank_score_prompt"].append(
            {
                **base,
                "condition": "raw_pool_gtr_rerank_score_prompt",
                "k": int(k),
                "raw_pool_k": int(raw_pool_k),
                "candidate_statutes": reranked_candidates,
                "candidates": reranked_candidates,
                "candidate_gold_statutes": rerank_target,
                "gold_injection": rerank_injection,
                "gtr_scores": {statute: score for statute, _rank, score in reranked_with_scores},
                "gtr_ranks": {statute: rank for statute, rank, _score in reranked_with_scores},
                "prompt": gtr_score_prompt(facts, reranked_with_scores),
                "completion": target_json(rerank_target),
            }
        )

        done = local_idx + 1
        if done == total or done % progress_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0.0
            print(f"[rows {split_name} {done}/{total}] rate={rate:.1f}/s eta={eta:.1f}s", flush=True)
    return by_condition


def build(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, statutes, checkpoint_args = load_model(args.checkpoint, device)
    payload = np.load(args.embedding_cache)
    z_by_split = {
        "train": l2_normalize(np.asarray(payload["train"], dtype=np.float32)),
        "valid": l2_normalize(np.asarray(payload["valid"], dtype=np.float32)),
        "test": l2_normalize(np.asarray(payload["test"], dtype=np.float32)),
    }
    rows_by_split = {
        "train": read_jsonl(args.train_path),
        "valid": read_jsonl(args.valid_path),
        "test": read_jsonl(args.test_path),
    }
    limits = {"train": args.max_train_rows, "valid": args.max_valid_rows, "test": args.max_test_rows}
    seeds = {"train": args.seed, "valid": args.seed + 1, "test": args.seed + 2}

    manifest: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "embedding_cache": str(args.embedding_cache),
        "k": int(args.k),
        "raw_pool_k": int(args.raw_pool_k),
        "sample_strategy": args.sample_strategy,
        "conditions": {condition: {} for condition in CONDITIONS},
        "checkpoint_args_subset": {
            "model_name": checkpoint_args.get("model_name"),
            "seed": checkpoint_args.get("seed"),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        rows = rows_by_split[split_name]
        z_split = z_by_split[split_name]
        if z_split.shape[0] != len(rows):
            raise ValueError(f"{split_name} embedding/row mismatch: {z_split.shape[0]} vs {len(rows)}")
        selected = select_indices(len(rows), limits[split_name], seed=seeds[split_name], strategy=args.sample_strategy)
        print(f"Selected {len(selected)}/{len(rows)} rows for split={split_name}", flush=True)
        raw_scores, gtr_scores = score_selected(
            model,
            z_split,
            selected,
            batch_size=args.batch_size,
            device=device,
            split_name=split_name,
        )
        split_rows = make_rows_for_split(
            split_name=split_name,
            source_rows=rows,
            selected_indices=selected,
            raw_scores=raw_scores,
            gtr_scores=gtr_scores,
            statutes=statutes,
            k=args.k,
            raw_pool_k=args.raw_pool_k,
            inject_train_gold=split_name == "train",
            progress_every=args.progress_every,
        )
        for condition, condition_rows in split_rows.items():
            path = args.output_dir / f"{condition}_{split_name}.jsonl"
            write_jsonl(path, condition_rows)
            manifest["conditions"][condition][split_name] = {
                "path": str(path),
                "n_rows": len(condition_rows),
                "gold_injected_count": int(sum(len(row.get("gold_injection", {}).get("injected_gold_indices", [])) for row in condition_rows)),
                "empty_completion_count": int(sum(1 for row in condition_rows if not row.get("candidate_gold_statutes") and condition != "full_fact")),
            }
    write_json(args.output_dir / "manifest_cail_qwen_lora_sft_data.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt"))
    parser.add_argument("--embedding-cache", type=Path, default=Path("output/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data"))
    parser.add_argument("--max-train-rows", type=int, default=0, help="<=0 means full train split.")
    parser.add_argument("--max-valid-rows", type=int, default=0, help="<=0 means full valid split.")
    parser.add_argument("--max-test-rows", type=int, default=276, help="<=0 means full test split.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--raw-pool-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-strategy", choices=("random", "head"), default="random")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
