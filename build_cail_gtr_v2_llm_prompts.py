#!/usr/bin/env python3
"""Build descriptor-free CAIL2018 LLM prompts from a full GTR v2 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch

from lbox_raw_lowrank_eval import l2_normalize
from train_cail_hybrid_gtr_v2 import DEFAULT_TEST_PATH, HybridGTRv2, read_jsonl


CONDITIONS = (
    "full_fact",
    "raw_topk",
    "raw_pool_gtr_rerank",
    "raw_pool_gtr_rerank_score_prompt",
)


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


def select_indices(n_rows: int, max_rows: int, *, seed: int, strategy: str) -> List[int]:
    if max_rows <= 0 or max_rows >= n_rows:
        return list(range(n_rows))
    if strategy == "head":
        return list(range(max_rows))
    if strategy != "random":
        raise ValueError(f"Unknown sample strategy: {strategy}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(n_rows), max_rows))


def full_fact_prompt(facts: str) -> str:
    return f"""Task: Determine which Chinese criminal statutes apply to the case facts.

Return only a JSON object with the key "statutes".
Do not explain.

Return statutes exactly in this format: 中华人民共和国刑法 第234条
Select all and only applicable statutes.

Case:
{facts}

Output format:
{{"statutes": ["..."]}}

Answer:"""


def candidate_only_prompt(facts: str, candidates: Sequence[str]) -> str:
    candidate_block = "\n".join(f"{rank}. {statute}" for rank, statute in enumerate(candidates, 1))
    return f"""Task: Determine which Chinese criminal statutes apply to the case facts.

Return only a JSON object with the key "statutes".
Do not explain.

Select all and only applicable statutes from the candidate list.

Case:
{facts}

Candidate statutes:

{candidate_block}

Output format:
{{"statutes": ["..."]}}

Answer:"""


def gtr_score_prompt(facts: str, candidates: Sequence[tuple[str, int, float]]) -> str:
    lines = [
        f"{rank}. {statute}\n   GTR rank: {gtr_rank}\n   GTR score: {score:.6f}"
        for rank, (statute, gtr_rank, score) in enumerate(candidates, 1)
    ]
    return f"""Task: Determine which Chinese criminal statutes apply to the case facts.

Return only a JSON object with the key "statutes".
Do not explain.

Select all and only applicable statutes from the candidate list.

Case:
{facts}

Candidate statutes:

{chr(10).join(lines)}

Rules:
- GTR rank and score indicate legal relevance estimated by the GTR v2 model.
- Use them as supporting evidence only.
- Final decisions must be based on the case facts.
- Select only statutes whose legal elements are satisfied by the facts.

Output format:
{{"statutes": ["..."]}}

Answer:"""


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
    return model, [str(item) for item in checkpoint["statutes"]], args


@torch.no_grad()
def score_selected(
    model: HybridGTRv2,
    z_test: np.ndarray,
    selected_indices: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    raw_chunks: List[np.ndarray] = []
    gtr_chunks: List[np.ndarray] = []
    started = time.time()
    total = len(selected_indices)
    for start in range(0, total, batch_size):
        batch_indices = selected_indices[start : start + batch_size]
        batch = torch.from_numpy(z_test[np.asarray(batch_indices, dtype=np.int64)].astype(np.float32)).to(device)
        out = model(batch)
        raw_chunks.append(model.raw_head(batch).cpu().numpy())
        gtr_chunks.append(out["logits"].cpu().numpy())
        done = min(start + len(batch_indices), total)
        elapsed = max(time.time() - started, 1e-6)
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0.0
        print(f"[score {done}/{total}] rate={rate:.1f}/s eta={eta:.1f}s", flush=True)
    return np.concatenate(raw_chunks, axis=0), np.concatenate(gtr_chunks, axis=0)


def descending_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int32)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int32)
    return ranks


def build_rows(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    test_rows = read_jsonl(args.test_path)
    selected_indices = select_indices(len(test_rows), args.max_rows, seed=args.seed, strategy=args.sample_strategy)
    print(
        f"Loaded {len(test_rows)} test rows; selected {len(selected_indices)} rows "
        f"strategy={args.sample_strategy} seed={args.seed}.",
        flush=True,
    )

    model, statutes, checkpoint_args = load_model(args.checkpoint, device)
    payload = np.load(args.embedding_cache)
    z_test = l2_normalize(np.asarray(payload["test"], dtype=np.float32))
    if z_test.shape[0] != len(test_rows):
        raise ValueError(f"Embedding/test size mismatch: {z_test.shape[0]} vs {len(test_rows)}")

    raw_scores, gtr_scores = score_selected(
        model,
        z_test,
        selected_indices,
        batch_size=args.batch_size,
        device=device,
    )

    prompt_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []
    started = time.time()
    total = len(selected_indices)
    for local_idx, original_idx in enumerate(selected_indices):
        row = test_rows[original_idx]
        facts = str(row.get("facts") or row.get("text") or "").strip()
        true_statutes = normalize_statutes(row.get("statutes") or row.get("label"))
        sample_id = row.get("id", original_idx)

        raw_row = raw_scores[local_idx]
        gtr_row = gtr_scores[local_idx]
        raw_order = np.argsort(-raw_row)
        gtr_ranks = descending_ranks(gtr_row)

        raw_top = raw_order[: args.k]
        raw_pool = raw_order[: args.raw_pool_k]
        reranked = raw_pool[np.argsort(-gtr_row[raw_pool], kind="mergesort")[: args.k]]

        raw_candidates = [(statutes[int(idx)], float(raw_row[int(idx)])) for idx in raw_top]
        reranked_statutes = [statutes[int(idx)] for idx in reranked]
        reranked_with_scores = [
            (statutes[int(idx)], int(gtr_ranks[int(idx)]), float(gtr_row[int(idx)]))
            for idx in reranked
        ]

        base = {
            "split": "test",
            "sample_index": int(original_idx),
            "sample_id": sample_id,
            "true_statutes": true_statutes,
            "source_checkpoint": str(args.checkpoint),
            "sample_strategy": args.sample_strategy,
            "sample_seed": int(args.seed),
        }
        prompt_rows.extend(
            [
                {
                    **base,
                    "condition": "full_fact",
                    "k": 0,
                    "raw_pool_k": None,
                    "candidate_statutes": [],
                    "candidates": [],
                    "prompt": full_fact_prompt(facts),
                },
                {
                    **base,
                    "condition": "raw_topk",
                    "condition_alias": "raw_k",
                    "k": int(args.k),
                    "raw_pool_k": None,
                    "candidate_statutes": [statute for statute, _score in raw_candidates],
                    "candidates": [statute for statute, _score in raw_candidates],
                    "raw_scores": {statute: score for statute, score in raw_candidates},
                    "prompt": candidate_only_prompt(facts, [statute for statute, _score in raw_candidates]),
                },
                {
                    **base,
                    "condition": "raw_pool_gtr_rerank",
                    "k": int(args.k),
                    "raw_pool_k": int(args.raw_pool_k),
                    "candidate_statutes": reranked_statutes,
                    "candidates": reranked_statutes,
                    "prompt": candidate_only_prompt(facts, reranked_statutes),
                },
                {
                    **base,
                    "condition": "raw_pool_gtr_rerank_score_prompt",
                    "k": int(args.k),
                    "raw_pool_k": int(args.raw_pool_k),
                    "candidate_statutes": reranked_statutes,
                    "candidates": reranked_statutes,
                    "gtr_scores": {statute: score for statute, _rank, score in reranked_with_scores},
                    "gtr_ranks": {statute: rank for statute, rank, _score in reranked_with_scores},
                    "prompt": gtr_score_prompt(facts, reranked_with_scores),
                },
            ]
        )
        preview_rows.append(
            {
                "sample_index": int(original_idx),
                "sample_id": sample_id,
                "true_statutes": true_statutes,
                "raw_topk": [statute for statute, _score in raw_candidates],
                "raw_pool_gtr_rerank": reranked_statutes,
            }
        )
        done = local_idx + 1
        if done == total or done % args.progress_every == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0.0
            print(f"[prompts {done}/{total}] rows={len(prompt_rows)} rate={rate:.1f}/s eta={eta:.1f}s", flush=True)

    counts = {condition: 0 for condition in CONDITIONS}
    for row in prompt_rows:
        counts[str(row["condition"])] += 1
    summary = {
        "prompt_rows": len(prompt_rows),
        "num_cases": len(selected_indices),
        "conditions": list(CONDITIONS),
        "condition_counts": counts,
        "k": int(args.k),
        "raw_pool_k": int(args.raw_pool_k),
        "sample_strategy": args.sample_strategy,
        "sample_seed": int(args.seed),
        "selected_indices": selected_indices,
        "source_checkpoint": str(args.checkpoint),
        "source_embedding_cache": str(args.embedding_cache),
        "checkpoint_args_subset": {
            "model_name": checkpoint_args.get("model_name"),
            "max_train_samples": checkpoint_args.get("max_train_samples"),
            "max_valid_samples": checkpoint_args.get("max_valid_samples"),
            "max_test_samples": checkpoint_args.get("max_test_samples"),
            "sample_strategy": checkpoint_args.get("sample_strategy"),
            "seed": checkpoint_args.get("seed"),
        },
        "preview": preview_rows[:10],
    }
    return {"prompt_rows": prompt_rows, "summary": summary, "preview_rows": preview_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--checkpoint", type=Path, default=Path("output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt"))
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("output/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/cail2018_gtr_v2_only/full/llm_prompts"))
    parser.add_argument("--max-rows", type=int, default=276, help="Cases to sample; <=0 means all test cases.")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--raw-pool-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-strategy", choices=("random", "head"), default="random")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / f"prompts_{args.max_rows if args.max_rows > 0 else 'all'}.jsonl"
    summary_path = args.output_dir / f"prompts_{args.max_rows if args.max_rows > 0 else 'all'}_summary.json"
    preview_path = args.output_dir / f"prompts_{args.max_rows if args.max_rows > 0 else 'all'}_preview.jsonl"
    write_jsonl(combined_path, output["prompt_rows"])
    write_json(summary_path, output["summary"])
    write_jsonl(preview_path, output["preview_rows"])

    # GPT and Gemini runners both consume the same descriptor-free prompt rows.
    for provider in ("gpt", "gemini"):
        provider_dir = args.output_dir / provider
        write_jsonl(provider_dir / combined_path.name, output["prompt_rows"])
        write_json(provider_dir / summary_path.name, output["summary"])

    print(f"Saved combined prompts: {combined_path}", flush=True)
    print(f"Saved GPT prompts: {args.output_dir / 'gpt' / combined_path.name}", flush=True)
    print(f"Saved Gemini prompts: {args.output_dir / 'gemini' / combined_path.name}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
