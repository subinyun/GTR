#!/usr/bin/env python3
"""Train GTR v2-only on CAIL2018 statute classification.

This is the CAIL2018 counterpart of train_hybrid_gtr_v2.py. It keeps the same
non-LLM model family but swaps in Chinese statute labels and CAIL axis
supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from build_cail_axis_labels import make_axis_labels
from cail_gtr_axis_schema import get_axis_ids
from lbox_raw_lowrank_eval import (
    DEFAULT_MODEL_NAME,
    exact_match_score,
    extract_ovr_linear_probe_weights,
    fit_raw_linear_probe,
    l2_normalize,
    normalize_labels,
)
from train_hybrid_gtr_v2 import (
    HybridGTRv2,
    json_ready,
    masked_axis_bce,
    regularization_loss,
    set_stage_trainability,
    trainable_parameters,
    write_json,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"Label not .* is present in all training examples\.", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "final_all_data/cail2018_statute_classification"
DEFAULT_TRAIN_PATH = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_VALID_PATH = DEFAULT_DATA_DIR / "valid.jsonl"
DEFAULT_TEST_PATH = DEFAULT_DATA_DIR / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/cail2018_gtr_v2_only"


@dataclass
class Splits:
    train_z: np.ndarray
    valid_z: np.ndarray
    test_z: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    axis_train: np.ndarray
    axis_valid: np.ndarray
    axis_test: np.ndarray
    statutes: List[str]
    train_rows: List[Dict[str, Any]]
    valid_rows: List[Dict[str, Any]]
    test_rows: List[Dict[str, Any]]


class GTRDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray, axis: np.ndarray) -> None:
        self.z = torch.from_numpy(z.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.axis = torch.from_numpy(axis.astype(np.float32))

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z[idx], self.y[idx], self.axis[idx]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def sample_rows(rows: Sequence[Dict[str, Any]], limit: int | None, *, seed: int, strategy: str) -> List[Dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return list(rows)
    if strategy == "head":
        return list(rows[:limit])
    if strategy != "random":
        raise ValueError(f"Unknown sample strategy: {strategy}")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(rows)), limit))
    return [dict(rows[i]) for i in indices]


def rows_to_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [str(row.get("facts") or row.get("text") or "").strip() for row in rows],
            "label": [normalize_labels(row.get("statutes") or row.get("labels") or row.get("label")) for row in rows],
            "casename": [str(row.get("casename", "")).strip() for row in rows],
        }
    )


def collect_statutes_from_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    labels = {label for row in rows for label in normalize_labels(row.get("statutes") or row.get("labels") or row.get("label"))}
    return sorted(labels)


def rows_to_axis_labels(rows: Sequence[Mapping[str, Any]], *, use_keyword_rules: bool, allow_unknown_axis: bool) -> np.ndarray:
    axis_ids = get_axis_ids()
    labels = np.full((len(rows), len(axis_ids)), -1 if allow_unknown_axis else 0, dtype=np.int32)
    for i, row in enumerate(rows):
        text = str(row.get("facts") or row.get("text") or "").strip()
        statutes = normalize_labels(row.get("statutes") or row.get("labels") or row.get("label"))
        axis_labels, _sources = make_axis_labels(
            text,
            statutes,
            use_keyword_rules=use_keyword_rules,
            allow_unknown_axis=allow_unknown_axis,
        )
        labels[i] = np.asarray([int(axis_labels[axis_id]) for axis_id in axis_ids], dtype=np.int32)
    return labels


def embedding_cache_path(args: argparse.Namespace) -> Path:
    if args.embed_cache is not None:
        return args.embed_cache
    tag = (
        f"{args.sample_strategy}_"
        f"tr{args.max_train_samples or 'full'}_"
        f"va{args.max_valid_samples or 'full'}_"
        f"te{args.max_test_samples or 'full'}_"
        f"seed{args.seed}"
    )
    return args.output_dir / "cache" / f"bge_m3_cail_embeddings_{tag}.npz"


def encode_cases(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
    *,
    device: str | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = SentenceTransformer(model_name, device=device)
    train_emb = model.encode(
        train_df["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    valid_emb = model.encode(
        valid_df["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    test_emb = model.encode(
        test_df["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return (
        l2_normalize(np.asarray(train_emb, dtype=np.float32)),
        l2_normalize(np.asarray(valid_emb, dtype=np.float32)),
        l2_normalize(np.asarray(test_emb, dtype=np.float32)),
    )


def load_or_encode(args: argparse.Namespace, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = embedding_cache_path(args)
    if cache_path.exists() and not args.force_reencode:
        payload = np.load(cache_path)
        meta = json.loads(str(payload["meta"])) if "meta" in payload else {}
        expected = {
            "n_train": int(len(train_df)),
            "n_valid": int(len(valid_df)),
            "n_test": int(len(test_df)),
            "model_name": args.model_name,
            "encoder_device": args.encoder_device,
        }
        if all(meta.get(k) == v for k, v in expected.items()):
            return (
                l2_normalize(np.asarray(payload["train"], dtype=np.float32)),
                l2_normalize(np.asarray(payload["valid"], dtype=np.float32)),
                l2_normalize(np.asarray(payload["test"], dtype=np.float32)),
            )
    train_z, valid_z, test_z = encode_cases(
        args.model_name,
        train_df,
        valid_df,
        test_df,
        args.batch_size,
        device=args.encoder_device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": args.model_name,
        "encoder_device": args.encoder_device,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "sample_strategy": args.sample_strategy,
        "seed": int(args.seed),
    }
    np.savez_compressed(cache_path, train=train_z, valid=valid_z, test=test_z, meta=json.dumps(meta, ensure_ascii=False))
    return l2_normalize(train_z), l2_normalize(valid_z), l2_normalize(test_z)


def load_splits(args: argparse.Namespace) -> Splits:
    full_train_rows = read_jsonl(args.train_path)
    full_valid_rows = read_jsonl(args.valid_path)
    full_test_rows = read_jsonl(args.test_path)
    statutes = sorted(set(collect_statutes_from_rows(full_train_rows)) | set(collect_statutes_from_rows(full_valid_rows)) | set(collect_statutes_from_rows(full_test_rows)))

    train_rows = sample_rows(full_train_rows, args.max_train_samples, seed=args.seed, strategy=args.sample_strategy)
    valid_rows = sample_rows(full_valid_rows, args.max_valid_samples, seed=args.seed + 1, strategy=args.sample_strategy)
    test_rows = sample_rows(full_test_rows, args.max_test_samples, seed=args.seed + 2, strategy=args.sample_strategy)

    train_df = rows_to_frame(train_rows)
    valid_df = rows_to_frame(valid_rows)
    test_df = rows_to_frame(test_rows)

    mlb = MultiLabelBinarizer(classes=statutes)
    y_train = mlb.fit_transform(train_df["label"]).astype(np.float32)
    y_valid = mlb.transform(valid_df["label"]).astype(np.float32)
    y_test = mlb.transform(test_df["label"]).astype(np.float32)
    train_z, valid_z, test_z = load_or_encode(args, train_df, valid_df, test_df)

    return Splits(
        train_z=train_z.astype(np.float32),
        valid_z=valid_z.astype(np.float32),
        test_z=test_z.astype(np.float32),
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        axis_train=rows_to_axis_labels(train_rows, use_keyword_rules=args.use_keyword_rules, allow_unknown_axis=args.allow_unknown_axis).astype(np.float32),
        axis_valid=rows_to_axis_labels(valid_rows, use_keyword_rules=args.use_keyword_rules, allow_unknown_axis=args.allow_unknown_axis).astype(np.float32),
        axis_test=rows_to_axis_labels(test_rows, use_keyword_rules=args.use_keyword_rules, allow_unknown_axis=args.allow_unknown_axis).astype(np.float32),
        statutes=statutes,
        train_rows=train_rows,
        valid_rows=valid_rows,
        test_rows=test_rows,
    )


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    y_pred = (logits >= threshold).astype(np.int32)
    return metrics_from_predictions(y_true, y_pred)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": exact_match_score(y_true, y_pred),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "gold_label_cardinality": float(np.mean(y_true.sum(axis=1))) if y_true.size else 0.0,
        "pred_label_cardinality": float(np.mean(y_pred.sum(axis=1))) if y_pred.size else 0.0,
    }


def tune_threshold(valid_logits: np.ndarray, y_valid: np.ndarray) -> Tuple[float, Dict[str, float]]:
    candidates = np.unique(np.concatenate([np.array([0.0], dtype=np.float32), np.quantile(valid_logits.reshape(-1), np.linspace(0.02, 0.98, 49))]))
    best_threshold = 0.0
    best_metrics: Dict[str, float] = {}
    best_f1 = -1.0
    for threshold in candidates:
        metrics = metrics_from_logits(valid_logits, y_valid, threshold=float(threshold))
        if metrics["micro_f1"] > best_f1:
            best_f1 = metrics["micro_f1"]
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


@torch.no_grad()
def predict_components(model: HybridGTRv2, z: np.ndarray, batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    tensor = torch.from_numpy(z.astype(np.float32))
    chunks: Dict[str, List[np.ndarray]] = {
        "full_hybrid": [],
        "raw_head_on_z": [],
        "raw_plus_field": [],
        "raw_head_on_z_prime": [],
        "field_logits": [],
        "axis_logits": [],
        "q": [],
        "residual_norm": [],
    }
    for start in range(0, tensor.shape[0], batch_size):
        batch = tensor[start : start + batch_size].to(device)
        out = model(batch)
        raw_z = model.raw_head(batch)
        chunks["full_hybrid"].append(out["logits"].cpu().numpy())
        chunks["raw_head_on_z"].append(raw_z.cpu().numpy())
        chunks["raw_plus_field"].append((raw_z + out["field_logits"]).cpu().numpy())
        chunks["raw_head_on_z_prime"].append(out["raw_logits"].cpu().numpy())
        chunks["field_logits"].append(out["field_logits"].cpu().numpy())
        chunks["axis_logits"].append(out["axis_logits"].cpu().numpy())
        chunks["q"].append(out["q"].cpu().numpy())
        chunks["residual_norm"].append(torch.linalg.norm(out["residual"], dim=1).cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def axis_metrics(axis_logits: np.ndarray, axis_labels: np.ndarray) -> Dict[str, Any]:
    rows = []
    probs = 1.0 / (1.0 + np.exp(-np.clip(axis_logits, -50.0, 50.0)))
    for j, axis_id in enumerate(get_axis_ids()):
        mask = axis_labels[:, j] >= 0
        y = axis_labels[mask, j]
        p = probs[mask, j]
        if len(np.unique(y)) < 2:
            rows.append({"axis_id": axis_id, "auc": None, "ap": None, "n_labeled": int(mask.sum())})
            continue
        rows.append(
            {
                "axis_id": axis_id,
                "auc": float(roc_auc_score(y, p)),
                "ap": float(average_precision_score(y, p)),
                "n_labeled": int(mask.sum()),
                "positive": int(np.sum(y == 1)),
                "negative": int(np.sum(y == 0)),
            }
        )
    aucs = [r["auc"] for r in rows if r["auc"] is not None]
    return {"mean_auc": float(np.mean(aucs)) if aucs else None, "per_axis": rows}


def retrieval_metrics(logits: np.ndarray, y_true: np.ndarray, statutes: Sequence[str], *, k: int) -> Dict[str, Any]:
    rows = []
    for j, statute in enumerate(statutes):
        positives = np.where(y_true[:, j] > 0)[0]
        if positives.size == 0:
            continue
        order = np.argsort(-logits[:, j])
        top = order[: min(k, len(order))]
        rel = y_true[top, j].astype(np.float32)
        precision_at_k = float(np.mean(rel)) if rel.size else 0.0
        recall_at_k = float(np.sum(rel) / positives.size)
        gains = rel / np.log2(np.arange(2, rel.size + 2))
        dcg = float(np.sum(gains))
        ideal_rel = np.ones(min(positives.size, rel.size), dtype=np.float32)
        ideal_dcg = float(np.sum(ideal_rel / np.log2(np.arange(2, ideal_rel.size + 2)))) if ideal_rel.size else 0.0
        rows.append(
            {
                "statute": statute,
                "positives": int(positives.size),
                "precision_at_k": precision_at_k,
                "recall_at_k": recall_at_k,
                "ndcg_at_k": float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0,
            }
        )
    return {
        "k": int(k),
        "macro_precision_at_k": float(np.mean([r["precision_at_k"] for r in rows])) if rows else None,
        "macro_recall_at_k": float(np.mean([r["recall_at_k"] for r in rows])) if rows else None,
        "macro_ndcg_at_k": float(np.mean([r["ndcg_at_k"] for r in rows])) if rows else None,
        "per_statute": rows,
    }


def warm_start_raw_head(model: HybridGTRv2, data: Splits, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.warm_start_raw_head:
        return {"enabled": False}
    cache_dir = args.raw_probe_cache_dir or (args.output_dir / "raw_probe_cache")
    metrics, _pred_test, clf = fit_raw_linear_probe(
        data.train_z,
        data.valid_z,
        data.test_z,
        data.y_train,
        data.y_valid,
        data.y_test,
        model_dir=cache_dir,
    )
    w_raw, b_raw = extract_ovr_linear_probe_weights(clf)
    with torch.no_grad():
        model.raw_head.weight.copy_(torch.from_numpy(w_raw.T.astype(np.float32)))
        model.raw_head.bias.copy_(torch.from_numpy(b_raw.astype(np.float32)))
    if args.freeze_raw_head:
        for param in model.raw_head.parameters():
            param.requires_grad = False
    return {
        "enabled": True,
        "frozen": bool(args.freeze_raw_head),
        "raw_probe_metrics": metrics,
        "weight_shape": list(w_raw.shape),
    }


def evaluate_component_family(
    valid_components: Mapping[str, np.ndarray],
    test_components: Mapping[str, np.ndarray],
    y_valid: np.ndarray,
    y_test: np.ndarray,
    *,
    component_names: Sequence[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for name in component_names:
        threshold, valid_metrics = tune_threshold(valid_components[name], y_valid)
        test_metrics = metrics_from_logits(test_components[name], y_test, threshold=threshold)
        output[name] = {
            "threshold": threshold,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }
    return output


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_splits(args)

    if data.axis_train.shape[1] != 20:
        raise ValueError(f"Expected CAIL axis_dim=20, got {data.axis_train.shape[1]}")
    if len(data.statutes) != 183:
        raise ValueError(f"Expected CAIL num_statutes=183, got {len(data.statutes)}")

    train_loader = DataLoader(
        GTRDataset(data.train_z, data.y_train, data.axis_train),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    model = HybridGTRv2(
        dim=data.train_z.shape[1],
        num_axes=data.axis_train.shape[1],
        num_statutes=data.y_train.shape[1],
        axis_hidden=args.axis_hidden,
        field_hidden=args.field_hidden,
        residual_rank=args.residual_rank,
        residual_scale=args.residual_scale,
    ).to(device)
    warm_start_info = warm_start_raw_head(model, data, args)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_valid = -1.0
    best_epoch = -1
    history: List[Dict[str, Any]] = []
    patience_left = args.patience
    global_epoch = 0

    initial_valid = predict_components(model, data.valid_z, args.eval_batch_size, device)
    initial_threshold, initial_valid_metrics = tune_threshold(initial_valid["full_hybrid"], data.y_valid)
    best_valid = initial_valid_metrics["micro_f1"]
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history.append(
        {
            "epoch": 0,
            "stage": "raw_warm_start",
            "train_loss": None,
            "valid_micro_f1": initial_valid_metrics["micro_f1"],
            "valid_exact_match": initial_valid_metrics["exact_match"],
            "valid_threshold": initial_threshold,
            "mean_residual_norm": float(np.mean(initial_valid["residual_norm"])),
        }
    )

    stages = [
        ("field", args.field_pretrain_epochs, args.lr),
        ("residual", args.epochs, args.residual_lr if args.residual_lr is not None else args.lr),
    ]
    for stage_name, stage_epochs, stage_lr in stages:
        if stage_epochs <= 0:
            continue
        set_stage_trainability(model, stage=stage_name, freeze_raw_head=args.freeze_raw_head)
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=stage_lr, weight_decay=args.weight_decay)
        if stage_name == "residual":
            patience_left = args.patience
        for _stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            model.train()
            losses = []
            for z, y, axis in train_loader:
                z = z.to(device)
                y = y.to(device)
                axis = axis.to(device)
                out = model(z)
                statute_loss = F.binary_cross_entropy_with_logits(out["logits"], y)
                axis_loss = masked_axis_bce(out["axis_logits"], axis)
                reg_loss = regularization_loss(
                    model,
                    out,
                    lambda_residual=args.lambda_residual,
                    lambda_field_l1=args.lambda_field_l1,
                )
                loss = statute_loss + args.lambda_axis * axis_loss + reg_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_parameters(model), args.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            valid_components = predict_components(model, data.valid_z, args.eval_batch_size, device)
            threshold, valid_metrics = tune_threshold(valid_components["full_hybrid"], data.y_valid)
            row = {
                "epoch": global_epoch,
                "stage": stage_name,
                "train_loss": float(np.mean(losses)),
                "valid_micro_f1": valid_metrics["micro_f1"],
                "valid_exact_match": valid_metrics["exact_match"],
                "valid_threshold": threshold,
                "mean_residual_norm": float(np.mean(valid_components["residual_norm"])),
            }
            history.append(row)
            print(json.dumps(json_ready(row), ensure_ascii=False), flush=True)
            if valid_metrics["micro_f1"] > best_valid:
                best_valid = valid_metrics["micro_f1"]
                best_epoch = global_epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = args.patience
            elif stage_name == "residual":
                patience_left -= 1
                if patience_left <= 0:
                    break
        if stage_name == "residual" and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    set_stage_trainability(model, stage="residual", freeze_raw_head=args.freeze_raw_head)

    valid_components = predict_components(model, data.valid_z, args.eval_batch_size, device)
    test_components = predict_components(model, data.test_z, args.eval_batch_size, device)
    threshold, valid_metrics = tune_threshold(valid_components["full_hybrid"], data.y_valid)
    test_metrics = metrics_from_logits(test_components["full_hybrid"], data.y_test, threshold=threshold)
    component_metrics = evaluate_component_family(
        valid_components,
        test_components,
        data.y_valid,
        data.y_test,
        component_names=("raw_head_on_z", "raw_plus_field", "raw_head_on_z_prime", "full_hybrid"),
    )
    axis_eval = axis_metrics(test_components["axis_logits"], data.axis_test)
    retrieval = retrieval_metrics(test_components["full_hybrid"], data.y_test, data.statutes, k=args.retrieval_k)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "hybrid_gtr_v2_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "statutes": data.statutes,
            "axis_ids": get_axis_ids(),
            "best_epoch": best_epoch,
            "threshold": threshold,
        },
        checkpoint_path,
    )
    report = {
        "meta": {
            "pipeline": "cail2018_hybrid_gtr_v2_only",
            "dataset": "cail2018_statute_classification",
            "legal_system": "chinese_criminal_law",
            "model_name": args.model_name,
            "encoder": args.model_name,
            "n_train": int(data.train_z.shape[0]),
            "n_valid": int(data.valid_z.shape[0]),
            "n_test": int(data.test_z.shape[0]),
            "n_statutes": len(data.statutes),
            "n_axes": len(get_axis_ids()),
            "axis_dim": int(data.axis_train.shape[1]),
            "num_statutes": int(data.y_train.shape[1]),
            "sample_strategy": args.sample_strategy,
            "best_epoch": best_epoch,
            "checkpoint": checkpoint_path,
            "embedding_cache": embedding_cache_path(args),
        },
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "component_metrics": component_metrics,
        "axis_metrics": axis_eval,
        "retrieval": retrieval,
        "training_history": history,
        "raw_head_initialization": warm_start_info,
        "selected_threshold": threshold,
        "mean_test_residual_norm": float(np.mean(test_components["residual_norm"])),
    }
    write_json(args.output_dir / "hybrid_gtr_v2_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CAIL2018 GTR v2-only model with CAIL axis supervision.")
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    p.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    p.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--encoder-device", default=None, help="SentenceTransformer device for BGE encoding, e.g. cuda, cuda:1, or cpu.")
    p.add_argument("--embed-cache", type=Path, default=None)
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-valid-samples", type=int, default=None)
    p.add_argument("--max-test-samples", type=int, default=None)
    p.add_argument("--sample-strategy", choices=("random", "head"), default="random")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--field-pretrain-epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--residual-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--axis-hidden", type=int, default=256)
    p.add_argument("--field-hidden", type=int, default=0)
    p.add_argument("--residual-rank", type=int, default=32)
    p.add_argument("--residual-scale", type=float, default=0.02)
    p.add_argument("--lambda-axis", type=float, default=0.2)
    p.add_argument("--lambda-residual", type=float, default=0.5)
    p.add_argument("--lambda-field-l1", type=float, default=1e-5)
    p.add_argument("--warm-start-raw-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-raw-head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--raw-probe-cache-dir", type=Path, default=None)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--retrieval-k", type=int, default=10)
    p.add_argument("--use-keyword-rules", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-unknown-axis", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    report = train(parse_args())
    print(f"Saved CAIL GTR v2 report to {report['meta']['checkpoint'].parent / 'hybrid_gtr_v2_report.json'}")
    print("Test exact match:", report["test_metrics"]["exact_match"])
    print("Test micro-F1:", report["test_metrics"]["micro_f1"])
    print("Test macro-F1:", report["test_metrics"]["macro_f1"])


if __name__ == "__main__":
    main()
