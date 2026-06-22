#!/usr/bin/env python3
"""Train Hybrid GTR v2: raw embedding + calibrated axes + decision field.

Implements the model described in the GTR v2 guide:

    z = E(x)
    q = sigmoid(axis_head(z))
    z' = z + B h(q)
    s_g = w_g^T z' + F_g(q)

The implementation uses cached sentence embeddings as z and weak legal-element
axis labels as supervision for q. It reports statute prediction metrics,
axis-head metrics, and statute-to-situation retrieval metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from build_lbox_axis_labels import make_axis_labels
from gtr_axis_schema import get_axis_ids
from lbox_raw_lowrank_eval import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    collect_unique_statutes,
    encode_cases,
    exact_match_score,
    extract_ovr_linear_probe_weights,
    fit_raw_linear_probe,
    l2_normalize,
    load_jsonl,
    load_lbox_splits,
    normalize_labels,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"Label not .* is present in all training examples\.", category=UserWarning)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output/hybrid_gtr_v2"
DEFAULT_EMB_CACHE = REPO_ROOT / "output/gtr_v2_experiments/bge_m3_lbox_embeddings.npz"


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
    test_texts: List[str]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_metric_labels(value: Any) -> List[str]:
    return normalize_labels(value)


def rows_to_axis_labels(rows: Sequence[Mapping[str, Any]], *, use_keyword_rules: bool, allow_unknown_axis: bool) -> np.ndarray:
    axis_ids = get_axis_ids()
    labels = np.full((len(rows), len(axis_ids)), -1 if allow_unknown_axis else 0, dtype=np.int32)
    for i, row in enumerate(rows):
        text = str(row.get("facts") or row.get("text") or "").strip()
        statutes = normalize_metric_labels(row.get("statutes") or row.get("labels") or row.get("label"))
        axis_labels, _sources = make_axis_labels(
            text,
            statutes,
            use_keyword_rules=use_keyword_rules,
            allow_unknown_axis=allow_unknown_axis,
        )
        labels[i] = np.asarray([int(axis_labels[axis_id]) for axis_id in axis_ids], dtype=np.int32)
    return labels


def load_or_encode(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_df, valid_df, test_df = load_lbox_splits(args.train_path, args.valid_path, args.test_path)
    if args.embed_cache.exists() and not args.force_reencode:
        payload = np.load(args.embed_cache)
        return (
            l2_normalize(np.asarray(payload["train"], dtype=np.float32)),
            l2_normalize(np.asarray(payload["valid"], dtype=np.float32)),
            l2_normalize(np.asarray(payload["test"], dtype=np.float32)),
        )
    _, train_z, valid_z, test_z = encode_cases(args.model_name, train_df, valid_df, test_df, args.batch_size)
    args.embed_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.embed_cache, train=train_z, valid=valid_z, test=test_z, model=args.model_name)
    return l2_normalize(train_z), l2_normalize(valid_z), l2_normalize(test_z)


def load_splits(args: argparse.Namespace) -> Splits:
    train_df, valid_df, test_df = load_lbox_splits(args.train_path, args.valid_path, args.test_path)
    statutes = collect_unique_statutes(train_df, valid_df, test_df)
    mlb = MultiLabelBinarizer(classes=statutes)
    y_train = mlb.fit_transform(train_df["label"]).astype(np.float32)
    y_valid = mlb.transform(valid_df["label"]).astype(np.float32)
    y_test = mlb.transform(test_df["label"]).astype(np.float32)
    train_z, valid_z, test_z = load_or_encode(args)
    train_rows = load_jsonl(args.train_path)
    valid_rows = load_jsonl(args.valid_path)
    test_rows = load_jsonl(args.test_path)
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
        test_texts=[str(row.get("facts", "")) for row in test_rows],
    )


class GTRDataset(Dataset):
    def __init__(self, z: np.ndarray, y: np.ndarray, axis: np.ndarray) -> None:
        self.z = torch.from_numpy(z.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.axis = torch.from_numpy(axis.astype(np.float32))

    def __len__(self) -> int:
        return int(self.z.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z[idx], self.y[idx], self.axis[idx]


def pairwise_q(q: torch.Tensor) -> torch.Tensor:
    pieces = [q]
    n = q.shape[1]
    if n > 1:
        pieces.append(torch.cat([q[:, i : i + 1] * q[:, j : j + 1] for i in range(n) for j in range(i + 1, n)], dim=1))
    return torch.cat(pieces, dim=1)


class HybridGTRv2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_axes: int,
        num_statutes: int,
        *,
        axis_hidden: int = 256,
        field_hidden: int = 0,
        residual_rank: int = 32,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_axes = int(num_axes)
        self.residual_scale = float(residual_scale)
        self.axis_head = nn.Sequential(
            nn.Linear(dim, axis_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(axis_hidden, num_axes),
        )
        field_dim = num_axes + (num_axes * (num_axes - 1)) // 2
        if field_hidden > 0:
            self.field_head = nn.Sequential(nn.Linear(field_dim, field_hidden), nn.ReLU(), nn.Linear(field_hidden, num_statutes))
        else:
            self.field_head = nn.Linear(field_dim, num_statutes)
        self.residual_h = nn.Sequential(
            nn.Linear(num_axes, max(16, residual_rank)),
            nn.ReLU(),
            nn.Linear(max(16, residual_rank), residual_rank),
        )
        self.residual_b = nn.Linear(residual_rank, dim, bias=False)
        self.raw_head = nn.Linear(dim, num_statutes)
        self.residual_enabled = True
        self._init_correction_modules()

    def _init_correction_modules(self) -> None:
        """Start from the raw baseline: F_g(q)=0 and Bh(q)=0 before training."""
        if isinstance(self.field_head, nn.Linear):
            nn.init.zeros_(self.field_head.weight)
            nn.init.zeros_(self.field_head.bias)
        elif isinstance(self.field_head, nn.Sequential):
            last = self.field_head[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
        nn.init.zeros_(self.residual_b.weight)

    def forward(self, z: torch.Tensor, *, residual_scale_override: Optional[float] = None) -> Dict[str, torch.Tensor]:
        axis_logits = self.axis_head(z)
        q = torch.sigmoid(axis_logits)
        scale = self.residual_scale if residual_scale_override is None else float(residual_scale_override)
        if not self.residual_enabled:
            scale = 0.0
        residual = scale * self.residual_b(self.residual_h(q))
        z_prime = z + residual
        field_features = pairwise_q(q)
        field_logits = self.field_head(field_features)
        raw_logits = self.raw_head(z_prime)
        return {
            "logits": raw_logits + field_logits,
            "raw_logits": raw_logits,
            "field_logits": field_logits,
            "axis_logits": axis_logits,
            "q": q,
            "z_prime": z_prime,
            "residual": residual,
        }


def masked_axis_bce(axis_logits: torch.Tensor, axis_labels: torch.Tensor) -> torch.Tensor:
    mask = axis_labels >= 0
    if not torch.any(mask):
        return axis_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(axis_logits[mask], axis_labels[mask])


def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    y_pred = (logits >= threshold).astype(np.int32)
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": exact_match_score(y_true, y_pred),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
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
def predict(model: nn.Module, z: np.ndarray, batch_size: int, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    loader = DataLoader(torch.from_numpy(z.astype(np.float32)), batch_size=batch_size, shuffle=False)
    chunks: Dict[str, List[np.ndarray]] = {"logits": [], "raw_logits": [], "field_logits": [], "axis_logits": [], "q": [], "residual_norm": []}
    for batch in loader:
        out = model(batch.to(device))
        for key in ("logits", "raw_logits", "field_logits", "axis_logits", "q"):
            chunks[key].append(out[key].cpu().numpy())
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


def regularization_loss(model: HybridGTRv2, out: Mapping[str, torch.Tensor], *, lambda_residual: float, lambda_field_l1: float) -> torch.Tensor:
    residual_penalty = out["residual"].pow(2).mean()
    l1 = torch.zeros((), device=out["residual"].device)
    for param in model.field_head.parameters():
        l1 = l1 + param.abs().mean()
    return lambda_residual * residual_penalty + lambda_field_l1 * l1


def warm_start_raw_head(model: HybridGTRv2, data: Splits, args: argparse.Namespace) -> Dict[str, Any]:
    if not args.warm_start_raw_head:
        return {"enabled": False}
    metrics, _pred_test, clf = fit_raw_linear_probe(
        data.train_z,
        data.valid_z,
        data.test_z,
        data.y_train,
        data.y_valid,
        data.y_test,
        model_dir=args.raw_probe_cache_dir,
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


def set_stage_trainability(model: HybridGTRv2, *, stage: str, freeze_raw_head: bool) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if freeze_raw_head:
        for param in model.raw_head.parameters():
            param.requires_grad = False
    if stage == "field":
        model.residual_enabled = False
        for param in model.residual_h.parameters():
            param.requires_grad = False
        for param in model.residual_b.parameters():
            param.requires_grad = False
    elif stage == "residual":
        model.residual_enabled = True
    else:
        raise ValueError(f"Unknown training stage: {stage}")


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_splits(args)
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

    initial_pred = predict(model, data.valid_z, args.eval_batch_size, device)
    initial_threshold, initial_valid_metrics = tune_threshold(initial_pred["logits"], data.y_valid)
    best_valid = initial_valid_metrics["micro_f1"]
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history.append(
        {
            "epoch": 0,
            "stage": "raw_warm_start",
            "train_loss": None,
            "valid_micro_f1": initial_valid_metrics["micro_f1"],
            "valid_threshold": initial_threshold,
            "mean_residual_norm": float(np.mean(initial_pred["residual_norm"])),
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
            epoch = global_epoch
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

            valid_pred = predict(model, data.valid_z, args.eval_batch_size, device)
            threshold, valid_metrics = tune_threshold(valid_pred["logits"], data.y_valid)
            row = {
                "epoch": epoch,
                "stage": stage_name,
                "train_loss": float(np.mean(losses)),
                "valid_micro_f1": valid_metrics["micro_f1"],
                "valid_threshold": threshold,
                "mean_residual_norm": float(np.mean(valid_pred["residual_norm"])),
            }
            history.append(row)
            if valid_metrics["micro_f1"] > best_valid:
                best_valid = valid_metrics["micro_f1"]
                best_epoch = epoch
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

    valid_pred = predict(model, data.valid_z, args.eval_batch_size, device)
    threshold, valid_metrics = tune_threshold(valid_pred["logits"], data.y_valid)
    test_pred = predict(model, data.test_z, args.eval_batch_size, device)
    test_metrics = metrics_from_logits(test_pred["logits"], data.y_test, threshold=threshold)
    raw_component_metrics = metrics_from_logits(test_pred["raw_logits"], data.y_test, threshold=threshold)
    field_component_metrics = metrics_from_logits(test_pred["field_logits"], data.y_test, threshold=threshold)
    axis_eval = axis_metrics(test_pred["axis_logits"], data.axis_test)
    retrieval = retrieval_metrics(test_pred["logits"], data.y_test, data.statutes, k=args.retrieval_k)

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
            "pipeline": "hybrid_gtr_v2",
            "model_name": args.model_name,
            "n_train": int(data.train_z.shape[0]),
            "n_valid": int(data.valid_z.shape[0]),
            "n_test": int(data.test_z.shape[0]),
            "n_statutes": len(data.statutes),
            "n_axes": len(get_axis_ids()),
            "best_epoch": best_epoch,
            "checkpoint": checkpoint_path,
        },
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "component_metrics": {
            "raw_head_on_z_prime": raw_component_metrics,
            "decision_field_only": field_component_metrics,
        },
        "axis_metrics": axis_eval,
        "retrieval": retrieval,
        "training_history": history,
        "raw_head_initialization": warm_start_info,
        "selected_threshold": threshold,
        "mean_test_residual_norm": float(np.mean(test_pred["residual_norm"])),
    }
    write_json(args.output_dir / "hybrid_gtr_v2_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Hybrid GTR v2 with residual correction and decision field.")
    p.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    p.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    p.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--embed-cache", type=Path, default=DEFAULT_EMB_CACHE)
    p.add_argument("--force-reencode", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
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
    p.add_argument("--raw-probe-cache-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "raw_probe_cache")
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--retrieval-k", type=int, default=10)
    p.add_argument("--use-keyword-rules", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--allow-unknown-axis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If false, axes not triggered by statute/keyword rules are treated as negatives for axis supervision.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return p.parse_args()


def main() -> None:
    report = train(parse_args())
    print(f"Saved Hybrid GTR v2 report to {report['meta']['checkpoint'].parent / 'hybrid_gtr_v2_report.json'}")
    print("Test micro-F1:", report["test_metrics"]["micro_f1"])
    print("Retrieval macro nDCG@K:", report["retrieval"]["macro_ndcg_at_k"])


if __name__ == "__main__":
    main()
