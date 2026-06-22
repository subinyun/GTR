#!/usr/bin/env python3
"""
Run and evaluate only two LBOX representations:
  1. Raw BGE-M3 embeddings + OneVsRest linear probe
  2. Fixed-probe low-rank residual GTR operator
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings(
    "ignore",
    message=r"Label not .* is present in all training examples\.",
    category=UserWarning,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - only used in minimal environments.
    def tqdm(iterable: Iterable[Any], **_: Any) -> Iterable[Any]:
        return iterable


DEFAULT_TRAIN_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/train.jsonl")
DEFAULT_VALID_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/valid.jsonl")
DEFAULT_TEST_PATH = Path("/home/sbyoon/patrol-law-llm/LBOX/statute_classification/test.jsonl")
DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_DESCRIPTOR_PATH = Path("statute_descriptors_improved.json")
DEFAULT_DESCRIPTOR_AXIS_PATH = Path("statute_axis_descriptor.npy")
DEFAULT_HARD_NEGATIVE_DESCRIPTOR_PATH = Path("statute_hard_negative_descriptors.json")
DEFAULT_HARD_NEGATIVE_AXIS_PATH = Path("statute_axis_hard_negative.npy")
DEFAULT_OUTPUT_PATH = Path("lbox_raw_lowrank_eval_results.json")
DEFAULT_LOWRANK_OUTPUT_PATH = Path("lbox_lowrank_gtr_fixed_probe_results.json")
DEFAULT_MODEL_CACHE_DIR = Path("output")
DEFAULT_RAW_PREDICTION_PATH = DEFAULT_MODEL_CACHE_DIR / "raw_predictions.json"
DEFAULT_LOWRANK_PREDICTION_PATH = DEFAULT_MODEL_CACHE_DIR / "lowrank_predictions.json"
DEFAULT_HARD_NEGATIVE_TOP_K = 50
DEFAULT_LOWRANK_RANK_GRID = (8, 16)
DEFAULT_LOWRANK_LAMBDA_OP_GRID = (0.1, 0.3, 0.5)
DEFAULT_LOWRANK_LR = 1e-4
DEFAULT_LOWRANK_EPOCHS = 20
DEFAULT_LOWRANK_PATIENCE = 5
DEFAULT_LOWRANK_BATCH_SIZE = 64
DEFAULT_LOWRANK_LAMBDA_DELTA_GRID = (0.0, 0.01)
DEFAULT_LOWRANK_SEED = 42
DEFAULT_C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)

STOPWORDS = {
    "피고인",
    "피해자",
    "사건",
    "공소",
    "범행",
    "경우",
    "이후",
    "당시",
    "그곳",
    "같은",
    "관련",
    "위하여",
    "대하여",
    "대한",
    "통하여",
    "인하여",
    "그리고",
    "그러나",
    "또는",
    "하는",
    "한다",
    "하여",
    "하고",
    "하며",
    "하자",
    "하였다",
    "하였다가",
    "있다",
    "있는",
    "없다",
    "없는",
    "되었다",
    "자신",
    "자신의",
    "명의",
    "정도",
    "약",
    "회",
    "명",
    "시",
    "분",
    "경",
    "소재",
    "장소",
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "제주",
    "강원",
    "경기",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "이하",
    "이라",
    "위반",
    "위반하였",
    "피고인은",
    "피해자는",
}


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


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


def normalize_labels(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;|]", value) if part.strip()]
    text = str(value).strip()
    return [text] if text else []


def rows_to_frame(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    texts = [str(row.get("facts", "")).strip() for row in rows]
    labels = [normalize_labels(row.get("statutes")) for row in rows]
    case_names = [str(row.get("casename", "")).strip() for row in rows]
    return pd.DataFrame({"text": texts, "label": labels, "casename": case_names})


def load_lbox_splits(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = rows_to_frame(load_jsonl(train_path))
    valid_df = rows_to_frame(load_jsonl(valid_path))
    test_df = rows_to_frame(load_jsonl(test_path))
    return train_df, valid_df, test_df


def collect_unique_statutes(*frames: pd.DataFrame) -> List[str]:
    return sorted(set().union(*(set(labels) for frame in frames for labels in frame["label"])))


def clean_facts_for_keywords(text: str) -> str:
    text = re.sub(r"\d{4}\.\s*\d{1,2}\.\s*\d{1,2}\.?", " ", text)
    text = re.sub(r"\d{1,2}:\d{2}", " ", text)
    text = re.sub(r"\d+(?:-\d+)?호|제?\d+조(?:의\d+)?(?:\s*제\d+항)?", " ", text)
    text = re.sub(r"[A-Z](?:\([^)]+\))?", " ", text)
    text = re.sub(r"\b[가-힣]\b", " ", text)
    text = re.sub(r"[^가-힣\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_keyword_token(token: str) -> str:
    token = token.strip()
    suffixes = (
        "으로부터",
        "로부터",
        "에게",
        "에서",
        "부터",
        "까지",
        "으로",
        "라고",
        "이라",
        "이며",
        "이고",
        "에는",
        "에도",
        "와",
        "과",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "로",
        "에",
        "만",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                changed = True
                break
    return token


def has_final_consonant(text: str) -> bool:
    for char in reversed(text.strip()):
        if "가" <= char <= "힣":
            return (ord(char) - ord("가")) % 28 != 0
    return False


def topic_josa(text: str) -> str:
    return "은" if has_final_consonant(text) else "는"


def extract_keywords_from_facts(
    facts_list: Sequence[str],
    top_k: int = 15,
    max_docs: int = 200,
) -> List[str]:
    counter: Counter[str] = Counter()
    for facts in facts_list[:max_docs]:
        cleaned = clean_facts_for_keywords(facts)
        tokens = [
            normalized
            for token in re.findall(r"[가-힣]{2,}", cleaned)
            for normalized in [normalize_keyword_token(token)]
            if normalized not in STOPWORDS and len(normalized) >= 2
        ]
        counter.update(tokens)

        for left, right in zip(tokens, tokens[1:]):
            phrase = f"{left} {right}"
            if left in STOPWORDS or right in STOPWORDS:
                continue
            counter[phrase] += 1

    keywords: List[str] = []
    for keyword, _ in counter.most_common(top_k * 3):
        if any(keyword in selected or selected in keyword for selected in keywords):
            continue
        keywords.append(keyword)
        if len(keywords) >= top_k:
            break
    return keywords


def build_improved_descriptor_records(
    train_df: pd.DataFrame,
    statutes: Sequence[str],
    top_keyword_count: int = 15,
) -> List[Dict[str, Any]]:
    statute_to_facts: Dict[str, List[str]] = defaultdict(list)
    statute_to_casenames: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        labels = getattr(row, "label")
        facts = str(getattr(row, "text"))
        case_name = str(getattr(row, "casename")).strip()
        for statute in labels:
            statute_to_facts[statute].append(facts)
            if case_name:
                statute_to_casenames[statute][case_name] += 1

    records: List[Dict[str, Any]] = []
    fallback_template = "{statute}에 해당하는 법적 구성요건과 사건 사실의 관련성을 판단한다."
    for statute in tqdm(statutes, desc="Building improved statute descriptors", unit="statute"):
        facts_list = statute_to_facts.get(statute, [])
        case_counter = statute_to_casenames.get(statute, Counter())
        if not facts_list:
            records.append(
                {
                    "statute": statute,
                    "train_frequency": 0,
                    "top_casenames": [],
                    "descriptor": fallback_template.format(statute=statute),
                    "fallback": True,
                }
            )
            continue

        top_casenames = [name for name, _ in case_counter.most_common(3)]
        keywords = extract_keywords_from_facts(facts_list, top_k=top_keyword_count)
        keyword_text = ", ".join(keywords) if keywords else "행위, 피해, 고의, 수단, 대상, 결과"
        case_text = ", ".join(top_casenames) if top_casenames else "관련 형사"
        descriptor = (
            f"{statute}{topic_josa(statute)} 주로 {case_text} 사건에 적용된다. "
            f"핵심 판단 기준은 사건 사실에서 {keyword_text} 등과 관련된 "
            "행위, 피해, 고의, 수단, 대상, 결과가 나타나는지 여부이다."
        )
        records.append(
            {
                "statute": statute,
                "train_frequency": len(facts_list),
                "top_casenames": top_casenames,
                "descriptor": descriptor,
                "fallback": False,
            }
        )
    return records


def save_descriptor_records(records: Sequence[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def embed_descriptor_axis(
    records: Sequence[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
    output_path: Path,
) -> np.ndarray:
    descriptors = [record["descriptor"] for record in records]
    axis_matrix = model.encode(
        descriptors,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    axis_matrix = l2_normalize(np.asarray(axis_matrix, dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, axis_matrix)
    return axis_matrix


def build_and_save_descriptor_axis(
    train_df: pd.DataFrame,
    statutes: Sequence[str],
    model: SentenceTransformer,
    batch_size: int,
    descriptor_path: Path,
    descriptor_axis_path: Path,
) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    records = build_improved_descriptor_records(train_df, statutes)
    save_descriptor_records(records, descriptor_path)
    axis_matrix = embed_descriptor_axis(records, model, batch_size, descriptor_axis_path)
    return [record["statute"] for record in records], axis_matrix, records


def collect_statute_evidence(
    train_df: pd.DataFrame,
) -> Tuple[Counter[str], Dict[str, List[str]], Dict[str, Counter[str]]]:
    statute_counter: Counter[str] = Counter()
    statute_to_facts: Dict[str, List[str]] = defaultdict(list)
    statute_to_casenames: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        labels = list(getattr(row, "label"))
        facts = str(getattr(row, "text"))
        case_name = str(getattr(row, "casename")).strip()
        statute_counter.update(labels)
        for statute in labels:
            statute_to_facts[statute].append(facts)
            if case_name:
                statute_to_casenames[statute][case_name] += 1
    return statute_counter, statute_to_facts, statute_to_casenames


def find_hard_negative_pairs(
    train_df: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    statute_counter, _, _ = collect_statute_evidence(train_df)
    cooccurrence: Counter[Tuple[str, str]] = Counter()
    casename_to_statutes: Dict[str, Counter[str]] = defaultdict(Counter)

    for row in train_df.itertuples(index=False):
        labels = sorted(set(getattr(row, "label")))
        case_name = str(getattr(row, "casename")).strip()
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                cooccurrence[(left, right)] += 1
        if case_name:
            casename_to_statutes[case_name].update(labels)

    same_casename_score: Counter[Tuple[str, str]] = Counter()
    for statute_counts in casename_to_statutes.values():
        statutes = sorted(statute_counts)
        for i, left in enumerate(statutes):
            for right in statutes[i + 1 :]:
                same_casename_score[(left, right)] += min(statute_counts[left], statute_counts[right])

    candidate_pairs = set(cooccurrence) | set(same_casename_score)
    scored_pairs: List[Dict[str, Any]] = []
    for left, right in candidate_pairs:
        co = int(cooccurrence.get((left, right), 0))
        same_case = int(same_casename_score.get((left, right), 0))
        support_min = min(statute_counter[left], statute_counter[right])
        score = (3.0 * co) + same_case + (0.05 * support_min)
        scored_pairs.append(
            {
                "statute_a": left,
                "statute_b": right,
                "cooccurrence": co,
                "same_casename_score": same_case,
                "support_a": int(statute_counter[left]),
                "support_b": int(statute_counter[right]),
                "hard_negative_score": float(score),
            }
        )

    scored_pairs.sort(key=lambda item: item["hard_negative_score"], reverse=True)
    return scored_pairs[:top_k]


def format_slot_terms(terms: Sequence[str]) -> str:
    return ", ".join(terms) if terms else "없음"


def condition_text(statute: str, keywords: Sequence[str], casenames: Sequence[str]) -> str:
    parts = list(dict.fromkeys(keywords[:8]))
    if casenames:
        parts = list(dict.fromkeys(list(casenames[:2]) + parts))
    if not parts:
        return f"{statute}의 고유한 구성요건이 사건 사실에서 확인되는 경우"
    return ", ".join(parts)


def build_hard_negative_descriptor_records(
    train_df: pd.DataFrame,
    top_k: int,
) -> List[Dict[str, Any]]:
    _, statute_to_facts, statute_to_casenames = collect_statute_evidence(train_df)
    pairs = find_hard_negative_pairs(train_df, top_k=top_k)
    keyword_cache: Dict[str, List[str]] = {}

    def keywords_for(statute: str) -> List[str]:
        if statute not in keyword_cache:
            keyword_cache[statute] = extract_keywords_from_facts(
                statute_to_facts.get(statute, []),
                top_k=20,
            )
        return keyword_cache[statute]

    records: List[Dict[str, Any]] = []
    for pair in tqdm(pairs, desc="Building hard-negative descriptors", unit="pair"):
        statute_a = pair["statute_a"]
        statute_b = pair["statute_b"]
        keywords_a = keywords_for(statute_a)
        keywords_b = keywords_for(statute_b)
        common_keywords = sorted(set(keywords_a) & set(keywords_b))
        diff_a = [keyword for keyword in keywords_a if keyword not in common_keywords][:10]
        diff_b = [keyword for keyword in keywords_b if keyword not in common_keywords][:10]
        casenames_a = [name for name, _ in statute_to_casenames.get(statute_a, Counter()).most_common(3)]
        casenames_b = [name for name, _ in statute_to_casenames.get(statute_b, Counter()).most_common(3)]
        condition_a = condition_text(statute_a, diff_a, casenames_a)
        condition_b = condition_text(statute_b, diff_b, casenames_b)
        boundary = (
            f"{statute_a}{topic_josa(statute_a)} {condition_a} 요건이 두드러질 때 적용되고, "
            f"{statute_b}{topic_josa(statute_b)} {condition_b} 요건이 두드러질 때 적용된다. "
            f"두 법령의 핵심 차이는 {statute_a}의 구별 키워드({format_slot_terms(diff_a[:6])})와 "
            f"{statute_b}의 구별 키워드({format_slot_terms(diff_b[:6])})가 사건 사실에서 "
            "어느 쪽 구성요건에 더 직접적으로 대응하는지이다."
        )
        records.append(
            {
                **pair,
                "top_casenames_a": casenames_a,
                "top_casenames_b": casenames_b,
                "diff_keywords_a": diff_a,
                "diff_keywords_b": diff_b,
                "common_keywords": common_keywords[:10],
                "descriptor": boundary,
            }
        )
    return records


def embed_hard_negative_axis(
    records: Sequence[Dict[str, Any]],
    model: SentenceTransformer,
    batch_size: int,
    output_path: Path,
) -> np.ndarray:
    descriptors = [record["descriptor"] for record in records]
    axis_matrix = model.encode(
        descriptors,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    axis_matrix = l2_normalize(np.asarray(axis_matrix, dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, axis_matrix)
    return axis_matrix


def build_and_save_hard_negative_axis(
    train_df: pd.DataFrame,
    model: SentenceTransformer,
    batch_size: int,
    top_k: int,
    descriptor_path: Path,
    axis_path: Path,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    records = build_hard_negative_descriptor_records(train_df, top_k=top_k)
    descriptor_path.parent.mkdir(parents=True, exist_ok=True)
    with descriptor_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    axis_matrix = embed_hard_negative_axis(records, model, batch_size, axis_path)
    return axis_matrix, records


def encode_cases(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int,
) -> Tuple[SentenceTransformer, np.ndarray, np.ndarray, np.ndarray]:
    model = SentenceTransformer(model_name)
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
    train_emb = l2_normalize(np.asarray(train_emb, dtype=np.float32))
    valid_emb = l2_normalize(np.asarray(valid_emb, dtype=np.float32))
    test_emb = l2_normalize(np.asarray(test_emb, dtype=np.float32))
    return model, train_emb, valid_emb, test_emb


def reorder_axis_to_classes(
    axis_statutes: Sequence[str],
    axis_matrix: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    statute_to_idx = {statute: idx for idx, statute in enumerate(axis_statutes)}
    missing = [label for label in classes if label not in statute_to_idx]
    if missing:
        raise ValueError(
            "Missing labels in statute axis: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    return axis_matrix[[statute_to_idx[label] for label in classes]]


def exact_match_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for exact match: y_true={y_true.shape}, y_pred={y_pred.shape}")
    if y_true.shape[0] == 0:
        return 0.0
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def fit_raw_linear_probe(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    model_dir: Path | None = None,
) -> Tuple[Dict[str, float], np.ndarray, OneVsRestClassifier]:
    best_c = 1.0
    best_valid_micro = -1.0
    best_clf: OneVsRestClassifier | None = None
    best_pred_valid: np.ndarray | None = None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    for C in c_grid:
        model_path = model_dir / f"linear_probe_C={C}.joblib" if model_dir is not None else None
        if model_path is not None and model_path.exists():
            clf = joblib.load(model_path)
            print("Loaded model from", model_path)
        else:
            if model_path is not None:
                print("Training new model...")
            clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=-1,
                )
            )
            clf.fit(x_train, y_train)
            if model_path is not None:
                joblib.dump(clf, model_path)
                print("Saved model to", model_path)
        pred_valid = clf.predict(x_valid)
        valid_micro = f1_score(y_valid, pred_valid, average="micro", zero_division=0)
        if valid_micro > best_valid_micro:
            best_valid_micro = valid_micro
            best_c = C
            best_clf = clf
            best_pred_valid = pred_valid

    if best_clf is None or best_pred_valid is None:
        raise ValueError("C grid must contain at least one value.")

    clf = best_clf
    pred_valid = best_pred_valid
    pred_test = clf.predict(x_test)

    metrics = {
        "micro_f1": float(f1_score(y_test, pred_test, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, pred_test, average="macro", zero_division=0)),
        "exact_match": exact_match_score(y_test, pred_test),
        "precision_micro": float(precision_score(y_test, pred_test, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_test, pred_test, average="micro", zero_division=0)),
        "best_C_from_valid_micro_f1": float(best_c),
        "valid_micro_f1_at_best_C": float(best_valid_micro),
        "valid_macro_f1_at_best_C": float(
            f1_score(y_valid, pred_valid, average="macro", zero_division=0)
        ),
        "valid_exact_match_at_best_C": exact_match_score(y_valid, pred_valid),
        "valid_precision_micro_at_best_C": float(
            precision_score(y_valid, pred_valid, average="micro", zero_division=0)
        ),
        "valid_recall_micro_at_best_C": float(
            recall_score(y_valid, pred_valid, average="micro", zero_division=0)
        ),
    }
    return metrics, pred_test, clf


def extract_ovr_linear_probe_weights(clf: OneVsRestClassifier) -> Tuple[np.ndarray, np.ndarray]:
    if not clf.estimators_:
        raise ValueError("Cannot extract weights from an unfitted OneVsRestClassifier.")

    n_features = int(getattr(clf, "n_features_in_", 0))
    if n_features <= 0:
        first_linear_estimator = next(
            (estimator for estimator in clf.estimators_ if hasattr(estimator, "coef_")),
            None,
        )
        if first_linear_estimator is None:
            raise ValueError("Could not infer feature dimension from OneVsRestClassifier.")
        n_features = int(first_linear_estimator.coef_.shape[1])

    coef_rows: List[np.ndarray] = []
    intercept_values: List[float] = []
    zero_probe = np.zeros((1, n_features), dtype=np.float32)
    for estimator in clf.estimators_:
        if hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
            coef_rows.append(estimator.coef_.reshape(-1).astype(np.float32))
            intercept_values.append(float(estimator.intercept_.reshape(-1)[0]))
            continue

        constant_pred = int(np.asarray(estimator.predict(zero_probe)).reshape(-1)[0])
        coef_rows.append(np.zeros(n_features, dtype=np.float32))
        intercept_values.append(1.0 if constant_pred == 1 else -1.0)

    coef = np.vstack(coef_rows).astype(np.float32)
    intercept = np.asarray(intercept_values, dtype=np.float32)
    return coef.T.copy(), intercept


def metrics_from_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": exact_match_score(y_true, y_pred),
        "precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def save_prediction_payload(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    classes: Sequence[str] | None = None,
    y_score: np.ndarray | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "y_true": np.asarray(y_true, dtype=np.int32).tolist(),
        "y_pred": np.asarray(y_pred, dtype=np.int32).tolist(),
    }
    if y_score is not None:
        payload["y_score"] = np.asarray(y_score, dtype=np.float32).tolist()
    if classes is not None:
        payload["classes"] = [str(label) for label in classes]
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def predict_raw_scores(clf: OneVsRestClassifier, x_test: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        scores = np.asarray(clf.predict_proba(x_test), dtype=np.float32)
    elif hasattr(clf, "decision_function"):
        scores = sigmoid_np(np.asarray(clf.decision_function(x_test), dtype=np.float32))
    else:
        raise ValueError("Raw classifier must expose predict_proba or decision_function.")
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    return scores.astype(np.float32)


class LowRankGTRDataset(Dataset):
    """Dataset for fixed-probe low-rank operator training."""

    def __init__(
        self,
        z: np.ndarray,
        descriptor_axis: np.ndarray,
        hard_negative_axis: np.ndarray,
        y: np.ndarray,
    ) -> None:
        z_norm = l2_normalize(np.asarray(z, dtype=np.float32))
        descriptor_axis_norm = l2_normalize(np.asarray(descriptor_axis, dtype=np.float32))
        hard_negative_axis_norm = l2_normalize(np.asarray(hard_negative_axis, dtype=np.float32))
        g_desc = z_norm @ descriptor_axis_norm.T
        g_hn = z_norm @ hard_negative_axis_norm.T

        self.z_norm = torch.from_numpy(z_norm.astype(np.float32))
        self.g = torch.from_numpy(np.concatenate([g_desc, g_hn], axis=1).astype(np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.z_norm.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.z_norm[idx], self.g[idx], self.y[idx]


class LowRankGTROperator(nn.Module):
    def __init__(
        self,
        d: int = 1024,
        g_dim: int = 219,
        rank: int = 16,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.U = nn.Parameter(torch.randn(d, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(d, rank) * 0.02)
        self.alpha_mlp = nn.Sequential(
            nn.Linear(d + g_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, rank),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        g: torch.Tensor,
        lambda_op: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concat_input = torch.cat([z, g], dim=1)
        alpha = self.alpha_mlp(concat_input)
        r = z @ self.V
        delta_low = alpha * r
        delta_z = delta_low @ self.U.T
        z_tilde = z + lambda_op * delta_z
        return z_tilde, alpha, delta_z


class FixedProbeLowRankGTRModel(nn.Module):
    def __init__(
        self,
        w_raw: np.ndarray,
        b_raw: np.ndarray,
        d: int = 1024,
        g_dim: int = 219,
        rank: int = 16,
        hidden: int = 256,
        lambda_op: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_op = float(lambda_op)
        self.operator = LowRankGTROperator(
            d=d,
            g_dim=g_dim,
            rank=rank,
            hidden=hidden,
        )
        self.register_buffer("w_raw", torch.from_numpy(np.asarray(w_raw, dtype=np.float32)))
        self.register_buffer("b_raw", torch.from_numpy(np.asarray(b_raw, dtype=np.float32)))
        self.w_raw.requires_grad_(False)
        self.b_raw.requires_grad_(False)

    def forward(
        self,
        z: torch.Tensor,
        g: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_tilde, alpha, delta_z = self.operator(z, g, self.lambda_op)
        logits = z_tilde @ self.w_raw + self.b_raw
        return logits, z_tilde, alpha, delta_z


def set_torch_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_lowrank_model(
    model: FixedProbeLowRankGTRModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    alpha_sum: torch.Tensor | None = None
    alpha_sq_sum: torch.Tensor | None = None
    delta_norm_sum = 0.0
    sample_count = 0

    for z, g, y in loader:
        z = z.to(device)
        g = g.to(device)
        logits, _, alpha, delta_z = model(z, g)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(y.numpy())

        batch_size = z.shape[0]
        alpha_batch_sum = alpha.detach().sum(dim=0)
        alpha_batch_sq_sum = alpha.detach().pow(2).sum(dim=0)
        if alpha_sum is None:
            alpha_sum = alpha_batch_sum
            alpha_sq_sum = alpha_batch_sq_sum
        else:
            alpha_sum += alpha_batch_sum
            alpha_sq_sum += alpha_batch_sq_sum
        delta_norm_sum += float(torch.linalg.vector_norm(delta_z.detach(), dim=1).sum().item())
        sample_count += int(batch_size)

    logits_np = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(np.int32)
    y_pred = (logits_np >= 0.0).astype(np.int32)
    metrics = metrics_from_binary_predictions(y_true, y_pred)

    if alpha_sum is None or alpha_sq_sum is None or sample_count == 0:
        diagnostics = {"mean_alpha": 0.0, "std_alpha": 0.0, "mean_delta_norm": 0.0}
    else:
        alpha_mean_by_rank = alpha_sum / sample_count
        alpha_var_by_rank = alpha_sq_sum / sample_count - alpha_mean_by_rank.pow(2)
        diagnostics = {
            "mean_alpha": float(alpha_mean_by_rank.mean().item()),
            "std_alpha": float(torch.sqrt(torch.clamp(alpha_var_by_rank, min=0.0)).mean().item()),
            "mean_delta_norm": float(delta_norm_sum / sample_count),
        }
    return metrics, diagnostics


@torch.no_grad()
def predict_lowrank_model_with_scores(
    model: FixedProbeLowRankGTRModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    for z, g, _ in loader:
        z = z.to(device)
        g = g.to(device)
        logits, _, _, _ = model(z, g)
        all_logits.append(logits.detach().cpu().numpy())
    logits_np = np.concatenate(all_logits, axis=0)
    y_pred = (logits_np >= 0.0).astype(np.int32)
    y_score = sigmoid_np(logits_np).astype(np.float32)
    return y_pred, y_score


def train_one_lowrank_config(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    w_raw: np.ndarray,
    b_raw: np.ndarray,
    d: int,
    g_dim: int,
    rank: int,
    lambda_op: float,
    lr: float,
    epochs: int,
    patience: int,
    lambda_delta: float,
    seed: int,
    device: torch.device,
    model_dir: Path | None = None,
) -> Dict[str, Any]:
    set_torch_seed(seed)
    model = FixedProbeLowRankGTRModel(
        w_raw=w_raw,
        b_raw=b_raw,
        d=d,
        g_dim=g_dim,
        rank=rank,
        lambda_op=lambda_op,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.operator.parameters(), lr=lr)

    best_valid_macro = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: Dict[str, torch.Tensor] | None = None
    history: List[Dict[str, float]] = []

    model_path: Path | None = None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / (
            f"lowrank_rank={rank}_lambda={lambda_op}"
            f"_ldelta={lambda_delta}_lr={lr}"
            f"_epochs={epochs}_patience={patience}_seed={seed}.pt"
        )

    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model from", model_path)
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    else:
        if model_path is not None:
            print("Training new model...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for z, g, y in train_loader:
                z = z.to(device)
                g = g.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits, _, _, delta_z = model(z, g)
                loss_task = criterion(logits, y)
                loss_delta = delta_z.pow(2).sum(dim=1).mean()
                loss = loss_task + lambda_delta * loss_delta
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * int(z.shape[0])

            valid_metrics, _ = eval_lowrank_model(model, valid_loader, device)
            epoch_row = {
                "epoch": float(epoch),
                "train_loss": float(total_loss / len(train_loader.dataset)),
                "valid_micro_f1": valid_metrics["micro_f1"],
                "valid_macro_f1": valid_metrics["macro_f1"],
            }
            history.append(epoch_row)
            print(
                "Low-rank GTR "
                f"rank={rank} lambda_op={lambda_op} lambda_delta={lambda_delta} lr={lr} "
                f"epoch={epoch} valid_micro={valid_metrics['micro_f1']:.4f} "
                f"valid_macro={valid_metrics['macro_f1']:.4f}"
            )

            if valid_metrics["macro_f1"] > best_valid_macro:
                best_valid_macro = valid_metrics["macro_f1"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print("Saved model to", model_path)

    train_metrics, train_diag = eval_lowrank_model(model, train_loader, device)
    valid_metrics, valid_diag = eval_lowrank_model(model, valid_loader, device)
    test_metrics, test_diag = eval_lowrank_model(model, test_loader, device)
    for metrics in (train_metrics, valid_metrics, test_metrics):
        metrics["best_C_from_valid_micro_f1"] = None
        metrics["valid_micro_f1_at_best_C"] = valid_metrics["micro_f1"]
        metrics["valid_macro_f1_at_best_config"] = valid_metrics["macro_f1"]
        metrics["valid_exact_match_at_best_C"] = valid_metrics["exact_match"]

    return {
        "hyperparameters": {
            "rank": int(rank),
            "lambda_op": float(lambda_op),
            "lambda_delta": float(lambda_delta),
            "lr": float(lr),
        },
        "best_epoch": int(best_epoch),
        "history": history,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "diagnostics": {
            "train": train_diag,
            "valid": valid_diag,
            "test": test_diag,
        },
        "model_state_dict": best_state,
        "critical_checks": {
            "classifier_source": "Raw BGE-M3 OneVsRest LogisticRegression linear probe",
            "logits": "z_tilde @ W_raw + b_raw",
            "optimizer_parameters": "operator.parameters() only",
            "threshold": "logits >= 0.0, matching sklearn LogisticRegression predict decision threshold",
        },
    }


def lowrank_effective_rank(model_state_dict: Dict[str, torch.Tensor], tol: float = 1e-6) -> Dict[str, Any]:
    u = model_state_dict["operator.U"].detach().cpu().numpy()
    v = model_state_dict["operator.V"].detach().cpu().numpy()
    _, r_u = np.linalg.qr(u, mode="reduced")
    _, r_v = np.linalg.qr(v, mode="reduced")
    singular_values = np.linalg.svd(r_u @ r_v.T, compute_uv=False)
    threshold = float(tol * max(u.shape) * singular_values[0]) if singular_values.size else 0.0
    return {
        "matrix": "U @ V.T",
        "numeric_rank": int(np.sum(singular_values > threshold)),
        "threshold": threshold,
        "singular_values": [float(value) for value in singular_values[: min(32, len(singular_values))]],
    }


def run_lowrank_gtr_operator_experiment(
    train_z: np.ndarray,
    valid_z: np.ndarray,
    test_z: np.ndarray,
    descriptor_axis: np.ndarray,
    hard_negative_axis: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    w_raw: np.ndarray,
    b_raw: np.ndarray,
    raw_metrics: Dict[str, Any],
    classes: Sequence[str],
    output_path: Path,
    rank_grid: Sequence[int],
    lambda_op_grid: Sequence[float],
    lambda_delta_grid: Sequence[float],
    lr: float,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    model_dir: Path | None = None,
) -> Dict[str, Any]:
    train_dataset = LowRankGTRDataset(train_z, descriptor_axis, hard_negative_axis, y_train)
    valid_dataset = LowRankGTRDataset(valid_z, descriptor_axis, hard_negative_axis, y_valid)
    test_dataset = LowRankGTRDataset(test_z, descriptor_axis, hard_negative_axis, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Low-rank Residual GTR Operator ({device}) ===")
    print("Fixed probe check: logits = z_tilde @ W_raw + b_raw, optimizer = operator only.")

    best_result: Dict[str, Any] | None = None
    all_runs: List[Dict[str, Any]] = []
    for rank in rank_grid:
        for lambda_op in lambda_op_grid:
            for lambda_delta in lambda_delta_grid:
                result = train_one_lowrank_config(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    test_loader=test_loader,
                    w_raw=w_raw,
                    b_raw=b_raw,
                    d=int(train_dataset.z_norm.shape[1]),
                    g_dim=int(train_dataset.g.shape[1]),
                    rank=int(rank),
                    lambda_op=float(lambda_op),
                    lr=float(lr),
                    epochs=epochs,
                    patience=patience,
                    lambda_delta=float(lambda_delta),
                    seed=seed,
                    device=device,
                    model_dir=model_dir,
                )
                comparable_result = {k: v for k, v in result.items() if k != "model_state_dict"}
                all_runs.append(comparable_result)
                if best_result is None or result["valid_metrics"]["macro_f1"] > best_result[
                    "valid_metrics"
                ]["macro_f1"]:
                    best_result = result

    if best_result is None or best_result["model_state_dict"] is None:
        raise ValueError("Low-rank hyperparameter grids must contain at least one run.")

    best_params = best_result["hyperparameters"]
    best_model = FixedProbeLowRankGTRModel(
        w_raw=w_raw,
        b_raw=b_raw,
        d=int(train_dataset.z_norm.shape[1]),
        g_dim=int(train_dataset.g.shape[1]),
        rank=int(best_params["rank"]),
        lambda_op=float(best_params["lambda_op"]),
    ).to(device)
    best_model.load_state_dict(best_result["model_state_dict"])
    lowrank_pred_test, lowrank_scores = predict_lowrank_model_with_scores(
        best_model,
        test_loader,
        device,
    )
    save_prediction_payload(
        y_test,
        lowrank_pred_test,
        DEFAULT_LOWRANK_PREDICTION_PATH,
        classes,
        lowrank_scores,
    )
    print(f"Saved low-rank predictions to {DEFAULT_LOWRANK_PREDICTION_PATH.resolve()}")
    print("Saved low-rank scores")

    effective_rank = lowrank_effective_rank(best_result["model_state_dict"])
    lowrank_test_metrics = best_result["test_metrics"]
    comparison_table = [
        {
            "Representation": "Raw BGE-M3 + Linear Probe",
            "Micro-F1": raw_metrics["micro_f1"],
            "Macro-F1": raw_metrics["macro_f1"],
            "EM": raw_metrics["exact_match"],
            "Precision": raw_metrics["precision_micro"],
            "Recall": raw_metrics["recall_micro"],
        },
        {
            "Representation": "Low-rank Residual GTR Operator",
            "Micro-F1": lowrank_test_metrics["micro_f1"],
            "Macro-F1": lowrank_test_metrics["macro_f1"],
            "EM": lowrank_test_metrics["exact_match"],
            "Precision": lowrank_test_metrics["precision_micro"],
            "Recall": lowrank_test_metrics["recall_micro"],
        },
    ]

    payload = {
        "algorithm": "fixed raw probe: logits = (z_norm + lambda_op * delta_z) @ W_raw + b_raw",
        "selection_metric": "valid_macro_f1",
        "best_hyperparameters": best_result["hyperparameters"],
        "best_epoch": best_result["best_epoch"],
        "train_metrics": best_result["train_metrics"],
        "valid_metrics": best_result["valid_metrics"],
        "test_metrics": best_result["test_metrics"],
        "raw_probe_metrics": raw_metrics,
        "raw_vs_lowrank": {
            "valid_micro_f1_delta": float(
                best_result["valid_metrics"]["micro_f1"] - raw_metrics["valid_micro_f1_at_best_C"]
            ),
            "valid_macro_f1_delta": float(
                best_result["valid_metrics"]["macro_f1"] - raw_metrics["valid_macro_f1_at_best_C"]
            ),
            "valid_exact_match_delta": float(
                best_result["valid_metrics"]["exact_match"] - raw_metrics["valid_exact_match_at_best_C"]
            ),
            "test_micro_f1_delta": float(best_result["test_metrics"]["micro_f1"] - raw_metrics["micro_f1"]),
            "test_macro_f1_delta": float(best_result["test_metrics"]["macro_f1"] - raw_metrics["macro_f1"]),
            "test_exact_match_delta": float(
                best_result["test_metrics"]["exact_match"] - raw_metrics["exact_match"]
            ),
        },
        "comparison_table": comparison_table,
        "mean_alpha": best_result["diagnostics"]["test"]["mean_alpha"],
        "std_alpha": best_result["diagnostics"]["test"]["std_alpha"],
        "mean_delta_norm": best_result["diagnostics"]["test"]["mean_delta_norm"],
        "diagnostics": best_result["diagnostics"],
        "effective_rank_approximation": effective_rank,
        "all_runs": all_runs,
        "regularization": {
            "lambda_delta_grid": [float(value) for value in lambda_delta_grid],
        },
        "raw_probe": {
            "W_raw_shape": list(w_raw.shape),
            "b_raw_shape": list(b_raw.shape),
            "requires_grad": False,
            "threshold": "logits >= 0.0",
        },
        "critical_checks": best_result["critical_checks"],
        "shapes": {
            "z_tilde": [None, int(train_dataset.z_norm.shape[1])],
            "concat_g": [None, int(train_dataset.g.shape[1])],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved low-rank GTR operator results to {output_path.resolve()}")
    return payload


def run_raw_and_lowrank_eval(args: argparse.Namespace) -> Dict[str, Any]:
    train_df, valid_df, test_df = load_lbox_splits(args.train_path, args.valid_path, args.test_path)
    model, train_emb, valid_emb, test_emb = encode_cases(
        args.model_name,
        train_df,
        valid_df,
        test_df,
        args.batch_size,
    )

    all_statutes = collect_unique_statutes(train_df, valid_df, test_df)
    mlb = MultiLabelBinarizer(classes=all_statutes)
    y_train = mlb.fit_transform(train_df["label"].tolist())
    y_valid = mlb.transform(valid_df["label"].tolist())
    y_test = mlb.transform(test_df["label"].tolist())

    descriptor_axis_statutes, descriptor_axis_matrix, descriptor_records = build_and_save_descriptor_axis(
        train_df=train_df,
        statutes=all_statutes,
        model=model,
        batch_size=args.batch_size,
        descriptor_path=args.descriptor_path,
        descriptor_axis_path=args.descriptor_axis_path,
    )
    descriptor_axis_matrix = reorder_axis_to_classes(
        descriptor_axis_statutes,
        descriptor_axis_matrix,
        mlb.classes_,
    )
    descriptor_axis_matrix = l2_normalize(descriptor_axis_matrix)

    hard_negative_axis_matrix, hard_negative_records = build_and_save_hard_negative_axis(
        train_df=train_df,
        model=model,
        batch_size=args.batch_size,
        top_k=args.hard_negative_top_k,
        descriptor_path=args.hard_negative_descriptor_path,
        axis_path=args.hard_negative_axis_path,
    )
    hard_negative_axis_matrix = l2_normalize(hard_negative_axis_matrix)

    raw_metrics, pred_raw_test, raw_probe_clf = fit_raw_linear_probe(
        train_emb,
        valid_emb,
        test_emb,
        y_train,
        y_valid,
        y_test,
        c_grid=args.c_grid,
        model_dir=DEFAULT_MODEL_CACHE_DIR / "linear_probe",
    )
    raw_scores = predict_raw_scores(raw_probe_clf, test_emb)
    save_prediction_payload(
        y_test,
        pred_raw_test,
        DEFAULT_RAW_PREDICTION_PATH,
        mlb.classes_,
        raw_scores,
    )
    print(f"Saved raw predictions to {DEFAULT_RAW_PREDICTION_PATH.resolve()}")
    print("Saved raw scores")

    w_raw, b_raw = extract_ovr_linear_probe_weights(raw_probe_clf)
    lowrank_results = run_lowrank_gtr_operator_experiment(
        train_z=train_emb,
        valid_z=valid_emb,
        test_z=test_emb,
        descriptor_axis=descriptor_axis_matrix,
        hard_negative_axis=hard_negative_axis_matrix,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        w_raw=w_raw,
        b_raw=b_raw,
        raw_metrics=raw_metrics,
        classes=mlb.classes_,
        output_path=args.lowrank_output_path,
        rank_grid=args.lowrank_rank_grid,
        lambda_op_grid=args.lowrank_lambda_op_grid,
        lambda_delta_grid=args.lowrank_lambda_delta_grid,
        lr=args.lowrank_lr,
        epochs=args.lowrank_epochs,
        patience=args.lowrank_patience,
        batch_size=args.lowrank_batch_size,
        seed=args.lowrank_seed,
        model_dir=DEFAULT_MODEL_CACHE_DIR / "lowrank_gtr",
    )

    payload = {
        "embedding_model": args.model_name,
        "raw_embedding_shape": {
            "train": list(train_emb.shape),
            "valid": list(valid_emb.shape),
            "test": list(test_emb.shape),
        },
        "descriptor_path": str(args.descriptor_path.resolve()),
        "descriptor_axis_path": str(args.descriptor_axis_path.resolve()),
        "descriptor_axis_shape": list(descriptor_axis_matrix.shape),
        "hard_negative_descriptor_path": str(args.hard_negative_descriptor_path.resolve()),
        "hard_negative_axis_path": str(args.hard_negative_axis_path.resolve()),
        "hard_negative_axis_shape": list(hard_negative_axis_matrix.shape),
        "hard_negative_top_k": int(args.hard_negative_top_k),
        "hard_negative_pair_count": int(len(hard_negative_records)),
        "n_descriptor_fallbacks": int(sum(1 for record in descriptor_records if record["fallback"])),
        "raw_probe_metrics": raw_metrics,
        "lowrank_gtr_operator_output_path": str(args.lowrank_output_path.resolve()),
        "lowrank_gtr_operator": lowrank_results,
        "n_labels": int(y_train.shape[1]),
        "classes": [str(label) for label in mlb.classes_],
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw embedding and low-rank residual GTR eval for LBOX.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--descriptor-path", type=Path, default=DEFAULT_DESCRIPTOR_PATH)
    parser.add_argument("--descriptor-axis-path", type=Path, default=DEFAULT_DESCRIPTOR_AXIS_PATH)
    parser.add_argument(
        "--hard-negative-descriptor-path",
        type=Path,
        default=DEFAULT_HARD_NEGATIVE_DESCRIPTOR_PATH,
    )
    parser.add_argument("--hard-negative-axis-path", type=Path, default=DEFAULT_HARD_NEGATIVE_AXIS_PATH)
    parser.add_argument("--hard-negative-top-k", type=int, default=DEFAULT_HARD_NEGATIVE_TOP_K)
    parser.add_argument("--lowrank-output-path", type=Path, default=DEFAULT_LOWRANK_OUTPUT_PATH)
    parser.add_argument(
        "--lowrank-rank-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_LOWRANK_RANK_GRID),
        help="Rank values for the low-rank residual GTR operator.",
    )
    parser.add_argument(
        "--lowrank-lambda-op-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LOWRANK_LAMBDA_OP_GRID),
        help="lambda_op values for z_tilde = z_norm + lambda_op * delta_z.",
    )
    parser.add_argument("--lowrank-lr", type=float, default=DEFAULT_LOWRANK_LR)
    parser.add_argument("--lowrank-epochs", type=int, default=DEFAULT_LOWRANK_EPOCHS)
    parser.add_argument("--lowrank-patience", type=int, default=DEFAULT_LOWRANK_PATIENCE)
    parser.add_argument("--lowrank-batch-size", type=int, default=DEFAULT_LOWRANK_BATCH_SIZE)
    parser.add_argument(
        "--lowrank-lambda-delta-grid",
        type=float,
        nargs="+",
        default=list(DEFAULT_LOWRANK_LAMBDA_DELTA_GRID),
    )
    parser.add_argument("--lowrank-seed", type=int, default=DEFAULT_LOWRANK_SEED)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_raw_and_lowrank_eval(args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {args.output_path.resolve()}")


if __name__ == "__main__":
    main()
