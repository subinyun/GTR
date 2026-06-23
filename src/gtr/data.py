"""Data path helpers for the GTR reproducibility package."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LBOX_SPLIT_DIR = REPO_ROOT / "LBOX/statute_classification"
CAIL_SPLIT_DIR = REPO_ROOT / "final_all_data/cail2018_statute_classification"


def lbox_split_path(split: str) -> Path:
    return LBOX_SPLIT_DIR / f"{split}.jsonl"


def cail_split_path(split: str) -> Path:
    return CAIL_SPLIT_DIR / f"{split}.jsonl"
