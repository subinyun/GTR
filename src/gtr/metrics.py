"""Small metric helpers used by tests and result table scripts."""

from __future__ import annotations

from typing import Iterable, Sequence


def exact_match(gold: Sequence[Iterable[str]], pred: Sequence[Iterable[str]]) -> float:
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have the same length")
    if not gold:
        return 0.0
    matches = [set(g) == set(p) for g, p in zip(gold, pred)]
    return float(sum(matches) / len(matches))


def label_space(rows: Iterable[dict]) -> set[str]:
    labels: set[str] = set()
    for row in rows:
        labels.update(str(label) for label in row.get("statutes", []))
    return labels
