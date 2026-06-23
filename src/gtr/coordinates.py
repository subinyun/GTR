"""Coordinate helpers for GTR experiments."""

from __future__ import annotations

from typing import Iterable, Sequence


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    return float(sum(x * y for x, y in zip(a, b)))


def project(axis: Sequence[float], z: Sequence[float]) -> float:
    """Compute an A^T z-style coordinate for one axis."""

    return dot(axis, z)


def project_many(axes: Iterable[Sequence[float]], z: Sequence[float]) -> list[float]:
    """Compute raw coordinates for multiple axes."""

    return [project(axis, z) for axis in axes]


def calibrate_logistic(raw: float, alpha: float, beta: float) -> float:
    """Map a raw coordinate to a calibrated q score."""

    import math

    return float(1.0 / (1.0 + math.exp(-(alpha * raw + beta))))
