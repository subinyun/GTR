"""Model interfaces used by the review-facing experiment wrappers.

These lightweight classes make the model taxonomy explicit without changing the
legacy training scripts that produced the included artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


Vector = Mapping[str, float]


class BaseGTRModel(Protocol):
    """Common interface for ablations."""

    name: str

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        """Return statute scores for one case representation."""


@dataclass(frozen=True)
class RawModel:
    """Raw-only baseline: w^T z."""

    name: str = "raw_only"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        return dict(z)


@dataclass(frozen=True)
class CoordinateModel:
    """Coordinate-only model: statute decisions from q."""

    name: str = "coordinate_only"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        return dict(q or {})


@dataclass(frozen=True)
class ConcatModel:
    """Concatenation baseline: [z; q]."""

    name: str = "concat_z_q"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        merged = dict(z)
        for key, value in (q or {}).items():
            merged[f"coord::{key}"] = value
        return merged


@dataclass(frozen=True)
class ResidualGTRModel:
    """Residual coordinate correction: z' = z + B h(q)."""

    name: str = "residual_gtr"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        # The trained residual operator lives in the legacy checkpoint. This
        # interface keeps the result ablation taxonomy explicit.
        return dict(z)


@dataclass(frozen=True)
class DecisionFieldModel:
    """Statute-specific decision field F_g(q)."""

    name: str = "decision_field"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        return dict(q or {})


@dataclass(frozen=True)
class HybridGTRModel:
    """Hybrid score: w^T z' + F_g(q)."""

    name: str = "hybrid_gtr"

    def score(self, z: Vector, q: Vector | None = None) -> Vector:
        scores = dict(z)
        for key, value in (q or {}).items():
            scores[key] = scores.get(key, 0.0) + value
        return scores
