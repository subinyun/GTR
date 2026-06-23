"""Paper-facing GTR interfaces.

The current reproducibility scripts reuse the original experiment files at the
repository root. This package documents the stable interfaces used to organize
the paper experiments.
"""

from .model import (
    BaseGTRModel,
    ConcatModel,
    CoordinateModel,
    DecisionFieldModel,
    HybridGTRModel,
    RawModel,
    ResidualGTRModel,
)

__all__ = [
    "BaseGTRModel",
    "RawModel",
    "CoordinateModel",
    "ConcatModel",
    "ResidualGTRModel",
    "DecisionFieldModel",
    "HybridGTRModel",
]
