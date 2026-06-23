"""Encoder interface placeholder for review-facing organization."""

from __future__ import annotations

from typing import Protocol, Sequence


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode case facts or statute text into dense vectors."""
