"""Axis-bank helpers."""

from __future__ import annotations

from typing import Any


def load_cail_axis_schema() -> list[dict[str, Any]]:
    from cail_gtr_axis_schema import AXIS_SCHEMA

    return list(AXIS_SCHEMA)
