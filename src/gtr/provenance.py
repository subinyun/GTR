"""Provenance helpers for reproducible paper artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

IGNORED_DIRTY_PREFIXES = (
    "outputs/metrics/",
    "outputs/configs/",
    "outputs/logs/",
    "outputs/predictions/",
)


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def config_hash(config_path: str | Path | None) -> str | None:
    if not config_path:
        return None
    path = Path(config_path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _status_path(line: str) -> str:
    # Porcelain format is usually "XY path"; rename lines can be
    # "R  old -> new", where the new path is what matters for generated files.
    path = line[3:] if len(line) > 3 else ""
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip()


def source_tree_dirty() -> bool:
    status = _git(["status", "--porcelain"])
    if not status:
        return False
    for line in status.splitlines():
        path = _status_path(line)
        if not any(path.startswith(prefix) for prefix in IGNORED_DIRTY_PREFIXES):
            return True
    return False


def build_provenance(config_path: str | Path | None, *, seed: int = 42) -> dict[str, Any]:
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": source_tree_dirty(),
        "config_path": str(config_path) if config_path else None,
        "config_hash": config_hash(config_path),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
