#!/usr/bin/env python3
"""Export threshold-vs-GTR metrics for Table 3."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.gtr.provenance import build_provenance


src = Path("artifacts/supporting_claims/threshold_vs_gtr_report.json")
dst = Path("outputs/metrics/threshold_vs_gtr.json")
dst.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "provenance": build_provenance("configs/exp/threshold_vs_gtr.yaml", seed=42),
    "source": str(src),
    "report": json.loads(src.read_text(encoding="utf-8")),
}
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {dst}")
