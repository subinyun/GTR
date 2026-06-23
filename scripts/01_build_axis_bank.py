#!/usr/bin/env python3
"""Build a review-facing axis-bank summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cail_gtr_axis_schema import AXIS_SCHEMA
from src.gtr.provenance import build_provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data/lbox.yaml")
    parser.add_argument("--output", default="outputs/metrics/axis_bank_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": build_provenance(args.config, seed=42),
        "config": args.config,
        "num_axes": len(AXIS_SCHEMA),
        "axis_ids": [axis["axis_id"] for axis in AXIS_SCHEMA],
        "source": "cail_gtr_axis_schema.py",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
