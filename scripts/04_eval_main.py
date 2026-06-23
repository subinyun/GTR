#!/usr/bin/env python3
"""Assemble main-table evaluation artifacts from included reports."""

from __future__ import annotations

import subprocess


subprocess.run(["python", "scripts/run_all_main_experiments.py"], check=True)
