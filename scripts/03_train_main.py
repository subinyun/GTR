#!/usr/bin/env python3
"""Wrapper for retraining the main CAIL GTR checkpoint.

This delegates to `repro/scripts/01_train_cail_gtr.sh`. Full CAIL split files
must be present before running this script.
"""

from __future__ import annotations

import subprocess


subprocess.run(["bash", "repro/scripts/01_train_cail_gtr.sh"], check=True)
