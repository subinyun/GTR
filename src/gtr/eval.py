"""Evaluation entrypoint vocabulary."""

from __future__ import annotations


EVAL_ENTRYPOINTS = {
    "saved_gpt54_metrics": "repro/scripts/08_recompute_saved_gpt_metrics.sh",
    "main_table": "scripts/run_all_main_experiments.py",
    "result_tables": "scripts/make_result_tables.py",
}
