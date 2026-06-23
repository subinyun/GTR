# GTR

This repository contains the PI-facing reproducibility package for the GTR experiments before paper submission.

Start here:

- `pi_gtr_v2_repro/docs/claim_map.md`: seven-claim paper narrative and evidence map.
- `pi_gtr_v2_repro/README.md`: review path and quick reproduction commands.
- `pi_gtr_v2_repro/docs/reproducibility.md`: full rerun instructions.
- `pi_gtr_v2_repro/docs/file_map.md`: source-file map for the final GTR path.
- `pi_gtr_v2_repro/results/expected_metrics.json`: expected metrics for comparison.

The repository is organized around seven claims: axis validity, coordinate calibration, hybrid superiority, interaction necessity, selective suppression, hard-negative margin improvement, and LLM routing improvement. It includes the core source files, wrapper scripts, supporting report artifacts, the trained GTR checkpoint, generated prompt artifacts, saved GPT-5.4 results, compact metrics, the full `LBOX/statute_classification/` splits, and a small CAIL label-vocabulary placeholder at `final_all_data/cail2018_statute_classification/train.jsonl` for evaluator-only reproduction. It intentionally excludes the full CAIL2018 split files and embedding cache because they are too large for a normal GitHub push.

For a no-API sanity check:

```bash
bash pi_gtr_v2_repro/scripts/08_recompute_saved_gpt_metrics.sh
```

For full CAIL retraining, place the CAIL2018 split files under `final_all_data/cail2018_statute_classification/` as described in `pi_gtr_v2_repro/docs/reproducibility.md`. The LBOX statute-classification splits are already included.

