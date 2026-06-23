# Reproduce

Run commands from the repository root.

## Artifact Reproduction

These are the primary review-facing commands:

```bash
python scripts/01_build_axis_bank.py --config configs/data/lbox.yaml
python scripts/run_all_main_experiments.py --config configs/exp/main_table.yaml
python scripts/make_result_tables.py --input outputs/metrics
```

Outputs:

- `outputs/metrics/axis_bank_summary.json`
- `outputs/metrics/main_table.json`
- `outputs/metrics/result_tables.md`

Each generated metrics JSON stores the git commit, dirty status, config path, config hash, seed, and timestamp under `provenance`.

## Environment Check

```bash
bash repro/scripts/00_check_environment.sh
```

This checks source files, included LBOX data, the CAIL label-vocabulary placeholder, optional API keys, and core Python packages.

## No-API Metric Check

This is the most stable reproducibility check. It recomputes metrics from the saved GPT-5.4 result file.

```bash
bash repro/scripts/08_recompute_saved_gpt_metrics.sh
```

Expected output:

- `raw_pool_gtr_rerank`: exact match `0.7428`, micro-F1 `0.7987`.
- `raw_pool_gtr_rerank_score_prompt`: exact match `0.7609`, micro-F1 `0.8039`.

## Rerun GPT-5.4 Decoding

Requires an OpenAI API key.

```bash
export OPENAI_API_KEY=...
bash repro/scripts/03_run_gpt54_rerank_eval.sh
```

This reruns the two final GTR-reranked prompt conditions:

- `raw_pool_gtr_rerank`
- `raw_pool_gtr_rerank_score_prompt`

## Rebuild CAIL Prompts

The repository includes the trained GTR checkpoint and saved prompt artifacts. To rebuild prompts:

```bash
DEVICE=cpu bash repro/scripts/02_build_llm_prompts.sh
```

Full CAIL retraining requires the real CAIL split files at:

- `final_all_data/cail2018_statute_classification/train.jsonl`
- `final_all_data/cail2018_statute_classification/valid.jsonl`
- `final_all_data/cail2018_statute_classification/test.jsonl`

The GitHub repo includes only a small CAIL `train.jsonl` label-vocabulary placeholder because the full CAIL train split is too large for a normal GitHub push.

## Rebuild The CAIL GTR Checkpoint

Requires the full CAIL split files and a CUDA-capable GPU.

```bash
DEVICE=cuda ENCODER_DEVICE=cuda bash repro/scripts/01_train_cail_gtr.sh
```

## Qwen LoRA Path

Build Qwen LoRA data:

```bash
DEVICE=cpu bash repro/scripts/05_build_qwen_lora_data.sh
```

Smoke run:

```bash
GPU=0 bash repro/scripts/06_run_qwen_lora_smoke.sh
```

Full run:

```bash
GPU=0 EPOCHS=1 LR=1e-4 bash repro/scripts/07_run_qwen_lora_full.sh
```

## Data Included

Included:

- Full `LBOX/statute_classification/` splits.
- Trained CAIL GTR checkpoint.
- Saved CAIL prompt artifacts.
- Saved GPT-5.4 results and metrics.
- Supporting claim reports.

Excluded:

- Full CAIL2018 split files.
- BGE embedding cache under `artifacts/cail2018_gtr_v2_only/full/cache/`.
- Large training caches and generated adapter checkpoints.

