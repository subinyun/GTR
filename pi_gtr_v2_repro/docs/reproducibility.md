# Reproducibility Guide

All commands below should be run from the repository root.

## 0. Environment Check

```bash
bash pi_gtr_v2_repro/scripts/00_check_environment.sh
```

This checks source files, available data files, optional API keys, and core Python packages.

Recommended environment:

- Python 3.10 or newer
- CUDA-capable GPU for training the GTR model and Qwen LoRA adapters
- `torch`, `numpy`, `pandas`, `scikit-learn`, `sentence-transformers`
- `openai` if rerunning GPT-5.4
- `google-genai` or compatible Gemini SDK if rerunning Gemini
- `transformers`, `peft`, `accelerate`, `bitsandbytes` for Qwen LoRA

The GitHub package includes the full `LBOX/statute_classification/` split files. It intentionally excludes the full CAIL2018 split files and the embedding cache. For CAIL, it includes a small `train.jsonl` label-vocabulary placeholder so the saved-result evaluator can run without the full dataset. Full CAIL retraining requires replacing/adding the real local split files under `final_all_data/cail2018_statute_classification/`:

- `train.jsonl`
- `valid.jsonl`
- `test.jsonl`

The BGE embedding cache under `output/cail2018_gtr_v2_only/full/cache/` can be regenerated.

Included LBOX files:

- `LBOX/statute_classification/train.jsonl`
- `LBOX/statute_classification/valid.jsonl`
- `LBOX/statute_classification/test.jsonl`
- `LBOX/statute_classification/test2.jsonl`
- `LBOX/statute_classification/plus_train.jsonl`
- `LBOX/statute_classification/plus_valid.jsonl`
- `LBOX/statute_classification/plus_test.jsonl`

## 1. Rebuild The GTR Checkpoint

```bash
DEVICE=cuda ENCODER_DEVICE=cuda bash pi_gtr_v2_repro/scripts/01_train_cail_gtr.sh
```

Main outputs:

- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt`
- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`
- `output/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz`

The run uses seed `42` and the full CAIL2018 train/valid/test splits under `final_all_data/cail2018_statute_classification/`.

## 2. Rebuild LLM Prompts

```bash
DEVICE=cpu bash pi_gtr_v2_repro/scripts/02_build_llm_prompts.sh
```

This creates 276 sampled test cases and four prompt conditions:

- `full_fact`
- `raw_topk`
- `raw_pool_gtr_rerank`
- `raw_pool_gtr_rerank_score_prompt`

Main outputs:

- `output/cail2018_gtr_v2_only/full/llm_prompts/prompts_276.jsonl`
- `output/cail2018_gtr_v2_only/full/llm_prompts/gpt/prompts_276.jsonl`
- `output/cail2018_gtr_v2_only/full/llm_prompts/gemini/prompts_276.jsonl`

## 3. Rerun GPT-5.4 Decoding And Evaluation

```bash
export OPENAI_API_KEY=...
bash pi_gtr_v2_repro/scripts/03_run_gpt54_rerank_eval.sh
```

This runs only the two final GTR-reranked GPT conditions:

- `raw_pool_gtr_rerank`
- `raw_pool_gtr_rerank_score_prompt`

Main outputs:

- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/results_gpt54_rerank_276.jsonl`
- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json`

For a no-API check, recompute metrics from the saved result file:

```bash
bash pi_gtr_v2_repro/scripts/08_recompute_saved_gpt_metrics.sh
```

Output:

- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/recomputed/metrics_gpt54_rerank_276.json`

## 4. Optional Gemini Rerun

```bash
export GOOGLE_API_KEY=...
MODEL=gemini-3.5-flash bash pi_gtr_v2_repro/scripts/04_run_gemini_all4_eval.sh
```

This reruns all four prompt conditions. Hosted Gemini behavior may differ over time, so treat this mainly as a decoding/evaluation reproducibility check.

## 5. Rebuild Qwen LoRA Data

```bash
DEVICE=cpu bash pi_gtr_v2_repro/scripts/05_build_qwen_lora_data.sh
```

This builds the full Qwen3-8B LoRA SFT data under:

- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data_full/`

## 6. Qwen LoRA Smoke Run

```bash
GPU=0 bash pi_gtr_v2_repro/scripts/06_run_qwen_lora_smoke.sh
```

This is the faster sanity check. It trains for 200 steps per condition.

## 7. Qwen LoRA Full Run

```bash
GPU=0 EPOCHS=1 LR=1e-4 bash pi_gtr_v2_repro/scripts/07_run_qwen_lora_full.sh
```

Main output:

- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json`

## Reproducibility Caveats

Metrics from saved result files should reproduce exactly when rerunning only the evaluator. End-to-end retraining or hosted LLM reruns can differ slightly because of GPU kernels, package versions, model-server updates, API-side sampling behavior, and checkpoint selection.

