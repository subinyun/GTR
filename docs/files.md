# Files

This is the repository map.

## Top-Level Source Files

Core GTR training:

- `train_cail_hybrid_gtr_v2.py`: CAIL GTR training script.
- `train_hybrid_gtr_v2.py`: shared GTR model utilities.
- `build_cail_axis_labels.py`: CAIL axis-label construction.
- `cail_gtr_axis_schema.py`: 20 Chinese criminal-law axes.
- `lbox_raw_lowrank_eval.py`: embedding, normalization, and raw-probe utilities.

LLM routing:

- `build_cail_gtr_v2_llm_prompts.py`: builds CAIL LLM prompt rows.
- `run_openai_prompt_jsonl.py`: runs OpenAI models on prompt JSONL files.
- `run_gemini_prompt_jsonl.py`: runs Gemini models on prompt JSONL files.
- `evaluate_llm_candidate_routing_results.py`: evaluates LLM statute predictions.
- `evaluate_gtr_llm_decoder.py`: shared statute normalization and metric code.

Qwen LoRA:

- `build_cail_qwen_lora_sft_data.py`: builds Qwen instruction-tuning data.
- `train_qwen_completion_lora.py`: trains Qwen3-8B LoRA adapters.
- `run_qwen_completion_eval.py`: runs Qwen adapter predictions.
- `evaluate_qwen_completion_results.py`: evaluates Qwen outputs.

## Result-Facing Structure

- `configs/data/lbox.yaml`: included LBOX split paths.
- `configs/model/raw.yaml`: raw-only baseline model config.
- `configs/model/gtr_v2_hybrid.yaml`: Hybrid GTR model config.
- `configs/exp/main_table.yaml`: main result table inputs and outputs.
- `configs/exp/ablation.yaml`: ablation, suppression, and hard-negative inputs.
- `configs/exp/threshold_vs_gtr.yaml`: threshold-vs-GTR comparison.
- `configs/exp/llm_routing.yaml`: LLM routing condition map.
- `src/gtr/`: small stable interfaces for models, coordinates, metrics, and routing condition names.
- `scripts/`: result table/claim wrappers.
- `experiments/`: named experiment buckets matching result sections.
- `outputs/`: normalized metrics, predictions, figures, logs, and copied configs.

## Reproduction Helpers

- `repro/scripts/`: wrapper scripts for environment checks, prompt generation, GPT reruns, Qwen LoRA, and saved-result metric recomputation.
- `repro/expected_metrics.json`: machine-readable expected metrics.
- `repro/requirements-gtr.txt`: package list for the reproduction path.

## Data

- `LBOX/statute_classification/`: full LBOX statute-classification splits.
- `final_all_data/cail2018_statute_classification/train.jsonl`: CAIL label-vocabulary placeholder for evaluator-only reproduction, not the full CAIL train split.

## Outputs

CAIL GTR:

- `artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt`: trained checkpoint included for prompt rebuilding.
- `artifacts/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`: checkpoint report.

LLM routing:

- `artifacts/cail2018_gtr_v2_only/full/llm_prompts/`: saved prompt artifacts.
- `artifacts/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/`: saved GPT-5.4 results and metrics.

Qwen LoRA:

- `artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json`
- `artifacts/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/smoke/summary_metrics.json`

Result-claim support:

- `artifacts/supporting_claims/proposal_axis_validity_report.json`
- `artifacts/supporting_claims/coordinate_calibration_report.json`
- `artifacts/supporting_claims/primitive_seeded_calibration_report.json`
- `artifacts/supporting_claims/hybrid_gtr_v2_ablation_report.json`
- `artifacts/supporting_claims/hybrid_gtr_v2_mechanism_report.json`
- `artifacts/supporting_claims/hard_negative_margin_report.json`
- `artifacts/supporting_claims/threshold_vs_gtr_report.json`

## Output Policy

Commit small review artifacts:

- `outputs/metrics/*.json`
- `outputs/metrics/result_tables.md`

Do not commit generated large or sensitive artifacts:

- `outputs/checkpoints/`
- `outputs/predictions/*.jsonl`
- `outputs/logs/`
- full CAIL raw data
- embeddings and caches
- API outputs that may contain sensitive text

