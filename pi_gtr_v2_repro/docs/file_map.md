# File Map

This document separates the final GTR path from exploratory or older experiments.

## Final CAIL2018 GTR Path

Core model training:

- `train_cail_hybrid_gtr_v2.py`: trains the CAIL2018 GTR model with Chinese legal-element axis supervision.
- `build_cail_axis_labels.py`: builds axis supervision labels from the CAIL2018 statute facts.
- `cail_gtr_axis_schema.py`: defines the 20 Chinese criminal-law axis schema.
- `train_hybrid_gtr_v2.py`: shared GTR model utilities used by the CAIL trainer.
- `lbox_raw_lowrank_eval.py`: shared embedding, normalization, and raw-probe utilities.

Prompt construction and LLM evaluation:

- `build_cail_gtr_v2_llm_prompts.py`: builds four prompt conditions from the trained checkpoint.
- `run_openai_prompt_jsonl.py`: runs OpenAI models on prompt JSONL files.
- `run_gemini_prompt_jsonl.py`: runs Gemini models on prompt JSONL files.
- `evaluate_llm_candidate_routing_results.py`: computes exact match, micro-F1, macro-F1, precision, recall, parse rate, and invalid-label counts.
- `evaluate_gtr_llm_decoder.py`: shared statute normalization and metric utilities.

Qwen LoRA decoder:

- `build_cail_qwen_lora_sft_data.py`: builds train/valid/test instruction data using the same GTR candidate conditions.
- `train_qwen_completion_lora.py`: trains Qwen3-8B LoRA adapters.
- `run_qwen_completion_eval.py`: runs a trained adapter on test prompts.
- `evaluate_qwen_completion_results.py`: evaluates Qwen completion outputs.
- `run_cail_qwen3_8b_lora_smoke.sh`: short 200-step smoke run.
- `run_cail_qwen3_8b_lora_full.sh`: full Qwen3-8B LoRA run.

## Main Outputs

GTR model:

- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_best.pt`
- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`
- `output/cail2018_gtr_v2_only/full/cache/bge_m3_cail_embeddings_random_trfull_vafull_tefull_seed42.npz`

Prompt artifacts:

- `output/cail2018_gtr_v2_only/full/llm_prompts/prompts_276.jsonl`
- `output/cail2018_gtr_v2_only/full/llm_prompts/prompts_276_summary.json`
- `output/cail2018_gtr_v2_only/full/llm_prompts/gpt/prompts_276.jsonl`
- `output/cail2018_gtr_v2_only/full/llm_prompts/gemini/prompts_276.jsonl`

GPT-5.4 artifacts:

- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/results_gpt54_rerank_276.jsonl`
- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json`
- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.csv`

Qwen3-8B LoRA artifacts:

- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/data_full/`
- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json`
- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/smoke/summary_metrics.json`

Supporting claim artifacts:

- `output/supporting_claims/proposal_axis_validity_report.json`: axis validity.
- `output/supporting_claims/coordinate_calibration_report.json`: coordinate calibration.
- `output/supporting_claims/primitive_seeded_calibration_report.json`: primitive-seeded coordinate calibration.
- `output/supporting_claims/hybrid_gtr_v2_ablation_report.json`: hybrid superiority and interaction necessity.
- `output/supporting_claims/hybrid_gtr_v2_mechanism_report.json`: selective false-positive suppression.
- `output/supporting_claims/hard_negative_margin_report.json`: hard-negative margin improvement.

## Descriptor And Axis Exploration

The `gtr_final/` folder contains the later descriptor and axis-construction work. It is useful for explaining the broader GTR design, but it is not the shortest path for rerunning the final CAIL2018 reranking metrics.

Important files there include:

- `gtr_final/frozen_descriptor_pipeline.py`: train-only frozen LBOX descriptor construction.
- `gtr_final/build_axis_grid_primitive_seeded.py`: primitive-seeded axis-grid construction.
- `gtr_final/build_axis_grid_debiased.py`: debiased axis-grid construction.
- `gtr_final/audit_coordinate_geometry.py`: coordinate sanity/audit tooling.

