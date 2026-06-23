# Paper Claims

This file maps the intended paper narrative to the code and artifacts in this repository.

## One-Sentence Story

GTR turns legal criteria into calibrated coordinates, uses statute-specific decision fields to refine candidate statutes, suppresses false positives, improves hard-negative separation, and gives LLMs a cleaner routing context.

## 1. Axis Validity

Claim: the axes measure primitive legal concepts rather than arbitrary latent dimensions.

Evidence:

- `cail_gtr_axis_schema.py`: Chinese criminal-law axis schema.
- `build_cail_axis_labels.py`: converts legal-element definitions into axis supervision.
- `output/supporting_claims/proposal_axis_validity_report.json`: supporting axis-validity report.

Interpretation: the model is supervised to align coordinates with legal concepts such as violence, deception, public-safety danger, traffic driving, drugs, and public-official status.

## 2. Coordinate Calibration

Claim: calibrated coordinates are more stable and interpretable than raw projection scores.

Evidence:

- `output/supporting_claims/coordinate_calibration_report.json`
- `output/supporting_claims/primitive_seeded_calibration_report.json`

Interpretation: raw `A^T z` projections are useful features, but calibrated `A^\dagger z` and `q` are closer to legal criterion satisfaction scores.

## 3. Hybrid Superiority

Claim: Hybrid GTR improves over raw-only prediction.

Evidence:

- `output/supporting_claims/hybrid_gtr_v2_ablation_report.json`
- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`

Key supporting numbers:

- `raw_head_on_z`: test micro-F1 `0.7724`, exact match `0.4493`.
- `full_hybrid`: test micro-F1 `0.8613`, exact match `0.7138`.

## 4. Interaction Necessity

Claim: statute-specific decision fields are needed; additive coordinates alone are not enough.

Evidence:

- `train_hybrid_gtr_v2.py`: shared model implementation.
- `train_cail_hybrid_gtr_v2.py`: CAIL trainer using raw, field, and residual components.
- `output/supporting_claims/hybrid_gtr_v2_ablation_report.json`: raw-only, field-only, additive, residual, and full-hybrid variants.

Interpretation: the same legal criterion can matter differently by statute, so the model needs statute-specific interactions.

## 5. Suppression

Claim: GTR selectively removes false positives.

Evidence:

- `output/supporting_claims/hybrid_gtr_v2_mechanism_report.json`

Key supporting numbers:

- `raw_plus_field_no_residual_vs_raw` removed `145` predictions.
- `122` of those removed predictions were false positives.
- Suppression precision: `0.8414`.

## 6. Hard-Negative Margin

Claim: GTR improves margins for confusing statute pairs.

Evidence:

- `output/supporting_claims/hard_negative_margin_report.json`

Aggregate signal:

- `raw_head_on_z_mean_margin`: `5.2277`
- `full_hybrid_mean_margin`: `5.3782`
- `raw_head_on_z_confuser_fp_rate`: `0.1192`
- `full_hybrid_confuser_fp_rate`: `0.0155`

## 7. LLM Routing

Claim: GTR candidate refinement improves downstream LLM legal judgment.

Evidence:

- `build_cail_gtr_v2_llm_prompts.py`
- `run_openai_prompt_jsonl.py`
- `evaluate_llm_candidate_routing_results.py`
- `output/cail2018_gtr_v2_only/full/llm_prompts/`
- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json`

Condition roles:

| Condition | Role |
| --- | --- |
| `full_fact + LLM` | LLM-only baseline |
| `raw_topk + LLM` | retrieval baseline |
| `gtr_topk + LLM` | direct GTR router ablation |
| `raw_pool_gtr_rerank + LLM` | best router |
| `raw_pool_gtr_rerank_score_prompt + LLM` | strongest setting |

Saved GPT-5.4 result:

- `raw_pool_gtr_rerank`: exact match `0.7428`, micro-F1 `0.7987`.
- `raw_pool_gtr_rerank_score_prompt`: exact match `0.7609`, micro-F1 `0.8039`.

Note: the saved CAIL prompt artifact contains `full_fact`, `raw_topk`, `raw_pool_gtr_rerank`, and `raw_pool_gtr_rerank_score_prompt`. `gtr_topk` is listed as the direct-router ablation for the paper scenario.

