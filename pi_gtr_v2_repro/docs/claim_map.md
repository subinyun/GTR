# Paper Claim Map

This document organizes the repository around the intended paper narrative. The goal is to let a reviewer read the code and artifacts by claim, rather than by experiment timestamp.

## Central Story

GTR is a legally structured routing layer. It does not only retrieve candidate statutes; it builds calibrated legal coordinates, applies statute-specific decision fields, suppresses false positives, improves hard-negative margins, and then gives the LLM a cleaner legal candidate set.

## Claim 1. Axis Validity

Claim: the axes measure primitive legal concepts rather than arbitrary latent dimensions.

Evidence in this repository:

- `cail_gtr_axis_schema.py`: Chinese criminal-law axis schema used for CAIL experiments.
- `build_cail_axis_labels.py`: converts legal-element definitions into axis supervision.
- `output/supporting_claims/proposal_axis_validity_report.json`: supporting axis-validity report from earlier proposal experiments.

How to read it:

- The axis schema defines primitive legal concepts such as personal violence, deception/fraud, public-safety danger, traffic driving, drugs, and public-official status.
- Axis labels are used as supervision, so the model is asked to align coordinates with legal concepts rather than only optimizing final statute classification.

## Claim 2. Coordinate Calibration

Claim: calibrated coordinates are more stable and interpretable than raw projection scores.

Evidence in this repository:

- `output/supporting_claims/coordinate_calibration_report.json`: raw vs calibrated coordinate analysis.
- `output/supporting_claims/primitive_seeded_calibration_report.json`: primitive-seeded calibration report.

Interpretation:

- Raw `A^T z`-style scores are useful as projections but should not be read directly as criterion satisfaction probabilities.
- Calibrated coordinates, including `A^\dagger z` and `q`, are intended to behave more like stable legal criterion satisfaction scores.

## Claim 3. Hybrid Superiority

Claim: Hybrid GTR improves over raw-only prediction.

Evidence in this repository:

- `output/supporting_claims/hybrid_gtr_v2_ablation_report.json`: model ablation table.
- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`: CAIL GTR checkpoint report.

Key comparison from the supporting ablation report:

- `raw_head_on_z`: test micro-F1 `0.7724`, exact match `0.4493`.
- `full_hybrid`: test micro-F1 `0.8613`, exact match `0.7138`.

The CAIL checkpoint report is included as the reproducible artifact used to build the LLM candidate prompts.

## Claim 4. Interaction Necessity

Claim: statute-specific decision fields are needed; an additive representation is not enough.

Evidence in this repository:

- `train_hybrid_gtr_v2.py`: shared GTR model implementation.
- `train_cail_hybrid_gtr_v2.py`: CAIL trainer using raw, field, and residual components.
- `output/supporting_claims/hybrid_gtr_v2_ablation_report.json`: compares raw-only, field-only, additive, residual, and full-hybrid variants.

Interpretation:

- A legal criterion can matter differently depending on the target statute.
- The decision field lets the model express statute-specific interaction patterns instead of treating every coordinate as a globally additive feature.

## Claim 5. Suppression

Claim: GTR selectively removes false positives instead of only adding more candidates.

Evidence in this repository:

- `output/supporting_claims/hybrid_gtr_v2_mechanism_report.json`: suppression and label-change analysis.

Key example from the mechanism report:

- `raw_plus_field_no_residual_vs_raw` removed `145` predictions, of which `122` were false positives.
- Suppression precision for that comparison is `0.8414`.

Interpretation:

- The model's gain is not just a recall effect.
- A major mechanism is selective suppression of legally implausible candidate statutes.

## Claim 6. Hard-Negative Margin

Claim: GTR improves margins for confusing statute pairs.

Evidence in this repository:

- `output/supporting_claims/hard_negative_margin_report.json`: pair-level hard-negative margin analysis.

Aggregate signal:

- `full_hybrid_mean_margin`: `5.3782`.
- `raw_head_on_z_mean_margin`: `5.2277`.
- `full_hybrid_confuser_fp_rate`: `0.0155`.
- `raw_head_on_z_confuser_fp_rate`: `0.1192`.

Interpretation:

- The relevant test is not only global F1.
- GTR should improve separation on legally adjacent statute pairs that are easy to confuse.

## Claim 7. LLM Routing

Claim: GTR candidate refinement improves downstream LLM legal judgment.

Evidence in this repository:

- `build_cail_gtr_v2_llm_prompts.py`: builds the LLM prompt conditions.
- `run_openai_prompt_jsonl.py`: runs GPT models on prompt JSONL rows.
- `evaluate_llm_candidate_routing_results.py`: evaluates LLM outputs.
- `output/cail2018_gtr_v2_only/full/llm_prompts/`: saved prompt artifacts.
- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json`: saved GPT-5.4 metrics.

Condition roles:

| Condition | Role |
| --- | --- |
| `full_fact + LLM` | LLM-only baseline |
| `raw_topk + LLM` | retrieval baseline |
| `gtr_topk + LLM` | direct GTR router ablation |
| `raw_pool_gtr_rerank + LLM` | best router |
| `raw_pool_gtr_rerank_score_prompt + LLM` | strongest setting |

Current saved GPT-5.4 result:

- `raw_pool_gtr_rerank`: exact match `0.7428`, micro-F1 `0.7987`.
- `raw_pool_gtr_rerank_score_prompt`: exact match `0.7609`, micro-F1 `0.8039`.

Note:

- The saved CAIL prompt artifact contains `full_fact`, `raw_topk`, `raw_pool_gtr_rerank`, and `raw_pool_gtr_rerank_score_prompt`.
- `gtr_topk` is the direct-router ablation in the paper scenario; if needed for the final table, add it as a separate prompt condition or report it from the corresponding expanded routing run.

## Suggested Review Order

1. Start with this claim map.
2. Read `docs/results_summary.md` for the compact numbers.
3. Use `docs/file_map.md` to locate the source files behind each claim.
4. Use `docs/reproducibility.md` to rerun the LLM routing path.
5. Run `scripts/08_recompute_saved_gpt_metrics.sh` for the no-API verification.

