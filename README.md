# GTR

This repository is a compact review and reproducibility package for the GTR experiments. It is organized around the paper story, not around experiment timestamps.

## What This Repository Shows

GTR is a legally structured routing layer. The main claim is that legal-coordinate structure improves statute prediction and also gives LLMs cleaner candidate statutes.

The evidence is organized into seven claims:

| Claim | Short version | Main evidence |
| --- | --- | --- |
| 1. Axis validity | axes measure primitive legal concepts | `cail_gtr_axis_schema.py`, `output/supporting_claims/proposal_axis_validity_report.json` |
| 2. Coordinate calibration | calibrated coordinates are more stable than raw projections | `output/supporting_claims/coordinate_calibration_report.json` |
| 3. Hybrid superiority | Hybrid GTR improves over raw-only | `output/supporting_claims/hybrid_gtr_v2_ablation_report.json` |
| 4. Interaction necessity | statute-specific decision fields help beyond additive features | `train_hybrid_gtr_v2.py`, ablation report |
| 5. Suppression | GTR selectively removes false positives | `output/supporting_claims/hybrid_gtr_v2_mechanism_report.json` |
| 6. Hard-negative margin | confusing statute pairs get better separated | `output/supporting_claims/hard_negative_margin_report.json` |
| 7. LLM routing | GTR candidate filtering improves LLM decisions | `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/` |

## Where To Read

- `docs/paper_claims.md`: the seven claims and evidence map.
- `docs/reproduce.md`: how to rerun the saved-result check, GPT decoding, CAIL path, and Qwen LoRA path.
- `docs/files.md`: what each important folder/file is for.

## Key LLM Result

Saved GPT-5.4 result on the 276-case CAIL sample:

| Condition | Role | Exact match | Micro-F1 |
| --- | --- | ---: | ---: |
| `raw_pool_gtr_rerank + LLM` | best router | `0.7428` | `0.7987` |
| `raw_pool_gtr_rerank_score_prompt + LLM` | strongest setting | `0.7609` | `0.8039` |

The full prompt design also includes:

| Condition | Role |
| --- | --- |
| `full_fact + LLM` | LLM-only baseline |
| `raw_topk + LLM` | retrieval baseline |
| `gtr_topk + LLM` | direct GTR router ablation |
| `raw_pool_gtr_rerank + LLM` | best router |
| `raw_pool_gtr_rerank_score_prompt + LLM` | strongest setting |

## Quick Check

Run the no-API evaluator check:

```bash
bash repro/scripts/08_recompute_saved_gpt_metrics.sh
```

The repository includes the full `LBOX/statute_classification/` splits. For CAIL2018, it includes a small label-vocabulary placeholder for evaluator-only reproduction; the full CAIL2018 split files are excluded because the train file is too large for a normal GitHub push.

