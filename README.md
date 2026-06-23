# GTR

This repository is organized as a review-facing reproducibility package. The code is arranged by claim/table, so a reviewer can see which script produces which result.

## Reproduce Artifacts

```bash
python scripts/01_build_axis_bank.py --config configs/data/lbox.yaml
python scripts/run_all_main_experiments.py --config configs/exp/main_table.yaml
python scripts/make_result_tables.py --input outputs/metrics
```

The commands above write normalized outputs under:

- `outputs/metrics/`
- `outputs/predictions/`
- `outputs/logs/`
- `outputs/configs/`

## Claim Map

| Claim | Role | Script/artifact |
| --- | --- | --- |
| Axis validity | Figure 2 / method validation | `scripts/01_build_axis_bank.py`, `artifacts/supporting_claims/proposal_axis_validity_report.json` |
| Coordinate calibration | coordinate sanity | `artifacts/supporting_claims/coordinate_calibration_report.json` |
| Hybrid superiority | Table 1 main results | `scripts/run_all_main_experiments.py` |
| Interaction necessity | Table 2 ablation | `scripts/05_ablation.py` |
| Threshold vs GTR | Table 3 threshold comparison | `scripts/06_threshold_vs_gtr.py` |
| Suppression | mechanism analysis | `scripts/07_suppression_analysis.py` |
| Hard-negative margin | Figure 3 hard negatives | `scripts/08_hard_negative_margin.py` |
| LLM routing | Table 4 LLM routing | `scripts/09_llm_routing_export.py` |

See `docs/claims.md` for the detailed claim-by-claim evidence map.

## Result Table Outputs

| Item | Output file |
| --- | --- |
| Table 1 Main results | `outputs/metrics/main_table.json` |
| Table 2 Ablation | `outputs/metrics/ablation.json` |
| Table 3 Threshold vs GTR | `outputs/metrics/threshold_vs_gtr.json` |
| Table 4 LLM routing | `outputs/metrics/llm_routing.json` |
| Figure 3 Hard-negative margin | `outputs/metrics/hard_negative.json` |
| Rendered compact tables | `outputs/metrics/result_tables.md` |

Every generated metrics JSON includes provenance:

```json
{
  "git_commit": "...",
  "git_dirty": true,
  "config_path": "configs/exp/main_table.yaml",
  "config_hash": "...",
  "seed": 42,
  "timestamp": "..."
}
```

## LLM Routing Conditions

| Condition | Role |
| --- | --- |
| `full_fact + LLM` | LLM-only baseline |
| `raw_topk + LLM` | retrieval baseline |
| `gtr_topk + LLM` | direct GTR router ablation |
| `raw_pool_gtr_rerank + LLM` | best router |
| `raw_pool_gtr_rerank_score_prompt + LLM` | strongest setting |

Saved GPT-5.4 result on the 276-case CAIL sample:

| Condition | Exact match | Micro-F1 |
| --- | ---: | ---: |
| `raw_pool_gtr_rerank` | `0.7428` | `0.7987` |
| `raw_pool_gtr_rerank_score_prompt` | `0.7609` | `0.8039` |

## Repository Layout

```text
configs/      YAML configs for data, model variants, and experiments
src/gtr/      review-facing GTR interfaces and helper utilities
scripts/      result table/claim wrappers
repro/        low-level reproduction scripts and expected metrics
experiments/  named experiment buckets
outputs/      normalized metrics, predictions, figures, logs, configs
docs/         claim map, reproduction guide, and file map
```

The repository includes the full `LBOX/statute_classification/` splits. For CAIL2018, it includes a small label-vocabulary placeholder for evaluator-only reproduction; the full CAIL2018 split files are excluded because the train file is too large for a normal GitHub push.

## Output Policy

Committed:

- `configs/`
- `src/`
- `scripts/`
- `tests/`
- `docs/`
- small `outputs/metrics/*.json`
- `outputs/metrics/result_tables.md`

Ignored or kept out of Git:

- raw full CAIL data
- checkpoints and adapters except the small released GTR checkpoint
- embeddings and caches
- large predictions
- logs
- API outputs that may contain sensitive text

