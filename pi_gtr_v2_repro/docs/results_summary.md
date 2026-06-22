# Results Summary

This summary records the representative metrics already produced in the repository. Use `results/expected_metrics.json` as the machine-readable comparison target.

## GTR Checkpoint

Source:

- `output/cail2018_gtr_v2_only/full/hybrid_gtr_v2_report.json`

Full GTR test metrics:

- Exact match: `0.6118493909`
- Micro-F1: `0.7492519614`
- Macro-F1: `0.6055095945`

The checkpoint is used to build candidate lists for the LLM and LoRA decoder experiments.

## GPT-5.4 Rerank Decoder

Source:

- `output/cail2018_gtr_v2_only/full/gpt54_gtr_v2_rerank_276/metrics_gpt54_rerank_276.json`

Metrics on the 276-case sampled test set:

| Condition | Exact Match | Micro-F1 | Macro-F1 | Parse OK |
| --- | ---: | ---: | ---: | ---: |
| `raw_pool_gtr_rerank` | `0.7427536232` | `0.7987012987` | `0.3781722478` | `1.0000` |
| `raw_pool_gtr_rerank_score_prompt` | `0.7608695652` | `0.8039215686` | `0.3773177668` | `1.0000` |

## Qwen3-8B LoRA Full

Source:

- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/full/summary_metrics.json`

Metrics on the 276-case sampled test set:

| Condition | Exact Match | Micro-F1 | Macro-F1 | Parse OK |
| --- | ---: | ---: | ---: | ---: |
| `raw_pool_gtr_rerank` | `0.7391304348` | `0.7919254658` | `0.3798632680` | `0.9928` |
| `raw_pool_gtr_rerank_score_prompt` | `0.7500000000` | `0.7919876733` | `0.3901935908` | `0.9928` |

## Qwen3-8B LoRA Smoke

Source:

- `output/cail2018_gtr_v2_only/full/qwen3_8b_lora_sft/smoke/summary_metrics.json`

Metrics on the 276-case sampled test set:

| Condition | Exact Match | Micro-F1 | Macro-F1 | Parse OK |
| --- | ---: | ---: | ---: | ---: |
| `raw_pool_gtr_rerank` | `0.7355072464` | `0.7744000000` | `0.3702399430` | `0.9928` |
| `raw_pool_gtr_rerank_score_prompt` | `0.7355072464` | `0.7898089172` | `0.3784866615` | `0.9928` |

## Gemini 3.5 Flash

Source:

- `output/cail2018_gtr_v2_only/full/gemini35_gtr_v2_all4_276/metrics_gemini35_all4_276.json`

The existing Gemini 3.5 Flash run has very low parse rates, so it is retained as an audit artifact rather than the main result. Hosted model behavior may also change over time.

