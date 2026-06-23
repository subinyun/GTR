# Paper Tables

## Table 1. Main Results

| Condition | Exact Match | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: |
| CAIL GTR checkpoint | 0.6118 | 0.7493 | 0.6055 |
| GPT-5.4 `raw_pool_gtr_rerank` | 0.7428 | 0.7987 | 0.3782 |
| GPT-5.4 `raw_pool_gtr_rerank_score_prompt` | 0.7609 | 0.8039 | 0.3773 |
| Qwen3-8B LoRA `raw_pool_gtr_rerank` | 0.7391 | 0.7919 | 0.3799 |
| Qwen3-8B LoRA `raw_pool_gtr_rerank_score_prompt` | 0.7500 | 0.7920 | 0.3902 |

## Table 3. Threshold vs GTR

| Method | Exact Match | Micro-F1 | Macro-F1 |
| --- | ---: | ---: | ---: |
| Raw + default threshold | 0.4493 | 0.7724 | 0.3016 |
| Raw + global threshold | 0.4493 | 0.7724 | 0.3016 |
| Raw + class-wise threshold | 0.5362 | 0.7824 | 0.3190 |
| Raw + temperature scaling | 0.4493 | 0.7724 | 0.3016 |
| GTR v2 | 0.7138 | 0.8613 | 0.3253 |
| GTR v2 + threshold tuning | 0.7138 | 0.8613 | 0.3253 |

## Figure/Table Support. Hard-Negative Margin

| Metric | Value |
| --- | ---: |
| raw mean margin | 5.2277 |
| full hybrid mean margin | 5.3782 |
| raw confuser FP rate | 0.1192 |
| full hybrid confuser FP rate | 0.0155 |
