"""LLM routing condition names used in the result."""

from __future__ import annotations


CONDITION_ROLES = {
    "full_fact": "LLM-only baseline",
    "raw_topk": "retrieval baseline",
    "gtr_topk": "direct GTR router ablation",
    "raw_pool_gtr_rerank": "best router",
    "raw_pool_gtr_rerank_score_prompt": "strongest setting",
}
