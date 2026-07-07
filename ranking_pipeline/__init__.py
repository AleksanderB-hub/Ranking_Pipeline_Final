"""LLM skill-ranking pipeline for SKILL-XL / ESCO.

Module map:
  config      -- LLM registry, provider selection, API-key loading
  esco        -- ESCO skills download / vocabulary manager
  data        -- SKILL-XL query building and vocabulary assembly
  retrieval   -- dense retrieval with cached skill embeddings
  gate        -- binary relevance gate and its standalone metrics
  prompts     -- prompt construction and response parsing (ranking + gate audit)
  fewshot     -- few-shot pool building and run-time example selection
  llm_clients -- async LLM clients, disk cache, rate limiter
  metrics     -- P/R/F1, NDCG@k, MAP@k, retriever diagnostics
  reporting   -- console formatting and per-sentence prediction traces

Entry points live at the repository root: llm_ranking_pipeline.py (evaluation)
and prepare_ranking_fewshot.py (few-shot pool preparation).
"""
