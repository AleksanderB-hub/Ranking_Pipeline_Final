"""Async LLM clients for the two LLM roles in the pipeline, plus the disk
cache and rate limiter they share.

  LLMRanker   -- orders the retrieved ESCO candidates for a sentence.
  GateAuditor -- audits sentences in the relevance gate's uncertain band
                 (the cascade refinement enabled with --gate-llm-refine).

Both dispatch to one of three providers (local vLLM via the OpenAI-compatible
API, OpenAI, Anthropic) and cache every decision on disk keyed by
(prompt version, model, sentence, candidates), so re-runs and metric
recomputations never re-pay for LLM calls.
"""

import asyncio
import hashlib
import json
import os
from typing import Any

import numpy as np
from tqdm.asyncio import tqdm as atqdm

# The config module is imported as a whole (not `from .config import ...`)
# because QWEN_ENABLE_THINKING can be flipped at runtime by --qwen-thinking;
# reading it through the module picks up that change.
from . import config
from .config import ENV_FILE
from .prompts import (
    GATE_AUDIT_PROMPT_VERSION,
    build_gate_audit_messages,
    build_rank_messages,
    parse_gate_decision,
    parse_ranked_indices,
)

# ---------------------------------------------------------------------------
# Disk cache for LLM decisions
# ---------------------------------------------------------------------------


class LLMCache:
    """JSON-file cache of LLM decisions, one file per model slug."""

    def __init__(self, cache_dir: str, model_slug: str, autoflush_every: int = 50):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, f"{model_slug}.json")
        self.store: dict[str, Any] = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.store = json.load(f)
        self._dirty = False
        self._since_flush = 0
        self._autoflush_every = autoflush_every

    @staticmethod
    def key(prompt_version: str, model: str, sentence: str, candidates: list[str],
            variant: str = "") -> str:
        blob = "\x00".join([prompt_version, variant, model, sentence, *candidates])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any:
        """Return the cached value (any JSON-serialisable type) or None if absent."""
        return self.store.get(key)

    def put(self, key: str, value: Any) -> None:
        self.store[key] = value
        self._dirty = True
        self._since_flush += 1
        if self._since_flush >= self._autoflush_every:
            self.flush()   # periodic save so a crash doesn't lose LLM calls

    def flush(self) -> None:
        if self._dirty:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.store, f)
            self._dirty = False
        self._since_flush = 0


# ---------------------------------------------------------------------------
# Rate limiter (API providers only)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Caps request *starts* to `rate_per_sec` per second across all concurrent
    tasks. rate_per_sec=None disables it -- used for the local LLM, which is
    only bounded by max_concurrent and the cache."""

    def __init__(self, rate_per_sec: float | None):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec else 0.0
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self) -> None:
        if self.min_interval <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self._lock:        # reserve the next slot (fast, no sleep here)
            now = loop.time()
            slot = max(now, self._next_time)
            self._next_time = slot + self.min_interval
        delay = slot - loop.time()
        if delay > 0:
            await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Provider-dispatched client base
# ---------------------------------------------------------------------------


class _LLMClientBase:
    """Shared plumbing for LLMRanker and GateAuditor: provider dispatch,
    API-key resolution, per-model cache, concurrency and rate caps."""

    def __init__(self, name: str, cfg: dict, cache_dir: str, cache_slug: str,
                 max_tokens: int = 512):
        self.name = name
        self.provider = cfg["provider"]
        self.model = cfg["model"]
        self.base_url = cfg.get("base_url")
        self.max_concurrent = cfg.get("max_concurrent", 8)
        self.max_tokens = max_tokens
        self.cache = LLMCache(cache_dir, cache_slug)

        key_env = cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env) if key_env else None
        if self.provider in {"openai", "anthropic"} and not self.api_key:
            raise SystemExit(f"[{name}] env var {key_env} not set. Add it to {ENV_FILE}.")
        # vLLM ignores the key but the OpenAI client requires a non-empty string.
        if self.provider == "openai_compat" and not self.api_key:
            self.api_key = "EMPTY"

        self.rate_limiter = RateLimiter(cfg.get("rate_limit"))
        self._client = None  # built lazily inside the event loop

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider in {"openai", "openai_compat"}:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown provider {self.provider}")
        return self._client

    async def _call_openai(self, messages: list[dict]) -> str:
        client = self._get_client()
        extra: dict = {}
        if self.provider == "openai_compat":
            # Qwen3 thinking toggle, honoured by vLLM's chat template.
            extra["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": config.QWEN_ENABLE_THINKING}
            }
        # Newer OpenAI models reject temperature!=1 and/or use max_completion_tokens.
        # Try the classic params first, then fall back.
        try:
            resp = await client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=0, max_tokens=self.max_tokens, **extra,
            )
        except Exception:
            resp = await client.chat.completions.create(
                model=self.model, messages=messages,
                max_completion_tokens=self.max_tokens, **extra,
            )
        return resp.choices[0].message.content or ""

    async def _call_anthropic(self, messages: list[dict]) -> str:
        client = self._get_client()
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        turns = [{"role": m["role"], "content": m["content"]}
                 for m in messages if m["role"] != "system"]
        resp = await client.messages.create(
            model=self.model, system=system, messages=turns,
            max_tokens=self.max_tokens, temperature=0,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def _raw(self, messages: list[dict]) -> str:
        if self.provider in {"openai", "openai_compat"}:
            return await self._call_openai(messages)
        return await self._call_anthropic(messages)


# ---------------------------------------------------------------------------
# Candidate ranker
# ---------------------------------------------------------------------------


class LLMRanker(_LLMClientBase):
    def __init__(self, name: str, cfg: dict, cache_dir: str,
                 prompt_version: str, system_prompt: str,
                 few_shot: list[dict] | None, max_tokens: int = 512):
        super().__init__(name, cfg, cache_dir, name.replace("/", "_"), max_tokens)
        self.few_shot = few_shot
        self.prompt_version = prompt_version
        self.system_prompt = system_prompt

    async def rank(self, sentence: str, candidates: list[str],
                   sem: asyncio.Semaphore) -> list[str]:
        """Return the subset of `candidates` the LLM marks relevant, ordered
        most-to-least relevant."""
        if not candidates:
            return []
        # Thinking mode changes Qwen's output, so it partitions the cache.
        variant = (f"think={config.QWEN_ENABLE_THINKING}"
                   if self.provider == "openai_compat" else "")
        ck = self.cache.key(self.prompt_version, self.model, sentence, candidates, variant)
        cached = self.cache.get(ck)
        if cached is not None:
            return [candidates[i] for i in cached if i < len(candidates)]

        messages = build_rank_messages(sentence, candidates, self.system_prompt, self.few_shot)
        async with sem:
            await self.rate_limiter.wait()   # no-op for local (rate_limit=None)
            try:
                text = await self._raw(messages)
            except Exception as e:  # network / API error -> empty, do not cache
                print(f"    [{self.name}] call failed: {e}")
                return []
        idx = parse_ranked_indices(text, len(candidates))
        self.cache.put(ck, idx)   # order matters here -- do not sort
        return [candidates[i] for i in idx]


async def run_ranker_over_sentences(
    ranker: LLMRanker, sentences: list[str], candidate_lists: list[list[str]],
) -> list[list[str]]:
    """Run the LLM on every sentence once (concurrency- and rate-capped).

    Gated vs non-gated predictions are derived afterwards by masking, since
    the gate only suppresses outputs -- it does not change the LLM's
    per-sentence call. Cached sentences return immediately without occupying
    a slot.
    """
    sem = asyncio.Semaphore(ranker.max_concurrent)
    tasks = [ranker.rank(sentences[i], candidate_lists[i], sem)
             for i in range(len(sentences))]
    # tqdm.asyncio.gather preserves input order while showing live progress.
    preds_list = await atqdm.gather(*tasks, desc=f"  {ranker.name}", unit="sent")
    ranker.cache.flush()
    return list(preds_list)


# ---------------------------------------------------------------------------
# Gate auditor (LLM cascade refinement of the relevance gate)
# ---------------------------------------------------------------------------


class GateAuditor(_LLMClientBase):
    def __init__(self, name: str, cfg: dict, cache_dir: str, max_tokens: int = 150):
        super().__init__(name, cfg, cache_dir,
                         f"gate_audit__{name.replace('/', '_')}", max_tokens)

    async def audit(self, sentence: str, sem: asyncio.Semaphore) -> int:
        """Return 1 (keep) / 0 (drop) for one sentence."""
        ck = self.cache.key(GATE_AUDIT_PROMPT_VERSION, self.model, sentence, [])
        cached = self.cache.get(ck)
        if cached is not None:
            return int(cached)

        messages = build_gate_audit_messages(sentence)
        async with sem:
            await self.rate_limiter.wait()
            try:
                text = await self._raw(messages)
            except Exception as e:  # network / API error -> conservative keep, do not cache
                print(f"    [{self.name}] gate-audit call failed: {e}")
                return 1
        decision = parse_gate_decision(text)
        self.cache.put(ck, decision)
        return decision


async def run_gate_cascade(
    auditor: GateAuditor, sentences: list[str], gate_probs: np.ndarray,
    threshold: float, t_upper: float,
) -> tuple[np.ndarray, dict]:
    """Cascade router over the gate's probabilities: auto-reject
    prob < threshold, auto-accept prob >= t_upper, LLM-audit the band in
    between.

    Returns (keep_mask, meta); meta reports how many sentences were routed to
    the LLM so the tradeoff is visible in the run log.
    """
    n = len(sentences)
    keep_mask = gate_probs >= threshold
    audit_idx = [i for i in range(n) if threshold <= gate_probs[i] < t_upper]

    if audit_idx:
        sem = asyncio.Semaphore(auditor.max_concurrent)
        tasks = [auditor.audit(sentences[i], sem) for i in audit_idx]
        decisions = await atqdm.gather(*tasks, desc=f"  gate-audit[{auditor.name}]",
                                       unit="sent")
        auditor.cache.flush()
        for i, d in zip(audit_idx, decisions):
            keep_mask[i] = bool(d)

    return keep_mask, {"n_audited": len(audit_idx), "n_total": n, "t_upper": t_upper}
