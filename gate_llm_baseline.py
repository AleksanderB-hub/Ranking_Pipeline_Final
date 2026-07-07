"""LLM-as-gate baseline: classify EVERY sentence as skill-bearing or not with
the LLM alone, for comparison with the RoBERTa gate and the RoBERTa+LLM
cascade.

Variants (--variants, default all):
  zero-shot  minimal instruction prompt, answer RELEVANT / NON-RELEVANT
  3-shot     same prompt + three demonstrations drawn deterministically from
             the validation split (a multi-skill positive, a single-skill
             positive, and a "hard" negative: flagged relevant by SKILL-XL
             but mapping to no skill)
  audit-v1   the cascade's chain-of-thought audit prompt (6 embedded worked
             examples), applied to all sentences instead of only the gate's
             uncertain band. Shares the cascade's LLM cache, so the band
             sentences are already paid for.

All decisions are cached in --llm-cache-dir keyed by prompt version, so
re-runs are free. Prints and stores gate_classification_metrics for each
variant plus the accept-all floor.

Prerequisite: the local vLLM server must be up (for the qwen-local default).

Example:
    python gate_llm_baseline.py --llm qwen-local \\
        --output results/gate_baselines.json
"""

import argparse
import asyncio
import json
import os
import sys

import numpy as np
from tqdm.asyncio import tqdm as atqdm

from ranking_pipeline.config import LLM_REGISTRY, load_api_keys
from ranking_pipeline.data import build_queries_with_clusters
from ranking_pipeline.gate import gate_classification_metrics
from ranking_pipeline.llm_clients import GateAuditor, _LLMClientBase
from ranking_pipeline.prompts import parse_gate_decision

GATE_BASELINE_SYSTEM_PROMPT = (
    "You classify job-description sentences for an HR skill-extraction "
    "system. Decide whether the sentence contains an EXPLICIT, EXTRACTABLE "
    "skill or qualification: concrete hard or soft skills, core professional "
    "competencies, specific experience requirements, or domain-specific "
    "tasks that require professional expertise. Vague or generic job duties, "
    "work conditions, company perks, marketing language, and generic "
    "behavioural expectations are NOT relevant. "
    "Answer with exactly one word: RELEVANT or NON-RELEVANT."
)


def pick_demos(dev_queries: list[dict]) -> list[tuple[str, int]]:
    """Three deterministic demonstrations from the validation split:
    a multi-skill positive, a single-skill positive, and a hard negative
    (SKILL-XL flagged the sentence relevant, but it maps to no skill)."""
    multi = next(q for q in dev_queries if len(q["gold"]) >= 2)
    single = next(q for q in dev_queries if len(q["gold"]) == 1)
    hard_neg = next(q for q in dev_queries if not q["gold"] and q["relevant_flag"])
    return [(multi["sentence"], 1), (single["sentence"], 1), (hard_neg["sentence"], 0)]


def build_baseline_messages(sentence: str,
                            demos: list[tuple[str, int]] | None) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": GATE_BASELINE_SYSTEM_PROMPT}]
    for demo_sentence, label in demos or []:
        msgs.append({"role": "user", "content": f'Sentence: "{demo_sentence}"'})
        msgs.append({"role": "assistant",
                     "content": "RELEVANT" if label else "NON-RELEVANT"})
    msgs.append({"role": "user", "content": f'Sentence: "{sentence}"'})
    return msgs


class PromptedGate(_LLMClientBase):
    """LLM gate with a fixed prompt variant; caches per (variant, sentence)."""

    def __init__(self, name: str, cfg: dict, cache_dir: str, variant: str,
                 demos: list[tuple[str, int]] | None, max_tokens: int):
        super().__init__(name, cfg, cache_dir,
                         f"gate_baseline__{variant}__{name.replace('/', '_')}",
                         max_tokens)
        self.prompt_version = f"gate-baseline-{variant}-v1"
        self.demos = demos

    async def classify(self, sentence: str, sem: asyncio.Semaphore) -> int:
        ck = self.cache.key(self.prompt_version, self.model, sentence, [])
        cached = self.cache.get(ck)
        if cached is not None:
            return int(cached)
        messages = build_baseline_messages(sentence, self.demos)
        async with sem:
            await self.rate_limiter.wait()
            try:
                text = await self._raw(messages)
            except Exception as e:  # network error -> conservative keep, no cache
                print(f"    [{self.name}] call failed: {e}")
                return 1
        decision = parse_gate_decision(text)
        self.cache.put(ck, decision)
        return decision


async def classify_all(gate, sentences: list[str], desc: str) -> np.ndarray:
    sem = asyncio.Semaphore(gate.max_concurrent)
    call = gate.classify if isinstance(gate, PromptedGate) else gate.audit
    decisions = await atqdm.gather(*[call(s, sem) for s in sentences],
                                   desc=f"  {desc}", unit="sent")
    gate.cache.flush()
    return np.array(decisions, dtype=bool)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="qwen-local",
                   help=f"Registry key: {list(LLM_REGISTRY)}")
    p.add_argument("--variants", nargs="+", default=["zero-shot", "3shot", "audit-v1"],
                   choices=["zero-shot", "3shot", "audit-v1"])
    p.add_argument("--dev-dataset", default="./data/development.csv",
                   help="Validation split the 3-shot demonstrations are drawn from")
    p.add_argument("--dataset", default="TechWolf/Skill-XL")
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default=None)
    p.add_argument("--max-tokens", type=int, default=32,
                   help="Max tokens for zero/3-shot answers (audit-v1 uses 150)")
    p.add_argument("--llm-cache-dir", default=".llm_cache")
    p.add_argument("--output", default="results/gate_baselines.json")
    args = p.parse_args()

    load_api_keys()
    if args.llm not in LLM_REGISTRY:
        raise SystemExit(f"Unknown --llm {args.llm!r}. Use one of: {list(LLM_REGISTRY)}")
    cfg = LLM_REGISTRY[args.llm]

    target_split = None if args.dataset.endswith(".csv") else args.split
    queries = build_queries_with_clusters(args.dataset, target_split, args.subset)
    sentences = [q["sentence"] for q in queries]

    demos = None
    if "3shot" in args.variants:
        dev_queries = build_queries_with_clusters(args.dev_dataset, None, args.subset)
        demos = pick_demos(dev_queries)
        print("[demos] " + " | ".join(
            f"{'REL' if lab else 'NON-REL'}: {s[:60]}..." for s, lab in demos))

    report = {"config": vars(args), "n_queries": len(queries), "variants": {}}

    # Accept-all floor: P = class prior, R = 1.
    accept_all = gate_classification_metrics(queries, np.ones(len(queries), dtype=bool))
    report["variants"]["accept-all"] = accept_all

    for variant in args.variants:
        if variant == "audit-v1":
            gate = GateAuditor(args.llm, cfg, args.llm_cache_dir, max_tokens=150)
        else:
            gate = PromptedGate(args.llm, cfg, args.llm_cache_dir, variant,
                                demos if variant == "3shot" else None,
                                args.max_tokens)
        mask = asyncio.run(classify_all(gate, sentences, f"gate[{variant}]"))
        metrics = gate_classification_metrics(queries, mask)
        metrics["n_accepted"] = int(mask.sum())
        report["variants"][variant] = metrics

    print(f"\n{'variant':<12} {'P':>8} {'R':>8} {'F1':>8} {'acc':>8} {'accepted':>9}")
    for name, m in report["variants"].items():
        print(f"{name:<12} {m['precision']:>8.4f} {m['recall']:>8.4f} "
              f"{m['f1']:>8.4f} {m['accuracy']:>8.4f} "
              f"{m.get('n_accepted', len(queries)):>9}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[done] Report written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
