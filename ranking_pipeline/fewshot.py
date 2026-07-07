"""Few-shot example handling: building the demonstration pool (used by
prepare_ranking_fewshot.py) and selecting examples from it at run time
(used by llm_ranking_pipeline.py).

Why a separate pool-building step: the ranking LLM sees a numbered candidate
list per sentence (the retriever's top-N), and a good few-shot demonstration
has to show the *complete* correct ranking for its own candidate list. If
retrieval missed one of a demo sentence's gold skills, that skill couldn't
appear anywhere in the demonstrated answer -- silently teaching the model an
incomplete ranking. build_pool therefore injects any missed gold skills into
each demo's candidate list before deriving the answer.
"""

import json
import random


def build_pool(queries: list[dict], top_labels: list[list[str]]) -> list[dict]:
    """Turn validation queries + their retrieved candidates into a flat pool
    of demonstration entries.

    For each sentence:
      1. Start from the retriever's top-N candidates (the same list the live
         pipeline would show the LLM).
      2. Inject any gold skill retrieval missed, so every gold skill is
         present in the candidate list and can be ranked.
      3. Derive the "ranked" answer: gold candidates ordered by cluster
         (1 = most important first), tie-broken on candidate position,
         expressed as 1-based indices into the candidate list -- the same
         numbering scheme the live pipeline uses.

    Gold-bearing sentences (n_gold > 0) become ranking demos; no-gold
    sentences become rejection demos ("ranked": []).
    """
    pool = []
    for i, q in enumerate(queries):
        candidates = list(top_labels[i])
        gold_clusters = q["gold_clusters"]
        missing = sorted(s for s in q["gold"] if s not in candidates)
        candidates.extend(missing)  # guarantee every gold skill is rankable in the demo

        gold_in_candidates = [s for s in candidates if s in gold_clusters]
        # Most important (lowest cluster) first; tie-break on candidate
        # position so the ordering is deterministic.
        gold_in_candidates.sort(key=lambda s: (gold_clusters[s], candidates.index(s)))
        ranked_numbers = [candidates.index(s) + 1 for s in gold_in_candidates]

        pool.append({
            "doc_id": q["doc_id"],
            "sentence": q["sentence"],
            "candidates": candidates,
            "ranked": ranked_numbers,
            "n_gold": len(q["gold"]),
            "n_gold_in_candidates": len(gold_in_candidates),
            "n_missing_injected": len(missing),
            "n_distinct_clusters": len(set(gold_clusters.values())),
            # For n_gold==0 entries: True if SKILL-XL still flagged the
            # sentence "relevant" despite no mapped skill -- a harder,
            # more instructive rejection demo than an obviously unrelated
            # sentence. Unused for n_gold>0 entries.
            "relevant_flag": q["relevant_flag"],
        })
    return pool


def load_fewshot_examples(pool_path: str, n_shot: int, n_shot_no_gold: int,
                          selection: str, seed: int) -> list[dict]:
    """Pick `n_shot` ranking demonstrations (pool entries with n_gold > 0)
    plus `n_shot_no_gold` rejection demonstrations (n_gold == 0, "ranked": [])
    from a pool built by build_pool / prepare_ranking_fewshot.py.

    The two counts are drawn from disjoint subsets of the pool, so
    n_shot_no_gold=0 (the default) reproduces skill-only few-shot behaviour
    exactly, no matter what the pool file itself contains -- the pool includes
    no-gold entries by default, but they're only ever used if you opt in.

    selection='diverse' (default): for ranking demos, deterministically pick
    the examples spanning the most distinct gold-cluster tiers; for rejection
    demos, prefer "hard" negatives (relevant_flag=True: SKILL-XL still
    flagged the sentence relevant despite no mapped skill) over obviously
    unrelated filler. selection='random': uniform sample with `seed` from
    each subset independently.
    """
    if n_shot == 0 and n_shot_no_gold == 0:
        return []
    with open(pool_path, encoding="utf-8") as f:
        pool = json.load(f)

    gold_pool = [r for r in pool if r["n_gold"] > 0]
    no_gold_pool = [r for r in pool if r["n_gold"] == 0]

    def pick(sub_pool: list[dict], n: int, rank_key, seed_offset: int) -> list[dict]:
        if n == 0:
            return []
        if len(sub_pool) < n:
            raise SystemExit(f"--fewshot-pool {pool_path!r} has only "
                             f"{len(sub_pool)} usable examples, need {n}")
        if selection == "random":
            return random.Random(seed + seed_offset).sample(sub_pool, n)
        return sorted(sub_pool, key=rank_key, reverse=True)[:n]

    ranking_demos = pick(
        gold_pool, n_shot,
        lambda r: (r["n_distinct_clusters"], r["n_gold_in_candidates"]), 0)
    rejection_demos = pick(
        no_gold_pool, n_shot_no_gold,
        lambda r: (r["relevant_flag"],), 1)
    return ranking_demos + rejection_demos
