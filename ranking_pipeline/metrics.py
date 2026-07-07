"""All evaluation metrics used by the pipeline.

Three families:

  Set metrics       -- micro P/R/F1 treating the ranked list as a set
                       (score_micro / both_modes), in two no-skill regimes.
  Ranking metrics   -- NDCG@k with graded relevance derived from SKILL-XL's
                       `cluster` label, plus binary-relevance MAP@k, each in
                       three variants (ranking_score_diagnostics).
  Retriever metrics -- MAP@k / Recall@k / MRR of the raw retrieval order,
                       relevant sentences only (ranking_diagnostics); pure
                       retriever quality, no gate or LLM involved.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Graded relevance from SKILL-XL clusters
# ---------------------------------------------------------------------------


def cluster_to_relevance(cluster, max_tier: int = 5) -> int:
    """Map a SKILL-XL cluster (1=most important, unbounded upward) to a
    bounded graded-relevance score for NDCG, where `max_tier` is the grade
    given to cluster 1. Clusters beyond max_tier collapse to grade 1 (still
    relevant, just the lowest tier) rather than 0 -- they're rare but
    genuinely gold, so they shouldn't be scored as irrelevant.
    """
    if cluster is None:
        return 0
    tier = min(int(cluster), max_tier)
    return max_tier + 1 - tier


def _dcg(relevances: list[float]) -> float:
    # Linear gain (not 2**rel - 1): cluster tiers are ordinal ranks, not
    # counts, so a linear discount is the more defensible gain function here.
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(predicted: list[str], gold_relevance: dict[str, int],
              k: int | None = None) -> float | None:
    """NDCG@k of `predicted` (ordered skill list, most relevant first)
    against `gold_relevance` ({skill: grade}, from cluster_to_relevance).

    IDCG is computed from ALL of `gold_relevance` -- including gold skills
    `predicted` never had a chance to rank because retrieval missed them --
    so this is deployment-faithful: it penalises retrieval misses, not just
    LLM mis-ranking. Pass a `gold_relevance` restricted to the candidate list
    to instead isolate ranking quality given the retriever's ceiling.

    Returns None if `gold_relevance` is empty (NDCG undefined for a sentence
    with no gold skills) -- callers should skip such sentences.
    """
    if not gold_relevance:
        return None
    order = predicted if k is None else predicted[:k]
    dcg = _dcg([gold_relevance.get(s, 0) for s in order])
    ideal = sorted(gold_relevance.values(), reverse=True)
    if k is not None:
        ideal = ideal[:k]
    idcg = _dcg(ideal)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(predicted: list[str], gold: set[str],
                           k: int | None = None) -> float | None:
    """Binary-relevance Average Precision@k of `predicted` (ordered, most
    relevant first) against `gold` (unordered set of relevant skills).

    Complements ndcg_at_k: MAP ignores the graded cluster relevance (a skill
    is either gold or not) and instead measures how early the ranking
    surfaces *any* gold skill, which is a more standard, threshold-free view
    of ranking quality. Returns None if `gold` is empty (undefined), same
    convention as ndcg_at_k -- callers should skip such sentences.
    """
    if not gold:
        return None
    order = predicted if k is None else predicted[:k]
    hits, ap = 0, 0.0
    for rank, skill in enumerate(order, 1):
        if skill in gold:
            hits += 1
            ap += hits / rank
    denom = min(len(gold), k) if k is not None else len(gold)
    return ap / denom if denom else 0.0


def deployment_score(graded_score: float | None, predicted: list[str]) -> float:
    """Turn an ndcg_at_k/average_precision_at_k score (None for a no-gold
    sentence, since ranking quality is undefined when nothing is relevant)
    into a full-population deployment score.

    For a no-gold sentence there's no ordering to grade, so this is binary
    rather than graded: 1.0 if the pipeline correctly predicted nothing,
    0.0 if it predicted anything. Averaging this alongside the graded score
    for gold-bearing sentences gives a single metric over the *whole*
    sentence population, which is what you need to see a gate's benefit:
    restricting to gold-bearing sentences only (as ndcg_at_k /
    average_precision_at_k do on their own) can never reflect a gate
    correctly suppressing no-gold noise.
    """
    if graded_score is not None:
        return graded_score
    return 1.0 if not predicted else 0.0


# ---------------------------------------------------------------------------
# Set metrics: micro P/R/F1 in two no-skill regimes
# ---------------------------------------------------------------------------


def score_micro(queries: list[dict], pred_sets: list[set[str]],
                mode: str) -> dict:
    """Micro P/R/F1 over sentences, ranked lists treated as sets.

    mode='all'           : include every sentence; predictions on no-gold
                           sentences are false positives (deployment-faithful).
    mode='skill_bearing' : skip sentences whose gold set is empty.
    """
    tp = fp = fn = 0
    n_pred = 0
    n_used = 0
    for q, preds in zip(queries, pred_sets):
        gold = q["gold"]
        if mode == "skill_bearing" and not gold:
            continue
        n_used += 1
        n_pred += len(preds)
        tp += len(preds & gold)
        fp += len(preds - gold)
        fn += len(gold - preds)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "n_sentences": n_used,
            "avg_predictions": (n_pred / n_used) if n_used else 0.0}


def both_modes(queries: list[dict], pred_sets: list[set[str]]) -> dict:
    """score_micro in both no-skill regimes, keyed by mode."""
    return {"all": score_micro(queries, pred_sets, "all"),
            "skill_bearing": score_micro(queries, pred_sets, "skill_bearing")}


# ---------------------------------------------------------------------------
# Ranking metrics: NDCG@k / MAP@k in three variants
# ---------------------------------------------------------------------------


def ranking_score_diagnostics(queries: list[dict], candidate_lists: list[list[str]],
                              ranked_preds: list[list[str]], k_values: list[int],
                              max_cluster_tier: int) -> dict:
    """NDCG@k and MAP@k, each in three variants:

    *_end_to_end@k       -- skill-bearing sentences only; IDCG/AP-denominator
                            from ALL gold skills, so retrieval misses are
                            penalised too. Deployment-faithful *conditional on
                            the sentence actually having skills*.
    *_given_candidates@k -- skill-bearing sentences only; IDCG/AP-denominator
                            from only the gold skills present in the
                            candidate list actually shown to the LLM,
                            isolating ranking quality from the retriever's
                            recall ceiling.
    *_deployment@k       -- ALL sentences, including no-gold ones. A no-gold
                            sentence scores 1.0 if the pipeline predicted
                            nothing (correct) or 0.0 if it predicted anything
                            (see deployment_score). Use this variant to see a
                            gate's benefit: the other two variants only ever
                            look at skill-bearing sentences, so a gate
                            correctly suppressing no-gold noise is
                            structurally invisible to them.

    NDCG uses the graded cluster relevance; MAP treats every gold skill as
    equally relevant (binary), so the two can disagree about *how much* a
    mis-ranking costs even when they agree on P/R/F1.
    """
    n = len(queries)
    rel_idx = [i for i, q in enumerate(queries) if q["gold_clusters"]]
    out = {"n_relevant_sentences": len(rel_idx), "n_sentences": n}

    for k in k_values:
        ndcg_e2e_all: list[float | None] = []
        map_e2e_all: list[float | None] = []
        ndcg_given, map_given = [], []

        for i in range(n):
            gold_clusters = queries[i]["gold_clusters"]
            full_rel = {s: cluster_to_relevance(c, max_cluster_tier)
                        for s, c in gold_clusters.items()}
            ndcg_e2e_all.append(ndcg_at_k(ranked_preds[i], full_rel, k))
            map_e2e_all.append(average_precision_at_k(ranked_preds[i], queries[i]["gold"], k))

            if gold_clusters:  # skill-bearing -> also compute the ceiling-isolated variant
                cand_set = set(candidate_lists[i])
                given_rel = {s: r for s, r in full_rel.items() if s in cand_set}
                # given_rel (and so given_gold) can be empty here -- retrieval
                # may have missed every one of this sentence's gold skills --
                # in which case both return None (undefined), same as e2e.
                given_gold = set(given_rel)
                ndcg_given.append(ndcg_at_k(ranked_preds[i], given_rel, k))
                map_given.append(average_precision_at_k(ranked_preds[i], given_gold, k))

        ndcg_e2e = [s for s in ndcg_e2e_all if s is not None]
        map_e2e = [s for s in map_e2e_all if s is not None]
        ndcg_given = [s for s in ndcg_given if s is not None]
        map_given = [s for s in map_given if s is not None]
        out[f"ndcg_end_to_end@{k}"] = float(np.mean(ndcg_e2e)) if ndcg_e2e else 0.0
        out[f"map_end_to_end@{k}"] = float(np.mean(map_e2e)) if map_e2e else 0.0
        out[f"ndcg_given_candidates@{k}"] = float(np.mean(ndcg_given)) if ndcg_given else 0.0
        out[f"map_given_candidates@{k}"] = float(np.mean(map_given)) if map_given else 0.0

        out[f"ndcg_deployment@{k}"] = float(np.mean(
            [deployment_score(s, ranked_preds[i]) for i, s in enumerate(ndcg_e2e_all)]))
        out[f"map_deployment@{k}"] = float(np.mean(
            [deployment_score(s, ranked_preds[i]) for i, s in enumerate(map_e2e_all)]))

    return out


# ---------------------------------------------------------------------------
# Retriever diagnostics
# ---------------------------------------------------------------------------


def ranking_diagnostics(queries: list[dict], top_labels: list[list[str]],
                        k_values: list[int], max_k: int) -> dict:
    """MAP@k, Recall@k (for each k in k_values), and MRR -- relevant sentences only.

    These are pure retriever quality metrics: gate and LLM are not involved.
    Include the LLM candidate count in k_values to see the retriever's recall
    ceiling for the LLM input.
    """
    rel_idx = [i for i, q in enumerate(queries) if q["gold"]]
    out = {"n_relevant_sentences": len(rel_idx)}

    rr = []
    for i in rel_idx:
        gold = queries[i]["gold"]
        first = next((r for r, lab in enumerate(top_labels[i][:max_k], 1)
                      if lab in gold), None)
        rr.append(1.0 / first if first else 0.0)
    out["mrr"] = float(np.mean(rr)) if rr else 0.0

    for k in k_values:
        aps, recalls = [], []
        for i in rel_idx:
            gold = queries[i]["gold"]
            hits, ap = 0, 0.0
            for rank, lab in enumerate(top_labels[i][:k], 1):
                if lab in gold:
                    hits += 1
                    ap += hits / rank
            denom = min(len(gold), k)
            aps.append(ap / denom if denom else 0.0)
            recalls.append(hits / len(gold))
        out[f"map@{k}"] = float(np.mean(aps)) if aps else 0.0
        out[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
    return out
