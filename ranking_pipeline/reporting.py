"""Console formatting of metric dicts and the per-sentence prediction trace
written next to the aggregate results.
"""

import json
import os

from .metrics import cluster_to_relevance


def format_micro(m: dict, primary: str) -> str:
    """One-line summary of a both_modes() dict, led by the primary regime."""
    p = m[primary]
    other = "skill_bearing" if primary == "all" else "all"
    return (f"P={p['precision']:.4f} R={p['recall']:.4f} F1={p['f1']:.4f}"
            f" avg_preds={p['avg_predictions']:.2f}"
            f"  ({other} F1={m[other]['f1']:.4f})")


def format_ranking_scores(d: dict, metric: str, k_values: list[int]) -> str:
    """One-line summary of a ranking_score_diagnostics() dict for one metric
    ('ndcg' or 'map'), showing all three variants at each k."""
    return " ".join(
        f"{metric.upper()}@{k}["
        f"e2e={d[f'{metric}_end_to_end@{k}']:.4f} "
        f"given_cand={d[f'{metric}_given_candidates@{k}']:.4f} "
        f"deploy={d[f'{metric}_deployment@{k}']:.4f}]"
        for k in k_values
    )


def write_ranking_predictions(path, queries, gate_probs, gate_mask,
                              candidate_lists, top_scores, ranked_preds,
                              llm_candidates, max_cluster_tier):
    """Write one JSON record per sentence with the full pipeline trace.

    Each record contains:
      sentence       -- original text
      gate_score     -- gate probability (null if no gate used)
      gate_decision  -- 1/0 gate accept/reject (null if no gate used)
      retrieved      -- the candidates shown to the LLM, with retrieval scores
      llm_ranked     -- the LLM's ranking, annotated with each skill's gold
                        cluster and NDCG relevance grade (0 = not gold)
      predicted      -- the ranked skills as a plain list
      gold           -- ground-truth skills (empty = no label)
      gold_clusters  -- {skill: cluster} graded labels
    """
    records = []
    for i, q in enumerate(queries):
        cands = candidate_lists[i]
        retr_sc = top_scores[i][:llm_candidates]
        retrieved = [{"skill": s, "retrieval_score": round(float(sc), 6)}
                     for s, sc in zip(cands, retr_sc)]
        gold_clusters = q["gold_clusters"]
        ranked = [
            {
                "rank": pos + 1,
                "skill": s,
                "gold_cluster": gold_clusters.get(s),
                "relevance_grade": cluster_to_relevance(gold_clusters[s], max_cluster_tier)
                                   if s in gold_clusters else 0,
            }
            for pos, s in enumerate(ranked_preds[i])
        ]
        records.append({
            "doc_id": q["doc_id"],
            "sentence": q["sentence"],
            "gate_score": round(float(gate_probs[i]), 6) if gate_probs is not None else None,
            "gate_decision": int(gate_mask[i]) if gate_mask is not None else None,
            "retrieved": retrieved,
            "llm_ranked": ranked,
            "predicted": list(ranked_preds[i]),
            "gold": sorted(q["gold"]),
            "gold_clusters": gold_clusters,
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False, default=str)
    print(f"  [predictions] {len(records)} records -> {path}")
