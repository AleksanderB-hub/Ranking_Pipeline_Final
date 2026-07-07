"""Cosine-threshold selection baseline for the LLM ranking stage.

Instead of asking an LLM which retrieved candidates the sentence expresses,
keep every candidate whose retrieval cosine score is >= tau, in cosine order.
tau is tuned on the validation split (maximising micro-F1) and applied to the
test split. This answers: "is the LLM ranking stage needed, or does a tuned
similarity cutoff on the retriever's own scores do the job?"

The baseline implicitly gates as well: a sentence whose top-1 score falls
below tau receives an empty prediction, so the run also reports the
implicit-gate classification metrics (sentence kept iff top-1 >= tau) for
comparison with the dedicated binary relevance gate.

The output JSON uses the same schema as llm_ranking_pipeline.py results
(the baseline appears as a pseudo-LLM named "cos-threshold"), so
summarize_results.py includes it in the comparison table automatically.

Example:
    python threshold_ranking_baseline.py \\
        --retriever Aleksandruz/skillmatch-mpnet-curriculum-retriever \\
        --output results/skillmatch-mpnet-curriculum-retriever/cos_threshold.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ranking_pipeline.data import build_queries_with_clusters, load_skill_vocab
from ranking_pipeline.gate import gate_classification_metrics
from ranking_pipeline.metrics import (
    both_modes,
    ranking_diagnostics,
    ranking_score_diagnostics,
    score_micro,
)
from ranking_pipeline.reporting import format_micro, format_ranking_scores
from ranking_pipeline.retrieval import retrieve_topk


def select_by_threshold(top_labels: list[list[str]], top_scores, n_cand: int,
                        tau: float) -> list[list[str]]:
    """Keep candidates scoring >= tau, in the retriever's cosine order."""
    return [[lab for lab, sc in zip(labels[:n_cand], scores[:n_cand]) if sc >= tau]
            for labels, scores in zip(top_labels, top_scores)]


def tune_tau(queries: list[dict], top_labels: list[list[str]], top_scores,
             n_cand: int, mode: str) -> tuple[float, float]:
    """Grid-search tau over the observed score quantiles, maximising
    micro-F1 in the given no-skill regime. Returns (best_tau, best_f1)."""
    all_scores = np.concatenate([np.asarray(s[:n_cand]) for s in top_scores])
    taus = np.unique(np.quantile(all_scores, np.linspace(0.0, 1.0, 401)))
    best_tau, best_f1 = float(taus[0]), -1.0
    for tau in taus:
        preds = select_by_threshold(top_labels, top_scores, n_cand, tau)
        f1 = score_micro(queries, [set(p) for p in preds], mode)["f1"]
        if f1 > best_f1:
            best_tau, best_f1 = float(tau), f1
    return best_tau, best_f1


def retrieve_for_split(model, retriever: str, queries: list[dict], args,
                       max_k: int):
    vocab = load_skill_vocab(args.esco_version, args.esco_language, queries,
                             add_missing_gold=not args.no_add_missing_gold)
    top_idx, top_scores = retrieve_topk(model, retriever, queries, vocab,
                                        max_k, args.cache_dir, args.batch_size)
    top_labels = [[vocab[j] for j in row] for row in top_idx]
    return top_labels, top_scores


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retriever", required=True)
    p.add_argument("--llm-candidates", type=int, default=20,
                   help="Candidate pool depth, matching the LLM pipeline (default 20)")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10])
    p.add_argument("--max-cluster-tier", type=int, default=5)
    p.add_argument("--tune-mode", choices=["all", "skill_bearing"], default="all",
                   help="No-skill regime whose micro-F1 tau is tuned on (default all)")
    p.add_argument("--dev-dataset", default="./data/development.csv",
                   help="Validation split used to tune tau")
    p.add_argument("--dataset", default="TechWolf/Skill-XL")
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default=None)
    p.add_argument("--esco-version", default="1.1.0")
    p.add_argument("--esco-language", default="en")
    p.add_argument("--no-add-missing-gold", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=".emb_cache")
    p.add_argument("--output", default="results/cos_threshold.json")
    args = p.parse_args()

    max_retrieve = max(max(args.k), args.llm_candidates)
    diag_k = sorted(set(args.k) | {args.llm_candidates})
    n_cand = args.llm_candidates

    model = SentenceTransformer(args.retriever, device=args.device)

    # ---- Tune tau on the validation split ----
    print(f"[tune] retrieving on {args.dev_dataset}")
    dev_queries = build_queries_with_clusters(args.dev_dataset, None, args.subset)
    dev_labels, dev_scores = retrieve_for_split(model, args.retriever,
                                                dev_queries, args, n_cand)
    tau, dev_f1 = tune_tau(dev_queries, dev_labels, dev_scores, n_cand, args.tune_mode)
    print(f"[tune] tau={tau:.6f} (dev micro-F1[{args.tune_mode}]={dev_f1:.4f})")

    # ---- Evaluate on the test split ----
    target_split = None if args.dataset.endswith(".csv") else args.split
    queries = build_queries_with_clusters(args.dataset, target_split, args.subset)
    top_labels, top_scores = retrieve_for_split(model, args.retriever,
                                                queries, args, max_retrieve)
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    candidate_lists = [top_labels[i][:n_cand] for i in range(len(queries))]
    preds = select_by_threshold(top_labels, top_scores, n_cand, tau)

    micro = both_modes(queries, [set(pr) for pr in preds])
    scores = ranking_score_diagnostics(queries, candidate_lists, preds,
                                       diag_k, args.max_cluster_tier)
    # Implicit gate: sentence kept iff its best candidate clears tau.
    keep_mask = np.array([len(pr) > 0 for pr in preds])
    implicit_gate = gate_classification_metrics(queries, keep_mask)

    print(f"  [retrieval->cos>={tau:.4f}]  {format_micro(micro, 'all')}")
    print(f"                             {format_ranking_scores(scores, 'ndcg', diag_k)}")
    print(f"                             {format_ranking_scores(scores, 'map', diag_k)}")
    print(f"  [implicit gate] P={implicit_gate['precision']:.4f} "
          f"R={implicit_gate['recall']:.4f} F1={implicit_gate['f1']:.4f} | "
          f"{int(keep_mask.sum())}/{len(queries)} sentences kept")

    results = {
        "config": {**vars(args), "baseline": "cos-thresh",
                   "tuned_tau": tau, "dev_f1": dev_f1},
        "n_queries": len(queries),
        "gate_classification": implicit_gate,
        "models": {
            args.retriever: {
                "diagnostics_relevant_only": ranking_diagnostics(
                    queries, top_labels, diag_k, max_retrieve),
                "baselines": {},
                "llms": {
                    "cos-threshold": {
                        "retrieval_llm": micro,
                        "retrieval_llm_scores": scores,
                    }
                },
            }
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[done] Results written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
