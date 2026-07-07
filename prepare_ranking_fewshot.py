"""Build a pool of ranking few-shot examples for llm_ranking_pipeline.py from
the SKILL-XL validation split (data/development.csv by default).

Why a separate prep step: the ranking LLM sees a numbered candidate list per
sentence (the retriever's top-N), and a good few-shot demonstration has to
show the *complete* correct ranking for its own candidate list. If retrieval
missed one of a demo sentence's gold skills, that skill couldn't appear
anywhere in the demonstrated answer -- silently teaching the model an
incomplete ranking. So for each validation sentence this script retrieves the
same top-N candidates the main pipeline would show, injects any gold skill
retrieval missed, and derives the correctly-ordered answer (see
ranking_pipeline.fewshot.build_pool for details).

The pool includes BOTH gold-bearing sentences (n_gold > 0, a real ranking to
demonstrate) and no-gold sentences (n_gold == 0, "ranked": [] -- a rejection
to demonstrate). llm_ranking_pipeline.py's default few-shot selection
(--n-shot) only ever draws from the gold-bearing subset;
--fewshot-no-gold-shots opts into also including rejection demonstrations.
No-gold entries carry `relevant_flag` (from SKILL-XL's own `relevant` column)
so that selection can prefer "hard" negatives -- sentences that read as
job-relevant but map to no ESCO skill -- over obviously unrelated filler.

The output JSON is a flat pool; llm_ranking_pipeline.py samples examples from
it at runtime (--fewshot-selection diverse|random).

Example:
    python prepare_ranking_fewshot.py \\
        --retriever Aleksandruz/skillmatch-mpnet-curriculum-retriever \\
        --dataset ./data/development.csv \\
        --esco-version 1.1.0 --esco-language en \\
        --llm-candidates 20 \\
        --output data/ranking_fewshot_pool.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ranking_pipeline.data import build_queries_with_clusters, load_skill_vocab
from ranking_pipeline.fewshot import build_pool
from ranking_pipeline.retrieval import retrieve_topk


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retriever", required=True,
                   help="Sentence-transformer retriever; must match the one "
                        "llm_ranking_pipeline.py will use, since the pool's "
                        "candidate lists are retriever-specific")
    p.add_argument("--llm-candidates", type=int, default=20,
                   help="Top-N retrieval candidates per example, before gold "
                        "injection (default 20, matches llm_ranking_pipeline.py)")
    p.add_argument("--dataset", default="./data/development.csv",
                   help="HF dataset name OR path to a local .csv file "
                        "(default: the SKILL-XL validation split)")
    p.add_argument("--split", default=None)
    p.add_argument("--subset", default=None)
    p.add_argument("--esco-version", default="1.1.0")
    p.add_argument("--esco-language", default="en")
    p.add_argument("--no-add-missing-gold", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=".emb_cache",
                   help="Shared with the main pipeline's retrieval cache "
                        "(default .emb_cache) so embeddings aren't recomputed")
    p.add_argument("--output", default="data/ranking_fewshot_pool.json")
    return p


def main():
    args = build_arg_parser().parse_args()

    target_split = None if args.dataset.endswith(".csv") else args.split
    queries = build_queries_with_clusters(args.dataset, target_split, args.subset)
    n_gold_bearing = sum(1 for q in queries if q["gold"])
    print(f"[pool] {n_gold_bearing}/{len(queries)} sentences have >=1 gold skill "
          f"(ranking demos); the rest become rejection demos (\"ranked\": [])")

    vocab = load_skill_vocab(args.esco_version, args.esco_language, queries,
                             add_missing_gold=not args.no_add_missing_gold)

    model = SentenceTransformer(args.retriever, device=args.device)
    top_idx, _ = retrieve_topk(model, args.retriever, queries, vocab,
                               args.llm_candidates, args.cache_dir, args.batch_size)
    top_labels = [[vocab[j] for j in row] for row in top_idx]
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    pool = build_pool(queries, top_labels)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2, ensure_ascii=False, default=str)

    gold_entries = [r for r in pool if r["n_gold"] > 0]
    no_gold_entries = [r for r in pool if r["n_gold"] == 0]
    n_with_missing = sum(1 for r in gold_entries if r["n_missing_injected"] > 0)
    n_hard_negatives = sum(1 for r in no_gold_entries if r["relevant_flag"])
    print(f"[pool] wrote {len(pool)} examples -> {args.output} "
          f"({len(gold_entries)} ranking demos, {len(no_gold_entries)} rejection demos)")
    if gold_entries:
        print(f"[pool] {n_with_missing}/{len(gold_entries)} ranking demos needed >=1 gold "
              f"skill injected (avg {np.mean([r['n_missing_injected'] for r in gold_entries]):.2f}/example)")
        print(f"[pool] avg distinct gold cluster tiers per ranking demo: "
              f"{np.mean([r['n_distinct_clusters'] for r in gold_entries]):.2f}")
    if no_gold_entries:
        print(f"[pool] {n_hard_negatives}/{len(no_gold_entries)} rejection demos are \"hard\" "
              f"(SKILL-XL flagged relevant despite no mapped skill)")


if __name__ == "__main__":
    sys.exit(main())
