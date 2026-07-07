"""Evaluate skill-*ranking* pipelines that use an LLM to order the retrieved
ESCO candidates by relevance, rather than just accepting/rejecting them.

For each retriever the script evaluates:

  Baselines (no LLM, retriever only):
    1. Retrieval @k              (no gate)
    2. Gate -> Retrieval @k      (gated)
    3. Retrieval order           (the retriever's own cosine-similarity
                                  ranking, used as an NDCG reference point)

  LLM pipelines (per selected LLM):
    4. Retrieval -> LLM ranking          (LLM orders the top-N candidates; no gate)
    5. Gate -> Retrieval -> LLM ranking  (gate pre-filters; LLM runs on accepted sentences)

The LLM is given the sentence plus the top-N retrieved ESCO candidates and
returns the candidates it judges relevant, *ordered most-to-least relevant,
by index* -- it never emits free-text skill names, so it cannot paraphrase
or invent ESCO labels.

SKILL-XL grades each gold skill with a `cluster` (1 = most central to the
sentence, larger = less central). This lets us score the LLM's ranking with
NDCG against that graded relevance, not just P/R/F1 (which are still
reported, treating the ranked list as a set).

Few-shot examples are not hardcoded here: --n-shot {0,1,3,5} pulls that many
demonstrations from a pool prepared ahead of time by prepare_ranking_fewshot.py
(--fewshot-pool). See that script's docstring for why a prep step is needed.

Optionally, the gate itself can be refined with an LLM cascade auditor
(--gate-llm-refine): sentences the gate is confident about (prob <
--gate-threshold or >= --gate-t-upper) are decided by the gate alone;
sentences in between are routed to --gate-audit-llm for a
relevant/non-relevant call. This only changes which sentences pass the gate --
pipelines 2 and 5 above then run on the refined accept set.

Primary metrics:
  - Micro-averaged Precision/Recall/F1 over unique sentences (ranked list
    treated as a set), in two no-skill regimes ('all' deployment-faithful
    vs 'skill_bearing').
  - NDCG@k / MAP@k against the gold cluster grades, in three variants
    (end-to-end, given-candidates, deployment) -- see
    ranking_pipeline.metrics.ranking_score_diagnostics.

Diagnostics (retriever quality, relevant sentences only): MAP@k, Recall@k for
each k plus Recall@llm-candidates (the retriever ceiling for the LLM), and MRR.

LLM calls are cached on disk keyed by (model, prompt version, sentence,
candidates); the prompt version is partitioned by --n-shot / --fewshot-selection
/ --fewshot-seed since those change the prompt content.

Example:
    python prepare_ranking_fewshot.py \\
        --retriever Aleksandruz/skillmatch-mpnet-curriculum-retriever \\
        --dataset ./data/development.csv --llm-candidates 20 \\
        --output data/ranking_fewshot_pool.json

    python llm_ranking_pipeline.py \\
        --retrievers Aleksandruz/skillmatch-mpnet-curriculum-retriever \\
        --gate Aleksandruz/Binary-Relevance-RoBERTa \\
        --llms local \\
        --dataset TechWolf/Skill-XL --split test \\
        --esco-version 1.1.0 --esco-language en \\
        --llm-candidates 20 --k 5 10 \\
        --n-shot 3 --fewshot-pool data/ranking_fewshot_pool.json \\
        --output llm_ranking_results.json
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys

import torch
from sentence_transformers import SentenceTransformer

from ranking_pipeline import config as pipeline_config
from ranking_pipeline.config import LLM_REGISTRY, load_api_keys, resolve_llms
from ranking_pipeline.data import (
    balance_queries,
    build_queries_with_clusters,
    load_skill_vocab,
)
from ranking_pipeline.fewshot import load_fewshot_examples
from ranking_pipeline.gate import RelevanceGate, gate_classification_metrics
from ranking_pipeline.llm_clients import (
    GateAuditor,
    LLMRanker,
    run_gate_cascade,
    run_ranker_over_sentences,
)
from ranking_pipeline.metrics import (
    both_modes,
    ranking_diagnostics,
    ranking_score_diagnostics,
)
from ranking_pipeline.prompts import GATE_AUDIT_PROMPT_VERSION, RANK_SYSTEM_PROMPT
from ranking_pipeline.reporting import (
    format_micro,
    format_ranking_scores,
    write_ranking_predictions,
)
from ranking_pipeline.retrieval import retrieve_topk


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--retrievers", nargs="+", required=True,
                   help="One or more sentence-transformer model names/paths")

    # Gate
    p.add_argument("--gate", default=None,
                   help="Optional binary relevance gate model name/path")
    p.add_argument("--gate-threshold", type=float, default=None,
                   help="Override gate decision threshold (default: model config or 0.5)")
    p.add_argument("--gate-llm-refine", action="store_true",
                   help="Route sentences in the gate's uncertain band "
                        "([--gate-threshold, --gate-t-upper)) to an LLM auditor "
                        "instead of accepting them outright (cascade router). "
                        "Requires --gate.")
    p.add_argument("--gate-t-upper", type=float, default=0.90,
                   help="Gate auto-accept threshold used by --gate-llm-refine; "
                        "sentences scoring below this (and >= --gate-threshold) "
                        "are sent to the LLM auditor (default 0.90)")
    p.add_argument("--gate-audit-llm", default="qwen-local",
                   help="Registry key of the LLM used for --gate-llm-refine "
                        f"(default qwen-local): {list(LLM_REGISTRY)}")
    p.add_argument("--gate-audit-max-tokens", type=int, default=150,
                   help="Max tokens for the gate auditor's CoT + decision (default 150)")

    # Few-shot
    p.add_argument("--n-shot", type=int, choices=[0, 1, 2, 3, 5], default=0,
                   help="Number of ranking few-shot demonstrations to include "
                        "(0=zero-shot, default). Drawn from --fewshot-pool's "
                        "gold-bearing entries only.")
    p.add_argument("--fewshot-no-gold-shots", type=int, default=0,
                   help="Number of rejection few-shot demonstrations "
                        "(no-gold sentences, \"ranked\": []) to ADD on top of "
                        "--n-shot, drawn from --fewshot-pool's no-gold entries "
                        "(default 0)")
    p.add_argument("--fewshot-pool", default=None,
                   help="Path to the JSON pool built by prepare_ranking_fewshot.py. "
                        "Required if --n-shot > 0 or --fewshot-no-gold-shots > 0.")
    p.add_argument("--fewshot-selection", choices=["diverse", "random"], default="diverse",
                   help="'diverse' (default): deterministically pick the pool's "
                        "most cluster-diverse ranking demos and, for rejection "
                        "demos, the 'hardest' negatives (relevant_flag=True). "
                        "'random': uniform sample with --fewshot-seed.")
    p.add_argument("--fewshot-seed", type=int, default=0)

    # LLM ranking
    p.add_argument("--llms", nargs="+", default=["local"],
                   help="'local' (default), 'all', or registry keys: "
                        f"{list(LLM_REGISTRY)}")
    p.add_argument("--llm-candidates", type=int, default=20,
                   help="Top-N retrieval candidates fed to the LLM (default 20)")
    p.add_argument("--llm-max-tokens", type=int, default=512)
    p.add_argument("--qwen-thinking", action="store_true",
                   help="Enable Qwen3 thinking mode for the local vLLM provider "
                        "(slower; use --llm-max-tokens 2048+ so the JSON isn't "
                        "truncated). Partitions the LLM cache, so thinking and "
                        "non-thinking results never mix. Ignored for API models.")

    # Scoring
    p.add_argument("--k", type=int, nargs="+", default=[5, 10],
                   help="k values for the retrieval-only baselines and NDCG@k (default 5 10)")
    p.add_argument("--max-cluster-tier", type=int, default=5,
                   help="Cluster value mapped to the highest NDCG relevance grade; "
                        "clusters beyond this collapse to the lowest tier (default 5)")
    p.add_argument("--scoring", choices=["all", "skill_bearing"], default="all",
                   help="Primary no-skill regime to print (both are stored). "
                        "'all' = deployment-faithful (default).")

    # Data
    p.add_argument("--dataset", default="TechWolf/Skill-XL",
                   help="HF dataset name OR path to a local .csv file")
    p.add_argument("--split", default="test")
    p.add_argument("--subset", default=None,
                   help="Optional value of the 'subset' column to filter on")
    p.add_argument("--esco-version", default="1.1.0")
    p.add_argument("--esco-language", default="en")
    p.add_argument("--no-add-missing-gold", action="store_true",
                   help="Do not add gold labels missing from the vocab to the pool")
    p.add_argument("--data-even", action="store_true",
                   help="Evaluate on an even 50/50 split of gold-bearing vs "
                        "no-gold sentences: the majority class is randomly "
                        "downsampled to the minority class size. Changes the "
                        "class prior the 'all'-regime and deployment metrics "
                        "are computed under.")
    p.add_argument("--data-even-seed", type=int, default=0,
                   help="Seed for the --data-even downsampling, for "
                        "reproducible splits (default 0)")

    # Runtime / output
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for retriever and gate models")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", default=".emb_cache")
    p.add_argument("--llm-cache-dir", default=".llm_cache")
    p.add_argument("--output", default="llm_ranking_results.json")
    p.add_argument("--predictions-dir", default="predictions",
                   help="Directory for per-sentence prediction JSON files "
                        "(one file per retriever x LLM x pipeline)")
    return p


def apply_gate(args, queries: list[dict], sentences: list[str]):
    """Run the relevance gate (and optionally its LLM cascade refinement)
    once -- gate decisions are retriever-independent.

    Returns (gate, gate_probs, keep_mask, gate_report, audit_meta); all None
    when no gate is configured.
    """
    if not args.gate:
        return None, None, None, None, None

    gate = RelevanceGate(args.gate, args.device, args.gate_threshold, args.batch_size)
    gate_probs = gate.predict_proba(sentences)

    if args.gate_llm_refine:
        if args.gate_audit_llm not in LLM_REGISTRY:
            raise SystemExit(
                f"Unknown --gate-audit-llm {args.gate_audit_llm!r}. "
                f"Use one of: {list(LLM_REGISTRY)}"
            )
        # Gate-alone performance before the LLM audit, for comparison.
        pre_report = gate_classification_metrics(queries, gate_probs >= gate.threshold)
        print(f"[gate] pre-refine  P={pre_report['precision']:.4f} "
              f"R={pre_report['recall']:.4f} F1={pre_report['f1']:.4f} | "
              f"{int((gate_probs >= gate.threshold).sum())}/{len(queries)} accepted")
        auditor = GateAuditor(args.gate_audit_llm, LLM_REGISTRY[args.gate_audit_llm],
                              args.llm_cache_dir, args.gate_audit_max_tokens)
        keep_mask, audit_meta = asyncio.run(run_gate_cascade(
            auditor, sentences, gate_probs, gate.threshold, args.gate_t_upper))
        audit_meta["prompt_version"] = GATE_AUDIT_PROMPT_VERSION
        audit_meta["pre_refine_classification"] = pre_report
        print(f"[gate] LLM refinement ({args.gate_audit_llm}, "
              f"prompt={GATE_AUDIT_PROMPT_VERSION}): "
              f"{audit_meta['n_audited']}/{audit_meta['n_total']} sentences "
              f"audited (threshold={gate.threshold}, t_upper={args.gate_t_upper})")
        report_label = "post-refine"
    else:
        keep_mask = gate_probs >= gate.threshold
        audit_meta = None
        report_label = "standalone"

    gate_report = gate_classification_metrics(queries, keep_mask)
    print(f"[gate] {report_label} P={gate_report['precision']:.4f} "
          f"R={gate_report['recall']:.4f} F1={gate_report['f1']:.4f} | "
          f"{int(keep_mask.sum())}/{len(queries)} accepted")
    return gate, gate_probs, keep_mask, gate_report, audit_meta


def evaluate_retriever(model_name: str, args, queries, sentences, vocab,
                       selected_llms, few_shot, prompt_version,
                       keep_mask_gated, gate_probs) -> dict:
    """Run baselines and all LLM pipelines for one retriever."""
    max_retrieve = max(max(args.k), args.llm_candidates)
    diag_k = sorted(set(args.k) | {args.llm_candidates})
    gated = keep_mask_gated is not None

    model = SentenceTransformer(model_name, device=args.device)
    top_idx, top_scores = retrieve_topk(
        model, model_name, queries, vocab, max_retrieve,
        args.cache_dir, args.batch_size)
    top_labels = [[vocab[j] for j in row] for row in top_idx]
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    entry = {
        "diagnostics_relevant_only": ranking_diagnostics(
            queries, top_labels, diag_k, max_retrieve),
        "baselines": {},
        "llms": {},
    }

    # ---- Baselines: retrieval@k, with/without gate ----
    for k in args.k:
        no_gate_preds = [set(top_labels[i][:k]) for i in range(len(queries))]
        ng = both_modes(queries, no_gate_preds)
        entry["baselines"][f"no_gate@{k}"] = ng
        print(f"  [retrieval@{k}]   no-gate  {format_micro(ng, args.scoring)}")
        if gated:
            gated_preds = [set(top_labels[i][:k]) if keep_mask_gated[i] else set()
                           for i in range(len(queries))]
            g = both_modes(queries, gated_preds)
            entry["baselines"][f"gated@{k}"] = g
            print(f"  [retrieval@{k}]     gated  {format_micro(g, args.scoring)}")

    # ---- Retrieval's own cosine-similarity order, as a reference point for
    # "did the LLM re-ranking actually help" ----
    candidate_lists = [top_labels[i][:args.llm_candidates]
                       for i in range(len(queries))]
    retrieval_scores = ranking_score_diagnostics(queries, candidate_lists, candidate_lists,
                                                 diag_k, args.max_cluster_tier)
    entry["baselines"]["retrieval_order_scores"] = retrieval_scores
    print(f"  [retrieval order]  {format_ranking_scores(retrieval_scores, 'ndcg', diag_k)}")
    print(f"  [retrieval order]  {format_ranking_scores(retrieval_scores, 'map', diag_k)}")

    # ---- LLM pipelines ----
    model_slug = model_name.replace("/", "_")
    for name in selected_llms:
        ranker = LLMRanker(name, LLM_REGISTRY[name], args.llm_cache_dir,
                           prompt_version, RANK_SYSTEM_PROMPT,
                           few_shot, args.llm_max_tokens)
        print(f"  [{name}] ranking top-{args.llm_candidates} candidates "
              f"for {len(queries)} sentences...")
        ranked_all = asyncio.run(run_ranker_over_sentences(ranker, sentences, candidate_lists))

        llm_entry = {}

        # Pipeline 4: Retrieval -> LLM ranking (no gate)
        ng = both_modes(queries, [set(r) for r in ranked_all])
        ng_scores = ranking_score_diagnostics(queries, candidate_lists, ranked_all,
                                              diag_k, args.max_cluster_tier)
        llm_entry["retrieval_llm"] = ng
        llm_entry["retrieval_llm_scores"] = ng_scores
        print(f"  [retrieval->llm]        {name:>14}  no-gate  {format_micro(ng, args.scoring)}")
        print(f"                          {name:>14}           {format_ranking_scores(ng_scores, 'ndcg', diag_k)}")
        print(f"                          {name:>14}           {format_ranking_scores(ng_scores, 'map', diag_k)}")
        write_ranking_predictions(
            os.path.join(args.predictions_dir,
                         f"{model_slug}__{name}__retrieval_llm.json"),
            queries, None, None, candidate_lists, top_scores,
            ranked_all, args.llm_candidates, args.max_cluster_tier)

        # Pipeline 5: Gate -> Retrieval -> LLM ranking
        if gated:
            ranked_gated = [ranked_all[i] if keep_mask_gated[i] else []
                            for i in range(len(queries))]
            g = both_modes(queries, [set(r) for r in ranked_gated])
            g_scores = ranking_score_diagnostics(queries, candidate_lists, ranked_gated,
                                                 diag_k, args.max_cluster_tier)
            llm_entry["gate_retrieval_llm"] = g
            llm_entry["gate_retrieval_llm_scores"] = g_scores
            print(f"  [gate->retrieval->llm]  {name:>14}    gated  {format_micro(g, args.scoring)}")
            print(f"                          {name:>14}           {format_ranking_scores(g_scores, 'ndcg', diag_k)}")
            print(f"                          {name:>14}           {format_ranking_scores(g_scores, 'map', diag_k)}")
            write_ranking_predictions(
                os.path.join(args.predictions_dir,
                             f"{model_slug}__{name}__gate_retrieval_llm.json"),
                queries, gate_probs, keep_mask_gated, candidate_lists,
                top_scores, ranked_gated, args.llm_candidates, args.max_cluster_tier)

        entry["llms"][name] = llm_entry

    d = entry["diagnostics_relevant_only"]
    print("  [diagnostics, relevant only] " + " ".join(
        f"MAP@{k}={d[f'map@{k}']:.4f} Recall@{k}={d[f'recall@{k}']:.4f}"
        for k in diag_k) + f" MRR={d['mrr']:.4f}")

    return entry


def main():
    args = build_arg_parser().parse_args()
    load_api_keys()
    if args.qwen_thinking:
        pipeline_config.QWEN_ENABLE_THINKING = True
        print("[llm] Qwen thinking mode ENABLED (cache variant think=True)")

    if (args.n_shot > 0 or args.fewshot_no_gold_shots > 0) and not args.fewshot_pool:
        raise SystemExit("--n-shot > 0 or --fewshot-no-gold-shots > 0 requires "
                         "--fewshot-pool (built by prepare_ranking_fewshot.py)")

    selected_llms = resolve_llms(args.llms)
    print(f"[llms] {selected_llms}")

    target_split = None if args.dataset.endswith(".csv") else args.split
    queries = build_queries_with_clusters(args.dataset, target_split, args.subset)
    if args.data_even:
        queries = balance_queries(queries, args.data_even_seed)
    vocab = load_skill_vocab(args.esco_version, args.esco_language, queries,
                             add_missing_gold=not args.no_add_missing_gold)
    sentences = [q["sentence"] for q in queries]

    gate, gate_probs, keep_mask_gated, gate_report, gate_audit_meta = apply_gate(
        args, queries, sentences)

    # Few-shot examples (dynamic, from a pre-built pool) and prompt version.
    # The prompt version keys the LLM cache, so anything that changes the
    # prompt content must appear in it -- including a fingerprint of the
    # selected demonstrations themselves, so a rebuilt pool file can never
    # silently reuse answers cached under the old demonstrations.
    few_shot = load_fewshot_examples(args.fewshot_pool, args.n_shot, args.fewshot_no_gold_shots,
                                     args.fewshot_selection, args.fewshot_seed)
    if few_shot:
        fs_blob = json.dumps(
            [[ex["sentence"], ex["candidates"], ex["ranked"]] for ex in few_shot],
            ensure_ascii=False)
        fs_fingerprint = hashlib.sha256(fs_blob.encode("utf-8")).hexdigest()[:8]
        prompt_version = (
            f"rank-v2_n{args.n_shot}_nng{args.fewshot_no_gold_shots}_"
            f"{args.fewshot_selection}_{args.fewshot_seed}_fs{fs_fingerprint}"
        )
    else:
        prompt_version = "rank-v2_zero"
    print(f"[prompt] {prompt_version} | {len(few_shot)} few-shot example(s) "
          f"({args.n_shot} ranking + {args.fewshot_no_gold_shots} rejection)")

    results = {
        "config": vars(args),
        "n_queries": len(queries),
        "scoring_primary": args.scoring,
        "gate_classification": gate_report,
        "gate_audit": gate_audit_meta,
        "models": {},
    }

    for model_name in args.retrievers:
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")
        results["models"][model_name] = evaluate_retriever(
            model_name, args, queries, sentences, vocab,
            selected_llms, few_shot, prompt_version,
            keep_mask_gated, gate_probs)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[done] Results written to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
