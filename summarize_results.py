"""Summarise one or more llm_ranking_results JSON files into a single
comparison table (GitHub markdown to stdout, optionally CSV).

One row per (results file x retriever x LLM x pipeline). The config label is
derived from each file's stored config block, so files can be renamed freely:
'zero' or 'n<ranking demos>+ng<rejection demos>', with '+even<seed>' when the
run used --data-even and '+think' when it used --qwen-thinking. The gate
column shows 'refined' (--gate-llm-refine), 'plain' (gate only), or 'none'.

Usage:
    python summarize_results.py results/*.json
    python summarize_results.py results/*.json --k 10 --csv results/summary.csv
    python summarize_results.py results/zero_shot.json --include-baselines
"""

import argparse
import csv
import glob
import json
import sys

COLUMNS = [
    "config", "retriever", "gate", "pipeline", "P", "R", "F1(all)", "F1(sb)",
    "avg_preds", "NDCG e2e", "NDCG gc", "NDCG dep",
    "MAP e2e", "MAP gc", "MAP dep", "gate F1",
]


def config_label(cfg: dict) -> str:
    if cfg.get("baseline"):
        label = cfg["baseline"]
    else:
        n = cfg.get("n_shot", 0)
        ng = cfg.get("fewshot_no_gold_shots", 0)
        label = "zero" if (n == 0 and ng == 0) else f"n{n}+ng{ng}"
    if cfg.get("data_even"):
        label += f"+even{cfg.get('data_even_seed', 0)}"
    if cfg.get("qwen_thinking"):
        label += "+think"
    return label


def gate_label(cfg: dict) -> str:
    if not cfg.get("gate"):
        return "none"
    return "refined" if cfg.get("gate_llm_refine") else "plain"


def fmt(x, digits=4) -> str:
    return "" if x is None else f"{x:.{digits}f}"


def metric_row(config: str, retriever: str, gate: str, pipeline: str,
               micro: dict | None, scores: dict | None, k: int, gate_f1) -> dict:
    """Build one table row; micro is a both_modes() dict, scores a
    ranking_score_diagnostics() dict -- either may be None (baselines)."""
    row = {c: "" for c in COLUMNS}
    row.update({"config": config, "retriever": retriever, "gate": gate,
                "pipeline": pipeline, "gate F1": fmt(gate_f1)})
    if micro is not None:
        row["P"] = fmt(micro["all"]["precision"])
        row["R"] = fmt(micro["all"]["recall"])
        row["F1(all)"] = fmt(micro["all"]["f1"])
        row["F1(sb)"] = fmt(micro["skill_bearing"]["f1"])
        row["avg_preds"] = fmt(micro["all"]["avg_predictions"], 2)
    if scores is not None:
        row["NDCG e2e"] = fmt(scores.get(f"ndcg_end_to_end@{k}"))
        row["NDCG gc"] = fmt(scores.get(f"ndcg_given_candidates@{k}"))
        row["NDCG dep"] = fmt(scores.get(f"ndcg_deployment@{k}"))
        row["MAP e2e"] = fmt(scores.get(f"map_end_to_end@{k}"))
        row["MAP gc"] = fmt(scores.get(f"map_given_candidates@{k}"))
        row["MAP dep"] = fmt(scores.get(f"map_deployment@{k}"))
    return row


def rows_for_file(path: str, k: int, include_baselines: bool) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not (isinstance(data, dict) and "config" in data and "models" in data):
        # Not a results file (e.g. a per-sentence prediction trace or a
        # few-shot pool caught by a recursive glob) -- skip it.
        print(f"[warn] skipping {path}: not a results file", file=sys.stderr)
        return []
    cfg = data["config"]
    config = config_label(cfg)
    gate = gate_label(cfg)
    gate_cls = data.get("gate_classification") or {}
    gate_f1 = gate_cls.get("f1")

    rows = []
    for retriever, entry in data["models"].items():
        retr = retriever.split("/")[-1]
        baselines = entry.get("baselines", {})
        if include_baselines:
            for key, micro in baselines.items():
                if key == "retrieval_order_scores":
                    rows.append(metric_row(config, retr, gate, "retrieval order",
                                           None, micro, k, gate_f1))
                else:
                    rows.append(metric_row(config, retr, gate, key,
                                           micro, None, k, gate_f1))
        for llm, llm_entry in entry.get("llms", {}).items():
            rows.append(metric_row(
                config, retr, gate, f"retr->llm [{llm}]",
                llm_entry.get("retrieval_llm"),
                llm_entry.get("retrieval_llm_scores"), k, gate_f1))
            if "gate_retrieval_llm" in llm_entry:
                rows.append(metric_row(
                    config, retr, gate, f"gate->retr->llm [{llm}]",
                    llm_entry["gate_retrieval_llm"],
                    llm_entry.get("gate_retrieval_llm_scores"), k, gate_f1))
    return rows


def print_markdown(rows: list[dict]) -> None:
    widths = {c: max(len(c), *(len(r[c]) for r in rows)) if rows else len(c)
              for c in COLUMNS}
    print("| " + " | ".join(c.ljust(widths[c]) for c in COLUMNS) + " |")
    print("|" + "|".join("-" * (widths[c] + 2) for c in COLUMNS) + "|")
    for r in rows:
        print("| " + " | ".join(r[c].ljust(widths[c]) for c in COLUMNS) + " |")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+",
                   help="Results JSON files (globs OK); row order follows input order")
    p.add_argument("--k", type=int, default=5,
                   help="Which k to show for the NDCG/MAP columns (default 5; "
                        "must be one of the run's --k values or --llm-candidates)")
    p.add_argument("--csv", default=None, metavar="PATH",
                   help="Also write the table to a CSV file")
    p.add_argument("--include-baselines", action="store_true",
                   help="Add each file's retrieval@k / retrieval-order baseline rows "
                        "(identical across configs that share retriever and data)")
    args = p.parse_args()

    paths: list[str] = []
    for pattern in args.files:
        matched = sorted(glob.glob(pattern, recursive=True))
        if not matched:
            print(f"[warn] no files match {pattern!r}", file=sys.stderr)
        paths.extend(matched)
    if not paths:
        raise SystemExit("No results files found.")

    rows = []
    for path in paths:
        rows.extend(rows_for_file(path, args.k, args.include_baselines))

    print(f"k={args.k} | {len(paths)} results file(s)\n")
    print_markdown(rows)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[csv] table written to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
