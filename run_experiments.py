"""Sequential driver for the few-shot experiment grid.

Runs llm_ranking_pipeline.py once per configuration below, writing each run's
aggregate results to results/<name>.json and its per-sentence traces to
results/predictions/<name>/. Each config gets its own predictions directory
because prediction filenames do not encode few-shot settings -- consecutive
runs would otherwise overwrite each other.

The grid varies only the few-shot composition (--n-shot ranking demos vs
--fewshot-no-gold-shots rejection demos); everything else -- retriever, gate,
LLM, dataset -- is held fixed. All grid runs use the LLM-refined gate; the
single zero_shot_plain_gate run measures the refined-vs-plain gate delta,
which applies to every config since the gate mask is independent of the
few-shot settings (see llm_ranking_pipeline.py: gated pipelines are derived
by masking a shared LLM pass).

Prerequisite: the local vLLM server must be up (see README). The suite is
resumable -- configs whose results file already exists are skipped unless
--force is given.

Usage:
    python run_experiments.py                 # run everything missing
    python run_experiments.py --dry-run       # print the commands only
    python run_experiments.py --only fs_n3_ng2 fs_n3_ng3
    python run_experiments.py --force --only zero_shot
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

RETRIEVER = "TechWolf/ConTeXT-Skill-Extraction-base"
GATE = "Aleksandruz/Binary-Relevance-RoBERTa"
FEWSHOT_POOL = "data/ranking_fewshot_pool_ConTeXT.json"

# Each retriever's grid lives in its own subfolder (results/<RUN_TAG>/ and
# results/predictions/<RUN_TAG>/), so switching RETRIEVER above never
# collides with -- or overwrites -- a previous retriever's finished runs.
# Remember to switch FEWSHOT_POOL together with RETRIEVER: demo candidate
# lists are retriever-specific (rebuild with prepare_ranking_fewshot.py).
RUN_TAG = RETRIEVER.split("/")[-1]

# (name, n_shot, no_gold_shots, gate_llm_refine)
CONFIGS: list[tuple[str, int, int, bool]] = [
    ("zero_shot",            0, 0, True),
    ("fs_n1_ng0",            1, 0, True),
    ("fs_n1_ng1",            1, 1, True),
    ("fs_n0_ng1",            0, 1, True),
    ("fs_n3_ng0",            3, 0, True),
    ("fs_n0_ng3",            0, 3, True),
    ("fs_n3_ng2",            3, 2, True),
    ("fs_n3_ng3",            3, 3, True),
    ("fs_n2_ng3",            2, 3, True),
    # Plain (non-refined) gate reference: the gate mask is few-shot
    # independent, so this single run quantifies refined-vs-plain for the
    # whole grid. Fully served from the LLM cache after zero_shot has run.
    ("zero_shot_plain_gate", 0, 0, False),
]


def plain_gate_variant(configs: list[tuple[str, int, int, bool]]
                       ) -> list[tuple[str, int, int, bool]]:
    """Rewrite the grid to run WITHOUT the LLM gate refinement, under
    '<name>_plain' result names so the refined runs are never overwritten.
    The zero_shot_plain_gate reference is dropped (it would duplicate
    zero_shot_plain). Fully served from the LLM ranking cache when the
    refined grid has already run -- the gate config never affects the
    ranking calls."""
    return [(f"{name}_plain", n, ng, False)
            for name, n, ng, refine in configs if refine]


def build_command(name: str, n_shot: int, no_gold_shots: int,
                  gate_llm_refine: bool) -> list[str]:
    cmd = [
        sys.executable, "llm_ranking_pipeline.py",
        "--retrievers", RETRIEVER,
        "--gate", GATE,
        "--llms", "qwen-local",
        "--dataset", "TechWolf/Skill-XL", "--split", "test",
        "--esco-version", "1.1.0", "--esco-language", "en",
        "--llm-candidates", "20",
        "--k", "5", "10",
        "--output", f"results/{RUN_TAG}/{name}.json",
        "--predictions-dir", f"results/predictions/{RUN_TAG}/{name}",
    ]
    if gate_llm_refine:
        cmd.append("--gate-llm-refine")
    if n_shot or no_gold_shots:
        cmd += ["--n-shot", str(n_shot),
                "--fewshot-no-gold-shots", str(no_gold_shots),
                "--fewshot-pool", FEWSHOT_POOL]
    return cmd


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the commands without executing anything")
    p.add_argument("--only", nargs="+", default=None, metavar="NAME",
                   help="Run only these config names "
                        f"(choices: {[c[0] for c in CONFIGS]})")
    p.add_argument("--force", action="store_true",
                   help="Re-run configs whose results file already exists")
    p.add_argument("--plain-gate", action="store_true",
                   help="Run the grid WITHOUT --gate-llm-refine, saving to "
                        "separate '<name>_plain' files (never overwrites the "
                        "refined runs; cheap when the refined grid is cached)")
    args = p.parse_args()

    selected = plain_gate_variant(CONFIGS) if args.plain_gate else CONFIGS
    if args.only:
        known = {c[0] for c in selected}
        bad = [n for n in args.only if n not in known]
        if bad:
            raise SystemExit(f"Unknown config names {bad}. Choices: {sorted(known)}")
        selected = [c for c in selected if c[0] in args.only]

    outcomes: dict[str, str] = {}
    for name, n_shot, no_gold_shots, refine in selected:
        cmd = build_command(name, n_shot, no_gold_shots, refine)
        result_path = RESULTS_DIR / RUN_TAG / f"{name}.json"

        if args.dry_run:
            print(" ".join(cmd))
            continue
        if result_path.exists() and not args.force:
            print(f"[skip] {name}: {result_path} exists (use --force to re-run)")
            outcomes[name] = "skipped"
            continue

        print(f"\n{'#' * 70}\n# {name}\n{'#' * 70}")
        proc = subprocess.run(cmd, cwd=ROOT)
        outcomes[name] = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"

    if args.dry_run:
        return 0

    print(f"\n{'=' * 70}\nSuite summary\n{'=' * 70}")
    for name, status in outcomes.items():
        print(f"  {name:<24} {status}")
    failed = [n for n, s in outcomes.items() if s.startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} config(s) failed; re-run with: "
              f"python run_experiments.py --only {' '.join(failed)}")
        return 1
    print("\nNext steps: summarize with\n"
          f"  python summarize_results.py results/{RUN_TAG}/*.json\n"
          "  (or results/**/*.json to compare retrievers)\n"
          "then run the follow-ups for the best config (see README).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
