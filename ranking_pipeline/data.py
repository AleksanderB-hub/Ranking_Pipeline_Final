"""Dataset and candidate-vocabulary loading.

SKILL-XL rows are (document, sentence, skill) triples: a sentence appears
once per gold skill, and each gold skill carries a `cluster` grade
(1 = the skill most central to the sentence, larger = less central; values
run from 1 up to ~12 in practice, with 1-5 covering the vast majority of
rows). This module collapses those rows into unique per-sentence queries and
builds the ESCO skill vocabulary the retriever searches over.
"""

import os
import random

from datasets import load_dataset

from .esco import ESCO, Language


def build_queries_with_clusters(dataset_input: str, split: str | None,
                                subset_filter: str | None = None) -> list[dict]:
    """Collapse duplicated rows into unique (doc_id, sentence) queries.

    Each query carries:
      gold          -- set of all non-null skill labels for the sentence
      gold_clusters -- {skill: cluster}, the graded-relevance label needed
                       for NDCG. A skill's cluster is the minimum (most
                       important) value seen across its duplicate rows,
                       mirroring how `gold` unions duplicate skill rows.
      relevant_flag -- SKILL-XL's own `relevant` column (True if any
                       duplicate row was flagged relevant)

    Sentences with no gold skill are retained -- predictions on them count
    as false positives in the deployment-faithful metrics.
    """
    if os.path.exists(dataset_input) and dataset_input.endswith(".csv"):
        ds = load_dataset("csv", data_files=dataset_input, split=split)
        # Local CSVs have no HF split metadata; datasets puts them in 'train'.
        ds = ds["train"]
    else:
        ds = load_dataset(dataset_input, split=split)

    if subset_filter is not None and "subset" in ds.column_names:
        ds = ds.filter(lambda r: r["subset"] == subset_filter)

    grouped: dict[tuple, dict] = {}
    for row in ds:
        key = (row["ID"], row["sentence"])
        if row["sentence"] is None or not str(row["sentence"]).strip():
            continue
        entry = grouped.setdefault(
            key,
            {
                "doc_id": row["ID"],
                "sentence": row["sentence"],
                "gold": set(),
                "gold_clusters": {},
                "relevant_flag": bool(row["relevant"]),
            },
        )
        skill = row.get("skill")
        if skill:
            entry["gold"].add(skill)
            cluster = row.get("cluster")
            if cluster is not None:
                prev = entry["gold_clusters"].get(skill)
                entry["gold_clusters"][skill] = min(prev, cluster) if prev is not None else cluster
        entry["relevant_flag"] = entry["relevant_flag"] or bool(row["relevant"])

    queries = list(grouped.values())
    n_no_skill = sum(1 for q in queries if not q["gold"])
    print(
        f"[data] {len(queries)} unique sentences | "
        f"{sum(1 for q in queries if q['gold'])} with gold skills | "
        f"{n_no_skill} with no skill label"
    )
    return queries


def balance_queries(queries: list[dict], seed: int) -> list[dict]:
    """Downsample to an even 50/50 split of gold-bearing vs no-gold sentences.

    The majority class is randomly downsampled (seeded, so runs are
    reproducible) to the size of the minority class; the minority class is
    kept whole. Original dataset order is preserved, so the same seed always
    yields the same evaluation set in the same order.

    Use this to measure pipeline performance under an artificial balanced
    class prior, as opposed to the dataset's natural skew towards no-skill
    sentences. Note that this changes what the 'all'-regime and deployment
    metrics mean: they become scores under the 50/50 prior, not the
    deployment-faithful one.
    """
    gold_idx = [i for i, q in enumerate(queries) if q["gold"]]
    no_gold_idx = [i for i, q in enumerate(queries) if not q["gold"]]
    n = min(len(gold_idx), len(no_gold_idx))
    rng = random.Random(seed)
    if len(gold_idx) <= len(no_gold_idx):
        keep = set(gold_idx) | set(rng.sample(no_gold_idx, n))
    else:
        keep = set(no_gold_idx) | set(rng.sample(gold_idx, n))
    balanced = [q for i, q in enumerate(queries) if i in keep]
    print(f"[data] even split (seed={seed}): {n} gold + {n} no-gold "
          f"= {len(balanced)} sentences (downsampled from {len(queries)})")
    return balanced


def load_skill_vocab(esco_version: str, esco_language: str, queries: list[dict],
                     add_missing_gold: bool) -> list[str]:
    """Load the ESCO skill vocabulary the retriever searches over.

    With add_missing_gold=True (the default behaviour), any gold label that is
    absent from the ESCO vocabulary is appended to the pool -- otherwise those
    labels could never be retrieved and the recall ceiling would sit below 1.0.
    """
    lang = getattr(Language, esco_language.upper())
    esco = ESCO(version=esco_version, language=lang, auto_download=True)
    vocab = list(esco.get_skills_vocabulary())
    print(f"[vocab] ESCO {esco_version} ({esco_language}): {len(vocab)} skills loaded")
    vocab_set = set(vocab)

    gold_all = set().union(*(q["gold"] for q in queries)) if queries else set()
    missing = sorted(gold_all - vocab_set)
    if missing:
        print(f"[vocab] WARNING: {len(missing)} gold labels not in skill vocab "
              f"(e.g. {missing[:5]})")
        if add_missing_gold:
            vocab.extend(missing)
            print(f"[vocab] Added missing gold labels (pool size now {len(vocab)})")
        else:
            print("[vocab] These labels can never be retrieved -> recall ceiling < 1.0")
    print(f"[vocab] Final candidate pool: {len(vocab)} skills")
    return vocab
