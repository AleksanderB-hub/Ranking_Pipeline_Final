"""Dense retrieval: rank the ESCO vocabulary against each sentence by cosine
similarity, with the (large, static) skill embeddings cached on disk.
"""

import hashlib
import os

import numpy as np
from sentence_transformers import SentenceTransformer


def encode_skills_cached(model: SentenceTransformer, model_name: str,
                         vocab: list[str], cache_dir: str,
                         batch_size: int) -> np.ndarray:
    """Encode the skill vocabulary, reusing a cached copy when available.

    The cache key covers the model name and the full vocabulary, so a changed
    vocab (e.g. injected gold labels) or a different retriever never reuses
    stale embeddings.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5(
        (model_name + "\x00" + "\x00".join(vocab)).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"skills_{key}.npy")
    if os.path.exists(cache_path):
        print(f"[encode] Loaded cached skill embeddings: {cache_path}")
        return np.load(cache_path)
    emb = model.encode(vocab, batch_size=batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(cache_path, emb)
    return emb


def retrieve_topk(model: SentenceTransformer, model_name: str,
                  queries: list[dict], vocab: list[str],
                  max_k: int, cache_dir: str,
                  batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the top max_k vocabulary skills for every query sentence.

    Returns (indices, scores), both of shape [n_queries, max_k]; scores are
    cosine similarities (embeddings are L2-normalised before the dot product).
    """
    skill_emb = encode_skills_cached(model, model_name, vocab, cache_dir, batch_size)
    sent_emb = model.encode(
        [q["sentence"] for q in queries], batch_size=batch_size,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True,
    )
    sims = sent_emb @ skill_emb.T
    top_idx = np.argsort(-sims, axis=1)[:, :max_k]
    top_scores = np.take_along_axis(sims, top_idx, axis=1)
    return top_idx, top_scores
