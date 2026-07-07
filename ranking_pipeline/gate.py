"""Binary relevance gate: a sequence classifier that decides, per sentence,
whether it is worth running skill extraction at all. Sentences scoring below
the decision threshold are dropped before retrieval results are used.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RelevanceGate:
    """Sentence-level relevance classifier with a configurable threshold.

    The threshold is resolved in priority order: explicit `threshold`
    argument (CLI override) > `decision_threshold` in the model config > 0.5.
    """

    def __init__(self, model_name: str, device: str, threshold: float | None,
                 batch_size: int, max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        cfg_thr = getattr(self.model.config, "decision_threshold", None)
        self.threshold = threshold if threshold is not None else (cfg_thr or 0.5)
        source = "CLI override" if threshold is not None else ("from config" if cfg_thr else "default 0.5")
        print(f"[gate] {model_name} | threshold={self.threshold} ({source})")

    @torch.no_grad()
    def predict_proba(self, sentences: list[str]) -> np.ndarray:
        """Return the per-sentence probability of being skill-relevant."""
        probs = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            enc = self.tokenizer(batch, truncation=True, padding=True,
                                 max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits
            if logits.shape[-1] == 1:
                p = torch.sigmoid(logits.squeeze(-1))
            else:
                p = torch.softmax(logits, dim=-1)[:, 1]
            probs.append(p.cpu().numpy())
        return np.concatenate(probs)


def gate_classification_metrics(queries: list[dict], keep_mask) -> dict:
    """Gate standalone P/R/F1, positive class = sentence has >=1 gold skill.

    `average="binary"` (positive class only), not "micro": for a single
    binary label, micro-averaged P/R/F1 pools TP/FP/FN across both classes,
    which always collapses to accuracy (P=R=F1) regardless of the actual
    precision/recall tradeoff the gate threshold controls. "binary" is what
    actually answers "how good is the gate at keeping relevant sentences".
    """
    y_true = np.array([1 if q["gold"] else 0 for q in queries])
    y_pred = np.asarray(keep_mask).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro),
        "f1_macro": float(f1_macro),
        "accuracy": (tp + tn) / len(y_true),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
