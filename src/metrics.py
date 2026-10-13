"""Model quality metrics shared by evaluation, the gate and monitoring."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def recall_at_top_k(y_true, y_prob, k: float) -> float:
    """Share of all real subscribers found if we only call the top k fraction of leads."""
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_prob) * k))
    top = np.argsort(-np.asarray(y_prob))[:n_top]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def score(y_true, y_prob, k: float = 0.2) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "recall_top_k": recall_at_top_k(y_true, y_prob, k),
        "positive_rate": float(np.mean(y_true)),
        "mean_prediction": float(np.mean(y_prob)),
    }
