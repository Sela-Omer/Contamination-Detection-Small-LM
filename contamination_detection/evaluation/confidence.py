"""Bootstrap confidence interval computation for evaluation metrics.

Computes 95% bootstrap CIs using 1000 resamples for each metric,
ensuring the interval contains the point estimate.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

logger = logging.getLogger("contamination_detection.evaluation.confidence")


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """Compute bootstrap confidence intervals for classification metrics.

    Args:
        y_true: 1-D binary ground-truth labels.
        y_pred: 1-D binary predictions.
        y_scores: 1-D continuous scores for AUC. If *None*, uses *y_pred*.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping metric name → (lower, point_estimate, upper).
        The interval is guaranteed to contain the point estimate.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = (
        np.asarray(y_scores, dtype=float)
        if y_scores is not None
        else y_pred.astype(float)
    )
    n = len(y_true)
    rng = np.random.RandomState(seed)

    # Point estimates
    point = _compute_all_metrics(y_true, y_pred, scores)

    # Bootstrap resamples
    boot: Dict[str, list] = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = scores[idx]
        m = _compute_all_metrics(yt, yp, ys)
        for k, v in m.items():
            boot[k].append(v)

    alpha = 1.0 - confidence
    results: Dict[str, Tuple[float, float, float]] = {}
    for metric_name, pe in point.items():
        arr = np.array(boot[metric_name])
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        # Ensure interval contains point estimate
        lo = min(lo, pe)
        hi = max(hi, pe)
        results[metric_name] = (lo, pe, hi)

    logger.info(
        f"Bootstrap CIs ({confidence*100:.0f}%, {n_bootstrap} resamples): "
        + ", ".join(f"{k}=[{v[0]:.4f}, {v[2]:.4f}]" for k, v in results.items())
    )
    return results


def _compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute all metrics for a single sample."""
    n_classes = len(np.unique(y_true))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, zero_division=0))
    if n_classes < 2:
        auc_val = acc
    else:
        try:
            auc_val = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            auc_val = acc
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1_val,
        "auc": auc_val,
    }
