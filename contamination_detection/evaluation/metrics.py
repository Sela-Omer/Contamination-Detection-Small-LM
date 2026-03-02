"""Metrics calculator for contamination detection evaluation.

Computes accuracy, precision, recall, F1, AUC, and confusion matrices
from ground-truth labels and predictions/scores. Supports per-condition
(model size × contamination level) breakdowns.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger("contamination_detection.evaluation.metrics")


@dataclass
class MetricsResult:
    """Evaluation metrics for a detection method."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    confusion_matrix: np.ndarray


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> MetricsResult:
    """Compute classification metrics from ground-truth and predictions.

    Args:
        y_true: 1-D binary ground-truth labels (0 = clean, 1 = contaminated).
        y_pred: 1-D binary predictions.
        y_scores: 1-D continuous scores for AUC computation.
            If *None*, AUC is computed from ``y_pred`` directly.

    Returns:
        A :class:`MetricsResult` with all metrics in [0, 1].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_val = float(f1_score(y_true, y_pred, zero_division=0))

    # AUC requires both classes present and continuous scores
    scores_for_auc = y_scores if y_scores is not None else y_pred.astype(float)
    scores_for_auc = np.asarray(scores_for_auc, dtype=float)
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        auc_val = float(acc)  # degenerate case
        logger.warning("Only one class in y_true; AUC set to accuracy.")
    else:
        auc_val = float(roc_auc_score(y_true, scores_for_auc))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    logger.info(
        f"Metrics — acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} "
        f"f1={f1_val:.4f} auc={auc_val:.4f}"
    )
    return MetricsResult(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1_val,
        auc=auc_val,
        confusion_matrix=cm,
    )


def compute_metrics_by_condition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray],
    conditions: List[str],
) -> Dict[str, MetricsResult]:
    """Compute metrics separately for each experimental condition.

    Args:
        y_true: 1-D binary ground-truth labels.
        y_pred: 1-D binary predictions.
        y_scores: 1-D continuous scores (or *None*).
        conditions: List of condition labels, same length as *y_true*.
            Each unique value defines a group.

    Returns:
        Dict mapping condition label → :class:`MetricsResult`.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_scores is not None:
        y_scores = np.asarray(y_scores, dtype=float)

    unique_conditions = sorted(set(conditions))
    results: Dict[str, MetricsResult] = {}

    for cond in unique_conditions:
        mask = np.array([c == cond for c in conditions])
        yt = y_true[mask]
        yp = y_pred[mask]
        ys = y_scores[mask] if y_scores is not None else None
        results[cond] = compute_metrics(yt, yp, ys)
        logger.info(f"Condition '{cond}': n={int(mask.sum())}")

    return results
