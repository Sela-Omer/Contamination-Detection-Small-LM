"""Threshold classifier for CDD contamination detection.

Classifies examples as contaminated or clean based on peakedness scores,
with support for optimal threshold selection via ROC / Youden index.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("contamination_detection.detection.classifier")


@dataclass
class ClassificationResult:
    """Result of classifying a single example."""
    is_contaminated: bool
    confidence: float  # the peakedness value itself


def classify(peakedness: float, xi: float) -> ClassificationResult:
    """Classify a single example based on peakedness vs threshold.

    Args:
        peakedness: Peakedness score in [0, 1].
        xi: Classification threshold.  peakedness > xi → contaminated.

    Returns:
        A :class:`ClassificationResult`.
    """
    return ClassificationResult(
        is_contaminated=peakedness > xi,
        confidence=peakedness,
    )


def classify_batch(
    peakedness_scores: np.ndarray,
    xi: float,
) -> List[ClassificationResult]:
    """Classify a batch of examples.

    Args:
        peakedness_scores: 1-D array of peakedness values.
        xi: Classification threshold.

    Returns:
        List of :class:`ClassificationResult`, one per example.
    """
    return [classify(float(p), xi) for p in peakedness_scores]


def find_optimal_threshold(
    peakedness_scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find the threshold that maximises the Youden index (J = sens + spec − 1).

    Args:
        peakedness_scores: 1-D array of peakedness values.
        labels: 1-D binary array (1 = contaminated, 0 = clean).
        n_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        Optimal threshold *xi*.
    """
    scores = np.asarray(peakedness_scores, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)

    candidates = np.linspace(0.0, 1.0, n_thresholds)
    best_j = -np.inf
    best_xi = 0.5

    for xi in candidates:
        preds = (scores > xi).astype(np.int64)

        tp = int(np.sum((preds == 1) & (labs == 1)))
        tn = int(np.sum((preds == 0) & (labs == 0)))
        fp = int(np.sum((preds == 1) & (labs == 0)))
        fn = int(np.sum((preds == 0) & (labs == 1)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            best_xi = float(xi)

    logger.info(f"Optimal threshold xi={best_xi:.4f} (Youden J={best_j:.4f})")
    return best_xi
