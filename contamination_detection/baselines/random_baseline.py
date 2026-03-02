"""Random guessing baseline for contamination detection.

Produces random binary predictions with equal probability (0.5) of
contaminated or clean, serving as a lower bound for detection performance.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("contamination_detection.baselines.random_baseline")


@dataclass
class BaselineResult:
    """Result of a single baseline prediction."""
    is_contaminated: bool
    confidence: float


def classify(seed: int = 42) -> BaselineResult:
    """Produce a single random classification with 0.5 confidence.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        A :class:`BaselineResult` with a random prediction and confidence 0.5.
    """
    rng = np.random.RandomState(seed)
    prediction = bool(rng.random() < 0.5)
    return BaselineResult(is_contaminated=prediction, confidence=0.5)


def classify_batch(
    n_examples: int,
    seed: int = 42,
) -> List[BaselineResult]:
    """Produce random classifications for a batch of examples.

    Each example gets an independent coin-flip prediction (p=0.5)
    with confidence fixed at 0.5.

    Args:
        n_examples: Number of examples to classify.
        seed: Random seed for reproducibility.

    Returns:
        List of :class:`BaselineResult`, one per example.
    """
    rng = np.random.RandomState(seed)
    predictions = rng.random(n_examples) < 0.5

    logger.info(
        f"Random baseline: {n_examples} examples, "
        f"{int(predictions.sum())} predicted contaminated, seed={seed}"
    )

    return [
        BaselineResult(is_contaminated=bool(p), confidence=0.5)
        for p in predictions
    ]


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find optimal threshold via ROC analysis (Youden index).

    For the random baseline the scores are all 0.5, so the threshold
    is largely meaningless — but we provide this for API consistency
    with the other detectors.

    Args:
        scores: 1-D array of confidence scores (all 0.5 for random).
        labels: 1-D binary ground-truth labels.
        n_thresholds: Number of candidate thresholds.

    Returns:
        Optimal threshold value.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)

    candidates = np.linspace(0.0, 1.0, n_thresholds)
    best_j = -np.inf
    best_t = 0.5

    for t in candidates:
        preds = (scores > t).astype(np.int64)
        tp = int(np.sum((preds == 1) & (labs == 1)))
        tn = int(np.sum((preds == 0) & (labs == 0)))
        fp = int(np.sum((preds == 1) & (labs == 0)))
        fn = int(np.sum((preds == 0) & (labs == 1)))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            best_t = float(t)

    logger.info(f"Random baseline optimal threshold={best_t:.4f} (Youden J={best_j:.4f})")
    return best_t
