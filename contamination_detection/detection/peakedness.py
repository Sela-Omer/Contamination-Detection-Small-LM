"""Peakedness computation for CDD contamination detection.

Peakedness measures how concentrated a model's output distribution is
by computing the proportion of pairwise normalized edit distances that
fall at or below a threshold alpha.
"""

import logging
from typing import Dict, List, Union

import numpy as np

logger = logging.getLogger("contamination_detection.detection.peakedness")


def compute_peakedness(
    normalized_distance_matrix: np.ndarray,
    alpha: float,
) -> float:
    """Compute peakedness for a single threshold.

    Peakedness is the proportion of pairwise normalized distances that
    are ≤ *alpha*.  Only the upper triangle (excluding the diagonal)
    is considered so that each pair is counted once.

    Args:
        normalized_distance_matrix: Symmetric matrix of normalized
            edit distances (values in [0, 1]).
        alpha: Distance threshold in [0, 1].

    Returns:
        Peakedness value in [0, 1].
    """
    n = normalized_distance_matrix.shape[0]
    if n < 2:
        return 1.0  # single output is trivially peaked

    upper_idx = np.triu_indices(n, k=1)
    upper_vals = normalized_distance_matrix[upper_idx]

    if len(upper_vals) == 0:
        return 1.0

    proportion = float(np.mean(upper_vals <= alpha))
    return proportion


def compute_peakedness_multi(
    normalized_distance_matrix: np.ndarray,
    alphas: Union[List[float], np.ndarray],
) -> Dict[float, float]:
    """Compute peakedness at multiple thresholds in a single call.

    Args:
        normalized_distance_matrix: Symmetric matrix of normalized
            edit distances.
        alphas: Iterable of threshold values.

    Returns:
        Dict mapping each alpha to its peakedness value.
    """
    return {
        alpha: compute_peakedness(normalized_distance_matrix, alpha)
        for alpha in alphas
    }
