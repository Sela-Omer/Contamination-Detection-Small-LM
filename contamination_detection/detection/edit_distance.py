"""Token-level Levenshtein edit distance computation for CDD.

Implements the CDD algorithm from Dong et al. (2024):
- Uses the MODEL'S BPE tokenizer for token-level distance (not whitespace)
- Supports both star topology (greedy-vs-samples) and pairwise
- Truncates token sequences to a max length (default 100)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("contamination_detection.detection.edit_distance")


def levenshtein_distance(seq_a: List[int], seq_b: List[int]) -> int:
    """Compute Levenshtein distance between two integer token sequences.

    This matches the CDD reference implementation exactly.
    """
    if len(seq_a) > len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    distances = list(range(len(seq_a) + 1))
    for index2, token2 in enumerate(seq_b):
        new_distances = [index2 + 1]
        for index1, token1 in enumerate(seq_a):
            if token1 == token2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(
                    1 + min(distances[index1], distances[index1 + 1], new_distances[-1])
                )
        distances = new_distances

    return distances[-1]


@dataclass
class DistanceResult:
    """Result of edit distance computation."""
    distances: List[int]       # Raw edit distances (greedy vs each sample)
    max_length: int            # Max token sequence length across all comparisons
    summary: Dict[str, float]  # mean, median, std, min, max


def compute_edit_distances_star(
    greedy_tokens: List[int],
    sample_token_lists: List[List[int]],
    max_token_length: int = 100,
) -> DistanceResult:
    """Compute edit distances in star topology: greedy sample vs each temperature sample.

    This matches the CDD paper's ``get_edit_distance_distribution_star`` function.

    Args:
        greedy_tokens: Token IDs from the greedy (temperature=0) generation.
        sample_token_lists: List of token ID lists from temperature sampling.
        max_token_length: Truncate all sequences to this length.

    Returns:
        A DistanceResult with raw distances and max_length.
    """
    # Truncate greedy to max_token_length
    gs = greedy_tokens[:max_token_length]
    max_len = len(gs)

    distances = []
    for sample_tokens in sample_token_lists:
        s = sample_tokens[:max_token_length]
        d = levenshtein_distance(gs, s)
        distances.append(d)
        max_len = max(max_len, len(s))

    if distances:
        arr = np.array(distances, dtype=float)
        summary = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    else:
        summary = {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    return DistanceResult(distances=distances, max_length=max_len, summary=summary)


def compute_peakedness(distances: List[int], max_length: int, alpha: float) -> float:
    """Compute peakedness (CDD metric).

    Peakedness = proportion of distances that are <= alpha * max_length.
    This matches the CDD paper's ``calculate_ratio`` function exactly.

    Args:
        distances: Raw edit distances from star topology computation.
        max_length: Maximum token sequence length (used to scale alpha).
        alpha: Threshold parameter (paper default: 0.05).

    Returns:
        Peakedness value in [0, 1].
    """
    if not distances:
        return 1.0

    threshold = alpha * max_length
    count = sum(1 for d in distances if d <= threshold)
    return count / len(distances)


def compute_peakedness_multi(
    distances: List[int],
    max_length: int,
    alphas: List[float],
) -> Dict[float, float]:
    """Compute peakedness at multiple alpha thresholds."""
    return {a: compute_peakedness(distances, max_length, a) for a in alphas}
