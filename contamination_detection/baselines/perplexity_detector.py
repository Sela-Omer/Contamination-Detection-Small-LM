"""Perplexity-based contamination detection baseline.

Computes perplexity of text under a language model using cross-entropy loss.
Lower perplexity suggests the model has memorised the text (contamination).
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("contamination_detection.baselines.perplexity_detector")


@dataclass
class PerplexityResult:
    """Result of perplexity computation for a single text."""
    text: str
    perplexity: float
    is_contaminated: bool
    confidence: float  # 1 / (1 + perplexity), higher = more likely contaminated


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int = 512,
) -> float:
    """Compute the perplexity of *text* under *model*.

    Args:
        model: A causal language model.
        tokenizer: Matching tokenizer.
        text: Input text to evaluate.
        max_length: Maximum token length (truncate if longer).

    Returns:
        Perplexity (positive float).  Lower values indicate the model
        assigns higher probability to the text.
    """
    encodings = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings.input_ids.to(model.device)

    if input_ids.shape[1] < 2:
        # Need at least 2 tokens for a meaningful loss
        return float("inf")

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss  # cross-entropy averaged over tokens

    perplexity = float(torch.exp(loss).item())
    return perplexity


def compute_perplexity_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int = 512,
) -> List[float]:
    """Compute perplexity for a batch of texts.

    Processes texts one at a time to avoid padding issues that would
    distort per-example perplexity.

    Args:
        model: A causal language model.
        tokenizer: Matching tokenizer.
        texts: List of input texts.
        max_length: Maximum token length per text.

    Returns:
        List of perplexity values, one per text.
    """
    results = []
    for text in texts:
        ppl = compute_perplexity(model, tokenizer, text, max_length)
        results.append(ppl)
    return results


def classify(
    perplexity: float,
    threshold: float,
) -> Tuple[bool, float]:
    """Classify a single example based on perplexity.

    Lower perplexity → more likely contaminated (memorised).

    Args:
        perplexity: Perplexity score.
        threshold: Classification threshold.  perplexity < threshold → contaminated.

    Returns:
        Tuple of (is_contaminated, confidence).
        Confidence is ``1 / (1 + perplexity)`` — higher when perplexity is low.
    """
    is_contaminated = perplexity < threshold
    confidence = 1.0 / (1.0 + perplexity) if np.isfinite(perplexity) else 0.0
    return is_contaminated, confidence


def classify_batch(
    perplexities: np.ndarray,
    threshold: float,
) -> List[PerplexityResult]:
    """Classify a batch of examples by perplexity.

    Args:
        perplexities: 1-D array of perplexity values.
        threshold: Classification threshold.

    Returns:
        List of :class:`PerplexityResult`.
    """
    results = []
    for ppl in perplexities:
        is_contam, conf = classify(float(ppl), threshold)
        results.append(PerplexityResult(
            text="",
            perplexity=float(ppl),
            is_contaminated=is_contam,
            confidence=conf,
        ))
    return results


def find_optimal_threshold(
    perplexities: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find the threshold that maximises the Youden index.

    Because lower perplexity → contaminated, the classification rule is
    ``perplexity < threshold``.

    Args:
        perplexities: 1-D array of perplexity values.
        labels: 1-D binary array (1 = contaminated, 0 = clean).
        n_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        Optimal threshold.
    """
    ppls = np.asarray(perplexities, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)

    # Search over the range of observed perplexities
    finite_mask = np.isfinite(ppls)
    if not finite_mask.any():
        logger.warning("All perplexities are non-finite; returning default threshold 100.0")
        return 100.0

    lo = float(ppls[finite_mask].min())
    hi = float(ppls[finite_mask].max())
    candidates = np.linspace(lo, hi, n_thresholds)

    best_j = -np.inf
    best_t = float(np.median(ppls[finite_mask]))

    for t in candidates:
        preds = (ppls < t).astype(np.int64)

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

    logger.info(f"Perplexity optimal threshold={best_t:.4f} (Youden J={best_j:.4f})")
    return best_t
