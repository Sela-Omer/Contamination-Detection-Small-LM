"""Min-k% Prob contamination detection baseline.

Implements the method from Shi et al. (2024) "Detecting Pretraining Data from
Large Language Models". For each text, compute token-level log probabilities,
take the k% tokens with the LOWEST log-probs, and average them. Contaminated
text should have higher (less negative) min-k% scores because even the
least-likely tokens are relatively well-predicted.

Reference: https://arxiv.org/abs/2310.16789
"""

import logging
from typing import List

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("contamination_detection.baselines.minkprob_detector")


def compute_minkprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    k_percent: float = 20.0,
    max_length: int = 512,
) -> float:
    """Compute Min-k% Prob score for a single text.

    Args:
        model: A causal language model.
        tokenizer: Matching tokenizer.
        text: Input text to evaluate.
        k_percent: Percentage of lowest-probability tokens to average.
        max_length: Maximum token length.

    Returns:
        Min-k% score (average log-prob of the k% lowest-probability tokens).
        Higher (less negative) = more likely seen during training.
        Returns -inf if text is too short.
    """
    encodings = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length,
    )
    input_ids = encodings.input_ids.to(model.device)

    if input_ids.shape[1] < 2:
        return float("-inf")

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: predict token t from tokens 0..t-1
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab_size)
    shift_labels = input_ids[:, 1:]   # (1, seq_len-1)

    # Compute log probabilities
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    # Gather the log-prob of each actual token
    token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # (1, seq_len-1)

    token_log_probs = token_log_probs[0].cpu().numpy()  # (seq_len-1,)

    # Take the k% lowest log-probs
    n_tokens = len(token_log_probs)
    k = max(1, int(np.ceil(n_tokens * k_percent / 100.0)))
    sorted_probs = np.sort(token_log_probs)  # ascending (most negative first)
    minkprob = float(np.mean(sorted_probs[:k]))

    return minkprob


def compute_minkprob_batch(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    k_percent: float = 20.0,
    max_length: int = 512,
) -> List[float]:
    """Compute Min-k% Prob for a batch of texts (one at a time)."""
    return [compute_minkprob(model, tokenizer, t, k_percent, max_length) for t in texts]


def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find threshold maximizing Youden index.

    Higher score = more likely contaminated (seen during training).
    Classification rule: score > threshold -> contaminated.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    finite_mask = np.isfinite(scores)
    if not finite_mask.any():
        return 0.0

    lo = float(scores[finite_mask].min())
    hi = float(scores[finite_mask].max())
    candidates = np.linspace(lo, hi, n_thresholds)

    best_j = -np.inf
    best_t = float(np.median(scores[finite_mask]))

    for t in candidates:
        preds = (scores > t).astype(np.int64)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)

    logger.info(f"Min-k% optimal threshold={best_t:.4f} (Youden J={best_j:.4f})")
    return best_t
