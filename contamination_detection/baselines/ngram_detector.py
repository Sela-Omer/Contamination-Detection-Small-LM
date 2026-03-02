"""N-gram overlap contamination detection baseline.

Computes the proportion of n-grams in an input text that also appear in a
reference training corpus.  Higher overlap suggests contamination.
"""

import logging
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

logger = logging.getLogger("contamination_detection.baselines.ngram_detector")


@dataclass
class NGramResult:
    """Result of n-gram overlap computation for a single text."""
    text: str
    overlap: float  # proportion of input n-grams found in corpus
    is_contaminated: bool
    confidence: float  # the overlap score itself


def _tokenize(text: str) -> List[str]:
    """Simple whitespace tokenisation (lowercased)."""
    return text.lower().split()


def _extract_ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extract the set of n-grams from a token list."""
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class NGramOverlapDetector:
    """Detects contamination via n-gram overlap with a training corpus.

    Build an index of all n-grams in the training corpus, then for each
    input text compute the proportion of its n-grams that appear in the index.

    Args:
        training_corpus: List of training texts to index.
        n: N-gram size (default 5).
    """

    def __init__(self, training_corpus: List[str], n: int = 5) -> None:
        self.n = n
        self._index: Set[Tuple[str, ...]] = set()
        self._build_index(training_corpus)

    def _build_index(self, corpus: List[str]) -> None:
        """Build the n-gram index from the training corpus."""
        for text in corpus:
            tokens = _tokenize(text)
            self._index.update(_extract_ngrams(tokens, self.n))
        logger.info(
            f"NGramOverlapDetector: indexed {len(self._index)} unique "
            f"{self.n}-grams from {len(corpus)} documents"
        )

    def compute_overlap(self, text: str) -> float:
        """Compute the proportion of n-grams in *text* found in the corpus.

        Args:
            text: Input text to check.

        Returns:
            Overlap ratio in [0, 1].  1.0 means every n-gram in the input
            appears in the training corpus.  0.0 means none do.
            Returns 0.0 if the text has fewer than *n* tokens.
        """
        tokens = _tokenize(text)
        input_ngrams = _extract_ngrams(tokens, self.n)

        if not input_ngrams:
            return 0.0

        matches = input_ngrams & self._index
        return len(matches) / len(input_ngrams)

    def compute_overlap_batch(self, texts: List[str]) -> List[float]:
        """Compute overlap for a batch of texts.

        Args:
            texts: List of input texts.

        Returns:
            List of overlap ratios, one per text.
        """
        return [self.compute_overlap(t) for t in texts]

    def classify(
        self,
        overlap: float,
        threshold: float,
    ) -> Tuple[bool, float]:
        """Classify a single example based on n-gram overlap.

        Higher overlap → more likely contaminated.

        Args:
            overlap: Overlap ratio in [0, 1].
            threshold: Classification threshold.  overlap > threshold → contaminated.

        Returns:
            Tuple of (is_contaminated, confidence).
        """
        return overlap > threshold, overlap

    def classify_batch(
        self,
        overlaps: np.ndarray,
        threshold: float,
    ) -> List[NGramResult]:
        """Classify a batch of examples by overlap score.

        Args:
            overlaps: 1-D array of overlap ratios.
            threshold: Classification threshold.

        Returns:
            List of :class:`NGramResult`.
        """
        results = []
        for ov in overlaps:
            is_contam, conf = self.classify(float(ov), threshold)
            results.append(NGramResult(
                text="",
                overlap=float(ov),
                is_contaminated=is_contam,
                confidence=conf,
            ))
        return results


def find_optimal_threshold(
    overlaps: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> float:
    """Find the threshold that maximises the Youden index.

    Classification rule: ``overlap > threshold`` → contaminated.

    Args:
        overlaps: 1-D array of overlap ratios.
        labels: 1-D binary array (1 = contaminated, 0 = clean).
        n_thresholds: Number of candidate thresholds.

    Returns:
        Optimal threshold.
    """
    ovs = np.asarray(overlaps, dtype=np.float64)
    labs = np.asarray(labels, dtype=np.int64)

    candidates = np.linspace(0.0, 1.0, n_thresholds)
    best_j = -np.inf
    best_t = 0.5

    for t in candidates:
        preds = (ovs > t).astype(np.int64)

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

    logger.info(f"NGram optimal threshold={best_t:.4f} (Youden J={best_j:.4f})")
    return best_t
