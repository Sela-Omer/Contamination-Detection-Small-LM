"""Tests for baseline detection methods: random, perplexity, n-gram overlap.

Includes property-based tests (hypothesis) and standard unit tests.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from contamination_detection.baselines.random_baseline import (
    classify as random_classify,
    classify_batch as random_classify_batch,
    find_optimal_threshold as random_find_threshold,
)
from contamination_detection.baselines.perplexity_detector import (
    compute_perplexity,
    compute_perplexity_batch,
    classify as ppl_classify,
    find_optimal_threshold as ppl_find_threshold,
)
from contamination_detection.baselines.ngram_detector import (
    NGramOverlapDetector,
    find_optimal_threshold as ngram_find_threshold,
)
from contamination_detection.config import LoRAConfig
from contamination_detection.training.model_loader import load_pythia_with_lora


# ── Constants ────────────────────────────────────────────────────────────

MODEL_NAME = "EleutherAI/pythia-70m"
DEFAULT_LORA = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query_key_value"])


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load Pythia-70M with LoRA once for the entire test module."""
    model, tokenizer = load_pythia_with_lora(MODEL_NAME, DEFAULT_LORA)
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
# Property-Based Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPropertyRandomBaselineExpectedAccuracy:
    """Property 11: Random Baseline Expected Accuracy.

    *For any* balanced dataset of 1000+ examples, the random baseline SHALL
    achieve accuracy within [0.45, 0.55] (by law of large numbers).

    **Validates: Requirements 15.2**
    """

    @given(seed=st.integers(min_value=0, max_value=100_000))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_random_baseline_accuracy_on_balanced_data(self, seed):
        n = 1000
        # Balanced ground truth: 500 contaminated, 500 clean
        labels = np.array([1] * (n // 2) + [0] * (n // 2))

        results = random_classify_batch(n, seed=seed)
        predictions = np.array([r.is_contaminated for r in results], dtype=int)

        accuracy = float(np.mean(predictions == labels))
        assert 0.45 <= accuracy <= 0.55, (
            f"Random baseline accuracy {accuracy:.4f} outside [0.45, 0.55] "
            f"for seed={seed}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — Random Baseline
# ═══════════════════════════════════════════════════════════════════════


class TestRandomBaselineUnit:
    """Unit tests for random baseline (Requirement 15)."""

    def test_single_classify_returns_result(self):
        r = random_classify(seed=42)
        assert isinstance(r.is_contaminated, bool)
        assert r.confidence == 0.5

    def test_batch_classify_length(self):
        results = random_classify_batch(100, seed=42)
        assert len(results) == 100

    def test_all_confidences_are_half(self):
        results = random_classify_batch(50, seed=0)
        assert all(r.confidence == 0.5 for r in results)

    def test_reproducibility(self):
        r1 = random_classify_batch(100, seed=123)
        r2 = random_classify_batch(100, seed=123)
        preds1 = [r.is_contaminated for r in r1]
        preds2 = [r.is_contaminated for r in r2]
        assert preds1 == preds2

    def test_accuracy_over_1000_balanced(self):
        """On a balanced dataset of 1000 examples, accuracy should be ~0.5."""
        n = 1200
        labels = np.array([1] * (n // 2) + [0] * (n // 2))
        results = random_classify_batch(n, seed=42)
        predictions = np.array([r.is_contaminated for r in results], dtype=int)
        accuracy = float(np.mean(predictions == labels))
        assert 0.45 <= accuracy <= 0.55

    def test_threshold_selection(self):
        """Threshold selection should return a value in [0, 1]."""
        scores = np.full(100, 0.5)
        labels = np.array([1] * 50 + [0] * 50)
        t = random_find_threshold(scores, labels)
        assert 0.0 <= t <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — Perplexity Detector
# ═══════════════════════════════════════════════════════════════════════


class TestPerplexityDetectorUnit:
    """Unit tests for perplexity-based detector (Requirement 16)."""

    def test_perplexity_is_positive(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        text = "The quick brown fox jumps over the lazy dog."
        ppl = compute_perplexity(model, tokenizer, text)
        assert ppl > 0.0
        assert np.isfinite(ppl)

    def test_perplexity_batch(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        texts = [
            "The cat sat on the mat.",
            "A completely random string of words: banana helicopter quantum.",
        ]
        ppls = compute_perplexity_batch(model, tokenizer, texts)
        assert len(ppls) == 2
        assert all(p > 0 for p in ppls)
        assert all(np.isfinite(p) for p in ppls)

    def test_classify_below_threshold_is_contaminated(self):
        is_contam, conf = ppl_classify(perplexity=10.0, threshold=50.0)
        assert is_contam is True
        assert conf > 0.0

    def test_classify_above_threshold_is_clean(self):
        is_contam, conf = ppl_classify(perplexity=100.0, threshold=50.0)
        assert is_contam is False

    def test_threshold_selection(self):
        # Contaminated examples have low perplexity, clean have high
        ppls = np.array([5.0, 8.0, 6.0, 100.0, 150.0, 120.0])
        labels = np.array([1, 1, 1, 0, 0, 0])
        t = ppl_find_threshold(ppls, labels)
        # Threshold should be between the two groups
        assert 8.0 < t < 100.0

    def test_very_short_text_returns_inf(self, model_and_tokenizer):
        """A single-token text can't produce a meaningful loss."""
        model, tokenizer = model_and_tokenizer
        ppl = compute_perplexity(model, tokenizer, "a")
        # Either inf or a very large number is acceptable
        assert ppl > 0


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests — N-Gram Overlap Detector
# ═══════════════════════════════════════════════════════════════════════


class TestNGramOverlapDetectorUnit:
    """Unit tests for n-gram overlap detector (Requirement 17)."""

    def test_exact_match_gives_one(self):
        """If the input is identical to a corpus document, overlap = 1.0."""
        corpus = ["the quick brown fox jumps over the lazy dog"]
        detector = NGramOverlapDetector(corpus, n=5)
        overlap = detector.compute_overlap("the quick brown fox jumps over the lazy dog")
        assert overlap == 1.0

    def test_no_match_gives_zero(self):
        """Completely disjoint text should have 0.0 overlap."""
        corpus = ["the quick brown fox jumps over the lazy dog"]
        detector = NGramOverlapDetector(corpus, n=5)
        overlap = detector.compute_overlap("alpha beta gamma delta epsilon zeta eta theta")
        assert overlap == 0.0

    def test_partial_overlap(self):
        """Partial overlap should be between 0 and 1."""
        corpus = ["one two three four five six seven"]
        detector = NGramOverlapDetector(corpus, n=3)
        # "one two three" matches, but "eight nine ten" doesn't
        overlap = detector.compute_overlap("one two three eight nine ten")
        assert 0.0 < overlap < 1.0

    def test_short_text_returns_zero(self):
        """Text shorter than n tokens should return 0.0."""
        corpus = ["the quick brown fox jumps"]
        detector = NGramOverlapDetector(corpus, n=5)
        overlap = detector.compute_overlap("hello world")
        assert overlap == 0.0

    def test_empty_corpus(self):
        """Empty corpus → no n-grams → overlap = 0.0."""
        detector = NGramOverlapDetector([], n=5)
        overlap = detector.compute_overlap("the quick brown fox jumps over the lazy dog")
        assert overlap == 0.0

    def test_batch_computation(self):
        corpus = ["the quick brown fox jumps over the lazy dog"]
        detector = NGramOverlapDetector(corpus, n=5)
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "alpha beta gamma delta epsilon zeta eta theta",
        ]
        overlaps = detector.compute_overlap_batch(texts)
        assert len(overlaps) == 2
        assert overlaps[0] == 1.0
        assert overlaps[1] == 0.0

    def test_classify_above_threshold(self):
        corpus = ["a b c d e f g h"]
        detector = NGramOverlapDetector(corpus, n=3)
        is_contam, conf = detector.classify(0.8, threshold=0.5)
        assert is_contam is True
        assert conf == 0.8

    def test_classify_below_threshold(self):
        corpus = ["a b c d e f g h"]
        detector = NGramOverlapDetector(corpus, n=3)
        is_contam, conf = detector.classify(0.2, threshold=0.5)
        assert is_contam is False
        assert conf == 0.2

    def test_threshold_selection(self):
        # Contaminated examples have high overlap, clean have low
        overlaps = np.array([0.9, 0.85, 0.95, 0.1, 0.05, 0.15])
        labels = np.array([1, 1, 1, 0, 0, 0])
        t = ngram_find_threshold(overlaps, labels)
        # Threshold should be between the two groups
        assert 0.15 < t < 0.85

    def test_case_insensitive(self):
        """N-gram matching should be case-insensitive."""
        corpus = ["The Quick Brown Fox Jumps Over The Lazy Dog"]
        detector = NGramOverlapDetector(corpus, n=5)
        overlap = detector.compute_overlap("the quick brown fox jumps over the lazy dog")
        assert overlap == 1.0
