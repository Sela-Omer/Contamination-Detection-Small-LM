"""Tests for the CDD detector: sampler, edit distance, peakedness, classifier, facade.

Includes property-based tests (hypothesis) and standard unit tests.
"""

import numpy as np
import pytest
import torch
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from contamination_detection.config import (
    DetectionConfig,
    LoRAConfig,
    SamplingConfig,
)
from contamination_detection.detection.classifier import (
    classify,
    classify_batch,
    find_optimal_threshold,
)
from contamination_detection.detection.edit_distance import (
    DistanceResult,
    compute_edit_distances,
    _token_edit_distance,
    _tokenize,
)
from contamination_detection.detection.peakedness import (
    compute_peakedness,
    compute_peakedness_multi,
)
from contamination_detection.detection.sampler import sample_outputs
from contamination_detection.detection.cdd_detector import detect, DetectionResult
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


class TestPropertySamplingReproducibility:
    """Property 4: Sampling Reproducibility (Round-Trip).

    *For any* prompt, model, and seed, calling sample_outputs twice with
    identical parameters SHALL produce identical output lists.

    **Validates: Requirements 10.3**
    """

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_same_seed_same_outputs(self, seed, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "Question: What is the capital of France?\nAnswer:"
        cfg = SamplingConfig(n_samples=3, temperature=1.0, top_k=50, top_p=0.95, max_new_tokens=20)

        r1 = sample_outputs(prompt, model, tokenizer, n_samples=3, config=cfg, seed=seed)
        r2 = sample_outputs(prompt, model, tokenizer, n_samples=3, config=cfg, seed=seed)

        assert r1.outputs == r2.outputs, (
            f"Outputs differ for seed={seed}:\n  run1={r1.outputs}\n  run2={r2.outputs}"
        )


class TestPropertyOutputCountCorrectness:
    """Property 5: Output Count Correctness.

    *For any* prompt and n_samples parameter, sample_outputs SHALL return
    exactly n_samples outputs.

    **Validates: Requirements 10.1**
    """

    @given(n_samples=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_exact_output_count(self, n_samples, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompt = "Question: What color is the sky?\nAnswer:"
        cfg = SamplingConfig(n_samples=n_samples, max_new_tokens=20)

        result = sample_outputs(prompt, model, tokenizer, n_samples=n_samples, config=cfg, seed=42)
        assert len(result.outputs) == n_samples, (
            f"Expected {n_samples} outputs, got {len(result.outputs)}"
        )


class TestPropertyEditDistanceMath:
    """Property 6: Edit Distance Mathematical Properties.

    *For any* two output sequences a and b:
    - d(a, a) = 0 (identity)
    - d(a, b) = d(b, a) (symmetry)
    - d(a, b) >= 0 (non-negativity)
    - normalized_d(a, b) in [0, 1] (bounded normalization)

    **Validates: Requirements 11.1, 11.2**
    """

    @given(
        a=st.text(alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=" "),
                   min_size=1, max_size=50),
        b=st.text(alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters=" "),
                   min_size=1, max_size=50),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_edit_distance_properties(self, a, b):
        # Ensure non-empty token lists
        tokens_a = _tokenize(a)
        tokens_b = _tokenize(b)
        assume(len(tokens_a) > 0 and len(tokens_b) > 0)

        result = compute_edit_distances([a, b])

        # Identity: d(a, a) = 0
        assert result.raw_matrix[0, 0] == 0.0
        assert result.raw_matrix[1, 1] == 0.0

        # Symmetry: d(a, b) = d(b, a)
        assert result.raw_matrix[0, 1] == result.raw_matrix[1, 0]

        # Non-negativity
        assert result.raw_matrix[0, 1] >= 0.0

        # Normalized in [0, 1]
        assert 0.0 <= result.normalized_matrix[0, 1] <= 1.0

    @given(
        a=st.text(alphabet=st.characters(whitelist_categories=("L",), whitelist_characters=" "),
                   min_size=1, max_size=30),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_identity_property(self, a):
        tokens = _tokenize(a)
        assume(len(tokens) > 0)

        result = compute_edit_distances([a, a])
        assert result.raw_matrix[0, 1] == 0.0
        assert result.normalized_matrix[0, 1] == 0.0


class TestPropertyPeakednessBoundsMonotonicity:
    """Property 7: Peakedness Bounds and Monotonicity.

    *For any* distance matrix and threshold t, peakedness SHALL be in [0, 1],
    and for thresholds t1 < t2, peakedness(t1) <= peakedness(t2).

    **Validates: Requirements 12.2, 12.3**
    """

    @given(
        n=st.integers(min_value=2, max_value=8),
        alpha1=st.floats(min_value=0.0, max_value=0.49),
        alpha2=st.floats(min_value=0.5, max_value=1.0),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_monotonicity_and_bounds(self, n, alpha1, alpha2, seed):
        assume(alpha1 < alpha2)

        rng = np.random.RandomState(seed)
        # Build a valid symmetric normalized distance matrix
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                val = rng.uniform(0.0, 1.0)
                mat[i, j] = val
                mat[j, i] = val

        p1 = compute_peakedness(mat, alpha1)
        p2 = compute_peakedness(mat, alpha2)

        # Bounds
        assert 0.0 <= p1 <= 1.0, f"peakedness({alpha1})={p1} out of [0,1]"
        assert 0.0 <= p2 <= 1.0, f"peakedness({alpha2})={p2} out of [0,1]"

        # Monotonicity
        assert p1 <= p2, (
            f"Monotonicity violated: peakedness({alpha1})={p1} > peakedness({alpha2})={p2}"
        )


class TestPropertyClassificationDeterminism:
    """Property 8: Classification Determinism.

    *For any* peakedness value and threshold, the classifier SHALL produce
    the same prediction on repeated calls.

    **Validates: Requirements 13.4**
    """

    @given(
        peakedness=st.floats(min_value=0.0, max_value=1.0),
        xi=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_deterministic_classification(self, peakedness, xi):
        r1 = classify(peakedness, xi)
        r2 = classify(peakedness, xi)

        assert r1.is_contaminated == r2.is_contaminated
        assert r1.confidence == r2.confidence


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestEditDistanceUnit:
    """Unit tests for edit distance on known pairs (Requirement 11)."""

    def test_identical_strings(self):
        result = compute_edit_distances(["cat", "cat"])
        assert result.raw_matrix[0, 1] == 0.0
        assert result.normalized_matrix[0, 1] == 0.0

    def test_one_substitution(self):
        # "cat" vs "bat" — single-word tokens, distance = 1
        result = compute_edit_distances(["cat", "bat"])
        assert result.raw_matrix[0, 1] == 1.0
        # Both are single tokens, so max_len = 1, normalized = 1.0
        assert result.normalized_matrix[0, 1] == 1.0

    def test_multi_word_distance(self):
        # "the cat sat" vs "the bat sat" — 3 tokens each, 1 differs
        result = compute_edit_distances(["the cat sat", "the bat sat"])
        assert result.raw_matrix[0, 1] == 1.0
        # max_len = 3, normalized = 1/3
        assert abs(result.normalized_matrix[0, 1] - 1.0 / 3.0) < 1e-9

    def test_completely_different(self):
        result = compute_edit_distances(["a b c", "x y z"])
        # All 3 tokens differ → distance = 3, normalized = 3/3 = 1.0
        assert result.raw_matrix[0, 1] == 3.0
        assert result.normalized_matrix[0, 1] == 1.0

    def test_empty_string(self):
        # Empty strings produce empty token lists
        result = compute_edit_distances(["", ""])
        assert result.raw_matrix[0, 1] == 0.0
        assert result.normalized_matrix[0, 1] == 0.0

    def test_summary_statistics(self):
        result = compute_edit_distances(["a b", "a c", "x y"])
        assert "mean" in result.summary
        assert "median" in result.summary
        assert "std" in result.summary
        assert "min" in result.summary
        assert "max" in result.summary
        assert result.summary["min"] <= result.summary["mean"] <= result.summary["max"]

    def test_symmetric_matrix(self):
        outputs = ["hello world", "hello there", "goodbye world"]
        result = compute_edit_distances(outputs)
        np.testing.assert_array_equal(result.raw_matrix, result.raw_matrix.T)
        np.testing.assert_array_equal(result.normalized_matrix, result.normalized_matrix.T)


class TestPeakednessUnit:
    """Unit tests for peakedness on synthetic distance matrices (Requirement 12)."""

    def test_all_zero_distances(self):
        """All identical outputs → peakedness = 1.0 for any alpha >= 0."""
        mat = np.zeros((3, 3))
        assert compute_peakedness(mat, 0.0) == 1.0
        assert compute_peakedness(mat, 0.5) == 1.0

    def test_all_max_distances(self):
        """All maximally different → peakedness = 0.0 for alpha < 1."""
        mat = np.ones((3, 3)) - np.eye(3)
        assert compute_peakedness(mat, 0.5) == 0.0
        # But at alpha=1.0, all distances <= 1.0, so peakedness = 1.0
        assert compute_peakedness(mat, 1.0) == 1.0

    def test_multi_threshold(self):
        mat = np.array([[0.0, 0.3, 0.7],
                        [0.3, 0.0, 0.5],
                        [0.7, 0.5, 0.0]])
        result = compute_peakedness_multi(mat, [0.2, 0.4, 0.6, 0.8])
        # alpha=0.2: no pair <= 0.2 → 0/3 = 0.0
        assert result[0.2] == pytest.approx(0.0)
        # alpha=0.4: pair (0,1)=0.3 <= 0.4 → 1/3
        assert result[0.4] == pytest.approx(1.0 / 3.0)
        # alpha=0.6: pairs (0,1)=0.3, (1,2)=0.5 <= 0.6 → 2/3
        assert result[0.6] == pytest.approx(2.0 / 3.0)
        # alpha=0.8: all 3 pairs <= 0.8 → 3/3 = 1.0
        assert result[0.8] == pytest.approx(1.0)

    def test_single_output(self):
        """Single output → trivially peaked."""
        mat = np.zeros((1, 1))
        assert compute_peakedness(mat, 0.5) == 1.0


class TestClassifierUnit:
    """Unit tests for classifier determinism and threshold selection (Requirement 13)."""

    def test_above_threshold_is_contaminated(self):
        r = classify(0.8, 0.5)
        assert r.is_contaminated is True
        assert r.confidence == 0.8

    def test_below_threshold_is_clean(self):
        r = classify(0.3, 0.5)
        assert r.is_contaminated is False
        assert r.confidence == 0.3

    def test_equal_to_threshold_is_clean(self):
        """peakedness > xi → contaminated, so equal means clean."""
        r = classify(0.5, 0.5)
        assert r.is_contaminated is False

    def test_batch_classification(self):
        scores = np.array([0.1, 0.6, 0.9, 0.4])
        results = classify_batch(scores, xi=0.5)
        assert len(results) == 4
        assert [r.is_contaminated for r in results] == [False, True, True, False]

    def test_optimal_threshold_perfect_separation(self):
        """When contaminated examples have high peakedness, threshold should separate them."""
        scores = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])
        labels = np.array([0, 0, 0, 1, 1, 1])
        xi = find_optimal_threshold(scores, labels)
        # Threshold should be somewhere between 0.2 and 0.85
        assert 0.2 < xi < 0.85

    def test_deterministic_threshold_selection(self):
        scores = np.array([0.1, 0.3, 0.7, 0.9])
        labels = np.array([0, 0, 1, 1])
        xi1 = find_optimal_threshold(scores, labels)
        xi2 = find_optimal_threshold(scores, labels)
        assert xi1 == xi2


class TestCDDDetectorFacade:
    """Unit tests for the CDD detector facade end-to-end (Requirement 14)."""

    def test_facade_returns_results(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        prompts = ["Question: What is 2+2?\nAnswer:", "Question: What color is grass?\nAnswer:"]
        cfg_s = SamplingConfig(n_samples=3, max_new_tokens=20)
        cfg_d = DetectionConfig(alpha=0.3, xi=0.5)

        results = detect(model, tokenizer, prompts, sampling_config=cfg_s, detection_config=cfg_d)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, DetectionResult)
            assert isinstance(r.is_contaminated, bool)
            assert 0.0 <= r.confidence <= 1.0
            assert 0.0 <= r.peakedness <= 1.0

    def test_facade_skips_failed_prompts(self, model_and_tokenizer):
        """If a prompt somehow fails, the facade should skip it and continue."""
        model, tokenizer = model_and_tokenizer
        prompts = ["Question: Hello?\nAnswer:"]
        cfg_s = SamplingConfig(n_samples=3, max_new_tokens=20)
        cfg_d = DetectionConfig(alpha=0.3, xi=0.5)

        results = detect(model, tokenizer, prompts, sampling_config=cfg_s, detection_config=cfg_d)
        # Should succeed for a normal prompt
        assert len(results) == 1

    def test_facade_empty_prompts(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        results = detect(model, tokenizer, [], sampling_config=SamplingConfig(n_samples=3, max_new_tokens=20))
        assert results == []
