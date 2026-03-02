"""Tests for the data pipeline: loading, splitting, formatting, contamination, serialization.

Includes property-based tests (hypothesis) and standard unit tests.
"""

import tempfile
from collections import Counter

import numpy as np
import pytest
from datasets import Dataset
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from contamination_detection.data.loader import (
    load_qa_dataset,
    load_saved_dataset,
    save_dataset,
)
from contamination_detection.data.splitter import create_splits
from contamination_detection.data.formatter import format_prompts
from contamination_detection.data.contamination import create_contaminated_training_set


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_dataset(n: int, seed: int = 0) -> Dataset:
    """Create a small synthetic QA dataset for testing."""
    rng = np.random.RandomState(seed)
    return Dataset.from_dict(
        {
            "question": [f"Question {i}?" for i in range(n)],
            "answer": [f"Answer {i}" for i in range(n)],
            "uid": list(range(n)),  # unique id for overlap checks
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# Property-Based Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPropertyDataSplitNonOverlap:
    """Property 1: Data Split Non-Overlap.

    *For any* dataset and split configuration, the contamination set and
    evaluation set SHALL have zero intersection.

    **Validates: Requirements 2.2**
    """

    @given(
        n=st.integers(min_value=10, max_value=200),
        train_ratio=st.floats(min_value=0.1, max_value=0.5),
        contam_ratio=st.floats(min_value=0.05, max_value=0.3),
        eval_ratio=st.floats(min_value=0.05, max_value=0.3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_contamination_eval_no_overlap(
        self, n, train_ratio, contam_ratio, eval_ratio, seed
    ):
        """Contamination and evaluation sets must never share examples."""
        # Skip if ratios exceed 1.0
        if train_ratio + contam_ratio + eval_ratio > 1.0:
            return

        ds = _make_dataset(n)
        splits = create_splits(ds, train_ratio, contam_ratio, eval_ratio, seed)

        contam_uids = set(splits.contamination["uid"])
        eval_uids = set(splits.evaluation["uid"])
        assert contam_uids.isdisjoint(eval_uids), (
            f"Overlap found: {contam_uids & eval_uids}"
        )

    @given(
        n=st.integers(min_value=10, max_value=200),
        train_ratio=st.floats(min_value=0.1, max_value=0.5),
        contam_ratio=st.floats(min_value=0.05, max_value=0.3),
        eval_ratio=st.floats(min_value=0.05, max_value=0.3),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_three_splits_no_overlap(
        self, n, train_ratio, contam_ratio, eval_ratio, seed
    ):
        """No pair of splits should share any example."""
        if train_ratio + contam_ratio + eval_ratio > 1.0:
            return

        ds = _make_dataset(n)
        splits = create_splits(ds, train_ratio, contam_ratio, eval_ratio, seed)

        train_uids = set(splits.train["uid"])
        contam_uids = set(splits.contamination["uid"])
        eval_uids = set(splits.evaluation["uid"])

        assert train_uids.isdisjoint(contam_uids)
        assert train_uids.isdisjoint(eval_uids)
        assert contam_uids.isdisjoint(eval_uids)


class TestPropertyRandomSubsetSelection:
    """Property 3: Random Subset Selection.

    *For any* subset selection with seed s, the selected indices SHALL NOT
    simply be the first-N records — they must be shuffled.

    **Validates: Requirements 2.3**
    """

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_split_is_not_first_n(self, seed):
        """Selected examples should not be the sequential first-N."""
        n = 100
        ds = _make_dataset(n)
        splits = create_splits(ds, 0.5, 0.25, 0.25, seed)

        train_uids = splits.train["uid"]
        first_n = list(range(len(train_uids)))

        # It's astronomically unlikely that a random shuffle of 100 items
        # produces the exact sequential order, so this is a valid check.
        assert train_uids != first_n, (
            "Train split appears to be the first-N records (not shuffled)"
        )


class TestPropertyContaminationRatioAccuracy:
    """Property 2: Contamination Ratio Accuracy.

    *For any* valid contamination_epochs, the contaminated training set SHALL
    contain exactly contamination_epochs × |contamination_set| contamination
    examples added to the clean set.

    **Validates: Requirements 4.1**
    """

    @given(
        n_clean=st.integers(min_value=5, max_value=100),
        n_contam=st.integers(min_value=1, max_value=50),
        epochs=st.integers(min_value=0, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_contaminated_set_size(self, n_clean, n_contam, epochs, seed):
        """Combined set size must equal clean + epochs * contamination."""
        clean = _make_dataset(n_clean, seed=0)
        contam = _make_dataset(n_contam, seed=9999)

        combined = create_contaminated_training_set(clean, contam, epochs, seed)
        expected = n_clean + n_contam * epochs
        assert len(combined) == expected, (
            f"Expected {expected}, got {len(combined)}"
        )

    @given(
        epochs=st.integers(min_value=1, max_value=10),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_zero_epochs_returns_clean_only(self, epochs, seed):
        """With epochs=0, the result should have only clean examples."""
        clean = _make_dataset(20, seed=0)
        contam = _make_dataset(5, seed=9999)

        combined = create_contaminated_training_set(clean, contam, 0, seed)
        assert len(combined) == len(clean)


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestDatasetLoading:
    """Unit tests for dataset loading (Requirement 1)."""

    def test_load_qasc(self):
        ds = load_qa_dataset("QASC")
        assert len(ds) > 0
        assert "question" in ds.column_names

    def test_load_strategyqa(self):
        ds = load_qa_dataset("StrategyQA")
        assert len(ds) > 0
        assert "question" in ds.column_names

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_qa_dataset("NonExistent")


class TestDatasetSplitting:
    """Unit tests for dataset splitting (Requirement 2)."""

    def test_split_sizes(self):
        ds = _make_dataset(100)
        splits = create_splits(ds, 0.6, 0.2, 0.2, seed=42)
        assert len(splits.train) == 60
        assert len(splits.contamination) == 20
        assert len(splits.evaluation) == 20

    def test_invalid_ratios_raise(self):
        ds = _make_dataset(100)
        with pytest.raises(ValueError, match="exceeds 1.0"):
            create_splits(ds, 0.5, 0.4, 0.3, seed=42)

    def test_metadata_recorded(self):
        ds = _make_dataset(50)
        splits = create_splits(ds, 0.6, 0.2, 0.2, seed=99)
        assert splits.metadata["seed"] == 99
        assert splits.metadata["total_examples"] == 50

    def test_reproducibility(self):
        ds = _make_dataset(100)
        s1 = create_splits(ds, 0.6, 0.2, 0.2, seed=42)
        s2 = create_splits(ds, 0.6, 0.2, 0.2, seed=42)
        assert s1.train["uid"] == s2.train["uid"]
        assert s1.contamination["uid"] == s2.contamination["uid"]
        assert s1.evaluation["uid"] == s2.evaluation["uid"]


class TestPromptFormatting:
    """Unit tests for prompt formatting (Requirement 3)."""

    def test_format_basic(self):
        examples = [{"question": "What is 2+2?"}, {"question": "Why is the sky blue?"}]
        prompts = format_prompts(examples)
        assert len(prompts) == 2
        assert prompts[0] == "Question: What is 2+2?\nAnswer:"
        assert prompts[1] == "Question: Why is the sky blue?\nAnswer:"

    def test_preserves_original_text(self):
        text = "A question with special chars: é, ñ, ü?"
        prompts = format_prompts([{"question": text}])
        assert text in prompts[0]

    def test_output_length_matches_input(self):
        ds = _make_dataset(25)
        prompts = format_prompts(ds)
        assert len(prompts) == 25

    def test_format_hf_dataset(self):
        ds = _make_dataset(10)
        prompts = format_prompts(ds)
        assert all(p.startswith("Question:") for p in prompts)
        assert all(p.endswith("Answer:") for p in prompts)


class TestContaminatedTrainingSet:
    """Unit tests for contaminated training set creation (Requirement 4)."""

    def test_epochs_zero(self):
        clean = _make_dataset(20)
        contam = _make_dataset(5, seed=99)
        result = create_contaminated_training_set(clean, contam, 0, seed=42)
        assert len(result) == 20

    def test_epochs_one(self):
        clean = _make_dataset(20)
        contam = _make_dataset(5, seed=99)
        result = create_contaminated_training_set(clean, contam, 1, seed=42)
        assert len(result) == 25

    def test_epochs_five(self):
        clean = _make_dataset(20)
        contam = _make_dataset(5, seed=99)
        result = create_contaminated_training_set(clean, contam, 5, seed=42)
        assert len(result) == 45

    def test_negative_epochs_raises(self):
        clean = _make_dataset(10)
        contam = _make_dataset(5, seed=99)
        with pytest.raises(ValueError):
            create_contaminated_training_set(clean, contam, -1)


class TestDatasetSerialization:
    """Unit tests for dataset serialization (Requirement 5)."""

    def test_round_trip(self):
        ds = _make_dataset(30)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dataset(ds, tmpdir)
            loaded = load_saved_dataset(tmpdir)
            assert len(loaded) == len(ds)
            assert loaded["question"] == ds["question"]
            assert loaded["uid"] == ds["uid"]

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_saved_dataset("/tmp/nonexistent_dataset_path_12345")
