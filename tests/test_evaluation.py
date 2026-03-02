"""Tests for the evaluation framework: metrics, confidence intervals, significance, exporter.

Includes property-based tests (hypothesis) and standard unit tests.
"""

import json
import os
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from contamination_detection.evaluation.metrics import (
    MetricsResult,
    compute_metrics,
    compute_metrics_by_condition,
)
from contamination_detection.evaluation.confidence import (
    bootstrap_confidence_intervals,
)
from contamination_detection.evaluation.significance import (
    mcnemar_test,
    paired_bootstrap_test,
)
from contamination_detection.evaluation.exporter import (
    export_csv,
    export_latex,
    export_json,
)


# ═══════════════════════════════════════════════════════════════════════
# Property-Based Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPropertyMetricBoundsAndRelationships:
    """Property 9: Metric Bounds and Relationships.

    *For any* set of predictions and ground truth labels:
    - All metrics (accuracy, precision, recall, F1, AUC) SHALL be in [0, 1]
    - F1 SHALL equal 2 * (precision * recall) / (precision + recall) when both > 0

    **Validates: Requirements 18.1, 18.3**
    """

    @given(
        data=st.lists(
            st.tuples(st.integers(min_value=0, max_value=1), st.integers(min_value=0, max_value=1)),
            min_size=2,
            max_size=200,
        ),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_all_metrics_in_unit_interval(self, data):
        y_true = np.array([d[0] for d in data])
        y_pred = np.array([d[1] for d in data])
        # Need both classes for meaningful metrics
        assume(len(np.unique(y_true)) == 2)

        result = compute_metrics(y_true, y_pred)

        assert 0.0 <= result.accuracy <= 1.0, f"accuracy={result.accuracy}"
        assert 0.0 <= result.precision <= 1.0, f"precision={result.precision}"
        assert 0.0 <= result.recall <= 1.0, f"recall={result.recall}"
        assert 0.0 <= result.f1 <= 1.0, f"f1={result.f1}"
        assert 0.0 <= result.auc <= 1.0, f"auc={result.auc}"

    @given(
        data=st.lists(
            st.tuples(st.integers(min_value=0, max_value=1), st.integers(min_value=0, max_value=1)),
            min_size=2,
            max_size=200,
        ),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_f1_formula_when_both_positive(self, data):
        y_true = np.array([d[0] for d in data])
        y_pred = np.array([d[1] for d in data])
        assume(len(np.unique(y_true)) == 2)

        result = compute_metrics(y_true, y_pred)

        if result.precision > 0 and result.recall > 0:
            expected_f1 = (
                2 * result.precision * result.recall
                / (result.precision + result.recall)
            )
            assert abs(result.f1 - expected_f1) < 1e-9, (
                f"F1={result.f1} != 2*P*R/(P+R)={expected_f1} "
                f"(P={result.precision}, R={result.recall})"
            )


class TestPropertyConfidenceIntervalCoverage:
    """Property 10: Confidence Interval Coverage.

    *For any* bootstrap confidence interval computation with confidence level alpha,
    the interval SHALL contain the point estimate.

    **Validates: Requirements 19.2**
    """

    @given(
        data=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=1),
                st.integers(min_value=0, max_value=1),
            ),
            min_size=10,
            max_size=100,
        ),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_ci_contains_point_estimate(self, data, seed):
        y_true = np.array([d[0] for d in data])
        y_pred = np.array([d[1] for d in data])
        assume(len(np.unique(y_true)) == 2)

        cis = bootstrap_confidence_intervals(
            y_true, y_pred, n_bootstrap=200, confidence=0.95, seed=seed
        )

        for metric_name, (lo, pe, hi) in cis.items():
            assert lo <= pe <= hi, (
                f"{metric_name}: CI [{lo}, {hi}] does not contain "
                f"point estimate {pe}"
            )


# ═══════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestMetricsUnit:
    """Unit tests for metrics computation (Requirement 18)."""

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        result = compute_metrics(y_true, y_pred)
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        assert result.accuracy == 0.0

    def test_f1_formula_correctness(self):
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1, 0, 0])
        result = compute_metrics(y_true, y_pred)
        # precision = 2/3, recall = 2/3
        expected_f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
        assert abs(result.f1 - expected_f1) < 1e-9

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = compute_metrics(y_true, y_pred)
        assert result.confusion_matrix.shape == (2, 2)

    def test_metrics_by_condition(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        conditions = ["70M", "70M", "70M", "160M", "160M", "160M"]
        results = compute_metrics_by_condition(y_true, y_pred, None, conditions)
        assert "70M" in results
        assert "160M" in results
        assert results["70M"].accuracy == 1.0
        assert results["160M"].accuracy == 1.0


class TestConfidenceIntervalsUnit:
    """Unit tests for bootstrap CI computation (Requirement 19)."""

    def test_ci_on_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1] * 25)
        y_pred = np.array([0, 0, 1, 1] * 25)
        cis = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=500, seed=42)
        # Perfect predictions → all CIs should be [1.0, 1.0, 1.0]
        assert cis["accuracy"][1] == 1.0
        assert cis["accuracy"][0] == 1.0
        assert cis["accuracy"][2] == 1.0

    def test_ci_on_synthetic_data(self):
        rng = np.random.RandomState(123)
        y_true = rng.randint(0, 2, size=200)
        y_pred = y_true.copy()
        # Flip 20% of predictions
        flip_idx = rng.choice(200, size=40, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

        cis = bootstrap_confidence_intervals(y_true, y_pred, n_bootstrap=500, seed=42)
        for metric_name, (lo, pe, hi) in cis.items():
            assert lo <= pe <= hi, f"{metric_name}: [{lo}, {hi}] vs {pe}"
            assert 0.0 <= lo and hi <= 1.0


class TestSignificanceUnit:
    """Unit tests for significance tests (Requirement 19)."""

    def test_mcnemar_identical_methods(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        # Same predictions → p = 1.0
        p = mcnemar_test(y_true, y_pred, y_pred)
        assert p == 1.0

    def test_mcnemar_different_methods(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 10)
        y_pred_a = y_true.copy()  # perfect
        y_pred_b = 1 - y_true  # all wrong
        p = mcnemar_test(y_true, y_pred_a, y_pred_b)
        # Should be highly significant
        assert p < 0.05

    def test_paired_bootstrap_identical(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        p = paired_bootstrap_test(y_true, y_pred, y_pred, n_bootstrap=500, seed=42)
        # Identical methods → p should be high (all diffs are 0, and |0| >= |0|)
        assert p >= 0.5

    def test_paired_bootstrap_different(self):
        rng = np.random.RandomState(99)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        # Method A: 90% correct
        y_pred_a = y_true.copy()
        flip_a = rng.choice(n, size=20, replace=False)
        y_pred_a[flip_a] = 1 - y_pred_a[flip_a]
        # Method B: 50% correct (random)
        y_pred_b = rng.randint(0, 2, size=n)
        p = paired_bootstrap_test(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, seed=42)
        # 90% vs ~50% should be significant
        assert p < 0.05


class TestExporterUnit:
    """Unit tests for CSV/LaTeX/JSON export (Requirement 18, 24)."""

    @pytest.fixture
    def sample_results(self):
        return {
            "70M_epoch0": MetricsResult(
                accuracy=0.85, precision=0.80, recall=0.90,
                f1=0.8471, auc=0.88,
                confusion_matrix=np.array([[40, 10], [5, 45]]),
            ),
            "160M_epoch5": MetricsResult(
                accuracy=0.92, precision=0.91, recall=0.93,
                f1=0.9199, auc=0.95,
                confusion_matrix=np.array([[45, 5], [3, 47]]),
            ),
        }

    def test_csv_export(self, sample_results, tmp_path):
        path = str(tmp_path / "metrics.csv")
        export_csv(sample_results, path)
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "condition" in content
        assert "accuracy" in content
        assert "70M_epoch0" in content
        assert "160M_epoch5" in content
        # Check it's valid CSV with correct number of rows
        lines = content.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_latex_export(self, sample_results, tmp_path):
        path = str(tmp_path / "metrics.tex")
        export_latex(sample_results, path, caption="Test table", label="tab:test")
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert r"\begin{table}" in content
        assert r"\begin{tabular}" in content
        assert r"\end{table}" in content
        assert r"\toprule" in content
        assert r"\bottomrule" in content
        assert r"\caption{Test table}" in content
        assert r"\label{tab:test}" in content
        assert "Accuracy" in content

    def test_json_export(self, sample_results, tmp_path):
        path = str(tmp_path / "metrics.json")
        export_json(sample_results, path)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert "70M_epoch0" in data
        assert "160M_epoch5" in data
        assert data["70M_epoch0"]["accuracy"] == 0.85
        assert "confusion_matrix" in data["70M_epoch0"]
        assert isinstance(data["70M_epoch0"]["confusion_matrix"], list)

    def test_csv_creates_directory(self, sample_results, tmp_path):
        path = str(tmp_path / "subdir" / "metrics.csv")
        export_csv(sample_results, path)
        assert os.path.exists(path)

    def test_latex_escapes_underscores(self, sample_results, tmp_path):
        path = str(tmp_path / "metrics.tex")
        export_latex(sample_results, path)
        with open(path) as f:
            content = f.read()
        # Underscores in condition names should be escaped
        assert r"70M\_epoch0" in content
