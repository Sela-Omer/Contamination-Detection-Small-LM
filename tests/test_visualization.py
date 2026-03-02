"""Unit tests for the visualization module.

Each test verifies that a plot function generates a valid, non-empty PDF file
using synthetic data. No display is needed (Agg backend).
"""

import os

import numpy as np
import pytest

from contamination_detection.visualization.plots import (
    setup_publication_style,
    plot_roc_curves,
    plot_accuracy_vs_model_size,
    plot_accuracy_vs_contamination_level,
    plot_performance_heatmap,
    plot_peakedness_distributions,
    plot_training_loss_curves,
)


@pytest.fixture(autouse=True)
def _apply_style():
    """Apply publication style before every test."""
    setup_publication_style()


class TestPublicationStyle:
    """11.7 — Consistent publication style configuration."""

    def test_setup_does_not_raise(self):
        setup_publication_style()

    def test_font_size_at_least_12(self):
        import matplotlib
        setup_publication_style()
        assert matplotlib.rcParams["font.size"] >= 12
        assert matplotlib.rcParams["axes.labelsize"] >= 12
        assert matplotlib.rcParams["xtick.labelsize"] >= 12
        assert matplotlib.rcParams["ytick.labelsize"] >= 12


class TestROCCurves:
    """11.1 — ROC curve generator."""

    def test_generates_pdf(self, tmp_path):
        rng = np.random.RandomState(0)
        y_true = {"cond_A": np.array([0, 0, 1, 1, 0, 1, 0, 1])}
        y_scores = {"cond_A": rng.rand(8)}
        path = str(tmp_path / "roc.pdf")

        result = plot_roc_curves(y_true, y_scores, output_path=path)

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_multiple_conditions(self, tmp_path):
        rng = np.random.RandomState(1)
        y_true = {
            "70M_epoch1": np.array([0, 0, 1, 1, 0, 1]),
            "160M_epoch5": np.array([0, 1, 1, 0, 1, 1]),
        }
        y_scores = {k: rng.rand(len(v)) for k, v in y_true.items()}
        path = str(tmp_path / "roc_multi.pdf")

        plot_roc_curves(y_true, y_scores, output_path=path)

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestAccuracyVsModelSize:
    """11.2 — Accuracy vs. model size plots."""

    def test_generates_pdf(self, tmp_path):
        accuracies = {
            "CDD": {"70M": 0.75, "160M": 0.80, "410M": 0.85, "1B": 0.90},
            "Perplexity": {"70M": 0.60, "160M": 0.65, "410M": 0.70, "1B": 0.75},
        }
        path = str(tmp_path / "acc_model.pdf")

        result = plot_accuracy_vs_model_size(accuracies, output_path=path)

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_with_error_bars(self, tmp_path):
        accuracies = {
            "CDD": {"70M": 0.75, "160M": 0.80, "410M": 0.85, "1B": 0.90},
        }
        ci_lo = {"CDD": {"70M": 0.70, "160M": 0.75, "410M": 0.80, "1B": 0.85}}
        ci_hi = {"CDD": {"70M": 0.80, "160M": 0.85, "410M": 0.90, "1B": 0.95}}
        path = str(tmp_path / "acc_model_ci.pdf")

        plot_accuracy_vs_model_size(
            accuracies, ci_lower=ci_lo, ci_upper=ci_hi, output_path=path,
        )

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestAccuracyVsContaminationLevel:
    """11.3 — Accuracy vs. contamination level plots."""

    def test_generates_pdf(self, tmp_path):
        accuracies = {
            "70M": {0: 0.50, 1: 0.60, 5: 0.75, 10: 0.85},
            "1B": {0: 0.55, 1: 0.70, 5: 0.85, 10: 0.92},
        }
        path = str(tmp_path / "acc_contam.pdf")

        result = plot_accuracy_vs_contamination_level(accuracies, output_path=path)

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_with_error_bars(self, tmp_path):
        accuracies = {"70M": {0: 0.50, 1: 0.60, 5: 0.75, 10: 0.85}}
        ci_lo = {"70M": {0: 0.45, 1: 0.55, 5: 0.70, 10: 0.80}}
        ci_hi = {"70M": {0: 0.55, 1: 0.65, 5: 0.80, 10: 0.90}}
        path = str(tmp_path / "acc_contam_ci.pdf")

        plot_accuracy_vs_contamination_level(
            accuracies, ci_lower=ci_lo, ci_upper=ci_hi, output_path=path,
        )

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPerformanceHeatmap:
    """11.4 — Performance heatmaps."""

    def test_generates_pdf(self, tmp_path):
        values = np.array([
            [0.50, 0.60, 0.75, 0.85],
            [0.55, 0.65, 0.80, 0.88],
            [0.60, 0.72, 0.85, 0.92],
            [0.65, 0.78, 0.90, 0.95],
        ])
        path = str(tmp_path / "heatmap.pdf")

        result = plot_performance_heatmap(values, output_path=path, metric_name="F1")

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_custom_labels(self, tmp_path):
        values = np.array([[0.7, 0.8], [0.85, 0.9]])
        path = str(tmp_path / "heatmap_custom.pdf")

        plot_performance_heatmap(
            values,
            row_labels=["Small", "Large"],
            col_labels=["Low", "High"],
            output_path=path,
        )

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestPeakednessDistributions:
    """11.5 — Peakedness distribution plots."""

    def test_histogram_generates_pdf(self, tmp_path):
        rng = np.random.RandomState(42)
        contaminated = rng.beta(5, 2, size=100)
        clean = rng.beta(2, 5, size=100)
        path = str(tmp_path / "peaked_hist.pdf")

        result = plot_peakedness_distributions(
            contaminated, clean, output_path=path, kind="histogram",
        )

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_violin_generates_pdf(self, tmp_path):
        rng = np.random.RandomState(42)
        contaminated = rng.beta(5, 2, size=50)
        clean = rng.beta(2, 5, size=50)
        path = str(tmp_path / "peaked_violin.pdf")

        plot_peakedness_distributions(
            contaminated, clean, output_path=path, kind="violin",
        )

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0


class TestTrainingLossCurves:
    """11.6 — Training loss curves."""

    def test_generates_pdf(self, tmp_path):
        losses = {
            "70M_epoch5": [2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.4, 1.3],
            "160M_epoch5": [2.4, 2.1, 1.8, 1.6, 1.4, 1.2, 1.1, 1.0],
        }
        path = str(tmp_path / "loss.pdf")

        result = plot_training_loss_curves(losses, output_path=path)

        assert result == path
        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0

    def test_single_run(self, tmp_path):
        losses = {"70M_epoch1": [3.0, 2.5, 2.0, 1.8]}
        path = str(tmp_path / "loss_single.pdf")

        plot_training_loss_curves(losses, output_path=path)

        assert os.path.isfile(path)
        assert os.path.getsize(path) > 0
