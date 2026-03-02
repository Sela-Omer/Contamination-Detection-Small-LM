"""Unit tests for scale analysis module.

Tests regression fitting on synthetic data, threshold effect detection,
and visualization PDF generation.

Requirements: 21.1–21.5
"""

import math
import os
import tempfile

import numpy as np
import pytest

from contamination_detection.analysis.scale_analysis import (
    MODEL_PARAM_COUNTS,
    RegressionResult,
    ScaleAnalysisResult,
    ThresholdEffect,
    detect_threshold_effects,
    fit_scale_regression,
    plot_scale_analysis,
    run_scale_analysis,
)


class TestFitScaleRegression:
    """Test regression fitting on synthetic data."""

    def test_perfect_positive_trend(self):
        """Accuracy increases linearly with log(params) → positive slope, R²≈1."""
        sizes = ["70M", "160M", "410M", "1B"]
        log_params = [math.log10(MODEL_PARAM_COUNTS[s]) for s in sizes]
        # Create a perfect linear relationship
        accs = [0.3 + 0.2 * (lp - log_params[0]) / (log_params[-1] - log_params[0])
                for lp in log_params]

        result = fit_scale_regression(sizes, accs, method="CDD", contamination_level=5)

        assert isinstance(result, RegressionResult)
        assert result.slope > 0, "Slope should be positive for increasing accuracy"
        assert result.r_squared > 0.99, "R² should be ~1 for perfect linear data"
        assert result.method == "CDD"
        assert result.contamination_level == 5

    def test_flat_trend(self):
        """Constant accuracy → slope ≈ 0."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.5, 0.5, 0.5, 0.5]

        result = fit_scale_regression(sizes, accs)

        assert abs(result.slope) < 1e-10
        assert result.intercept == pytest.approx(0.5, abs=0.01)

    def test_negative_trend(self):
        """Decreasing accuracy → negative slope."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.9, 0.7, 0.5, 0.3]

        result = fit_scale_regression(sizes, accs)

        assert result.slope < 0

    def test_single_point(self):
        """Single data point → degenerate result."""
        result = fit_scale_regression(["70M"], [0.8])

        assert result.slope == 0.0
        assert result.intercept == 0.8
        assert result.r_squared == 0.0

    def test_two_points(self):
        """Two points → perfect fit (R²=1)."""
        sizes = ["70M", "1B"]
        accs = [0.4, 0.8]

        result = fit_scale_regression(sizes, accs)

        assert result.r_squared == pytest.approx(1.0, abs=1e-10)
        assert result.slope > 0


class TestDetectThresholdEffects:
    """Test threshold effect detection."""

    def test_large_jump_detected(self):
        """A 20pp jump between adjacent sizes should be detected."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.5, 0.5, 0.5, 0.7]  # 20pp jump at 410M→1B

        effects = detect_threshold_effects(
            sizes, accs, method="CDD", contamination_level=5
        )

        assert len(effects) == 1
        assert effects[0].smaller_model == "410M"
        assert effects[0].larger_model == "1B"
        assert effects[0].accuracy_change_pp == pytest.approx(20.0)
        assert effects[0].method == "CDD"
        assert effects[0].contamination_level == 5

    def test_no_threshold_effect(self):
        """Gradual changes below 10pp should not be flagged."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.50, 0.55, 0.60, 0.65]  # 5pp each

        effects = detect_threshold_effects(sizes, accs)

        assert len(effects) == 0

    def test_multiple_jumps(self):
        """Multiple large jumps should all be detected."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.3, 0.5, 0.5, 0.7]  # 20pp at 70M→160M and 410M→1B

        effects = detect_threshold_effects(sizes, accs)

        assert len(effects) == 2

    def test_negative_jump(self):
        """A large negative change should also be detected."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.8, 0.8, 0.5, 0.5]  # -30pp at 160M→410M

        effects = detect_threshold_effects(sizes, accs)

        assert len(effects) == 1
        assert effects[0].accuracy_change_pp < 0

    def test_custom_threshold(self):
        """Custom threshold_pp should be respected."""
        sizes = ["70M", "160M", "410M", "1B"]
        accs = [0.50, 0.56, 0.58, 0.60]  # 6pp, 2pp, 2pp

        effects_5pp = detect_threshold_effects(sizes, accs, threshold_pp=5.0)
        effects_10pp = detect_threshold_effects(sizes, accs, threshold_pp=10.0)

        assert len(effects_5pp) == 1  # only the 6pp jump
        assert len(effects_10pp) == 0


class TestRunScaleAnalysis:
    """Test the full scale analysis pipeline."""

    def test_basic_analysis(self):
        """Run analysis on synthetic data for two methods."""
        results = {
            "CDD": {
                0: {"70M": 0.5, "160M": 0.6, "410M": 0.7, "1B": 0.8},
                5: {"70M": 0.6, "160M": 0.7, "410M": 0.8, "1B": 0.9},
            },
            "Random": {
                0: {"70M": 0.5, "160M": 0.5, "410M": 0.5, "1B": 0.5},
                5: {"70M": 0.5, "160M": 0.5, "410M": 0.5, "1B": 0.5},
            },
        }

        analysis = run_scale_analysis(results)

        assert isinstance(analysis, ScaleAnalysisResult)
        assert "CDD" in analysis.regressions
        assert "Random" in analysis.regressions
        assert analysis.method_comparison["CDD"] > analysis.method_comparison["Random"]
        assert len(analysis.model_sizes) == 4

    def test_with_threshold_effects(self):
        """Analysis should detect threshold effects in the data."""
        results = {
            "CDD": {
                5: {"70M": 0.5, "160M": 0.5, "410M": 0.5, "1B": 0.75},
            },
        }

        analysis = run_scale_analysis(results)

        assert len(analysis.threshold_effects) >= 1
        assert any(e.accuracy_change_pp > 10 for e in analysis.threshold_effects)


class TestPlotScaleAnalysis:
    """Test scale analysis visualization generates PDF."""

    def test_generates_pdf(self):
        """Plot function should create a valid PDF file."""
        results = {
            "CDD": {
                0: {"70M": 0.5, "160M": 0.6, "410M": 0.7, "1B": 0.8},
                5: {"70M": 0.6, "160M": 0.7, "410M": 0.8, "1B": 0.9},
            },
            "Perplexity": {
                0: {"70M": 0.5, "160M": 0.55, "410M": 0.6, "1B": 0.65},
                5: {"70M": 0.55, "160M": 0.6, "410M": 0.65, "1B": 0.7},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scale.pdf")
            result_path = plot_scale_analysis(results, output_path=path)

            assert result_path == path
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_empty_data(self):
        """Plot should handle empty data gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "empty.pdf")
            result_path = plot_scale_analysis({}, output_path=path)

            assert os.path.exists(path)

    def test_single_method_single_level(self):
        """Plot should work with minimal data."""
        results = {
            "CDD": {
                0: {"70M": 0.5, "160M": 0.6, "410M": 0.7, "1B": 0.8},
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single.pdf")
            result_path = plot_scale_analysis(results, output_path=path)

            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
