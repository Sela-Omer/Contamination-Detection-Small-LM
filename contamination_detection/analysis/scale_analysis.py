"""Scale analysis module for model size vs detection accuracy.

Fits regression models to quantify the relationship between model parameter
count and detection accuracy, identifies threshold effects, and compares
scale effects across detection methods.

Requirements: 21.1, 21.2, 21.3, 21.4, 21.5
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats

from contamination_detection.visualization.plots import (
    COLOR_PALETTE,
    setup_publication_style,
    _ensure_dir,
)

logger = logging.getLogger("contamination_detection.analysis.scale_analysis")

# Canonical model sizes in parameters
MODEL_PARAM_COUNTS: Dict[str, float] = {
    "70M": 70e6,
    "160M": 160e6,
    "410M": 410e6,
    "1B": 1e9,
}


@dataclass
class RegressionResult:
    """Result of fitting a regression model."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float
    method: str = ""
    contamination_level: Optional[int] = None


@dataclass
class ThresholdEffect:
    """A detected threshold effect between adjacent model sizes."""
    smaller_model: str
    larger_model: str
    accuracy_change_pp: float  # percentage points
    method: str
    contamination_level: int


@dataclass
class ScaleAnalysisResult:
    """Complete result of scale analysis."""
    regressions: Dict[str, List[RegressionResult]]  # method -> list per contam level
    threshold_effects: List[ThresholdEffect]
    method_comparison: Dict[str, float]  # method -> average slope
    model_sizes: List[str]
    param_counts: List[float]


def fit_scale_regression(
    model_sizes: List[str],
    accuracies: List[float],
    method: str = "",
    contamination_level: Optional[int] = None,
) -> RegressionResult:
    """Fit a linear regression: log(param_count) → detection accuracy.

    Args:
        model_sizes: List of model size labels (e.g. ["70M", "160M"]).
        accuracies: Corresponding detection accuracies.
        method: Detection method name for labelling.
        contamination_level: Contamination level for labelling.

    Returns:
        A RegressionResult with slope, intercept, R², p-value, std error.
    """
    if len(model_sizes) < 2:
        return RegressionResult(
            slope=0.0, intercept=accuracies[0] if accuracies else 0.0,
            r_squared=0.0, p_value=1.0, std_err=0.0,
            method=method, contamination_level=contamination_level,
        )

    x = np.array([math.log10(MODEL_PARAM_COUNTS.get(s, 1.0)) for s in model_sizes])
    y = np.array(accuracies)

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

    result = RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r_value ** 2),
        p_value=float(p_value),
        std_err=float(std_err),
        method=method,
        contamination_level=contamination_level,
    )
    logger.info(
        f"Regression [{method}, contam={contamination_level}]: "
        f"slope={slope:.4f}, R²={r_value**2:.4f}, p={p_value:.4f}"
    )
    return result


def detect_threshold_effects(
    model_sizes: List[str],
    accuracies: List[float],
    method: str = "",
    contamination_level: int = 0,
    threshold_pp: float = 10.0,
) -> List[ThresholdEffect]:
    """Identify threshold effects (>threshold_pp change between adjacent sizes).

    Args:
        model_sizes: Ordered list of model size labels.
        accuracies: Corresponding detection accuracies.
        method: Detection method name.
        contamination_level: Contamination level.
        threshold_pp: Minimum percentage-point change to flag (default 10).

    Returns:
        List of ThresholdEffect instances for each detected jump.
    """
    effects: List[ThresholdEffect] = []

    for i in range(len(model_sizes) - 1):
        change_pp = (accuracies[i + 1] - accuracies[i]) * 100.0
        if abs(change_pp) > threshold_pp:
            effect = ThresholdEffect(
                smaller_model=model_sizes[i],
                larger_model=model_sizes[i + 1],
                accuracy_change_pp=change_pp,
                method=method,
                contamination_level=contamination_level,
            )
            effects.append(effect)
            logger.info(
                f"Threshold effect [{method}, contam={contamination_level}]: "
                f"{model_sizes[i]} → {model_sizes[i+1]}: {change_pp:+.1f} pp"
            )

    return effects


def run_scale_analysis(
    results_by_method: Dict[str, Dict[int, Dict[str, float]]],
    model_sizes: Optional[List[str]] = None,
    error_bars: Optional[Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]] = None,
) -> ScaleAnalysisResult:
    """Run full scale analysis across methods and contamination levels.

    Args:
        results_by_method: Nested dict:
            ``{method_name: {contam_level: {model_size: accuracy}}}``.
        model_sizes: Ordered model size labels. Defaults to
            ``["70M", "160M", "410M", "1B"]``.
        error_bars: Optional CI bounds:
            ``{method: {contam_level: {model_size: (lower, upper)}}}``.

    Returns:
        A ScaleAnalysisResult with regressions, threshold effects, and
        method comparison.
    """
    if model_sizes is None:
        model_sizes = ["70M", "160M", "410M", "1B"]

    param_counts = [MODEL_PARAM_COUNTS.get(s, 0.0) for s in model_sizes]

    all_regressions: Dict[str, List[RegressionResult]] = {}
    all_threshold_effects: List[ThresholdEffect] = []
    method_slopes: Dict[str, List[float]] = {}

    for method, by_contam in results_by_method.items():
        all_regressions[method] = []
        method_slopes[method] = []

        for contam_level, by_size in sorted(by_contam.items()):
            sizes_present = [s for s in model_sizes if s in by_size]
            accs = [by_size[s] for s in sizes_present]

            if len(sizes_present) < 2:
                continue

            # Fit regression within this contamination level
            reg = fit_scale_regression(
                sizes_present, accs,
                method=method, contamination_level=contam_level,
            )
            all_regressions[method].append(reg)
            method_slopes[method].append(reg.slope)

            # Detect threshold effects
            effects = detect_threshold_effects(
                sizes_present, accs,
                method=method, contamination_level=contam_level,
            )
            all_threshold_effects.extend(effects)

    # Compare scale effects across methods (average slope)
    method_comparison = {
        m: float(np.mean(slopes)) if slopes else 0.0
        for m, slopes in method_slopes.items()
    }

    logger.info(
        f"Scale analysis complete: {len(all_threshold_effects)} threshold effects, "
        f"method slopes: {method_comparison}"
    )

    return ScaleAnalysisResult(
        regressions=all_regressions,
        threshold_effects=all_threshold_effects,
        method_comparison=method_comparison,
        model_sizes=model_sizes,
        param_counts=param_counts,
    )


def plot_scale_analysis(
    results_by_method: Dict[str, Dict[int, Dict[str, float]]],
    output_path: str = "scale_analysis.pdf",
    model_sizes: Optional[List[str]] = None,
    error_bars: Optional[Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]] = None,
    title: str = "Detection Accuracy vs. Model Scale",
) -> str:
    """Generate publication-quality figure showing scale-detection relationship.

    Plots accuracy vs. log(model parameters) for each method, with one
    subplot per contamination level. Includes regression lines and error bars.

    Args:
        results_by_method: ``{method: {contam_level: {model_size: accuracy}}}``.
        output_path: Destination PDF path.
        model_sizes: Ordered model size labels.
        error_bars: Optional ``{method: {contam_level: {model_size: (lo, hi)}}}``.
        title: Overall figure title.

    Returns:
        The output_path written.
    """
    setup_publication_style()
    _ensure_dir(output_path)

    if model_sizes is None:
        model_sizes = ["70M", "160M", "410M", "1B"]

    # Collect all contamination levels across methods
    all_contam_levels = sorted({
        cl for by_contam in results_by_method.values() for cl in by_contam
    })

    n_panels = len(all_contam_levels)
    if n_panels == 0:
        # Nothing to plot — create an empty figure
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
        fig.savefig(output_path, format="pdf")
        plt.close(fig)
        return output_path

    n_cols = min(n_panels, 2)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)

    x_positions = np.arange(len(model_sizes))
    log_params = [math.log10(MODEL_PARAM_COUNTS.get(s, 1.0)) for s in model_sizes]

    for panel_idx, contam_level in enumerate(all_contam_levels):
        row, col = divmod(panel_idx, n_cols)
        ax = axes[row][col]

        for method_idx, (method, by_contam) in enumerate(sorted(results_by_method.items())):
            if contam_level not in by_contam:
                continue

            by_size = by_contam[contam_level]
            y = [by_size.get(s, np.nan) for s in model_sizes]
            color = COLOR_PALETTE[method_idx % len(COLOR_PALETTE)]

            # Error bars
            yerr = None
            if error_bars and method in error_bars and contam_level in error_bars[method]:
                eb = error_bars[method][contam_level]
                yerr_lo = [by_size.get(s, 0) - eb.get(s, (by_size.get(s, 0), 0))[0]
                           for s in model_sizes]
                yerr_hi = [eb.get(s, (0, by_size.get(s, 0)))[1] - by_size.get(s, 0)
                           for s in model_sizes]
                yerr = [yerr_lo, yerr_hi]

            ax.errorbar(
                x_positions, y, yerr=yerr, label=method, color=color,
                marker="o", capsize=4, capthick=1.5, linewidth=2,
            )

            # Regression line
            valid = [(xp, yv) for xp, yv in zip(log_params, y) if not np.isnan(yv)]
            if len(valid) >= 2:
                xr = np.array([v[0] for v in valid])
                yr = np.array([v[1] for v in valid])
                slope, intercept, _, _, _ = scipy_stats.linregress(xr, yr)
                x_line = np.linspace(min(log_params), max(log_params), 50)
                # Map x_line back to x_positions space for plotting
                x_plot = np.interp(x_line, log_params, x_positions)
                y_line = slope * x_line + intercept
                ax.plot(x_plot, y_line, color=color, linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_sizes)
        ax.set_xlabel("Model Size")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Contamination Level = {contam_level}")
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for panel_idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(panel_idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Scale analysis figure saved to {output_path}")
    return output_path
