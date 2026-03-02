"""Publication-quality visualization functions for contamination detection experiments.

All plots use a consistent style (font size >= 12, PDF backend, shared color palette)
and save output as PDF files suitable for inclusion in an ACL-format LaTeX report.

Usage:
    from contamination_detection.visualization.plots import (
        setup_publication_style,
        plot_roc_curves,
        plot_accuracy_vs_model_size,
        plot_accuracy_vs_contamination_level,
        plot_performance_heatmap,
        plot_peakedness_distributions,
        plot_training_loss_curves,
    )

    setup_publication_style()
    plot_roc_curves(y_true, y_scores, conditions, output_path="roc.pdf")
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger("contamination_detection.visualization.plots")

# ── Consistent color palette ──────────────────────────────────────────
COLOR_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]

MODEL_SIZES = ["70M", "160M", "410M", "1B"]
CONTAMINATION_LEVELS = [0, 1, 5, 10]


def setup_publication_style() -> None:
    """Configure matplotlib for publication-quality figures.

    Sets font size >= 12, consistent color palette, PDF-friendly backend,
    and clean axis styling. Call once before generating any plots.
    """
    matplotlib.use("Agg")
    plt.rcParams.update({
        # Font
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        # Axes
        "axes.prop_cycle": plt.cycler("color", COLOR_PALETTE),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Figure
        "figure.figsize": (7, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # PDF backend
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    logger.info("Publication style configured.")


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if they don't exist."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# ── 11.1  ROC Curve Generator ────────────────────────────────────────


def plot_roc_curves(
    y_true_by_condition: Dict[str, np.ndarray],
    y_scores_by_condition: Dict[str, np.ndarray],
    output_path: str = "roc_curves.pdf",
    title: str = "ROC Curves by Experimental Condition",
) -> str:
    """Generate ROC curves for each experimental condition.

    Args:
        y_true_by_condition: Mapping of condition label (e.g. ``"70M_epoch5"``)
            to 1-D binary ground-truth arrays.
        y_scores_by_condition: Matching mapping to continuous score arrays.
        output_path: Destination PDF path.
        title: Plot title.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots()

    for idx, (cond, y_true) in enumerate(sorted(y_true_by_condition.items())):
        y_scores = y_scores_by_condition[cond]
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        ax.plot(fpr, tpr, color=color, label=f"{cond} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"ROC curves saved to {output_path}")
    return output_path


# ── 11.2  Accuracy vs. Model Size ────────────────────────────────────


def plot_accuracy_vs_model_size(
    accuracies: Dict[str, Dict[str, float]],
    ci_lower: Optional[Dict[str, Dict[str, float]]] = None,
    ci_upper: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: str = "accuracy_vs_model_size.pdf",
    title: str = "Detection Accuracy vs. Model Size",
    model_sizes: Optional[List[str]] = None,
) -> str:
    """Plot detection accuracy vs. model size with error bars.

    Args:
        accuracies: ``{series_label: {model_size: accuracy}}``.
            Each series is one contamination level or detection method.
        ci_lower: Optional matching dict of CI lower bounds.
        ci_upper: Optional matching dict of CI upper bounds.
        output_path: Destination PDF path.
        title: Plot title.
        model_sizes: Ordered list of model size labels for the x-axis.
            Defaults to ``["70M", "160M", "410M", "1B"]``.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    if model_sizes is None:
        model_sizes = MODEL_SIZES

    fig, ax = plt.subplots()
    x = np.arange(len(model_sizes))

    for idx, (series, acc_by_size) in enumerate(sorted(accuracies.items())):
        y = [acc_by_size.get(ms, np.nan) for ms in model_sizes]
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        yerr_lo = None
        yerr_hi = None
        if ci_lower is not None and ci_upper is not None:
            lo = ci_lower.get(series, {})
            hi = ci_upper.get(series, {})
            yerr_lo = [acc_by_size.get(ms, 0) - lo.get(ms, acc_by_size.get(ms, 0))
                       for ms in model_sizes]
            yerr_hi = [hi.get(ms, acc_by_size.get(ms, 0)) - acc_by_size.get(ms, 0)
                       for ms in model_sizes]
            yerr = [yerr_lo, yerr_hi]
        else:
            yerr = None

        ax.errorbar(
            x, y, yerr=yerr, label=series, color=color,
            marker="o", capsize=4, capthick=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_sizes)
    ax.set_xlabel("Model Size")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, 1.05])

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Accuracy vs. model size saved to {output_path}")
    return output_path


# ── 11.3  Accuracy vs. Contamination Level ───────────────────────────


def plot_accuracy_vs_contamination_level(
    accuracies: Dict[str, Dict[int, float]],
    ci_lower: Optional[Dict[str, Dict[int, float]]] = None,
    ci_upper: Optional[Dict[str, Dict[int, float]]] = None,
    output_path: str = "accuracy_vs_contamination.pdf",
    title: str = "Detection Accuracy vs. Contamination Level",
    contamination_levels: Optional[List[int]] = None,
) -> str:
    """Plot detection accuracy vs. contamination level with error bars.

    Args:
        accuracies: ``{series_label: {contam_level: accuracy}}``.
            Each series is one model size or detection method.
        ci_lower: Optional matching dict of CI lower bounds.
        ci_upper: Optional matching dict of CI upper bounds.
        output_path: Destination PDF path.
        title: Plot title.
        contamination_levels: Ordered contamination epoch counts.
            Defaults to ``[0, 1, 5, 10]``.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    if contamination_levels is None:
        contamination_levels = CONTAMINATION_LEVELS

    fig, ax = plt.subplots()
    x = np.arange(len(contamination_levels))

    for idx, (series, acc_by_level) in enumerate(sorted(accuracies.items())):
        y = [acc_by_level.get(cl, np.nan) for cl in contamination_levels]
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        yerr = None
        if ci_lower is not None and ci_upper is not None:
            lo = ci_lower.get(series, {})
            hi = ci_upper.get(series, {})
            yerr_lo = [acc_by_level.get(cl, 0) - lo.get(cl, acc_by_level.get(cl, 0))
                       for cl in contamination_levels]
            yerr_hi = [hi.get(cl, acc_by_level.get(cl, 0)) - acc_by_level.get(cl, 0)
                       for cl in contamination_levels]
            yerr = [yerr_lo, yerr_hi]

        ax.errorbar(
            x, y, yerr=yerr, label=series, color=color,
            marker="s", capsize=4, capthick=1.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(cl) for cl in contamination_levels])
    ax.set_xlabel("Contamination Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, 1.05])

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Accuracy vs. contamination level saved to {output_path}")
    return output_path


# ── 11.4  Performance Heatmaps ───────────────────────────────────────


def plot_performance_heatmap(
    values: np.ndarray,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    output_path: str = "heatmap.pdf",
    title: str = "Detection Performance",
    metric_name: str = "Accuracy",
    cmap: str = "YlOrRd",
) -> str:
    """Plot a heatmap of detection performance (model size × contamination level).

    Args:
        values: 2-D array of shape ``(n_model_sizes, n_contamination_levels)``.
        row_labels: Labels for rows (model sizes). Defaults to MODEL_SIZES.
        col_labels: Labels for columns (contamination levels). Defaults to
            string representations of CONTAMINATION_LEVELS.
        output_path: Destination PDF path.
        title: Plot title.
        metric_name: Name of the metric shown (for the colorbar label).
        cmap: Matplotlib colormap name.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    if row_labels is None:
        row_labels = MODEL_SIZES
    if col_labels is None:
        col_labels = [str(c) for c in CONTAMINATION_LEVELS]

    fig, ax = plt.subplots()
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Contamination Epochs")
    ax.set_ylabel("Model Size")
    ax.set_title(title)

    # Annotate cells
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text_color = "white" if val > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_name)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Heatmap ({metric_name}) saved to {output_path}")
    return output_path


# ── 11.5  Peakedness Distribution Plots ──────────────────────────────


def plot_peakedness_distributions(
    contaminated_scores: np.ndarray,
    clean_scores: np.ndarray,
    output_path: str = "peakedness_dist.pdf",
    title: str = "Peakedness Score Distribution",
    kind: str = "histogram",
) -> str:
    """Plot peakedness score distributions for contaminated vs. clean examples.

    Args:
        contaminated_scores: 1-D array of peakedness scores for contaminated examples.
        clean_scores: 1-D array of peakedness scores for clean examples.
        output_path: Destination PDF path.
        title: Plot title.
        kind: ``"histogram"`` or ``"violin"``.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots()

    if kind == "violin":
        parts = ax.violinplot(
            [clean_scores, contaminated_scores],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )
        for idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(COLOR_PALETTE[idx])
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Clean", "Contaminated"])
    else:
        bins = np.linspace(0, 1, 30)
        ax.hist(clean_scores, bins=bins, alpha=0.6, label="Clean",
                color=COLOR_PALETTE[0], edgecolor="white")
        ax.hist(contaminated_scores, bins=bins, alpha=0.6, label="Contaminated",
                color=COLOR_PALETTE[1], edgecolor="white")
        ax.legend()

    ax.set_xlabel("Peakedness Score")
    ax.set_ylabel("Count" if kind == "histogram" else "Density")
    ax.set_title(title)

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Peakedness distribution ({kind}) saved to {output_path}")
    return output_path


# ── 11.6  Training Loss Curves ───────────────────────────────────────


def plot_training_loss_curves(
    loss_histories: Dict[str, List[float]],
    output_path: str = "training_loss.pdf",
    title: str = "Training Loss Curves",
) -> str:
    """Plot training loss over steps for each fine-tuning run.

    Args:
        loss_histories: ``{run_label: [loss_step0, loss_step1, ...]}``.
        output_path: Destination PDF path.
        title: Plot title.

    Returns:
        The *output_path* written.
    """
    _ensure_dir(output_path)
    fig, ax = plt.subplots()

    for idx, (label, losses) in enumerate(sorted(loss_histories.items())):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        steps = list(range(len(losses)))
        ax.plot(steps, losses, label=label, color=color)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    logger.info(f"Training loss curves saved to {output_path}")
    return output_path
