"""Visualization module for publication-quality figures."""

from contamination_detection.visualization.plots import (
    setup_publication_style,
    plot_roc_curves,
    plot_accuracy_vs_model_size,
    plot_accuracy_vs_contamination_level,
    plot_performance_heatmap,
    plot_peakedness_distributions,
    plot_training_loss_curves,
    COLOR_PALETTE,
    MODEL_SIZES,
    CONTAMINATION_LEVELS,
)

__all__ = [
    "setup_publication_style",
    "plot_roc_curves",
    "plot_accuracy_vs_model_size",
    "plot_accuracy_vs_contamination_level",
    "plot_performance_heatmap",
    "plot_peakedness_distributions",
    "plot_training_loss_curves",
    "COLOR_PALETTE",
    "MODEL_SIZES",
    "CONTAMINATION_LEVELS",
]
