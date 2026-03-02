"""Results exporter for evaluation metrics.

Exports metric tables as CSV, LaTeX-formatted tables (ACL-style), and JSON.
"""

import csv
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np

from contamination_detection.evaluation.metrics import MetricsResult

logger = logging.getLogger("contamination_detection.evaluation.exporter")


def export_csv(
    results: Dict[str, MetricsResult],
    output_path: str,
) -> str:
    """Export metrics as a CSV file.

    Args:
        results: Dict mapping condition label → :class:`MetricsResult`.
        output_path: Path to write the CSV file.

    Returns:
        The output path written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fieldnames = ["condition", "accuracy", "precision", "recall", "f1", "auc"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition, m in sorted(results.items()):
            writer.writerow(
                {
                    "condition": condition,
                    "accuracy": f"{m.accuracy:.4f}",
                    "precision": f"{m.precision:.4f}",
                    "recall": f"{m.recall:.4f}",
                    "f1": f"{m.f1:.4f}",
                    "auc": f"{m.auc:.4f}",
                }
            )

    logger.info(f"CSV exported to {output_path} ({len(results)} conditions)")
    return output_path


def export_latex(
    results: Dict[str, MetricsResult],
    output_path: str,
    caption: str = "Detection performance metrics",
    label: str = "tab:metrics",
) -> str:
    """Export metrics as a LaTeX tabular table suitable for ACL reports.

    Args:
        results: Dict mapping condition label → :class:`MetricsResult`.
        output_path: Path to write the .tex file.
        caption: Table caption.
        label: LaTeX label for cross-referencing.

    Returns:
        The output path written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Condition & Accuracy & Precision & Recall & F1 & AUC \\",
        r"\midrule",
    ]

    for condition in sorted(results.keys()):
        m = results[condition]
        # Escape underscores in condition names for LaTeX
        safe_cond = condition.replace("_", r"\_")
        lines.append(
            f"{safe_cond} & {m.accuracy:.4f} & {m.precision:.4f} & "
            f"{m.recall:.4f} & {m.f1:.4f} & {m.auc:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"LaTeX table exported to {output_path}")
    return output_path


def export_json(
    results: Dict[str, MetricsResult],
    output_path: str,
) -> str:
    """Export metrics as a JSON file.

    Args:
        results: Dict mapping condition label → :class:`MetricsResult`.
        output_path: Path to write the JSON file.

    Returns:
        The output path written.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = {}
    for condition, m in sorted(results.items()):
        data[condition] = {
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "auc": m.auc,
            "confusion_matrix": m.confusion_matrix.tolist(),
        }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"JSON exported to {output_path} ({len(results)} conditions)")
    return output_path
