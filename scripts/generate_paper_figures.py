#!/usr/bin/env python
"""Generate publication-quality figures and metrics for the paper.

Reads all detection/baseline results across all experimental conditions
and produces:
  - Heatmaps: CDD accuracy by model size × contamination level, per ft method
  - Line plots: CDD accuracy vs contamination level, grouped by model size
  - Line plots: CDD accuracy vs model size, grouped by contamination level
  - Bar chart: comparison across ft methods for 410M contam=10
  - Loss curves per ft method
  - Peakedness distribution histograms
  - Summary table with confidence intervals (CSV + LaTeX)

Usage:
    python scripts/generate_paper_figures.py [--output_dir outputs/paper_figures]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from contamination_detection.detection.edit_distance import compute_peakedness

# ── Configuration ────────────────────────────────────────────────────
BASE_DIR = os.path.expanduser("~/final_proj/outputs/gpu_full_run") if os.path.exists(os.path.expanduser("~/final_proj")) else "outputs/gpu_full_run"
MODELS = ["70M", "160M", "410M"]
CONTAM_LEVELS = [0, 1, 5, 10]

# Map of ft_tag -> (detection_prefix, loss_prefix, display_name)
FT_METHODS = {
    "lora8_3ep":     ("",           "",           "LoRA r=8, 3ep"),
    "lora256_3ep":   ("lora256/",   "lora256/",   "LoRA r=256, 3ep"),
    "full_3ep":      ("full/",      "full/",      "Full FT, 3ep"),
    "lora8_20ep":    ("lora8_ep20/","lora8_ep20/","LoRA r=8, 20ep"),
    "lora256_20ep":  ("lora256_ep20/","lora256_ep20/","LoRA r=256, 20ep"),
    "full_20ep":     ("full_ep20/", "full_ep20/", "Full FT, 20ep"),
}

ALPHA = 0.05
SEED = 42


def load_detection(ft_tag, model, ce):
    """Load detection results, return (peaks, distances, max_lengths) or None."""
    prefix = FT_METHODS[ft_tag][0]
    path = os.path.join(BASE_DIR, "detection", f"{prefix}{model}_contam{ce}.npz")
    if not os.path.exists(path):
        return None
    data = dict(np.load(path))
    peaks = data["peakedness"]
    distances = data.get("distances", None)
    max_lengths = data.get("max_lengths", None)
    return peaks, distances, max_lengths


def load_baselines(ft_tag, model, ce):
    """Load baseline results or None."""
    prefix = FT_METHODS[ft_tag][0]
    path = os.path.join(BASE_DIR, "baselines", f"{prefix}{model}_contam{ce}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def load_losses(ft_tag, model, ce):
    """Load loss history or None."""
    prefix = FT_METHODS[ft_tag][1]
    path = os.path.join(BASE_DIR, "loss_histories", f"{prefix}{model}_contam{ce}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def optimal_accuracy(peaks, labels):
    """Find best accuracy over threshold sweep."""
    best = 0
    for xi in np.linspace(0, 1, 500):
        acc = np.mean((peaks > xi).astype(int) == labels)
        if acc > best:
            best = acc
    return best


def bootstrap_accuracy(peaks, labels, n_boot=1000, seed=42):
    """Bootstrap 95% CI for optimal accuracy."""
    rng = np.random.RandomState(seed)
    accs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        accs.append(optimal_accuracy(peaks[idx], labels[idx]))
    lo, mid, hi = np.percentile(accs, [2.5, 50, 97.5])
    return mid, lo, hi


labels = np.array([1] * 100 + [0] * 100)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/paper_figures")
    parser.add_argument("--base_dir", default=None)
    args = parser.parse_args()

    global BASE_DIR
    if args.base_dir:
        BASE_DIR = args.base_dir

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    setup_style()

    # ── Collect all results ──────────────────────────────────────────
    results = {}  # (ft_tag, model, ce) -> accuracy
    results_ci = {}  # (ft_tag, model, ce) -> (mid, lo, hi)
    peak_data = {}  # (ft_tag, model, ce) -> (contam_peaks, clean_peaks)
    loss_data = {}  # (ft_tag, model, ce) -> [losses]

    available_fts = []
    for ft_tag in FT_METHODS:
        has_any = False
        for model in MODELS:
            for ce in CONTAM_LEVELS:
                det = load_detection(ft_tag, model, ce)
                if det is None:
                    continue
                has_any = True
                peaks = det[0]
                acc = optimal_accuracy(peaks, labels)
                results[(ft_tag, model, ce)] = acc

                mid, lo, hi = bootstrap_accuracy(peaks, labels, n_boot=500, seed=SEED)
                results_ci[(ft_tag, model, ce)] = (mid, lo, hi)

                peak_data[(ft_tag, model, ce)] = (peaks[:100], peaks[100:])

                losses = load_losses(ft_tag, model, ce)
                if losses:
                    loss_data[(ft_tag, model, ce)] = losses
        if has_any:
            available_fts.append(ft_tag)

    print(f"Loaded {len(results)} conditions across {len(available_fts)} ft methods")
    print(f"Available: {available_fts}")

    # ── Figure 1: Heatmaps per ft method ─────────────────────────────
    for ft_tag in available_fts:
        display = FT_METHODS[ft_tag][2]
        mat = np.full((len(MODELS), len(CONTAM_LEVELS)), np.nan)
        for i, m in enumerate(MODELS):
            for j, ce in enumerate(CONTAM_LEVELS):
                if (ft_tag, m, ce) in results:
                    mat[i, j] = results[(ft_tag, m, ce)]

        fig, ax = plt.subplots(figsize=(5, 3))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(CONTAM_LEVELS)))
        ax.set_xticklabels([str(c) for c in CONTAM_LEVELS])
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS)
        ax.set_xlabel("Contamination Level")
        ax.set_ylabel("Model Size")
        ax.set_title(f"CDD Accuracy — {display}")
        for i in range(len(MODELS)):
            for j in range(len(CONTAM_LEVELS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=10, color="black" if mat[i,j] > 0.6 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(out, f"heatmap_{ft_tag}.pdf"))
        plt.close(fig)
        print(f"  Saved heatmap_{ft_tag}.pdf")

    # ── Figure 2: Accuracy vs contamination level (one plot per model) ──
    for model in MODELS:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for ft_tag in available_fts:
            display = FT_METHODS[ft_tag][2]
            xs, ys = [], []
            for ce in CONTAM_LEVELS:
                if (ft_tag, model, ce) in results:
                    xs.append(ce)
                    ys.append(results[(ft_tag, model, ce)])
            if xs:
                ax.plot(xs, ys, "o-", label=display, markersize=5)
        ax.set_xlabel("Contamination Level")
        ax.set_ylabel("CDD Accuracy")
        ax.set_title(f"CDD Accuracy vs Contamination — {model}")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(os.path.join(out, f"acc_vs_contam_{model}.pdf"))
        plt.close(fig)
        print(f"  Saved acc_vs_contam_{model}.pdf")

    # ── Figure 3: Accuracy vs model size (one plot per contam level) ──
    model_params = {"70M": 70, "160M": 160, "410M": 410}
    for ce in CONTAM_LEVELS:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for ft_tag in available_fts:
            display = FT_METHODS[ft_tag][2]
            xs, ys = [], []
            for model in MODELS:
                if (ft_tag, model, ce) in results:
                    xs.append(model_params[model])
                    ys.append(results[(ft_tag, model, ce)])
            if xs:
                ax.plot(xs, ys, "o-", label=display, markersize=5)
        ax.set_xlabel("Model Parameters (M)")
        ax.set_ylabel("CDD Accuracy")
        ax.set_title(f"CDD Accuracy vs Model Size — contam={ce}")
        ax.set_ylim(0.4, 1.05)
        ax.set_xscale("log")
        ax.set_xticks([70, 160, 410])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
        ax.legend(loc="best", fontsize=7)
        fig.savefig(os.path.join(out, f"acc_vs_size_contam{ce}.pdf"))
        plt.close(fig)
        print(f"  Saved acc_vs_size_contam{ce}.pdf")

    # ── Figure 4: Bar chart comparing ft methods for 410M contam=10 ──
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ft_names = []
    ft_accs = []
    ft_errs = []
    for ft_tag in available_fts:
        key = (ft_tag, "410M", 10)
        if key in results_ci:
            mid, lo, hi = results_ci[key]
            ft_names.append(FT_METHODS[ft_tag][2])
            ft_accs.append(mid)
            ft_errs.append([mid - lo, hi - mid])
    if ft_names:
        x = range(len(ft_names))
        errs = np.array(ft_errs).T
        ax.bar(x, ft_accs, yerr=errs, capsize=4, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ft_names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("CDD Accuracy")
        ax.set_title("CDD Accuracy by Fine-Tuning Method (410M, contam=10)")
        ax.set_ylim(0.4, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        fig.savefig(os.path.join(out, "bar_ft_methods_410M_c10.pdf"))
        plt.close(fig)
        print("  Saved bar_ft_methods_410M_c10.pdf")

    # ── Figure 5: Loss curves for contam=10 across models ────────────
    for ft_tag in available_fts:
        display = FT_METHODS[ft_tag][2]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        has_data = False
        for model in MODELS:
            key = (ft_tag, model, 10)
            if key in loss_data:
                losses = loss_data[key]
                steps = [(i + 1) * 5 for i in range(len(losses))]
                ax.plot(steps, losses, label=model)
                has_data = True
        if has_data:
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
            ax.set_title(f"Training Loss (contam=10) — {display}")
            ax.legend()
            fig.savefig(os.path.join(out, f"loss_curves_{ft_tag}.pdf"))
            print(f"  Saved loss_curves_{ft_tag}.pdf")
        plt.close(fig)

    # ── Figure 6: Peakedness distributions for key conditions ────────
    key_conditions = [
        ("lora256_20ep", "410M", 0, "410M, LoRA256 20ep, contam=0"),
        ("lora256_20ep", "410M", 10, "410M, LoRA256 20ep, contam=10"),
        ("full_3ep", "410M", 0, "410M, Full FT 3ep, contam=0"),
        ("full_3ep", "410M", 10, "410M, Full FT 3ep, contam=10"),
    ]
    for ft_tag, model, ce, title in key_conditions:
        if (ft_tag, model, ce) not in peak_data:
            continue
        contam_p, clean_p = peak_data[(ft_tag, model, ce)]
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = np.linspace(0, 1, 30)
        ax.hist(contam_p, bins=bins, alpha=0.6, label="Contaminated", color="red", density=True)
        ax.hist(clean_p, bins=bins, alpha=0.6, label="Clean", color="blue", density=True)
        ax.set_xlabel("Peakedness")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        fname = f"peakedness_{ft_tag}_{model}_c{ce}.pdf"
        fig.savefig(os.path.join(out, fname))
        plt.close(fig)
        print(f"  Saved {fname}")

    # ── Summary table (CSV) ──────────────────────────────────────────
    csv_path = os.path.join(out, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("ft_method,model,contam_level,cdd_accuracy,ci_lo,ci_hi,peak_contam_mean,peak_clean_mean\n")
        for ft_tag in available_fts:
            for model in MODELS:
                for ce in CONTAM_LEVELS:
                    key = (ft_tag, model, ce)
                    if key not in results:
                        continue
                    acc = results[key]
                    mid, lo, hi = results_ci.get(key, (acc, acc, acc))
                    cp, ep = peak_data.get(key, (np.array([0]), np.array([0])))
                    f.write(f"{FT_METHODS[ft_tag][2]},{model},{ce},{acc:.4f},{lo:.4f},{hi:.4f},{cp.mean():.4f},{ep.mean():.4f}\n")
    print(f"\n  Saved {csv_path}")

    # ── Print summary to stdout ──────────────────────────────────────
    print("\n=== KEY RESULTS ===")
    for ft_tag in available_fts:
        display = FT_METHODS[ft_tag][2]
        print(f"\n{display}:")
        for model in MODELS:
            for ce in [0, 10]:
                key = (ft_tag, model, ce)
                if key in results_ci:
                    mid, lo, hi = results_ci[key]
                    print(f"  {model} c={ce:2d}: {mid:.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
