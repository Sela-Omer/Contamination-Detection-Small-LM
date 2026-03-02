#!/usr/bin/env python
"""Generate publication-quality figures for the paper (v2).

Key figures:
  1. Main result: 3x1 grid of heatmaps (one per model size) showing CDD accuracy
     across ft_method x contamination_level. This is the "memorization threshold" figure.
  2. CDD vs baselines: grouped bar chart for 410M contam=10 across ft methods,
     showing CDD, perplexity, n-gram, random side by side.
  3. Peakedness separation: 2-panel figure showing peaked vs flat distributions.
  4. Training loss vs CDD accuracy scatter: does lower loss = higher CDD accuracy?
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE_DIR = "outputs/gpu_full_run"
MODELS = ["70M", "160M", "410M"]
CONTAM_LEVELS = [0, 1, 5, 10]
SEED = 42

FT_METHODS = {
    "lora8_3ep":     ("",            "",            "LoRA r=8\n3 epochs"),
    "lora256_3ep":   ("lora256/",    "lora256/",    "LoRA r=256\n3 epochs"),
    "full_3ep":      ("full/",       "full/",       "Full FT\n3 epochs"),
    "lora8_20ep":    ("lora8_ep20/", "lora8_ep20/", "LoRA r=8\n20 epochs"),
    "lora256_20ep":  ("lora256_ep20/","lora256_ep20/","LoRA r=256\n20 epochs"),
    "full_20ep":     ("full_ep20/",  "full_ep20/",  "Full FT\n20 epochs"),
}

FT_ORDER = ["lora8_3ep", "lora256_3ep", "full_3ep", "lora8_20ep", "lora256_20ep", "full_20ep"]

labels = np.array([1]*100 + [0]*100)


def load_det(ft_tag, model, ce):
    prefix = FT_METHODS[ft_tag][0]
    path = os.path.join(BASE_DIR, "detection", f"{prefix}{model}_contam{ce}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["peakedness"]


def load_base(ft_tag, model, ce):
    prefix = FT_METHODS[ft_tag][0]
    path = os.path.join(BASE_DIR, "baselines", f"{prefix}{model}_contam{ce}.npz")
    if not os.path.exists(path):
        return None
    return dict(np.load(path))


def load_loss(ft_tag, model, ce):
    prefix = FT_METHODS[ft_tag][1]
    path = os.path.join(BASE_DIR, "loss_histories", f"{prefix}{model}_contam{ce}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def opt_acc(peaks):
    best = 0
    for xi in np.linspace(0, 1, 500):
        acc = np.mean((peaks > xi).astype(int) == labels)
        if acc > best:
            best = acc
    return best


def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def main():
    out = "outputs/paper_figures_v2"
    os.makedirs(out, exist_ok=True)
    setup()

    # Collect all results
    results = {}
    available = []
    for ft in FT_ORDER:
        has = False
        for m in MODELS:
            for ce in CONTAM_LEVELS:
                p = load_det(ft, m, ce)
                if p is not None:
                    results[(ft, m, ce)] = opt_acc(p)
                    has = True
        if has:
            available.append(ft)

    print(f"Available: {available}")

    # ── Figure 1: Main heatmap (ft_method x contam_level, one per model) ──
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), sharey=True)
    cmap = LinearSegmentedColormap.from_list("cdd", ["#d73027", "#fee08b", "#1a9850"])

    for col, model in enumerate(MODELS):
        ax = axes[col]
        mat = np.full((len(available), len(CONTAM_LEVELS)), np.nan)
        for i, ft in enumerate(available):
            for j, ce in enumerate(CONTAM_LEVELS):
                if (ft, model, ce) in results:
                    mat[i, j] = results[(ft, model, ce)]

        im = ax.imshow(mat, cmap=cmap, vmin=0.45, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(CONTAM_LEVELS)))
        ax.set_xticklabels([str(c) for c in CONTAM_LEVELS])
        ax.set_xlabel("Contam. level")
        ax.set_title(f"Pythia-{model}", fontsize=11, fontweight="bold")

        if col == 0:
            ax.set_yticks(range(len(available)))
            ax.set_yticklabels([FT_METHODS[ft][2] for ft in available], fontsize=8)
        else:
            ax.set_yticks([])

        for i in range(len(available)):
            for j in range(len(CONTAM_LEVELS)):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    color = "white" if v < 0.6 else "black"
                    ax.text(j, i, f".{int(v*100):02d}" if v < 1 else "1.0",
                            ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="CDD Accuracy")
    fig.savefig(os.path.join(out, "fig1_main_heatmap.pdf"))
    plt.close(fig)
    print("  fig1_main_heatmap.pdf")

    # ── Figure 2: CDD vs baselines for 410M contam=10 ──
    fig, ax = plt.subplots(figsize=(7, 3))
    methods_to_show = [ft for ft in available if (ft, "410M", 10) in results]
    x = np.arange(len(methods_to_show))
    width = 0.2

    cdd_accs, ppl_accs, ngram_accs, rand_accs = [], [], [], []
    for ft in methods_to_show:
        cdd_accs.append(results.get((ft, "410M", 10), 0.5))
        base = load_base(ft, "410M", 10)
        if base is not None:
            ppl_accs.append(np.mean(base["ppl_preds"].astype(int) == labels))
            ngram_accs.append(np.mean(base["ngram_preds"].astype(int) == labels))
            rand_accs.append(np.mean(base["random_preds"].astype(int) == labels))
        else:
            ppl_accs.append(0.5)
            ngram_accs.append(0.5)
            rand_accs.append(0.5)

    ax.bar(x - 1.5*width, cdd_accs, width, label="CDD", color="#2166ac")
    ax.bar(x - 0.5*width, ppl_accs, width, label="Perplexity", color="#92c5de")
    ax.bar(x + 0.5*width, ngram_accs, width, label="N-gram", color="#f4a582")
    ax.bar(x + 1.5*width, rand_accs, width, label="Random", color="#d6d6d6")

    ax.set_xticks(x)
    ax.set_xticklabels([FT_METHODS[ft][2].replace("\n", ", ") for ft in methods_to_show],
                       rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Detection Methods on Pythia-410M, contam=10", fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(out, "fig2_cdd_vs_baselines.pdf"))
    plt.close(fig)
    print("  fig2_cdd_vs_baselines.pdf")

    # ── Figure 3: Peakedness distributions (2 panels) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
    bins = np.linspace(0, 1, 25)

    # Panel A: CDD fails (lora8_3ep, 410M, contam=10)
    p = load_det("lora8_3ep", "410M", 10)
    if p is not None:
        ax1.hist(p[:100], bins=bins, alpha=0.7, label="Contaminated", color="#d73027", density=True)
        ax1.hist(p[100:], bins=bins, alpha=0.7, label="Clean", color="#4575b4", density=True)
    ax1.set_xlabel("Peakedness")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) LoRA r=8, 3ep: CDD fails", fontsize=10)
    ax1.legend(fontsize=7)

    # Panel B: CDD works (best available for 410M contam=10)
    for ft_try in ["full_20ep", "full_3ep", "lora256_20ep"]:
        p2 = load_det(ft_try, "410M", 10)
        if p2 is not None:
            ax2.hist(p2[:100], bins=bins, alpha=0.7, label="Contaminated", color="#d73027", density=True)
            ax2.hist(p2[100:], bins=bins, alpha=0.7, label="Clean", color="#4575b4", density=True)
            ax2.set_title(f"(b) {FT_METHODS[ft_try][2].replace(chr(10),', ')}: CDD works", fontsize=10)
            ax2.legend(fontsize=7)
            break
    ax2.set_xlabel("Peakedness")

    fig.savefig(os.path.join(out, "fig3_peakedness_contrast.pdf"))
    plt.close(fig)
    print("  fig3_peakedness_contrast.pdf")

    # ── Figure 4: Training loss vs CDD accuracy scatter ──
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {"70M": "#d73027", "160M": "#fc8d59", "410M": "#4575b4"}
    markers = {"3": "o", "20": "s"}

    for ft in available:
        for m in MODELS:
            for ce in CONTAM_LEVELS:
                if ce == 0:
                    continue
                loss_hist = load_loss(ft, m, ce)
                if loss_hist is None or (ft, m, ce) not in results:
                    continue
                final_loss = loss_hist[-1]
                acc = results[(ft, m, ce)]
                ep = "20" if "20" in ft else "3"
                ax.scatter(final_loss, acc, c=colors[m], marker=markers[ep],
                          s=30, alpha=0.7, edgecolors="black", linewidths=0.3)

    # Legend
    for m in MODELS:
        ax.scatter([], [], c=colors[m], marker="o", s=30, label=m, edgecolors="black", linewidths=0.3)
    ax.scatter([], [], c="gray", marker="o", s=30, label="3 epochs", edgecolors="black", linewidths=0.3)
    ax.scatter([], [], c="gray", marker="s", s=30, label="20 epochs", edgecolors="black", linewidths=0.3)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Final Training Loss")
    ax.set_ylabel("CDD Accuracy")
    ax.set_title("CDD Accuracy vs. Training Loss", fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0.4, 1.05)
    fig.savefig(os.path.join(out, "fig4_loss_vs_accuracy.pdf"))
    plt.close(fig)
    print("  fig4_loss_vs_accuracy.pdf")

    # ── Summary CSV ──
    csv_path = os.path.join(out, "results_summary.csv")
    with open(csv_path, "w") as f:
        f.write("ft_method,model,contam,cdd_acc,final_loss\n")
        for ft in available:
            for m in MODELS:
                for ce in CONTAM_LEVELS:
                    acc = results.get((ft, m, ce), None)
                    loss_hist = load_loss(ft, m, ce)
                    fl = loss_hist[-1] if loss_hist else ""
                    if acc is not None:
                        f.write(f"{FT_METHODS[ft][2].replace(chr(10),' ')},{m},{ce},{acc:.4f},{fl}\n")
    print(f"  {csv_path}")


if __name__ == "__main__":
    main()
