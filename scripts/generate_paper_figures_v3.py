#!/usr/bin/env python
"""Generate publication-quality figures for the paper (v3).

Fixes:
  - Fig 1: row labels visible on all panels
  - Fig 2: replaced bar chart with loss-vs-CDD-accuracy scatter (more informative)
  - Fig 3: vertical layout, no alpha overlap, cleaner histograms
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

FT_METHODS = {
    "lora8_3ep":     ("",            "",            "LoRA r=8, 3ep"),
    "lora256_3ep":   ("lora256/",    "lora256/",    "LoRA r=256, 3ep"),
    "full_3ep":      ("full/",       "full/",       "Full FT, 3ep"),
    "lora8_20ep":    ("lora8_ep20/", "lora8_ep20/", "LoRA r=8, 20ep"),
    "lora256_20ep":  ("lora256_ep20/","lora256_ep20/","LoRA r=256, 20ep"),
    "full_20ep":     ("full_ep20/",  "full_ep20/",  "Full FT, 20ep"),
}
FT_ORDER = ["lora8_3ep", "lora256_3ep", "full_3ep", "lora8_20ep", "lora256_20ep", "full_20ep"]
labels = np.array([1]*100 + [0]*100)


def load_det(ft, m, ce):
    p = os.path.join(BASE_DIR, "detection", f"{FT_METHODS[ft][0]}{m}_contam{ce}.npz")
    return np.load(p)["peakedness"] if os.path.exists(p) else None

def load_loss(ft, m, ce):
    p = os.path.join(BASE_DIR, "loss_histories", f"{FT_METHODS[ft][1]}{m}_contam{ce}.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def opt_acc(peaks):
    best = 0
    for xi in np.linspace(0, 1, 500):
        acc = np.mean((peaks > xi).astype(int) == labels)
        if acc > best: best = acc
    return best

def setup():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    })

def main():
    out = "outputs/paper_figures_v3"
    os.makedirs(out, exist_ok=True)
    setup()

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
        if has: available.append(ft)
    print(f"Available: {available}")

    # ── Figure 1: Main heatmap with clear row labels ──
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5), sharey=True,
                             gridspec_kw={"wspace": 0.08})
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
        ax.set_xlabel("Contamination level")
        ax.set_title(f"Pythia-{model}", fontweight="bold")

        # Row labels on every panel
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels([FT_METHODS[ft][2] for ft in available], fontsize=7.5)

        for i in range(len(available)):
            for j in range(len(CONTAM_LEVELS)):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    color = "white" if v < 0.65 else "black"
                    txt = f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7.5, color=color, fontweight="bold")

    fig.subplots_adjust(right=0.87)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="CDD Accuracy")
    fig.savefig(os.path.join(out, "fig1_main_heatmap.pdf"))
    plt.close(fig)
    print("  fig1_main_heatmap.pdf")

    # ── Figure 2: Training loss vs CDD accuracy scatter ──
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    colors_m = {"70M": "#e41a1c", "160M": "#ff7f00", "410M": "#377eb8"}
    marker_ep = {"3": "o", "20": "s"}

    for ft in available:
        ep_key = "20" if "20" in ft else "3"
        for m in MODELS:
            for ce in [1, 5, 10]:  # skip contam=0
                loss_hist = load_loss(ft, m, ce)
                if loss_hist is None or (ft, m, ce) not in results: continue
                ax.scatter(loss_hist[-1], results[(ft, m, ce)],
                          c=colors_m[m], marker=marker_ep[ep_key],
                          s=35, alpha=0.8, edgecolors="black", linewidths=0.4)

    for m in MODELS:
        ax.scatter([], [], c=colors_m[m], marker="o", s=35, label=m,
                  edgecolors="black", linewidths=0.4)
    ax.scatter([], [], c="gray", marker="o", s=35, label="3 epochs",
              edgecolors="black", linewidths=0.4)
    ax.scatter([], [], c="gray", marker="s", s=35, label="20 epochs",
              edgecolors="black", linewidths=0.4)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Final training loss")
    ax.set_ylabel("CDD accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=7, loc="center right")
    fig.savefig(os.path.join(out, "fig2_loss_vs_accuracy.pdf"))
    plt.close(fig)
    print("  fig2_loss_vs_accuracy.pdf")

    # ── Figure 3: Peakedness distributions (vertical, no overlap) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4.5), sharex=True)

    # Panel A: CDD fails
    p1 = load_det("lora8_3ep", "410M", 10)
    if p1 is not None:
        bins = np.linspace(-0.01, 1.01, 30)
        ax1.hist(p1[:100], bins=bins, color="#d73027", edgecolor="black",
                linewidth=0.5, label="Contaminated (n=100)")
        ax1.hist(p1[100:], bins=bins, color="#4575b4", edgecolor="black",
                linewidth=0.5, label="Clean (n=100)")
        ax1.set_ylabel("Count")
        ax1.set_title("(a) LoRA r=8, 3 epochs: CDD fails", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.set_xlim(-0.05, 1.05)

    # Panel B: CDD works
    for ft_try in ["full_20ep", "full_3ep", "lora256_20ep"]:
        p2 = load_det(ft_try, "410M", 10)
        if p2 is not None:
            ax2.hist(p2[:100], bins=bins, color="#d73027", edgecolor="black",
                    linewidth=0.5, label="Contaminated (n=100)")
            ax2.hist(p2[100:], bins=bins, color="#4575b4", edgecolor="black",
                    linewidth=0.5, label="Clean (n=100)")
            title_name = FT_METHODS[ft_try][2]
            ax2.set_title(f"(b) {title_name}: CDD works", fontsize=10)
            ax2.legend(fontsize=8)
            break
    ax2.set_xlabel("Peakedness")
    ax2.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_peakedness_contrast.pdf"))
    plt.close(fig)
    print("  fig3_peakedness_contrast.pdf")

if __name__ == "__main__":
    main()
