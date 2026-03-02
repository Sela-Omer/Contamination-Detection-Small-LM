#!/usr/bin/env python
"""Figure 2 v5: loss vs CDD accuracy scatter, colored by model only.
No shape distinction for epochs. Annotated regions instead."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "outputs/gpu_full_run"
MODELS = ["70M", "160M", "410M"]
labels = np.array([1]*100 + [0]*100)

FT_INFO = {
    "lora8_3ep":    ("",            ""),
    "lora256_3ep":  ("lora256/",    "lora256/"),
    "full_3ep":     ("full/",       "full/"),
    "lora8_20ep":   ("lora8_ep20/", "lora8_ep20/"),
    "lora256_20ep": ("lora256_ep20/","lora256_ep20/"),
}
FT_ORDER = list(FT_INFO.keys())

def load_det(prefix, m, ce):
    p = os.path.join(BASE, "detection", f"{prefix}{m}_contam{ce}.npz")
    return np.load(p)["peakedness"] if os.path.exists(p) else None

def load_loss(prefix, m, ce):
    p = os.path.join(BASE, "loss_histories", f"{prefix}{m}_contam{ce}.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

def opt_acc(peaks):
    best = 0
    for xi in np.linspace(0, 1, 500):
        acc = np.mean((peaks > xi).astype(int) == labels)
        if acc > best: best = acc
    return best

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 12, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

out = "outputs/paper_figures_v3"
fig, ax = plt.subplots(figsize=(4.5, 3.5))

colors = {"70M": "#e41a1c", "160M": "#ff7f00", "410M": "#377eb8"}

for ft in FT_ORDER:
    det_prefix = FT_INFO[ft][0]
    loss_prefix = FT_INFO[ft][1]
    for m in MODELS:
        for ce in [1, 5, 10]:
            p = load_det(det_prefix, m, ce)
            l = load_loss(loss_prefix, m, ce)
            if p is None or l is None: continue
            ax.scatter(l[-1], opt_acc(p), c=colors[m], s=40, alpha=0.75,
                      edgecolors="black", linewidths=0.4, zorder=3)

# Legend for model colors
for m in MODELS:
    ax.scatter([], [], c=colors[m], s=40, label=f"Pythia-{m}",
              edgecolors="black", linewidths=0.4)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# Annotate regions
# (removed text annotations - let the data speak for itself)

ax.set_xlabel("Final training loss")
ax.set_ylabel("CDD accuracy")
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=8, loc="center right")

fig.savefig(os.path.join(out, "fig2_loss_vs_accuracy.pdf"))
plt.close(fig)
print("Saved fig2_loss_vs_accuracy.pdf")
