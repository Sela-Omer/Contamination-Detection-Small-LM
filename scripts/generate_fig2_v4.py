#!/usr/bin/env python
"""Figure 2 v4: grouped bar chart showing CDD accuracy by ft method, one group per model size.
Focused on contam=10 only. Clean and simple."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "outputs/gpu_full_run"
MODELS = ["70M", "160M", "410M"]
labels = np.array([1]*100 + [0]*100)

FT_INFO = {
    "lora8_3ep":    ("",            "LoRA r=8\n3ep"),
    "lora256_3ep":  ("lora256/",    "LoRA r=256\n3ep"),
    "full_3ep":     ("full/",       "Full FT\n3ep"),
    "lora8_20ep":   ("lora8_ep20/", "LoRA r=8\n20ep"),
    "lora256_20ep": ("lora256_ep20/","LoRA r=256\n20ep"),
}
FT_ORDER = ["lora8_3ep","lora256_3ep","full_3ep","lora8_20ep","lora256_20ep"]

def load_det(prefix, m, ce):
    p = os.path.join(BASE, "detection", f"{prefix}{m}_contam{ce}.npz")
    return np.load(p)["peakedness"] if os.path.exists(p) else None

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
os.makedirs(out, exist_ok=True)

# Collect data for contam=10
available = []
for ft in FT_ORDER:
    if load_det(FT_INFO[ft][0], "410M", 10) is not None:
        available.append(ft)

fig, ax = plt.subplots(figsize=(5.5, 3.2))

colors = {"70M": "#e41a1c", "160M": "#ff7f00", "410M": "#377eb8"}
n_ft = len(available)
n_models = len(MODELS)
width = 0.22
x = np.arange(n_ft)

for i, model in enumerate(MODELS):
    accs = []
    for ft in available:
        p = load_det(FT_INFO[ft][0], model, 10)
        accs.append(opt_acc(p) if p is not None else 0.5)
    offset = (i - (n_models-1)/2) * width
    bars = ax.bar(x + offset, accs, width, label=f"Pythia-{model}",
                  color=colors[model], edgecolor="black", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels([FT_INFO[ft][1] for ft in available], fontsize=8)
ax.set_ylabel("CDD Accuracy")
ax.set_ylim(0.4, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("CDD Accuracy at Contamination Level 10", fontweight="bold")

fig.savefig(os.path.join(out, "fig2_loss_vs_accuracy.pdf"))
plt.close(fig)
print("Saved fig2_loss_vs_accuracy.pdf")
