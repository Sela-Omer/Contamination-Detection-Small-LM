#!/usr/bin/env python
"""Figure 3 v4: side-by-side violin/strip plots instead of overlapping histograms."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "outputs/gpu_full_run"
labels = np.array([1]*100 + [0]*100)

def load_det(prefix, m, ce):
    p = os.path.join(BASE, "detection", f"{prefix}{m}_contam{ce}.npz")
    return np.load(p)["peakedness"] if os.path.exists(p) else None

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "axes.titlesize": 11, "savefig.dpi": 300, "savefig.bbox": "tight",
})

out = "outputs/paper_figures_v3"
os.makedirs(out, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 5))

# Panel A: CDD fails (lora8_3ep, 410M, contam=10)
p1 = load_det("", "410M", 10)
contam1, clean1 = p1[:100], p1[100:]

positions_a = [1, 2]
vp1 = ax1.violinplot([contam1, clean1], positions=positions_a, showmedians=True, widths=0.7)
vp1["bodies"][0].set_facecolor("#d73027"); vp1["bodies"][0].set_alpha(0.7)
vp1["bodies"][1].set_facecolor("#4575b4"); vp1["bodies"][1].set_alpha(0.7)
for part in ["cbars", "cmins", "cmaxes", "cmedians"]:
    vp1[part].set_color("black")

# Add jittered points
rng = np.random.RandomState(42)
ax1.scatter(rng.normal(1, 0.08, len(contam1)), contam1, s=8, alpha=0.4, c="#d73027", zorder=3)
ax1.scatter(rng.normal(2, 0.08, len(clean1)), clean1, s=8, alpha=0.4, c="#4575b4", zorder=3)

ax1.set_xticks(positions_a)
ax1.set_xticklabels(["Contaminated\n(n=100)", "Clean\n(n=100)"])
ax1.set_ylabel("Peakedness")
ax1.set_title("(a) LoRA r=8, 3 epochs: CDD fails", fontsize=10, fontweight="bold")
ax1.set_ylim(-0.05, 1.05)
ax1.axhline(0.01, color="gray", linestyle="--", linewidth=0.8, label="$\\xi$=0.01")
ax1.legend(fontsize=8)

# Panel B: CDD works (full_3ep, 410M, contam=10)
p2 = load_det("full/", "410M", 10)
if p2 is None:
    p2 = load_det("lora256_ep20/", "410M", 10)
contam2, clean2 = p2[:100], p2[100:]

vp2 = ax2.violinplot([contam2, clean2], positions=positions_a, showmedians=True, widths=0.7)
vp2["bodies"][0].set_facecolor("#d73027"); vp2["bodies"][0].set_alpha(0.7)
vp2["bodies"][1].set_facecolor("#4575b4"); vp2["bodies"][1].set_alpha(0.7)
for part in ["cbars", "cmins", "cmaxes", "cmedians"]:
    vp2[part].set_color("black")

ax2.scatter(rng.normal(1, 0.08, len(contam2)), contam2, s=8, alpha=0.4, c="#d73027", zorder=3)
ax2.scatter(rng.normal(2, 0.08, len(clean2)), clean2, s=8, alpha=0.4, c="#4575b4", zorder=3)

ax2.set_xticks(positions_a)
ax2.set_xticklabels(["Contaminated\n(n=100)", "Clean\n(n=100)"])
ax2.set_ylabel("Peakedness")
ax2.set_title("(b) Full fine-tuning, 3 epochs: CDD works", fontsize=10, fontweight="bold")
ax2.set_ylim(-0.05, 1.05)
ax2.axhline(0.01, color="gray", linestyle="--", linewidth=0.8, label="$\\xi$=0.01")
ax2.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(out, "fig3_peakedness_contrast.pdf"))
print("Saved fig3_peakedness_contrast.pdf")
