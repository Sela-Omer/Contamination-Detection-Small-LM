#!/usr/bin/env python
"""Aggregate results from parallel runs and produce evaluation + visualizations.

Run this AFTER all 12 conditions from launch_parallel.sh have completed.
Reads cached detection/baseline .npz files and produces:
  - Metrics (CSV, LaTeX, JSON) per condition
  - All PDF figures (ROC, heatmaps, accuracy plots, peakedness, scale analysis)
  - Scale analysis summary

Usage:
    cd ~/final_proj && conda run -n cdd python scripts/run_aggregate.py 2>&1 | tee outputs/gpu_full_run/aggregate.log
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from contamination_detection.utils import set_global_seed
from contamination_detection.data.loader import load_saved_dataset
from contamination_detection.data.formatter import format_prompts, format_training_texts
from contamination_detection.detection.classifier import find_optimal_threshold
from contamination_detection.evaluation.metrics import compute_metrics
from contamination_detection.evaluation.confidence import bootstrap_confidence_intervals
from contamination_detection.evaluation.significance import mcnemar_test
from contamination_detection.evaluation.exporter import export_csv, export_latex, export_json
from contamination_detection.visualization.plots import (
    setup_publication_style,
    plot_roc_curves,
    plot_accuracy_vs_model_size,
    plot_accuracy_vs_contamination_level,
    plot_performance_heatmap,
    plot_peakedness_distributions,
    plot_training_loss_curves,
)
from contamination_detection.analysis.scale_analysis import run_scale_analysis, plot_scale_analysis

SEED = 42
OUTPUT_DIR = os.path.expanduser("~/final_proj/outputs/gpu_full_run")
MODEL_SIZES = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
MODEL_SHORT = {"EleutherAI/pythia-70m": "70M", "EleutherAI/pythia-160m": "160M", "EleutherAI/pythia-410m": "410M"}
CONTAM_EPOCHS = [0, 1, 5, 10]
METHODS = ["CDD", "CDD_xi001", "Random", "Perplexity", "NGram"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aggregate")


def main():
    set_global_seed(SEED)
    setup_publication_style()

    data_dir = os.path.join(OUTPUT_DIR, "data")
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Load eval/contam prompts for labels
    splits_contam = load_saved_dataset(os.path.join(data_dir, "contamination"))
    splits_eval = load_saved_dataset(os.path.join(data_dir, "evaluation"))
    contam_prompts = format_prompts(splits_contam)
    eval_prompts = format_prompts(splits_eval)
    test_labels = np.array([1] * len(contam_prompts) + [0] * len(eval_prompts))

    model_size_labels = [MODEL_SHORT[m] for m in MODEL_SIZES]

    # ── Load all results ─────────────────────────────────────────────
    all_metrics = {}
    all_cis = {}
    all_peaks = {}
    all_baselines = {}

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"

            det_path = os.path.join(OUTPUT_DIR, "detection", f"{condition}.npz")
            base_path = os.path.join(OUTPUT_DIR, "baselines", f"{condition}.npz")

            if not os.path.exists(det_path) or not os.path.exists(base_path):
                log.warning(f"Missing results for {condition}, skipping")
                continue

            peaks = np.load(det_path)["peakedness"]
            base = dict(np.load(base_path))
            all_peaks[(model_name, ce)] = peaks
            all_baselines[(model_name, ce)] = base

            # Recompute peakedness at multiple alphas if raw distances available
            det_data = dict(np.load(det_path))
            if "distances" in det_data and "max_lengths" in det_data:
                distances = det_data["distances"]
                max_lengths = det_data["max_lengths"]
                for alpha_val in [0.05, 0.10, 0.20]:
                    recomputed = np.array([
                        sum(1 for d in distances[i] if d <= alpha_val * max_lengths[i]) / len(distances[i])
                        for i in range(len(distances))
                    ])
                    key = f"peaks_a{alpha_val:.2f}"
                    if (model_name, ce) not in all_peaks:
                        all_peaks[(model_name, ce)] = {}
                    # Store multi-alpha peaks for later analysis
                    det_data[key] = recomputed

            # CDD classification — optimal threshold (Youden)
            cdd_thresh = find_optimal_threshold(peaks, test_labels)
            cdd_preds = (peaks > cdd_thresh).astype(int)

            # CDD classification — paper's fixed threshold ξ=0.01
            XI_FIXED = 0.01
            cdd_fixed_preds = (peaks > XI_FIXED).astype(int)

            methods_data = {
                "CDD": (cdd_preds, peaks),
                "CDD_xi001": (cdd_fixed_preds, peaks),
                "Random": (base["random_preds"].astype(int), base["random_scores"]),
                "Perplexity": (base["ppl_preds"].astype(int), base["ppl_scores"]),
                "NGram": (base["ngram_preds"].astype(int), base["ngram_scores"]),
            }

            log.info(f"\n{'='*40}")
            log.info(f"{condition} (CDD optimal_xi={cdd_thresh:.4f}, fixed_xi={XI_FIXED})")

            export_dict = {}
            for method_name, (preds, scores) in methods_data.items():
                m = compute_metrics(test_labels, preds, scores)
                ci = bootstrap_confidence_intervals(test_labels, preds, scores, n_bootstrap=1000, seed=SEED)
                all_metrics[(model_name, ce, method_name)] = m
                all_cis[(model_name, ce, method_name)] = ci
                export_dict[f"{condition}_{method_name}"] = m
                log.info(f"  {method_name:12s}: Acc={m.accuracy:.3f} [{ci['accuracy'][0]:.3f},{ci['accuracy'][2]:.3f}] F1={m.f1:.3f} AUC={m.auc:.3f}")

            # Significance
            for baseline in ["Random", "Perplexity", "NGram"]:
                p = mcnemar_test(test_labels, cdd_preds, methods_data[baseline][0])
                log.info(f"  McNemar CDD vs {baseline}: p={p:.4f}")
                p_fixed = mcnemar_test(test_labels, cdd_fixed_preds, methods_data[baseline][0])
                log.info(f"  McNemar CDD_xi001 vs {baseline}: p={p_fixed:.4f}")

            export_csv(export_dict, os.path.join(eval_dir, f"{condition}_metrics.csv"))
            export_latex(export_dict, os.path.join(eval_dir, f"{condition}_metrics.tex"))
            export_json(export_dict, os.path.join(eval_dir, f"{condition}_metrics.json"))

    # ── Visualizations ───────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("Generating visualizations...")

    # Training loss curves
    loss_dir = os.path.join(OUTPUT_DIR, "loss_histories")
    if os.path.exists(loss_dir):
        loss_histories = {}
        for f in sorted(os.listdir(loss_dir)):
            if f.endswith(".json"):
                with open(os.path.join(loss_dir, f)) as fh:
                    loss_histories[f.replace(".json", "")] = json.load(fh)
        if loss_histories:
            plot_training_loss_curves(loss_histories, os.path.join(fig_dir, "training_loss.pdf"))

    # ROC curves
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            if (model_name, ce) not in all_peaks:
                continue
            condition = f"{short}_contam{ce}"
            peaks = all_peaks[(model_name, ce)]
            base = all_baselines[(model_name, ce)]
            y_true_d = {"CDD": test_labels, "Perplexity": test_labels, "NGram": test_labels}
            y_scores_d = {"CDD": peaks, "Perplexity": base["ppl_scores"], "NGram": base["ngram_scores"]}
            plot_roc_curves(y_true_d, y_scores_d, os.path.join(fig_dir, f"roc_{condition}.pdf"), f"ROC — {condition}")

    # Peakedness distributions
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            if (model_name, ce) not in all_peaks:
                continue
            peaks = all_peaks[(model_name, ce)]
            plot_peakedness_distributions(
                peaks[test_labels == 1], peaks[test_labels == 0],
                os.path.join(fig_dir, f"peakedness_{short}_contam{ce}.pdf"),
                f"Peakedness — {short} contam={ce}",
            )

    # Accuracy vs model size
    for ce in CONTAM_EPOCHS:
        acc_data, ci_lo, ci_hi = {}, {}, {}
        for method in METHODS:
            acc_data[method], ci_lo[method], ci_hi[method] = {}, {}, {}
            for model_name in MODEL_SIZES:
                short = MODEL_SHORT[model_name]
                if (model_name, ce, method) not in all_metrics:
                    continue
                m = all_metrics[(model_name, ce, method)]
                ci = all_cis[(model_name, ce, method)]
                acc_data[method][short] = m.accuracy
                ci_lo[method][short] = ci["accuracy"][0]
                ci_hi[method][short] = ci["accuracy"][2]
        plot_accuracy_vs_model_size(
            acc_data, ci_lo, ci_hi,
            os.path.join(fig_dir, f"acc_vs_size_contam{ce}.pdf"),
            f"Accuracy vs Model Size (contam={ce})", model_sizes=model_size_labels,
        )

    # Accuracy vs contamination level
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        acc_data = {}
        for method in METHODS:
            acc_data[method] = {}
            for ce in CONTAM_EPOCHS:
                if (model_name, ce, method) in all_metrics:
                    acc_data[method][ce] = all_metrics[(model_name, ce, method)].accuracy
        plot_accuracy_vs_contamination_level(
            acc_data, output_path=os.path.join(fig_dir, f"acc_vs_contam_{short}.pdf"),
            title=f"Accuracy vs Contamination ({short})", contamination_levels=CONTAM_EPOCHS,
        )

    # Heatmaps
    for method in ["CDD", "CDD_xi001", "Perplexity", "NGram"]:
        for metric_name in ["accuracy", "f1", "auc"]:
            vals = np.full((len(MODEL_SIZES), len(CONTAM_EPOCHS)), np.nan)
            for i, model_name in enumerate(MODEL_SIZES):
                for j, ce in enumerate(CONTAM_EPOCHS):
                    if (model_name, ce, method) in all_metrics:
                        vals[i, j] = getattr(all_metrics[(model_name, ce, method)], metric_name)
            plot_performance_heatmap(
                vals, row_labels=model_size_labels,
                col_labels=[str(c) for c in CONTAM_EPOCHS],
                output_path=os.path.join(fig_dir, f"heatmap_{metric_name}_{method}.pdf"),
                title=f"{metric_name.upper()} — {method}", metric_name=metric_name.upper(),
            )

    # Scale analysis
    scale_data = {}
    for method in METHODS:
        scale_data[method] = {}
        for ce in CONTAM_EPOCHS:
            scale_data[method][ce] = {}
            for model_name in MODEL_SIZES:
                short = MODEL_SHORT[model_name]
                if (model_name, ce, method) in all_metrics:
                    scale_data[method][ce][short] = all_metrics[(model_name, ce, method)].accuracy

    analysis = run_scale_analysis(scale_data)
    plot_scale_analysis(scale_data, os.path.join(fig_dir, "scale_analysis.pdf"))

    scale_summary = {
        "method_comparison": analysis.method_comparison,
        "threshold_effects": [
            {"smaller": e.smaller_model, "larger": e.larger_model,
             "change_pp": e.accuracy_change_pp, "method": e.method, "contam_level": e.contamination_level}
            for e in analysis.threshold_effects
        ],
        "regressions": {
            method: [{"slope": r.slope, "r_squared": r.r_squared, "contam_level": r.contamination_level}
                     for r in regs]
            for method, regs in analysis.regressions.items()
        },
    }
    with open(os.path.join(eval_dir, "scale_analysis.json"), "w") as f:
        json.dump(scale_summary, f, indent=2)

    # ── Final summary ────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("AGGREGATION COMPLETE")
    log.info("=" * 60)

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            if (model_name, ce, "CDD") not in all_metrics:
                continue
            log.info(f"\n  {short}_contam{ce}:")
            for method in METHODS:
                m = all_metrics[(model_name, ce, method)]
                log.info(f"    {method:12s}: Acc={m.accuracy:.3f} F1={m.f1:.3f} AUC={m.auc:.3f}")

    log.info(f"\nScale slopes: {analysis.method_comparison}")
    log.info(f"Threshold effects: {len(analysis.threshold_effects)}")
    for e in analysis.threshold_effects:
        log.info(f"  {e.method} contam={e.contamination_level}: {e.smaller_model}→{e.larger_model} {e.accuracy_change_pp:+.1f}pp")

    n_files = sum(len(files) for _, _, files in os.walk(OUTPUT_DIR))
    log.info(f"\nTotal output files: {n_files}")
    log.info(f"Figures: {fig_dir}")
    log.info(f"Evaluation: {eval_dir}")


if __name__ == "__main__":
    main()
