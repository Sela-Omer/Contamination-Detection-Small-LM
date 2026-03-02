#!/usr/bin/env python
"""Full experiment run on GPU server.

Runs the complete CDD pipeline on all Pythia model sizes (70M, 160M, 410M)
with all contamination levels (0, 1, 5, 10) using QASC dataset.

Designed for a multi-GPU server with A100 40GB GPUs.
Uses more data, more samples, and longer generation than the local tiny run.
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from contamination_detection.utils import set_global_seed, record_dependency_versions, timer
from contamination_detection.data.loader import load_qa_dataset, save_dataset, load_saved_dataset
from contamination_detection.data.splitter import create_splits
from contamination_detection.data.formatter import format_prompts
from contamination_detection.data.contamination import create_contaminated_training_set
from contamination_detection.training.model_loader import load_pythia_with_lora, save_checkpoint, load_checkpoint
from contamination_detection.training.trainer import fine_tune
from contamination_detection.detection.sampler import sample_outputs
from contamination_detection.detection.edit_distance import compute_edit_distances
from contamination_detection.detection.peakedness import compute_peakedness, compute_peakedness_multi
from contamination_detection.detection.classifier import classify, find_optimal_threshold
from contamination_detection.baselines.random_baseline import classify_batch as random_classify_batch
from contamination_detection.baselines.perplexity_detector import compute_perplexity_batch, find_optimal_threshold as ppl_find_threshold
from contamination_detection.baselines.ngram_detector import NGramOverlapDetector, find_optimal_threshold as ngram_find_threshold
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
from contamination_detection.config import LoRAConfig, TrainingConfig, SamplingConfig

# ── Configuration ────────────────────────────────────────────────────
SEED = 42
MODEL_SIZES = ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
MODEL_SHORT = {"EleutherAI/pythia-70m": "70M", "EleutherAI/pythia-160m": "160M", "EleutherAI/pythia-410m": "410M"}
N_EXAMPLES = 500          # QASC examples to use
TRAIN_RATIO = 0.6         # 300 train
CONTAM_RATIO = 0.2        # 100 contamination
EVAL_RATIO = 0.2          # 100 eval
CONTAM_EPOCHS = [0, 1, 5, 10]
N_SAMPLES = 20            # samples per prompt for CDD
MAX_NEW_TOKENS = 100
LORA_R = 8
TRAIN_EPOCHS = 3
TRAIN_BATCH = 8
TRAIN_LR = 2e-4
OUTPUT_DIR = os.path.expanduser("~/final_proj/outputs/gpu_full_run")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gpu_full_run")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_global_seed(SEED)
    setup_publication_style()

    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f} GB)")

    versions = record_dependency_versions()
    log.info(f"Dependency versions: {json.dumps(versions, indent=2)}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Data Preparation
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 1: Data Preparation")
    log.info("=" * 60)

    data_dir = os.path.join(OUTPUT_DIR, "data")
    eval_cache = os.path.join(data_dir, "evaluation")

    if os.path.exists(eval_cache):
        log.info("Data already prepared, loading from cache...")
        splits_train = load_saved_dataset(os.path.join(data_dir, "train"))
        splits_contam = load_saved_dataset(os.path.join(data_dir, "contamination"))
        splits_eval = load_saved_dataset(os.path.join(data_dir, "evaluation"))
    else:
        with timer("Load QASC dataset", log):
            full_ds = load_qa_dataset("QASC")
            ds = full_ds.shuffle(seed=SEED).select(range(N_EXAMPLES))
            log.info(f"Using {len(ds)} examples from QASC")

        with timer("Create splits", log):
            splits = create_splits(ds, TRAIN_RATIO, CONTAM_RATIO, EVAL_RATIO, seed=SEED)
            splits_train = splits.train
            splits_contam = splits.contamination
            splits_eval = splits.evaluation

        save_dataset(splits_train, os.path.join(data_dir, "train"))
        save_dataset(splits_contam, os.path.join(data_dir, "contamination"))
        save_dataset(splits_eval, os.path.join(data_dir, "evaluation"))

    eval_prompts = format_prompts(splits_eval)
    contam_prompts = format_prompts(splits_contam)
    log.info(f"Train: {len(splits_train)}, Contam: {len(splits_contam)}, Eval: {len(splits_eval)}")

    # Create contaminated training sets
    contam_train_sets = {}
    for ce in CONTAM_EPOCHS:
        cache_path = os.path.join(data_dir, f"train_contam_{ce}")
        if os.path.exists(cache_path):
            contam_train_sets[ce] = load_saved_dataset(cache_path)
        else:
            ct = create_contaminated_training_set(splits_train, splits_contam, ce, seed=SEED)
            save_dataset(ct, cache_path)
            contam_train_sets[ce] = ct
        log.info(f"Contamination epochs={ce}: {len(contam_train_sets[ce])} training examples")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Fine-Tuning (all models × all contamination levels)
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 2: Fine-Tuning")
    log.info("=" * 60)

    lora_cfg = LoRAConfig(r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.05, target_modules=["query_key_value"])
    training_cfg = TrainingConfig(
        learning_rate=TRAIN_LR, batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=2, num_epochs=TRAIN_EPOCHS,
        warmup_ratio=0.1, seed=SEED, logging_steps=5,
    )

    all_loss_histories = {}
    model_dirs = {}

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"
            model_dir = os.path.join(OUTPUT_DIR, "models", condition)
            model_dirs[(model_name, ce)] = model_dir

            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                log.info(f"[{condition}] Checkpoint exists, skipping")
                continue

            log.info(f"[{condition}] Fine-tuning...")
            with timer(f"Fine-tune {condition}", log):
                model, tokenizer = load_pythia_with_lora(model_name, lora_cfg)
                train_texts = format_prompts(contam_train_sets[ce])

                result = fine_tune(
                    model=model, tokenizer=tokenizer,
                    train_texts=train_texts, training_config=training_cfg,
                    lora_config=lora_cfg, output_dir=model_dir, max_length=256,
                )

                all_loss_histories[condition] = result.loss_history
                log.info(f"[{condition}] Loss: {result.initial_loss:.4f} → {result.final_loss:.4f}")
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Plot training loss
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    if all_loss_histories:
        plot_training_loss_curves(all_loss_histories, os.path.join(fig_dir, "training_loss.pdf"))

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: CDD Detection
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 3: CDD Detection")
    log.info("=" * 60)

    sampling_cfg = SamplingConfig(
        n_samples=N_SAMPLES, temperature=1.0, top_k=50, top_p=0.95,
        max_new_tokens=MAX_NEW_TOKENS, seed=SEED,
    )

    test_prompts = contam_prompts + eval_prompts
    test_labels = np.array([1] * len(contam_prompts) + [0] * len(eval_prompts))
    all_detection = {}

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"
            cache_path = os.path.join(OUTPUT_DIR, "detection", f"{condition}.npz")

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                all_detection[(model_name, ce)] = {"peakedness": data["peakedness"], "labels": test_labels}
                log.info(f"[{condition}] Detection loaded from cache")
                continue

            model_dir = model_dirs[(model_name, ce)]
            log.info(f"[{condition}] Running CDD detection...")

            with timer(f"CDD {condition}", log):
                model, tokenizer, _ = load_checkpoint(model_dir)
                model.eval()

                peaks = []
                for idx, prompt in enumerate(test_prompts):
                    sr = sample_outputs(
                        prompt=prompt, model=model, tokenizer=tokenizer,
                        n_samples=N_SAMPLES, config=sampling_cfg,
                        seed=SEED + idx,
                    )
                    dist = compute_edit_distances(sr.outputs)
                    peak = compute_peakedness(dist.normalized_matrix, alpha=0.1)
                    peaks.append(peak)

                    if idx % 20 == 0:
                        log.info(f"  [{condition}] {idx+1}/{len(test_prompts)} prompts done")

                peaks_arr = np.array(peaks)
                os.makedirs(os.path.join(OUTPUT_DIR, "detection"), exist_ok=True)
                np.savez(cache_path, peakedness=peaks_arr)
                all_detection[(model_name, ce)] = {"peakedness": peaks_arr, "labels": test_labels}

                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Baselines
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 4: Baselines")
    log.info("=" * 60)

    all_baselines = {}
    n_test = len(test_labels)

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"
            cache_path = os.path.join(OUTPUT_DIR, "baselines", f"{condition}.npz")

            if os.path.exists(cache_path):
                data = np.load(cache_path)
                all_baselines[(model_name, ce)] = dict(data)
                log.info(f"[{condition}] Baselines loaded from cache")
                continue

            model_dir = model_dirs[(model_name, ce)]
            log.info(f"[{condition}] Computing baselines...")

            # Random
            rr = random_classify_batch(n_test, seed=SEED)
            random_preds = np.array([r.is_contaminated for r in rr], dtype=int)
            random_scores = np.full(n_test, 0.5)

            # Perplexity
            model, tokenizer, _ = load_checkpoint(model_dir)
            model.eval()
            ppls = compute_perplexity_batch(model, tokenizer, test_prompts, max_length=256)
            ppls_arr = np.array(ppls)
            ppl_thresh = ppl_find_threshold(ppls_arr, test_labels)
            ppl_preds = (ppls_arr < ppl_thresh).astype(int)
            ppl_scores = 1.0 / (1.0 + ppls_arr)
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # N-gram
            train_texts = format_prompts(contam_train_sets[ce])
            ngram_det = NGramOverlapDetector(train_texts, n=3)
            overlaps = np.array(ngram_det.compute_overlap_batch(test_prompts))
            ngram_thresh = ngram_find_threshold(overlaps, test_labels)
            ngram_preds = (overlaps > ngram_thresh).astype(int)

            baseline_data = {
                "random_preds": random_preds, "random_scores": random_scores,
                "ppl_preds": ppl_preds, "ppl_scores": ppl_scores,
                "ngram_preds": ngram_preds, "ngram_scores": overlaps,
            }
            os.makedirs(os.path.join(OUTPUT_DIR, "baselines"), exist_ok=True)
            np.savez(cache_path, **baseline_data)
            all_baselines[(model_name, ce)] = baseline_data
            log.info(f"[{condition}] PPL thresh={ppl_thresh:.2f}, NGram thresh={ngram_thresh:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: Evaluation
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 5: Evaluation")
    log.info("=" * 60)

    eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    all_metrics = {}  # (model_name, ce, method) → MetricsResult
    all_cis = {}

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"
            det = all_detection[(model_name, ce)]
            base = all_baselines[(model_name, ce)]
            labels = det["labels"]
            peaks = det["peakedness"]

            cdd_thresh = find_optimal_threshold(peaks, labels)
            cdd_preds = (peaks > cdd_thresh).astype(int)

            methods = {
                "CDD": (cdd_preds, peaks),
                "Random": (base["random_preds"], base["random_scores"]),
                "Perplexity": (base["ppl_preds"], base["ppl_scores"]),
                "NGram": (base["ngram_preds"], base["ngram_scores"]),
            }

            log.info(f"\n{'='*40}")
            log.info(f"{condition} (CDD thresh={cdd_thresh:.4f})")

            export_dict = {}
            for method_name, (preds, scores) in methods.items():
                m = compute_metrics(labels, preds, scores)
                ci = bootstrap_confidence_intervals(labels, preds, scores, n_bootstrap=1000, seed=SEED)
                all_metrics[(model_name, ce, method_name)] = m
                all_cis[(model_name, ce, method_name)] = ci
                export_dict[f"{condition}_{method_name}"] = m
                log.info(f"  {method_name:12s}: Acc={m.accuracy:.3f} [{ci['accuracy'][0]:.3f},{ci['accuracy'][2]:.3f}] F1={m.f1:.3f} AUC={m.auc:.3f}")

            export_csv(export_dict, os.path.join(eval_dir, f"{condition}_metrics.csv"))
            export_latex(export_dict, os.path.join(eval_dir, f"{condition}_metrics.tex"))
            export_json(export_dict, os.path.join(eval_dir, f"{condition}_metrics.json"))

    # ═══════════════════════════════════════════════════════════════
    # STAGE 6: Visualizations + Scale Analysis
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 6: Visualizations & Scale Analysis")
    log.info("=" * 60)

    model_size_labels = [MODEL_SHORT[m] for m in MODEL_SIZES]

    # ROC curves per condition
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            condition = f"{short}_contam{ce}"
            det = all_detection[(model_name, ce)]
            base = all_baselines[(model_name, ce)]
            labels = det["labels"]
            y_true_d = {"CDD": labels, "Perplexity": labels, "NGram": labels}
            y_scores_d = {"CDD": det["peakedness"], "Perplexity": base["ppl_scores"], "NGram": base["ngram_scores"]}
            plot_roc_curves(y_true_d, y_scores_d, os.path.join(fig_dir, f"roc_{condition}.pdf"), f"ROC — {condition}")

    # Peakedness distributions
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            det = all_detection[(model_name, ce)]
            labels = det["labels"]
            peaks = det["peakedness"]
            plot_peakedness_distributions(
                peaks[labels == 1], peaks[labels == 0],
                os.path.join(fig_dir, f"peakedness_{short}_contam{ce}.pdf"),
                f"Peakedness — {short} contam={ce}",
            )

    # Accuracy vs model size (one line per method, one plot per contam level)
    for ce in CONTAM_EPOCHS:
        acc_data = {}
        ci_lo_data = {}
        ci_hi_data = {}
        for method in ["CDD", "Random", "Perplexity", "NGram"]:
            acc_data[method] = {}
            ci_lo_data[method] = {}
            ci_hi_data[method] = {}
            for model_name in MODEL_SIZES:
                short = MODEL_SHORT[model_name]
                m = all_metrics[(model_name, ce, method)]
                ci = all_cis[(model_name, ce, method)]
                acc_data[method][short] = m.accuracy
                ci_lo_data[method][short] = ci["accuracy"][0]
                ci_hi_data[method][short] = ci["accuracy"][2]
        plot_accuracy_vs_model_size(
            acc_data, ci_lo_data, ci_hi_data,
            os.path.join(fig_dir, f"acc_vs_size_contam{ce}.pdf"),
            f"Accuracy vs Model Size (contam={ce})",
            model_sizes=model_size_labels,
        )

    # Accuracy vs contamination level (one line per method, one plot per model)
    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        acc_data = {}
        for method in ["CDD", "Random", "Perplexity", "NGram"]:
            acc_data[method] = {}
            for ce in CONTAM_EPOCHS:
                acc_data[method][ce] = all_metrics[(model_name, ce, method)].accuracy
        plot_accuracy_vs_contamination_level(
            acc_data, output_path=os.path.join(fig_dir, f"acc_vs_contam_{short}.pdf"),
            title=f"Accuracy vs Contamination ({short})",
            contamination_levels=CONTAM_EPOCHS,
        )

    # Heatmaps
    for method in ["CDD", "Perplexity", "NGram"]:
        for metric_name in ["accuracy", "f1", "auc"]:
            vals = np.zeros((len(MODEL_SIZES), len(CONTAM_EPOCHS)))
            for i, model_name in enumerate(MODEL_SIZES):
                for j, ce in enumerate(CONTAM_EPOCHS):
                    vals[i, j] = getattr(all_metrics[(model_name, ce, method)], metric_name)
            plot_performance_heatmap(
                vals, row_labels=model_size_labels,
                col_labels=[str(c) for c in CONTAM_EPOCHS],
                output_path=os.path.join(fig_dir, f"heatmap_{metric_name}_{method}.pdf"),
                title=f"{metric_name.upper()} — {method}",
                metric_name=metric_name.upper(),
            )

    # Scale analysis
    scale_data = {}
    for method in ["CDD", "Random", "Perplexity", "NGram"]:
        scale_data[method] = {}
        for ce in CONTAM_EPOCHS:
            scale_data[method][ce] = {}
            for model_name in MODEL_SIZES:
                short = MODEL_SHORT[model_name]
                scale_data[method][ce][short] = all_metrics[(model_name, ce, method)].accuracy

    analysis = run_scale_analysis(scale_data)
    plot_scale_analysis(scale_data, os.path.join(fig_dir, "scale_analysis.pdf"))

    # Save scale analysis results
    scale_summary = {
        "method_comparison": analysis.method_comparison,
        "threshold_effects": [
            {"smaller": e.smaller_model, "larger": e.larger_model,
             "change_pp": e.accuracy_change_pp, "method": e.method,
             "contam_level": e.contamination_level}
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

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("RUN COMPLETE")
    log.info("=" * 60)

    for model_name in MODEL_SIZES:
        short = MODEL_SHORT[model_name]
        for ce in CONTAM_EPOCHS:
            log.info(f"\n  {short}_contam{ce}:")
            for method in ["CDD", "Random", "Perplexity", "NGram"]:
                m = all_metrics[(model_name, ce, method)]
                log.info(f"    {method:12s}: Acc={m.accuracy:.3f} F1={m.f1:.3f} AUC={m.auc:.3f}")

    log.info(f"\nScale analysis — average slopes: {analysis.method_comparison}")
    log.info(f"Threshold effects: {len(analysis.threshold_effects)}")
    for e in analysis.threshold_effects:
        log.info(f"  {e.method} contam={e.contamination_level}: {e.smaller_model}→{e.larger_model} {e.accuracy_change_pp:+.1f}pp")

    # Count output files
    n_files = sum(len(files) for _, _, files in os.walk(OUTPUT_DIR))
    log.info(f"\nTotal output files: {n_files}")
    log.info(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
