#!/usr/bin/env python
"""Local end-to-end run on Mac with Pythia-70M and a tiny QASC subset.

This script runs the full pipeline manually (no Hydra) so we can see
every step and debug interactively. Uses:
  - Pythia-70M only
  - 50 QASC examples (30 train, 10 contamination, 10 eval)
  - Contamination epochs: [0, 1]
  - 5 samples per prompt (fast)
  - 20 max new tokens
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

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
    plot_accuracy_vs_contamination_level,
    plot_performance_heatmap,
    plot_peakedness_distributions,
    plot_training_loss_curves,
)
from contamination_detection.config import LoRAConfig, TrainingConfig, SamplingConfig

# ── Configuration ────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "EleutherAI/pythia-70m"
N_EXAMPLES = 50          # total QASC examples to use
TRAIN_RATIO = 0.6        # 30 examples
CONTAM_RATIO = 0.2       # 10 examples
EVAL_RATIO = 0.2         # 10 examples
CONTAM_EPOCHS = [0, 1]   # clean vs 1-epoch contamination
N_SAMPLES = 5            # samples per prompt for CDD
MAX_NEW_TOKENS = 20
LORA_R = 4
TRAIN_EPOCHS = 3
TRAIN_BATCH = 2
TRAIN_LR = 5e-4
OUTPUT_DIR = "outputs/local_tiny_run"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("local_run")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_global_seed(SEED)
    setup_publication_style()

    versions = record_dependency_versions()
    log.info(f"Dependency versions: {json.dumps(versions, indent=2)}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 1: Data Preparation
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 1: Data Preparation")
    log.info("=" * 60)

    with timer("Load QASC dataset", log):
        full_ds = load_qa_dataset("QASC")
        # Take a small subset
        ds = full_ds.shuffle(seed=SEED).select(range(N_EXAMPLES))
        log.info(f"Using {len(ds)} examples from QASC (out of {len(full_ds)})")

    with timer("Create splits", log):
        splits = create_splits(ds, TRAIN_RATIO, CONTAM_RATIO, EVAL_RATIO, seed=SEED)
        log.info(f"Train: {len(splits.train)}, Contam: {len(splits.contamination)}, Eval: {len(splits.evaluation)}")

    # Save splits
    data_dir = os.path.join(OUTPUT_DIR, "data")
    save_dataset(splits.train, os.path.join(data_dir, "train"))
    save_dataset(splits.contamination, os.path.join(data_dir, "contamination"))
    save_dataset(splits.evaluation, os.path.join(data_dir, "evaluation"))

    # Format prompts
    eval_prompts = format_prompts(splits.evaluation)
    train_prompts_clean = format_prompts(splits.train)
    contam_prompts = format_prompts(splits.contamination)
    log.info(f"Eval prompts: {len(eval_prompts)}")
    log.info(f"Sample eval prompt: {eval_prompts[0][:100]}...")

    # Create contaminated training sets
    contam_train_sets = {}
    for ce in CONTAM_EPOCHS:
        contam_train = create_contaminated_training_set(
            splits.train, splits.contamination, contamination_epochs=ce, seed=SEED
        )
        contam_train_sets[ce] = contam_train
        contam_texts = format_prompts(contam_train)
        log.info(f"Contamination epochs={ce}: {len(contam_train)} training examples")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 2: Fine-Tuning
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 2: Fine-Tuning (Pythia-70M)")
    log.info("=" * 60)

    lora_cfg = LoRAConfig(r=LORA_R, lora_alpha=LORA_R * 2, lora_dropout=0.0, target_modules=["query_key_value"])
    training_cfg = TrainingConfig(
        learning_rate=TRAIN_LR,
        batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=1,
        num_epochs=TRAIN_EPOCHS,
        warmup_ratio=0.0,
        seed=SEED,
        logging_steps=1,
    )

    all_loss_histories = {}
    model_dirs = {}

    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        model_dir = os.path.join(OUTPUT_DIR, "models", condition)
        model_dirs[ce] = model_dir

        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            log.info(f"[{condition}] Checkpoint exists, skipping training")
            continue

        log.info(f"[{condition}] Loading model...")
        with timer(f"Load {condition}", log):
            model, tokenizer = load_pythia_with_lora(MODEL_NAME, lora_cfg)

        train_texts = format_prompts(contam_train_sets[ce])
        log.info(f"[{condition}] Training on {len(train_texts)} examples for {TRAIN_EPOCHS} epochs")

        with timer(f"Fine-tune {condition}", log):
            result = fine_tune(
                model=model,
                tokenizer=tokenizer,
                train_texts=train_texts,
                training_config=training_cfg,
                lora_config=lora_cfg,
                output_dir=model_dir,
                max_length=128,
            )

        all_loss_histories[condition] = result.loss_history
        log.info(f"[{condition}] Loss: {result.initial_loss:.4f} → {result.final_loss:.4f} (sanity={result.sanity_passed})")
        del model, tokenizer

    # Plot training loss curves
    if all_loss_histories:
        fig_dir = os.path.join(OUTPUT_DIR, "figures")
        os.makedirs(fig_dir, exist_ok=True)
        plot_training_loss_curves(all_loss_histories, os.path.join(fig_dir, "training_loss.pdf"))
        log.info("Training loss curves saved")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 3: CDD Detection (Sampling + Edit Distance + Peakedness)
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 3: CDD Detection")
    log.info("=" * 60)

    sampling_cfg = SamplingConfig(
        n_samples=N_SAMPLES,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        max_new_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )

    # Ground truth: contamination set examples are "contaminated" (label=1),
    # eval set examples that are NOT in contamination set are "clean" (label=0).
    # For our eval set, we know which are contaminated because we control the splits.
    # The eval set is entirely clean (no overlap with contamination set).
    # So we need to evaluate on BOTH contamination examples AND eval examples.
    # Contamination examples → label=1, Eval examples → label=0.

    all_detection_results = {}  # condition → {peakedness_scores, labels, ...}

    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        model_dir = model_dirs[ce]

        log.info(f"[{condition}] Loading checkpoint...")
        with timer(f"Load checkpoint {condition}", log):
            model, tokenizer, meta = load_checkpoint(model_dir)
        model.eval()

        # Evaluate on contamination examples (should be detected as contaminated)
        # and eval examples (should be detected as clean)
        test_prompts = contam_prompts + eval_prompts
        test_labels = [1] * len(contam_prompts) + [0] * len(eval_prompts)

        peakedness_scores = []
        log.info(f"[{condition}] Sampling {N_SAMPLES} outputs for {len(test_prompts)} prompts...")

        for idx, prompt in enumerate(test_prompts):
            with timer(f"Prompt {idx+1}/{len(test_prompts)}", log):
                sr = sample_outputs(
                    prompt=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    n_samples=N_SAMPLES,
                    config=sampling_cfg,
                    seed=SEED + idx,
                )
                dist = compute_edit_distances(sr.outputs)
                peak = compute_peakedness(dist.normalized_matrix, alpha=0.1)
                peakedness_scores.append(peak)

            if idx < 3:
                log.info(f"  Prompt: {prompt[:60]}...")
                log.info(f"  Outputs[0]: {sr.outputs[0][:80]}...")
                log.info(f"  Peakedness: {peak:.4f}, Edit dist mean: {dist.summary['mean']:.4f}")

        all_detection_results[condition] = {
            "peakedness": np.array(peakedness_scores),
            "labels": np.array(test_labels),
        }
        del model, tokenizer

    # ═══════════════════════════════════════════════════════════════
    # STAGE 4: Baselines
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 4: Baselines")
    log.info("=" * 60)

    all_baseline_results = {}

    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        model_dir = model_dirs[ce]
        labels = all_detection_results[condition]["labels"]
        n_test = len(labels)

        # Random baseline
        random_preds = random_classify_batch(n_test, seed=SEED)
        random_pred_arr = np.array([r.is_contaminated for r in random_preds], dtype=int)
        random_scores = np.array([r.confidence for r in random_preds])

        # Perplexity baseline
        log.info(f"[{condition}] Computing perplexity baseline...")
        model, tokenizer, _ = load_checkpoint(model_dir)
        model.eval()
        test_prompts = contam_prompts + eval_prompts
        ppls = compute_perplexity_batch(model, tokenizer, test_prompts, max_length=128)
        ppls_arr = np.array(ppls)
        ppl_threshold = ppl_find_threshold(ppls_arr, labels)
        ppl_preds = (ppls_arr < ppl_threshold).astype(int)
        ppl_scores = 1.0 / (1.0 + ppls_arr)

        # N-gram overlap baseline
        train_texts_for_ngram = format_prompts(contam_train_sets[ce])
        ngram_det = NGramOverlapDetector(train_texts_for_ngram, n=3)  # n=3 for short texts
        overlaps = ngram_det.compute_overlap_batch(test_prompts)
        overlaps_arr = np.array(overlaps)
        ngram_threshold = ngram_find_threshold(overlaps_arr, labels)
        ngram_preds = (overlaps_arr > ngram_threshold).astype(int)

        all_baseline_results[condition] = {
            "random_preds": random_pred_arr,
            "random_scores": random_scores,
            "ppl_preds": ppl_preds,
            "ppl_scores": ppl_scores,
            "ppl_threshold": ppl_threshold,
            "ngram_preds": ngram_preds,
            "ngram_scores": overlaps_arr,
            "ngram_threshold": ngram_threshold,
        }
        del model, tokenizer
        log.info(f"[{condition}] Perplexity threshold: {ppl_threshold:.2f}, N-gram threshold: {ngram_threshold:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 5: Evaluation
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 5: Evaluation")
    log.info("=" * 60)

    eval_dir = os.path.join(OUTPUT_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    all_metrics = {}
    all_cis = {}

    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        det = all_detection_results[condition]
        base = all_baseline_results[condition]
        labels = det["labels"]
        peak_scores = det["peakedness"]

        # Find optimal CDD threshold
        cdd_threshold = find_optimal_threshold(peak_scores, labels)
        cdd_preds = (peak_scores > cdd_threshold).astype(int)

        log.info(f"\n{'='*40}")
        log.info(f"Condition: {condition}")
        log.info(f"CDD threshold (optimal): {cdd_threshold:.4f}")
        log.info(f"{'='*40}")

        methods = {
            "CDD": (cdd_preds, peak_scores),
            "Random": (base["random_preds"], base["random_scores"]),
            "Perplexity": (base["ppl_preds"], base["ppl_scores"]),
            "NGram": (base["ngram_preds"], base["ngram_scores"]),
        }

        condition_metrics = {}
        condition_cis = {}

        for method_name, (preds, scores) in methods.items():
            m = compute_metrics(labels, preds, scores)
            ci = bootstrap_confidence_intervals(labels, preds, scores, n_bootstrap=500, seed=SEED)

            condition_metrics[method_name] = m
            condition_cis[method_name] = ci

            log.info(f"  {method_name:12s}: Acc={m.accuracy:.3f} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} AUC={m.auc:.3f}")
            log.info(f"    Acc CI: [{ci['accuracy'][0]:.3f}, {ci['accuracy'][2]:.3f}]")

        # Significance tests: CDD vs each baseline
        for baseline_name in ["Random", "Perplexity", "NGram"]:
            p = mcnemar_test(labels, cdd_preds, methods[baseline_name][0])
            log.info(f"  McNemar CDD vs {baseline_name}: p={p:.4f}")

        all_metrics[condition] = condition_metrics
        all_cis[condition] = condition_cis

    # Export results
    for condition, cond_metrics in all_metrics.items():
        from contamination_detection.evaluation.metrics import MetricsResult as MR
        export_dict = {}
        for method_name, m in cond_metrics.items():
            key = f"{condition}_{method_name}"
            export_dict[key] = m
        export_csv(export_dict, os.path.join(eval_dir, f"{condition}_metrics.csv"))
        export_latex(export_dict, os.path.join(eval_dir, f"{condition}_metrics.tex"))
        export_json(export_dict, os.path.join(eval_dir, f"{condition}_metrics.json"))

    log.info(f"Evaluation results exported to {eval_dir}")

    # ═══════════════════════════════════════════════════════════════
    # STAGE 6: Visualizations
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("STAGE 6: Visualizations")
    log.info("=" * 60)

    # ROC curves per condition
    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        det = all_detection_results[condition]
        base = all_baseline_results[condition]
        labels = det["labels"]

        y_true_dict = {
            f"CDD": labels,
            f"Perplexity": labels,
            f"NGram": labels,
        }
        y_scores_dict = {
            f"CDD": det["peakedness"],
            f"Perplexity": base["ppl_scores"],
            f"NGram": base["ngram_scores"],
        }
        plot_roc_curves(
            y_true_dict, y_scores_dict,
            output_path=os.path.join(fig_dir, f"roc_{condition}.pdf"),
            title=f"ROC Curves — {condition}",
        )

    # Peakedness distributions
    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        det = all_detection_results[condition]
        labels = det["labels"]
        peaks = det["peakedness"]

        contam_peaks = peaks[labels == 1]
        clean_peaks = peaks[labels == 0]

        plot_peakedness_distributions(
            contam_peaks, clean_peaks,
            output_path=os.path.join(fig_dir, f"peakedness_dist_{condition}.pdf"),
            title=f"Peakedness Distribution — {condition}",
            kind="histogram",
        )

    # Accuracy vs contamination level
    acc_by_contam = {}
    for method_name in ["CDD", "Random", "Perplexity", "NGram"]:
        acc_by_contam[method_name] = {}
        for ce in CONTAM_EPOCHS:
            condition = f"pythia-70m_contam{ce}"
            acc_by_contam[method_name][ce] = all_metrics[condition][method_name].accuracy

    plot_accuracy_vs_contamination_level(
        acc_by_contam,
        output_path=os.path.join(fig_dir, "accuracy_vs_contamination.pdf"),
        title="Detection Accuracy vs Contamination Level (Pythia-70M)",
        contamination_levels=CONTAM_EPOCHS,
    )

    # Heatmap (just 1 model size × 2 contam levels for now)
    for metric_name in ["accuracy", "f1", "auc"]:
        for method_name in ["CDD", "Perplexity", "NGram"]:
            vals = []
            for ce in CONTAM_EPOCHS:
                condition = f"pythia-70m_contam{ce}"
                m = all_metrics[condition][method_name]
                vals.append(getattr(m, metric_name))
            vals_arr = np.array(vals).reshape(1, -1)
            plot_performance_heatmap(
                vals_arr,
                row_labels=["70M"],
                col_labels=[str(c) for c in CONTAM_EPOCHS],
                output_path=os.path.join(fig_dir, f"heatmap_{metric_name}_{method_name}.pdf"),
                title=f"{metric_name.upper()} — {method_name}",
                metric_name=metric_name.upper(),
            )

    log.info(f"All figures saved to {fig_dir}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("RUN COMPLETE — Summary")
    log.info("=" * 60)
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"Model: {MODEL_NAME}")
    log.info(f"Dataset: QASC ({N_EXAMPLES} examples)")
    log.info(f"Contamination epochs: {CONTAM_EPOCHS}")
    log.info(f"Samples per prompt: {N_SAMPLES}")

    for ce in CONTAM_EPOCHS:
        condition = f"pythia-70m_contam{ce}"
        log.info(f"\n  {condition}:")
        for method_name in ["CDD", "Random", "Perplexity", "NGram"]:
            m = all_metrics[condition][method_name]
            log.info(f"    {method_name:12s}: Acc={m.accuracy:.3f} F1={m.f1:.3f} AUC={m.auc:.3f}")

    # List output files
    log.info(f"\nOutput files:")
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), OUTPUT_DIR)
            size = os.path.getsize(os.path.join(root, f))
            log.info(f"  {rel} ({size:,} bytes)")


if __name__ == "__main__":
    main()
