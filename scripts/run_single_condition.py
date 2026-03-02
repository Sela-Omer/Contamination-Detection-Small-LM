#!/usr/bin/env python
"""Run a single experimental condition (one model × one contamination level × one ft method).

Designed to be launched multiple times in parallel, one per GPU.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_single_condition.py \
        --model EleutherAI/pythia-70m --contam_epochs 0 --ft_method lora8

All stages: fine-tune → CDD detect → baselines.
Evaluation and visualization happen in run_aggregate.py.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from contamination_detection.utils import set_global_seed, timer
from contamination_detection.data.loader import load_saved_dataset
from contamination_detection.data.formatter import format_prompts, format_training_texts
from contamination_detection.training.model_loader import (
    load_pythia_with_lora, load_pythia_full, load_checkpoint,
)
from contamination_detection.training.trainer import fine_tune
from contamination_detection.detection.sampler import sample_outputs_cdd
from contamination_detection.detection.edit_distance import compute_edit_distances_star, compute_peakedness
from contamination_detection.detection.classifier import classify
from contamination_detection.baselines.random_baseline import classify_batch as random_classify_batch
from contamination_detection.baselines.perplexity_detector import (
    compute_perplexity_batch, find_optimal_threshold as ppl_find_threshold,
)
from contamination_detection.baselines.ngram_detector import (
    NGramOverlapDetector, find_optimal_threshold as ngram_find_threshold,
)
from contamination_detection.config import LoRAConfig, TrainingConfig, SamplingConfig

# ── Fixed experiment parameters ──────────────────────────────────────
SEED = 42
N_SAMPLES = 50
MAX_NEW_TOKENS = 100
TRAIN_EPOCHS = 3
TRAIN_BATCH = 8
TRAIN_LR = 2e-4

MODEL_SHORT = {
    "EleutherAI/pythia-70m": "70M",
    "EleutherAI/pythia-160m": "160M",
    "EleutherAI/pythia-410m": "410M",
    "EleutherAI/pythia-1b": "1B",
}

# Fine-tuning method configs
FT_METHODS = {
    "lora8":   {"type": "lora", "r": 8},
    "lora256": {"type": "lora", "r": 256},
    "full":    {"type": "full"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Run single experimental condition")
    p.add_argument("--model", required=True, help="HF model name, e.g. EleutherAI/pythia-70m")
    p.add_argument("--contam_epochs", type=int, required=True, help="Contamination epochs: 0, 1, 5, or 10")
    p.add_argument("--ft_method", default="lora8", choices=list(FT_METHODS.keys()),
                   help="Fine-tuning method: lora8, lora256, or full")
    p.add_argument("--train_epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--output_dir", default=os.path.expanduser("~/final_proj/outputs/gpu_full_run"))
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def _checkpoint_exists(model_dir, ft_method):
    """Check if a checkpoint already exists for this ft method."""
    if FT_METHODS[ft_method]["type"] == "lora":
        return os.path.exists(os.path.join(model_dir, "adapter_config.json"))
    else:
        return os.path.exists(os.path.join(model_dir, "config.json"))


def main():
    args = parse_args()
    model_name = args.model
    ce = args.contam_epochs
    ft = args.ft_method
    output_dir = args.output_dir
    short = MODEL_SHORT.get(model_name, model_name.split("/")[-1])
    condition = f"{short}_contam{ce}"
    ft_cfg = FT_METHODS[ft]

    # Organize outputs by ft_method: models/lora8/70M_contam0/, detection/lora8/70M_contam0.npz, etc.
    # Include epoch count in path if non-default to avoid conflicts
    ft_tag = ft if args.train_epochs == 3 else f"{ft}_ep{args.train_epochs}"
    model_dir = os.path.join(output_dir, "models", ft_tag, condition)
    detection_dir = os.path.join(output_dir, "detection", ft_tag)
    baselines_dir = os.path.join(output_dir, "baselines", ft_tag)
    loss_dir = os.path.join(output_dir, "loss_histories", ft_tag)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(baselines_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)

    tag = f"{ft_tag}_{condition}"

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{tag}] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(tag)

    set_global_seed(args.seed)

    gpu_info = "CPU"
    if torch.cuda.is_available():
        gpu_info = f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'auto')}: {torch.cuda.get_device_name(0)}"
    log.info(f"Starting {tag} on {gpu_info}")

    # ── Load pre-prepared data ───────────────────────────────────────
    data_dir = os.path.join(output_dir, "data")
    eval_path = os.path.join(data_dir, "evaluation")
    contam_cache = os.path.join(data_dir, f"train_contam_{ce}")

    if not os.path.exists(eval_path) or not os.path.exists(contam_cache):
        log.error(f"Data not prepared! Run 'python scripts/prepare_data.py --output_dir {output_dir}' first.")
        sys.exit(1)

    log.info("Loading pre-prepared data splits...")
    splits_contam = load_saved_dataset(os.path.join(data_dir, "contamination"))
    splits_eval = load_saved_dataset(eval_path)
    contam_train = load_saved_dataset(contam_cache)

    eval_prompts = format_prompts(splits_eval)
    contam_prompts = format_prompts(splits_contam)
    train_texts = format_training_texts(contam_train)
    log.info(f"Contam: {len(splits_contam)}, Eval: {len(splits_eval)}, Train (contam={ce}): {len(contam_train)}")

    # ── Fine-tuning ──────────────────────────────────────────────────
    training_cfg = TrainingConfig(
        learning_rate=TRAIN_LR, batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=2, num_epochs=args.train_epochs,
        warmup_ratio=0.1, seed=args.seed, logging_steps=5,
    )

    if _checkpoint_exists(model_dir, ft):
        log.info(f"Checkpoint exists at {model_dir}, skipping fine-tuning")
    else:
        log.info(f"Fine-tuning ({ft}) on {len(train_texts)} examples...")
        with timer(f"Fine-tuning ({ft})", log):
            if ft_cfg["type"] == "lora":
                r = ft_cfg["r"]
                lora_cfg = LoRAConfig(r=r, lora_alpha=r * 2, lora_dropout=0.05,
                                      target_modules=["query_key_value"])
                model, tokenizer = load_pythia_with_lora(model_name, lora_cfg)
            else:
                lora_cfg = None
                model, tokenizer = load_pythia_full(model_name)

            result = fine_tune(
                model=model, tokenizer=tokenizer,
                train_texts=train_texts, training_config=training_cfg,
                lora_config=lora_cfg, output_dir=model_dir, max_length=256,
            )
            log.info(f"Loss: {result.initial_loss:.4f} -> {result.final_loss:.4f} (sanity={result.sanity_passed})")

            # Save loss history
            with open(os.path.join(loss_dir, f"{condition}.json"), "w") as f:
                json.dump(result.loss_history, f)

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── CDD Detection ────────────────────────────────────────────────
    test_prompts = contam_prompts + eval_prompts
    test_labels = np.array([1] * len(contam_prompts) + [0] * len(eval_prompts))

    detection_path = os.path.join(detection_dir, f"{condition}.npz")

    if os.path.exists(detection_path):
        log.info("Detection results cached, skipping")
    else:
        log.info(f"CDD detection: {len(test_prompts)} prompts x {N_SAMPLES} samples...")
        sampling_cfg = SamplingConfig(
            n_samples=N_SAMPLES, temperature=0.8, top_k=0, top_p=1.0,
            max_new_tokens=MAX_NEW_TOKENS, seed=args.seed,
        )

        with timer("CDD detection", log):
            model, tokenizer, _ = load_checkpoint(model_dir)
            model.eval()

            peaks = []
            all_distances = []
            all_max_lengths = []
            for idx, prompt in enumerate(test_prompts):
                sr = sample_outputs_cdd(
                    prompt=prompt, model=model, tokenizer=tokenizer,
                    n_samples=N_SAMPLES, config=sampling_cfg, seed=args.seed + idx,
                )
                dist = compute_edit_distances_star(
                    greedy_tokens=sr.greedy_tokens,
                    sample_token_lists=sr.sample_token_lists,
                    max_token_length=100,
                )
                peak = compute_peakedness(dist.distances, dist.max_length, alpha=0.05)
                peaks.append(peak)
                all_distances.append(dist.distances)
                all_max_lengths.append(dist.max_length)

                if (idx + 1) % 20 == 0:
                    log.info(f"  {idx+1}/{len(test_prompts)} prompts done")

            np.savez(detection_path,
                     peakedness=np.array(peaks),
                     distances=np.array(all_distances),
                     max_lengths=np.array(all_max_lengths))

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Baselines ────────────────────────────────────────────────────
    baselines_path = os.path.join(baselines_dir, f"{condition}.npz")

    if os.path.exists(baselines_path):
        log.info("Baselines cached, skipping")
    else:
        log.info("Computing baselines...")
        n_test = len(test_labels)

        rr = random_classify_batch(n_test, seed=args.seed)
        random_preds = np.array([r.is_contaminated for r in rr], dtype=int)
        random_scores = np.full(n_test, 0.5)

        with timer("Perplexity baseline", log):
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

        ngram_det = NGramOverlapDetector(train_texts, n=3)
        overlaps = np.array(ngram_det.compute_overlap_batch(test_prompts))
        ngram_thresh = ngram_find_threshold(overlaps, test_labels)
        ngram_preds = (overlaps > ngram_thresh).astype(int)

        np.savez(baselines_path,
                 random_preds=random_preds, random_scores=random_scores,
                 ppl_preds=ppl_preds, ppl_scores=ppl_scores,
                 ngram_preds=ngram_preds, ngram_scores=overlaps)

        log.info(f"PPL thresh={ppl_thresh:.2f}, NGram thresh={ngram_thresh:.4f}")

    log.info(f"Condition {tag} COMPLETE")


if __name__ == "__main__":
    main()
