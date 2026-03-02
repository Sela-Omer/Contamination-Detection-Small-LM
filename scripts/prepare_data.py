#!/usr/bin/env python
"""Prepare shared data splits ONCE before launching parallel training jobs.

This script creates:
  - data/train, data/contamination, data/evaluation (base splits)
  - data/train_contam_0, data/train_contam_1, data/train_contam_5, data/train_contam_10

Run this BEFORE launch_parallel.sh to avoid race conditions.

Usage:
    cd ~/final_proj && conda run -n cdd python scripts/prepare_data.py \
        --output_dir outputs/gpu_full_run 2>&1 | tee outputs/gpu_full_run/prepare_data.log
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contamination_detection.utils import set_global_seed
from contamination_detection.data.loader import load_qa_dataset, save_dataset, load_saved_dataset
from contamination_detection.data.splitter import create_splits
from contamination_detection.data.contamination import create_contaminated_training_set

SEED = 42
N_EXAMPLES = 500
TRAIN_RATIO = 0.6
CONTAM_RATIO = 0.2
EVAL_RATIO = 0.2
CONTAM_EPOCHS = [0, 1, 5, 10]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=os.path.expanduser("~/final_proj/outputs/gpu_full_run"))
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dataset", default="GSM8K", help="Dataset name: GSM8K, QASC, etc.")
    args = p.parse_args()

    set_global_seed(args.seed)
    data_dir = os.path.join(args.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # ── Base splits ──────────────────────────────────────────────────
    train_path = os.path.join(data_dir, "train")
    contam_path = os.path.join(data_dir, "contamination")
    eval_path = os.path.join(data_dir, "evaluation")

    if os.path.exists(eval_path):
        log.info("Base splits already exist, loading...")
        splits_train = load_saved_dataset(train_path)
        splits_contam = load_saved_dataset(contam_path)
    else:
        log.info(f"Creating base splits from {args.dataset}...")
        full_ds = load_qa_dataset(args.dataset)
        ds = full_ds.shuffle(seed=args.seed).select(range(N_EXAMPLES))
        splits = create_splits(ds, TRAIN_RATIO, CONTAM_RATIO, EVAL_RATIO, seed=args.seed)

        save_dataset(splits.train, train_path)
        save_dataset(splits.contamination, contam_path)
        save_dataset(splits.evaluation, eval_path)
        log.info(f"Saved: train={len(splits.train)}, contam={len(splits.contamination)}, eval={len(splits.evaluation)}")

        splits_train = splits.train
        splits_contam = splits.contamination

    # ── Contaminated training sets for each level ────────────────────
    for ce in CONTAM_EPOCHS:
        cache_path = os.path.join(data_dir, f"train_contam_{ce}")
        if os.path.exists(cache_path):
            log.info(f"train_contam_{ce} already exists, skipping")
            continue

        log.info(f"Creating train_contam_{ce}...")
        contam_train = create_contaminated_training_set(
            splits_train, splits_contam, ce, seed=args.seed
        )
        save_dataset(contam_train, cache_path)
        log.info(f"Saved train_contam_{ce}: {len(contam_train)} examples")

    log.info("Data preparation complete!")


if __name__ == "__main__":
    main()
