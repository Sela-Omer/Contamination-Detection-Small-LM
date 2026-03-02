"""Dataset splitting into non-overlapping train / contamination / evaluation subsets."""

import logging
from dataclasses import dataclass
from typing import Dict, Any

from datasets import Dataset

logger = logging.getLogger("contamination_detection")


@dataclass
class DataSplits:
    """Container for the three non-overlapping dataset splits."""
    train: Dataset
    contamination: Dataset
    evaluation: Dataset
    metadata: Dict[str, Any]


def create_splits(
    dataset: Dataset,
    train_ratio: float = 0.6,
    contamination_ratio: float = 0.2,
    eval_ratio: float = 0.2,
    seed: int = 42,
) -> DataSplits:
    """Partition *dataset* into non-overlapping train / contamination / eval splits.

    Examples are selected via seeded random shuffling (not first-N).

    Args:
        dataset: The full HuggingFace Dataset.
        train_ratio: Fraction for the training set.
        contamination_ratio: Fraction for the contamination set.
        eval_ratio: Fraction for the evaluation set.
        seed: Random seed for reproducibility.

    Returns:
        A DataSplits instance.

    Raises:
        ValueError: If ratios are invalid.
    """
    total = train_ratio + contamination_ratio + eval_ratio
    if total > 1.0:
        raise ValueError(
            f"Split ratios sum to {total:.4f}, which exceeds 1.0. "
            f"train={train_ratio}, contamination={contamination_ratio}, "
            f"eval={eval_ratio}. Each ratio must be in [0, 1] and their sum <= 1.0."
        )
    for name, val in [
        ("train_ratio", train_ratio),
        ("contamination_ratio", contamination_ratio),
        ("eval_ratio", eval_ratio),
    ]:
        if not 0.0 <= val <= 1.0:
            raise ValueError(f"{name}={val} is out of range [0, 1].")

    n = len(dataset)
    n_train = int(n * train_ratio)
    n_contam = int(n * contamination_ratio)
    n_eval = int(n * eval_ratio)

    # Shuffle the full dataset with the given seed, then slice
    shuffled = dataset.shuffle(seed=seed)

    train_ds = shuffled.select(range(n_train))
    contam_ds = shuffled.select(range(n_train, n_train + n_contam))
    eval_ds = shuffled.select(range(n_train + n_contam, n_train + n_contam + n_eval))

    metadata = {
        "total_examples": n,
        "train_size": len(train_ds),
        "contamination_size": len(contam_ds),
        "evaluation_size": len(eval_ds),
        "seed": seed,
        "train_ratio": train_ratio,
        "contamination_ratio": contamination_ratio,
        "eval_ratio": eval_ratio,
    }

    logger.info(
        f"Created splits (seed={seed}): "
        f"train={len(train_ds)}, contamination={len(contam_ds)}, "
        f"eval={len(eval_ds)} from {n} total examples"
    )

    return DataSplits(
        train=train_ds,
        contamination=contam_ds,
        evaluation=eval_ds,
        metadata=metadata,
    )
