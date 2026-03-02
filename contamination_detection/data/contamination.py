"""Create contaminated training sets by mixing clean data with repeated contamination examples."""

import logging
from typing import Optional

from datasets import Dataset, concatenate_datasets

logger = logging.getLogger("contamination_detection")


def create_contaminated_training_set(
    clean_train: Dataset,
    contamination_set: Dataset,
    contamination_epochs: int = 1,
    seed: int = 42,
) -> Dataset:
    """Combine clean training data with the contamination set repeated N times.

    Args:
        clean_train: The clean training split.
        contamination_set: Examples to inject as contamination.
        contamination_epochs: How many times to repeat the contamination set.
            0 means no contamination (return only clean data).
        seed: Random seed for shuffling the combined set.

    Returns:
        A shuffled Dataset containing clean + repeated contamination examples.
    """
    if contamination_epochs < 0:
        raise ValueError(f"contamination_epochs must be >= 0, got {contamination_epochs}")

    if contamination_epochs == 0:
        logger.info(
            f"contamination_epochs=0: returning clean training set only "
            f"({len(clean_train)} examples)"
        )
        return clean_train.shuffle(seed=seed)

    # Repeat contamination set N times and concatenate with clean data
    parts = [clean_train] + [contamination_set] * contamination_epochs
    combined = concatenate_datasets(parts)
    combined = combined.shuffle(seed=seed)

    n_contam = len(contamination_set) * contamination_epochs
    logger.info(
        f"Created contaminated training set: "
        f"{len(clean_train)} clean + {n_contam} contamination "
        f"({contamination_epochs} epochs × {len(contamination_set)}) = "
        f"{len(combined)} total"
    )

    return combined
