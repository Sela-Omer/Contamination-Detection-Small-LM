"""Dataset loading and serialization for QA datasets (GSM8K, QASC, StrategyQA)."""

import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk

logger = logging.getLogger("contamination_detection")

# HuggingFace dataset identifiers
_DATASET_REGISTRY = {
    "QASC": {"path": "allenai/qasc", "split": "train"},
    "StrategyQA": {"path": "ChilleD/StrategyQA", "split": "train"},
    "GSM8K": {"path": "openai/gsm8k", "name": "main", "split": "train"},
}


def load_qa_dataset(dataset_name: str, cache_dir: Optional[str] = None) -> Dataset:
    """Load a QA dataset from HuggingFace.

    Args:
        dataset_name: One of "QASC" or "StrategyQA".
        cache_dir: Optional local cache directory for downloads.

    Returns:
        A HuggingFace Dataset object.

    Raises:
        ValueError: If dataset_name is not recognised.
        RuntimeError: If the download fails.
    """
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(_DATASET_REGISTRY.keys())}"
        )

    info = _DATASET_REGISTRY[dataset_name]
    logger.info(f"Loading {dataset_name} from '{info['path']}' (split={info['split']})")

    try:
        kwargs = {"path": info["path"], "split": info["split"]}
        if "name" in info:
            kwargs["name"] = info["name"]
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        ds = load_dataset(**kwargs)
    except Exception as exc:
        msg = f"Failed to load {dataset_name} from HuggingFace: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    logger.info(f"Loaded {dataset_name}: {len(ds)} examples")
    return ds


# ── Serialization helpers ────────────────────────────────────────────────


def save_dataset(dataset: Dataset, path: str) -> None:
    """Serialize a HuggingFace Dataset to disk.

    Args:
        dataset: The dataset to save.
        path: Directory path to save into.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(path)
    logger.info(f"Saved dataset ({len(dataset)} rows) to {path}")


def load_saved_dataset(path: str) -> Dataset:
    """Load a previously serialized HuggingFace Dataset.

    Args:
        path: Directory path containing the saved dataset.

    Returns:
        The loaded Dataset.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"No saved dataset at {path}")
    ds = load_from_disk(path)
    logger.info(f"Loaded dataset ({len(ds)} rows) from {path}")
    return ds
