"""Utilities for seed setting, logging, timing, and dependency version recording."""

import logging
import os
import random
import time
from contextlib import contextmanager
from typing import Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Sets seeds for: random, numpy, torch (CPU & MPS), and transformers (via env var).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    # HuggingFace transformers reads this env var for its own seed handling
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(False)


def setup_logging(
    log_dir: str = "outputs/logs",
    level: int = logging.INFO,
    tensorboard: bool = True,
) -> logging.Logger:
    """Configure logging with optional TensorBoard integration.

    Returns the root logger configured with console and file handlers.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("contamination_detection")
    logger.setLevel(level)

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    tb_writer = None
    if tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
            logger.info(f"TensorBoard logging to {os.path.join(log_dir, 'tensorboard')}")
        except ImportError:
            logger.warning("TensorBoard not available, skipping TB logging.")

    return logger


@contextmanager
def timer(description: str = "Operation", logger: Optional[logging.Logger] = None):
    """Context manager that times a block and logs the elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"{description} completed in {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def record_dependency_versions() -> Dict[str, str]:
    """Record versions of key dependencies for reproducibility."""
    versions = {}
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("hydra-core", "hydra"),
        ("omegaconf", "omegaconf"),
        ("hypothesis", "hypothesis"),
        ("python-Levenshtein", "Levenshtein"),
    ]
    for display_name, import_name in packages:
        try:
            mod = __import__(import_name)
            versions[display_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[display_name] = "not installed"
    return versions
