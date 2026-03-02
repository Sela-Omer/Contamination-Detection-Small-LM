"""Statistical significance tests for comparing detection methods.

Implements McNemar's test and paired bootstrap test, returning p-values.
"""

import logging
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score

logger = logging.getLogger("contamination_detection.evaluation.significance")


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> float:
    """McNemar's test comparing two methods' binary predictions.

    Tests whether the two methods disagree in a systematic way.

    Args:
        y_true: 1-D binary ground-truth labels.
        y_pred_a: 1-D binary predictions from method A.
        y_pred_b: 1-D binary predictions from method B.

    Returns:
        Two-sided p-value.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    n = b + c
    if n == 0:
        logger.info("McNemar: both methods agree on all examples, p=1.0")
        return 1.0

    # McNemar's test with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0

    # Compute p-value from chi-squared distribution with 1 df
    from scipy.stats import chi2 as chi2_dist

    p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))
    logger.info(f"McNemar: b={b}, c={c}, chi2={chi2:.4f}, p={p_value:.6f}")
    return p_value


def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test comparing two methods' accuracy.

    Uses the shift method: under H0 (no difference), the bootstrap
    distribution of the difference is centred at zero. We count how
    often a bootstrap difference (shifted to be centred at 0) is at
    least as extreme as the observed difference.

    Args:
        y_true: 1-D binary ground-truth labels.
        y_pred_a: 1-D binary predictions from method A.
        y_pred_b: 1-D binary predictions from method B.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Two-sided p-value.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_a = np.asarray(y_pred_a, dtype=int)
    y_pred_b = np.asarray(y_pred_b, dtype=int)
    n = len(y_true)
    rng = np.random.RandomState(seed)

    observed_diff = float(
        accuracy_score(y_true, y_pred_a) - accuracy_score(y_true, y_pred_b)
    )

    # Bootstrap under the null: shift differences to be centred at 0
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = float(
            accuracy_score(y_true[idx], y_pred_a[idx])
            - accuracy_score(y_true[idx], y_pred_b[idx])
        )

    # Centre the bootstrap distribution at 0 (null hypothesis)
    boot_diffs_centred = boot_diffs - boot_diffs.mean()

    # Two-sided p-value
    count = int(np.sum(np.abs(boot_diffs_centred) >= abs(observed_diff)))
    p_value = count / n_bootstrap

    logger.info(
        f"Paired bootstrap: observed_diff={observed_diff:.4f}, "
        f"p={p_value:.4f} ({n_bootstrap} resamples)"
    )
    return float(p_value)
