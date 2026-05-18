"""Generic bootstrap confidence intervals for sklearn-style metrics.

The headline profit and improvement numbers already carry bootstrap
CIs via :mod:`src.credit.eval`. This module extends that uncertainty
treatment to AUC, Brier score, recall, precision, F1, and any other
metric of the shape ``metric(y_true, y_pred) -> float``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_metric_ci(
    metric_fn,
    y_true,
    y_pred,
    n_boot=2000,
    ci=0.95,
    seed=42,
):
    """Bootstrap CI for any metric of the form ``metric(y_true, y_pred)``.

    Parameters
    ----------
    metric_fn : callable
        Two-argument metric. For probability-based metrics (AUC,
        Brier), pass continuous predictions. For decision-based
        metrics (recall, precision), pass already-binarised
        predictions.
    y_true, y_pred : array-like
        Must be the same length.
    n_boot : int
        Number of bootstrap resamples (default 2000).
    ci : float
        Confidence level in (0, 1).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: ``point`` (unbootstrapped metric value),
    ``lo`` (lower CI bound), ``hi`` (upper CI bound), ``boot``
    (1-D array of bootstrap replicates).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred shape mismatch: {y_true.shape} vs {y_pred.shape}"
        )
    n = y_true.shape[0]
    if n == 0:
        raise ValueError("y_true is empty")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1); got {ci}")

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(metric_fn(y_true[idx], y_pred[idx]))

    point = float(metric_fn(y_true, y_pred))
    alpha = (1 - ci) / 2
    return {
        "point": point,
        "lo": float(np.quantile(boot, alpha)),
        "hi": float(np.quantile(boot, 1 - alpha)),
        "boot": boot,
    }


def threshold_predictions(probs, threshold):
    """Convenience: binarise probabilities at a fixed threshold."""
    probs = np.asarray(probs, dtype=float)
    return (probs >= threshold).astype(int)
