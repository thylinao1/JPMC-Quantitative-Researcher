"""Profit and threshold evaluation for the credit-risk model.

A single ``profit_at_threshold`` function is the source of truth
for the cost matrix; every other function in this module and every
notebook call route through it. The cost convention:

    TN: good loan approved   -> + loan_amount * margin
    FN: defaulter approved   -> - loan_amount * lgd
    FP: good loan rejected   -> - loan_amount * margin    (lost margin)
    TP: defaulter rejected   ->   0
"""

from __future__ import annotations

import numpy as np


def profit_at_threshold(probs, labels, threshold, loan_amount, margin, lgd):
    """Expected profit at a single classification threshold.

    A loan is *rejected* when ``probs >= threshold``. Returns a dict
    with the profit and the confusion-matrix counts that produced it.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.shape != labels.shape:
        raise ValueError(
            f"probs and labels shape mismatch: {probs.shape} vs {labels.shape}"
        )
    predictions = (probs >= threshold).astype(int)
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    tp = int(((predictions == 1) & (labels == 1)).sum())
    profit = (
        tn * loan_amount * margin
        - fp * loan_amount * margin
        - fn * loan_amount * lgd
    )
    return {
        "profit": float(profit),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(labels.shape[0]),
    }


def optimal_threshold(
    probs, labels, loan_amount, margin, lgd, grid=None
):
    """Sweep thresholds and return the one with highest profit.

    Uses the same ``profit_at_threshold`` function so the optimum
    cannot disagree with the per-threshold profit. Returns a dict
    with the threshold, the profit at that threshold, and the full
    sweep arrays for plotting.
    """
    if grid is None:
        grid = np.linspace(0, 1, 101)
    profits = np.array([
        profit_at_threshold(probs, labels, t, loan_amount, margin, lgd)["profit"]
        for t in grid
    ])
    best = int(np.argmax(profits))
    return {
        "threshold": float(grid[best]),
        "profit": float(profits[best]),
        "grid": grid,
        "profits": profits,
    }


def bootstrap_profit_ci(
    probs, labels, threshold, loan_amount, margin, lgd,
    n_boot=2000, ci=0.95, seed=42,
):
    """Bootstrap confidence interval on profit at a fixed threshold."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = probs.shape[0]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = profit_at_threshold(
            probs[idx], labels[idx], threshold, loan_amount, margin, lgd
        )["profit"]
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))
    point = profit_at_threshold(
        probs, labels, threshold, loan_amount, margin, lgd
    )["profit"]
    return {"point": point, "lo": lo, "hi": hi, "boot": boot}


def bootstrap_threshold_ci(
    probs, labels, loan_amount, margin, lgd,
    n_boot=2000, ci=0.95, seed=42, grid=None,
):
    """Bootstrap CI on the optimal threshold itself."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    n = probs.shape[0]
    rng = np.random.default_rng(seed)
    grid = grid if grid is not None else np.linspace(0, 1, 101)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = optimal_threshold(
            probs[idx], labels[idx], loan_amount, margin, lgd, grid=grid
        )["threshold"]
    alpha = (1 - ci) / 2
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))
    point = optimal_threshold(
        probs, labels, loan_amount, margin, lgd, grid=grid
    )["threshold"]
    return {"point": point, "lo": lo, "hi": hi, "boot": boot}
