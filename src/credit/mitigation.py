"""Mitigations for the cohort-generalisation failure.

The credit model trained on long-tenure customers loses money when
applied to the short-tenure cohort. This module implements two
standard responses and a diagnostic that says which (if either) is
the right tool:

1. Covariate-shift importance weighting. A domain classifier
   estimates the density ratio p_target(x) / p_source(x); the source
   model is refit with those weights so it leans toward the target
   covariate distribution.
2. Cohort-adaptive threshold. The operating point is re-selected on
   a dev split of the target cohort instead of inherited from the
   source cohort.

Neither is guaranteed to work. Importance weighting only addresses
covariate shift (a change in p(x)); it cannot fix concept shift (a
change in p(y|x)) or a base-rate problem where the target cohort is
simply unprofitable under the cost matrix. ``cohort_mitigation_report``
also computes the oracle (best-possible) threshold profit and the
reject-all profit so the caller can see whether any classifier rule
beats declining the cohort outright.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .eval import optimal_threshold, profit_at_threshold


def domain_classifier_weights(X_source, X_target, seed=42, clip=1e-3):
    """Importance weights for the source rows via a domain classifier.

    Fits a logistic regression to separate source rows (label 0) from
    target rows (label 1). For each source row the weight is the
    density ratio ``p(target|x) / p(source|x)``, clipped away from 0
    and 1 for numerical stability, then normalised to mean 1.

    A source row that looks like the target distribution gets a
    weight above 1; a row unlike the target gets a weight below 1.
    """
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    if X_source.ndim != 2 or X_target.ndim != 2:
        raise ValueError("X_source and X_target must be 2-D")
    if X_source.shape[1] != X_target.shape[1]:
        raise ValueError("X_source and X_target must have the same number of columns")

    X = np.vstack([X_source, X_target])
    y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    dom = LogisticRegression(max_iter=1000, random_state=seed)
    dom.fit(X, y)

    p_target = np.clip(dom.predict_proba(X_source)[:, 1], clip, 1 - clip)
    weights = p_target / (1 - p_target)
    return weights / weights.mean()


def fit_importance_weighted(estimator, X, y, weights):
    """Clone ``estimator`` and fit it with per-sample importance weights."""
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(y):
        raise ValueError(
            f"weights length {len(weights)} != y length {len(y)}"
        )
    model = estimator.__class__(**estimator.get_params())
    model.fit(X, y, sample_weight=weights)
    return model


def breakeven_margin(probs_dev, y_dev, probs_test, y_test,
                     loan_amount, lgd, margin_grid):
    """Smallest margin in ``margin_grid`` at which the cohort breaks even.

    For each candidate margin the threshold is chosen on the dev
    split and the profit is measured on the test split. Returns the
    first margin whose test profit is non-negative, or ``None`` if no
    margin in the grid reaches break-even.
    """
    for margin in margin_grid:
        t = optimal_threshold(probs_dev, y_dev, loan_amount, margin, lgd)["threshold"]
        p = profit_at_threshold(probs_test, y_test, t, loan_amount, margin, lgd)["profit"]
        if p >= 0:
            return float(margin)
    return None


def cohort_mitigation_report(
    estimator,
    X_source, y_source,
    X_target, y_target,
    loan_amount, margin, lgd,
    seed=42, target_dev_frac=0.5,
):
    """Run the baseline and both mitigations on a target cohort.

    The target cohort is split into a dev half (for threshold
    re-selection) and a test half (for reporting). Returns a dict
    with the test-half profit for: the baseline rule, the
    cohort-adaptive threshold, importance weighting, the two
    combined, the oracle threshold, and reject-all.
    """
    from sklearn.model_selection import train_test_split

    y_source = np.asarray(y_source)
    y_target = np.asarray(y_target)
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)

    # Base model + source operating point.
    base = estimator.__class__(**estimator.get_params())
    base.fit(X_source, y_source)
    t_source = optimal_threshold(
        base.predict_proba(X_source)[:, 1], y_source, loan_amount, margin, lgd
    )["threshold"]

    # Importance-weighted model + its own source operating point.
    weights = domain_classifier_weights(np.asarray(X_source), np.asarray(X_target), seed=seed)
    weighted = fit_importance_weighted(estimator, X_source, y_source, weights)
    t_source_w = optimal_threshold(
        weighted.predict_proba(X_source)[:, 1], y_source, loan_amount, margin, lgd
    )["threshold"]

    # Split the target cohort: dev for threshold re-selection, test for reporting.
    idx = np.arange(len(y_target))
    dev_idx, test_idx = train_test_split(
        idx, test_size=1 - target_dev_frac, random_state=seed, stratify=y_target
    )
    Xt_dev, Xt_test = X_target[dev_idx], X_target[test_idx]
    yt_dev, yt_test = y_target[dev_idx], y_target[test_idx]

    p_dev = base.predict_proba(Xt_dev)[:, 1]
    p_test = base.predict_proba(Xt_test)[:, 1]
    p_dev_w = weighted.predict_proba(Xt_dev)[:, 1]
    p_test_w = weighted.predict_proba(Xt_test)[:, 1]

    t_adaptive = optimal_threshold(p_dev, yt_dev, loan_amount, margin, lgd)["threshold"]
    t_adaptive_w = optimal_threshold(p_dev_w, yt_dev, loan_amount, margin, lgd)["threshold"]

    def prof(probs, t):
        return profit_at_threshold(probs, yt_test, t, loan_amount, margin, lgd)["profit"]

    grid = np.linspace(0, 1, 101)
    oracle = max(prof(p_test, t) for t in grid)

    n_test = int(len(yt_test))
    return {
        "n_test": n_test,
        "t_source": float(t_source),
        "t_adaptive": float(t_adaptive),
        "baseline": float(prof(p_test, t_source)),
        "adaptive_threshold": float(prof(p_test, t_adaptive)),
        "importance_weighted": float(prof(p_test_w, t_source_w)),
        "combined": float(prof(p_test_w, t_adaptive_w)),
        "oracle": float(oracle),
        "reject_all": 0.0,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
