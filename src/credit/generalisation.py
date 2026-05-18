"""Cohort-based generalisation tests for the credit model.

The Forage loan data has no time column, but ``years_employed`` is a
monotone proxy: a model trained on long-tenure customers and tested
on short-tenure ones is a partial substitute for an out-of-time
split. The metric drop across cohorts measures how much of the
model's lift relies on the population it was trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .eval import optimal_threshold, profit_at_threshold
from .operational import operational_profile


def cohort_split(df, column, threshold):
    """Split a DataFrame on a monotone column at ``threshold``.

    Returns ``(df_train, df_test)`` where ``df_train`` is rows with
    ``column >= threshold`` and ``df_test`` is the strict complement.
    Raises ``ValueError`` if either side is empty or the column is
    missing.
    """
    if column not in df.columns:
        raise ValueError(f"column not found: {column}")
    train_mask = df[column] >= threshold
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"cohort split on {column} at {threshold} produced an empty side"
        )
    return df[train_mask].copy(), df[test_mask].copy()


def cohort_generalisation_report(
    estimator,
    X_train, y_train,
    X_test, y_test,
    loan_amount, margin, lgd,
):
    """Train ``estimator`` on the train cohort; report metrics on test.

    The threshold is chosen on the train cohort so the test cohort is
    a pure held-out generalisation check, never touched by training
    or threshold selection.
    """
    estimator_fitted = estimator.__class__(**estimator.get_params())
    estimator_fitted.fit(X_train, y_train)

    probs_train = estimator_fitted.predict_proba(X_train)[:, 1]
    probs_test = estimator_fitted.predict_proba(X_test)[:, 1]

    # Threshold chosen on the train cohort only.
    sweep = optimal_threshold(probs_train, y_train, loan_amount, margin, lgd)
    t_star = sweep["threshold"]

    pt_train = profit_at_threshold(probs_train, y_train, t_star, loan_amount, margin, lgd)
    pt_test = profit_at_threshold(probs_test, y_test, t_star, loan_amount, margin, lgd)
    op_test = operational_profile(probs_test, y_test, t_star, loan_amount)

    return {
        "t_star": float(t_star),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "profit_train": float(pt_train["profit"]),
        "profit_test": float(pt_test["profit"]),
        "drop": float(pt_train["profit"] - pt_test["profit"]),
        "per_loan_train": float(pt_train["profit"] / pt_train["n"]),
        "per_loan_test": float(pt_test["profit"] / pt_test["n"]),
        "operational_test": op_test,
    }
