"""Tests for cohort generalisation helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.credit.generalisation import (
    cohort_generalisation_report,
    cohort_split,
)


def test_cohort_split_partitions_at_threshold():
    df = pd.DataFrame({
        "years_employed": [1, 2, 3, 4, 5],
        "x": list(range(5)),
    })
    train, test = cohort_split(df, "years_employed", threshold=3)
    assert set(train["years_employed"]) == {3, 4, 5}
    assert set(test["years_employed"]) == {1, 2}
    assert len(train) + len(test) == len(df)


def test_cohort_split_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError):
        cohort_split(df, "missing", threshold=2)


def test_cohort_split_empty_side_raises():
    df = pd.DataFrame({"x": [1, 1, 1]})
    with pytest.raises(ValueError):
        cohort_split(df, "x", threshold=10)


def test_report_has_expected_keys():
    rng = np.random.default_rng(0)
    n = 400
    X_tr = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y_tr = (rng.uniform(size=n) < 0.2).astype(int)
    X_te = pd.DataFrame(rng.normal(size=(n // 2, 3)), columns=["a", "b", "c"])
    y_te = (rng.uniform(size=n // 2) < 0.2).astype(int)
    lr = LogisticRegression(max_iter=1000)
    out = cohort_generalisation_report(
        lr, X_tr, y_tr, X_te, y_te,
        loan_amount=10_000, margin=0.15, lgd=0.90,
    )
    for k in (
        "t_star", "n_train", "n_test",
        "profit_train", "profit_test", "drop",
        "per_loan_train", "per_loan_test",
        "operational_test",
    ):
        assert k in out
    assert out["n_train"] == n
    assert out["n_test"] == n // 2


def test_report_drop_equals_train_minus_test_profit():
    rng = np.random.default_rng(1)
    n = 300
    X_tr = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y_tr = (X_tr["a"] + rng.normal(0, 0.5, size=n) > 0).astype(int)
    X_te = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y_te = (X_te["a"] + rng.normal(0, 0.5, size=n) > 0).astype(int)
    lr = LogisticRegression(max_iter=1000)
    out = cohort_generalisation_report(
        lr, X_tr, y_tr, X_te, y_te,
        loan_amount=10_000, margin=0.15, lgd=0.90,
    )
    assert out["drop"] == pytest.approx(out["profit_train"] - out["profit_test"])


def test_threshold_chosen_on_train_only():
    # If the threshold were also tuned on the test cohort, swapping
    # test labels would change the chosen t*. The contract here says
    # t* is fixed by the train cohort; swapping test labels must not
    # alter it.
    rng = np.random.default_rng(2)
    n = 400
    X_tr = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y_tr = (X_tr["a"] > 0).astype(int)
    X_te = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    y_te1 = (X_te["a"] > 0).astype(int)
    y_te2 = 1 - y_te1
    lr = LogisticRegression(max_iter=1000)
    a = cohort_generalisation_report(lr, X_tr, y_tr, X_te, y_te1, 10_000, 0.15, 0.90)
    b = cohort_generalisation_report(lr, X_tr, y_tr, X_te, y_te2, 10_000, 0.15, 0.90)
    assert a["t_star"] == b["t_star"]
