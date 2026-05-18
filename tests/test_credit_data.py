"""Tests for the credit data loader and the three-way split."""

import numpy as np
import pandas as pd
import pytest

from src.credit.data import (
    EXPECTED_COLUMNS,
    load_loan_data,
    restricted_features,
    train_threshold_test_split,
    verify_no_overlap,
)


def test_load_loan_data_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_loan_data(tmp_path / "nope.csv")


def test_load_loan_data_rejects_wrong_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        load_loan_data(p)


def test_load_loan_data_loads_good_file(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text(",".join(EXPECTED_COLUMNS) + "\n" + ",".join(["1"] * len(EXPECTED_COLUMNS)) + "\n")
    df = load_loan_data(p)
    assert list(df.columns) == list(EXPECTED_COLUMNS)


def test_restricted_features_matches_notebook_subset():
    feats = restricted_features()
    assert feats == ["income", "years_employed", "fico_score", "loan_amt_outstanding"]


def test_train_threshold_test_split_partitions_disjointly():
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"a": rng.random(n), "b": rng.random(n)})
    y = rng.binomial(1, 0.2, size=n)
    Xtr, Xth, Xte, ytr, yth, yte = train_threshold_test_split(X, y, seed=0)
    assert len(Xtr) + len(Xth) + len(Xte) == n
    # Use original indices to confirm no overlap.
    verify_no_overlap(Xtr.index, Xth.index, Xte.index)


def test_train_threshold_test_split_is_stratified():
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame({"a": rng.random(n)})
    y = rng.binomial(1, 0.15, size=n)
    Xtr, Xth, Xte, ytr, yth, yte = train_threshold_test_split(X, y, seed=1)
    overall = y.mean()
    # Stratification means each split mirrors the overall positive rate
    # within a couple of percentage points on n=2000.
    for split in (ytr, yth, yte):
        assert abs(split.mean() - overall) < 0.02


def test_train_threshold_test_split_rejects_bad_sizes():
    X = pd.DataFrame({"a": np.arange(10)})
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        train_threshold_test_split(X, y, sizes=(0.5, 0.4, 0.2))
    with pytest.raises(ValueError):
        train_threshold_test_split(X, y, sizes=(0.0, 0.5, 0.5))


def test_verify_no_overlap_catches_overlap():
    a = np.array([1, 2, 3])
    b = np.array([3, 4, 5])
    with pytest.raises(ValueError):
        verify_no_overlap(a, b)
