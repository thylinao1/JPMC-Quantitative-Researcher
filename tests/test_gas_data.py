"""Tests for the gas data loader."""

import pandas as pd
import pytest

from src.gas.data import chronological_split, load_prices


def test_load_prices_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "nope.csv")


def test_load_prices_rejects_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("Dates,Prices\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_load_prices_rejects_wrong_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Day,Cost\n2020-01-01,10\n")
    with pytest.raises(ValueError):
        load_prices(p)


def test_load_prices_sorts_and_parses(tmp_path):
    p = tmp_path / "ok.csv"
    p.write_text("Dates,Prices\n2020-02-29,11\n2020-01-31,10\n")
    df = load_prices(p)
    assert list(df["Prices"]) == [10, 11]
    assert df["Dates"].iloc[0] == pd.Timestamp("2020-01-31")


def test_chronological_split_partitions_correctly():
    df = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=6, freq="ME"),
        "Prices": list(range(6)),
    })
    train, test = chronological_split(df, n_train_months=4)
    assert len(train) == 4
    assert len(test) == 2
    assert list(train["Prices"]) == [0, 1, 2, 3]
    assert list(test["Prices"]) == [4, 5]


def test_chronological_split_rejects_degenerate():
    df = pd.DataFrame({
        "Dates": pd.date_range("2020-01-31", periods=3, freq="ME"),
        "Prices": [1, 2, 3],
    })
    for n in (0, 3, 5, -1):
        with pytest.raises(ValueError):
            chronological_split(df, n)
