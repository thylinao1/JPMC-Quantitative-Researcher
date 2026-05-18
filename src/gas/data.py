"""Data loading and validation for the natural gas price series."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


EXPECTED_N_MONTHS = 48


def load_prices(path):
    """Load the monthly Henry Hub price series.

    Returns a DataFrame with columns ``Dates`` (datetime64[ns]) and
    ``Prices`` (float), sorted ascending by date. Raises
    ``ValueError`` if the file is empty or the date column has
    duplicates.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"price file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("price file is empty")

    if "Dates" not in df.columns or "Prices" not in df.columns:
        raise ValueError(
            f"expected columns Dates, Prices; got {list(df.columns)}"
        )

    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)

    if df["Dates"].duplicated().any():
        raise ValueError("duplicate dates in price series")

    return df


def chronological_split(prices_df, n_train_months):
    """Split a sorted price DataFrame into train and test along time."""
    n = len(prices_df)
    if n_train_months <= 0 or n_train_months >= n:
        raise ValueError(
            f"n_train_months must be in (0, {n}); got {n_train_months}"
        )
    train = prices_df.iloc[:n_train_months].reset_index(drop=True)
    test = prices_df.iloc[n_train_months:].reset_index(drop=True)
    return train, test
