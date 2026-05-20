"""Loaders and splits for the credit-risk dataset.

The training/threshold-selection/test split keeps threshold tuning
out of the data the headline profit is reported on, which is the
only way to avoid selecting the empirically best operating point
on the same fold that scores it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


EXPECTED_COLUMNS = (
    "customer_id",
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
    "default",
)

# Subset of features used in the "restricted" track: the full feature
# set is trivially separable on this synthetic dataset, so the
# analysis restricts to these four for a realistic modelling regime.
RESTRICTED_FEATURES = (
    "income",
    "years_employed",
    "fico_score",
    "loan_amt_outstanding",
)


def load_loan_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"loan data not found: {path}")
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {path.name}: {sorted(missing)}")
    return df


def restricted_features():
    return list(RESTRICTED_FEATURES)


def train_threshold_test_split(X, y, sizes=(0.6, 0.2, 0.2), seed=42):
    """Stratified three-way split: train, threshold-selection, test.

    The threshold-selection set is held out from training and used
    to pick the operating point. The test set is held out from both
    and used only to report the final profit. Returns six arrays in
    the order: X_train, X_thresh, X_test, y_train, y_thresh, y_test.
    """
    if abs(sum(sizes) - 1.0) > 1e-9:
        raise ValueError(f"sizes must sum to 1.0; got {sum(sizes)}")
    train_frac, thresh_frac, test_frac = sizes
    if min(sizes) <= 0:
        raise ValueError(f"each split must be positive; got {sizes}")

    # First split off test, then split the remaining into train/thresh.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=y
    )
    rel_thresh = thresh_frac / (train_frac + thresh_frac)
    X_train, X_thresh, y_train, y_thresh = train_test_split(
        X_rest, y_rest, test_size=rel_thresh, random_state=seed, stratify=y_rest
    )
    return X_train, X_thresh, X_test, y_train, y_thresh, y_test


def verify_no_overlap(*indices):
    """Cheap leakage guard. Raises if any pair of index sets intersects."""
    sets = [set(np.asarray(idx).ravel().tolist()) for idx in indices]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = sets[i] & sets[j]
            if inter:
                raise ValueError(
                    f"index sets {i} and {j} overlap on {len(inter)} rows"
                )
    return True
