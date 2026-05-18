"""Tests for calibration helpers."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.credit.calibration import (
    compare_calibration,
    expected_calibration_error,
    fit_isotonic_calibrator,
    reliability_curve,
)


def test_reliability_curve_perfect_calibration_gives_zero_gap():
    # Construct data where the predicted probability really matches.
    rng = np.random.default_rng(0)
    n = 2000
    probs = rng.uniform(0, 1, size=n)
    labels = rng.binomial(1, probs)
    curve = reliability_curve(probs, labels, n_bins=10)
    # Each (pred_avg, true_avg) pair should be close; the average
    # absolute gap should be small.
    gap = np.abs(curve["true_avg"] - curve["pred_avg"]).mean()
    assert gap < 0.05


def test_reliability_curve_skips_empty_bins():
    # All probs in [0.4, 0.6]; only the middle bin is populated.
    probs = np.full(50, 0.5)
    labels = np.zeros(50, dtype=int)
    curve = reliability_curve(probs, labels, n_bins=10)
    assert len(curve["pred_avg"]) == 1
    assert curve["pred_avg"][0] == pytest.approx(0.5)


def test_reliability_curve_shape_mismatch_raises():
    with pytest.raises(ValueError):
        reliability_curve(np.array([0.1, 0.2]), np.array([0]), n_bins=5)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(1)
    n = 5000
    probs = rng.uniform(0, 1, size=n)
    labels = rng.binomial(1, probs)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    # With perfect generating process and 5k samples, ECE should be near zero.
    assert ece < 0.03


def test_ece_large_for_always_wrong():
    # Predicts 0.9 for every sample but the true rate is 0.1.
    n = 1000
    probs = np.full(n, 0.9)
    labels = np.zeros(n, dtype=int)
    labels[: int(0.1 * n)] = 1
    ece = expected_calibration_error(probs, labels, n_bins=10)
    # Gap should be roughly 0.9 - 0.1 = 0.8.
    assert ece > 0.5


def test_ece_bounded_in_unit_interval():
    rng = np.random.default_rng(2)
    n = 500
    probs = rng.uniform(0, 1, size=n)
    labels = rng.binomial(1, 0.3, size=n)
    ece = expected_calibration_error(probs, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_fit_isotonic_calibrator_returns_fitted_estimator():
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(0, 0.5, size=n) > 0).astype(int)
    lr = LogisticRegression(max_iter=1000)
    cal = fit_isotonic_calibrator(lr, X, y, cv=3)
    # Calibrated wrapper exposes predict_proba.
    probs = cal.predict_proba(X)[:, 1]
    assert probs.shape == (n,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_compare_calibration_returns_both_eces_and_briers():
    rng = np.random.default_rng(4)
    n = 600
    labels = rng.binomial(1, 0.3, size=n)
    probs_uncal = np.clip(0.3 + 0.5 * labels + rng.normal(0, 0.1, size=n), 0, 1)
    # Calibrated probs that better match the empirical rate per bin.
    probs_cal = np.clip(0.05 + 0.85 * labels + rng.normal(0, 0.05, size=n), 0, 1)
    out = compare_calibration(probs_uncal, probs_cal, labels, n_bins=10)
    for k in ("ece_uncal", "ece_cal", "brier_uncal", "brier_cal", "curve_uncal", "curve_cal"):
        assert k in out


def test_compare_calibration_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compare_calibration(np.array([0.1, 0.2]), np.array([0.1]), np.array([0, 1]))
