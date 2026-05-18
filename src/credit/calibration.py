"""Calibration diagnostics and isotonic post-hoc calibration.

The base logistic regression in this project is already reasonably
calibrated, but post-hoc calibration via
``sklearn.calibration.CalibratedClassifierCV`` is a useful comparison
even without resampling: it gives a second, methodologically
independent estimate of the same probabilities, and it lets us
report an Expected Calibration Error before and after.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def reliability_curve(probs, labels, n_bins=10):
    """Per-bin (mean predicted probability, fraction of positives).

    Empty bins are skipped. Returns a dict with ``pred_avg``,
    ``true_avg``, and ``count_per_bin`` (each a 1-D array of the
    same length).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: {probs.shape} vs {labels.shape}"
        )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize: indices in [1, n_bins]; clip extreme right-edge
    # values to the last bin.
    idx = np.clip(np.digitize(probs, bins[1:-1], right=False), 0, n_bins - 1)

    pred_avg, true_avg, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        pred_avg.append(float(probs[mask].mean()))
        true_avg.append(float(labels[mask].mean()))
        count.append(n)
    return {
        "pred_avg": np.asarray(pred_avg),
        "true_avg": np.asarray(true_avg),
        "count_per_bin": np.asarray(count, dtype=int),
    }


def expected_calibration_error(probs, labels, n_bins=10):
    """Expected calibration error.

    ECE = sum_b (n_b / N) * |true_avg_b - pred_avg_b|.
    Bounded in [0, 1]. 0 is perfect calibration.
    """
    curve = reliability_curve(probs, labels, n_bins=n_bins)
    n_total = curve["count_per_bin"].sum()
    if n_total == 0:
        return float("nan")
    gap = np.abs(curve["true_avg"] - curve["pred_avg"])
    return float(np.sum(curve["count_per_bin"] * gap) / n_total)


def fit_isotonic_calibrator(estimator, X, y, cv=5):
    """Wrap ``estimator`` in a CalibratedClassifierCV with isotonic.

    Uses an internal cross-validation fold over the training set to
    fit the isotonic calibrator without leaking into the calibration
    estimate. Returns the fitted CalibratedClassifierCV.
    """
    cal = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
    cal.fit(X, y)
    return cal


def compare_calibration(probs_uncal, probs_cal, labels, n_bins=10):
    """Side-by-side calibration comparison.

    Returns a dict with ECE and Brier score for both probability
    series, plus the per-bin reliability curves keyed as
    ``curve_uncal`` and ``curve_cal``.
    """
    probs_uncal = np.asarray(probs_uncal, dtype=float)
    probs_cal = np.asarray(probs_cal, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if not (probs_uncal.shape == probs_cal.shape == labels.shape):
        raise ValueError("probs_uncal, probs_cal, labels must all be the same length")

    return {
        "ece_uncal": expected_calibration_error(probs_uncal, labels, n_bins=n_bins),
        "ece_cal":   expected_calibration_error(probs_cal,   labels, n_bins=n_bins),
        "brier_uncal": float(brier_score_loss(labels, probs_uncal)),
        "brier_cal":   float(brier_score_loss(labels, probs_cal)),
        "curve_uncal": reliability_curve(probs_uncal, labels, n_bins=n_bins),
        "curve_cal":   reliability_curve(probs_cal,   labels, n_bins=n_bins),
    }
