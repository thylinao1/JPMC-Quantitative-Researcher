"""Tests for the generic bootstrap-CI helper."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, recall_score, roc_auc_score

from src.credit.metrics import bootstrap_metric_ci, threshold_predictions


@pytest.fixture
def synthetic_probs_labels():
    rng = np.random.default_rng(7)
    n = 800
    labels = rng.binomial(1, 0.2, size=n)
    # Probabilities correlated with labels with controlled noise.
    probs = np.clip(0.1 + 0.7 * labels + rng.normal(0, 0.1, size=n), 0, 1)
    return labels, probs


def test_point_estimate_equals_metric_on_raw_inputs(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    out = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=200, seed=0)
    assert out["point"] == pytest.approx(roc_auc_score(labels, probs))


def test_ci_contains_point_estimate(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    out = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=500, seed=0)
    # Point lies inside the CI provided the metric is roughly stable.
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["lo"] < out["hi"]


def test_deterministic_under_seed(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    a = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=300, seed=42)
    b = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=300, seed=42)
    assert np.array_equal(a["boot"], b["boot"])


def test_different_seeds_diverge(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    a = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=300, seed=1)
    b = bootstrap_metric_ci(roc_auc_score, labels, probs, n_boot=300, seed=2)
    assert not np.array_equal(a["boot"], b["boot"])


def test_works_with_brier_score(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    out = bootstrap_metric_ci(brier_score_loss, labels, probs, n_boot=200, seed=0)
    # Brier is bounded in [0, 1].
    assert 0.0 <= out["lo"] <= out["hi"] <= 1.0


def test_works_with_recall_score_on_binarised_predictions(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    preds = threshold_predictions(probs, 0.5)
    out = bootstrap_metric_ci(recall_score, labels, preds, n_boot=200, seed=0)
    assert 0.0 <= out["lo"] <= out["hi"] <= 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_metric_ci(roc_auc_score, np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_empty_input_raises():
    with pytest.raises(ValueError):
        bootstrap_metric_ci(roc_auc_score, np.array([]), np.array([]))


def test_ci_out_of_range_raises(synthetic_probs_labels):
    labels, probs = synthetic_probs_labels
    with pytest.raises(ValueError):
        bootstrap_metric_ci(roc_auc_score, labels, probs, ci=1.5)
    with pytest.raises(ValueError):
        bootstrap_metric_ci(roc_auc_score, labels, probs, ci=0.0)


def test_threshold_predictions_returns_integer_binary():
    probs = np.array([0.1, 0.49, 0.5, 0.9])
    preds = threshold_predictions(probs, 0.5)
    assert preds.dtype == int
    assert list(preds) == [0, 0, 1, 1]
