"""Tests for the operational profile of a credit decision rule."""

import numpy as np
import pytest

from src.credit.operational import operational_profile


def test_threshold_zero_rejects_all():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    out = operational_profile(probs, labels, threshold=0.0, loan_amount=10_000)
    assert out["rejected"] == 4
    assert out["approved"] == 0
    assert out["rejection_rate"] == pytest.approx(1.0)
    assert out["false_rejection_count"] == 2
    assert out["approved_volume"] == pytest.approx(0.0)


def test_threshold_above_one_approves_all():
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([0, 0, 1, 1])
    out = operational_profile(probs, labels, threshold=2.0, loan_amount=10_000)
    assert out["approved"] == 4
    assert out["rejection_rate"] == pytest.approx(0.0)
    assert out["approved_volume"] == pytest.approx(4 * 10_000)
    assert out["approved_default_rate"] == pytest.approx(0.5)


def test_false_rejection_count_matches_definition():
    probs = np.array([0.1, 0.6, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])  # 0.6 is a false positive (good loan, rejected)
    out = operational_profile(probs, labels, threshold=0.5, loan_amount=10_000)
    assert out["rejected"] == 3
    assert out["false_rejection_count"] == 1
    assert out["approved_default_rate"] == pytest.approx(0.0)
