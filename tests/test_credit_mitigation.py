"""Tests for the cohort-generalisation mitigations."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.credit.mitigation import (
    breakeven_margin,
    cohort_mitigation_report,
    domain_classifier_weights,
    fit_importance_weighted,
)


def _two_cohorts(seed=0, n=600):
    """Source and target cohorts with a deliberate covariate shift."""
    rng = np.random.default_rng(seed)
    # Source: features centred at 0. Target: shifted by +1 on column 0.
    X_source = rng.normal(0.0, 1.0, size=(n, 3))
    X_target = rng.normal(0.0, 1.0, size=(n, 3))
    X_target[:, 0] += 1.0
    y_source = (X_source[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    y_target = (X_target[:, 0] + rng.normal(0, 0.5, n) > 0).astype(int)
    return X_source, y_source, X_target, y_target


def test_domain_weights_are_positive_and_mean_one():
    Xs, _, Xt, _ = _two_cohorts()
    w = domain_classifier_weights(Xs, Xt, seed=0)
    assert (w > 0).all()
    assert w.mean() == pytest.approx(1.0, rel=1e-9)
    assert len(w) == len(Xs)


def test_domain_weights_uniform_when_no_shift():
    # Source and target drawn from the same distribution -> weights ~ 1.
    rng = np.random.default_rng(1)
    Xs = rng.normal(size=(800, 3))
    Xt = rng.normal(size=(800, 3))
    w = domain_classifier_weights(Xs, Xt, seed=1)
    # Spread should be small: no row looks much more "target" than another.
    assert w.std() < 0.25


def test_domain_weights_reject_column_mismatch():
    Xs = np.zeros((10, 3))
    Xt = np.zeros((10, 4))
    with pytest.raises(ValueError):
        domain_classifier_weights(Xs, Xt)


def test_fit_importance_weighted_returns_fitted_model():
    Xs, ys, Xt, _ = _two_cohorts()
    w = domain_classifier_weights(Xs, Xt, seed=0)
    model = fit_importance_weighted(LogisticRegression(max_iter=1000), Xs, ys, w)
    probs = model.predict_proba(Xs)[:, 1]
    assert probs.shape == (len(ys),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_fit_importance_weighted_rejects_length_mismatch():
    Xs, ys, _, _ = _two_cohorts()
    with pytest.raises(ValueError):
        fit_importance_weighted(LogisticRegression(), Xs, ys, np.ones(len(ys) + 1))


def test_mitigation_report_has_expected_keys():
    Xs, ys, Xt, yt = _two_cohorts()
    out = cohort_mitigation_report(
        LogisticRegression(max_iter=1000),
        Xs, ys, Xt, yt,
        loan_amount=10_000, margin=0.15, lgd=0.90, seed=0,
    )
    for k in ("n_test", "t_source", "t_adaptive", "baseline",
              "adaptive_threshold", "importance_weighted", "combined",
              "oracle", "reject_all", "weight_min", "weight_max"):
        assert k in out


def test_mitigation_report_reject_all_is_zero():
    Xs, ys, Xt, yt = _two_cohorts()
    out = cohort_mitigation_report(
        LogisticRegression(max_iter=1000),
        Xs, ys, Xt, yt,
        loan_amount=10_000, margin=0.15, lgd=0.90, seed=0,
    )
    assert out["reject_all"] == 0.0


def test_oracle_dominates_every_classifier_rule():
    # The oracle threshold is, by construction, the best achievable on the
    # test half. No baseline/mitigation rule may beat it.
    Xs, ys, Xt, yt = _two_cohorts()
    out = cohort_mitigation_report(
        LogisticRegression(max_iter=1000),
        Xs, ys, Xt, yt,
        loan_amount=10_000, margin=0.15, lgd=0.90, seed=0,
    )
    for rule in ("baseline", "adaptive_threshold", "combined"):
        assert out[rule] <= out["oracle"] + 1e-6


def test_breakeven_margin_found_on_profitable_grid():
    # A cohort with a low base rate breaks even at a low margin.
    rng = np.random.default_rng(3)
    n = 800
    probs = np.clip(rng.beta(2, 8, n), 0, 1)   # mostly low default probs
    y = rng.binomial(1, probs)
    half = n // 2
    m = breakeven_margin(probs[:half], y[:half], probs[half:], y[half:],
                         loan_amount=10_000, lgd=0.90,
                         margin_grid=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    assert m is not None
    assert 0.0 < m <= 0.50


def test_breakeven_margin_returns_none_when_never_profitable():
    # High default rate (60%) with a probability signal uncorrelated with the
    # label: approving loses to defaults, rejecting loses good-loan margin, so
    # no threshold breaks even at a low margin.
    rng = np.random.default_rng(5)
    n = 800
    y = rng.binomial(1, 0.60, n)
    probs = rng.uniform(0, 1, n)  # carries no information about y
    half = n // 2
    m = breakeven_margin(probs[:half], y[:half], probs[half:], y[half:],
                         loan_amount=10_000, lgd=0.90,
                         margin_grid=[0.02, 0.05, 0.08])
    assert m is None
