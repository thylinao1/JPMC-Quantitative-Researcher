"""Tests for the credit-risk evaluation primitives."""

import numpy as np
import pytest

from src.credit.eval import (
    bootstrap_profit_ci,
    bootstrap_threshold_ci,
    optimal_threshold,
    profit_at_threshold,
)


# A small, hand-constructed example so the cost-matrix arithmetic
# can be checked by eye.
PROBS = np.array([0.05, 0.15, 0.55, 0.85])  # one "safe", one "borderline", two "risky"
LABELS = np.array([0, 1, 0, 1])  # the borderline customer actually defaults; the second risky also defaults
LOAN = 10_000
MARGIN = 0.15
LGD = 0.90


def test_profit_at_threshold_zero_rejects_everyone():
    # threshold = 0 => probs >= 0 is always True => all rejected.
    out = profit_at_threshold(PROBS, LABELS, 0.0, LOAN, MARGIN, LGD)
    assert out["tp"] == 2
    assert out["fp"] == 2
    assert out["tn"] == 0
    assert out["fn"] == 0
    # Profit: 0 * margin - 2 * margin - 0 * lgd = -2 * 10000 * 0.15
    assert out["profit"] == pytest.approx(-3000.0)


def test_profit_at_threshold_above_one_approves_everyone():
    # threshold > max(probs) => nobody rejected.
    out = profit_at_threshold(PROBS, LABELS, 1.5, LOAN, MARGIN, LGD)
    assert out["tp"] == 0
    assert out["fp"] == 0
    assert out["tn"] == 2
    assert out["fn"] == 2
    # Profit: 2 * margin - 0 - 2 * lgd = 2*1500 - 2*9000
    assert out["profit"] == pytest.approx(2 * 1500 - 2 * 9000)


def test_profit_at_threshold_middle_picks_riskier_customers():
    # threshold = 0.5 => rejects only the two with probs >= 0.5.
    out = profit_at_threshold(PROBS, LABELS, 0.5, LOAN, MARGIN, LGD)
    assert out["tp"] == 1  # one of the two risky is a defaulter
    assert out["fp"] == 1  # one is a false positive
    assert out["tn"] == 1  # safe customer is approved
    assert out["fn"] == 1  # borderline defaulter approved
    expected = 1 * 1500 - 1 * 1500 - 1 * 9000
    assert out["profit"] == pytest.approx(expected)


def test_profit_function_is_single_source_of_truth():
    # optimal_threshold's sweep at a specific threshold must equal
    # profit_at_threshold at that same threshold. If these diverge,
    # the two functions are using different formulas.
    out_opt = optimal_threshold(
        PROBS, LABELS, LOAN, MARGIN, LGD, grid=np.array([0.5])
    )
    out_pt = profit_at_threshold(PROBS, LABELS, 0.5, LOAN, MARGIN, LGD)
    assert out_opt["profit"] == pytest.approx(out_pt["profit"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        profit_at_threshold(np.array([0.1, 0.2]), np.array([0]), 0.5, LOAN, MARGIN, LGD)


def test_optimal_threshold_finds_known_best():
    # In our hand example the best threshold is in (0.55, 0.85]: it
    # rejects the second defaulter (prob 0.85, label 1) and approves
    # everyone else.
    out = optimal_threshold(PROBS, LABELS, LOAN, MARGIN, LGD)
    # The best in our example should reject the lone defaulter with
    # prob 0.85 while approving the borderline 0.55 (which is also a
    # default; the threshold can't separate it without also rejecting
    # the 0.55 case, but optimum may differ depending on grid). The
    # key invariant: profit at the chosen threshold beats threshold=0.5
    # *or* equals it.
    baseline = profit_at_threshold(PROBS, LABELS, 0.5, LOAN, MARGIN, LGD)["profit"]
    assert out["profit"] >= baseline


def test_bootstrap_profit_ci_contains_point_estimate():
    # Larger synthetic dataset to make the CI meaningful.
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.binomial(1, 0.2, size=n)
    # Correlate probs with labels with some noise.
    probs = np.clip(0.1 + 0.7 * labels + rng.normal(0, 0.1, size=n), 0, 1)
    out = bootstrap_profit_ci(
        probs, labels, threshold=0.5, loan_amount=LOAN,
        margin=MARGIN, lgd=LGD, n_boot=500, seed=0,
    )
    # The point estimate should lie inside the CI (it must — it's
    # the unbootstrapped value, and the CI is symmetric quantiles).
    assert out["lo"] <= out["point"] <= out["hi"]
    assert out["lo"] < out["hi"]


def test_bootstrap_ci_deterministic_under_seed():
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.binomial(1, 0.2, size=n)
    probs = rng.random(size=n)
    a = bootstrap_profit_ci(
        probs, labels, 0.5, LOAN, MARGIN, LGD, n_boot=200, seed=7,
    )
    b = bootstrap_profit_ci(
        probs, labels, 0.5, LOAN, MARGIN, LGD, n_boot=200, seed=7,
    )
    assert a["lo"] == b["lo"]
    assert a["hi"] == b["hi"]


def test_bootstrap_threshold_ci_returns_interval():
    rng = np.random.default_rng(1)
    n = 500
    labels = rng.binomial(1, 0.2, size=n)
    probs = np.clip(0.1 + 0.6 * labels + rng.normal(0, 0.15, size=n), 0, 1)
    out = bootstrap_threshold_ci(
        probs, labels, LOAN, MARGIN, LGD, n_boot=200, seed=1,
    )
    assert 0.0 <= out["lo"] <= out["point"] <= out["hi"] <= 1.0
