"""Tests for the gas baseline strategies."""

import numpy as np
import pandas as pd
import pytest

from src.gas.baselines import (
    DEFAULT_STORAGE_COST,
    buy_and_hold_profit,
    seasonal_swing_profit,
)


def test_buy_and_hold_zero_storage_simple():
    p = np.array([10.0, 11.0, 12.0])
    assert buy_and_hold_profit(p, units=1, storage_cost_per_unit_per_month=0) == pytest.approx(2.0)


def test_buy_and_hold_charges_storage_per_held_month():
    p = np.array([10.0, 10.5, 12.0])
    profit = buy_and_hold_profit(p, units=1, storage_cost_per_unit_per_month=0.5)
    # Held for 2 months across 3 observations.
    assert profit == pytest.approx(2.0 - 0.5 * 2)


def test_buy_and_hold_scales_with_units():
    p = np.array([10.0, 12.0])
    profit = buy_and_hold_profit(p, units=10, storage_cost_per_unit_per_month=0)
    assert profit == pytest.approx(20.0)


def test_buy_and_hold_can_be_negative_with_storage():
    p = np.array([10.0, 10.05])
    profit = buy_and_hold_profit(p, units=1, storage_cost_per_unit_per_month=0.5)
    assert profit < 0


def test_buy_and_hold_rejects_singleton():
    with pytest.raises(ValueError):
        buy_and_hold_profit(np.array([10.0]))


def test_seasonal_swing_picks_cheapest_inject_and_dearest_withdraw():
    df = pd.DataFrame({
        "Dates": pd.to_datetime([
            "2021-04-30", "2021-05-31", "2021-12-31", "2022-01-31",
        ]),
        "Prices": [8.0, 9.0, 12.0, 14.0],
    })
    out = seasonal_swing_profit(df, units=1, storage_cost_per_unit_per_month=0)
    assert out["inject_price"] == pytest.approx(8.0)
    assert out["withdraw_price"] == pytest.approx(14.0)
    assert out["months_held"] == 9
    assert out["profit"] == pytest.approx(6.0)


def test_seasonal_swing_charges_storage_correctly():
    df = pd.DataFrame({
        "Dates": pd.to_datetime(["2021-04-30", "2021-12-31"]),
        "Prices": [8.0, 12.0],
    })
    out = seasonal_swing_profit(df, units=10, storage_cost_per_unit_per_month=0.1)
    assert out["months_held"] == 8
    assert out["profit"] == pytest.approx(40.0 - 8.0)


def test_seasonal_swing_requires_inject_before_withdraw():
    df = pd.DataFrame({
        "Dates": pd.to_datetime(["2021-12-31", "2022-04-30"]),
        "Prices": [12.0, 8.0],
    })
    with pytest.raises(ValueError):
        seasonal_swing_profit(df, units=1)


def test_default_storage_cost_is_documented():
    assert DEFAULT_STORAGE_COST == 0.05
