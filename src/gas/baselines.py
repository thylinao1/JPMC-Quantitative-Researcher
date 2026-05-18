"""Baseline trading strategies for the gas storage problem.

All baselines and the RL agent share the same per-month storage cost
convention so that the comparison between them is on equal footing:

    storage cost per period = units_held * storage_cost_per_unit_per_month

charged once per month for every month the units are in storage.
This is the single source of truth used by
:class:`src.gas.env.GasStorageEnv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_STORAGE_COST = 0.05  # $/unit/month, applied per month held


def buy_and_hold_profit(prices, units=10, storage_cost_per_unit_per_month=DEFAULT_STORAGE_COST):
    """Profit from buying at the first price and selling at the last.

    Storage cost is charged for every month the units sit in
    inventory: that is ``len(prices) - 1`` months.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        raise ValueError("need at least two price observations")
    revenue = (prices[-1] - prices[0]) * units
    months_held = len(prices) - 1
    storage = units * storage_cost_per_unit_per_month * months_held
    return float(revenue - storage)


def seasonal_swing_profit(
    prices_df,
    units=10,
    inject_months=(4, 5, 6, 7, 8, 9),
    withdraw_months=(11, 12, 1, 2, 3),
    storage_cost_per_unit_per_month=DEFAULT_STORAGE_COST,
):
    """One-cycle seasonal swing using calendar-month definitions.

    Defaults to the conventional Northern Hemisphere convention:
    injection season is April through September (gas is cheap during
    cooling season), withdrawal season is November through March
    (gas is dear during heating season). The returned trade is the
    inject-then-withdraw pair that maximises profit after carrying
    cost.
    """
    if "Dates" not in prices_df.columns or "Prices" not in prices_df.columns:
        raise ValueError("prices_df needs Dates and Prices columns")
    df = prices_df.copy()
    df["Dates"] = pd.to_datetime(df["Dates"])
    df = df.sort_values("Dates").reset_index(drop=True)
    df["month"] = df["Dates"].dt.month

    inj = df[df["month"].isin(inject_months)]
    wd = df[df["month"].isin(withdraw_months)]
    if inj.empty or wd.empty:
        raise ValueError("no qualifying inject or withdraw months in data")

    best = {"profit": float("-inf")}
    for _, ir in inj.iterrows():
        wd_later = wd[wd["Dates"] > ir["Dates"]]
        if wd_later.empty:
            continue
        for _, wr in wd_later.iterrows():
            months_held = (
                (wr["Dates"].year - ir["Dates"].year) * 12
                + (wr["Dates"].month - ir["Dates"].month)
            )
            revenue = (wr["Prices"] - ir["Prices"]) * units
            storage = units * storage_cost_per_unit_per_month * months_held
            profit = revenue - storage
            if profit > best["profit"]:
                best = {
                    "profit": float(profit),
                    "inject_date": ir["Dates"],
                    "inject_price": float(ir["Prices"]),
                    "withdraw_date": wr["Dates"],
                    "withdraw_price": float(wr["Prices"]),
                    "months_held": int(months_held),
                }
    if best["profit"] == float("-inf"):
        raise ValueError("no valid inject-then-withdraw pair found")
    return best
