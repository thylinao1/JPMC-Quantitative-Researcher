"""Tests for the GasStorageEnv."""

import numpy as np
import pytest

from src.gas.env import (
    ACTION_HOLD,
    ACTION_INJECT,
    ACTION_WITHDRAW,
    GasStorageEnv,
)


def test_reset_returns_zero_inventory_zero_time():
    env = GasStorageEnv(prices=np.array([10.0, 11.0, 12.0]))
    assert env.reset() == (0, 0)


def test_inject_at_full_capacity_is_illegal():
    env = GasStorageEnv(
        prices=np.array([10.0, 11.0]), max_inventory=10, unit_size=10,
        storage_cost_per_unit_per_month=0, terminal_liquidation=False,
    )
    env.reset()
    state, _, _, info = env.step(ACTION_INJECT)
    assert info["illegal"] is False
    assert state[0] == 10
    state, _, _, info = env.step(ACTION_INJECT)
    assert info["illegal"] is True
    assert info["actual_action"] == ACTION_HOLD
    assert state[0] == 10


def test_withdraw_at_zero_inventory_is_illegal():
    env = GasStorageEnv(
        prices=np.array([10.0, 11.0]), max_inventory=10, unit_size=10,
        storage_cost_per_unit_per_month=0, terminal_liquidation=False,
    )
    env.reset()
    state, _, _, info = env.step(ACTION_WITHDRAW)
    assert info["illegal"] is True
    assert info["actual_action"] == ACTION_HOLD
    assert state[0] == 0


def test_storage_cost_is_charged_per_step():
    env = GasStorageEnv(
        prices=np.array([10.0, 10.0, 10.0]),
        max_inventory=10, unit_size=10,
        storage_cost_per_unit_per_month=0.5,
        terminal_liquidation=False,
    )
    env.reset()
    env.step(ACTION_INJECT)
    _, reward, _, _ = env.step(ACTION_HOLD)
    assert reward == pytest.approx(-5.0)


def test_terminal_liquidation_settles_remaining_inventory():
    env = GasStorageEnv(
        prices=np.array([10.0, 11.0, 12.0]),
        max_inventory=10, unit_size=10,
        storage_cost_per_unit_per_month=0,
    )
    env.reset()
    env.step(ACTION_INJECT)
    env.step(ACTION_HOLD)
    state, reward, done, _ = env.step(ACTION_HOLD)
    assert done is True
    assert reward == pytest.approx(12.0 * 10)
    assert state[0] == 0


def test_state_to_idx_is_injective_over_valid_states():
    env = GasStorageEnv(prices=np.zeros(5), max_inventory=20, unit_size=10)
    seen = set()
    for inv in (0, 10, 20):
        for t in range(env.n_periods):
            idx = env.state_to_idx(inv, t)
            assert idx not in seen
            seen.add(idx)


def test_unknown_action_raises():
    env = GasStorageEnv(prices=np.array([10.0, 11.0]))
    env.reset()
    with pytest.raises(ValueError):
        env.step(action=99)
