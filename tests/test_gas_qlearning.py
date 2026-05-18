"""Tests for the Q-learning trainer and evaluator."""

import numpy as np
import pytest

from src.gas.qlearning import QLearningConfig, evaluate, train


SHORT_ENV_KWARGS = {
    "max_inventory": 10,
    "unit_size": 10,
    "storage_cost_per_unit_per_month": 0,
    "terminal_liquidation": True,
}


def test_train_is_deterministic_under_seed():
    prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
    config = QLearningConfig(n_episodes=200, seed=123)
    q1 = train(prices, env_kwargs=SHORT_ENV_KWARGS, config=config)
    q2 = train(prices, env_kwargs=SHORT_ENV_KWARGS, config=config)
    assert np.array_equal(q1, q2)


def test_train_different_seeds_diverge():
    prices = np.array([10.0, 11.0, 12.0, 11.0, 10.0])
    q1 = train(prices, env_kwargs=SHORT_ENV_KWARGS,
               config=QLearningConfig(n_episodes=200, seed=1))
    q2 = train(prices, env_kwargs=SHORT_ENV_KWARGS,
               config=QLearningConfig(n_episodes=200, seed=2))
    assert not np.array_equal(q1, q2)


def test_agent_learns_to_arbitrage_obvious_uptrend():
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    config = QLearningConfig(n_episodes=2000, seed=42)
    Q = train(prices, env_kwargs=SHORT_ENV_KWARGS, config=config)
    out = evaluate(Q, prices, env_kwargs=SHORT_ENV_KWARGS)
    assert out["total_profit"] >= 20.0


def test_evaluate_raises_on_size_mismatch():
    Q = train(np.zeros(5), env_kwargs=SHORT_ENV_KWARGS,
              config=QLearningConfig(n_episodes=10, seed=0))
    with pytest.raises(ValueError):
        evaluate(Q, np.zeros(8), env_kwargs=SHORT_ENV_KWARGS)


def test_evaluate_returns_aligned_action_and_inventory_traces():
    prices = np.array([10.0, 11.0, 12.0])
    Q = train(prices, env_kwargs=SHORT_ENV_KWARGS,
              config=QLearningConfig(n_episodes=50, seed=0))
    out = evaluate(Q, prices, env_kwargs=SHORT_ENV_KWARGS)
    assert len(out["actions"]) == len(prices)
    assert len(out["inventories"]) == len(prices) + 1
