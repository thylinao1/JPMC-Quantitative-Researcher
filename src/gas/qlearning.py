"""Tabular Q-learning for the gas storage environment.

Training and evaluation are deliberately separated so the agent can
be trained on one price series and rolled out on another. That makes
out-of-sample evaluation possible instead of the in-sample number an
agent essentially memorises when trained and tested on the same
series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import GasStorageEnv, N_ACTIONS


@dataclass
class QLearningConfig:
    n_episodes: int = 5000
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 0.3
    epsilon_end: float = 0.05
    seed: int = 42


def train(train_prices, env_kwargs=None, config=None):
    """Train a tabular Q-learning agent on a single price series.

    Returns the Q-table indexed by ``env.state_to_idx(inventory, t)``.
    Epsilon decays linearly from ``epsilon_start`` to ``epsilon_end``
    over the episodes.
    """
    config = config or QLearningConfig()
    env_kwargs = dict(env_kwargs or {})
    env = GasStorageEnv(prices=train_prices, **env_kwargs)
    n_states = env.n_inventory_levels * env.n_periods
    rng = np.random.default_rng(config.seed)

    Q = np.zeros((n_states, N_ACTIONS))
    if config.n_episodes <= 1:
        eps_schedule = np.array([config.epsilon_start])
    else:
        eps_schedule = np.linspace(
            config.epsilon_start, config.epsilon_end, config.n_episodes
        )

    for ep in range(config.n_episodes):
        state = env.reset()
        epsilon = eps_schedule[ep]
        while True:
            s_idx = env.state_to_idx(*state)
            if rng.random() < epsilon:
                action = int(rng.integers(N_ACTIONS))
            else:
                action = int(np.argmax(Q[s_idx]))
            next_state, reward, done, _ = env.step(action)
            if not done:
                next_idx = env.state_to_idx(*next_state)
                Q[s_idx, action] += config.alpha * (
                    reward + config.gamma * np.max(Q[next_idx]) - Q[s_idx, action]
                )
            else:
                Q[s_idx, action] += config.alpha * (reward - Q[s_idx, action])
                break
            state = next_state
    return Q


def evaluate(Q, prices, env_kwargs=None):
    """Roll out the greedy policy of a Q-table on a price series."""
    env_kwargs = dict(env_kwargs or {})
    env = GasStorageEnv(prices=prices, **env_kwargs)
    if Q.shape[0] != env.n_inventory_levels * env.n_periods:
        raise ValueError(
            "Q-table shape does not match evaluation env "
            f"({Q.shape[0]} vs {env.n_inventory_levels * env.n_periods})"
        )
    state = env.reset()
    total_profit = 0.0
    actions = []
    inventories = [state[0]]
    illegal_attempts = 0
    while True:
        s_idx = env.state_to_idx(*state)
        action = int(np.argmax(Q[s_idx]))
        next_state, reward, done, info = env.step(action)
        total_profit += reward
        actions.append(action)
        inventories.append(next_state[0])
        if info["illegal"]:
            illegal_attempts += 1
        if done:
            break
        state = next_state
    return {
        "total_profit": float(total_profit),
        "actions": actions,
        "inventories": inventories,
        "illegal_attempts": illegal_attempts,
    }
