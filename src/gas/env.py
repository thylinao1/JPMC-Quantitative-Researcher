"""Gas storage environment for tabular RL.

A finite-horizon environment: monthly periods, discrete inventory
levels in steps of ``unit_size``, three actions (hold, inject,
withdraw). Storage cost units agree with :mod:`src.gas.baselines`
(per unit per month, charged once per step).

Illegal actions become ``hold`` and a flag is returned; the agent
cannot silently "withdraw" from an empty store or "inject" into a
full one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baselines import DEFAULT_STORAGE_COST


ACTION_HOLD = 0
ACTION_INJECT = 1
ACTION_WITHDRAW = 2
N_ACTIONS = 3


@dataclass
class GasStorageEnv:
    prices: np.ndarray
    max_inventory: int = 100
    unit_size: int = 10
    storage_cost_per_unit_per_month: float = DEFAULT_STORAGE_COST
    terminal_liquidation: bool = True
    inventory: int = field(init=False, default=0)
    t: int = field(init=False, default=0)

    def __post_init__(self):
        self.prices = np.asarray(self.prices, dtype=float)
        if self.max_inventory % self.unit_size != 0:
            raise ValueError("max_inventory must be a multiple of unit_size")
        self.n_periods = len(self.prices)
        self.n_inventory_levels = self.max_inventory // self.unit_size + 1

    def reset(self):
        self.inventory = 0
        self.t = 0
        return (self.inventory, self.t)

    def step(self, action):
        if action not in (ACTION_HOLD, ACTION_INJECT, ACTION_WITHDRAW):
            raise ValueError(f"unknown action: {action}")

        price = float(self.prices[self.t])
        info = {"illegal": False, "actual_action": action}
        reward = -self.inventory * self.storage_cost_per_unit_per_month

        if action == ACTION_INJECT:
            if self.inventory + self.unit_size > self.max_inventory:
                info["illegal"] = True
                info["actual_action"] = ACTION_HOLD
            else:
                self.inventory += self.unit_size
                reward -= price * self.unit_size
        elif action == ACTION_WITHDRAW:
            if self.inventory - self.unit_size < 0:
                info["illegal"] = True
                info["actual_action"] = ACTION_HOLD
            else:
                self.inventory -= self.unit_size
                reward += price * self.unit_size

        self.t += 1
        done = self.t >= self.n_periods
        if done and self.terminal_liquidation and self.inventory > 0:
            reward += self.inventory * float(self.prices[-1])
            self.inventory = 0

        return (self.inventory, self.t), float(reward), done, info

    def state_to_idx(self, inventory, t):
        inv_idx = inventory // self.unit_size
        return int(inv_idx * self.n_periods + t)
