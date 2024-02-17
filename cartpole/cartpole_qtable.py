"""
Agent for Reinforcement learning using qtable
"""

import random
import numpy as np
import numpy.random as nprnd
import cpagent


class QTableAgent(cpagent.CartPoleAgentABC):
    def __init__(self, alpha=0.1, epsilon=0.9, gamma=0.9) -> None:
        self.bins = [
            np.array([-0.1, 0.0, 0.1]),  # Cartpos
            np.array([-0.5, 0.0, 0.5]),  # Cartvel
            np.array([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]),  # Poleang
            np.array([-0.6, -0.2, 0.0, 0.2, 0.6]),  # Polevel
            np.array([]), # Pos deviation => ignore
        ]

        self.alpha = alpha  # Learning factor
        self.epsilon = epsilon  # Exploration Factor
        self.gamma = gamma  # Discount factor

        self.init_qtable()

    def init_qtable(self):
        shape = (self.get_state_size(), 2)
        # self.qtable = nprnd.normal(0.1, 0.02, shape)
        self.qtable = np.zeros(shape)

    def idx(self, env_state):
        factor = 1
        result = 0
        for i, b in enumerate(self.bins):
            state_int = np.digitize(env_state[i], b)
            result += factor * state_int
            factor *= b.shape[0] + 1

        return result

    def get_state_size(self) -> int:
        result = 1
        for b in self.bins:
            result = result * (1 + b.shape[0])
        return result

    def reset(self) -> None:
        pass

    def step(self, env_state: np.ndarray) -> int:
        state_idx = self.idx(env_state)
        return self._step_inner(state_idx)

    def _step_inner(self, state_idx: int):
        if random.random() < self.epsilon:
            # Exploration
            return 1 if random.random() > 0.5 else 0

        return np.argmin(self.qtable[state_idx, :])

    def after_step(
        self,
        old_env_state: np.ndarray,
        new_env_state: np.ndarray,
        action: int,
        env_reward: float,
    ) -> None:
        """
        Update Q-Table
        """

        old_state_idx = self.idx(old_env_state)
        new_state_idx = self.idx(new_env_state)
        old_val = self.qtable[old_state_idx, action]

        new_val = old_val + self.alpha * (
            env_reward + self.gamma * np.max(self.qtable[new_state_idx, :])
        )

        self.qtable[old_state_idx, action] = new_val
