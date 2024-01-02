"""
Modified cartpole with changing setpoint for cart position
"""

import abc
from math import nan
import random
from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym


class _CartPoleCommon(gym.Env, abc.ABC):
    inner: gym.Env = None
    t: int = 0
    log: np.ndarray = None

    def __init__(self) -> None:
        self.inner = gym.make("CartPole-v1")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)  # Sets seed
        inner_state, info = self.inner.reset()
        self.t = 0
        self.log = np.zeros((501, 5))

        state, reward = self.calc_state_and_reward(inner_state, nan)

        self.log[self.t, :] = np.hstack(state, reward)

        return state, info

    @abc.abstractmethod
    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        pass

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        inner_state, inner_reward, terminated, truncated, info = self.inner.step(action)
        self.t = self.t + 1

        state, reward = self.calc_state_and_reward(inner_state, inner_reward)

        self.log[self.t, :] = np.hstack(state, reward)

        return state, reward, terminated, truncated, info

    def render(self) -> None:
        return None

    def close(self):
        if self.inner is not None:
            self.inner.close()
            self.inner = None


class StandardCartPoleEnv(_CartPoleCommon):
    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        return np.concatenate([inner_state, inner_state[0]]), inner_reward


class MovingCartpoleEnv(_CartPoleCommon):
    setpoints: np.ndarray = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.generate_setpoints()
        return super().reset(seed=seed, options=options)

    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        distance = self.calc_distance(inner_state)
        dist_reward = max(0.0, 1.0 - (distance * distance))

        return np.concatenate([inner_state, distance]), (
            dist_reward + inner_reward
        ) / 2.0

    def calc_distance(self, inner_state: np.ndarray):
        return inner_state[0] - self.setpoints[self.t]

    def generate_setpoints(self):
        sp = np.zeros((500,))

        t1 = random.randint(50, 150)  # Around t=100
        t2 = random.randint(200, 300)  # Around t=250
        t3 = random.randint(350, 450)  # Around t=400

        direction1 = random.choice([-1.0, 1.0])
        direction2 = -1.0 * direction1

        len1 = random.normalvariate(0.75, 0.2)
        len2 = random.normalvariate(0.75, 0.2)

        sp[t1:t2] = direction1 * len1
        sp[t2:t3] = direction2 * len2
        sp[t3:] = 0.0

        self.setpoints = sp
