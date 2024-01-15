"""
PID controller for cart pole
"""

import numpy as np
import pandas as pd

import cpagent


class PidController:
    def __init__(self, KP: float, KI: float, KD: float) -> None:
        self.KP = KP
        self.KI = KI
        self.KD = KD

    def reset(self) -> None:
        self.error_sum = 0.0
        self.error_diff = 0.0
        self.prev_error = 0.0

    def step(self, error: float, delta_t: float = 1.0) -> float:
        self.error_sum += error * delta_t
        self.error_diff = (error - self.prev_error) / delta_t

        pid = self.KP * error + self.KI * self.error_sum + self.KD * self.error_diff

        self.prev_error = error

        return pid


class PidAgent(cpagent.CartPoleAgentABC):
    def __init__(
        self,
        KPID_pole: tuple[float, float, float],
        KPID_cart: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.controller_pole = PidController(*KPID_pole)
        self.controller_cart = PidController(*KPID_cart)

    def reset(self) -> None:
        self.controller_pole.reset()
        self.controller_cart.reset()

    def step(self, env_state: np.ndarray) -> int:
        delta_t = 1.0

        error_pole = 0.0 - env_state[2]
        error_cart = 0.0 - env_state[4]

        pid_pole = self.controller_pole.step(error_pole, delta_t)
        pid_cart = self.controller_cart.step(error_cart, delta_t)

        pid = pid_cart + pid_pole

        action = 0 if pid >= 0 else 1

        return action
