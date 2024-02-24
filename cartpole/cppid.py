"""
PID controller for cart pole
"""

import math
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

    def get_target_value_pole_ang(self, env_state: np.ndarray):
        return 0.0

    def calc_pid(self, env_state: np.ndarray) -> float:
        delta_t = 0.02

        error_pole = self.get_target_value_pole_ang(env_state) - env_state[2]
        error_cart = 0.0 - env_state[4]

        pid_pole = self.controller_pole.step(error_pole, delta_t)
        pid_cart = self.controller_cart.step(error_cart, delta_t)

        return pid_cart + pid_pole

    def step(self, env_state: np.ndarray) -> int:
        pid = self.calc_pid(env_state)
        action = 0 if pid >= 0 else 1
        return action

class PidAgentCont(PidAgent):
    def step(self, env_state: np.ndarray) -> int:
        return -np.clip(self.calc_pid(env_state), -10.0, 10.0)

class PidAgentMoving(PidAgent):
    def __init__(self, 
                 KPID_pole: tuple[float, float, float], 
                 KPID_cart: tuple[float, float, float] = (0, 0, 0), 
                 pos_threshold = 0.2, 
                 ang_degrees=1.0) -> None:
        super().__init__(KPID_pole, KPID_cart)
        self.pos_threshold = pos_threshold
        self.ang_degrees = ang_degrees

    def get_target_value_pole_ang(self, env_state: np.ndarray):
        posdelta = env_state[4]
        if posdelta > self.pos_threshold:
            return math.radians(-self.ang_degrees)
        if posdelta < -self.pos_threshold:
            return math.radians(self.ang_degrees)
        return 0.0
    
class PidAgentMoving2(PidAgent):
    def __init__(self, 
                 KPID_pole: tuple[float, float, float], 
                 KPID_cart: tuple[float, float, float] = (0, 0, 0), 
                 factorA = 1.0) -> None:
        super().__init__(KPID_pole, KPID_cart)
        self.factorA = factorA

    def get_target_value_pole_ang(self, env_state: np.ndarray):
        posdelta = env_state[4]
        return math.radians(-1.0 * self.factorA * posdelta)
