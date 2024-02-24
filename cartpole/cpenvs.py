"""
Modified cartpole with changing setpoint for cart position
"""

import abc
import math
from math import nan
import random
from typing import Any, SupportsFloat
import numpy as np
import pandas as pd
import gymnasium as gym


# Taken from gymnasium implementation
_gravity = 9.8
_masscart = 1.0
_masspole = 0.1
_total_mass = _masscart + _masspole
_length = 0.5  # actually half the pole's length
_polemass_length = _masspole * _length
_force_mag = 10.0
_tau = 0.02  # seconds between state updates
_kinematics_integrator = "euler"

# Angle at which to fail the episode
_theta_threshold_radians = 15 * 2 * math.pi / 360
_x_threshold = 2.4


_high = np.array([
    _x_threshold * 2,
    8.0, # Cart speed
    _theta_threshold_radians * 2,
    4.0, # Pole angle velocity
],dtype=np.float32)
_high2 = np.concatenate([_high, _high[0:1]]) # Reuse cart pos limit for position deviation

_norm = np.array([
    1.6,
    4.0, # Cart speed
    10 * 2 * math.pi / 360,
    2.0, # Pole angle velocity
],dtype=np.float32)
_norm2 = np.concatenate([_norm, _norm[0:1]])

def normalize_state(state:np.ndarray):
    if state.shape[0] == 4:
        return state / _norm
    elif state.shape[0] == 5:
        return state / _norm2
    
def denormalize_state(norm_state: np.ndarray):
    if norm_state.shape[0] == 4:
        return norm_state * _norm
    elif norm_state.shape[0] == 5:
        return norm_state * _norm2

def cartpole_simstep(state:np.ndarray, force:float, disturbance_thetaacc:float=0.0):
    """
    Taken from gymnasium implementation
    """

    x, x_dot, theta, theta_dot = tuple(state)
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    if isinstance(force, np.ndarray):
        force = float(force[0])

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (
        force + _polemass_length * theta_dot**2 * sintheta
    ) / _total_mass
    thetaacc = (_gravity * sintheta - costheta * temp) / (
        _length * (4.0 / 3.0 - _masspole * costheta**2 / _total_mass)
    )
    #####
    # Add turbulence
    #####
    thetaacc = thetaacc + disturbance_thetaacc
    #####

    xacc = temp - _polemass_length * thetaacc * costheta / _total_mass

    if _kinematics_integrator == "euler":
        x = x + _tau * x_dot
        x_dot = x_dot + _tau * xacc
        theta = theta + _tau * theta_dot
        theta_dot = theta_dot + _tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + _tau * xacc
        x = x + _tau * x_dot
        theta_dot = theta_dot + _tau * thetaacc
        theta = theta + _tau * theta_dot

    return np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

def check_terminate(state):
    x, x_dot, theta, theta_dot = tuple(state)
    return bool(
            x < -_x_threshold
            or x > _x_threshold
            or theta < -_theta_threshold_radians
            or theta > _theta_threshold_radians
        )

class _CartPoleCommon(gym.Env, abc.ABC):
    innerstate: np.ndarray
    t: int = 0
    max_episode_t:int = 500
    log: np.ndarray = None

    # Gives the agent a normalized state vector (values ca. in [-1;1])
    use_normalized_state:bool = False

    def __init__(self, use_normalized_state:bool=False) -> None:
        self.use_normalized_state = use_normalized_state
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-_high2, _high2, dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)  # Sets seed

        self.innerstate = self.np_random.uniform(-0.05, 0.05, size=(4,))
        self.t = 0
        self.log = np.zeros((501, 6))

        state, reward = self.calc_state_and_reward(self.innerstate, nan)

        self.log[self.t, :] = np.concatenate([state, np.array([reward])])

        if self.use_normalized_state:
            return np.array(normalize_state(state), dtype=np.float32), {}
        else:
            return np.array(state, dtype=np.float32), {}

    @abc.abstractmethod
    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        pass

    def perform_simulation_step(self, action):
        force = 10.0 if action == 1 else -10.0
        self.innerstate = cartpole_simstep(self.innerstate, force)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.t = self.t + 1
        self.perform_simulation_step(action)

        terminated = check_terminate(self.innerstate)
        inner_reward = 1.0 if not terminated else 0.0
        truncated = self.t >= self.max_episode_t

        state, reward = self.calc_state_and_reward(self.innerstate, inner_reward)

        self.log[self.t, :] = np.concatenate([state, np.array([reward])])

        if self.use_normalized_state:
            return np.array(normalize_state(state), dtype=np.float32), reward, terminated, truncated, {}
        return np.array(state, dtype=np.float32), reward, terminated, truncated, {}

    def render(self) -> None:
        return None

    def close(self):
        pass

    def log_to_df(self):
        df = pd.DataFrame(
            self.log[: self.t + 1],
            columns=[
                "cart_pos",
                "cart_vel",
                "pole_ang",
                "pole_vel",
                "pos_deviation",
                "reward",
            ],
        )
        df.index.name = "t"
        df["cart_pos_setpoint"] = df["cart_pos"] - df["pos_deviation"]
        return df


class StandardCartPoleEnv(_CartPoleCommon):
    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        return np.concatenate([inner_state, np.array([inner_state[0]])]), inner_reward
    
class StandardCartPoleEnvCont(StandardCartPoleEnv):
    def __init__(self, use_normalized_state: bool = False) -> None:
        super().__init__(use_normalized_state)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def perform_simulation_step(self, action):
        force = np.clip(10.0 * action, -10.0, 10.0)
        self.innerstate = cartpole_simstep(self.innerstate, force)

class RiskyCartPoleEnv(_CartPoleCommon):
    def calc_state_and_reward(
        self, inner_state: np.ndarray, inner_reward: np.ndarray
    ) -> tuple[np.ndarray, float]:
        
        angle = abs(inner_state[2])
        desired_angle = 0.04
        difference = abs(angle - desired_angle)

        angle_reward = max(0, 1.0 - difference/0.04)

        reward = 0.3 * inner_reward + 0.7 * angle_reward

        return np.concatenate([inner_state, np.array([inner_state[0]])]), reward


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

        return np.concatenate([inner_state, np.array([distance])]), (
            dist_reward + inner_reward
        ) / 2.0

    def calc_distance(self, inner_state: np.ndarray):
        return inner_state[0] - self.setpoints[self.t]

    def generate_setpoints(self):
        sp = np.zeros((501,))

        # t1 = random.randint(80, 120)  # Around t=100
        # t2 = random.randint(230, 270)  # Around t=250
        # t3 = random.randint(380, 420)  # Around t=400

        direction1 = random.choice([-1.0, 1.0])
        # direction2 = -1.0 * direction1

        len1 = random.normalvariate(0.75, 0.1)
        # len2 = random.normalvariate(0.75, 0.2)

        # sp[t1:t2] = direction1 * len1
        # sp[t2:t3] = direction2 * len2
        # sp[t3:] = 0.0

        sp[100:] = direction1 * len1

        self.setpoints = sp

class MovingCartpoleEnvEnvCont(MovingCartpoleEnv):
    def __init__(self, use_normalized_state: bool = False) -> None:
        super().__init__(use_normalized_state)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def perform_simulation_step(self, action):
        force = np.clip(10.0 * action, -10.0, 10.0)
        self.innerstate = cartpole_simstep(self.innerstate, force)
