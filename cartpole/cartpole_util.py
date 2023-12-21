"""
Utils for execution of cartpole env
"""

import abc
import typing
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt


class CartPoleAgentABC(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def step(
        self, env_state: np.ndarray, env_reward: float, cartpos_setpoint: float
    ) -> typing.Tuple[int, typing.Dict[str, object]]:
        pass


def execute_cartpole(
    agent: CartPoleAgentABC,
    num_episodes: int = 20,
    num_steps: int = 500,
    inital_state: typing.Optional[np.ndarray] = None,
    cartpos_setpoint: float | np.ndarray = 0.0,
) -> pd.DataFrame:
    env = None
    log = []
    try:
        env = gym.make("CartPole-v1")
        for i_episode in range(num_episodes):
            env_state, env_info = env.reset()
            env_reward = 0
            agent.reset()
            if inital_state is not None:
                env_state = inital_state
                env.state = inital_state

            for t in range(num_steps):
                curr_cartpos_setpoint = cartpos_setpoint if np.isscalar(cartpos_setpoint) else cartpos_setpoint[t]

                action, agent_log_dict = agent.step(env_state, env_reward, curr_cartpos_setpoint)
                env_state, env_reward, terminated, truncated, env_info = env.step(
                    action
                )

                step_log_dict = OrderedDict()
                step_log_dict["ep"] = i_episode
                step_log_dict["t"] = t
                step_log_dict["cart_pos_setpoint"] = curr_cartpos_setpoint
                step_log_dict["cart_pos"] = env_state[0]
                step_log_dict["cart_vel"] = env_state[1]
                step_log_dict["pole_ang"] = env_state[2]
                step_log_dict["pole_vel"] = env_state[3]

                for k, v in agent_log_dict.items():
                    step_log_dict["agent_" + k] = v

                step_log_dict["env_info"] = json.dumps(env_info)

                log.append(step_log_dict)

                if terminated or truncated:
                    break

        return pd.DataFrame(log)
    finally:
        if env is not None:
            env.close()

def render_cartpole_state(state:np.ndarray, pos_setpoint:float):
    # ToDo: Implment own rendering based on matplotlib etc.
    # Should be much faster and can also visualize velocities and pos setpoint

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    env.state = state
    pixels = env.render()

    return plt.imshow(pixels)
