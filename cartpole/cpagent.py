"""
Utils for execution of cartpole env
"""

import abc
import typing
import numpy as np
import pandas as pd
import gymnasium as gym

import stable_baselines3.common.base_class as sb_classes

import cpenvs




class CartPoleAgentABC(abc.ABC):
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def step(self, env_state: np.ndarray) -> int:
        pass

    def after_step(
        self,
        old_env_state: np.ndarray,
        new_env_state: np.ndarray,
        action: int,
        env_reward: float,
    ) -> None:
        pass

# Use for execution only, not training
class ModelCartPoleAgent(CartPoleAgentABC):
    def __init__(self, model: sb_classes.BaseAlgorithm) -> None:
        self.model = model

    def step(self, env_state: np.ndarray) -> int:
        action, _ = self.model.predict(env_state, deterministic=True)
        return action

def execute_cartpole(
    agent: CartPoleAgentABC,
    env: gym.Env = None,
    num_episodes: int = 20,
    num_steps: int = 500,
) -> pd.DataFrame:
    if env is None:
        env = cpenvs.StandardCartPoleEnv()

    try:
        dfs = []
        for i_episode in range(num_episodes):
            env_state, env_info = env.reset()
            agent.reset()

            for t in range(num_steps):
                action = agent.step(env_state)

                old_env_state = env_state
                env_state, env_reward, terminated, truncated, env_info = env.step(
                    action
                )

                agent.after_step(old_env_state, env_state, action, env_reward)

                if terminated or truncated:
                    break

            df_ep = env.log_to_df().reset_index()
            df_ep.insert(0, "ep", i_episode)
            dfs.append(df_ep)

        return pd.concat(dfs, ignore_index=True)
    finally:
        if env is not None:
            env.close()
