"""
Utils for execution of cartpole env
"""

import math
import abc
import typing
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
import gymnasium as gym

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plotly.subplots import make_subplots


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

            for t in range(num_steps):
                curr_cartpos_setpoint = (
                    cartpos_setpoint
                    if np.isscalar(cartpos_setpoint)
                    else cartpos_setpoint[t]
                )

                action, agent_log_dict = agent.step(
                    env_state, env_reward, curr_cartpos_setpoint
                )
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


def render_cartpole_state(row: pd.Series):
    return _render_cartpole_state(
        row["cart_pos"],
        row["cart_vel"],
        row["pole_ang"],
        row["pole_vel"],
        row["cart_pos_setpoint"],
    )


def _render_cartpole_state(
    cart_pos: float,
    cart_vel: float,
    pole_ang: float,
    pole_vel: float,
    pos_setpoint: float = 0.0,
):
    CART_WIDTH = 0.5
    CART_HEIGHT = 0.3
    POLE_LENGTH = 1.0
    POLE_WIDTH = 0.1

    fig, ax = plt.subplots()

    # Cart
    ax.add_patch(
        mpatches.Rectangle(
            xy=(cart_pos - CART_WIDTH / 2.0, 0.0),
            width=CART_WIDTH,
            height=CART_HEIGHT,
            color="black",
        )
    )

    joint_xy = (cart_pos, CART_HEIGHT * 0.75)

    # Pole left half
    ax.add_patch(
        mpatches.Rectangle(
            xy=joint_xy,
            width=-1.0 * POLE_WIDTH / 2.0,
            height=POLE_LENGTH,
            angle=math.degrees(-1.0 * pole_ang),
            color="brown",
        )
    )

    # Pole right half
    ax.add_patch(
        mpatches.Rectangle(
            xy=joint_xy,
            width=1.0 * POLE_WIDTH / 2.0,
            height=POLE_LENGTH,
            angle=math.degrees(-1.0 * pole_ang),
            color="brown",
        )
    )

    # Joint
    ax.add_patch(mpatches.Circle(xy=joint_xy, radius=POLE_WIDTH * 0.6, color="grey"))

    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(-1, 1), ylim=(-0.1, 1.5))

    return fig, ax


def lineplot(df, ep=None):
    if ep is not None:
        df = df.loc[df["ep"] == ep]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    t = df["t"]

    def trace(name, row):
        fig.add_scatter(x=t, y=df[name], name=name, showlegend=False, row=row, col=1)
        fig.update_yaxes(row=row, title_text=name)

    trace("cart_pos", 1)
    trace("cart_vel", 2)
    trace("pole_ang", 3)
    trace("pole_vel", 4)

    fig.update_xaxes(rangeslider_visible=True, row=4)
    fig.update_layout(
        hovermode="x unified",
        template="none",
        margin=dict(l=60, r=10, t=10, b=10),
        height=700,
    )

    return fig
