"""
Util functions for rendering cartpole state (e.g. in Jupyter Notebook)
"""



import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plotly.subplots import make_subplots


def render_cartpole_state(row: pd.Series):
    return _render_cartpole_state(
        row["cart_pos"],
        row["cart_vel"],
        row["pole_ang"],
        row["pole_vel"],
        row["cart_pos_setpoint"] if ("cart_pos_setpoint" in row) else row["cart_pos"] - row["pos_deviation"],
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
