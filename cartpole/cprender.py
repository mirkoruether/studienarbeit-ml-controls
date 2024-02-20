"""
Util functions for rendering cartpole state (e.g. in Jupyter Notebook)
"""


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.axes as maxes

from plotly.subplots import make_subplots

import cpenvs

def render_cartpole_state_df(df: pd.DataFrame, t: int, ep:int =None):
    if ep is not None:
        dff = df.loc[(df["ep"] == ep) & (df["t"] == t)]
    else:
        dff = df.loc[df["t"] == t]

    if dff.shape[0] != 1:
        return

    return render_cartpole_state(dff.iloc[0])

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
    pos_setpoint: float = None,
):
    CART_WIDTH = 0.5
    CART_HEIGHT = 0.3
    POLE_LENGTH = 1.0
    POLE_WIDTH = 0.1

    fig, ax = plt.subplots()
    ax:maxes.Axes = ax

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

    if pos_setpoint is not None:
        ax.arrow(pos_setpoint, -0.25, 0, 0.1, width=2e-2)

    ax.axhline(0, linewidth=1, color="grey")

    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=(-1.5, 1.5), ylim=(-0.3, 1.5))

    return fig, ax


def lineplot(df, ep=None, incl_velo=False):
    if ep is not None:
        df = df.loc[df["ep"] == ep]

    fig = make_subplots(rows=4 if incl_velo else 2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

    df = df.copy()
    df["pole_ang_deg"] = np.rad2deg(df["pole_ang"])
    df["pole_vel_deg/s"] = np.rad2deg(df["pole_vel"])

    t = df["t"]

    def trace(name, row):
        fig.add_scatter(
            x=t, y=df[name], name=name, showlegend=False, mode="lines", row=row, col=1
        )
        fig.update_yaxes(row=row, title_text=name)

    trace("cart_pos", 1)
    fig.add_scatter(
        x=t,
        y=df["cart_pos_setpoint"],
        mode="lines",
        name="cart_pos_setpoint",
        showlegend=False,
        row=1,
        col=1,
        line=dict(dash="dash", color="black"),
    )

    if incl_velo:
        trace("cart_vel", 2)
        trace("pole_ang_deg", 3)
        trace("pole_vel_deg/s", 4)
    else:
        trace("pole_ang_deg", 2)

    fig.update_xaxes(rangeslider_visible=True, row=4 if incl_velo else 2)

    if incl_velo:
        fig.update_yaxes(row=1, col=1, range=(-cpenvs._norm[0], cpenvs._norm[0]))
        fig.update_yaxes(row=2, col=1, range=(-cpenvs._norm[1], cpenvs._norm[1]))
        fig.update_yaxes(row=3, col=1, range=(np.rad2deg(-cpenvs._norm[2]), np.rad2deg(cpenvs._norm[2])))
        fig.update_yaxes(row=4, col=1, range=(np.rad2deg(-cpenvs._norm[3]), np.rad2deg(cpenvs._norm[3])))
    else:
        fig.update_yaxes(row=1, col=1, range=(-cpenvs._norm[0], cpenvs._norm[0]))
        fig.update_yaxes(row=2, col=1, range=(np.rad2deg(-cpenvs._norm[2]), np.rad2deg(cpenvs._norm[2])))

    fig.update_layout(
        hovermode="x unified",
        template="none",
        margin=dict(l=90, r=10, t=10, b=10),
        height=1200 if incl_velo else 800,
    )

    #fig.update_layout(title_font_size=16)
    fig.update_xaxes(tickfont_size=18)
    for i in range(4 if incl_velo else 2):
        fig.update_yaxes(row=i+1, col=1, tickfont_size=18, title_font_size=30)

    return fig
