# src/option_pricer/utils/data_processors/iv_surface_plot.py

import plotly.graph_objs as go
import pandas as pd


def plot_iv_surface(surface_df: pd.DataFrame):
    """
    Plot a 3D Implied Volatility surface.

    Parameters
    ----------
    surface_df : pd.DataFrame
        A pivoted DataFrame:
            index   = strike
            columns = expiry (tau)
            values  = implied volatility
    """

    # Ensure  ordering (if data did not pass through iv_surface_builder)
    surface_df = surface_df.sort_index(axis=0)       # sort strikes
    surface_df = surface_df.sort_index(axis=1)       # sort expiries

    strikes = surface_df.index.values
    expiries = surface_df.columns.values
    iv_values = surface_df.values

    fig = go.Figure(
        data=[
            go.Surface(
                x=expiries,
                y=strikes,
                z=iv_values,
                colorscale="Viridis",
                showscale=True,
            )
        ]
    )

    fig.update_layout(
        title="Implied Volatility Surface",
        autosize=True,
        scene=dict(
            xaxis_title="Time to Expiry (Years)",
            yaxis_title="Strike",
            zaxis_title="Implied Volatility",
        ),
        margin=dict(l=20, r=20, b=20, t=60)
    )

    return fig
