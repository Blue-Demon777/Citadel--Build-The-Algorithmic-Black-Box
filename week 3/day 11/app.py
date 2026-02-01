import pandas as pd
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dash import Dash, dcc, html

# --------------------------------------------------
# Load data (single source of truth)
# --------------------------------------------------
df = pd.read_csv("data/performance_dataset.csv")

df["timestamp"] = pd.to_numeric(df["timestamp"])

# --------------------------------------------------
# Prepare BUY / SELL subsets
# --------------------------------------------------
buy_df = df[df["action"] == "BUY"]
sell_df = df[df["action"] == "SELL"]

# --------------------------------------------------
# Create figure with 3 linked charts
# --------------------------------------------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.04
)

# -------- Chart 1: Portfolio vs Benchmarks --------
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["agent_portfolio_value"],
        mode="lines",
        name="RL Agent"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["benchmark_buy_hold_value"],
        mode="lines",
        name="Buy & Hold"
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["benchmark_random_value"],
        mode="lines",
        name="Random"
    ),
    row=1, col=1
)

# -------- Chart 2: Price + Actions --------
fig.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["price"],
        mode="lines",
        name="Market Price"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=buy_df["timestamp"],
        y=buy_df["price"],
        mode="markers",
        marker=dict(symbol="triangle-up", size=9),
        name="BUY"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(
        x=sell_df["timestamp"],
        y=sell_df["price"],
        mode="markers",
        marker=dict(symbol="triangle-down", size=9),
        name="SELL"
    ),
    row=2, col=1
)

# -------- Chart 3: PnL Distribution + Gaussian --------
fig.add_trace(
    go.Histogram(
        x=df["pnl_step"],
        nbinsx=50,
        name="PnL Distribution",
        opacity=0.75
    ),
    row=3, col=1
)

# Gaussian overlay
pnl = df["pnl_step"]
x = np.linspace(pnl.min(), pnl.max(), 200)
pdf = (
    np.exp(-0.5 * ((x - pnl.mean()) / pnl.std()) ** 2)
    / (pnl.std() * np.sqrt(2 * np.pi))
)

# scale to histogram height
pdf = pdf * len(pnl) * (pnl.max() - pnl.min()) / 50

fig.add_trace(
    go.Scatter(
        x=x,
        y=pdf,
        mode="lines",
        name="Gaussian Fit"
    ),
    row=3, col=1
)

# --------------------------------------------------
# Layout + interactivity
# --------------------------------------------------
fig.update_layout(
    height=900,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(t=40, b=40)
)

# --------------------------------------------------
# Dash App
# --------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    children=[
        dcc.Graph(figure=fig)
    ]
)

if __name__ == "__main__":
    app.run(debug=False)
