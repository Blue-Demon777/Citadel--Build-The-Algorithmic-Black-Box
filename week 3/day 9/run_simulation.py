import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm
from statsmodels.tsa.stattools import acf

from agents import NoiseTraderAgent, MarketMakerAgent, MomentumAgent
from fair_value import FairValueProcess
from order_book import OrderBook
from engine import MarketEngine
from environment import MarketEnvironment
from logger import Logger
from market_config import MarketConfig
from events import (
    AgentArrivalEvent,
    MarketCloseEvent,
    SnapshotEvent,
    FairValueUpdateEvent,
)

# -------------------------------
# Simulation
# -------------------------------

def run_simulation(seed=42, horizon=1000):
    random.seed(seed)
    np.random.seed(seed)

    book = OrderBook()
    logger = Logger()
    engine = MarketEngine(book, logger)
    env = MarketEnvironment(engine, MarketConfig(snapshot_interval=1.0))

    fv = FairValueProcess(initial_value=100.0, sigma=0.5, seed=seed)

    agents = [
        NoiseTraderAgent("N1", fv, arrival_rate=1.2),
        NoiseTraderAgent("N2", fv, arrival_rate=1.2),
        NoiseTraderAgent("N3", fv, arrival_rate=1.2),
        MarketMakerAgent("MM1", arrival_rate=0.5),
        MomentumAgent("M1", window=50, arrival_rate=0.8),
        MomentumAgent("M2", window=50, arrival_rate=0.8),
    ]

    for agent in agents:
        engine.agents[agent.agent_id] = agent
        t0 = agent.next_event_time(0)
        engine.schedule(AgentArrivalEvent(t0, agent, env))

    engine.schedule(SnapshotEvent(0, env))
    engine.schedule(FairValueUpdateEvent(0, fv, dt=1.0))
    engine.schedule(MarketCloseEvent(horizon))

    engine.run()
    return logger


# -------------------------------
# Stylized Facts Analysis
# -------------------------------

def analyze_stylized_facts(logger):
    l1 = logger.l1_df()
    assert not l1.empty, "No L1 data recorded"

    mid = l1["mid"].values
    returns = np.diff(np.log(mid))
    returns = returns[returns != 0]

    # ---- Plot 1: Return time series ----
    plt.figure(figsize=(10, 4))
    plt.plot(returns)
    plt.title("Log Returns Over Time")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.tight_layout()
    plt.show()

    # ---- Plot 2: ACF of absolute returns ----
    abs_returns = np.abs(returns)
    acf_vals = acf(abs_returns, nlags=20, fft=True)

    plt.figure(figsize=(6, 4))
    plt.stem(
        range(1, len(acf_vals)),   # skip lag 0
        acf_vals[1:],
    )
    plt.title("ACF of Absolute Returns")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.tight_layout()
    plt.show()

    # ---- Plot 3: Histogram vs Gaussian ----
    mu, sigma = returns.mean(), returns.std()
    x = np.linspace(returns.min(), returns.max(), 500)

    plt.figure(figsize=(6, 4))
    plt.hist(returns, bins=40, density=True, alpha=0.6, label="Empirical")
    plt.plot(x, norm.pdf(x, mu, sigma), "r--", label="Gaussian")
    plt.title("Return Distribution vs Gaussian")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Kurtosis ----
    k = kurtosis(returns, fisher=False)
    print(f"Empirical Kurtosis: {k:.2f}")

# Herding Analysis
def analyze_herding(logger, window=30):
    inv = logger.inventory_df()
    l1 = logger.l1_df().set_index("time")

    # focus on momentum agents only
    momentum_ids = ["M1", "M2"]
    inv = inv[inv.agent.isin(momentum_ids)]

    # pivot → time × agent matrix
    inv_mat = inv.pivot(index="time", columns="agent", values="inventory").fillna(0)

    # rolling correlation
    rolling_corr = (
        inv_mat["M1"]
        .rolling(window)
        .corr(inv_mat["M2"])
    )

    # volatility for alignment
    returns = np.diff(np.log(l1["mid"]))
    vol = pd.Series(returns).rolling(window).std()

    # ----- Plot -----
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(l1["mid"])
    ax[0].set_title("Mid Price")

    ax[1].plot(vol)
    ax[1].set_title("Rolling Volatility")

    ax[2].plot(rolling_corr)
    ax[2].set_title("Momentum Agent Position Correlation")

    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    logger = run_simulation()
    analyze_stylized_facts(logger)
    analyze_herding(logger)
