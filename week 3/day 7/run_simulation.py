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


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    logger = run_simulation()
    analyze_stylized_facts(logger)
