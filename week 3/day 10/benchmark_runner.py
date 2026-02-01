import os
import pandas as pd

from evaluate_rl_agent import evaluate_rl_agent
from evaluate_baselines import (
    evaluate_buy_and_hold,
    evaluate_random_agent,
)
from metrics import aggregate_metrics
from plots import plot_equity_curves, plot_drawdown_curves

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    N_EPISODES = 10
    SEED = 42
    MODEL_PATH = "ppo_trading_agent"

    results = {
        "RL Agent": evaluate_rl_agent(MODEL_PATH, N_EPISODES, SEED),
        "Buy & Hold": evaluate_buy_and_hold(N_EPISODES, SEED),
        "Random": evaluate_random_agent(N_EPISODES, SEED),
    }

    metrics = {}
    for agent, runs in results.items():
        metrics[agent] = aggregate_metrics(runs)

    df = pd.DataFrame(metrics).T
    df.to_csv(f"{RESULTS_DIR}/metrics.csv")

    plot_equity_curves(results, f"{RESULTS_DIR}/equity_curves.png")
    plot_drawdown_curves(results, f"{RESULTS_DIR}/drawdown_curves.png")

    print("\n=== Day 10 Results ===")
    print(df)

if __name__ == "__main__":
    main()
