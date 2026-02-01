import random
from TradingEnv import TradingEnv

def evaluate_buy_and_hold(n_episodes, seed):
    runs = []

    for ep in range(n_episodes):
        env = TradingEnv(seed=seed + ep)
        obs, _ = env.reset()
        done = False
        values = []

        # Buy once at t=0
        obs, _, _, _, info = env.step(1)

        while not done:
            obs, _, terminated, truncated, info = env.step(0)
            values.append(info["portfolio_value"])
            done = terminated or truncated

        runs.append(values)

    return runs


def evaluate_random_agent(n_episodes, seed):
    runs = []

    for ep in range(n_episodes):
        env = TradingEnv(seed=seed + ep)
        obs, _ = env.reset()
        done = False
        values = []

        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            values.append(info["portfolio_value"])
            done = terminated or truncated

        runs.append(values)

    return runs
