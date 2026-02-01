import numpy as np
from stable_baselines3 import PPO
from TradingEnv import TradingEnv

def evaluate_rl_agent(model_path, n_episodes, seed):
    model = PPO.load(model_path)
    results = []

    for ep in range(n_episodes):
        env = TradingEnv(seed=seed + ep)
        obs, _ = env.reset()
        done = False
        portfolio_values = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])
            done = terminated or truncated

        results.append(portfolio_values)

    return results
