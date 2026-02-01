import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from TradingEnv import TradingEnv

SEED = 42
EVAL_EPISODES = 5


def evaluate(model, seed=SEED):
    rewards = []

    for ep in range(EVAL_EPISODES):
        env = TradingEnv(seed=seed + ep)
        env = Monitor(env)

        obs, _ = env.reset()
        done, truncated = False, False
        ep_reward = 0.0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        env.close()

    return float(np.mean(rewards))


if __name__ == "__main__":
    model = PPO.load("ppo_trading_agent")  # or tuned model
    score = evaluate(model)
    print("Evaluation score:", score)
