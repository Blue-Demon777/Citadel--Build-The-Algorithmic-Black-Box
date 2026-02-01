from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np

from TradingEnv import TradingEnv
env = TradingEnv(seed=42)
env = Monitor(env)   

from stable_baselines3.common.callbacks import BaseCallback
TIME_STEPS = 50_000

class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.entropy_losses = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if "entropy_loss" in self.locals:
            loss = self.locals["entropy_loss"]
            self.episode_losses.append(loss)

        
        return True

logger = EpisodeLogger()

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

model.learn(
    total_timesteps=TIME_STEPS,
    callback=logger
)
import numpy as np

print("\n=== Day 5 Summary ===")
print("Episodes:", len(logger.episode_rewards))
print("Mean reward (first 10):", np.mean(logger.episode_rewards[:10]))
print("Mean reward (last 10):", np.mean(logger.episode_rewards[-10:]))

if logger.entropy_losses:
    print("Entropy (early):", np.mean(logger.entropy_losses[:100]))
    print("Entropy (late):", np.mean(logger.entropy_losses[-100:]))

import os
print("Saving model to:", os.getcwd())
model.save("ppo_day5")