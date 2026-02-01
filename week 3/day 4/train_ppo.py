from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import numpy as np

from TradingEnv import TradingEnv
env = TradingEnv(seed=42)
env = Monitor(env)   

from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True

logger = EpisodeLogger()

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
)

model.learn(
    total_timesteps=10_000,
    callback=logger
)
print("\nEpisode reward stats:")
print("Num episodes:", len(logger.episode_rewards))
print("Mean reward:", np.mean(logger.episode_rewards))
print("Std reward:", np.std(logger.episode_rewards))
print("First 5 rewards:", logger.episode_rewards[:5])
print("Last 5 rewards:", logger.episode_rewards[-5:])
