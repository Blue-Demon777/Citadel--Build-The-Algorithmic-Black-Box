import gymnasium as gym
from stable_baselines3 import PPO

from TradingEnv import TradingEnv


def main():
    env = TradingEnv(
        max_steps=200,
        book_depth=5,
        max_inventory=20,
        max_cash=100_000,
        transaction_cost=0.01,
        seed=42,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log="./tb_logs/",
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_trading_agent")


if __name__ == "__main__":
    main()
