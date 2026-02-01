import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from TradingEnv import TradingEnv

SEED = 42
TRAIN_TIMESTEPS = 50_000  # same scale as Day 5


def main():
    # Load best hyperparameters
    with open("best_params.json", "r") as f:
        best = json.load(f)

    params = best["params"]

    env = TradingEnv(seed=SEED)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        ent_coef=params["ent_coef"],
        n_steps=2048,
        batch_size=64,
        seed=SEED,
        verbose=1,
    )

    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    model.save("ppo_trading_agent")
    print(" Saved ppo_trading_agent.zip")

    env.close()


if __name__ == "__main__":
    main()
