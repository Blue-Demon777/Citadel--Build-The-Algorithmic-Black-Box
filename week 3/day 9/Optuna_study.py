import optuna
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from TradingEnv import TradingEnv

# =====================
# GLOBAL CONFIG
# =====================

SEED = 42
TRAIN_TIMESTEPS = 30_000
EVAL_EPISODES = 5

# =====================
# EVALUATION FUNCTION
# =====================

def evaluate(model, seed):
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


# =====================
# OPTUNA OBJECTIVE
# =====================

def objective(trial):

    # --- Sample hyperparameters ---
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-3, log=True
    )

    gamma = trial.suggest_float(
        "gamma", 0.90, 0.999
    )

    ent_coef = trial.suggest_float(
        "ent_coef", 1e-4, 1e-2, log=True
    )

    # --- Training env ---
    train_env = TradingEnv(seed=SEED)
    train_env = Monitor(train_env)

    # --- PPO model ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=2048,
        batch_size=64,
        verbose=0,
        seed=SEED,
    )

    # --- Train ---
    model.learn(total_timesteps=TRAIN_TIMESTEPS)

    # --- Evaluate ---
    score = evaluate(model, seed=SEED)

    train_env.close()

    return score


# =====================
# STUDY RUNNER
# =====================

if __name__ == "__main__":

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="ppo_trading_day9",
        storage="sqlite:///day9_optuna.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=20)

    print("\n=== BEST TRIAL ===")
    print("Value:", study.best_trial.value)
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")
