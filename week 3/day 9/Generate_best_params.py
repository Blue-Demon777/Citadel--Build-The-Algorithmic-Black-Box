import json
import optuna

study = optuna.load_study(
    study_name="ppo_trading_day9",
    storage="sqlite:///day9_optuna.db"
)

best = {
    "value": study.best_trial.value,
    "params": study.best_trial.params,
    "trial_number": study.best_trial.number,
}

with open("best_params.json", "w") as f:
    json.dump(best, f, indent=4)
