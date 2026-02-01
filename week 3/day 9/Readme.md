# Day 9 — Hyperparameter Tuning (Phase III)

## Objective

The goal of Day 9 is to systematically study the effect of PPO hyperparameters on agent performance **without changing the task**.

This phase does **not** aim to maximize returns.  
Instead, it aims to determine **whether performance is sensitive to optimizer-level choices** once the environment and reward are stable.

In other words:

> We optimize *how* the agent learns, not *what* it learns.

---

## Why Hyperparameter Tuning Is Isolated

Hyperparameter tuning is intentionally delayed until Phase III because:

- Early tuning can hide reward design flaws
- It can compensate for unstable environment dynamics
- It can overfit noise instead of signal

By Day 9:

- Environment dynamics are finalized
- Reward function is validated
- Learning behavior is confirmed (Days 5–8)

Only at this stage does hyperparameter tuning become meaningful.

---

## Experimental Protocol

To preserve experimental validity, **everything except PPO hyperparameters was fixed**.

### Fixed Components
- Trading environment (`TradingEnv`)
- Reward function
- Random seeds
- Training timesteps per trial
- Evaluation protocol
- Policy architecture (MLP)

### Tuned Hyperparameters
- `learning_rate` (log scale)
- `gamma`
- `ent_coef`

Each Optuna trial represents a **controlled experiment** with an identical training and evaluation budget.

---

## Optimization Setup

- Algorithm: Proximal Policy Optimization (PPO)
- Tuning framework: Optuna (TPE sampler, Median pruner)
- Objective metric: Mean evaluation episode reward
- Number of trials: ≥ 20
- Study storage: SQLite (`day9_optuna.db`)

Training reward was **not** optimized directly.  
Only **evaluation performance** was used as the objective.

---

## Results Summary

### Optimization History

The optimization history shows a **flat objective curve** across all trials.

All completed trials achieved **identical evaluation performance**.

This indicates:
- No hyperparameter configuration outperformed others
- PPO converged to the same policy behavior across the tested ranges

### Hyperparameter Importance

Hyperparameter importance analysis (fANOVA) could not be computed because:
- All objective values were identical
- Performance variance across trials was zero

This is not a failure of the tuning process.  
It is a valid outcome indicating **low sensitivity to optimizer parameters**.

---

## Final Model Artifact

Despite the lack of performance differentiation, a **canonical Day-9 model** was produced to complete the experimental pipeline.

Steps:
1. Best hyperparameters were frozen from the Optuna study
2. A single PPO agent was trained using these parameters
3. The trained model was saved as a reusable artifact

## Final Model Artifact

Despite the lack of performance differentiation across hyperparameter configurations, a
canonical Day-9 model was produced to complete the optimization pipeline.

Steps performed:
1. The best hyperparameter configuration was selected from the Optuna study
2. A PPO agent was retrained once using these frozen hyperparameters
3. The trained model was saved as a reusable artifact

Final model file:


This ensures:
- Reproducibility
- Clear separation between tuning and deployment
- A definitive Day-9 output model

---

## File Structure

day9/
├── optuna_study.py
├── day9_optuna.db
├── generate_best_params.py
├── best_params.json
├── train_best_agent.py
├── ppo_trading_agent.zip
├── evaluate_agent.py
├── analyze_optuna.ipynb
├── plot_optuna_results.py
└── optimization_history.html


---

## Key Takeaway

The absence of performance variation across hyperparameter configurations suggests that:

- The environment–reward system is **well-conditioned**
- PPO learning dynamics are **stable**
- Moderate changes in learning rate, discount factor, and entropy regularization do not materially affect outcomes

This result supports the conclusion that **earlier phases successfully removed structural and reward misalignment issues**, leaving little room for optimizer-level gains.

---