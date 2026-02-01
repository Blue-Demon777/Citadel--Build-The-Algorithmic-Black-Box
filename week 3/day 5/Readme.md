# Day 5 — Sanity Check Training Run (Learnability Stress Test)

## Objective

Verify whether the trading agent continues to learn under sustained optimization.
This day is a stress test for learnability, not profitability or strategy quality.

The core question addressed is:

**Does the agent actually learn when trained for longer, or does learning collapse?**

---

## Activity

A prolonged PPO training run was executed to expose:
- Reward noise masquerading as learning
- Initialization effects
- Structural flaws in environment or reward design

---

## Training Configuration

- Algorithm: Proximal Policy Optimization (PPO)
- Policy: MLP (default Stable-Baselines3)
- Timesteps: **50,000**
- Environment: unchanged from Day 4
- Reward: risk-aware (designed in Day 3)
- Hyperparameter tuning: **none**

Consistency with Day 4 was intentionally preserved.

---

## How to Run (Day 5)

From the Day 5 directory:

```bash
python train_ppo.py

```
---

## Directory Structure (Day 5)

Day5/
├── TradingEnv.py        # Gymnasium trading environment (unchanged from Day 4)
├── train_ppo.py         # PPO training script (50k timestep run)
├── README.md            # Day-5 documentation and verdict

---

## What `train_ppo.py` Does

- Initializes `TradingEnv` with a fixed seed
- Wraps the environment with SB3 `Monitor`
- Trains a PPO agent for **50,000 timesteps**
- Logs:
  - Episode-level rewards
  - Policy entropy (training signal)
- Prints a concise Day-5 summary at the end of training

---

---

## Directory Structure (Day 5)

Day5/
├── TradingEnv.py        # Gymnasium trading environment (unchanged from Day 4)
├── train_ppo.py         # PPO training script (50k timestep run)
├── README.md            # Day-5 documentation and verdict

---

## What `train_ppo.py` Does

- Initializes `TradingEnv` with a fixed seed
- Wraps the environment with SB3 `Monitor`
- Trains a PPO agent for **50,000 timesteps**
- Logs:
  - Episode-level rewards
  - Policy entropy (training signal)
- Prints a concise Day-5 summary at the end of training

---

## Directory Structure (Day 5)

Day5/
├── TradingEnv.py        # Gymnasium trading environment (unchanged from Day 4)
├── train_ppo.py         # PPO training script (50k timestep run)
├── README.md            # Day-5 documentation and verdict

---

## What `train_ppo.py` Does

- Initializes `TradingEnv` with a fixed seed
- Wraps the environment with SB3 `Monitor`
- Trains a PPO agent for **50,000 timesteps**
- Logs:
  - Episode-level rewards
  - Policy entropy (training signal)
- Prints a concise Day-5 summary at the end of training

---

## Expected Output

During training:
- PPO rollout and training statistics
- No NaNs or runtime errors
- Gradual changes in reward and entropy signals

After training, a summary similar to:

## === Day 5 Summary ===
Episodes: ~250
Mean reward (first 10): ~ -1.3
Mean reward (last 10): ~ 0.0
Entropy (early): higher magnitude
Entropy (late): lower magnitude

---


## Pass / Fail Criteria

### PASS
- Mean episode reward trends upward or stabilizes at a higher level
- Policy entropy decreases gradually
- Behavior changes meaningfully without collapsing

### FAIL
- Mean reward flat or erratic
- Entropy unchanged or collapses early
- Degenerate behavior (always hold / always trade)

If **FAIL**, return to:
- Reward engineering (Day 3)
- Environment dynamics (Day 2)

Do **not** tune hyperparameters to compensate for structural issues.

---

## Administrative Notes

- This run is a **baseline reference** for future changes
- Do not rerun Day 5 unless the environment or reward is modified
- Results here justify proceeding to reward refinement

---
