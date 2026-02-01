## Directory Structure (day 4 files to verify)

Day4/
├── TradingEnv.py          # Gymnasium trading environment
├── Train_SB3.py           # PPO training script (main entry point)
├── env_sanity_check.py    # Determinism & reset/step validation
├── README.md              # Day-4 documentation and conclusions

## How to Run (Day 4)

Step 1: (Optional) Verify environment determinism
```bash
python env_sanity_check.py
```
## Train the PPO baseline agent 
```bash
python Train_SB3.py
```
## What Train_SB3.py Does

- Initializes the TradingEnv with fixed seed
- Wraps the environment with SB3 Monitor
- Trains a PPO agent for 10,000 timesteps
- Logs episode-level rewards and lengths
- Prints summary statistics after training

## Expected Output

Training logs should show:
- PPO rollout and training statistics
- No NaNs or runtime errors

At the end of training, episode statistics similar to:

Num episodes: ~50
Mean reward: ~ -1.0
Early rewards: ~ -1.3 to -1.5
Late rewards: ~ -0.5 to -0.6

Improvement in episode reward confirms a valid learning signal.

## Interpretation

The PPO agent demonstrates a non-random learning signal, with episode rewards improving
consistently over training. Training remains numerically stable with no policy collapse.

The reward function primarily incentivizes loss avoidance and conservative behavior,
which is appropriate for early validation but may limit selective risk-taking.

## Known Limitations

- No profitability evaluation is performed
- Reward function weakly differentiates small gains vs small losses
- No hyperparameter tuning or architectural changes applied

## Day 4 Conclusion — Learning Validation

A baseline PPO agent was trained for 10,000 timesteps on the trading environment without any hyperparameter tuning. Training completed without numerical instability or runtime errors.

Episode-level rewards showed a clear improvement over time, with mean episode reward improving from approximately −1.3 in early episodes to approximately −0.5 in later episodes. This confirms the presence of a non-random learning signal.

Observed behavior suggests the reward function incentivizes reduction of large losses rather than aggressive profit-seeking, leading the agent to adopt more conservative and stable actions over time.

A key limitation identified is that the reward does not yet strongly differentiate between mildly negative and mildly positive outcomes, which may discourage selective risk-taking.

The next step is to refine the reward function to better balance loss avoidance with opportunity capture while maintaining stability.
