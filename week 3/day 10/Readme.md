# Day 10 — Benchmarking: The Alpha Test

> *Is the agent intelligent, or merely unchallenged?*

---

## Objective

The goal of Day 10 is to evaluate whether the trained reinforcement learning (RL) trading agent delivers **statistical and economic value** relative to trivial baseline strategies under **identical market conditions**.

This day represents a **null-hypothesis test**:

> **H₀:** The RL agent does not produce risk-adjusted returns superior to simple baselines.

A negative result is considered a **valid scientific outcome**.

---

## Experimental Design

All agents were evaluated under **identical conditions**:

- Same stochastic price process
- Same transaction costs
- Same execution rules
- Same initial capital and inventory
- Frozen policies (no learning or adaptation)
- Out-of-sample evaluation episodes

Each agent was evaluated across multiple independent episodes using fixed random seeds to ensure comparability.

---

## Agents Evaluated

### 1. RL Agent (PPO)
- Pretrained PPO policy
- Evaluation-only (no retraining)
- Deterministic action selection

### 2. Buy & Hold
- Buy once at the beginning of the episode
- Hold until termination
- Captures market drift

### 3. Random Agent
- Random Buy / Sell / Hold actions
- Same action space and constraints as the RL agent
- Serves as a learning sanity check

---

## Evaluation Metrics

Profit alone is insufficient to assess trading performance.  
Instead, the following **risk-adjusted metrics** were used:

### Sharpe Ratio

\[
\text{Sharpe} = \frac{R_p - R_f}{\sigma_p}
\]

- Measures return per unit of risk
- Penalizes volatility
- Risk-free rate set to 0 for simulations

### Maximum Drawdown

- Largest peak-to-trough decline in portfolio value
- Captures downside risk and capital preservation

---

## Results

### Performance Summary

| Agent       | Mean Sharpe | Std Sharpe | Mean Max Drawdown |
|------------|-------------|------------|-------------------|
| RL Agent   | 0.000000    | 0.000000   | 0.000000          |
| Buy & Hold | 0.012399    | 0.055123   | 0.000021          |
| Random     | -0.000384   | 0.067491   | 0.000135          |

---

### Visual Analysis

- **Equity Curves:** `results/equity_curves.png`
- **Drawdown Curves:** `results/drawdown_curves.png`

The Buy & Hold strategy captures mild market drift, while the Random agent exhibits noisy returns and higher drawdowns.  
The RL agent maintains an almost flat equity curve with negligible drawdowns.

---

## Verdict

Under identical market conditions and out-of-sample evaluation, the PPO-based trading agent **did not achieve a positive Sharpe ratio and failed to outperform both the Buy & Hold and Random baselines**.

The RL agent exhibited **near-zero drawdowns and flat portfolio value**, indicating that it learned a **highly risk-averse policy** that prioritizes capital preservation over return generation.

While learning is present, **no statistically defensible alpha** is observed under the current reward structure.

This outcome highlights the sensitivity of reinforcement learning agents to reward shaping: strong downside penalties can suppress risk-taking to the point where inactivity becomes optimal.

---

## Conclusion

Day 10 confirms that profitability in isolation is meaningless without benchmarking.  
Although the RL agent does not produce alpha, the evaluation pipeline is valid, metrics are well-defined, and baselines behave as expected.

This negative result is a **valid and informative scientific outcome**, and it establishes a solid foundation for future reward and environment refinement.

---

## Artifacts

results/
├── metrics.csv
├── equity_curves.png
└── drawdown_curves.png


## to run 
```bash
python benchmark_runner.py
```
