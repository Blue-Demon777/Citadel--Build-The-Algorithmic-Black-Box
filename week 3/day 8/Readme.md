# Day 8 — Gymnasium Trading Environment (RL Interface)

## Objective

The goal of **Day 8** is to wrap the previously built limit-order-book market simulator into a **Gymnasium-compatible reinforcement learning environment**.

This day answers a critical question:

**Can the market simulation be exposed as a stable, deterministic RL environment suitable for PPO training?**

This is an infrastructure day — no learning yet, only correctness and interface design.

---

## What Was Built

A custom `TradingEnv` class implementing the **Gymnasium API**, providing:

- Discrete action space
- Continuous observation space
- Risk-aware reward function
- Deterministic resets and transitions
- Compatibility with Stable-Baselines3

This environment internally embeds:
- Order book
- Event-driven market engine
- Inventory & cash accounting
- Reward shaping with drawdown penalty

---

## Environment Specification

### Action Space


| Action | Meaning |
|------|--------|
| 0 | Hold (no action) |
| 1 | Market Buy (1 unit) |
| 2 | Market Sell (1 unit) |

---

### Observation Space

A fixed-length normalized vector:

[ bid_prices,
bid_sizes,
ask_prices,
ask_sizes,
inventory,
cash ]


- Order book depth: configurable (`book_depth`)
- Prices normalized relative to mid-price
- Inventory and cash normalized to bounds


---

## Reward Function

The reward is **risk-aware**, not just PnL-based:

reward =
Δ(portfolio_value)
− transaction_cost
− λ × drawdown

Where:
- `portfolio_value = cash + inventory × mid_price`
- Drawdown is measured from peak equity
- λ (`lambda_risk`) controls risk aversion

This discourages:
- Over-trading
- Large inventory accumulation
- Deep equity drawdowns

---

## Episode Termination Conditions

An episode ends when **any** of the following occurs:

- Maximum number of steps reached
- Inventory exceeds allowed bounds
- Cash balance is depleted

Both `terminated` and `truncated` flags are handled explicitly.

---

## Determinism Guarantees

- Explicit random seeding
- Reproducible resets
- Identical trajectories for identical action sequences

A determinism test script verifies:
- Identical observations
- Identical termination flags
- No hidden stochasticity

---

## Files Introduced (Day 8)

Day8/
├── TradingEnv.py # Gymnasium-compatible trading environment
├── test_determinism.py # Determinism verification script
├── README.md # Day-8 documentation


---

## How to Run

### Determinism Test

From the Day 8 directory:

```bash
python test_determinism.py
```
