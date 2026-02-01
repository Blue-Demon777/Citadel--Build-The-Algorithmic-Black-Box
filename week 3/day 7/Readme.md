# Day 7 — Market Agents & Strategic Behavior

## Objective

The goal of **Day 7** is to introduce **agent-based behavior** into the event-driven market simulation.

This day answers the question:

**Can different types of trading agents interact realistically inside the same limit order book?**

The focus is on **strategy design and interaction**, not learning or optimization.

---

## What Was Built

A modular agent framework where multiple trader archetypes operate simultaneously in the market.

Each agent:
- Observes the current market state
- Decides an action (market / limit / cancel)
- Interacts asynchronously via the event engine
- Maintains its own inventory and cash state

This establishes **market ecology**, a prerequisite for RL.

---

## Agent Architecture

All agents inherit from a common abstract base:

Agent
├── RandomAgent
├── MarketMakerAgent
├── NoiseTraderAgent
└── MomentumAgent


Each agent implements:

```python
get_action(market_state)
on_trade(trade, side)
```
**Agent Types**
**RandomAgent**

A baseline, non-strategic trader.

Behavior:
- Random BUY / SELL decisions
- Random order type:
-- Market orders
-- Limit orders near mid-price
- Random quantity

Purpose:
- Provide background liquidity
- Serve as a control agent

### NoiseTraderAgent

A zero-intelligence trader anchored to an external **fair value process**.

#### Behavior
- Random BUY / SELL decisions
- Quantity bounded by available cash and inventory
- **70% market orders**
- **30% aggressive limit orders** near fair value

#### Constraints
- Cannot buy beyond available cash
- Cannot sell beyond available inventory

#### Purpose
- Inject stochastic order flow
- Simulate uninformed or retail market participants
- Provide liquidity and randomness to the market

---

### MarketMakerAgent

A liquidity-providing agent responsible for tightening spreads and absorbing order flow.

#### Behavior
- Simultaneously posts bid and ask limit orders
- Quotes centered around mid-price
- Applies **inventory skew** to encourage mean reversion
- Cancels and replaces all active quotes synchronously

#### Key Mechanisms
- Base spread parameter
- Inventory-sensitive quote adjustment
- Maximum inventory constraint

#### Purpose
- Reduce bid-ask spread
- Maintain market stability
- Earn profit from spread capture

---

### MomentumAgent

A trend-following trader based on price momentum.

#### Behavior
- Maintains a rolling window of mid-prices
- Computes Simple Moving Average (SMA)
- BUY if price > SMA
- SELL if price < SMA
- Uses aggressive market orders

#### Constraints
- Budget-limited
- Inventory-limited

#### Purpose
- Introduce directional pressure
- Stress-test market maker inventory management
- Create non-stationary price dynamics

---

## Market State Provided to Agents

Each agent receives a snapshot of the current market state:

{
best_bid,
best_ask,
mid,
l2_snapshot
}


This ensures:
- No access to future information
- No privileged state
- All agents operate under identical information constraints

---

## Event-Driven Interaction

Agents interact with the market via scheduled events:

- `AgentArrivalEvent`
- `OrderSubmissionEvent`
- `SnapshotEvent`
- `MarketCloseEvent`

All interactions are:
- Time-stamped
- Latency-aware
- Processed asynchronously

This preserves causality and prevents unrealistic execution assumptions.

---

## Files Introduced (Day 7)

Day7/
├── agents.py # Agent definitions and strategies
├── actions.py # Market actions (limit, market, cancel)
├── events.py # Event-driven agent interaction
├── README.md # Day-7 documentation


---

## Design Constraints Enforced

- Agents never bypass the order book
- All state changes occur via executed trades
- Cash and inventory are conserved
- No agent sees future prices

Any violation indicates a structural error.

---

## Pass / Fail Criteria

### PASS
- Multiple agent types coexist without runtime errors
- Market maker tightens spreads
- Noise traders generate random order flow
- Momentum traders introduce directional pressure
- Inventory and cash remain bounded

### FAIL
- Negative cash balances
- Inventory explosions
- No trades occurring
- Agents bypassing the event system

Failures must be fixed before reinforcement learning integration.

---

## Why This Matters

Without heterogeneous agents:
- Market dynamics are trivial
- Reinforcement learning agents overfit
- Learning signals are misleading

This agent ecosystem creates:
- Competition
- Noise
- Adversarial pressure

These conditions are required for meaningful PPO training.

---

## Verdict

**Day 7 successfully establishes realistic market behavior.**

The simulator now supports:
- Competing trading strategies
- Inventory-aware decision making
- Directional and stochastic order flow

This completes the **behavioral layer** of the system.

Day 8 builds on this by exposing the market as a Gymnasium-compatible RL environment.
