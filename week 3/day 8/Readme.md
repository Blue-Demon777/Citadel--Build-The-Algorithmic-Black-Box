# Day 6 — Scale-Up, Data Collection & Order Flow Visualization

## Objective

Run a multi-agent market simulation and instrument it for **full observability** in order to study emergent market microstructure dynamics.

This day focuses on **observation, not optimization**.  
All agent policies are frozen, and no learning occurs during the simulation.

---

## Motivation

Single-agent environments are controlled systems.

Multi-agent environments are **complex adaptive systems** where:

- Order flow interacts with liquidity
- Liquidity provision affects volatility
- Agent behavior creates feedback loops

The goal of Day 6 is to observe these dynamics under realistic market mechanics.

---

## Simulation Setup

### Agent Population

The simulation includes heterogeneous agents:

- **1 PPO Agent**
  - Uses a frozen policy trained in Day 5 (`ppo_day5.zip`)
  - Acts deterministically (no exploration)
  - Inventory and cash constrained

- **Noise Traders**
  - Randomized buy/sell behavior
  - Bounded inventory
  - Inject stochastic order flow

- **Market Makers**
  - Continuously quote bid/ask prices around mid-price
  - Provide liquidity and earn spread
  - Inventory constrained

- **Momentum Agent**
  - Trades based on recent price movement
  - Adds directional pressure to the market

No individual agent is “intelligent” in isolation.  
Market structure emerges from their interaction.

---

## Simulation Parameters

- Total timesteps: **5,000**
- Tick size: fixed
- Lot size: fixed
- Inventory bounds: enforced for all agents
- Learning: **disabled**
- Policy updates: **none**

This is a **measurement run**, not a training run.

---

## How to Run (Day 6)

From the Day 6 directory:

```bash
python run_simulation.py
```
### This script:

- Loads the frozen PPO policy
- Initializes the multi-agent population
- Runs the event-driven market simulation
- Logs trades, order book snapshots, and inventory
- Generates a consolidated market report (PDF)

## Date Logging & Observability
The simulation records full market microstructure data, including :
### Trade Data :
- Trade price
- Trade quantity
- Buyer and seller identifiers
### Orderbook Data :

- Best bid and ask (L1)
- Bid–ask spread
- Mid-price
- Depth snapshots (L2)
### Agent State:
- Inventory over time 
- Agent participation in Trades

All Data is persisted and can be reconstructed offline for analysis

## Visualization 
An orderbook heatmap and market summary visualizations are generated to make liquidity and price dynamics visible 
### Expected Characterstics
- Liquidity walls formed by market makers
- Continuos Price discovery through interaction with resting orders 
- Assymetry between bid-side and ask-side liquidity under pressure 
About price jumps without liquidity interaction would indicate a bug; none were observed 


## Results and Observations 
- Liquidity concentrated primarily around market maker quotes
- Price moved through liquidity rather than teleporting
- The PPO agent traded selectively and avoided extreme inventory accumulation
- Noise traders contributed stochastic flow without destabilizing the market

One anomaly observed was occasional thinning of liquidity during clustered agent arrivals, which may warrant further investigation.