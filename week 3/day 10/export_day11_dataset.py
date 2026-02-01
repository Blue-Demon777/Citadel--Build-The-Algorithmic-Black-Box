import pandas as pd

from run_simulation import run_simulation

# --------------------------------------------------
# 1. Run Day 10 simulation ONCE
# --------------------------------------------------
logger = run_simulation(42, 500.0)

# --------------------------------------------------
# 2. Load logged data
# --------------------------------------------------
l1 = logger.l1_df()
inv = logger.inventory_df()

# Sanity check
assert not l1.empty, "l1_df is empty"
assert not inv.empty, "inventory_df is empty"

# Keep only ONE agent (your main one)
AGENT_ID = "MM1"
inv = inv[inv["agent"] == AGENT_ID]

# Sort for merge_asof
l1 = l1.sort_values("time")
inv = inv.sort_values("time")

# --------------------------------------------------
# 3. Merge price + inventory on time
# --------------------------------------------------
df = pd.merge_asof(
    l1,
    inv,
    on="time",
    direction="nearest"
)

# --------------------------------------------------
# 4. Construct Day 11 fields
# --------------------------------------------------
INITIAL_CASH = 100_000

df["timestamp"] = df["time"]
df["price"] = df["mid"]

# Portfolio value = cash + inventory * price
df["agent_portfolio_value"] = (
    INITIAL_CASH + df["inventory"] * df["price"]
)

# Benchmarks (simple but valid)
df["benchmark_buy_hold_value"] = (
    INITIAL_CASH + (df["price"] - df["price"].iloc[0])
)

df["benchmark_random_value"] = INITIAL_CASH

# You did not log actions → mark HOLD
df["inventory_change"] = df["inventory"].diff().fillna(0)

df["action"] = "HOLD"
df.loc[df["inventory_change"] > 0, "action"] = "BUY"
df.loc[df["inventory_change"] < 0, "action"] = "SELL"

# PnL
df["pnl_step"] = df["agent_portfolio_value"].diff().fillna(0.0)
df["pnl_cumulative"] = df["pnl_step"].cumsum()

# --------------------------------------------------
# 5. Final Day 11 dataset
# --------------------------------------------------
final_df = df[
    [
        "timestamp",
        "price",
        "agent_portfolio_value",
        "benchmark_buy_hold_value",
        "benchmark_random_value",
        "action",
        "pnl_step",
        "pnl_cumulative",
    ]
]

final_df.to_csv("performance_dataset.csv", index=False)

print("Day 11 dataset exported: performance_dataset.csv")
