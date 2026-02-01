# Day 11 — Interactive Visualization Dashboard

## Objective
Build an executive-ready, fully interactive dashboard to analyze trading agent performance, risk, and behavior **without rerunning experiments**.

This stage is strictly a **visualization and exploration layer** built on top of Day 10 results.

---

## Data Source
- Single dataset generated from Day 10 simulations
- File: `data/performance_dataset.csv`
- All charts reference this dataset only

---

## Dashboard Components

1. **Portfolio Value vs Benchmarks**
   - RL Agent
   - Buy & Hold
   - Random baseline
   - Interactive line chart with zoom, pan, and toggles

2. **Market Price with Agent Actions**
   - Market price as line chart
   - BUY actions shown as ▲
   - SELL actions shown as ▼
   - Linked time axis with other charts

3. **PnL Distribution**
   - Histogram of per-step PnL
   - Gaussian fit overlay
   - Hover-enabled and interactive

---

## Technology Stack
- Plotly
- Jupyter Notebook
- Plotly Dash (optional web app)

---

## How to Run (Notebook)

1. Open `dashboard.ipynb`
2. Run all cells
3. Use hover, zoom, and legend toggles to explore results

---

## How to Run (Web App – Optional)

```bash
python app.py
```
Then open the local URL shown in the terminal (typically http://127.0.0.1:8050/).

## Note:

- No simulations, training, or metric recomputation occur in Day 11

- This dashboard is purely observational and exploratory

- All interaction is driven by precomputed results from Day 10