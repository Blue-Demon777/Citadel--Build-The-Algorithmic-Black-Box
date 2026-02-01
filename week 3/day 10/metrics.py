import numpy as np

def compute_returns(portfolio_values):
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns

def sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate
    std = np.std(excess)
    if std == 0:
        return 0.0
    return np.mean(excess) / std

def max_drawdown(portfolio_values):
    values = np.array(portfolio_values)
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        dd = (peak - v) / peak
        max_dd = max(max_dd, dd)
    return max_dd

def aggregate_metrics(runs):
    sharpes = []
    drawdowns = []

    for pv in runs:
        rets = compute_returns(pv)
        sharpes.append(sharpe_ratio(rets))
        drawdowns.append(max_drawdown(pv))

    return {
        "mean_sharpe": float(np.mean(sharpes)),
        "std_sharpe": float(np.std(sharpes)),
        "mean_max_drawdown": float(np.mean(drawdowns)),
    }
