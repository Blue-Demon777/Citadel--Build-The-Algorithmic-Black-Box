import matplotlib.pyplot as plt
import numpy as np

def _truncate_runs(runs):
    """
    Truncate all runs to the minimum length.
    This avoids padding bias and keeps time alignment correct.
    """
    min_len = min(len(r) for r in runs if len(r) > 0)
    return np.array([r[:min_len] for r in runs])

def plot_equity_curves(results, output_path):
    plt.figure(figsize=(10, 6))

    for agent, runs in results.items():
        valid_runs = [r for r in runs if len(r) > 0]
        if not valid_runs:
            continue

        aligned = _truncate_runs(valid_runs)
        mean_curve = aligned.mean(axis=0)

        plt.plot(mean_curve, label=agent)

    plt.title("Equity Curves (Mean, Aligned Episodes)")
    plt.xlabel("Time Step")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_drawdown_curves(results, output_path):
    plt.figure(figsize=(10, 6))

    for agent, runs in results.items():
        valid_runs = [r for r in runs if len(r) > 0]
        if not valid_runs:
            continue

        aligned = _truncate_runs(valid_runs)

        drawdowns = []
        for run in aligned:
            peak = run[0]
            dd = []
            for v in run:
                peak = max(peak, v)
                dd.append((peak - v) / peak)
            drawdowns.append(dd)

        mean_dd = np.mean(drawdowns, axis=0)
        plt.plot(mean_dd, label=agent)

    plt.title("Drawdown Curves (Mean, Aligned Episodes)")
    plt.xlabel("Time Step")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
