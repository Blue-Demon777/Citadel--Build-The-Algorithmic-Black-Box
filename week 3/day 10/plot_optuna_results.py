import optuna
import optuna.visualization as vis


def main():
    study = optuna.load_study(
        study_name="ppo_trading_day9",
        storage="sqlite:///day9_optuna.db"
    )

    # Plot optimization history
    fig = vis.plot_optimization_history(study)

    # Save as HTML (reproducible artifact)
    fig.write_html("optimization_history.html")

    print("optimization_history.html created successfully")


if __name__ == "__main__":
    main()
