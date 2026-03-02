import wandb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

api = wandb.Api()

# Parameters
ENTITY = "alvomatias_project"
PROJECT = "inventory_control"
SWEEP_ID = "vzkqo3hh"
FILTER_KEY = "pathwise_coef"
FILTER_VALUE = 1
X_AXIS = "n_stores"

PLOT_CONFIGS = [
    {
        "metrics": [
            "train/cosine/continuous/pathwise/mean/across_epochs_mean",
            "train/cosine/continuous/policy/mean/across_epochs_mean",
            "train/cosine/continuous/reinforce/mean/across_epochs_mean",
        ],
        "out_file": "cosine_similarity_plot.png",
        "ylabel": "Cosine Similarity (mean across epochs)",
        "title": f"Cosine Similarity vs n_stores (pathwise_coef={FILTER_VALUE})",
        "ylim": None,
        "label_index": -3,
    },
    {
        "metrics": [
            "train/cosine/continuous/pathwise_vs_policy_plus_pathwise/mean/across_epochs_mean",
            "train/cosine/continuous/pathwise_vs_reinforce_plus_pathwise/mean/across_epochs_mean",
            "train/cosine/continuous/policy_plus_pathwise/mean/across_epochs_mean",
            "train/cosine/continuous/reinforce_plus_pathwise/mean/across_epochs_mean",
        ],
        "out_file": "cosine_similarity_plot_2.png",
        "ylabel": "Cosine Similarity (mean across epochs)",
        "title": f"Cosine Similarity (combined) vs n_stores (pathwise_coef={FILTER_VALUE})",
        "ylim": None,
        "label_index": -3,
    },
    {
        "metrics": [
            "train/grad_analysis/policy/policy.heads_layers.continuous.weight",
            "train/grad_analysis/pathwise/policy.heads_layers.continuous.weight",
            "train/grad_analysis/reinforce/policy.heads_layers.continuous.weight",
        ],
        "out_file": "grad_analysis_plot.png",
        "ylabel": "Gradient Abs Mean",
        "title": f"Gradient Analysis vs n_stores (pathwise_coef={FILTER_VALUE})",
        "ylim": (0, 0.1),
        "label_index": -2,
    },
]


def get_metric_value(run, metric):
    """Pull metric from run history, averaging over epochs (and over vector if needed)."""
    history = run.history(keys=[metric])
    if metric not in history.columns or history[metric].isna().all():
        return None
    values = history[metric].dropna()
    epoch_means = values.apply(lambda x: np.clip(np.mean(x) if hasattr(x, '__iter__') else x, None, 1))
    return epoch_means.mean()


# Pull all runs once
sweep = api.sweep(f"{ENTITY}/{PROJECT}/{SWEEP_ID}")
all_metrics = [m for config in PLOT_CONFIGS for m in config["metrics"]]

print("Pulling runs...")
runs_data = []
for run in sweep.runs:
    if run.config.get(FILTER_KEY) != FILTER_VALUE:
        continue
    row = {X_AXIS: run.config.get(X_AXIS)}
    for metric in all_metrics:
        row[metric] = get_metric_value(run, metric)
    runs_data.append(row)

df = pd.DataFrame(runs_data)
print(df)

# Generate each plot
for config in PLOT_CONFIGS:
    metrics = config["metrics"]
    grouped = df.groupby(X_AXIS)[metrics].mean().reset_index()
    print(f"\n{config['out_file']}:")
    print(grouped)

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in metrics:
        ax.plot(grouped[X_AXIS], grouped[metric], marker='o', label=metric.split("/")[config["label_index"]])

    ax.set_xlabel(X_AXIS)
    ax.set_ylabel(config["ylabel"])
    ax.set_title(config["title"])
    if config["ylim"] is not None:
        ax.set_ylim(*config["ylim"])
    ax.legend()
    plt.tight_layout()
    plt.savefig(config["out_file"], dpi=150)
    plt.show()
    print(f"Saved to {config['out_file']}")