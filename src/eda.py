"""EDA: drop near-constant sensors, inspect correlation with RUL, plot degradation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_fd001
from rul_labeling import add_rul

ROOT = Path(__file__).resolve().parents[1]
STD_THRESHOLD = 1e-3


def main():
    train_df, test_df, rul_truth = load_fd001(ROOT / "data" / "raw")
    train_df = add_rul(train_df)

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    stds = train_df[sensor_cols].std()
    dropped = stds[stds < STD_THRESHOLD].index.tolist()
    kept = [c for c in sensor_cols if c not in dropped]

    corr = train_df[kept + ["rul"]].corr()["rul"].drop("rul").sort_values(key=abs, ascending=False)

    print(f"Dropped (near-constant, std<{STD_THRESHOLD}): {dropped}")
    print("Correlation with RUL (sorted by |corr|):")
    print(corr)

    config = {"feature_cols": kept, "dropped_sensors": dropped}
    with open(ROOT / "src" / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # degradation plot for 3 sample engines
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    top_sensor = corr.abs().idxmax()
    for ax, unit_id in zip(axes, [1, 2, 3]):
        g = train_df[train_df.unit_id == unit_id]
        ax.plot(g.cycle, g[top_sensor])
        ax.set_title(f"unit {unit_id}: {top_sensor} vs cycle")
        ax.set_xlabel("cycle")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "degradation_sample_units.png", dpi=120)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    corr.plot.barh(ax=ax2)
    ax2.set_title("Correlation of sensors with RUL")
    fig2.tight_layout()
    fig2.savefig(ROOT / "reports" / "figures" / "sensor_rul_correlation.png", dpi=120)

    print(f"feature_config.json written: {len(kept)} sensors kept, {len(dropped)} dropped")


if __name__ == "__main__":
    main()
