"""Phase 06: combine classical + deep learning results into one comparison report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    classical = json.load(open(ROOT / "reports" / "classical_results.json"))
    deep = json.load(open(ROOT / "reports" / "deep_results.json"))["results"]

    all_results = {**classical, **deep}
    names = list(all_results.keys())

    print(f"{'model':<15}{'RMSE':>10}{'NASA_score':>14}")
    for name in names:
        r = all_results[name]
        print(f"{name:<15}{r['rmse']:>10.3f}{r['nasa_score']:>14.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, [all_results[n]["rmse"] for n in names], color="steelblue")
    axes[0].set_title("RMSE by model")
    axes[0].set_ylabel("RMSE (cycles)")
    axes[1].bar(names, [all_results[n]["nasa_score"] for n in names], color="indianred")
    axes[1].set_title("NASA score by model (lower=better)")
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "comparison_bar.png", dpi=120)

    with open(ROOT / "reports" / "final_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    best_rmse = min(names, key=lambda n: all_results[n]["rmse"])
    best_score = min(names, key=lambda n: all_results[n]["nasa_score"])
    print(f"\nBest RMSE: {best_rmse} | Best NASA score: {best_score}")


if __name__ == "__main__":
    main()
