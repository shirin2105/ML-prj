"""NASA asymmetric scoring function (Saxena et al., PHM08 2008).

Penalizes late predictions (pred > true) more than early predictions.
"""
import numpy as np


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = d < 0
    score = np.where(early, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(score.sum())


if __name__ == "__main__":
    # sanity check: symmetric small errors -> late penalized more than early
    s_early = nasa_score([100], [95])   # d = -5 (early)
    s_late = nasa_score([100], [105])   # d = +5 (late)
    assert s_late > s_early, "late prediction must be penalized more than early"
    print(f"early(-5) score={s_early:.4f}, late(+5) score={s_late:.4f}")
