"""Sliding-window sample generation shared by classical ML and deep learning models."""
import numpy as np
import pandas as pd

WINDOW = 30


def make_sequences(df: pd.DataFrame, feature_cols: list[str], window: int = WINDOW,
                    label_col: str | None = "rul", last_window_only: bool = False):
    """Build 3D sequences (n_samples, window, n_features) per unit.

    Short units (<window cycles) are front-padded by repeating the first row.
    If `last_window_only`, only the final window of each unit is returned
    (used for the official test set, matched against RUL_FD001.txt).
    """
    X, y, unit_ids = [], [], []
    for unit_id, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("cycle")
        values = g[feature_cols].to_numpy()
        labels = g[label_col].to_numpy() if label_col is not None else None

        if len(values) < window:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            values = np.concatenate([pad, values], axis=0)
            if labels is not None:
                labels = np.concatenate([np.repeat(labels[:1], window - len(labels)), labels])

        n = len(values)
        starts = [n - window] if last_window_only else range(0, n - window + 1)
        for s in starts:
            X.append(values[s:s + window])
            if labels is not None:
                y.append(labels[s + window - 1])
            unit_ids.append(unit_id)

    X = np.stack(X)
    y = np.asarray(y) if y else None
    return X, y, np.asarray(unit_ids)


def _agg_window(col: np.ndarray, suffix: str, records: dict, name: str, full_stats: bool) -> None:
    """Aggregate one sensor's sub-window (last `col.shape[1]` cycles) into records."""
    length = col.shape[1]
    t_centered = np.arange(length) - (length - 1) / 2
    denom = (t_centered ** 2).sum()
    records[f"{name}_mean_{suffix}"] = col.mean(axis=1)
    records[f"{name}_std_{suffix}"] = col.std(axis=1)
    records[f"{name}_slope_{suffix}"] = (col * t_centered).sum(axis=1) / denom
    if full_stats:
        records[f"{name}_min_{suffix}"] = col.min(axis=1)
        records[f"{name}_max_{suffix}"] = col.max(axis=1)


def to_tabular(X_3d: np.ndarray, feature_names: list[str], subwindows: tuple[int, ...] = (10, 20, 30)) -> pd.DataFrame:
    """Multi-scale aggregation: mean/std/slope over the last 10/20/30 cycles of each
    window (min/max only on the full window) per sensor (for RF/XGBoost).

    Rationale: a single full-window aggregate blurs recent acceleration in
    degradation; short sub-windows (10, 20 cycles) capture that recent trend
    while the full 30-cycle window keeps the long-range signal.
    """
    n_samples, window, n_features = X_3d.shape
    records = {}
    for i, name in enumerate(feature_names):
        col = X_3d[:, :, i]
        for sw in subwindows:
            sw = min(sw, window)
            _agg_window(col[:, -sw:], str(sw), records, name, full_stats=(sw == window))
    return pd.DataFrame(records)
