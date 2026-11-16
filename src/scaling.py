"""Fit a scaler on train sensor columns only; reuse for test (no leakage)."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(train_df: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    return scaler


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler: MinMaxScaler) -> pd.DataFrame:
    out = df.copy()
    out[feature_cols] = scaler.transform(out[feature_cols])
    return out


def save_scaler(scaler: MinMaxScaler, path: str | Path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path: str | Path) -> MinMaxScaler:
    return joblib.load(path)
