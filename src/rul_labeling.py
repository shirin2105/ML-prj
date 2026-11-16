"""Piecewise RUL labeling for C-MAPSS train data."""
import pandas as pd

RUL_CAP = 125


def add_rul(train_df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Add a `rul` column: cycles remaining until failure, capped at `cap`."""
    df = train_df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df
