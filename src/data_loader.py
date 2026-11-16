"""Load NASA C-MAPSS FD001 raw text files into labeled DataFrames."""
from pathlib import Path

import pandas as pd

COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load train/test/RUL-truth for FD001.

    Returns:
        train_df: run-to-failure data, columns = COLUMNS
        test_df: truncated-run data, columns = COLUMNS
        rul_truth: Series indexed 1..100, true RUL at end of each test unit
    """
    raw_dir = Path(raw_dir)
    train_df = pd.read_csv(
        raw_dir / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        raw_dir / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS
    )
    rul_truth = pd.read_csv(
        raw_dir / "RUL_FD001.txt", sep=r"\s+", header=None
    )[0]
    rul_truth.index = range(1, len(rul_truth) + 1)
    return train_df, test_df, rul_truth


if __name__ == "__main__":
    train_df, test_df, rul_truth = load_fd001(Path(__file__).resolve().parents[1] / "data" / "raw")
    print(f"train shape: {train_df.shape}, units: {train_df.unit_id.nunique()}")
    print(f"test shape: {test_df.shape}, units: {test_df.unit_id.nunique()}")
    print(f"rul_truth: {len(rul_truth)} values")
