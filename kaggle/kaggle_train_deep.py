"""Self-contained Kaggle kernel script: train LSTM & CNN1D on C-MAPSS FD001.

Runs on Kaggle (GPU accelerator). Reads raw data from the attached
`palbha/cmapss-jet-engine-simulated-data` dataset at /kaggle/input/.
Writes results.json + model checkpoints to /kaggle/working/ (downloadable
via `kaggle kernels output`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
WINDOW = 30
RUL_CAP = 125
STD_THRESHOLD = 1e-3
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 128

torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = next(Path("/kaggle/input").glob("*/train_FD001.txt")).parent
OUT_DIR = Path("/kaggle/working")
COLUMNS = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_fd001(raw_dir: Path):
    train_df = pd.read_csv(raw_dir / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(raw_dir / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul_truth = pd.read_csv(raw_dir / "RUL_FD001.txt", sep=r"\s+", header=None)[0].to_numpy()
    return train_df, test_df, rul_truth


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=RUL_CAP)
    return df


def make_sequences(df, feature_cols, label_col="rul", last_window_only=False):
    X, y, groups = [], [], []
    for unit_id, g in df.groupby("unit_id", sort=False):
        g = g.sort_values("cycle")
        values = g[feature_cols].to_numpy()
        labels = g[label_col].to_numpy() if label_col else None
        if len(values) < WINDOW:
            pad = np.repeat(values[:1], WINDOW - len(values), axis=0)
            values = np.concatenate([pad, values], axis=0)
            if labels is not None:
                labels = np.concatenate([np.repeat(labels[:1], WINDOW - len(labels)), labels])
        n = len(values)
        starts = [n - WINDOW] if last_window_only else range(0, n - WINDOW + 1)
        for s in starts:
            X.append(values[s:s + WINDOW])
            if labels is not None:
                y.append(labels[s + WINDOW - 1])
            groups.append(unit_id)
    return np.stack(X), (np.asarray(y) if y else None), np.asarray(groups)


def nasa_score(y_true, y_pred) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(s.sum())


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTMRul(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNN1DRul(nn.Module):
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x).squeeze(-1)


def train_one_model(name, model, train_loader, val_loader, device):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    best_val, best_state, patience_left = float("inf"), None, PATIENCE
    history = []

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_rmse = float(np.sqrt(np.mean(val_losses)))
        history.append(val_rmse)
        print(f"[{name}] epoch {epoch + 1}/{EPOCHS} val_rmse={val_rmse:.3f}")

        if val_rmse < best_val:
            best_val, best_state, patience_left = val_rmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[{name}] early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, history


def evaluate(name, model, X_test, y_test, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    score = nasa_score(y_test, pred)
    print(f"[{name}] TEST RMSE={rmse:.3f}  NASA_score={score:.1f}")
    return {"rmse": rmse, "nasa_score": score}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_df, test_df, rul_truth = load_fd001(INPUT_DIR)
    train_df = add_rul(train_df)

    sensor_cols = [c for c in train_df.columns if c.startswith("sensor_")]
    stds = train_df[sensor_cols].std()
    feature_cols = stds[stds >= STD_THRESHOLD].index.tolist()

    scaler = MinMaxScaler().fit(train_df[feature_cols])
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    X_all, y_all, unit_ids = make_sequences(train_df, feature_cols)
    X_test, _, _ = make_sequences(test_df, feature_cols, label_col=None, last_window_only=True)
    y_test = rul_truth.clip(max=RUL_CAP)

    # val split: hold out last 20% of units (grouped, no leakage)
    units = np.unique(unit_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(units)
    val_units = set(units[: max(1, len(units) // 5)])
    val_mask = np.isin(unit_ids, list(val_units))

    train_loader = DataLoader(SeqDataset(X_all[~val_mask], y_all[~val_mask]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_all[val_mask], y_all[val_mask]), batch_size=BATCH_SIZE)

    n_features = len(feature_cols)
    results = {}

    lstm, lstm_hist = train_one_model("LSTM", LSTMRul(n_features), train_loader, val_loader, device)
    results["lstm"] = evaluate("LSTM", lstm, X_test, y_test, device)
    torch.save(lstm.state_dict(), OUT_DIR / "lstm_fd001.pt")

    cnn, cnn_hist = train_one_model("CNN1D", CNN1DRul(n_features), train_loader, val_loader, device)
    results["cnn1d"] = evaluate("CNN1D", cnn, X_test, y_test, device)
    torch.save(cnn.state_dict(), OUT_DIR / "cnn1d_fd001.pt")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"results": results, "feature_cols": feature_cols,
                   "lstm_val_history": lstm_hist, "cnn1d_val_history": cnn_hist}, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
