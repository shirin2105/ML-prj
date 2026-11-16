"""Phase 05: train LSTM and CNN1D locally (CPU) — small dataset, GPU not required."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_loader import load_fd001
from dataset import RULDataset
from models.cnn1d_model import CNN1DRul
from models.lstm_model import LSTMRul
from nasa_score import nasa_score
from rul_labeling import RUL_CAP, add_rul
from scaling import apply_scaler, fit_scaler, save_scaler
from windowing import WINDOW, make_sequences

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 128

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_data():
    train_df, test_df, rul_truth = load_fd001(ROOT / "data" / "raw")
    train_df = add_rul(train_df)
    feature_cols = json.load(open(ROOT / "src" / "feature_config.json"))["feature_cols"]

    scaler = fit_scaler(train_df, feature_cols)
    save_scaler(scaler, ROOT / "models" / "scaler.joblib")
    train_scaled = apply_scaler(train_df, feature_cols, scaler)
    test_scaled = apply_scaler(test_df, feature_cols, scaler)

    X_all, y_all, unit_ids = make_sequences(train_scaled, feature_cols, WINDOW)
    X_test, _, _ = make_sequences(test_scaled, feature_cols, WINDOW, label_col=None, last_window_only=True)
    y_test = rul_truth.to_numpy().clip(max=RUL_CAP)

    # scale RUL target to [0, 1]: raw 0-125 scale destabilizes LSTM gradients early in training
    y_all = y_all / RUL_CAP

    units = np.unique(unit_ids)
    rng = np.random.default_rng(SEED)
    rng.shuffle(units)
    val_units = set(units[: max(1, len(units) // 5)])
    val_mask = np.isin(unit_ids, list(val_units))

    return X_all, y_all, val_mask, X_test, y_test, len(feature_cols)


def train_one(name, model, train_loader, val_loader, device):
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())
        val_rmse = float(np.sqrt(np.mean(val_losses))) * RUL_CAP
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
        pred = model(torch.as_tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy() * RUL_CAP
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    score = nasa_score(y_test, pred)
    print(f"[{name}] TEST RMSE={rmse:.3f}  NASA_score={score:.1f}")
    return {"rmse": rmse, "nasa_score": score}


def main():
    device = torch.device("cpu")
    X_all, y_all, val_mask, X_test, y_test, n_features = build_data()

    train_loader = DataLoader(RULDataset(X_all[~val_mask], y_all[~val_mask]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RULDataset(X_all[val_mask], y_all[val_mask]), batch_size=BATCH_SIZE)

    results = {}

    lstm, lstm_hist = train_one("LSTM", LSTMRul(n_features), train_loader, val_loader, device)
    results["lstm"] = evaluate("LSTM", lstm, X_test, y_test, device)
    torch.save(lstm.state_dict(), ROOT / "models" / "lstm_fd001.pt")

    cnn, cnn_hist = train_one("CNN1D", CNN1DRul(n_features), train_loader, val_loader, device)
    results["cnn1d"] = evaluate("CNN1D", cnn, X_test, y_test, device)
    torch.save(cnn.state_dict(), ROOT / "models" / "cnn1d_fd001.pt")

    with open(ROOT / "reports" / "deep_results.json", "w") as f:
        json.dump({"results": results, "lstm_val_history": lstm_hist, "cnn1d_val_history": cnn_hist}, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
