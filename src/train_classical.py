"""Phase 04: train Random Forest and XGBoost on window-aggregated tabular features."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV
from xgboost import XGBRegressor

from data_loader import load_fd001
from nasa_score import nasa_score
from rul_labeling import add_rul
from scaling import apply_scaler, fit_scaler, save_scaler
from windowing import WINDOW, make_sequences, to_tabular

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def build_datasets():
    train_df, test_df, rul_truth = load_fd001(ROOT / "data" / "raw")
    train_df = add_rul(train_df)

    feature_cols = json.load(open(ROOT / "src" / "feature_config.json"))["feature_cols"]

    scaler = fit_scaler(train_df, feature_cols)
    save_scaler(scaler, ROOT / "models" / "scaler.joblib")
    train_scaled = apply_scaler(train_df, feature_cols, scaler)
    test_scaled = apply_scaler(test_df, feature_cols, scaler)

    X_train_3d, y_train, unit_ids = make_sequences(train_scaled, feature_cols, WINDOW)
    X_test_3d, _, _ = make_sequences(test_scaled, feature_cols, WINDOW, label_col=None, last_window_only=True)

    X_train = to_tabular(X_train_3d, feature_cols)
    X_test = to_tabular(X_test_3d, feature_cols)
    return X_train, y_train, unit_ids, X_test, rul_truth.to_numpy()


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    score = nasa_score(y_test, pred)
    print(f"[{name}] RMSE={rmse:.3f}  NASA_score={score:.1f}")
    return {"rmse": rmse, "nasa_score": score}


def select_features(X_train, y_train, unit_ids, cumulative_threshold: float = 0.95) -> list[str]:
    """Fit a quick XGBoost on the full multi-scale feature set, keep the smallest
    set of top-importance features whose cumulative gain covers `cumulative_threshold`.
    """
    probe = XGBRegressor(n_estimators=150, max_depth=5, random_state=SEED, n_jobs=-1)
    probe.fit(X_train, y_train)
    importances = pd.Series(probe.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    cum = importances.cumsum() / importances.sum()
    selected = cum[cum <= cumulative_threshold].index.tolist()
    if not selected:
        selected = [importances.index[0]]
    print(f"Feature selection: {len(selected)}/{len(X_train.columns)} features cover {cumulative_threshold:.0%} importance")
    print("  top 10:", importances.head(10).to_dict())
    return selected


def grid_search(estimator, param_grid, X_train, y_train, unit_ids, group_kfold):
    grid = GridSearchCV(estimator, param_grid, cv=group_kfold, scoring="neg_root_mean_squared_error", n_jobs=-1)
    grid.fit(X_train, y_train, groups=unit_ids)
    return grid


def main():
    X_train, y_train, unit_ids, X_test, y_test = build_datasets()
    group_kfold = GroupKFold(n_splits=5)
    results = {}

    rf_params = {"n_estimators": [100, 200], "max_depth": [10, 20, None], "min_samples_leaf": [1, 5]}
    xgb_params = {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1]}

    # baseline: full multi-scale feature set (165 features)
    rf_grid = grid_search(RandomForestRegressor(random_state=SEED), rf_params, X_train, y_train, unit_ids, group_kfold)
    results["random_forest"] = evaluate("RandomForest (full features)", rf_grid.best_estimator_, X_test, y_test)

    xgb_grid = grid_search(XGBRegressor(random_state=SEED, n_jobs=-1), xgb_params, X_train, y_train, unit_ids, group_kfold)
    results["xgboost"] = evaluate("XGBoost (full features)", xgb_grid.best_estimator_, X_test, y_test)

    # feature selection ablation: keep top-importance features covering 95% cumulative gain
    selected_cols = select_features(X_train, y_train, unit_ids)
    X_train_sel, X_test_sel = X_train[selected_cols], X_test[selected_cols]

    rf_grid_sel = grid_search(RandomForestRegressor(random_state=SEED), rf_params, X_train_sel, y_train, unit_ids, group_kfold)
    joblib.dump(rf_grid_sel.best_estimator_, ROOT / "models" / "rf_fd001.joblib")
    results["random_forest_selected"] = evaluate("RandomForest (selected features)", rf_grid_sel.best_estimator_, X_test_sel, y_test)
    print("  best params:", rf_grid_sel.best_params_)

    xgb_grid_sel = grid_search(XGBRegressor(random_state=SEED, n_jobs=-1), xgb_params, X_train_sel, y_train, unit_ids, group_kfold)
    joblib.dump(xgb_grid_sel.best_estimator_, ROOT / "models" / "xgb_fd001.joblib")
    results["xgboost_selected"] = evaluate("XGBoost (selected features)", xgb_grid_sel.best_estimator_, X_test_sel, y_test)
    print("  best params:", xgb_grid_sel.best_params_)

    with open(ROOT / "src" / "selected_features.json", "w") as f:
        json.dump(selected_cols, f, indent=2)
    with open(ROOT / "reports" / "classical_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
