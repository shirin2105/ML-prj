# Phase 04: Classical ML (Random Forest, XGBoost)

## Context
Baseline tabular dùng feature window-aggregated (mean/std/slope) từ Phase 3. Độc lập với Phase 5, có thể làm song song.

## Files
- Create: `src/models/random_forest.py`
- Create: `src/models/xgboost_model.py`
- Create: `src/train_classical.py` (entrypoint train + save model)

## Steps
1. Dùng `windowing.to_tabular()` để sinh feature set (mean, std, min, max, slope mỗi sensor trong window 30) + RUL label.
2. Random Forest: `RandomForestRegressor` (sklearn), tune cơ bản: `n_estimators`, `max_depth`, `min_samples_leaf` qua `GridSearchCV` nhỏ (3-5 fold, CV theo unit_id để tránh leak giữa window cùng 1 engine).
3. XGBoost: `XGBRegressor`, tune cơ bản tương tự (`n_estimators`, `max_depth`, `learning_rate`).
4. Lưu model: `models/rf_fd001.joblib`, `models/xgb_fd001.joblib`.
5. Predict trên test set (window cuối mỗi unit), tính RMSE + NASA score ngay để sanity-check sớm (full comparison ở Phase 6).

## Acceptance Criteria
- [ ] Cả 2 model train xong không lỗi, RMSE nằm trong khoảng hợp lý (~11-20) — nếu lệch xa, kiểm tra lại leak/scaling trước khi tiếp tục.
- [ ] Cross-validation split theo `unit_id` (GroupKFold), không theo row ngẫu nhiên — tránh leak giữa window cùng engine.
- [ ] Feature importance (RF/XGBoost) được log/plot — hữu ích cho báo cáo so sánh.

## Risks / Rollback
- Leak qua CV ngẫu nhiên (không group theo unit) là lỗi phổ biến nhất ở bước này — nếu RMSE CV thấp bất thường so với test RMSE, nghi ngờ leak đầu tiên.
