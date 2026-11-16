# Phase 03: Preprocessing & Shared Utilities

## Context
Pipeline tiền xử lý dùng chung cho cả 4 model: RUL labeling, normalization (fit trên train only), sliding window, và NASA score function — viết 1 lần, tái sử dụng ở Phase 4 & 5.

## Files
- Create: `src/rul_labeling.py` — piecewise RUL (cap tại 125)
- Create: `src/windowing.py` — sliding window=30, sinh sequence 3D (cho LSTM/CNN) và flattened/aggregated feature (cho RF/XGBoost)
- Create: `src/nasa_score.py` — hàm `nasa_score(y_true, y_pred)` theo công thức Saxena 2008
- Create: `src/scaling.py` — MinMax/StandardScaler fit trên train, lưu scaler (joblib) để dùng lại cho test

## Steps
1. `rul_labeling.py`: với mỗi unit trong train, `RUL = max_cycle - cycle`; áp piecewise: `RUL = min(RUL, 125)`.
2. `scaling.py`: fit scaler trên sensor columns của train (loại sensor từ Phase 2), transform cả train & test — **không fit lại trên test**.
3. `windowing.py`:
   - Với mỗi unit, sinh các window 30-cycle liên tiếp (stride=1) → `X: (n_samples, 30, n_sensors)`, `y: RUL tại cycle cuối window`.
   - Unit có <30 cycle: pad đầu bằng giá trị cycle đầu tiên (zero-padding alternative tùy chọn — ghi rõ lựa chọn trong code comment).
   - Với test set: chỉ cần window cuối cùng mỗi unit (so với `rul_truth`).
   - Thêm hàm `to_tabular(X_3d) -> X_2d` (flatten hoặc aggregate mean/std/slope mỗi sensor trong window) riêng cho RF/XGBoost.
4. `nasa_score.py`: implement đúng công thức mũ bất đối xứng (d = RUL_pred - RUL_true; score = sum(exp(-d/13)-1 if d<0 else exp(d/10)-1)).

## Acceptance Criteria
- [ ] Unit test nhỏ (hoặc assert trong `if __name__ == "__main__"`) verify: `nasa_score` khớp ví dụ tính tay từ paper Saxena 2008.
- [ ] `windowing.load_windows(train_df)` trả đúng số sample mong đợi (tổng cycle - 29*100 units, xử lý unit ngắn <30 riêng).
- [ ] Scaler chỉ fit 1 lần trên train, serialize ra `models/scaler.joblib`.

## Risks / Rollback
- Piecewise cap=125 là lựa chọn phổ biến nhưng không bắt buộc — nếu kết quả RMSE lệch bất thường so với literature, thử bỏ cap để so sánh (ghi rõ trong report, không âm thầm đổi).
