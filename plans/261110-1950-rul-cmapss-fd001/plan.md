---
status: done
created: 2026-11-10
slug: rul-cmapss-fd001
---

# Plan: RUL Prediction trên NASA C-MAPSS FD001

## Outcome
Xây dựng pipeline đầy đủ dự đoán Remaining Useful Life (RUL) của động cơ turbofan
từ dữ liệu cảm biến chuỗi thời gian (NASA C-MAPSS FD001), so sánh 4 model
(Random Forest, XGBoost, LSTM, CNN 1D) bằng RMSE và NASA asymmetric scoring
function (Saxena et al. 2008).

## Constraints
- Dataset: C-MAPSS FD001 only (100 train engine run-to-failure, 100 test engine, 21 sensor, 3 operational settings).
- Input window: 30 cycle gần nhất (theo chuẩn Li et al. 2018).
- Output: RUL (số cycle còn lại), dùng piecewise RUL clipping (cap ở 125) — chuẩn phổ biến trong literature để tránh model học early-cycle noise.
- Giữ nguyên train/test split gốc của NASA, không leak thông tin tương lai vào feature/scaler (fit scaler trên train, transform test).
- DL framework: PyTorch. Env: venv + pip. Git: đã init.
- Evaluation bắt buộc: RMSE + NASA score, trên đúng 100 test engine với RUL ground truth gốc.

## Non-goals
- Không làm FD002/FD003/FD004 (multi-condition, multi-fault).
- Không deploy (API, web app, Docker).
- Không chạy theo SOTA tuyệt đối (ensemble phức tạp) — mục tiêu là so sánh công bằng 4 model baseline chuẩn.
- Không hyperparameter search quy mô lớn (Optuna full sweep) — chỉ tuning cơ bản.

## Acceptance Criteria
- [x] Dataset FD001 tải về, verify đúng shape (20,631 train rows / 100 train engines, 100 test engines).
- [x] EDA report: phân phối sensor, sensor constant bị loại, tương quan với RUL.
- [x] Feature pipeline tái sử dụng được cho cả 4 model (shared preprocessing cho RF/XGBoost dạng tabular-window, và sequence 3D cho LSTM/CNN).
- [x] 4 model train xong, mỗi model có RMSE + NASA score trên test set.
- [x] Bảng so sánh + biểu đồ predicted vs actual RUL cho 4 model.
- [x] Toàn bộ pipeline reproducible (seed cố định, script chạy từ đầu đến cuối không lỗi).
- [x] README/docs mô tả cách chạy lại toàn bộ pipeline.

## Kết quả cuối (2026-11-10)
| Model | RMSE | NASA score |
|---|---|---|
| Random Forest | 14.184 | 316.3 |
| XGBoost | 13.387 | 257.5 |
| LSTM | 13.109 | 287.5 |
| CNN 1D | 16.452 | 365.2 |

LSTM/CNN1D train local (CPU) thay vì Kaggle do kernel GPU bị queue bất thường
lâu (xem `reports/comparison_report.md` phần Unresolved). Chi tiết: `reports/comparison_report.md`.

## Phases
1. [phase-01-environment-setup.md](phase-01-environment-setup.md) — venv, dependencies, folder structure, tải dataset
2. [phase-02-eda.md](phase-02-eda.md) — khám phá dữ liệu, loại sensor constant, visualize degradation
3. [phase-03-preprocessing.md](phase-03-preprocessing.md) — RUL labeling (piecewise), normalization, sliding window, NASA score function
4. [phase-04-classical-ml.md](phase-04-classical-ml.md) — Random Forest, XGBoost
5. [phase-05-deep-learning.md](phase-05-deep-learning.md) — LSTM, CNN 1D (PyTorch)
6. [phase-06-evaluation-report.md](phase-06-evaluation-report.md) — so sánh 4 model, visualize, viết báo cáo

## Dependencies
Phase 1 → 2 → 3 → (4 và 5 chạy độc lập, có thể song song) → 6

## Prior Art (tham chiếu)
- Saxena et al. 2008 — PHM08, nguồn dataset + NASA scoring function gốc.
- Li et al. 2018 — CNN sliding-window=30, RMSE FD001 ≈ 12.6 (baseline reference).
- Zheng et al. 2017 — LSTM trực tiếp trên sensor sequence.
- Kỳ vọng kết quả hợp lý cho đề tài: RMSE ~11–16 trên FD001 nếu giữ đúng protocol chuẩn (không leak).
