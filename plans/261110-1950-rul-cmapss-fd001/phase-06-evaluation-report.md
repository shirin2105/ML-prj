# Phase 06: Evaluation & Final Report

## Context
Tổng hợp kết quả 4 model (Phase 4 + 5), so sánh công bằng trên cùng test set 100 engine, viết báo cáo cuối.

## Files
- Create: `src/evaluate.py` — load 4 model, predict, tính RMSE + NASA score, xuất bảng
- Create: `reports/comparison_report.md` (hoặc `.ipynb`)
- Output: `reports/figures/comparison_*.png`

## Steps
1. Load 4 model đã train (RF, XGBoost, LSTM, CNN1D), predict trên đúng test set (100 window cuối mỗi unit).
2. Tính RMSE + NASA score cho mỗi model → bảng so sánh.
3. Plot: predicted vs actual RUL (scatter, per model), bar chart so sánh RMSE/score giữa 4 model.
4. Phân tích: model nào tốt ở early-prediction, model nào tốt ở late-prediction (xem distribution của d = pred - true).
5. Viết `reports/comparison_report.md`: tóm tắt method, bảng kết quả, nhận xét (đối chiếu với baseline Li 2018 ~12.6 RMSE, Zheng 2017), hạn chế, hướng mở rộng tương lai (liệt kê ngắn, không tự ý implement thêm).

## Acceptance Criteria
- [ ] Bảng so sánh 4 model đầy đủ (RMSE + NASA score) trong `comparison_report.md`.
- [ ] Ít nhất 3 hình so sánh trong `reports/figures/`.
- [ ] Kết luận model nào phù hợp nhất cho use case bảo trì dự đoán (trade-off RMSE vs NASA score vs cost tính toán vs interpretability).
- [ ] README cập nhật hướng dẫn chạy lại toàn bộ pipeline từ đầu (`phase-01` → `phase-06`).

## Risks / Rollback
- Không có rollback cần thiết — đây là bước tổng hợp, không thay đổi data/model.
