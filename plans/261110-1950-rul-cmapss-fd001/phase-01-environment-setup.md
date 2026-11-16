# Phase 01: Environment Setup

## Context
Project folder trống, chưa có code, git đã init. Cần dựng skeleton trước khi làm EDA/model.

## Files
- Create: `requirements.txt`, `README.md`, `.gitignore`
- Create folders: `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `models/`, `reports/figures/`
- Create: `src/data_loader.py` (tải/đọc raw C-MAPSS FD001)

## Steps
1. `python -m venv .venv` → activate.
2. `requirements.txt`: pandas, numpy, scikit-learn, xgboost, torch, matplotlib, seaborn, scipy, joblib.
3. `.gitignore`: `.venv/`, `data/raw/*`, `__pycache__/`, `*.pt`, `*.pkl`, `reports/figures/*.png` (giữ `.gitkeep`).
4. Tải FD001 (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) từ NASA PCoE repo hoặc Kaggle mirror vào `data/raw/`.
5. `src/data_loader.py`: hàm `load_fd001(raw_dir) -> (train_df, test_df, rul_truth)` — gán tên cột chuẩn: `unit_id, cycle, op_setting_1..3, sensor_1..21`.
6. Verify shape: train ≈ 20,631 rows / 100 units; test = 100 units; `rul_truth` = 100 giá trị.

## Acceptance Criteria
- [ ] `python -c "from src.data_loader import load_fd001; ..."` chạy không lỗi, in đúng shape kỳ vọng.
- [ ] Repo có cấu trúc folder như trên, git commit đầu tiên (skeleton, không chứa raw data lớn nếu >100MB — kiểm tra trước khi add).

## Risks / Rollback
- Nếu NASA repo không truy cập được, dùng Kaggle mirror (cùng schema). Rollback: xoá `data/raw/`, thử nguồn khác.
