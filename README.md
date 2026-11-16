# RUL Prediction — NASA C-MAPSS FD001

Dự đoán Remaining Useful Life (RUL) của động cơ turbofan từ sensor time-series,
so sánh Random Forest, XGBoost, LSTM, CNN 1D. Đánh giá bằng RMSE + NASA
asymmetric scoring function (Saxena et al. 2008).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Chạy pipeline (theo thứ tự)

```powershell
cd src
..\.venv\Scripts\python.exe data_loader.py      # phase 01: verify dataset
..\.venv\Scripts\python.exe eda.py              # phase 02: EDA -> feature_config.json
..\.venv\Scripts\python.exe nasa_score.py       # phase 03: sanity check scoring fn
..\.venv\Scripts\python.exe train_classical.py  # phase 04: RF + XGBoost (local)
```

## Deep learning (LSTM, CNN 1D) — train trên Kaggle

Script tự chứa tại `kaggle/kaggle_train_deep.py`, chạy trên Kaggle GPU kernel:

```powershell
cd kaggle
kaggle kernels push -p .
kaggle kernels status mikunakano1357/rul-cmapss-fd001-dl-training
kaggle kernels output mikunakano1357/rul-cmapss-fd001-dl-training -p ./output
```

Kết quả (`results.json`, `lstm_fd001.pt`, `cnn1d_fd001.pt`) tải về `kaggle/output/`.

## Tổng hợp kết quả cuối

```powershell
cd src
..\.venv\Scripts\python.exe evaluate.py   # phase 06: bảng + biểu đồ so sánh 4 model
```

## Cấu trúc

- `data/raw/` — raw FD001 txt (train/test/RUL truth)
- `src/` — data loader, preprocessing, model code, training scripts
- `kaggle/` — self-contained GPU training script cho LSTM/CNN1D + kernel metadata
- `models/` — model checkpoint đã train (scaler, RF, XGBoost)
- `reports/` — kết quả, figures, báo cáo so sánh cuối
- `plans/261110-1950-rul-cmapss-fd001/` — plan chi tiết 6 phase

## Dataset

NASA C-MAPSS FD001: 100 engine train (run-to-failure), 100 engine test, 21
sensor, 3 operational settings. Input window = 30 cycle gần nhất.
