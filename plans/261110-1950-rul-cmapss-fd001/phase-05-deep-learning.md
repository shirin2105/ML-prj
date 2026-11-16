# Phase 05: Deep Learning (LSTM, CNN 1D) — PyTorch

## Context
Dùng sequence 3D (n_samples, 30, n_sensors) từ Phase 3, không cần feature engineering thủ công. Độc lập với Phase 4.

## Files
- Create: `src/models/lstm_model.py` — class `LSTMRul(nn.Module)`
- Create: `src/models/cnn1d_model.py` — class `CNN1DRul(nn.Module)`
- Create: `src/train_deep.py` (entrypoint train loop, early stopping)
- Create: `src/dataset.py` — `torch.utils.data.Dataset` wrap windows 3D

## Steps
1. `dataset.py`: `RULDataset(X, y)` trả tensor `(seq_len=30, n_sensors)` — LSTM nhận trực tiếp, CNN1D cần permute thành `(n_sensors, seq_len)`.
2. `lstm_model.py`: 1-2 layer LSTM (hidden_size ~64-128) → FC head → RUL scalar. Dropout giữa layer để chống overfit.
3. `cnn1d_model.py`: 2-3 Conv1D block (kernel 3-5, với BatchNorm+ReLU) + global pooling → FC head.
4. `train_deep.py`: loss = MSE (hoặc thử thêm asymmetric loss nếu có thời gian — optional, không bắt buộc theo scope đề tài), optimizer Adam, early stopping theo validation RMSE (split train thành train/val theo unit_id, không random row).
5. Train cả 2 model, lưu checkpoint: `models/lstm_fd001.pt`, `models/cnn1d_fd001.pt`.
6. Predict test set, tính RMSE + NASA score sanity-check.

## Acceptance Criteria
- [ ] Train/val split theo `unit_id` (không leak cùng engine giữa train/val).
- [ ] Loss curve (train vs val) được log/plot — phát hiện overfit sớm.
- [ ] Cả 2 model đạt RMSE hợp lý (~11-18) trên test set.
- [ ] Seed cố định (`torch.manual_seed`, `numpy.random.seed`) để reproducible.

## Risks / Rollback
- LSTM/CNN dễ overfit với 100 engine (ít data) — nếu val loss tăng sớm, giảm model size / tăng dropout / giảm epoch thay vì đổi architecture phức tạp hơn.
