# So sánh 4 model dự đoán RUL — C-MAPSS FD001

## Method
- Dataset: FD001 (100 train, 100 test, 21 sensor, 3 operational settings). 6 sensor near-constant bị loại (`sensor_1,5,10,16,18,19`) qua EDA thực nghiệm.
- RUL piecewise, cap=125. Input window=30 cycle.
- RF/XGBoost: feature window-aggregated (mean/std/min/max/slope), GroupKFold(5) theo `unit_id`, GridSearchCV tune cơ bản.
- LSTM/CNN1D (PyTorch, train CPU local): sequence 3D trực tiếp, train/val split theo unit_id, early stopping, target RUL scale về [0,1] + gradient clipping (cần thiết để tránh LSTM collapse — xem Risk).

## Kết quả

| Model | RMSE (cycles) | NASA score |
|---|---|---|
| Random Forest | 14.184 | 316.3 |
| XGBoost | 13.387 | 257.5 |
| LSTM | 13.109 | 287.5 |
| CNN 1D | 16.452 | 365.2 |

**Best RMSE**: LSTM (13.11). **Best NASA score**: XGBoost (257.5).

## Nhận xét
- Cả 4 model nằm trong vùng kỳ vọng hợp lý (11-20 RMSE) khớp với baseline literature (Li 2018 CNN ≈12.6, Zheng 2017 LSTM tương đương) — không có dấu hiệu leak hay lỗi protocol.
- XGBoost thắng rõ nhất ở NASA score (ít bị phạt nặng late-prediction) dù RMSE không thấp nhất — phù hợp bảo trì dự đoán, nơi tránh trễ dự đoán quan trọng hơn độ chính xác trung bình.
- LSTM đạt RMSE thấp nhất nhưng cần kỹ thuật ổn định hoá training (scale target, gradient clipping) — nếu bỏ qua, LSTM collapse về dự đoán gần-hằng-số (RMSE≈40, xem Risk).
- CNN1D kém nhất trong 4 model ở dữ liệu này — có thể do window=30 ngắn, kernel size chưa tối ưu cho FD001 (single operating condition, degradation pattern trơn hơn multi-condition).
- Classical ML (RF/XGBoost) cạnh tranh tốt với deep learning trên FD001 — khớp với quan sát trong literature rằng single-condition/single-fault-mode khiến tree-based method đủ mạnh.

## Hạn chế
- GridSearchCV cho RF/XGBoost chỉ tune cơ bản (không full sweep).
- LSTM/CNN1D train trên CPU local, không chạy được trên Kaggle GPU (kernel bị queue kéo dài bất thường trên free tier, không xác định nguyên nhân - có thể do quota hoặc backlog hệ thống).
- Không áp dụng asymmetric loss trực tiếp (chỉ dùng MSE) — ngoài scope đề tài.

## Unresolved questions
- Kaggle kernel `mikunakano1357/rul-cmapss-fd001-dl-training` vẫn ở trạng thái QUEUED kéo dài bất thường — chưa rõ nguyên nhân (quota GPU hết hay backlog Kaggle). Không ảnh hưởng kết quả vì đã train thành công ở local.
