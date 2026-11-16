# Phase 02: EDA

## Context
Cần hiểu phân phối sensor, loại sensor vô dụng (constant/near-constant), và quan hệ degradation trước khi build feature.

## Files
- Create: `notebooks/01_eda.ipynb` (hoặc `src/eda.py` nếu muốn script hoá)
- Output: `reports/figures/*.png`

## Steps
1. Load train/test qua `data_loader.load_fd001`.
2. Thống kê describe() từng sensor + std=0 hoặc near-0 → list sensor loại bỏ (thường: 1, 5, 6, 10, 16, 18, 19 — verify lại trên data thật, không copy mù từ paper).
3. Plot 3-5 engine mẫu: sensor value theo cycle, quan sát trend suy giảm.
4. Tính correlation giữa mỗi sensor và RUL (sau khi gán RUL cho train) để chọn sensor quan trọng.
5. Check 3 operational settings — FD001 chỉ 1 condition nên settings gần như const; verify giả định này đúng trên data thật.

## Acceptance Criteria
- [ ] Danh sách sensor giữ lại (dùng cho Phase 3) có evidence cụ thể (std, correlation), không chỉ copy từ paper khác.
- [ ] Ít nhất 2 hình minh hoạ degradation pattern lưu vào `reports/figures/`.

## Risks / Rollback
- Nếu loại sensor sai (loại sensor có tín hiệu thật) → ảnh hưởng model sau. Rollback: giữ lại toàn bộ 21 sensor làm baseline đối chứng, so sánh RMSE có/không loại sensor ở Phase 6 nếu nghi ngờ.
