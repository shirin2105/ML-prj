"""Generate a comprehensive Excel workbook for NASA C-MAPSS FD001 dataset with descriptions and formatting."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_PATH = ROOT / "data" / "C-MAPSS_FD001_Dataset.xlsx"

COLUMNS = [
    "unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"
] + [f"sensor_{i}" for i in range(1, 22)]

SENSOR_INFO = [
    ("unit_id", "Mã định danh động cơ", "1 đến 100", "Định danh", "Có"),
    ("cycle", "Chu kỳ vận hành (Time in cycles)", "1, 2, ...", "Thời gian", "Có"),
    ("op_setting_1", "Cài đặt vận hành 1 (Altitude)", "Độ cao (ft)", "Vận hành", "Không (hằng số FD001)"),
    ("op_setting_2", "Cài đặt vận hành 2 (Mach Number)", "Tốc độ Mach", "Vận hành", "Không (hằng số FD001)"),
    ("op_setting_3", "Cài đặt vận hành 3 (TRA)", "Góc bướm ga (Throttle Resolver Angle)", "Vận hành", "Không (hằng số FD001)"),
    ("sensor_1", "Total temperature at fan inlet", "°R (Rankine)", "Nhiệt độ đầu vào quạt", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_2", "Total temperature at LPC outlet", "°R", "Nhiệt độ đầu ra máy nén áp thấp", "Dùng (15 sensors)"),
    ("sensor_3", "Total temperature at HPC outlet", "°R", "Nhiệt độ đầu ra máy nén áp cao", "Dùng (15 sensors)"),
    ("sensor_4", "Total temperature at LPT outlet", "°R", "Nhiệt độ đầu ra tuabin áp thấp", "Dùng (15 sensors)"),
    ("sensor_5", "Pressure at fan inlet", "psia", "Áp suất đầu vào quạt", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_6", "Total pressure in bypass-duct", "psia", "Áp suất đường ống bypass", "Dùng (15 sensors)"),
    ("sensor_7", "Total pressure at HPC outlet", "psia", "Áp suất đầu ra máy nén áp cao", "Dùng (15 sensors)"),
    ("sensor_8", "Physical fan speed", "rpm", "Tốc độ vòng quay cơ học quạt", "Dùng (15 sensors)"),
    ("sensor_9", "Physical core speed", "rpm", "Tốc độ vòng quay cơ học lõi động cơ", "Dùng (15 sensors)"),
    ("sensor_10", "Engine pressure ratio (P50/P2)", "--", "Tỷ số áp suất động cơ", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_11", "Static pressure at HPC outlet", "psia", "Áp suất tĩnh đầu ra HPC", "Dùng (15 sensors)"),
    ("sensor_12", "Ratio of fuel flow to Ps30", "pps/psia", "Tỷ lệ lưu lượng nhiên liệu / Ps30", "Dùng (15 sensors)"),
    ("sensor_13", "Corrected fan speed", "rpm", "Tốc độ vòng quay quạt hiệu chỉnh", "Dùng (15 sensors)"),
    ("sensor_14", "Corrected core speed", "rpm", "Tốc độ vòng quay lõi hiệu chỉnh", "Dùng (15 sensors)"),
    ("sensor_15", "Bypass Ratio (BPR)", "--", "Tỷ lệ dòng bypass", "Dùng (15 sensors)"),
    ("sensor_16", "Burner fuel-air ratio", "--", "Tỷ lệ nhiên liệu / không khí buồng đốt", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_17", "Bleed Enthalpy", "--", "Entanpi khí trích", "Dùng (15 sensors)"),
    ("sensor_18", "Demanded fan speed", "rpm", "Tốc độ quạt yêu cầu", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_19", "Demanded corrected fan speed", "rpm", "Tốc độ quạt hiệu chỉnh yêu cầu", "Loại bỏ (Variance ≈ 0)"),
    ("sensor_20", "HPT coolant bleed", "lbm/s", "Lưu lượng khí làm mát tuabin cao áp", "Dùng (15 sensors)"),
    ("sensor_21", "LPT coolant bleed", "lbm/s", "Lưu lượng khí làm mát tuabin áp thấp", "Dùng (15 sensors)"),
    ("raw_rul", "Remaining Useful Life thực tế (tính đến hỏng)", "cycles", "Nhãn mục tiêu gốc", "Nhãn huấn luyện"),
    ("clipped_rul", "Piecewise RUL (cap ở 125 cycles)", "cycles", "Nhãn đã cắt trần", "Target chính của Model"),
]


def create_excel():
    print("Loading raw files...")
    train_df = pd.read_csv(RAW_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    test_df = pd.read_csv(RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul_truth = pd.read_csv(RAW_DIR / "RUL_FD001.txt", sep=r"\s+", header=None)[0]

    # Calculate raw RUL and clipped RUL for train
    max_cycle_train = train_df.groupby("unit_id")["cycle"].transform("max")
    train_df["raw_rul"] = max_cycle_train - train_df["cycle"]
    train_df["clipped_rul"] = train_df["raw_rul"].clip(upper=125)

    # Build test summary
    test_summary = test_df.groupby("unit_id")["cycle"].max().reset_index()
    test_summary.rename(columns={"cycle": "last_cycle_in_test"}, inplace=True)
    test_summary["true_rul"] = rul_truth.values
    test_summary["clipped_rul"] = test_summary["true_rul"].clip(upper=125)
    test_summary["estimated_failure_cycle"] = test_summary["last_cycle_in_test"] + test_summary["true_rul"]

    # Build dictionary DataFrame
    dict_df = pd.DataFrame(SENSOR_INFO, columns=[
        "Tên biến / Cột", "Mô tả kỹ thuật (Technical Description)", "Đơn vị (Unit)", "Ý nghĩa vật lý", "Trạng thái trong mô hình"
    ])

    print("Writing to Excel:", OUTPUT_PATH)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        dict_df.to_excel(writer, sheet_name="Data_Dictionary", index=False)
        test_summary.to_excel(writer, sheet_name="RUL_Ground_Truth", index=False)
        test_df.to_excel(writer, sheet_name="test_FD001", index=False)
        train_df.to_excel(writer, sheet_name="train_FD001", index=False)

    print("Styling sheets with openpyxl...")
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.load_workbook(OUTPUT_PATH)

    header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    accent_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    border_thin = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        ws.views.sheetView[0].showGridLines = True
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        for col_idx in range(1, ws.max_column + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill if sheet_name != "Data_Dictionary" else accent_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

            # Auto-fit column width (limited between 12 and 45)
            max_len = max(len(str(ws.cell(row=r, column=col_idx).value or '')) for r in range(1, min(ws.max_row + 1, 150)))
            col_letter = get_column_letter(col_idx)
            ws.column_dimensions[col_letter].width = max(min(max_len + 4, 45), 12)

        ws.row_dimensions[1].height = 28

    wb.save(OUTPUT_PATH)
    print("Done! File saved successfully at:", OUTPUT_PATH)


if __name__ == "__main__":
    create_excel()

