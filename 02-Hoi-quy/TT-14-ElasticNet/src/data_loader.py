"""
Nạp bộ dữ liệu Energy Efficiency (UCI).
Hỗ trợ cả .csv và .xlsx (file gốc UCI là ENB2012_data.xlsx).

Cách dùng khi bạn tự tải dataset:
    - Tải file tại: https://archive.ics.uci.edu/dataset/242/energy+efficiency
    - Đặt file vào: data/ENB2012_data.xlsx  (hoặc .csv với cùng cột)
    - Hoặc gọi load_energy_data("đường/dẫn/của/bạn.xlsx")
"""
import pandas as pd

FEATURE_COLS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
CATEGORICAL_COLS = ["X6", "X8"]          # hướng nhà, phân bố kính -> one-hot
NUMERIC_COLS = ["X1", "X2", "X3", "X4", "X5", "X7"]
TARGET_COLS = ["Y1", "Y2"]

# Đường dẫn mặc định trong project. Đổi lại nếu bạn để file dataset chỗ khác.
DEFAULT_DATA_PATH = "data/ENB2012_data.csv"


def load_energy_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Đọc file dataset (.csv hoặc .xlsx), trả về DataFrame thô với đúng 10 cột."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df = df.dropna(how="all").dropna(axis=1, how="all")  # dọn dòng/cột trống thừa
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in FEATURE_COLS + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} trong file dữ liệu — kiểm tra lại header.")

    return df[FEATURE_COLS + TARGET_COLS].astype(float)


def make_design_matrix(df: pd.DataFrame):
    """One-hot hoá X6, X8 (biến phân loại dạng số) -> ma trận đặc trưng cuối cùng."""
    X_cat = pd.get_dummies(
        df[CATEGORICAL_COLS].astype(int).astype(str),
        prefix=CATEGORICAL_COLS, drop_first=True
    )
    X = pd.concat([df[NUMERIC_COLS], X_cat], axis=1)
    y1, y2 = df["Y1"], df["Y2"]
    return X, y1, y2
