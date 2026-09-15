"""
features.py — Nạp dữ liệu, chứng minh rò rỉ, mã hoá đặc trưng chu kỳ,
và chia dữ liệu THEO THỜI GIAN cho bài toán dự báo nhu cầu thuê xe đạp
theo giờ (TT-19 — XGBoost Regressor).

Dataset: UCI Bike Sharing Dataset — file hour.csv
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Các cột không dùng làm đặc trưng đầu vào (rò rỉ hoặc dư thừa)
LEAK_COLS = ["casual", "registered"]
DROP_COLS = ["instant", "dteday"]  # dteday đã được tách thành yr/mnth/hr/weekday
TARGET = "cnt"


def load_data(path: str) -> pd.DataFrame:
    """Nạp hour.csv. `dteday` được parse để tiện EDA nhưng không dùng làm đặc trưng
    trực tiếp (thông tin ngày đã có sẵn ở yr/mnth/hr/weekday)."""
    df = pd.read_csv(path)
    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"])
    return df


def prove_leakage(df: pd.DataFrame, target: str = TARGET) -> float:
    """⭐ BẪY 1 — Chứng minh rò rỉ trực tiếp: cnt = casual + registered.

    Train một hồi quy tuyến tính đơn giản CÓ giữ casual + registered để
    cho thấy R² ~ 1.0 một cách "ảo" — chứng minh vì sao 2 cột này PHẢI bị bỏ.
    Chỉ dùng để minh hoạ, KHÔNG dùng model này để dự báo.
    """
    assert all(c in df.columns for c in LEAK_COLS), "Thiếu cột casual/registered để chứng minh rò rỉ"
    X_leak = df[LEAK_COLS].values
    y = df[target].values
    model = LinearRegression().fit(X_leak, y)
    r2 = r2_score(y, model.predict(X_leak))
    return r2


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mã hoá sin/cos cho các đặc trưng có tính chu kỳ:
    hr (chu kỳ 24), mnth (chu kỳ 12), weekday (chu kỳ 7).
    Lý do: giờ 23 và giờ 0 liền kề nhau thực tế nhưng cách xa nhau về giá trị số.
    """
    df = df.copy()
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


def build_feature_matrix(df: pd.DataFrame, drop_leak: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Trả về (X, y) sẵn sàng cho model.

    drop_leak=True  -> pipeline chính thức (BỎ casual/registered)
    drop_leak=False -> CHỈ dùng trong prove_leakage(), không dùng để train model thật.
    """
    df = add_cyclical_features(df)
    drop_cols = list(DROP_COLS)
    if drop_leak:
        drop_cols += LEAK_COLS
    # Bỏ các cột gốc đã được thay bằng bản sin/cos để tránh trùng lặp thông tin thô
    # nhưng vẫn giữ nguyên các cột gốc hr/mnth/weekday vì XGBoost (cây quyết định)
    # có thể khai thác thêm thông tin thứ tự từ dạng số nguyên gốc song song với sin/cos.
    feature_cols = [c for c in df.columns if c not in drop_cols + [TARGET]]
    X = df[feature_cols]
    y = df[TARGET]
    return X, y


def time_split(
    df: pd.DataFrame,
    val_months: int = 9,
    test_months: int = 3,
):
    """Chia dữ liệu THEO THỜI GIAN, KHÔNG shuffle (chống nhìn thấy tương lai):

        - yr == 0 (năm 1)                              -> train
        - yr == 1, tháng 1..val_months                  -> validation
        - yr == 1, tháng (val_months+1)..12              -> test

    Mặc định: năm 1 train · 9 tháng đầu năm 2 validation · 3 tháng cuối năm 2 test
    (đúng theo yêu cầu README).
    """
    assert val_months + test_months == 12, "val_months + test_months phải bằng 12"
    df = df.sort_values(["yr", "mnth", "hr"]).reset_index(drop=True)

    train_mask = df["yr"] == 0
    val_mask = (df["yr"] == 1) & (df["mnth"] <= val_months)
    test_mask = (df["yr"] == 1) & (df["mnth"] > val_months)

    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()


def naive_baseline_predict(df_context: pd.DataFrame, df_target: pd.DataFrame) -> np.ndarray:
    """Baseline naive: dự báo cnt của (ngày, giờ) = cnt của CÙNG GIỜ, CÙNG THỨ
    TUẦN TRƯỚC (lệch 168 giờ = 7 ngày x 24h).

    df_context: dữ liệu ngay trước df_target theo thời gian (để lấy giá trị tuần trước
                cho các dòng đầu của df_target), thường là train+val nối với test.
    df_target:  tập cần dự báo (val hoặc test), phải có cột 'dteday' và 'hr'.
    """
    full = pd.concat([df_context, df_target], ignore_index=True).sort_values(["dteday", "hr"])
    full = full.drop_duplicates(subset=["dteday", "hr"]).reset_index(drop=True)
    full = full.set_index(pd.to_datetime(full["dteday"]).astype(str) + "-" + full["hr"].astype(str))

    ts = full.set_index(pd.to_datetime(full["dteday"]) + pd.to_timedelta(full["hr"], unit="h"))["cnt"]
    ts = ts[~ts.index.duplicated(keep="first")].sort_index()

    target_ts = pd.to_datetime(df_target["dteday"]) + pd.to_timedelta(df_target["hr"].values, unit="h")
    lagged_ts = target_ts - pd.Timedelta(days=7)

    preds = ts.reindex(lagged_ts).values
    # Nếu thiếu giá trị tuần trước (đầu chuỗi), điền bằng trung bình theo giờ trong df_context
    if np.isnan(preds).any():
        hourly_mean = df_context.groupby("hr")["cnt"].mean()
        fallback = df_target["hr"].map(hourly_mean).values
        preds = np.where(np.isnan(preds), fallback, preds)
    return np.clip(preds, 0, None)
