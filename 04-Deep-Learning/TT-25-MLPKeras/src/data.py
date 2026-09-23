"""
data.py
-------
Nạp và tiền xử lý dữ liệu cho bài toán chấm điểm khách hàng tiềm năng
mua bảo hiểm ô tô (Health Insurance Cross Sell Prediction).

Dataset gốc (Kaggle):
https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Cách dùng:
    1. Tải file train.csv từ Kaggle, đặt vào: data/train.csv
    2. from src.data import load_raw, build_preprocessor, split_data
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

TARGET = "Response"

# Các cột phân loại mức thấp -> one-hot bình thường
LOW_CARD_CATS = ["Gender", "Driving_License", "Previously_Insured", "Vehicle_Age", "Vehicle_Damage"]

# Các cột phân loại nhiều mức -> ứng viên cho Embedding (xem model.py)
HIGH_CARD_CATS = ["Region_Code", "Policy_Sales_Channel"]

NUMERIC_COLS = ["Age", "Annual_Premium", "Vintage"]


def load_raw(csv_path: str) -> pd.DataFrame:
    """Đọc file CSV gốc tải từ Kaggle (train.csv)."""
    df = pd.read_csv(csv_path)
    expected = set(LOW_CARD_CATS + HIGH_CARD_CATS + NUMERIC_COLS + [TARGET, "id"])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"File dữ liệu thiếu các cột bắt buộc: {missing}. "
            "Hãy chắc chắn bạn đã tải đúng file train.csv từ Kaggle."
        )
    return df


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """log1p cho Annual_Premium (lệch phải mạnh) + ép kiểu categorical."""
    df = df.copy()
    df["Annual_Premium_log1p"] = np.log1p(df["Annual_Premium"])
    df = df.drop(columns=["Annual_Premium"])

    # Chuẩn hoá kiểu dữ liệu cho các biến phân loại nhiều mức (số nguyên -> mã liên tục 0..n-1)
    for col in HIGH_CARD_CATS:
        df[col] = df[col].astype(int)
    return df


def encode_low_card(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame):
    """One-hot cho các biến phân loại ít mức. Fit trên train, transform trên val/test."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df_train[LOW_CARD_CATS])

    def _transform(df):
        arr = encoder.transform(df[LOW_CARD_CATS])
        cols = encoder.get_feature_names_out(LOW_CARD_CATS)
        return pd.DataFrame(arr, columns=cols, index=df.index)

    return _transform(df_train), _transform(df_val), _transform(df_test), encoder


def scale_numeric(df_train, df_val, df_test):
    """StandardScaler cho các biến số liên tục (fit trên train)."""
    numeric_cols = NUMERIC_COLS[:1] + ["Annual_Premium_log1p"] + NUMERIC_COLS[2:]
    scaler = StandardScaler()
    scaler.fit(df_train[numeric_cols])

    def _transform(df):
        arr = scaler.transform(df[numeric_cols])
        return pd.DataFrame(arr, columns=numeric_cols, index=df.index)

    return _transform(df_train), _transform(df_val), _transform(df_test), scaler, numeric_cols


def remap_high_card_ids(df_train, df_val, df_test):
    """
    Ánh xạ Region_Code, Policy_Sales_Channel về khoảng [0, n_unique-1] dựa trên train,
    để dùng làm chỉ số cho Embedding layer. Giá trị chưa thấy ở val/test -> bucket "unknown".
    Trả về: (train_ids, val_ids, test_ids, vocab_sizes: dict)
    """
    mappings = {}
    vocab_sizes = {}
    for col in HIGH_CARD_CATS:
        uniques = sorted(df_train[col].unique())
        mapping = {v: i for i, v in enumerate(uniques)}
        unknown_id = len(uniques)  # bucket dự phòng cho giá trị lạ
        mappings[col] = (mapping, unknown_id)
        vocab_sizes[col] = len(uniques) + 1  # +1 cho unknown

    def _transform(df):
        out = {}
        for col in HIGH_CARD_CATS:
            mapping, unknown_id = mappings[col]
            out[col] = df[col].map(mapping).fillna(unknown_id).astype(int).values
        return out

    return _transform(df_train), _transform(df_val), _transform(df_test), vocab_sizes


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """Chia train/val/test có stratify theo nhãn Response."""
    df_train, df_temp = train_test_split(
        df, test_size=(test_size + val_size), stratify=df[TARGET], random_state=random_state
    )
    rel_test = test_size / (test_size + val_size)
    df_val, df_test = train_test_split(
        df_temp, test_size=rel_test, stratify=df_temp[TARGET], random_state=random_state
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def build_dense_features(df_train, df_val, df_test):
    """
    Xây ma trận đặc trưng "dense" (one-hot + numeric) dùng cho:
      - baseline LightGBM/XGBoost
      - MLP không dùng Embedding (one-hot toàn bộ, kể cả biến nhiều mức)
    """
    df_train = basic_feature_engineering(df_train)
    df_val = basic_feature_engineering(df_val)
    df_test = basic_feature_engineering(df_test)

    oh_train, oh_val, oh_test, oh_encoder = encode_low_card(df_train, df_val, df_test)
    num_train, num_val, num_test, scaler, numeric_cols = scale_numeric(df_train, df_val, df_test)

    X_train = pd.concat([oh_train, num_train], axis=1)
    X_val = pd.concat([oh_val, num_val], axis=1)
    X_test = pd.concat([oh_test, num_test], axis=1)

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    artifacts = {"oh_encoder": oh_encoder, "scaler": scaler, "numeric_cols": numeric_cols}
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), artifacts


def build_dense_features_with_highcard_onehot(df_train, df_val, df_test):
    """Biến thể one-hot TOÀN BỘ (bao gồm Region_Code, Policy_Sales_Channel) -> ma trận thưa 200+ cột.
    Dùng để so sánh với hướng Embedding (mục 10 trong README)."""
    df_train = basic_feature_engineering(df_train)
    df_val = basic_feature_engineering(df_val)
    df_test = basic_feature_engineering(df_test)

    all_cats = LOW_CARD_CATS + HIGH_CARD_CATS
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(df_train[all_cats])

    def _oh(df):
        arr = encoder.transform(df[all_cats])
        cols = encoder.get_feature_names_out(all_cats)
        return pd.DataFrame(arr, columns=cols, index=df.index)

    num_train, num_val, num_test, scaler, numeric_cols = scale_numeric(df_train, df_val, df_test)

    X_train = pd.concat([_oh(df_train), num_train], axis=1)
    X_val = pd.concat([_oh(df_val), num_val], axis=1)
    X_test = pd.concat([_oh(df_test), num_test], axis=1)

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_embedding_features(df_train, df_val, df_test):
    """
    Xây bộ đặc trưng cho kiến trúc MLP + Embedding:
      - inputs["dense"]: one-hot (biến ít mức) + numeric, đã scale
      - inputs["Region_Code"], inputs["Policy_Sales_Channel"]: chỉ số nguyên cho Embedding layer
    """
    df_train = basic_feature_engineering(df_train)
    df_val = basic_feature_engineering(df_val)
    df_test = basic_feature_engineering(df_test)

    oh_train, oh_val, oh_test, oh_encoder = encode_low_card(df_train, df_val, df_test)
    num_train, num_val, num_test, scaler, numeric_cols = scale_numeric(df_train, df_val, df_test)

    dense_train = pd.concat([oh_train, num_train], axis=1).values.astype("float32")
    dense_val = pd.concat([oh_val, num_val], axis=1).values.astype("float32")
    dense_test = pd.concat([oh_test, num_test], axis=1).values.astype("float32")

    id_train, id_val, id_test, vocab_sizes = remap_high_card_ids(df_train, df_val, df_test)

    X_train = {"dense": dense_train, **{k: v for k, v in id_train.items()}}
    X_val = {"dense": dense_val, **{k: v for k, v in id_val.items()}}
    X_test = {"dense": dense_test, **{k: v for k, v in id_test.items()}}

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    meta = {
        "dense_dim": dense_train.shape[1],
        "vocab_sizes": vocab_sizes,
        "oh_encoder": oh_encoder,
        "scaler": scaler,
    }
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), meta
