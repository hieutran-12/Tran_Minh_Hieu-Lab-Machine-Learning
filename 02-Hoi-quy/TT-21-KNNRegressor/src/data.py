"""Nạp dữ liệu California Housing, xử lý outlier và dựng pipeline KNN."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DAC_TRUNG = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
             "Population", "AveOccup", "Latitude", "Longitude"]
NHAN = "MedHouseVal"
COT_VI_TRI = ["Latitude", "Longitude"]
RANDOM_STATE = 42

# Ngưỡng cắt outlier (giá trị vô lý do tỉ số tổng/hộ gia đình)
NGUONG_OUTLIER = {"AveRooms": 15.0, "AveBedrms": 4.0, "AveOccup": 10.0}


def tai_du_lieu(loc_outlier=True, bo_gia_bi_chan=False):
    """Trả về DataFrame gồm 8 đặc trưng + nhãn MedHouseVal (đơn vị 100.000 USD)."""
    df = fetch_california_housing(as_frame=True).frame.copy()

    if loc_outlier:
        mask = np.ones(len(df), dtype=bool)
        for cot, nguong in NGUONG_OUTLIER.items():
            mask &= df[cot] <= nguong
        df = df[mask]

    # Nhãn bị chặn trần tại 5.0 (500.000 USD) — tuỳ chọn loại bỏ
    if bo_gia_bi_chan:
        df = df[df[NHAN] < 5.0]

    return df.reset_index(drop=True)


def chia_du_lieu(df, test_size=0.2, random_state=RANDOM_STATE):
    """Chia train/test, giữ nguyên tên cột để còn tra cứu 'căn tương tự'."""
    X = df[DAC_TRUNG]
    y = df[NHAN]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


class TrongSoViTri(BaseEstimator, TransformerMixin):
    """Nhân các cột toạ độ với hệ số w SAU khi chuẩn hoá.

    w = 1 -> mọi đặc trưng bình đẳng. w > 1 -> ép KNN ưu tiên hàng xóm địa lý.
    """

    def __init__(self, w=1.0, cot_vi_tri=(6, 7)):
        self.w = w
        self.cot_vi_tri = cot_vi_tri

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, list(self.cot_vi_tri)] *= self.w
        return X


def tao_pipeline(k=10, weights="distance", metric="minkowski", p=2,
                 w_vi_tri=1.0, chuan_hoa=True):
    """Pipeline chuẩn: StandardScaler -> (trọng số vị trí) -> KNeighborsRegressor."""
    buoc = []
    if chuan_hoa:
        buoc.append(("scale", StandardScaler()))
    if w_vi_tri != 1.0:
        idx = tuple(DAC_TRUNG.index(c) for c in COT_VI_TRI)
        buoc.append(("vitri", TrongSoViTri(w=w_vi_tri, cot_vi_tri=idx)))
    buoc.append(("knn", KNeighborsRegressor(n_neighbors=k, weights=weights,
                                            metric=metric, p=p, n_jobs=-1)))
    return Pipeline(buoc)


def cac_can_tuong_tu(pipe, X_train, y_train, x_query, k=5):
    """Trả về (DataFrame k căn tương tự + giá + khoảng cách, giá dự đoán)."""
    x_query = pd.DataFrame(x_query).T if isinstance(x_query, pd.Series) else x_query
    tien_xu_ly = pipe[:-1]
    xq = tien_xu_ly.transform(x_query) if len(pipe) > 1 else np.asarray(x_query)
    kc, chi_so = pipe[-1].kneighbors(xq, n_neighbors=k)

    bang = X_train.iloc[chi_so[0]].copy()
    bang["Gia_thuc"] = y_train.iloc[chi_so[0]].values
    bang["Khoang_cach"] = kc[0]
    return bang, float(pipe.predict(x_query)[0])
