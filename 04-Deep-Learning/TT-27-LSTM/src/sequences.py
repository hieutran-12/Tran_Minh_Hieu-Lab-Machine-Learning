"""Chia theo thời gian, chuẩn hoá (không rò rỉ) và tạo chuỗi cửa sổ trượt."""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, SCALE_COLS


def tao_chuoi(X, y, do_dai=24, buoc_du_bao=1):
    """Cửa sổ trượt. do_dai = số giờ quá khứ làm đầu vào; nhãn = giá trị sau `buoc_du_bao` giờ.

    Mẫu i dùng X[i : i+do_dai] (chỉ quá khứ) để dự báo y[i+do_dai+buoc_du_bao-1].
    """
    n = len(X) - do_dai - buoc_du_bao + 1
    if n <= 0:
        return np.empty((0, do_dai, X.shape[1]), np.float32), np.empty(0, np.float32)
    Xs = sliding_window_view(X, do_dai, axis=0)[:n].transpose(0, 2, 1)
    start = do_dai + buoc_du_bao - 1
    return np.ascontiguousarray(Xs, np.float32), y[start:start + n].astype(np.float32)


def split_cuts(hourly, train=0.70, val=0.15):
    """Mốc thời gian bắt đầu val và test (chia theo thời gian trên các giờ hợp lệ, không shuffle)."""
    t = hourly.index[hourly["valid"]]
    return t[int(len(t) * train)], t[int(len(t) * (train + val))]


def fit_scaler(feats, hourly, t_val_start):
    """Scaler CHỈ fit trên phần train, rồi transform toàn bộ."""
    mask = (hourly["valid"] & (hourly.index < t_val_start)).to_numpy()
    scaler = StandardScaler().fit(feats.loc[mask, SCALE_COLS].to_numpy())
    scaled = feats.copy()
    scaled[SCALE_COLS] = scaler.transform(feats[SCALE_COLS].to_numpy())
    return scaled, scaler


def make_samples(scaled, hourly, cuts, seq_len=24, horizon=1):
    """Tạo cửa sổ TRONG TỪNG đoạn liên tục (không cửa sổ nào vắt qua lỗ hổng lớn).

    Mẫu thuộc train/val/test theo thời điểm ĐÍCH. Đầu vào luôn nằm trước thời điểm đích.
    Trả về {"train"|"val"|"test": (X, y, thời_điểm_đích)}.
    """
    X_all = scaled[FEATURES].to_numpy(np.float32)
    y_all = X_all[:, 0]  # traffic_volume (đã chuẩn hoá) là cột đầu
    seg = hourly["segment"].to_numpy()
    Xs, ys, ts = [], [], []
    for s in np.unique(seg[seg >= 0]):
        pos = np.flatnonzero(seg == s)
        a, b = pos[0], pos[-1] + 1
        X, y = tao_chuoi(X_all[a:b], y_all[a:b], seq_len, horizon)
        if len(y):
            first = a + seq_len + horizon - 1
            Xs.append(X), ys.append(y), ts.append(hourly.index[first:first + len(y)])
    X, y, t = np.concatenate(Xs), np.concatenate(ys), ts[0].append(ts[1:])
    t_val, t_test = cuts
    part = np.where(t < t_val, 0, np.where(t < t_test, 1, 2))
    return {name: (X[part == i], y[part == i], t[part == i])
            for i, name in enumerate(["train", "val", "test"])}
