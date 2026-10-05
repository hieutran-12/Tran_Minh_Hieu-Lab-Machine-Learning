"""Hàm tiện ích: seed, làm sạch văn bản, lưu/đọc kết quả, đo thời gian, calibration."""
import html
import json
import random
import re
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config import CLASSES, REPORTS, SEED


def set_seed(seed=SEED):
    """Cố định seed cho các framework đã được import."""
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    if "torch" in sys.modules:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def clean_text(s):
    """Giải mã HTML entity, bỏ ký tự '\\' thừa của AG News, gọn khoảng trắng."""
    s = html.unescape(str(s)).replace("\\", " ")
    return re.sub(r"\s+", " ", s).strip()


def tinh_metrics(y, logits):
    pred = np.argmax(logits, axis=1)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro"))}


def bao_cao_phan_loai(y, logits):
    """In classification report (precision/recall/F1 từng chuyên mục)."""
    print(classification_report(y, np.argmax(logits, axis=1), target_names=CLASSES, digits=4))


def do_ms_mot_tin(predict_one, texts, n=100, warmup=5):
    """Thời gian dự đoán 1 tin (ms): chạy predict_one(text) lần lượt trên n tin, lấy trung bình."""
    texts = list(texts)[:n]
    for t in texts[:warmup]:
        predict_one(t)
    t0 = time.perf_counter()
    for t in texts:
        predict_one(t)
    return (time.perf_counter() - t0) / len(texts) * 1000


def luu_ket_qua(name, metrics, val_logits, val_y, test_logits, test_y):
    """Lưu metrics (json) + logits val/test (npz) để so_sanh.py dùng lại."""
    metrics = dict(metrics, **tinh_metrics(test_y, test_logits))

    (REPORTS / f"metrics_{name}.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    np.savez_compressed(
        REPORTS / f"logits_{name}.npz",
        val_logits=val_logits,
        val_y=val_y,
        test_logits=test_logits,
        test_y=test_y
    )

    return metrics


def doc_ket_qua(name):
    m = json.loads((REPORTS / f"metrics_{name}.json").read_text())
    z = np.load(REPORTS / f"logits_{name}.npz")
    return m, {k: z[k] for k in z.files}


def fit_temperature(val_logits, val_y):
    """Temperature scaling: tìm T tối thiểu hoá NLL trên val -> xác suất tin cậy được hiệu chỉnh."""
    def nll(logT):
        lp = log_softmax(val_logits / np.exp(logT), axis=1)
        return -lp[np.arange(len(val_y)), val_y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def xac_suat(logits, T=1.0):
    return softmax(logits / T, axis=1)
