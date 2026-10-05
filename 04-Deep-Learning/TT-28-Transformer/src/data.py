"""Nạp AG News (tự tải qua Hugging Face datasets), chia train/val/test, khám phá dữ liệu."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import CLASSES, REPORTS, SEED, VAL_SIZE
from utils import clean_text


def load_ag_news(max_train=None, max_test=None):
    """Trả về dict {'train','val','test'}, mỗi phần là (list văn bản, mảng nhãn).

    max_train / max_test: giới hạn số mẫu (lấy phân tầng) để chạy thử nhanh; None = dùng đủ
    120.000 train + 7.600 test.
    """
    from datasets import load_dataset
    ds = load_dataset("ag_news")
    x_tr = [clean_text(t) for t in ds["train"]["text"]]
    y_tr = np.array(ds["train"]["label"])
    x_te = [clean_text(t) for t in ds["test"]["text"]]
    y_te = np.array(ds["test"]["label"])

    def sub(x, y, n):
        if n is None or n >= len(x):
            return x, y
        idx, _ = train_test_split(np.arange(len(x)), train_size=n, stratify=y, random_state=SEED)
        return [x[i] for i in idx], y[idx]

    x_tr, y_tr = sub(x_tr, y_tr, max_train)
    x_te, y_te = sub(x_te, y_te, max_test)
    i_tr, i_val = train_test_split(np.arange(len(x_tr)), test_size=VAL_SIZE, stratify=y_tr,
                                   random_state=SEED)
    return {"train": ([x_tr[i] for i in i_tr], y_tr[i_tr]),
            "val": ([x_tr[i] for i in i_val], y_tr[i_val]),
            "test": (x_te, y_te)}


def explore(data, n_mau=5):
    """Bước 1: kiểm tra cân bằng lớp, độ dài văn bản, xem n_mau mẫu mỗi lớp."""
    rows = []
    for split, (x, y) in data.items():
        counts = np.bincount(y, minlength=len(CLASSES))
        rows.append({"tập": split, "số mẫu": len(y), **{c: int(n) for c, n in zip(CLASSES, counts)}})
    df = pd.DataFrame(rows).set_index("tập")
    print(df.to_string(), "\n")

    x_tr, y_tr = data["train"]
    n_tu = pd.Series([len(t.split()) for t in x_tr])
    print(f"Số từ / tin (train): trung bình {n_tu.mean():.1f} · trung vị {n_tu.median():.0f} · "
          f"p95 {n_tu.quantile(.95):.0f} · tối đa {n_tu.max()}\n")

    for k, c in enumerate(CLASSES):
        print(f"=== {c} ===")
        for t in [t for t, y in zip(x_tr, y_tr) if y == k][:n_mau]:
            print("  -", t[:150] + ("…" if len(t) > 150 else ""))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    df[CLASSES].plot.bar(ax=ax[0], rot=0)
    ax[0].set_title("Số mẫu mỗi chuyên mục")
    ax[0].legend(fontsize=8)
    ax[1].hist(n_tu, bins=40, color="#4c72b0")
    ax[1].axvline(n_tu.quantile(.95), color="r", ls="--", label="p95")
    ax[1].set_title("Phân bố số từ / tin")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "kham_pha_du_lieu.png", dpi=130)
    plt.show()
    return df
