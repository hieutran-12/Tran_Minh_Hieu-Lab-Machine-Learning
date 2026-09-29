"""Dữ liệu: tải, đếm, tách validation, tf.data pipeline, hiển thị ảnh mẫu.

Quy ước nhãn: NORMAL = 0, PNEUMONIA = 1 (PNEUMONIA là lớp "dương tính").
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

KAGGLE_HANDLE = "paultimothymooney/chest-xray-pneumonia"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (224, 224)
IMAGE_EXTS = {".jpeg", ".jpg", ".png"}
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Tìm / tải dữ liệu
# --------------------------------------------------------------------------- #
def find_dataset_root(base: str | Path) -> Path:
    """Tìm thư mục chứa train/ và test/ (mỗi thư mục có NORMAL/ và PNEUMONIA/).

    Bản Kaggle thường có thư mục lồng nhau (chest_xray/chest_xray/...) và thư
    mục rác __MACOSX nên cần dò tìm thay vì giả định đường dẫn cố định.
    """
    base = Path(base)
    candidates = [base] + sorted(
        (p for p in base.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts),
        key=lambda p: len(p.parts),
    )
    for cand in candidates:
        if all((cand / split / cls).is_dir() for split in ("train", "test") for cls in CLASS_NAMES):
            return cand
    raise FileNotFoundError(
        f"Không tìm thấy cấu trúc train/test/{{NORMAL,PNEUMONIA}} trong '{base}'."
    )


def get_data_root(data_dir: str | Path | None = None) -> Path:
    """Trả về thư mục gốc của dataset.

    Thứ tự ưu tiên: --data_dir do người dùng chỉ định  ->  thư mục ./data
    (nếu đã có dữ liệu)  ->  tự tải từ Kaggle bằng kagglehub.
    """
    if data_dir is not None:
        return find_dataset_root(data_dir)
    if DEFAULT_DATA_DIR.exists():
        try:
            return find_dataset_root(DEFAULT_DATA_DIR)
        except FileNotFoundError:
            pass
    import kagglehub  # import muộn để không bắt buộc khi đã có dữ liệu local

    print(f"Đang tải dataset '{KAGGLE_HANDLE}' từ Kaggle (~1,2 GB)...")
    return find_dataset_root(kagglehub.dataset_download(KAGGLE_HANDLE))


# --------------------------------------------------------------------------- #
# 2. Liệt kê ảnh, đếm, tách validation
# --------------------------------------------------------------------------- #
_PNEUMONIA_ID = re.compile(r"^person(\d+)_")
_NORMAL_ID = re.compile(r"IM-(\d+)")


def patient_id(filename: str) -> str:
    """Suy ra mã bệnh nhân từ tên file (phục vụ chia theo bệnh nhân).

    person123_bacteria_456.jpeg -> P123 ; IM-0115-0001.jpeg -> N0115.
    Nếu không đoán được thì mỗi ảnh coi là một bệnh nhân riêng.
    """
    m = _PNEUMONIA_ID.match(filename)
    if m:
        return f"P{m.group(1)}"
    m = _NORMAL_ID.search(filename)
    if m:
        return f"N{m.group(1)}"
    return filename


def list_images(root: str | Path, split: str) -> pd.DataFrame:
    """Bảng (path, label, group) cho một tập gốc: 'train' | 'val' | 'test'."""
    rows = []
    for label, cls in enumerate(CLASS_NAMES):
        folder = Path(root) / split / cls
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("."):
                rows.append((str(p), label, patient_id(p.name)))
    return pd.DataFrame(rows, columns=["path", "label", "group"])


def count_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Đếm số ảnh mỗi lớp mỗi tập, kèm tỉ lệ PNEUMONIA."""
    rows = {}
    for name, df in splits.items():
        n_normal = int((df.label == 0).sum())
        n_pneu = int((df.label == 1).sum())
        total = n_normal + n_pneu
        rows[name] = {
            "NORMAL": n_normal,
            "PNEUMONIA": n_pneu,
            "Tổng": total,
            "% PNEUMONIA": round(100 * n_pneu / total, 1) if total else 0.0,
        }
    table = pd.DataFrame(rows).T
    return table.astype({"NORMAL": int, "PNEUMONIA": int, "Tổng": int})


def split_train_val(train_df: pd.DataFrame, val_fraction: float = 0.15):
    """Tách lại validation từ tập train gốc.

    - Phân tầng (stratify) theo nhãn để giữ tỉ lệ NORMAL/PNEUMONIA.
    - Nhóm theo bệnh nhân: một bệnh nhân có nhiều ảnh, nếu ảnh của cùng một
      người rơi vào cả train và val thì val bị "rò rỉ" và điểm số ảo.
    """
    n_splits = max(2, round(1 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(train_df, train_df.label, train_df.group))
    tr = train_df.iloc[tr_idx].reset_index(drop=True)
    va = train_df.iloc[va_idx].reset_index(drop=True)
    assert set(tr.group).isdisjoint(va.group), "Rò rỉ bệnh nhân giữa train và val"
    return tr, va


def compute_class_weights(df: pd.DataFrame) -> dict[int, float]:
    """class_weight cân bằng: w_c = N / (2 * N_c)."""
    n = len(df)
    return {c: n / (2 * int((df.label == c).sum())) for c in (0, 1)}


# --------------------------------------------------------------------------- #
# 3. Hiển thị ảnh mẫu
# --------------------------------------------------------------------------- #
def plot_samples(df: pd.DataFrame, n_per_class: int = 8, out_path: str | Path | None = None):
    """Vẽ n ảnh ngẫu nhiên mỗi lớp (mỗi hàng một lớp)."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(2, n_per_class, figsize=(2.2 * n_per_class, 5.2))
    for r, cls in enumerate(CLASS_NAMES):
        paths = df.loc[df.label == r, "path"].to_numpy()
        pick = rng.choice(paths, size=min(n_per_class, len(paths)), replace=False)
        for c in range(n_per_class):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(pick):
                ax.imshow(Image.open(pick[c]).convert("L"), cmap="gray")
                if c == 0:
                    ax.set_title(cls, loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("Ảnh mẫu: hàng trên NORMAL, hàng dưới PNEUMONIA")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    return fig


# --------------------------------------------------------------------------- #
# 4. tf.data pipeline
# --------------------------------------------------------------------------- #
def _load(path, label):
    """Đọc file -> RGB 224x224 (uint8) để cache gọn bộ nhớ."""
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, antialias=True)
    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    return img, tf.cast(label, tf.float32)


def load_image(path: str | Path) -> np.ndarray:
    """Ảnh float32 (224, 224, 3), thang 0-255, tiền xử lý y hệt pipeline."""
    img, _ = _load(tf.constant(str(path)), 0)
    return img.numpy().astype("float32")


def build_augmentation() -> tf.keras.Sequential:
    """Augmentation hợp lý với y tế.

    Chỉ dùng: xoay ±10°, dịch ±10%, phóng to/thu nhỏ ±10%, đổi sáng/tương phản nhẹ.
    KHÔNG lật ngang (tim nằm bên trái, lật là sai giải phẫu) và KHÔNG lật dọc.
    """
    L = tf.keras.layers
    return tf.keras.Sequential(
        [
            L.RandomRotation(10 / 360, fill_mode="constant"),
            L.RandomTranslation(0.1, 0.1, fill_mode="constant"),
            L.RandomZoom(0.1, fill_mode="constant"),
            L.RandomBrightness(0.1, value_range=(0, 255)),
            L.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def make_dataset(df: pd.DataFrame, batch_size: int = 32, training: bool = False, cache: bool = True):
    """Dataset (ảnh float32 0-255, nhãn). Augmentation CHỈ áp dụng khi training=True.

    Việc chuẩn hoá giá trị điểm ảnh nằm trong model (Rescaling / tiền xử lý
    của EfficientNet) nên ảnh ở đây giữ thang 0-255.
    """
    if training:  # xáo một lần; phần xáo mỗi epoch nhờ shuffle buffer bên dưới
        df = df.sample(frac=1.0, random_state=SEED)
    ds = tf.data.Dataset.from_tensor_slices((df.path.to_numpy(), df.label.to_numpy().astype("float32")))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        ds = ds.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
