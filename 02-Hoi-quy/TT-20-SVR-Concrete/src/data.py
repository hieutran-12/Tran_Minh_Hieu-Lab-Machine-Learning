"""
Tải và chuẩn bị bộ dữ liệu Concrete Compressive Strength (UCI, 1030 dòng).

Thứ tự ưu tiên khi nạp dữ liệu:
  1. File người dùng chỉ định qua --data
  2. File có sẵn trong thư mục data/ (Concrete_Data.xls / .csv)
  3. Tự tải từ UCI (cần mạng)
"""
from __future__ import annotations

import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

UCI_ZIP = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# Tên cột chuẩn hoá (theo đúng thứ tự cột gốc của UCI)
COLS = [
    "Cement",
    "BlastFurnaceSlag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAggregate",
    "FineAggregate",
    "Age",
    "Strength",
]

BASE_FEATURES = COLS[:-1]
DOMAIN_FEATURES = [
    "water_cement_ratio",
    "water_binder_ratio",
    "total_binder",
    "log_age",
    "aggregate_binder_ratio",
    "sp_binder_ratio",
]
TARGET = "Strength"


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def _read_table(path: Path) -> pd.DataFrame:
    """Đọc .xls/.xlsx/.csv và chuẩn hoá tên cột."""
    if path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if df.shape[1] != 9:
        raise ValueError(
            f"File {path.name} có {df.shape[1]} cột, cần đúng 9 cột "
            "(8 đặc trưng + cường độ)."
        )
    df.columns = COLS
    return df.astype(float)


def _download() -> Path:
    """Tải zip từ UCI, giải nén file .xls vào data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[data] Đang tải từ UCI: {UCI_ZIP}")
    with urllib.request.urlopen(UCI_ZIP, timeout=60) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv")))
        out = DATA_DIR / Path(name).name
        out.write_bytes(zf.read(name))
    print(f"[data] Đã lưu: {out}")
    return out


def load_raw(path: str | Path | None = None, drop_duplicates: bool = True) -> pd.DataFrame:
    """Trả về DataFrame gốc 9 cột đã chuẩn hoá tên."""
    if path is not None:
        df = _read_table(Path(path))
    else:
        local = [p for p in sorted(DATA_DIR.glob("*")) if p.suffix.lower() in {".xls", ".xlsx", ".csv"}]
        if local:
            df = _read_table(local[0])
        else:
            try:
                df = _read_table(_download())
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(
                    "Không tải được dữ liệu tự động.\n"
                    f"  Lý do: {exc}\n"
                    "  Cách xử lý: tải thủ công tại\n"
                    "  https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength\n"
                    f"  rồi đặt file Concrete_Data.xls vào thư mục: {DATA_DIR}"
                ) from exc

    n0 = len(df)
    if drop_duplicates:
        # Bộ UCI có ~25 dòng trùng hoàn toàn -> gây rò rỉ khi chia train/test
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] {n0} dòng -> {len(df)} dòng sau khi bỏ trùng lặp, {df.shape[1]} cột")
    return df


# --------------------------------------------------------------------------- #
# Đặc trưng từ kiến thức miền
# --------------------------------------------------------------------------- #
def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Thêm các đặc trưng mà model không tự nghĩ ra được:
      - water_cement_ratio : định luật Abrams, yếu tố số 1 quyết định cường độ
      - water_binder_ratio : tính cả tro bay + xỉ lò cao (chất kết dính bổ sung)
      - total_binder       : tổng chất kết dính
      - log_age            : cường độ tăng gần tuyến tính theo log(tuổi)
      - aggregate/sp ratio : tỉ lệ cốt liệu và phụ gia siêu dẻo trên chất kết dính
    """
    out = df.copy()
    eps = 1e-9
    binder = out["Cement"] + out["BlastFurnaceSlag"] + out["FlyAsh"]

    out["total_binder"] = binder
    out["water_cement_ratio"] = out["Water"] / (out["Cement"] + eps)
    out["water_binder_ratio"] = out["Water"] / (binder + eps)
    out["log_age"] = np.log1p(out["Age"])
    out["aggregate_binder_ratio"] = (out["CoarseAggregate"] + out["FineAggregate"]) / (binder + eps)
    out["sp_binder_ratio"] = out["Superplasticizer"] / (binder + eps)
    return out


def get_xy(df: pd.DataFrame, use_domain: bool = True):
    """Tách X, y. use_domain=False -> chỉ dùng 8 đặc trưng gốc."""
    cols = list(BASE_FEATURES)
    if use_domain:
        df = add_domain_features(df)
        cols += DOMAIN_FEATURES
    return df[cols].to_numpy(dtype=float), df[TARGET].to_numpy(dtype=float), cols
