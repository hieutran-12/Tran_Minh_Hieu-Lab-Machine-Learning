"""Tải và làm sạch Metro Interstate Traffic Volume (UCI) -> chuỗi theo giờ + đặc trưng."""
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
URL = "https://archive.ics.uci.edu/static/public/492/metro+interstate+traffic+volume.zip"
PAGE = "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume"
PREFIX = "Metro_Interstate_Traffic_Volume"

MAX_GAP = 3        # lỗ hổng <= 3 giờ: nội suy; dài hơn: cắt thành đoạn riêng
TEMP_MIN_K = 200   # temp thấp hơn (vd 0 K) là phi lý
RAIN_MAX_MM = 100  # rain_1h cao hơn (vd 9831 mm/giờ) là phi lý

WEATHER = ["Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
           "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"]
NUM_COLS = ["temp", "rain_1h", "snow_1h", "clouds_all"]
SCALE_COLS = ["traffic_volume", "temp", "rain_1h", "snow_1h", "clouds_all"]  # cột đầu = nhãn
FEATURES = (SCALE_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday", "is_filled"]
            + [f"w_{w}" for w in WEATHER])


def _find_raw(raw_dir):
    files = [p for p in sorted(raw_dir.rglob(f"{PREFIX}*")) if p.suffix in (".csv", ".gz")]
    return files[0] if files else None


def download(raw_dir=RAW_DIR):
    """Tải & giải nén dữ liệu UCI (bỏ qua nếu đã có sẵn trong data/raw)."""
    raw_dir = Path(raw_dir)
    path = _find_raw(raw_dir) if raw_dir.exists() else None
    if path:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Đang tải dữ liệu: {URL}")
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(raw_dir)
    except Exception as e:
        raise RuntimeError(
            f"Không tải được dữ liệu ({e}).\nHãy tải thủ công tại {PAGE}, giải nén vào {raw_dir} "
            "hoặc chạy với --csv <đường_dẫn_file>."
        ) from e
    path = _find_raw(raw_dir)
    if path is None:
        raise FileNotFoundError(f"Không thấy file {PREFIX}* sau khi giải nén vào {raw_dir}")
    return path


def load_raw(csv_path=None):
    path = Path(csv_path) if csv_path else download()
    df = pd.read_csv(path)  # tự nhận .csv / .csv.gz
    need = {"holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "date_time", "traffic_volume"}
    if not need <= set(df.columns):
        raise ValueError(f"File thiếu cột: {sorted(need - set(df.columns))}")
    return df


def _fill_short_gaps(s, max_gap=MAX_GAP):
    """Nội suy theo thời gian CHỈ các đoạn NaN ngắn (<= max_gap giờ); đoạn dài giữ nguyên NaN."""
    na = s.isna()
    run_len = na.groupby((na != na.shift()).cumsum()).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    return s.where(~(na & (run_len <= max_gap)), filled)


def clean(raw):
    """Xử lý 4 vấn đề dữ liệu. Trả về (hourly, report).

    hourly: chỉ số giờ liên tục; cột `valid` = giờ dùng được, `segment` = id đoạn liên tục (-1 nếu không hợp lệ).
    """
    df = raw.copy()
    df["date_time"] = pd.to_datetime(df["date_time"]).dt.floor("h")
    df["holiday"] = df["holiday"].fillna("None")  # pandas>=2 đọc chữ 'None' thành NaN

    # (4) holiday chỉ đánh dấu dòng đầu ngày lễ -> lan ra toàn bộ ngày
    holiday_days = pd.DatetimeIndex(df.loc[df["holiday"] != "None", "date_time"].dt.normalize().unique())

    # (2) ngoại lai phi lý -> NaN (sẽ nội suy nếu ngắn)
    bad_temp, bad_rain = df["temp"] < TEMP_MIN_K, df["rain_1h"] > RAIN_MAX_MM
    df.loc[bad_temp, "temp"] = np.nan
    df.loc[bad_rain, "rain_1h"] = np.nan

    # (3) khử trùng date_time (nhiều mô tả thời tiết cho cùng 1 giờ)
    agg = {c: "mean" for c in NUM_COLS + ["traffic_volume"]}
    agg["weather_main"] = "first"
    hourly = df.groupby("date_time").agg(agg)

    # (1) reindex theo giờ đầy đủ -> lỗ hổng lộ ra thành NaN
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"))
    hourly.index.name = "date_time"
    observed = hourly["traffic_volume"].notna()
    for c in NUM_COLS + ["traffic_volume"]:
        hourly[c] = _fill_short_gaps(hourly[c])
    valid = hourly["traffic_volume"].notna()
    hourly["is_filled"] = (valid & ~observed).astype(float)  # cột cờ: giờ được nội suy
    hourly[NUM_COLS] = hourly[NUM_COLS].fillna(hourly.loc[valid, NUM_COLS].median())
    hourly["weather_main"] = hourly["weather_main"].ffill().bfill()
    hourly["is_holiday"] = hourly.index.normalize().isin(holiday_days).astype(float)
    hourly["valid"] = valid
    hourly["segment"] = (valid != valid.shift()).cumsum().where(valid, -1)

    report = {
        "Khoảng thời gian": f"{hourly.index.min():%Y-%m-%d} → {hourly.index.max():%Y-%m-%d}",
        "Số dòng gốc": len(df),
        "(3) Dòng trùng date_time đã khử": len(df) - df["date_time"].nunique(),
        "(2) temp phi lý (<200K) đã lọc": int(bad_temp.sum()),
        "(2) rain_1h phi lý (>100mm) đã lọc": int(bad_rain.sum()),
        "(1) Giờ thiếu trong khoảng thời gian": int((~observed).sum()),
        f"(1)   - nội suy (lỗ hổng <= {MAX_GAP}h, có cờ is_filled)": int(hourly["is_filled"].sum()),
        f"(1)   - giữ trống, cắt đoạn (lỗ hổng > {MAX_GAP}h)": int((~valid).sum()),
        "(1) Số đoạn liên tục": int(hourly.loc[valid, "segment"].nunique()),
        "Số giờ dùng được": int(valid.sum()),
        "(4) Dòng holiday gốc → số giờ is_holiday=1 sau khi lan ra cả ngày":
            f"{int((df['holiday'] != 'None').sum())} → {int(hourly['is_holiday'].sum())}",
    }
    return hourly, report


def build_features(hourly):
    """Đặc trưng: lưu lượng, thời tiết, sin/cos giờ & thứ, cờ ngày lễ, cờ nội suy, one-hot weather_main."""
    idx = hourly.index
    f = pd.DataFrame(index=idx)
    f["traffic_volume"] = hourly["traffic_volume"]
    f["temp"] = hourly["temp"]
    f["rain_1h"] = np.log1p(hourly["rain_1h"])  # lệch phải mạnh -> log
    f["snow_1h"] = np.log1p(hourly["snow_1h"])
    f["clouds_all"] = hourly["clouds_all"]
    f["hour_sin"] = np.sin(2 * np.pi * idx.hour.to_numpy() / 24)
    f["hour_cos"] = np.cos(2 * np.pi * idx.hour.to_numpy() / 24)
    f["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek.to_numpy() / 7)
    f["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek.to_numpy() / 7)
    f["is_holiday"] = hourly["is_holiday"]
    f["is_filled"] = hourly["is_filled"]
    for w in WEATHER:
        f[f"w_{w}"] = (hourly["weather_main"] == w).astype(float)
    return f[FEATURES]
