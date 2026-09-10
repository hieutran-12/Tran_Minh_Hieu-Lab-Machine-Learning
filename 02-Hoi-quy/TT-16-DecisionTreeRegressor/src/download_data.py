"""
TT-16 — Tải DỮ LIỆU THẬT từ NYC TLC và chuẩn hoá đúng schema mà train.py cần.

Đây là bản THAY THẾ cho generate_data.py (dữ liệu mô phỏng) sau phản hồi chấm
điểm: "phải dùng dữ liệu thật, không dùng dữ liệu giả".

⚠️ VÌ SAO SCRIPT NÀY KHÔNG THỂ TỰ CHẠY TRONG MÔI TRƯỜNG CLAUDE SINH RA BÀI:
môi trường sandbox dùng để soạn bài chỉ được phép gọi mạng ra một danh sách
domain cố định (pypi, npm, github...), KHÔNG gồm nyc.gov hay CDN
d37ci6vzurychx.cloudfront.net nơi TLC lưu file yellow_tripdata_*.parquet. Vì
vậy Claude không tự tải và tự chạy lại toàn bộ pipeline trên dữ liệu thật được.

→ Cách dùng: chạy script này TRÊN MÁY CỦA BẠN (có mạng bình thường), nó sẽ:
  1. Tải file yellow_tripdata_<year-month>.parquet thật từ TLC
  2. Chỉ giữ các cột hợp lệ (không đụng tới tip_amount/tolls_amount/total_amount
     ở bước train — các cột này vẫn được giữ trong file trung gian để tính EDA/
     kiểm tra nếu cần, nhưng train.py đã tự loại khỏi FEATURES)
  3. Lấy mẫu ngẫu nhiên 200.000 dòng (random_state=42, giống đúng hướng dẫn gốc)
  4. Join thêm cột pickup_borough từ bảng tra chính thức data/taxi_zone_lookup.csv
     (LocationID -> Borough) để giữ nguyên feature "pickup_borough" đã dùng
  5. Ghi ra ĐÚNG file mà train.py đọc: data/yellow_tripdata_sample_raw.csv

Sau khi chạy xong, chạy lại `python src/train.py` bình thường — không cần sửa
gì thêm, vì output có đúng tên cột như generate_data.py từng sinh ra.

Chạy:
    pip install -r requirements.txt
    python src/download_data.py --year-month 2024-01

Nếu máy bạn cũng bị chặn mạng tới TLC (ví dụ mạng công ty/trường), tải file
.parquet bằng trình duyệt tại:
    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
rồi chạy:
    python src/download_data.py --parquet-path duong/dan/toi/yellow_tripdata_2024-01.parquet
"""

import argparse
import os
import sys

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
ZONE_LOOKUP_PATH = os.path.join(DATA_DIR, "taxi_zone_lookup.csv")
OUT_PATH = os.path.join(DATA_DIR, "yellow_tripdata_sample_raw.csv")
TLC_URL_TMPL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"

RANDOM_STATE = 42
SAMPLE_N = 200_000

# Đúng các cột thật có trong file TLC (không phải cột tự bịa) mà train.py cần:
KEEP_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "mta_tax",
    "improvement_surcharge",
    "congestion_surcharge",
    "tolls_amount",
    "tip_amount",
    "total_amount",
]


def download_parquet(year_month: str, dest: str) -> None:
    if requests is None:
        raise RuntimeError("Thiếu thư viện 'requests' -> pip install requests")
    url = TLC_URL_TMPL.format(ym=year_month)
    print(f"Đang tải {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done / total * 100
                    print(f"\r  {done/1e6:,.1f} / {total/1e6:,.1f} MB ({pct:.0f}%)", end="")
        print()
    print(f"Đã tải xong -> {dest}")


def main():
    ap = argparse.ArgumentParser(description="Tải + chuẩn hoá dữ liệu TLC thật cho TT-16.")
    ap.add_argument("--year-month", default="2024-01", help="vd 2024-01 (mặc định)")
    ap.add_argument(
        "--parquet-path",
        default=None,
        help="Nếu đã tự tải sẵn file .parquet (do mạng chặn tự động), trỏ thẳng đường dẫn vào đây.",
    )
    ap.add_argument("--sample-n", type=int, default=SAMPLE_N, help="Số dòng lấy mẫu (mặc định 200.000)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    raw_parquet = args.parquet_path
    if raw_parquet is None:
        raw_parquet = os.path.join(DATA_DIR, f"yellow_tripdata_{args.year_month}.parquet")
        if not os.path.exists(raw_parquet):
            try:
                download_parquet(args.year_month, raw_parquet)
            except Exception as e:
                print(f"\n[LỖI] Không tự tải được: {e}")
                print("-> Tải thủ công tại https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page")
                print(f"   sau đó chạy lại: python src/download_data.py --parquet-path <file .parquet vừa tải>")
                sys.exit(1)
    elif not os.path.exists(raw_parquet):
        print(f"[LỖI] Không tìm thấy {raw_parquet}")
        sys.exit(1)

    if not os.path.exists(ZONE_LOOKUP_PATH):
        print(f"[LỖI] Thiếu {ZONE_LOOKUP_PATH} (bảng tra Borough theo LocationID, đi kèm sẵn trong repo).")
        sys.exit(1)
    zones = pd.read_csv(ZONE_LOOKUP_PATH)[["LocationID", "Borough"]]

    print(f"Đang đọc {raw_parquet} ...")
    df = pd.read_parquet(raw_parquet, columns=KEEP_COLS)
    n_total = len(df)
    print(f"Tổng số chuyến trong file gốc: {n_total:,}")

    n = min(args.sample_n, n_total)
    df = df.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    df = df.merge(zones, how="left", left_on="PULocationID", right_on="LocationID")
    df = df.rename(columns={"Borough": "pickup_borough"}).drop(columns=["LocationID"])
    df["pickup_borough"] = df["pickup_borough"].fillna("Unknown")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nĐã lưu {len(df):,} dòng DỮ LIỆU THẬT (TLC {args.year_month}) -> {OUT_PATH}")
    print("Tiếp theo, chạy: python src/train.py")


if __name__ == "__main__":
    main()
