"""
TT-16 — [PHƯƠNG ÁN DỰ PHÒNG, KHÔNG PHẢI MẶC ĐỊNH] Sinh dữ liệu mô phỏng.

⚠️ ĐÃ ĐỔI SANG DÙNG DỮ LIỆU THẬT: pipeline chính giờ dùng `src/download_data.py`
để tải dữ liệu THẬT từ TLC (xem file đó). File `generate_data.py` này CHỈ còn
là phương án cuối cùng, dùng khi máy chạy bài không có internet chút nào (kể
cả không tải thủ công được) — không dùng để nộp bài chính thức nữa, vì mọi số
liệu sinh ra từ đây (MAE, MAPE, bảng tra cước) không phản ánh bài toán thực tế.

Nếu buộc phải dùng file này, PHẢI ghi rõ trong báo cáo đây là dữ liệu mô phỏng
và mọi kết luận về mức giá chỉ mang tính minh hoạ pipeline, không dùng được cho
tổng đài thật.

Cơ chế: sinh dữ liệu cùng cấu trúc cột với TLC thật (cùng tên cột, cùng các lỗi
bẩn thường gặp: fare âm, distance=0, passenger_count=0, outlier quãng đường...),
với `fare_amount` tính theo đúng công thức cước taxi NYC công khai (cước mở
cửa, cước theo dặm, phụ phí giờ cao điểm, phụ phí ban đêm, phụ phí tắc nghẽn
Manhattan, giá cố định sân bay JFK).

Chạy: python src/generate_data.py
Sinh ra: data/yellow_tripdata_sample_raw.csv (200.000 dòng, CÓ lẫn dữ liệu bẩn)
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 200_000
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "yellow_tripdata_sample_raw.csv",
)

BOROUGHS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island", "JFK Airport", "LGA Airport"]
BOROUGH_P = [0.42, 0.20, 0.18, 0.08, 0.02, 0.06, 0.04]


def gen_pickup_datetime(n):
    """Rải chuyến trong 1 tháng, tập trung nhiều hơn vào giờ cao điểm 7-10h và 16-20h."""
    days = RNG.integers(1, 29, size=n)
    hour_weights = np.array(
        [1, 1, 1, 1, 1, 2, 4, 7, 8, 6, 5, 5, 5, 5, 5, 6, 7, 8, 8, 7, 6, 4, 3, 2], dtype=float
    )
    hour_weights /= hour_weights.sum()
    hours = RNG.choice(24, size=n, p=hour_weights)
    minutes = RNG.integers(0, 60, size=n)
    seconds = RNG.integers(0, 60, size=n)
    dt = pd.to_datetime(
        {
            "year": 2024,
            "month": 1,
            "day": days,
            "hour": hours,
            "minute": minutes,
            "second": seconds,
        }
    )
    return dt


def main():
    n = N
    pickup_dt = gen_pickup_datetime(n)
    hour = pickup_dt.dt.hour.to_numpy()
    dow = pickup_dt.dt.dayofweek.to_numpy()  # 0=Mon .. 6=Sun

    borough = RNG.choice(BOROUGHS, size=n, p=BOROUGH_P)
    is_airport = np.isin(borough, ["JFK Airport", "LGA Airport"])

    # quãng đường: lognormal cho chuyến thường, dài hơn hẳn cho chuyến sân bay
    base_distance = RNG.lognormal(mean=0.9, sigma=0.65, size=n)
    airport_extra = np.where(is_airport, RNG.uniform(8, 16, size=n), 0.0)
    trip_distance = np.clip(base_distance + airport_extra, 0.1, None)

    passenger_count = RNG.choice([1, 1, 1, 2, 2, 3, 4, 5, 6], size=n)

    is_weekday = dow < 5
    is_rush = is_weekday & (((hour >= 7) & (hour <= 10)) | ((hour >= 16) & (hour <= 20)))
    is_night = (hour >= 22) | (hour <= 5)
    is_manhattan = borough == "Manhattan"

    payment_type = RNG.choice([1, 2], size=n, p=[0.72, 0.28])  # 1=card, 2=cash

    # ---- công thức cước thật (mô phỏng theo quy tắc TLC công khai) ----
    base_fare = 3.00
    per_mile = 2.80
    rush_surcharge = np.where(is_rush, 2.50, 0.0)
    night_surcharge = np.where(is_night, 1.00, 0.0)
    congestion_surcharge = np.where(is_manhattan, 2.75, 0.0)

    meter_fare = base_fare + per_mile * trip_distance + rush_surcharge + night_surcharge + congestion_surcharge
    noise = RNG.normal(0, 1.8, size=n)
    fare_amount = meter_fare + noise

    # giá cố định sân bay JFK<->Manhattan-ish (đặc thù thực tế NYC)
    jfk_flat = borough == "JFK Airport"
    fare_amount = np.where(jfk_flat, RNG.normal(70, 4, size=n), fare_amount)

    fare_amount = np.round(fare_amount, 2)

    mta_tax = np.full(n, 0.50)
    improvement_surcharge = np.full(n, 1.00)
    tolls_amount = np.where(is_airport, RNG.choice([0.0, 6.94], size=n, p=[0.4, 0.6]), 0.0)
    tip_rate = np.where(payment_type == 1, RNG.uniform(0.10, 0.25, size=n), RNG.uniform(0.0, 0.03, size=n))
    tip_amount = np.round(np.maximum(fare_amount, 0) * tip_rate, 2)
    total_amount = np.round(
        fare_amount + mta_tax + improvement_surcharge + tolls_amount + tip_amount + congestion_surcharge, 2
    )

    vendor_id = RNG.choice([1, 2], size=n)
    pu_location_id = RNG.integers(1, 264, size=n)
    do_location_id = RNG.integers(1, 264, size=n)

    df = pd.DataFrame(
        {
            "VendorID": vendor_id,
            "tpep_pickup_datetime": pickup_dt,
            "passenger_count": passenger_count,
            "trip_distance": np.round(trip_distance, 2),
            "PULocationID": pu_location_id,
            "DOLocationID": do_location_id,
            "pickup_borough": borough,  # cột phụ, mô phỏng tra cứu từ taxi-zone lookup
            "payment_type": payment_type,
            "fare_amount": fare_amount,
            "mta_tax": mta_tax,
            "improvement_surcharge": improvement_surcharge,
            "congestion_surcharge": congestion_surcharge,
            "tolls_amount": tolls_amount,
            "tip_amount": tip_amount,
            "total_amount": total_amount,
        }
    )

    # ---- chèn dữ liệu BẨN đúng như dữ liệu TLC thật hay gặp ----
    n_dirty = int(n * 0.03)
    dirty_idx = RNG.choice(n, size=n_dirty, replace=False)
    thirds = np.array_split(dirty_idx, 4)

    df.loc[thirds[0], "fare_amount"] = -RNG.uniform(2, 20, size=len(thirds[0])).round(2)  # fare âm (huỷ/hoàn tiền)
    df.loc[thirds[1], "trip_distance"] = 0.0  # distance = 0
    df.loc[thirds[2], "passenger_count"] = 0  # passenger_count = 0
    df.loc[thirds[3], "trip_distance"] = RNG.uniform(150, 400, size=len(thirds[3])).round(1)  # outlier quãng đường

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Đã sinh {len(df):,} dòng -> {OUT_PATH}")
    print(f"Trong đó có {n_dirty:,} dòng dữ liệu bẩn cố ý chèn vào (để luyện bước làm sạch).")


if __name__ == "__main__":
    main()
