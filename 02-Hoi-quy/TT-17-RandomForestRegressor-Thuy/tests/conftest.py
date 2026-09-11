"""Fixtures dùng chung cho toàn bộ test — sinh dữ liệu giả lập ĐÚNG SCHEMA của
Clean_Dataset.csv (Kaggle Flight Price Prediction) để test pipeline mà không
cần file dữ liệu thật (không được phép commit vào repo).
"""
import numpy as np
import pandas as pd
import pytest

RANDOM_STATE = 42


def make_synthetic_flight_df(n_rows: int = 4000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    airlines = ["SpiceJet", "AirAsia", "Vistara", "GO_FIRST", "Indigo", "Air_India"]
    cities = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"]
    times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    stops = ["zero", "one", "two_or_more"]
    classes = ["Economy", "Business"]

    n = n_rows
    airline = rng.choice(airlines, n)
    source_city = rng.choice(cities, n)
    # đảm bảo destination khác source
    destination_city = np.array([rng.choice([c for c in cities if c != s]) for s in source_city])
    departure_time = rng.choice(times, n)
    arrival_time = rng.choice(times, n)
    stop = rng.choice(stops, n, p=[0.2, 0.6, 0.2])
    klass = rng.choice(classes, n, p=[0.85, 0.15])
    duration = np.clip(rng.normal(10, 5, n), 1, 40)
    days_left = rng.randint(1, 50, n)
    flight_code = [f"{rng.choice(['SG','AI','UK','6E'])}-{rng.randint(1000,9999)}" for _ in range(n)]

    # Giá: phi tuyến theo days_left (tăng vọt khi gần ngày bay) + hạng vé + nhiễu
    base = 4000 + duration * 150
    days_effect = np.where(days_left <= 7, (8 - days_left) * 800, 0)
    class_effect = np.where(klass == "Business", 35000, 0)
    stops_effect = np.select([stop == "zero", stop == "one", stop == "two_or_more"], [1000, 0, -500])
    noise = rng.normal(0, 1500, n)
    price = base + days_effect + class_effect + stops_effect + noise
    price = np.clip(price, 1500, None)

    df = pd.DataFrame({
        "Unnamed: 0": np.arange(n),
        "airline": airline,
        "flight": flight_code,
        "source_city": source_city,
        "departure_time": departure_time,
        "stops": stop,
        "arrival_time": arrival_time,
        "destination_city": destination_city,
        "class": klass,
        "duration": duration,
        "days_left": days_left,
        "price": price,
    })
    return df


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return make_synthetic_flight_df()


@pytest.fixture(scope="session")
def synthetic_csv(tmp_path_factory, synthetic_df) -> str:
    path = tmp_path_factory.mktemp("data") / "Clean_Dataset.csv"
    synthetic_df.to_csv(path, index=False)
    return str(path)
