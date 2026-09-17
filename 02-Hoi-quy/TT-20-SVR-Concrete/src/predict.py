"""
Dự đoán cường độ chịu nén từ tỉ lệ phối trộn, dùng model đã lưu.

    python src/predict.py --csv mix_moi.csv          # file có 8 cột gốc
    python src/predict.py --cement 380 --water 175 --age 28
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from data import BASE_FEATURES, add_domain_features

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "svr_pipeline.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise SystemExit("Chưa có model. Chạy `python src/train.py` trước.")
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["features"]


def predict(df: pd.DataFrame) -> pd.Series:
    model, features = load_model()
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Thiếu cột: {missing}")
    X = add_domain_features(df)[features].to_numpy(dtype=float)
    return pd.Series(model.predict(X), index=df.index, name="Strength_pred_MPa")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV chứa 8 cột: " + ", ".join(BASE_FEATURES))
    for c in BASE_FEATURES:
        ap.add_argument(f"--{c.lower()}", type=float)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        defaults = dict(Cement=350, BlastFurnaceSlag=0, FlyAsh=0, Water=180,
                        Superplasticizer=6, CoarseAggregate=1000, FineAggregate=780, Age=28)
        row = {c: (getattr(args, c.lower()) if getattr(args, c.lower()) is not None else defaults[c])
               for c in BASE_FEATURES}
        df = pd.DataFrame([row])

    out = df.copy()
    out["Strength_pred_MPa"] = predict(df).round(2)
    out["water_cement_ratio"] = (df["Water"] / df["Cement"]).round(3)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
