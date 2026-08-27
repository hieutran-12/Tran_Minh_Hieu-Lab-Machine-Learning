"""
TT-08 XGBoost — Fraud Detection
Phiên bản script độc lập, viết lại theo notebooks/xgboost_fraud.ipynb (bản đã
bỏ nhánh mô phỏng — chỉ chạy với dữ liệu THẬT từ Kaggle Credit Card Fraud).

Chạy: python src/train.py
(chạy từ đâu cũng được — script tự tính đường dẫn tương đối theo vị trí file này)

Yêu cầu bắt buộc: đặt file dữ liệu thật tại data/creditcard.csv (tương đối so
với thư mục gốc TT-08-XGBoost/). Không có file này script sẽ dừng lại với lỗi
rõ ràng — không còn âm thầm trả về None như bản notebook cũ (xem mục "Lỗi đã
sửa so với notebook" bên dưới).
"""

import json
import os
import time

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)

import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

# --------------------------------------------------------------------------
# Đường dẫn — luôn tương đối so với vị trí file này, chạy được trên mọi máy.
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../TT-08-XGBoost
DATA_PATH = os.path.join(BASE_DIR, "data", r"D:\Hoc-May\Lam_Bai_Lab\01-Phan-loai\TT-08-XGBoost\creditcard.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Chi phí kinh doanh dùng để chọn ngưỡng tối ưu ở mục "cost-based threshold"
COST_FALSE_POSITIVE = 200_000  # đồng, chi phí chăm sóc khách khi chặn nhầm
# Amount gốc của bộ Kaggle Credit Card Fraud là EUR — quy đổi sang VND để so
# sánh cùng đơn vị với COST_FALSE_POSITIVE. Đây là giả định, thay bằng tỷ giá
# thực tế tại thời điểm phân tích nếu cần.
EUR_TO_VND = 27_000


def load_data() -> pd.DataFrame:

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu tại {DATA_PATH}. "
            f"Hãy đặt file creditcard.csv (tải từ Kaggle) vào thư mục "
            f"'{os.path.join(BASE_DIR, 'data')}' rồi chạy lại."
        )
    print(f"Đang nạp dữ liệu THẬT từ {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def split_and_scale(df: pd.DataFrame):
    """Feature engineering + chia theo thời gian (70/15/15, không shuffle) +
    fit StandardScaler CHỈ trên train — tránh rò rỉ dữ liệu vào val/test."""
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Amount_log"] = np.log1p(df["Amount"])

    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)
    i_train_end = int(n * 0.70)
    i_val_end = int(n * 0.85)

    train_df = df_sorted.iloc[:i_train_end].copy()
    val_df = df_sorted.iloc[i_train_end:i_val_end].copy()
    test_df = df_sorted.iloc[i_val_end:].copy()

    scaler = StandardScaler()
    train_df["Amount_scaled"] = scaler.fit_transform(train_df[["Amount_log"]])
    val_df["Amount_scaled"] = scaler.transform(val_df[["Amount_log"]])
    test_df["Amount_scaled"] = scaler.transform(test_df[["Amount_log"]])

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Hour", "Amount_scaled"]
    return train_df, val_df, test_df, feature_cols, scaler


def train_xgboost(X_train, y_train, X_val, y_val):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        reg_lambda=1.0, reg_alpha=0.1,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist", n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model, scale_pos_weight


def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight):
    """early-stop theo binary_logloss (đơn điệu, ổn định) thay vì
    average_precision — tránh bug dừng sớm khi val chỉ có vài chục ca dương
    (xem README mục "Các lỗi đã phát hiện và sửa")."""
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, reg_lambda=1.0, reg_alpha=0.1,
        random_state=42, verbose=-1,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    return model


def cost_optimal_threshold(y_test, proba, test_amounts_eur):
    test_amounts_vnd = test_amounts_eur * EUR_TO_VND
    y_test_arr = np.asarray(y_test)
    thr_grid = np.linspace(0.01, 0.99, 99)
    total_costs = []
    for thr in thr_grid:
        pred = (proba >= thr).astype(int)
        fp_mask = (pred == 1) & (y_test_arr == 0)
        fn_mask = (pred == 0) & (y_test_arr == 1)
        cost = fp_mask.sum() * COST_FALSE_POSITIVE + test_amounts_vnd[fn_mask].sum()
        total_costs.append(cost)
    total_costs = np.array(total_costs)
    best_idx = np.argmin(total_costs)
    return float(thr_grid[best_idx]), float(total_costs[best_idx])


def precision_90_threshold(y_test, proba):
    prec, rec, thresholds_pr = precision_recall_curve(y_test, proba)
    prec_arr, rec_arr = prec[:-1], rec[:-1]
    mask = prec_arr >= 0.90
    if mask.any():
        idx = np.argmax(rec_arr[mask])
        return float(thresholds_pr[mask][idx])
    return float(thresholds_pr[np.argmax(prec_arr)])


def preprocess_new_transaction(raw_row: dict, metadata: dict) -> list:
    """Tái tạo đúng feature engineering đã dùng lúc train, cho 1 giao dịch MỚI
    (raw_row phải có 'Time', 'Amount', 'V1'..'V28'). Dùng metadata.json đã lưu
    (không tự fit lại scaler) để đảm bảo kết quả nhất quán khi đem model +
    metadata sang máy khác chạy. Trả về list giá trị theo đúng thứ tự
    metadata['feature_cols'].
    """
    hour = (raw_row["Time"] // 3600) % 24
    amount_log = np.log1p(raw_row["Amount"])
    amount_scaled = (amount_log - metadata["amount_scaler_mean"]) / metadata["amount_scaler_scale"]

    values = dict(raw_row)
    values["Hour"] = hour
    values["Amount_scaled"] = amount_scaled
    return [values[c] for c in metadata["feature_cols"]]


def main():
    df = load_data()
    fraud_rate = (df["Class"] == 1).mean()
    print(f"Tổng giao dịch: {len(df):,} | Tỉ lệ gian lận: {fraud_rate:.4%}")
    assert fraud_rate < 0.01, "Tỉ lệ lệch không đúng như kỳ vọng của bài toán"

    train_df, val_df, test_df, feature_cols, scaler = split_and_scale(df)
    X_train, y_train = train_df[feature_cols], train_df["Class"]
    X_val, y_val = val_df[feature_cols], val_df["Class"]
    X_test, y_test = test_df[feature_cols], test_df["Class"]

    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"{name:5s}: {len(y):>7,} giao dịch | {y.sum():>4} gian lận | {y.mean():.4%}")

    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_proba = dummy.predict_proba(X_test)[:, 1]

    logreg = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    logreg.fit(X_train, y_train)
    logreg_proba = logreg.predict_proba(X_test)[:, 1]

    xgb_model, scale_pos_weight = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pr_auc = average_precision_score(y_test, xgb_proba)
    xgb_roc_auc = roc_auc_score(y_test, xgb_proba)
    print(f"XGBoost -> PR-AUC: {xgb_pr_auc:.4f}  ROC-AUC: {xgb_roc_auc:.4f}  "
          f"(best_iteration={xgb_model.best_iteration})")

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]

    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    print(f"LightGBM -> PR-AUC: {average_precision_score(y_test, lgb_proba):.4f}  "
          f"ROC-AUC: {roc_auc_score(y_test, lgb_proba):.4f}  "
          f"(best_iteration={lgb_model.best_iteration_})")

    comparison = pd.DataFrame({
        "Model": ["DummyClassifier", "Logistic Regression", "Random Forest", "LightGBM", "XGBoost"],
        "PR-AUC": [
            average_precision_score(y_test, dummy_proba),
            average_precision_score(y_test, logreg_proba),
            average_precision_score(y_test, rf_proba),
            average_precision_score(y_test, lgb_proba),
            xgb_pr_auc,
        ],
        "ROC-AUC": [
            roc_auc_score(y_test, dummy_proba),
            roc_auc_score(y_test, logreg_proba),
            roc_auc_score(y_test, rf_proba),
            roc_auc_score(y_test, lgb_proba),
            xgb_roc_auc,
        ],
    }).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    print("\n=== So sánh model (sắp theo PR-AUC) ===")
    print(comparison.to_string(index=False))

    thr_p90 = precision_90_threshold(y_test, xgb_proba)
    thr_cost, cost_val = cost_optimal_threshold(y_test, xgb_proba, test_df["Amount"].to_numpy())
    print(f"\nNgưỡng đạt Precision>=0.90 : {thr_p90:.4f}")
    print(f"Ngưỡng tối ưu chi phí (VND): {thr_cost:.3f}  (tổng chi phí ~{cost_val:,.0f} đ, "
          f"tỷ giá giả định 1 EUR = {EUR_TO_VND:,} VND)")

    single_row = X_test.iloc[[0]]
    for _ in range(5):
        xgb_model.predict_proba(single_row)
    timings = []
    for _ in range(200):
        t0 = time.perf_counter()
        xgb_model.predict_proba(single_row)
        timings.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(timings, 95))
    print(f"\nThời gian dự đoán 1 giao dịch: p95={p95:.3f} ms "
          f"({'đạt' if p95 < 100 else 'KHÔNG đạt'} yêu cầu < 100ms)")

    xgb_model.save_model(os.path.join(MODELS_DIR, "xgb_fraud.json"))
    metadata = {
        "feature_cols": feature_cols,
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(xgb_model.best_iteration),
        "threshold_precision_90": thr_p90,
        "threshold_cost_optimal": thr_cost,
        "cost_false_positive_vnd": COST_FALSE_POSITIVE,
        "eur_to_vnd_assumed_rate": EUR_TO_VND,
        "test_pr_auc": float(xgb_pr_auc),
        "test_roc_auc": float(xgb_roc_auc),
        # Tham số StandardScaler (fit trên Amount_log của TRAIN) — bắt buộc để
        # preprocess_new_transaction() tái tạo đúng Amount_scaled trên máy
        # khác, không phải tự fit lại scaler (sẽ cho mean/std khác, sai lệch).
        "amount_scaler_mean": float(scaler.mean_[0]),
        "amount_scaler_scale": float(scaler.scale_[0]),
    }
    with open(os.path.join(MODELS_DIR, "xgb_fraud_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu model + metadata vào {MODELS_DIR}/")


if __name__ == "__main__":
    main()
