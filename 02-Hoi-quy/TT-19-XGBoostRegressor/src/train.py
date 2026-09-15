"""
train.py — Pipeline end-to-end cho TT-19 (XGBoost Regressor):
dự báo nhu cầu thuê xe đạp công cộng theo giờ.

Cách chạy:
    python src/train.py --data data/hour.csv

Yêu cầu: tải hour.csv từ UCI Bike Sharing Dataset và đặt vào data/hour.csv
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Toàn bộ output (model, biểu đồ) được ghi vào models/ và reports/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

from features import (
    load_data,
    prove_leakage,
    build_feature_matrix,
    time_split,
    naive_baseline_predict,
)

RANDOM_STATE = 42


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, out_dir: Path):
    section("4/13. EDA: cnt trung bình theo giờ / mùa / thời tiết")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    hourly = df.groupby("hr")["cnt"].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color="#2b6cb0")
    axes[0].set_title("Cnt trung bình theo giờ trong ngày")
    axes[0].set_xlabel("Giờ")
    axes[0].set_ylabel("Cnt trung bình")
    axes[0].axvline(8, color="gray", linestyle="--", linewidth=0.8)
    axes[0].axvline(17.5, color="gray", linestyle="--", linewidth=0.8)

    season_names = {1: "Xuân", 2: "Hè", 3: "Thu", 4: "Đông"}
    seasonal = df.groupby("season")["cnt"].mean()
    axes[1].bar([season_names[s] for s in seasonal.index], seasonal.values, color="#38a169")
    axes[1].set_title("Cnt trung bình theo mùa")

    weather_names = {1: "Quang đãng", 2: "Sương mù/Mây", 3: "Mưa/Tuyết nhẹ", 4: "Mưa/Tuyết nặng"}
    weather = df.groupby("weathersit")["cnt"].mean()
    axes[2].bar([weather_names.get(w, str(w)) for w in weather.index], weather.values, color="#dd6b20")
    axes[2].set_title("Cnt trung bình theo thời tiết")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out_path = out_dir / "cnt_theo_gio.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"-> Đã lưu {out_path}")
    print(hourly.sort_values(ascending=False).head(3))


# ---------------------------------------------------------------------------
# 2. Train XGBoost chính thức
# ---------------------------------------------------------------------------
def train_xgb(X_train, y_train_log, X_val, y_val_log, params: dict | None = None):
    default_params = dict(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        min_child_weight=3,
        objective="reg:squarederror",
        early_stopping_rounds=100,
        eval_metric="rmse",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    if params:
        default_params.update(params)
    model = xgb.XGBRegressor(**default_params)
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        verbose=False,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/hour.csv")
    parser.add_argument("--search-iter", type=int, default=25,
                         help="Số vòng lặp cho RandomizedSearchCV (giảm để chạy nhanh khi test)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    reports_dir = root / "reports"
    models_dir = root / "models"
    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    # ---- 1. Nạp dữ liệu -----------------------------------------------
    section("1/13. Nạp dữ liệu")
    df = load_data(args.data)
    print(f"Shape: {df.shape}")
    required_cols = {"casual", "registered", "cnt", "dteday", "hr", "yr", "mnth"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc trong hour.csv: {missing}")

    # ---- 2. Chứng minh rò rỉ -------------------------------------------
    section("2/13. ⭐ CHỨNG MINH RÒ RỈ (giữ casual + registered)")
    r2_leak = prove_leakage(df)
    print(f"R² khi giữ casual + registered làm đặc trưng: {r2_leak:.6f}")
    print("=> cnt = casual + registered (đúng bằng tổng), nên PHẢI bỏ cả hai cột này")
    print("   trước khi train, nếu không model sẽ 'gian lận' thay vì học thật.")

    # ---- 3. Chia theo thời gian -----------------------------------------
    section("3/13. Chia dữ liệu THEO THỜI GIAN (không shuffle)")
    train_df, val_df, test_df = time_split(df, val_months=9, test_months=3)
    print(f"Train (năm 1): {len(train_df)} dòng | "
          f"Validation (năm 2, T1-T9): {len(val_df)} dòng | "
          f"Test (năm 2, T10-T12): {len(test_df)} dòng")

    # ---- 4. EDA -----------------------------------------------------------
    run_eda(df, reports_dir)

    # ---- 5+6. Mã hoá chu kỳ + baseline naive ------------------------------
    section("5-6/13. Mã hoá sin/cos + Baseline naive (cùng giờ tuần trước)")
    X_train, y_train = build_feature_matrix(train_df, drop_leak=True)
    X_val, y_val = build_feature_matrix(val_df, drop_leak=True)
    X_test, y_test = build_feature_matrix(test_df, drop_leak=True)
    print(f"Số đặc trưng: {X_train.shape[1]} -> {list(X_train.columns)}")

    baseline_val_pred = naive_baseline_predict(train_df, val_df)
    baseline_test_pred = naive_baseline_predict(pd.concat([train_df, val_df]), test_df)
    baseline_val_rmse = rmse(y_val, baseline_val_pred)
    baseline_test_rmse = rmse(y_test, baseline_test_pred)
    print(f"Baseline naive — RMSE validation: {baseline_val_rmse:.2f} | RMSE test: {baseline_test_rmse:.2f}")

    # ---- 7. Train XGBoost + early stopping ---------------------------------
    section("7/13. Train XGBoost (log1p target) + early stopping")
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    t0 = time.time()
    model_log = train_xgb(X_train, y_train_log, X_val, y_val_log)
    n_trees_log = model_log.best_iteration + 1 if model_log.best_iteration is not None else model_log.n_estimators
    pred_val_log = np.clip(np.expm1(model_log.predict(X_val)), 0, None)
    pred_test_log = np.clip(np.expm1(model_log.predict(X_test)), 0, None)
    print(f"Số cây thực tế dùng (early stopping): {n_trees_log} | thời gian train: {time.time()-t0:.1f}s")

    # ---- 8. So sánh có/không log1p ------------------------------------------
    section("8/13. So sánh có/không dùng log1p cho nhãn")
    model_raw = train_xgb(X_train, y_train, X_val, y_val)
    pred_val_raw = np.clip(model_raw.predict(X_val), 0, None)
    pred_test_raw = np.clip(model_raw.predict(X_test), 0, None)

    rmse_val_log = rmse(y_val, pred_val_log)
    rmse_val_raw = rmse(y_val, pred_val_raw)
    print(f"RMSE validation — có log1p: {rmse_val_log:.2f} | không log1p: {rmse_val_raw:.2f}")
    use_log = rmse_val_log <= rmse_val_raw
    print(f"=> Chọn phương án: {'log1p' if use_log else 'raw (không log1p)'}")

    final_model = model_log if use_log else model_raw
    final_pred_val = pred_val_log if use_log else pred_val_raw
    final_pred_test = pred_test_log if use_log else pred_test_raw

    # ---- 9. RandomizedSearchCV ------------------------------------------------
    section("9/13. Dò siêu tham số bằng RandomizedSearchCV")
    X_tr_search = pd.concat([X_train, X_val], ignore_index=True)
    y_tr_search = pd.concat([y_train, y_val], ignore_index=True)
    y_tr_search_used = np.log1p(y_tr_search) if use_log else y_tr_search
    # PredefinedSplit: -1 = luôn ở train, 0 = luôn ở validation (giữ đúng thứ tự thời gian,
    # KHÔNG cho CV ngẫu nhiên xáo trộn quá khứ/tương lai)
    test_fold = np.array([-1] * len(X_train) + [0] * len(X_val))
    ps = PredefinedSplit(test_fold)

    param_dist = {
        "n_estimators": [400, 600, 800, 1000],
        "learning_rate": [0.02, 0.03, 0.05, 0.08],
        "max_depth": [4, 5, 6, 7, 8],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "min_child_weight": [1, 3, 5, 7],
    }
    base_est = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    search = RandomizedSearchCV(
        base_est, param_distributions=param_dist,
        n_iter=args.search_iter, cv=ps, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
    )
    t0 = time.time()
    search.fit(X_tr_search, y_tr_search_used)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE (log-space nếu use_log): {-search.best_score_:.4f} | thời gian: {time.time()-t0:.1f}s")

    tuned_model = train_xgb(X_train, np.log1p(y_train) if use_log else y_train,
                             X_val, np.log1p(y_val) if use_log else y_val,
                             params=search.best_params_)
    tuned_pred_test = tuned_model.predict(X_test)
    if use_log:
        tuned_pred_test = np.expm1(tuned_pred_test)
    tuned_pred_test = np.clip(tuned_pred_test, 0, None)

    rmse_before = rmse(y_test, final_pred_test)
    rmse_after = rmse(y_test, tuned_pred_test)
    print(f"RMSE test — trước tuning: {rmse_before:.2f} | sau tuning: {rmse_after:.2f}")
    if rmse_after < rmse_before:
        final_model = tuned_model
        final_pred_test = tuned_pred_test
        print("=> Dùng model đã tuning (tốt hơn)")
    else:
        print("=> Giữ model mặc định (tuning không cải thiện trên tập test)")

    # ---- 10. Feature importance + SHAP -----------------------------------------
    section("10/13. Feature importance (gain) + SHAP summary")
    importance = final_model.get_booster().get_score(importance_type="gain")
    importance_series = pd.Series(importance).sort_values(ascending=False)
    print(importance_series.head(10))

    fig, ax = plt.subplots(figsize=(7, 5))
    importance_series.head(15).sort_values().plot(kind="barh", ax=ax, color="#805ad5")
    ax.set_title("Feature importance (gain) — top 15")
    plt.tight_layout()
    plt.savefig(reports_dir / "feature_importance.png", dpi=130)
    plt.close(fig)

    try:
        import shap
        explainer = shap.TreeExplainer(final_model)
        sample = X_test.sample(min(1000, len(X_test)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(sample)
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(reports_dir / "shap_summary.png", dpi=130)
        plt.close(fig)
        print(f"-> Đã lưu {reports_dir / 'shap_summary.png'}")
    except Exception as e:
        print(f"[Cảnh báo] Không tạo được SHAP summary plot: {e}")

    # ---- 11. Dự báo vs thực tế 2 tuần cuối --------------------------------------
    section("11/13. ⭐ Dự báo vs thực tế trên 2 tuần cuối (test set)")
    test_plot_df = test_df.copy().reset_index(drop=True)
    test_plot_df["timestamp"] = pd.to_datetime(test_plot_df["dteday"]) + pd.to_timedelta(test_plot_df["hr"], unit="h")
    test_plot_df["y_true"] = y_test.values
    test_plot_df["y_pred"] = final_pred_test
    last_2w = test_plot_df.sort_values("timestamp").tail(24 * 14)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(last_2w["timestamp"], last_2w["y_true"], label="Thực tế", color="#2b6cb0", linewidth=1.2)
    ax.plot(last_2w["timestamp"], last_2w["y_pred"], label="Dự báo (XGBoost)", color="#e53e3e",
            linewidth=1.2, alpha=0.85)
    ax.set_title("Dự báo vs Thực tế — 2 tuần cuối tập test")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Số lượt thuê / giờ (cnt)")
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(reports_dir / "du_bao_vs_thuc_te.png", dpi=130)
    plt.close(fig)
    print(f"-> Đã lưu {reports_dir / 'du_bao_vs_thuc_te.png'}")

    # ---- 12. Phân tích lỗi ---------------------------------------------------------
    section("12/13. Phân tích lỗi theo giờ / thời tiết")
    test_plot_df["abs_err"] = (test_plot_df["y_true"] - test_plot_df["y_pred"]).abs()
    err_by_hour = test_plot_df.groupby("hr")["abs_err"].mean().sort_values(ascending=False)
    err_by_weather = test_plot_df.groupby("weathersit")["abs_err"].mean().sort_values(ascending=False)
    print("MAE theo giờ (5 giờ sai nhiều nhất):")
    print(err_by_hour.head(5))
    print("\nMAE theo điều kiện thời tiết:")
    print(err_by_weather)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    err_by_hour.sort_index().plot(kind="bar", ax=axes[0], color="#dd6b20")
    axes[0].set_title("MAE theo giờ trong ngày")
    axes[0].set_xlabel("Giờ")
    weather_names = {1: "Quang đãng", 2: "Sương mù/Mây", 3: "Mưa/Tuyết nhẹ", 4: "Mưa/Tuyết nặng"}
    err_by_weather.rename(index=weather_names).plot(kind="bar", ax=axes[1], color="#c53030")
    axes[1].set_title("MAE theo thời tiết")
    plt.tight_layout()
    plt.savefig(reports_dir / "phan_tich_loi.png", dpi=130)
    plt.close(fig)
    print(f"-> Đã lưu {reports_dir / 'phan_tich_loi.png'}")

    # ---- 13. So sánh với Random Forest / Gradient Boosting ------------------------
    section("13/13. So sánh với Random Forest (TT-17) và Gradient Boosting (TT-18)")
    y_train_fit = np.log1p(y_train) if use_log else y_train
    rf = RandomForestRegressor(n_estimators=400, max_depth=14, n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train_fit)
    rf_pred = rf.predict(X_test)
    if use_log:
        rf_pred = np.expm1(rf_pred)
    rf_pred = np.clip(rf_pred, 0, None)

    gb = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE)
    gb.fit(X_train, y_train_fit)
    gb_pred = gb.predict(X_test)
    if use_log:
        gb_pred = np.expm1(gb_pred)
    gb_pred = np.clip(gb_pred, 0, None)

    comparison = pd.DataFrame({
        "model": ["Baseline naive", "Random Forest (TT-17)", "Gradient Boosting (TT-18)", "XGBoost (TT-19)"],
        "rmse_test": [
            baseline_test_rmse,
            rmse(y_test, rf_pred),
            rmse(y_test, gb_pred),
            rmse(y_test, final_pred_test),
        ],
        "r2_test": [
            r2_score(y_test, baseline_test_pred),
            r2_score(y_test, rf_pred),
            r2_score(y_test, gb_pred),
            r2_score(y_test, final_pred_test),
        ],
        "mae_test": [
            mean_absolute_error(y_test, baseline_test_pred),
            mean_absolute_error(y_test, rf_pred),
            mean_absolute_error(y_test, gb_pred),
            mean_absolute_error(y_test, final_pred_test),
        ],
    })
    print(comparison.to_string(index=False))
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)

    beats_baseline = rmse(y_test, final_pred_test) < baseline_test_rmse
    print(f"\nXGBoost thắng baseline naive: {beats_baseline}")

    # ---- Lưu model + tổng kết ----------------------------------------------------------
    final_model.save_model(str(models_dir / "xgb_bike.json"))
    summary = {
        "leakage_r2_with_casual_registered": r2_leak,
        "n_features": X_train.shape[1],
        "used_log1p_target": bool(use_log),
        "n_trees_used_early_stopping": int(n_trees_log),
        "baseline_naive_rmse_test": baseline_test_rmse,
        "final_model_rmse_test": rmse(y_test, final_pred_test),
        "final_model_r2_test": r2_score(y_test, final_pred_test),
        "final_model_mae_test": mean_absolute_error(y_test, final_pred_test),
        "beats_naive_baseline": bool(beats_baseline),
        "best_search_params": search.best_params_,
    }
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    section("KẾT QUẢ CUỐI CÙNG")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
