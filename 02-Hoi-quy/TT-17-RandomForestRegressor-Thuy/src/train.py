
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # chạy được cả khi không có màn hình (server/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("tt17")

RANDOM_STATE = 42
NUMERIC_COLS = ["duration", "days_left"]
CATEGORICAL_COLS = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]
TARGET_COL = "price"
STRATIFY_COL = "class"  # biến chi phối mạnh nhất -> giữ tỉ lệ khi split


# --------------------------------------------------------------------------
# 1. NẠP DỮ LIỆU + KIỂM TRA CHẤT LƯỢNG
# --------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            "-> Tải Clean_Dataset.csv (Kaggle: Flight Price Prediction) và đặt vào thư mục data/"
        )

    df = pd.read_csv(path)

    # Bỏ cột index thừa mà Kaggle export ra kèm ("Unnamed: 0" hoặc tương đương)
    unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # BẮT BUỘC bỏ cột `flight`: mã chuyến bay có hàng nghìn giá trị duy nhất
    # (VD: "SG-8709"). Nếu one-hot encode cột này, mô hình sẽ "học thuộc lòng"
    # từng mã chuyến thay vì học quy luật giá => overfit nặng, không tổng quát.
    if "flight" in df.columns:
        n_unique_flight = df["flight"].nunique()
        df = df.drop(columns=["flight"])
        log.info(f"Đã bỏ cột 'flight' ({n_unique_flight:,} giá trị duy nhất — tránh học thuộc lòng).")

    missing_expected = set(NUMERIC_COLS + CATEGORICAL_COLS + [TARGET_COL]) - set(df.columns)
    if missing_expected:
        raise ValueError(
            f"Thiếu cột bắt buộc trong dữ liệu: {missing_expected}. "
            "Kiểm tra lại file Clean_Dataset.csv có đúng cấu trúc Kaggle gốc không."
        )

    log.info(f"Dữ liệu nạp thành công: {df.shape[0]:,} dòng x {df.shape[1]} cột")
    _validate_data_quality(df)
    return df


def _validate_data_quality(df: pd.DataFrame) -> dict:
    """Kiểm tra dữ liệu trước khi train — thay vì tin mù vào CSV đầu vào.

    Không raise lỗi cho cảnh báo nhẹ (in ra để người dùng tự quyết định), nhưng
    raise lỗi cứng nếu có giá trị hoàn toàn vô lý (giá vé âm/bằng 0, ngày âm).
    """
    report = {}

    n_null = int(df.isna().sum().sum())
    report["so_gia_tri_null"] = n_null
    if n_null > 0:
        log.warning(f"Phát hiện {n_null} giá trị null:\n{df.isna().sum()[df.isna().sum() > 0]}")
    else:
        log.info("Kiểm tra dữ liệu: không có giá trị null.")

    n_dup = int(df.duplicated().sum())
    report["so_dong_trung_lap"] = n_dup
    if n_dup > 0:
        log.warning(f"Phát hiện {n_dup:,} dòng trùng lặp hoàn toàn ({n_dup / len(df):.1%}).")

    # Giá vé và duration/days_left phải dương -> nếu không, dữ liệu lỗi nghiêm trọng
    for col, must_be in [(TARGET_COL, ">0"), ("duration", ">0"), ("days_left", ">=0")]:
        if col not in df.columns:
            continue
        bad = (df[col] <= 0) if must_be == ">0" else (df[col] < 0)
        n_bad = int(bad.sum())
        report[f"so_dong_{col}_bat_thuong"] = n_bad
        if n_bad > 0:
            raise ValueError(
                f"Phát hiện {n_bad} dòng có `{col}` không hợp lệ (yêu cầu {must_be}). "
                "Dữ liệu có thể bị lỗi export — cần làm sạch trước khi train."
            )

    log.info(
        f"Kiểm tra dữ liệu: giá vé trong khoảng [{df[TARGET_COL].min():,.0f}, "
        f"{df[TARGET_COL].max():,.0f}], days_left trong [{df['days_left'].min()}, "
        f"{df['days_left'].max()}]."
    )
    return report


# --------------------------------------------------------------------------
# 2. EDA
# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame, reports_dir: Path) -> None:
    sns.set_style("whitegrid")

    # (a) Giá trung bình theo days_left -> quan hệ phi tuyến rõ rệt
    fig, ax = plt.subplots(figsize=(9, 5))
    by_days = df.groupby("days_left")[TARGET_COL].mean().sort_index()
    ax.plot(by_days.index, by_days.values, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Số ngày còn lại tới chuyến bay (days_left)")
    ax.set_ylabel("Giá vé trung bình (Rupee)")
    ax.set_title("Giá vé trung bình theo số ngày còn lại tới chuyến bay")
    ax.axvspan(0, 7, color="red", alpha=0.08, label="7 ngày cuối")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "gia_theo_days_left.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/gia_theo_days_left.png")

    # (b) Boxplot giá theo class
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df, x="class", y=TARGET_COL, ax=ax)
    ax.set_title("Phân phối giá vé theo hạng vé (class)")
    fig.tight_layout()
    fig.savefig(reports_dir / "boxplot_gia_theo_class.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/boxplot_gia_theo_class.png")

    # (c) Boxplot giá theo airline
    fig, ax = plt.subplots(figsize=(11, 5))
    order = df.groupby("airline")[TARGET_COL].median().sort_values().index
    sns.boxplot(data=df, x="airline", y=TARGET_COL, order=order, ax=ax)
    ax.set_title("Phân phối giá vé theo hãng bay (airline)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(reports_dir / "boxplot_gia_theo_airline.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/boxplot_gia_theo_airline.png")


# --------------------------------------------------------------------------
# 3. PIPELINE TIỀN XỬ LÝ
# --------------------------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    # Cây/ensemble KHÔNG cần scale numeric -> chỉ cần OneHot cho categorical
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# 4. BASELINE
# --------------------------------------------------------------------------
def run_baselines(X_train, X_test, y_train, y_test, preprocessor) -> dict:
    results = {}

    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Decision Tree (TT-16, single)": DecisionTreeRegressor(
            random_state=RANDOM_STATE, min_samples_leaf=2
        ),
    }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = _regression_metrics(y_test, pred)
        log.info(
            f"[Baseline] {name:32s} RMSE={results[name]['rmse']:.0f}  "
            f"MAE={results[name]['mae']:.0f}  R2={results[name]['r2']:.4f}"
        )

    return results


# --------------------------------------------------------------------------
# 5. RANDOM FOREST (mặc định)
# --------------------------------------------------------------------------
def train_random_forest(X_train, y_train, preprocessor, **rf_overrides) -> Pipeline:
    rf_params = dict(
        n_estimators=300,
        max_features=1.0,  # hồi quy: mặc định dùng hết đặc trưng (khác 'sqrt' của phân loại)
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=True,
        bootstrap=True,
    )
    rf_params.update(rf_overrides)
    rf = RandomForestRegressor(**rf_params)
    pipe = Pipeline([("prep", preprocessor), ("model", rf)])
    pipe.fit(X_train, y_train)
    log.info(f"Random Forest — oob_score_ (R^2 trên out-of-bag): {rf.oob_score_:.4f}")
    return pipe


# --------------------------------------------------------------------------
# 6. TINH CHỈNH SIÊU THAM SỐ (RandomizedSearchCV)
# --------------------------------------------------------------------------
def tune_random_forest(
    X_train,
    y_train,
    preprocessor,
    n_iter: int = 20,
    cv_folds: int = 3,
    search_sample_frac: float = 0.35,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Tìm siêu tham số bằng RandomizedSearchCV thay vì dùng mặc định "đoán bừa".

    Để tiết kiệm thời gian trên tập dữ liệu lớn, quá trình TÌM KIẾM chạy trên một
    tập con ngẫu nhiên (`search_sample_frac`) của tập train — đây là kỹ thuật
    thực dụng phổ biến. Sau khi tìm được cấu hình tốt nhất, model CUỐI CÙNG vẫn
    được refit trên TOÀN BỘ tập train (xem `main()`), search chỉ dùng để chọn
    tham số chứ không phải model cuối.
    """
    if 0 < search_sample_frac < 1.0:
        X_search = X_train.sample(frac=search_sample_frac, random_state=random_state)
        y_search = y_train.loc[X_search.index]
        log.info(
            f"Tìm siêu tham số trên {len(X_search):,}/{len(X_train):,} dòng "
            f"({search_sample_frac:.0%}) để tiết kiệm thời gian."
        )
    else:
        X_search, y_search = X_train, y_train

    param_distributions = {
        "model__n_estimators": [100, 200, 300, 400, 500],
        "model__max_depth": [None, 10, 15, 20, 30, 40],
        "model__min_samples_leaf": [1, 2, 4, 8, 16],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__max_features": [1.0, 0.7, 0.5, "sqrt"],
    }

    base_rf = RandomForestRegressor(
        n_jobs=-1, random_state=random_state, bootstrap=True, oob_score=False
    )
    pipe = Pipeline([("prep", preprocessor), ("model", base_rf)])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    t0 = time.time()
    search.fit(X_search, y_search)
    elapsed = time.time() - t0

    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    log.info(f"RandomizedSearchCV hoàn tất sau {elapsed:.1f}s ({n_iter} cấu hình x {cv_folds}-fold)")
    log.info(f"Best CV RMSE = {-search.best_score_:.0f} | Best params = {best_params}")

    return {
        "best_params": best_params,
        "best_cv_rmse": float(-search.best_score_),
        "n_iter": n_iter,
        "cv_folds": cv_folds,
        "search_sample_frac": search_sample_frac,
        "search_time_seconds": elapsed,
    }


# --------------------------------------------------------------------------
# 7. CROSS-VALIDATION CHO MODEL CUỐI
# --------------------------------------------------------------------------
def run_cross_validation(pipe: Pipeline, X, y, cv_folds: int = 5) -> dict:
    """Đánh giá bằng K-fold CV thay vì chỉ tin vào 1 lần train/test split.

    `cross_val_score` tự clone estimator ở mỗi fold nên không ảnh hưởng tới
    `pipe` đã fit trước đó (an toàn khi gọi sau khi đã có model cuối).
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    neg_rmse_scores = cross_val_score(
        pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2", n_jobs=-1)

    rmse_scores = -neg_rmse_scores
    log.info(
        f"{cv_folds}-fold CV — RMSE = {rmse_scores.mean():,.0f} ± {rmse_scores.std():,.0f}  "
        f"| R2 = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}"
    )
    return {
        "cv_folds": cv_folds,
        "rmse_mean": float(rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "r2_mean": float(r2_scores.mean()),
        "r2_std": float(r2_scores.std()),
        "rmse_per_fold": rmse_scores.tolist(),
        "r2_per_fold": r2_scores.tolist(),
    }


# --------------------------------------------------------------------------
# 8. RMSE THEO SỐ CÂY
# --------------------------------------------------------------------------
def rmse_vs_n_estimators(X_train, X_test, y_train, y_test, preprocessor, reports_dir: Path, n_values=None):
    n_values = n_values or [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]
    train_rmses, test_rmses = [], []

    # Transform 1 lần để không fit lại preprocessor mỗi vòng lặp (tiết kiệm thời gian)
    prep = preprocessor
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)

    for n in n_values:
        rf = RandomForestRegressor(
            n_estimators=n, max_features=1.0, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE,
        )
        rf.fit(X_train_t, y_train)
        train_rmses.append(rmse(y_train, rf.predict(X_train_t)))
        test_rmses.append(rmse(y_test, rf.predict(X_test_t)))
        log.info(f"[RMSE-vs-n] n_estimators={n:4d}  train={train_rmses[-1]:.0f}  test={test_rmses[-1]:.0f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, train_rmses, marker="o", label="Train RMSE")
    ax.plot(n_values, test_rmses, marker="o", label="Test RMSE")
    ax.set_xlabel("Số cây (n_estimators)")
    ax.set_ylabel("RMSE (Rupee)")
    ax.set_title("RMSE theo số lượng cây trong rừng")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "rmse_theo_so_cay.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/rmse_theo_so_cay.png")

    # Tìm điểm bão hoà tự động: n nhỏ nhất mà test RMSE nằm trong 0.5% của RMSE tốt nhất
    best_rmse = min(test_rmses)
    threshold = best_rmse * 1.005
    saturation_n = next(n for n, r in zip(n_values, test_rmses) if r <= threshold)
    log.info(f"[RMSE-vs-n] Điểm bão hoà ước lượng: n_estimators={saturation_n} (trong 0.5% của RMSE tốt nhất={best_rmse:.0f})")

    return {
        "n_estimators": n_values,
        "train_rmse": train_rmses,
        "test_rmse": test_rmses,
        "diem_bao_hoa_uoc_luong": saturation_n,
    }


# --------------------------------------------------------------------------
# 9. PERMUTATION IMPORTANCE (+ SO SÁNH VỚI GINI IMPORTANCE)
# --------------------------------------------------------------------------
def run_permutation_importance(pipe: Pipeline, X_test, y_test, reports_dir: Path):
    # QUAN TRỌNG: KHÔNG dùng rf.feature_importances_ mặc định — nó thiên vị các
    # biến phân loại có NHIỀU mức giá trị (VD: airline có ~6 hãng vs stops có 3 mức),
    # vì Gini/MSE importance có xu hướng cho điểm cao hơn với biến có nhiều điểm chia khả dĩ.
    # Permutation importance đo trực tiếp: xáo trộn 1 cột -> hiệu năng giảm bao nhiêu.
    result = permutation_importance(
        pipe, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, xerr=result.importances_std)
    ax.set_xlabel("Mức tăng RMSE khi xáo trộn cột (permutation importance)")
    ax.set_title("Permutation Importance (đo trên tập test)")
    fig.tight_layout()
    fig.savefig(reports_dir / "permutation_importance.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/permutation_importance.png")
    log.info(f"\n{importances.sort_values(ascending=False)}")

    # --- SO SÁNH với Gini/MSE importance mặc định để CHỨNG MINH claim thiên vị ---
    # feature_importances_ mặc định chỉ có ở mức đặc trưng SAU one-hot (VD:
    # "airline_Vistara"), nên cần gộp lại theo cột gốc trước khi so sánh công bằng.
    prep: ColumnTransformer = pipe.named_steps["prep"]
    rf: RandomForestRegressor = pipe.named_steps["model"]
    encoded_names = prep.get_feature_names_out()
    gini_raw = pd.Series(rf.feature_importances_, index=encoded_names)

    def _original_col(encoded_name: str) -> str:
        # "cat__airline_Vistara" -> "airline" ; "num__duration" -> "duration"
        name = encoded_name.split("__", 1)[-1]
        for col in CATEGORICAL_COLS:
            if name.startswith(col + "_"):
                return col
        return name  # numeric passthrough giữ nguyên tên

    gini_by_original_col = gini_raw.groupby(_original_col).sum().sort_values()

    compare_df = pd.DataFrame(
        {
            "permutation_importance": importances,
            "gini_importance_gop": gini_by_original_col.reindex(importances.index),
        }
    )
    # Chuẩn hoá cả 2 về [0, 1] để so sánh THỨ HẠNG (không phải đơn vị)
    compare_norm = compare_df / compare_df.sum()

    fig, ax = plt.subplots(figsize=(9, 5))
    compare_norm.sort_values("permutation_importance").plot(kind="barh", ax=ax)
    ax.set_xlabel("Tỉ trọng chuẩn hoá")
    ax.set_title("So sánh Permutation Importance vs Gini Importance (đã gộp theo cột gốc)")
    fig.tight_layout()
    fig.savefig(reports_dir / "so_sanh_permutation_vs_gini.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/so_sanh_permutation_vs_gini.png")
    log.info(f"\n{compare_norm.sort_values('permutation_importance', ascending=False)}")

    return {
        "permutation_importance": importances.to_dict(),
        "gini_importance_gop_theo_cot_goc": gini_by_original_col.to_dict(),
    }


# --------------------------------------------------------------------------
# 10. PARTIAL DEPENDENCE PLOT cho days_left
# --------------------------------------------------------------------------
def run_pdp_days_left(pipe: Pipeline, X_train, reports_dir: Path) -> dict:
    # sklearn >= 1.9 sẽ raise lỗi (hiện tại chỉ cảnh báo) nếu cột dùng cho PDP là
    # kiểu int -> ép sang float trước để code không "hết hạn" ở bản sklearn mới.
    X_train_pdp = X_train.copy()
    X_train_pdp["days_left"] = X_train_pdp["days_left"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 5))
    display = PartialDependenceDisplay.from_estimator(
        pipe, X_train_pdp, features=["days_left"], ax=ax, kind="average",
    )
    ax.set_title("Partial Dependence Plot — days_left")
    fig.tight_layout()
    fig.savefig(reports_dir / "pdp_days_left.png", dpi=150)
    plt.close(fig)
    log.info("Đã lưu reports/pdp_days_left.png")

    # Kết luận bằng con số tiền cụ thể: đặt vé sớm hơn 1 tuần (7 ngày) tiết kiệm bao nhiêu?
    pd_results = display.pd_results[0]
    grid = pd_results["grid_values"][0]
    avg = pd_results["average"][0]

    def value_at(days_target):
        idx = int(np.argmin(np.abs(grid - days_target)))
        return avg[idx]

    price_at_1 = value_at(1)     # mua sát ngày bay (1 ngày trước)
    price_at_8 = value_at(8)     # mua sớm hơn 7 ngày
    savings = price_at_1 - price_at_8

    conclusion = {
        "gia_du_kien_neu_mua_truoc_1_ngay": float(price_at_1),
        "gia_du_kien_neu_mua_truoc_8_ngay": float(price_at_8),
        "tiet_kiem_khi_mua_som_hon_1_tuan": float(savings),
    }
    log.info(
        f"[PDP] Mua trước 1 ngày ~ {price_at_1:,.0f}đ | "
        f"Mua trước 8 ngày ~ {price_at_8:,.0f}đ | "
        f"=> Đặt sớm hơn 1 tuần tiết kiệm trung bình ~ {savings:,.0f}đ"
    )
    return conclusion


# --------------------------------------------------------------------------
# 11. KHOẢNG DỰ BÁO 10–90% TỪ CÁC CÂY
# --------------------------------------------------------------------------
def run_prediction_interval(pipe: Pipeline, X_test, y_test, reports_dir: Path) -> dict:
    prep: ColumnTransformer = pipe.named_steps["prep"]
    rf: RandomForestRegressor = pipe.named_steps["model"]

    X_test_t = prep.transform(X_test)
    # Lấy dự đoán của TỪNG cây riêng lẻ -> có phân phối cho mỗi mẫu -> suy ra khoảng
    per_tree_preds = np.stack([tree.predict(X_test_t) for tree in rf.estimators_])
    low = np.percentile(per_tree_preds, 10, axis=0)
    high = np.percentile(per_tree_preds, 90, axis=0)

    y_test_arr = np.asarray(y_test)
    coverage = float(np.mean((y_test_arr >= low) & (y_test_arr <= high)))
    avg_width = float(np.mean(high - low))

    fig, ax = plt.subplots(figsize=(7, 7))
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(y_test_arr), size=min(300, len(y_test_arr)), replace=False)
    y_sample = y_test_arr[sample_idx]
    low_s, high_s = low[sample_idx], high[sample_idx]
    order = np.argsort(y_sample)
    ax.fill_between(
        range(len(order)), low_s[order], high_s[order],
        alpha=0.3, label="Khoảng dự báo 10–90%",
    )
    ax.plot(range(len(order)), y_sample[order], ".", color="black", markersize=3, label="Giá thật")
    ax.set_xlabel("Mẫu (sắp xếp theo giá thật)")
    ax.set_ylabel("Giá vé (Rupee)")
    ax.set_title(f"Khoảng dự báo 10–90% — độ phủ thực tế = {coverage:.1%}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "khoang_du_bao.png", dpi=150)
    plt.close(fig)
    log.info(
        f"Đã lưu reports/khoang_du_bao.png — độ phủ thực tế (kỳ vọng ~80%): {coverage:.1%}, "
        f"độ rộng khoảng trung bình: {avg_width:,.0f}đ"
    )

    return {"coverage_10_90": coverage, "do_rong_khoang_trung_binh": avg_width}


# --------------------------------------------------------------------------
# 12. THÍ NGHIỆM NGOẠI SUY: days_left = 100 (trên NHIỀU hồ sơ đại diện)
# --------------------------------------------------------------------------
def run_extrapolation_experiment(
    pipe: Pipeline, X_train, reports_dir: Path, n_profiles: int = 30
) -> dict:
    """Kiểm tra hiện tượng 'kẹp trần' khi ngoại suy.

    Bản v1 chỉ thử trên 1 dòng dữ liệu đầu tiên (X_train.iloc[[0]]) — dễ gây
    hiểu lầm nếu dòng đó không đại diện. Bản này lấy trung bình trên NHIỀU hồ sơ
    khách đại diện (mẫu ngẫu nhiên) để kết luận chắc chắn hơn về mặt thống kê.
    """
    max_days_in_train = int(X_train["days_left"].max())
    rng = np.random.RandomState(RANDOM_STATE)
    profile_idx = rng.choice(len(X_train), size=min(n_profiles, len(X_train)), replace=False)
    base_rows = X_train.iloc[profile_idx].copy()

    test_days = sorted(set(list(range(1, max_days_in_train + 1, 5)) + [max_days_in_train, 100]))

    mean_preds, std_preds = [], []
    for d in test_days:
        rows = base_rows.copy()
        rows["days_left"] = d
        preds_d = pipe.predict(rows)
        mean_preds.append(float(np.mean(preds_d)))
        std_preds.append(float(np.std(preds_d)))

    mean_preds = np.array(mean_preds)
    std_preds = np.array(std_preds)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(test_days, mean_preds, marker="o", markersize=3, label=f"Trung bình trên {len(base_rows)} hồ sơ")
    ax.fill_between(test_days, mean_preds - std_preds, mean_preds + std_preds, alpha=0.2, label="± 1 std")
    ax.axvline(max_days_in_train, color="red", linestyle="--",
               label=f"Giới hạn train (days_left_max={max_days_in_train})")
    ax.set_xlabel("days_left")
    ax.set_ylabel("Giá dự đoán trung bình (Rupee)")
    ax.set_title("Thí nghiệm ngoại suy: dự đoán khi days_left vượt khoảng train")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "ngoai_suy_days_left.png", dpi=150)
    plt.close(fig)

    idx_max = test_days.index(max_days_in_train)
    idx_100 = test_days.index(100) if 100 in test_days else None
    pred_at_max = mean_preds[idx_max]
    pred_at_100 = mean_preds[idx_100] if idx_100 is not None else None

    # So sánh tương đối (không dùng abs diff < 1e-6 như bản cũ — quá chặt khi
    # lấy trung bình nhiều hồ sơ có thể có sai số làm tròn nhỏ).
    is_capped = (
        pred_at_100 is not None
        and abs(pred_at_100 - pred_at_max) / max(pred_at_max, 1.0) < 0.01  # lệch < 1%
    )

    log.info(
        f"[Ngoại suy] days_left tối đa trong train = {max_days_in_train}. "
        f"Dự đoán TB tại max_train = {pred_at_max:,.0f}đ; "
        f"tại days_left=100 = {pred_at_100:,.0f}đ (trên {len(base_rows)} hồ sơ đại diện)."
    )
    if is_capped:
        log.info(
            "[Ngoại suy] => Kết quả bị 'kẹp trần' đúng như lý thuyết: cây quyết định "
            "chỉ có thể dự đoán giá trị trung bình của lá cuối cùng nó từng thấy trong "
            "train, không thể suy luận xu hướng tiếp tục ra ngoài khoảng đã học."
        )

    return {
        "max_days_left_trong_train": max_days_in_train,
        "so_ho_so_dai_dien": int(len(base_rows)),
        "du_doan_trung_binh_tai_max_train": float(pred_at_max),
        "du_doan_trung_binh_tai_days_left_100": float(pred_at_100) if pred_at_100 is not None else None,
        "bi_kep_tran": bool(is_capped),
    }


# --------------------------------------------------------------------------
# 13. SANITY CHECK CHỐNG RÒ RỈ DỮ LIỆU: SHUFFLED-TARGET TEST
# --------------------------------------------------------------------------
def run_shuffled_target_sanity_check(X_train, X_test, y_train, y_test, preprocessor) -> dict:
    """Nếu train lại với target bị xáo trộn ngẫu nhiên mà R² vẫn cao -> CHẮC CHẮN
    có rò rỉ dữ liệu (một cột nào đó vô tình "nhìn thấy" target thật, VD: do lỗi
    join/feature engineering). Đây là bài test bắt buộc trước khi tin vào R² cao.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    y_train_shuffled = pd.Series(
        rng.permutation(y_train.values), index=y_train.index, name=y_train.name
    )

    rf = RandomForestRegressor(
        n_estimators=100, max_features=1.0, min_samples_leaf=2,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    pipe = Pipeline([("prep", preprocessor), ("model", rf)])
    pipe.fit(X_train, y_train_shuffled)
    pred = pipe.predict(X_test)
    r2_shuffled = float(r2_score(y_test, pred))

    passed = r2_shuffled < 0.05  # kỳ vọng gần 0 (hoặc âm) nếu không rò rỉ
    log.info(
        f"[Sanity check] R² khi train trên target đã xáo trộn = {r2_shuffled:.4f} "
        f"(kỳ vọng ~0) -> {'HỢP LỆ, không rò rỉ' if passed else 'CẢNH BÁO: có thể rò rỉ dữ liệu!'}"
    )
    return {"r2_voi_target_xao_tron": r2_shuffled, "vuot_qua_kiem_tra": passed}


# --------------------------------------------------------------------------
# 14. (TUỲ CHỌN) SO SÁNH VỚI XGBOOST
# --------------------------------------------------------------------------
def try_xgboost_comparison(X_train, X_test, y_train, y_test, preprocessor) -> dict | None:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        log.info("[Bỏ qua] Chưa cài xgboost — bỏ qua bước so sánh (không bắt buộc, xem TT-19).")
        return None

    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric="rmse",
    )
    prep = preprocessor
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)
    # early stopping cần eval set riêng, tách từ train (không đụng tới test)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_t, y_train, test_size=0.15, random_state=RANDOM_STATE
    )
    xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    pred = xgb.predict(X_test_t)
    result = _regression_metrics(y_test, pred)
    result["best_n_estimators_sau_early_stopping"] = int(getattr(xgb, "best_iteration", xgb.n_estimators))
    log.info(
        f"[XGBoost] RMSE={result['rmse']:.0f}  MAE={result['mae']:.0f}  R2={result['r2']:.4f}  "
        f"(dừng sớm tại cây thứ {result['best_n_estimators_sau_early_stopping']})"
    )
    return result


# --------------------------------------------------------------------------
# METADATA — để tái lập kết quả
# --------------------------------------------------------------------------
def build_run_metadata(elapsed_seconds: float, args: argparse.Namespace) -> dict:
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "random_state": RANDOM_STATE,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_col": TARGET_COL,
        "cli_args": vars(args),
        "thoi_gian_chay_giay": elapsed_seconds,
    }


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TT-17 Random Forest Regressor — Flight Price")
    parser.add_argument("--data", type=str, default="data/Clean_Dataset.csv",
                         help="Đường dẫn tới Clean_Dataset.csv")
    parser.add_argument("--outdir", type=str, default=".",
                         help="Thư mục gốc chứa models/ và reports/")
    parser.add_argument("--no-tune", action="store_true",
                         help="Bỏ qua bước RandomizedSearchCV (dùng RF mặc định, chạy nhanh hơn)")
    parser.add_argument("--tune-iter", type=int, default=20,
                         help="Số cấu hình thử trong RandomizedSearchCV")
    parser.add_argument("--tune-sample-frac", type=float, default=0.35,
                         help="Tỉ lệ dữ liệu train dùng để tìm siêu tham số (0-1)")
    parser.add_argument("--cv-folds", type=int, default=5,
                         help="Số fold cho cross-validation đánh giá model cuối")
    parser.add_argument("--fast", action="store_true",
                         help="Chế độ debug nhanh: ít cây hơn, ít cấu hình tinh chỉnh hơn")
    args = parser.parse_args()

    root = Path(args.outdir)
    reports_dir = root / "reports"
    models_dir = root / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    summary = {}

    # 1. Load + kiểm tra chất lượng dữ liệu
    df = load_data(Path(args.data))

    # 2. EDA
    run_eda(df, reports_dir)

    # 3-4. Split X/y — CÓ PHÂN TẦNG theo class (biến chi phối mạnh nhất)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=X[STRATIFY_COL]
    )
    log.info(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} (stratify theo '{STRATIFY_COL}')")

    preprocessor = build_preprocessor()

    # 5. Baseline
    summary["baselines"] = run_baselines(X_train, X_test, y_train, y_test, preprocessor)

    # 6. Tinh chỉnh siêu tham số (mặc định BẬT — dùng --no-tune để bỏ qua)
    best_rf_params = {}
    if not args.no_tune:
        n_iter = 5 if args.fast else args.tune_iter
        tune_result = tune_random_forest(
            X_train, y_train, build_preprocessor(),
            n_iter=n_iter, cv_folds=3, search_sample_frac=args.tune_sample_frac,
        )
        summary["hyperparameter_tuning"] = tune_result
        best_rf_params = tune_result["best_params"]
    else:
        log.info("Bỏ qua bước tinh chỉnh siêu tham số (--no-tune).")

    # 7. Random Forest CUỐI CÙNG — refit trên TOÀN BỘ train với tham số tốt nhất
    #    (hoặc mặc định nếu --no-tune), dùng cho mọi bước phân tích phía sau.
    if args.fast:
        best_rf_params.setdefault("n_estimators", 100)
    rf_pipe = train_random_forest(X_train, y_train, build_preprocessor(), **best_rf_params)
    pred_test = rf_pipe.predict(X_test)
    summary["random_forest"] = {
        **_regression_metrics(y_test, pred_test),
        "oob_score_": float(rf_pipe.named_steps["model"].oob_score_),
        "final_params": {**best_rf_params} if best_rf_params else "mặc định (xem train_random_forest)",
    }
    log.info(
        f"[Random Forest] Test RMSE={summary['random_forest']['rmse']:.0f}  "
        f"MAE={summary['random_forest']['mae']:.0f}  R2={summary['random_forest']['r2']:.4f}"
    )

    # 8. Cross-validation cho model cuối (đánh giá ổn định, không chỉ 1 split)
    if not args.fast:
        summary["cross_validation"] = run_cross_validation(
            Pipeline([("prep", build_preprocessor()),
                      ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE, **best_rf_params))]),
            X, y, cv_folds=args.cv_folds,
        )

    # Kiểm tra rò rỉ / R2 quá cao bất thường
    if summary["random_forest"]["r2"] > 0.995:
        log.warning("R2 > 0.995 — kiểm tra lại có rò rỉ dữ liệu không (VD: cột nào đó suy ra trực tiếp từ price).")

    # 9. Sanity check chống rò rỉ dữ liệu (shuffled-target test)
    summary["sanity_check_shuffled_target"] = run_shuffled_target_sanity_check(
        X_train, X_test, y_train, y_test, build_preprocessor()
    )

    # 10. RMSE theo n_estimators
    n_values = [10, 20, 30, 50, 100, 200] if args.fast else None
    summary["rmse_vs_n_estimators"] = rmse_vs_n_estimators(
        X_train, X_test, y_train, y_test, build_preprocessor(), reports_dir, n_values=n_values
    )

    # 11. Permutation importance + so sánh với Gini importance
    summary["feature_importance"] = run_permutation_importance(rf_pipe, X_test, y_test, reports_dir)

    # 12. PDP days_left
    summary["pdp_days_left"] = run_pdp_days_left(rf_pipe, X_train, reports_dir)

    # 13. Khoảng dự báo 10-90%
    summary["prediction_interval"] = run_prediction_interval(rf_pipe, X_test, y_test, reports_dir)

    # 14. Ngoại suy (trên nhiều hồ sơ đại diện)
    summary["extrapolation"] = run_extrapolation_experiment(rf_pipe, X_train, reports_dir)

    # 15. XGBoost (tuỳ chọn)
    xgb_result = try_xgboost_comparison(X_train, X_test, y_train, y_test, build_preprocessor())
    if xgb_result:
        summary["xgboost_comparison"] = xgb_result

    elapsed = time.time() - t0
    summary["metadata"] = build_run_metadata(elapsed, args)

    # Lưu model + báo cáo JSON
    model_path = models_dir / "rf_reg.joblib"
    joblib.dump(rf_pipe, model_path)
    with open(reports_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"[HOÀN TẤT] Tổng thời gian: {elapsed:.1f}s")
    log.info(f"[HOÀN TẤT] Model lưu tại: {model_path}")
    log.info(f"[HOÀN TẤT] Báo cáo lưu tại: {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
