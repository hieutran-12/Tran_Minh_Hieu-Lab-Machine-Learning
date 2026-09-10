"""
TT-16 — Decision Tree Regressor
Định giá cước chuyến xe kiểu NYC Taxi -> bảng giá dạng LUẬT cho tổng đài.

Chạy: python src/train.py
Sinh ra:
  - reports/ham_bac_thang.png
  - reports/mae_theo_depth.png
  - reports/cay_quyet_dinh.png
  - reports/scatter_distance_fare.png
  - reports/gia_trung_binh_theo_gio.png
  - reports/overfit_train_vs_test.png
  - outputs/bang_tra_cuoc.csv          <- sản phẩm bàn giao cho tổng đài
  - outputs/so_sanh_model.csv
  - models/tree_reg.joblib
"""

import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(HERE, "data", "yellow_tripdata_sample_raw.csv")
REPORT_DIR = os.path.join(HERE, "reports")
MODEL_DIR = os.path.join(HERE, "models")
OUTPUT_DIR = os.path.join(HERE, "outputs")
for d in (REPORT_DIR, MODEL_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TARGET = "fare_amount"
LEAK_COLS = ["tip_amount", "tolls_amount", "total_amount"]  # ⚠️ chỉ biết SAU chuyến đi
FEATURES = ["trip_distance", "passenger_count", "hour", "dow", "is_weekend", "is_rush", "is_night", "pickup_borough"]
CAT_FEATURES = ["pickup_borough"]


# ---------------------------------------------------------------------------
# Bước 1-2: tải + làm sạch
# ---------------------------------------------------------------------------
def load_raw():
    df = pd.read_csv(RAW_PATH, parse_dates=["tpep_pickup_datetime"])
    return df


def clean_data(df):
    log = []
    n0 = len(df)
    log.append(("Ban đầu", n0, 0))

    df = df[df["fare_amount"] > 0]
    log.append(("Loại fare_amount <= 0", len(df), n0 - len(df)))
    n1 = len(df)

    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] <= 100)]
    log.append(("Loại trip_distance <=0 hoặc > 100 miles", len(df), n1 - len(df)))
    n2 = len(df)

    df = df[df["passenger_count"] > 0]
    log.append(("Loại passenger_count == 0", len(df), n2 - len(df)))

    for col in LEAK_COLS:
        assert col not in FEATURES, f"⚠️ RÒ RỈ: {col} không được dùng làm đặc trưng"

    log_df = pd.DataFrame(log, columns=["buoc", "so_dong_con_lai", "so_dong_bi_loai"])
    print("\n=== LÀM SẠCH DỮ LIỆU ===")
    print(log_df.to_string(index=False))
    return df.reset_index(drop=True), log_df


def engineer_features(df):
    df = df.copy()
    dt = df["tpep_pickup_datetime"]
    df["hour"] = dt.dt.hour
    df["dow"] = dt.dt.dayofweek  # 0=Thứ 2 .. 6=CN
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    is_weekday = df["dow"] < 5
    df["is_rush"] = (is_weekday & (((df["hour"] >= 7) & (df["hour"] <= 10)) | ((df["hour"] >= 16) & (df["hour"] <= 20)))).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    return df


def encode_features(df):
    X = df[FEATURES].copy()
    X = pd.get_dummies(X, columns=CAT_FEATURES, prefix="zone")
    return X


# ---------------------------------------------------------------------------
# Bước 4: EDA
# ---------------------------------------------------------------------------
def step_eda(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(8000, len(df)), random_state=RANDOM_STATE)
    ax.scatter(sample["trip_distance"], sample[TARGET], s=4, alpha=0.25, color="#2b6cb0")
    ax.set_xlabel("trip_distance (dặm)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title("EDA: quãng đường vs cước — quan hệ gần tuyến tính nhưng có phụ phí rời rạc")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "scatter_distance_fare.png"), dpi=130)
    plt.close(fig)

    avg_by_hour = df.groupby("hour")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(avg_by_hour.index, avg_by_hour.values, "o-", color="#c53030")
    ax.set_xlabel("Giờ trong ngày")
    ax.set_ylabel("Cước trung bình ($)")
    ax.set_title("Cước trung bình theo giờ — rõ hai đỉnh giờ cao điểm")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "gia_trung_binh_theo_gio.png"), dpi=130)
    plt.close(fig)
    print("Đã lưu reports/scatter_distance_fare.png và reports/gia_trung_binh_theo_gio.png")


# ---------------------------------------------------------------------------
# Bước 5: baseline
# ---------------------------------------------------------------------------
def step_baseline(X_train, X_test, y_train, y_test, dist_train, dist_test):
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    mae_dummy = mean_absolute_error(y_test, dummy.predict(X_test))

    # công thức tuyến tính thủ công kiểu tổng đài: 3.0 + 2.8*km (USD, khớp cấu trúc cước thật)
    manual_pred = 3.0 + 2.8 * dist_test
    mae_manual = mean_absolute_error(y_test, manual_pred)

    print("\n=== BASELINE ===")
    print(f"DummyRegressor (mean):        MAE = ${mae_dummy:.2f}")
    print(f"Công thức thủ công 3.0+2.8*km: MAE = ${mae_manual:.2f}")
    return mae_dummy, mae_manual


# ---------------------------------------------------------------------------
# Bước 6: cây không giới hạn độ sâu -> chứng minh overfit
# ---------------------------------------------------------------------------
def step_overfit_demo(X_train, X_test, y_train, y_test):
    tree_full = DecisionTreeRegressor(random_state=RANDOM_STATE)
    tree_full.fit(X_train, y_train)
    mae_train = mean_absolute_error(y_train, tree_full.predict(X_train))
    mae_test = mean_absolute_error(y_test, tree_full.predict(X_test))
    n_leaves = tree_full.get_n_leaves()
    depth = tree_full.get_depth()
    print("\n=== CÂY KHÔNG GIỚI HẠN ĐỘ SÂU (chứng minh overfit) ===")
    print(f"Độ sâu thực tế = {depth}, số lá = {n_leaves}")
    print(f"MAE train = ${mae_train:.3f}  |  MAE test = ${mae_test:.3f}  -> lệch {mae_test - mae_train:.3f}")
    print(">>> MAE train gần 0 trong khi MAE test cao hơn nhiều lần -> cây đã HỌC THUỘC LÒNG dữ liệu train (overfit).")
    return tree_full, mae_train, mae_test, n_leaves


# ---------------------------------------------------------------------------
# Bước 7: quét max_depth 1..20
# ---------------------------------------------------------------------------
def step_depth_sweep(X_train, X_test, y_train, y_test, max_depth_range=range(1, 21)):
    rows = []
    for d in max_depth_range:
        tree = DecisionTreeRegressor(max_depth=d, min_samples_leaf=500, random_state=RANDOM_STATE)
        tree.fit(X_train, y_train)
        mae_tr = mean_absolute_error(y_train, tree.predict(X_train))
        mae_te = mean_absolute_error(y_test, tree.predict(X_test))
        rows.append({"max_depth": d, "MAE_train": mae_tr, "MAE_test": mae_te, "so_la": tree.get_n_leaves()})
    table = pd.DataFrame(rows)

    best_row = table.loc[table["MAE_test"].idxmin()]
    best_depth_by_min = int(best_row["max_depth"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["max_depth"], table["MAE_train"], "o-", label="MAE train")
    ax.plot(table["max_depth"], table["MAE_test"], "o-", label="MAE test")
    ax.axvline(best_depth_by_min, color="gray", linestyle="--", alpha=0.6, label=f"MAE test nhỏ nhất tại depth={best_depth_by_min}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("MAE ($)")
    ax.set_title("MAE train/test theo max_depth (min_samples_leaf=500)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "mae_theo_depth.png"), dpi=130)
    plt.close(fig)
    print("\n=== QUÉT max_depth = 1..20 ===")
    print(table.to_string(index=False))
    print(f"Đã lưu reports/mae_theo_depth.png — MAE test nhỏ nhất tại depth={best_depth_by_min}")
    print(
        "⭐ Nhưng theo yêu cầu nghiệp vụ (tổng đài viên tra tay), bài chọn depth=5: "
        "cây sâu hơn cho MAE test thấp hơn không đáng kể nhưng tạo ra hàng chục/hàng "
        "trăm lá — không thể in thành bảng tra tay được."
    )
    return table, best_depth_by_min


# ---------------------------------------------------------------------------
# Bước 8: hàm bậc thang
# ---------------------------------------------------------------------------
def step_staircase_plot(tree_model, X_train, df_train_raw):
    grid_len = 400
    dist_grid = np.linspace(0.1, 20, grid_len)
    base_row = X_train.iloc[0:1].copy()
    for c in X_train.columns:
        if c != "trip_distance":
            base_row[c] = X_train[c].mode().iloc[0] if X_train[c].dtype != float else X_train[c].median()
    grid_df = pd.concat([base_row] * grid_len, ignore_index=True)
    grid_df["trip_distance"] = dist_grid
    pred = tree_model.predict(grid_df)

    fig, ax = plt.subplots(figsize=(8, 5))
    sample = df_train_raw.sample(min(4000, len(df_train_raw)), random_state=RANDOM_STATE)
    ax.scatter(sample["trip_distance"], sample[TARGET], s=4, alpha=0.15, color="#a0aec0", label="Dữ liệu thật")
    ax.plot(dist_grid, pred, color="#c53030", linewidth=2, label="Cây dự đoán (điều kiện khác giữ cố định)")
    ax.set_xlabel("trip_distance (dặm)")
    ax.set_ylabel("fare_amount ($)")
    ax.set_title("Hàm dự đoán của cây là hàm BẬC THANG, không mượt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "ham_bac_thang.png"), dpi=130)
    plt.close(fig)
    n_levels = len(np.unique(np.round(pred, 4)))
    print(f"\nĐã lưu reports/ham_bac_thang.png — số mức giá khác nhau khi quét quãng đường: {n_levels}")


# ---------------------------------------------------------------------------
# Bước 9: cây cuối max_depth=5
# ---------------------------------------------------------------------------
def step_final_tree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=500, random_state=RANDOM_STATE)
    tree.fit(X_train, y_train)
    pred_test = tree.predict(X_test)
    mae = mean_absolute_error(y_test, pred_test)
    mape = mean_absolute_percentage_error(y_test, pred_test)
    rmse = root_mean_squared_error(y_test, pred_test)
    print("\n=== CÂY CUỐI max_depth=5, min_samples_leaf=500 ===")
    print(f"Số lá = {tree.get_n_leaves()}  |  MAE test = ${mae:.3f}  |  MAPE test = {mape*100:.1f}%  |  RMSE test = ${rmse:.3f}")

    rules_text = export_text(tree, feature_names=list(X_train.columns))
    with open(os.path.join(REPORT_DIR, "cay_luat_text.txt"), "w") as f:
        f.write(rules_text)
    print(rules_text[:1500] + ("\n... (xem đủ ở reports/cay_luat_text.txt)" if len(rules_text) > 1500 else ""))

    fig, ax = plt.subplots(figsize=(22, 11))
    plot_tree(
        tree,
        feature_names=list(X_train.columns),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        precision=1,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "cay_quyet_dinh.png"), dpi=130)
    plt.close(fig)
    print("Đã lưu reports/cay_quyet_dinh.png")

    return tree, mae, mape, rmse, pred_test


# ---------------------------------------------------------------------------
# Bước 10: bảng tra cước (mỗi lá = 1 dòng luật, đọc trực tiếp cấu trúc cây)
# ---------------------------------------------------------------------------
def tree_to_rate_table(tree, feature_names):
    t = tree.tree_
    rows = []

    def recurse(node, conditions):
        if t.children_left[node] == t.children_right[node]:  # lá
            rows.append(
                {
                    "dieu_kien": " VÀ ".join(conditions) if conditions else "(mọi chuyến)",
                    "gia_du_doan_usd": round(float(t.value[node][0][0]), 2),
                    "so_chuyen_trong_du_lieu_train": int(t.n_node_samples[node]),
                }
            )
            return
        feat = feature_names[t.feature[node]]
        thr = t.threshold[node]
        left_cond = f"{feat} <= {thr:.2f}"
        right_cond = f"{feat} > {thr:.2f}"
        recurse(t.children_left[node], conditions + [left_cond])
        recurse(t.children_right[node], conditions + [right_cond])

    recurse(0, [])
    table = pd.DataFrame(rows).sort_values("gia_du_doan_usd").reset_index(drop=True)
    table.insert(0, "ma_muc_gia", [f"G{i+1:02d}" for i in range(len(table))])
    return table


def step_rate_table(tree, X_train):
    table = tree_to_rate_table(tree, list(X_train.columns))
    out_path = os.path.join(OUTPUT_DIR, "bang_tra_cuoc.csv")
    table.to_csv(out_path, index=False)
    print(f"\n=== BẢNG TRA CƯỚC ({len(table)} mức giá — bằng đúng số lá) ===")
    print(table.to_string(index=False))
    print(f"Đã lưu {out_path}")
    return table


# ---------------------------------------------------------------------------
# Bước 11: % chuyến trong sai số ±15%
# ---------------------------------------------------------------------------
def step_error_within_15pct(y_test, pred_test):
    pct_err = np.abs(pred_test - y_test.values) / y_test.values
    within = (pct_err <= 0.15).mean() * 100
    print(f"\n=== KIỂM TRA SAI SỐ ±15% (yêu cầu nghiệp vụ) ===")
    print(f"{within:.1f}% số chuyến test có sai số dự đoán trong khoảng ±15% giá thật.")
    return within


# ---------------------------------------------------------------------------
# Bước 12: so sánh Random Forest & Linear Regression
# ---------------------------------------------------------------------------
def step_compare_models(X_train, X_test, y_train, y_test, tree_mae, tree_mape, tree_rmse):
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    pred_lin = lin.predict(X_test)

    rows = [
        {
            "model": "DecisionTreeRegressor (depth=5, tra bảng được)",
            "MAE": tree_mae,
            "MAPE_%": tree_mape * 100,
            "RMSE": tree_rmse,
        },
        {
            "model": "RandomForestRegressor (200 cây, depth=10)",
            "MAE": mean_absolute_error(y_test, pred_rf),
            "MAPE_%": mean_absolute_percentage_error(y_test, pred_rf) * 100,
            "RMSE": root_mean_squared_error(y_test, pred_rf),
        },
        {
            "model": "LinearRegression",
            "MAE": mean_absolute_error(y_test, pred_lin),
            "MAPE_%": mean_absolute_percentage_error(y_test, pred_lin) * 100,
            "RMSE": root_mean_squared_error(y_test, pred_lin),
        },
    ]
    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUTPUT_DIR, "so_sanh_model.csv"), index=False)
    print("\n=== SO SÁNH 3 MODEL ===")
    print(table.to_string(index=False))
    print(
        "\n>>> Random Forest chính xác hơn cây đơn nhưng KHÔNG in được thành bảng tra tay "
        "(200 cây x hàng chục lá mỗi cây). Linear Regression không bắt được phụ phí rời rạc "
        "(giờ cao điểm, ban đêm, congestion) vì các phụ phí này CỘNG THÊM rời rạc chứ không "
        "tuyến tính theo biến liên tục -> MAE thường cao hơn cây depth=5."
    )
    return table


def main():
    raw = load_raw()
    clean, clean_log = clean_data(raw)
    clean_log.to_csv(os.path.join(OUTPUT_DIR, "log_lam_sach.csv"), index=False)

    df = engineer_features(clean)
    step_eda(df)

    X = encode_features(df)
    y = df[TARGET]
    dist = df["trip_distance"]

    X_train, X_test, y_train, y_test, dist_train, dist_test, df_train_raw, df_test_raw = train_test_split(
        X, y, dist, df, test_size=0.2, random_state=RANDOM_STATE
    )

    step_baseline(X_train, X_test, y_train, y_test, dist_train, dist_test)

    tree_full, mae_tr_full, mae_te_full, n_leaves_full = step_overfit_demo(X_train, X_test, y_train, y_test)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["MAE train", "MAE test"], [mae_tr_full, mae_te_full], color=["#2f855a", "#c53030"])
    ax.set_ylabel("MAE ($)")
    ax.set_title(f"Cây không giới hạn độ sâu (depth={tree_full.get_depth()}, {n_leaves_full} lá) — overfit rõ rệt")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT_DIR, "overfit_train_vs_test.png"), dpi=130)
    plt.close(fig)

    depth_table, best_depth = step_depth_sweep(X_train, X_test, y_train, y_test)

    final_tree_for_plot = DecisionTreeRegressor(max_depth=5, min_samples_leaf=500, random_state=RANDOM_STATE)
    final_tree_for_plot.fit(X_train, y_train)
    step_staircase_plot(final_tree_for_plot, X_train, df_train_raw)

    tree, mae, mape, rmse, pred_test = step_final_tree(X_train, X_test, y_train, y_test)

    rate_table = step_rate_table(tree, X_train)

    within15 = step_error_within_15pct(y_test, pred_test)

    compare_table = step_compare_models(X_train, X_test, y_train, y_test, mae, mape, rmse)

    joblib.dump(tree, os.path.join(MODEL_DIR, "tree_reg.joblib"))

    print("\n=== TÓM TẮT ===")
    print(f"Baseline DummyRegressor:            xem chi tiết ở trên")
    print(f"Cây depth=5 (model cuối, tra bảng):  MAE=${mae:.2f}  MAPE={mape*100:.1f}%  RMSE=${rmse:.2f}")
    print(f"% chuyến trong sai số ±15%:          {within15:.1f}%")
    print(f"Số mức giá trong bảng tra cước:      {len(rate_table)}")
    print("Đã lưu models/tree_reg.joblib")


if __name__ == "__main__":
    main()
