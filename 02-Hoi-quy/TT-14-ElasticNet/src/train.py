"""
TT-14 — ElasticNet: dự báo tải sưởi (Y1) và tải làm mát (Y2) của toà nhà.
Chạy: python src/train.py [--data đường/dẫn/file]
"""
import argparse
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_loader import DEFAULT_DATA_PATH, NUMERIC_COLS, load_energy_data, make_design_matrix

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
ALPHAS = np.logspace(-4, 1, 100)
CORRELATED_GROUP = ["X1", "X2", "X4", "X5"]  # đa cộng tuyến gần hoàn hảo


# ---------- 1. VIF ----------
def compute_vif(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """VIF trên các biến số liên tục (không tính biến phân loại đã one-hot).

    Với bộ Energy Efficiency, X2 ~ 2*(X3+X4)/X1 gần như là hệ thức hình học
    chính xác -> VIF của X2/X3/X4 tràn số (không hội tụ), con số cụ thể vô nghĩa,
    chỉ có giá trị NÊU: "đa cộng tuyến gần như tuyệt đối". Dùng cols để tính VIF
    hữu hạn sau khi bỏ bớt 1 biến trong nhóm đó (ví dụ bỏ X2).
    """
    use_cols = cols or NUMERIC_COLS
    X = df[use_cols].values
    vif = pd.DataFrame({
        "feature": use_cols,
        "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)
    return vif


# ---------- 3-4. Baseline + 3 model ----------
def build_models():
    return {
        "Dummy": Pipeline([("scale", StandardScaler()), ("m", DummyRegressor(strategy="mean"))]),
        "LinearRegression": Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]),
        "Ridge": Pipeline([("scale", StandardScaler()),
                            ("m", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "Lasso": Pipeline([("scale", StandardScaler()),
                            ("m", Lasso(alpha=0.01, max_iter=50000, random_state=RANDOM_STATE))]),
        "ElasticNetCV": Pipeline([("scale", StandardScaler()),
                                   ("m", ElasticNetCV(l1_ratio=L1_RATIOS, alphas=ALPHAS,
                                                       cv=5, max_iter=50000,
                                                       random_state=RANDOM_STATE))]),
    }


def n_nonzero(pipe: Pipeline) -> int:
    """Đếm hệ số > 1e-6. LƯU Ý: chỉ có ý nghĩa "chọn biến" thật sự với
    Lasso/ElasticNet (có thể đẩy hệ số về đúng 0). Ridge/LinearRegression hầu
    như không bao giờ zero-out một hệ số, nên với 2 model đó con số này gần
    như luôn bằng tổng số biến — không nên đọc là "Ridge chọn N biến"."""
    model = pipe.named_steps["m"]
    if not hasattr(model, "coef_"):
        return len(model.constant_) if hasattr(model, "constant_") else 0
    return int(np.sum(np.abs(model.coef_) > 1e-6))


def evaluate_target(X, y, target_name: str, feature_names):
    """Chạy 5 model, trả về bảng so sánh + model ElasticNet đã fit trên toàn bộ train."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    rows, fitted = [], {}
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        r2 = r2_score(y_test, pred)
        rows.append({"target": target_name, "model": name,
                      "n_features_kept": n_nonzero(pipe) if name != "Dummy" else 0,
                      "RMSE": round(rmse, 3), "R2": round(r2, 4)})
        fitted[name] = pipe

    en = fitted["ElasticNetCV"]
    print(f"[{target_name}] ElasticNetCV -> alpha={en.named_steps['m'].alpha_:.5f}, "
          f"l1_ratio={en.named_steps['m'].l1_ratio_}")

    return pd.DataFrame(rows), fitted, (X_train, X_test, y_train, y_test)


# ---------- 6. Hiệu ứng gom nhóm ----------
def grouping_effect(fitted: dict, feature_names) -> pd.DataFrame:
    coefs = {}
    for name in ["Lasso", "ElasticNetCV"]:
        model = fitted[name].named_steps["m"]
        coefs[name] = dict(zip(feature_names, model.coef_))
    rows = []
    for f in CORRELATED_GROUP:
        rows.append({"feature": f, "Lasso_coef": round(coefs["Lasso"].get(f, 0.0), 4),
                     "ElasticNet_coef": round(coefs["ElasticNetCV"].get(f, 0.0), 4)})
    return pd.DataFrame(rows)


# ---------- 7. Heatmap alpha x l1_ratio ----------
def alpha_l1ratio_heatmap(X, y, target_name: str, out_path: str):
    # Zoom vào vùng alpha nhỏ (nơi RMSE tối ưu thực sự nằm, xem ElasticNetCV.alpha_
    # ~1e-3–1e-4) để tránh cả lưới bị "bão hoà" một màu vàng như alpha lớn 0.01-10.
    alphas_grid = np.logspace(-4, 0, 15)
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    grid = np.zeros((len(L1_RATIOS), len(alphas_grid)))
    for i, l1 in enumerate(L1_RATIOS):
        for j, a in enumerate(alphas_grid):
            model = ElasticNet(alpha=a, l1_ratio=l1, max_iter=20000, random_state=RANDOM_STATE)
            scores = cross_val_score(model, Xs, y, cv=kf, scoring="neg_root_mean_squared_error")
            grid[i, j] = -scores.mean()

    plt.figure(figsize=(11, 5))
    sns.heatmap(grid, xticklabels=[f"{a:.1e}" for a in alphas_grid],
                yticklabels=L1_RATIOS, cmap="viridis_r", annot=False, cbar_kws={"label": "RMSE (CV)"})
    plt.xlabel("alpha")
    plt.ylabel("l1_ratio")
    plt.title(f"RMSE theo lưới alpha x l1_ratio — {target_name}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    return grid


# ---------- 9. Bootstrap ổn định hệ số ----------
def bootstrap_stability(X, y, feature_names, n_boot=100):
    rng = np.random.RandomState(RANDOM_STATE)
    coefs = np.zeros((n_boot, X.shape[1]))
    for b in range(n_boot):
        idx = rng.choice(len(X), len(X), replace=True)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        pipe = Pipeline([("scale", StandardScaler()),
                          ("m", ElasticNetCV(l1_ratio=L1_RATIOS, alphas=ALPHAS,
                                              cv=5, max_iter=50000, random_state=b))])
        pipe.fit(Xb, yb)
        coefs[b] = pipe.named_steps["m"].coef_
    return pd.DataFrame({
        "feature": feature_names,
        "coef_mean": coefs.mean(axis=0).round(4),
        "coef_std": coefs.std(axis=0).round(4),
    }).sort_values("coef_std", ascending=False)


def comparison_plot(compare_df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, target in enumerate(compare_df["target"].unique()):
        sub = compare_df[compare_df["target"] == target]
        ax[i].bar(sub["model"], sub["RMSE"], color="#4C72B0")
        ax[i].set_title(f"RMSE theo model — {target}")
        ax[i].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def main(data_path: str):
    df = load_energy_data(data_path)
    X, y1, y2 = make_design_matrix(df)
    feature_names = list(X.columns)

    print("== 1. VIF (đa cộng tuyến) — bộ đầy đủ (kỳ vọng tràn số) ==")
    vif_df = compute_vif(df)
    print(vif_df.to_string(index=False))
    vif_df.to_csv("reports/vif_table.csv", index=False)

    print("\n== 1b. VIF sau khi bỏ X2 (để có giá trị hữu hạn, dễ đọc) ==")
    cols_no_x2 = [c for c in NUMERIC_COLS if c != "X2"]
    vif_no_x2 = compute_vif(df, cols=cols_no_x2)
    print(vif_no_x2.to_string(index=False))
    vif_no_x2.to_csv("reports/vif_table_no_X2.csv", index=False)

    all_compare = []
    fitted_store = {}
    for name, y in [("Y1_heating", y1), ("Y2_cooling", y2)]:
        print(f"\n== {name}: baseline + Ridge/Lasso/ElasticNet ==")
        cmp_df, fitted, split = evaluate_target(X, y, name, feature_names)
        print(cmp_df.to_string(index=False))
        all_compare.append(cmp_df)
        fitted_store[name] = fitted

        print(f"\n== Hiệu ứng gom nhóm ({name}) ==")
        grp = grouping_effect(fitted, feature_names)
        print(grp.to_string(index=False))
        grp.to_csv(f"reports/grouping_effect_{name}.csv", index=False)

        print(f"\n== Heatmap alpha x l1_ratio ({name}) ==")
        alpha_l1ratio_heatmap(X, y, name, f"reports/heatmap_alpha_l1ratio_{name}.png")

        print(f"\n== Bootstrap ổn định hệ số ({name}, 100 lần) ==")
        boot_df = bootstrap_stability(X, y, feature_names, n_boot=100)
        print(boot_df.to_string(index=False))
        boot_df.to_csv(f"reports/bootstrap_stability_{name}.csv", index=False)

        joblib.dump(fitted["ElasticNetCV"], f"models/elasticnet_{name.split('_')[0]}.joblib")

    compare_all = pd.concat(all_compare, ignore_index=True)
    compare_all.to_csv("reports/so_sanh_3_model.csv", index=False)
    comparison_plot(compare_all, "reports/so_sanh_3_model.png")

    print("\nXong. Xem file trong reports/ và models/.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                         help="Đường dẫn tới ENB2012_data.csv/.xlsx của bạn")
    args = parser.parse_args()
    main(args.data)
