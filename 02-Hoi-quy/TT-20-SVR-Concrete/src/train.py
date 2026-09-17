"""
TT-20 — SVR dự đoán cường độ chịu nén bê tông.
Chạy toàn bộ thí nghiệm end-to-end và xuất hình + bảng vào reports/.

    python src/train.py              # chạy tất cả
    python src/train.py --steps 4 5  # chỉ chạy vài bước
    python src/train.py --data path/to/Concrete_Data.xls
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.compose import TransformedTargetRegressor  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, KFold, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR, LinearSVR  # noqa: E402

from data import DOMAIN_FEATURES, add_domain_features, get_xy, load_raw  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
SEED = 42

plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


# --------------------------------------------------------------------------- #
# Tiện ích
# --------------------------------------------------------------------------- #
def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def evaluate(model, Xtr, ytr, Xte, yte) -> dict:
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t0
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    return {
        "rmse_train": rmse(ytr, ptr),
        "rmse_test": rmse(yte, pte),
        "mae_test": float(mean_absolute_error(yte, pte)),
        "r2_train": float(r2_score(ytr, ptr)),
        "r2_test": float(r2_score(yte, pte)),
        "fit_seconds": fit_s,
    }


def make_svr(**kw) -> TransformedTargetRegressor:
    """SVR chuẩn: scale X trong pipeline + scale y qua TransformedTargetRegressor."""
    params = dict(kernel="rbf", C=100.0, gamma="scale", epsilon=0.1)
    params.update(kw)
    pipe = Pipeline([("scale", StandardScaler()), ("svr", SVR(**params))])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def save(fig, name: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORTS / name, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] reports/{name}")


# --------------------------------------------------------------------------- #
# B1 — EDA
# --------------------------------------------------------------------------- #
def step1_eda(df: pd.DataFrame) -> None:
    d = add_domain_features(df)
    cols = [c for c in d.columns if c != "Strength"]

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    for ax, c in zip(axes.ravel(), cols):
        ax.scatter(d[c], d["Strength"], s=6, alpha=0.4, color="#1f77b4")
        ax.set_xlabel(c, fontsize=8)
        ax.set_ylabel("MPa", fontsize=8)
        r = np.corrcoef(d[c], d["Strength"])[0, 1]
        ax.set_title(f"r = {r:+.2f}", fontsize=8)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Đặc trưng vs Cường độ chịu nén", fontsize=12)
    save(fig, "eda_scatter.png")

    corr = d.corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Ma trận tương quan (gồm đặc trưng miền)")
    ax.grid(False)
    save(fig, "correlation_heatmap.png")

    s = d.corr()["Strength"].drop("Strength").sort_values(key=abs, ascending=False)
    show("Tương quan với cường độ (|r| giảm dần)",
         pd.DataFrame({"Đặc trưng": s.index, "r": s.round(3).values}))

    # Phân bố Age lệch phải rất mạnh
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(d["Age"], bins=40, color="#d62728")
    axes[0].set_title(f"Age (skew = {d['Age'].skew():.2f})")
    axes[1].hist(d["log_age"], bins=40, color="#2ca02c")
    axes[1].set_title(f"log1p(Age) (skew = {d['log_age'].skew():.2f})")
    save(fig, "age_distribution.png")


# --------------------------------------------------------------------------- #
# B3 — Baseline
# --------------------------------------------------------------------------- #
def step3_baseline(Xtr, Xte, ytr, yte) -> pd.DataFrame:
    rows = []
    for name, m in [
        ("DummyRegressor (trung bình)", DummyRegressor(strategy="mean")),
        ("Linear Regression", Pipeline([("s", StandardScaler()), ("lr", LinearRegression())])),
    ]:
        r = evaluate(m, Xtr, ytr, Xte, yte)
        rows.append({"Model": name, "RMSE test": round(r["rmse_test"], 3), "R² test": round(r["r2_test"], 4)})
    out = pd.DataFrame(rows)
    show("BƯỚC 3 — Baseline", out)
    return out


# --------------------------------------------------------------------------- #
# B4 + B5 — Có / không chuẩn hoá
# --------------------------------------------------------------------------- #
def step45_scaling(Xtr, Xte, ytr, yte) -> pd.DataFrame:
    y_tr_s = StandardScaler().fit(ytr.reshape(-1, 1))

    variants = {
        "Không scale gì cả": SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
        "Chỉ scale X": Pipeline([("s", StandardScaler()),
                                 ("svr", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))]),
        "Scale cả X và y (đúng)": make_svr(C=100, epsilon=0.1),
    }

    rows = []
    for name, m in variants.items():
        r = evaluate(m, Xtr, ytr, Xte, yte)
        # Đếm support vector để thấy ε=0.1 vô nghĩa khi y chưa scale
        svr = m.regressor_.named_steps["svr"] if isinstance(m, TransformedTargetRegressor) else (
            m.named_steps["svr"] if isinstance(m, Pipeline) else m)
        n_sv = int(svr.support_.shape[0])
        rows.append({
            "Cấu hình": name,
            "RMSE test": round(r["rmse_test"], 3),
            "R² test": round(r["r2_test"], 4),
            "Số SV": n_sv,
            "% SV": f"{100 * n_sv / len(Xtr):.1f}%",
        })
    out = pd.DataFrame(rows)
    show("BƯỚC 4+5 — Ảnh hưởng của chuẩn hoá (ε = 0.1 cố định)", out)
    print(f"  Độ lệch chuẩn của y = {ytr.std():.2f} MPa -> ε=0.1 chỉ bằng "
          f"{0.1 / ytr.std() * 100:.2f}% độ lệch chuẩn khi KHÔNG scale y.")
    print(f"  Sau khi scale y (std = 1), ε=0.1 tương đương ±{0.1 * ytr.std():.2f} MPa — hợp lý.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    x = np.arange(len(out))
    axes[0].bar(x, out["RMSE test"], color=["#d62728", "#ff7f0e", "#2ca02c"])
    axes[0].set_xticks(x, out["Cấu hình"], rotation=12, fontsize=7)
    axes[0].set_ylabel("RMSE test (MPa)")
    for i, v in enumerate(out["RMSE test"]):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(x, out["R² test"], color=["#d62728", "#ff7f0e", "#2ca02c"])
    axes[1].set_xticks(x, out["Cấu hình"], rotation=12, fontsize=7)
    axes[1].set_ylabel("R² test")
    axes[1].set_ylim(min(0, out["R² test"].min() * 1.1), 1)
    for i, v in enumerate(out["R² test"]):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("SVR: tác động của việc chuẩn hoá X và y")
    save(fig, "scale_vs_noscale.png")
    return out


# --------------------------------------------------------------------------- #
# B6 — So sánh kernel
# --------------------------------------------------------------------------- #
def step6_kernels(Xtr, Xte, ytr, yte) -> pd.DataFrame:
    kernels = {
        "linear": dict(kernel="linear", C=100, epsilon=0.1),
        "poly (deg 2)": dict(kernel="poly", degree=2, C=100, gamma="scale", epsilon=0.1, coef0=1),
        "poly (deg 3)": dict(kernel="poly", degree=3, C=100, gamma="scale", epsilon=0.1, coef0=1),
        "rbf": dict(kernel="rbf", C=100, gamma="scale", epsilon=0.1),
    }
    rows = []
    for name, kw in kernels.items():
        m = make_svr(**kw)
        r = evaluate(m, Xtr, ytr, Xte, yte)
        rows.append({
            "Kernel": name,
            "RMSE test": round(r["rmse_test"], 3),
            "R² test": round(r["r2_test"], 4),
            "Thời gian fit (s)": round(r["fit_seconds"], 2),
        })
    out = pd.DataFrame(rows)
    show("BƯỚC 6 — So sánh kernel (C=100, ε=0.1)", out)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(out))
    ax.bar(x - 0.2, out["RMSE test"], 0.4, label="RMSE test (MPa)", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, out["R² test"], "o--", color="#d62728", label="R² test")
    ax2.set_ylabel("R² test")
    ax2.grid(False)
    ax.set_xticks(x, out["Kernel"])
    ax.set_ylabel("RMSE test (MPa)")
    ax.set_title("So sánh kernel của SVR")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    save(fig, "kernel_comparison.png")
    return out


# --------------------------------------------------------------------------- #
# B7 + B8 — GridSearchCV và heatmap C × gamma
# --------------------------------------------------------------------------- #
def step78_grid(Xtr, Xte, ytr, yte):
    grid = {
        "regressor__svr__C": [1, 10, 100, 1000],
        "regressor__svr__gamma": ["scale", 0.01, 0.1, 1],
        "regressor__svr__epsilon": [0.01, 0.1, 0.5],
    }
    gs = GridSearchCV(
        make_svr(), grid,
        scoring="neg_root_mean_squared_error",
        cv=KFold(5, shuffle=True, random_state=SEED),
        n_jobs=-1, refit=True,
    )
    t0 = time.perf_counter()
    gs.fit(Xtr, ytr)
    print(f"\n=== BƯỚC 7 — GridSearchCV ({len(gs.cv_results_['params'])} tổ hợp, "
          f"{time.perf_counter() - t0:.1f}s) ===")
    best = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    print(f"  Tham số tốt nhất : {best}")
    print(f"  RMSE CV          : {-gs.best_score_:.3f} MPa")

    res = pd.DataFrame(gs.cv_results_)
    res["C"] = res["param_regressor__svr__C"].astype(float)
    res["gamma"] = res["param_regressor__svr__gamma"].astype(str)
    res["rmse"] = -res["mean_test_score"]
    # với mỗi (C, gamma) lấy epsilon tốt nhất
    piv = res.pivot_table(index="C", columns="gamma", values="rmse", aggfunc="min")
    piv = piv[["scale"] + [c for c in piv.columns if c != "scale"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [f"{int(c)}" for c in piv.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="RMSE CV (MPa)")
    ax.set_title("RMSE cross-validation theo (C, gamma)")
    ax.grid(False)
    save(fig, "C_gamma_heatmap.png")

    show("BƯỚC 8 — Bảng RMSE CV theo (C, gamma)", piv.round(3).reset_index())
    return gs


# --------------------------------------------------------------------------- #
# B9 — Support vectors + chẩn đoán model cuối
# --------------------------------------------------------------------------- #
def step9_support_vectors(model, Xtr, ytr, Xte, yte) -> dict:
    svr = model.regressor_.named_steps["svr"]
    n_sv = int(svr.support_.shape[0])
    eps = svr.epsilon
    alpha = np.abs(svr.dual_coef_).ravel()
    n_bound = int(np.sum(alpha >= svr.C - 1e-6))

    r_tr, r_te = model.predict(Xtr), model.predict(Xte)
    print("\n=== BƯỚC 9 — Support vectors & chất lượng model cuối ===")
    print(f"  Số support vector : {n_sv}/{len(Xtr)}  ({100 * n_sv / len(Xtr):.1f}% dữ liệu train)")
    print(f"  SV chạm biên (|α| = C) : {n_bound} ({100 * n_bound / len(Xtr):.1f}%)")
    print(f"  ε = {eps} (trên thang y đã chuẩn hoá) ≈ ±{eps * ytr.std():.2f} MPa")
    print(f"  RMSE train/test : {rmse(ytr, r_tr):.3f} / {rmse(yte, r_te):.3f} MPa")
    print(f"  R²   train/test : {r2_score(ytr, r_tr):.4f} / {r2_score(yte, r_te):.4f}")

    resid = yte - r_te
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].scatter(yte, r_te, s=10, alpha=0.5)
    lim = [min(yte.min(), r_te.min()) - 2, max(yte.max(), r_te.max()) + 2]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set_xlabel("Thực tế (MPa)")
    axes[0].set_ylabel("Dự đoán (MPa)")
    axes[0].set_title("Dự đoán vs Thực tế (test)")
    axes[1].scatter(r_te, resid, s=10, alpha=0.5)
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].axhspan(-eps * ytr.std(), eps * ytr.std(), color="orange", alpha=0.2,
                    label=f"ống ε ≈ ±{eps * ytr.std():.1f} MPa")
    axes[1].set_xlabel("Dự đoán (MPa)")
    axes[1].set_ylabel("Phần dư (MPa)")
    axes[1].set_title("Phần dư")
    axes[1].legend(fontsize=7)
    axes[2].hist(resid, bins=30, color="#1f77b4")
    axes[2].set_title(f"Phân bố phần dư (μ={resid.mean():.2f})")
    axes[2].set_xlabel("MPa")
    save(fig, "residual_analysis.png")

    return {"n_support_vectors": n_sv, "pct_support_vectors": 100 * n_sv / len(Xtr),
            "n_at_bound": n_bound, "rmse_test": rmse(yte, r_te), "r2_test": float(r2_score(yte, r_te))}


# --------------------------------------------------------------------------- #
# B10 — Giá trị của đặc trưng miền
# --------------------------------------------------------------------------- #
def step10_domain_ablation(df: pd.DataFrame, best_params: dict) -> pd.DataFrame:
    rows = []
    for label, use_domain in [("8 đặc trưng gốc", False), ("+ đặc trưng miền (đầy đủ)", True)]:
        X, y, cols = get_xy(df, use_domain=use_domain)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
        r = evaluate(make_svr(**best_params), Xtr, ytr, Xte, yte)
        rows.append({"Bộ đặc trưng": label, "Số cột": len(cols),
                     "RMSE test": round(r["rmse_test"], 3), "R² test": round(r["r2_test"], 4)})

    # bỏ riêng water_cement_ratio để đo đóng góp của chính nó
    d = add_domain_features(df)
    cols_no_wc = [c for c in d.columns if c not in ("Strength", "water_cement_ratio")]
    X = d[cols_no_wc].to_numpy(float)
    y = d["Strength"].to_numpy(float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    r = evaluate(make_svr(**best_params), Xtr, ytr, Xte, yte)
    rows.insert(1, {"Bộ đặc trưng": "Đầy đủ − water_cement_ratio", "Số cột": len(cols_no_wc),
                    "RMSE test": round(r["rmse_test"], 3), "R² test": round(r["r2_test"], 4)})

    out = pd.DataFrame(rows)
    gain = out.iloc[0]["RMSE test"] - out.iloc[-1]["RMSE test"]
    show("BƯỚC 10 — Đóng góp của đặc trưng từ kiến thức miền", out)
    print(f"  Cải thiện RMSE nhờ nhóm đặc trưng miền: {gain:+.3f} MPa "
          f"({100 * gain / out.iloc[0]['RMSE test']:.1f}%)")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.barh(out["Bộ đặc trưng"], out["RMSE test"], color=["#d62728", "#ff7f0e", "#2ca02c"])
    for i, v in enumerate(out["RMSE test"]):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("RMSE test (MPa)")
    ax.set_title("Hiệu quả của đặc trưng miền")
    save(fig, "domain_feature_gain.png")
    return out


# --------------------------------------------------------------------------- #
# B11 — SVR vs XGBoost vs Random Forest vs LinearSVR
# --------------------------------------------------------------------------- #
def step11_model_comparison(Xtr, Xte, ytr, yte, best_params: dict) -> pd.DataFrame:
    models = {
        "SVR (rbf, đã tinh chỉnh)": make_svr(**best_params),
        "LinearSVR": TransformedTargetRegressor(
            regressor=Pipeline([("s", StandardScaler()),
                                ("m", LinearSVR(C=1.0, epsilon=0.1, max_iter=20000, random_state=SEED))]),
            transformer=StandardScaler()),
        "Random Forest": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
                                         subsample=0.8, colsample_bytree=0.8,
                                         random_state=SEED, n_jobs=-1)
    except ImportError:
        print("[warn] Chưa cài xgboost -> bỏ qua cột XGBoost (pip install xgboost)")

    rows = []
    for name, m in models.items():
        r = evaluate(m, Xtr, ytr, Xte, yte)
        t0 = time.perf_counter()
        m.predict(Xte)
        rows.append({
            "Model": name,
            "RMSE test": round(r["rmse_test"], 3),
            "MAE test": round(r["mae_test"], 3),
            "R² test": round(r["r2_test"], 4),
            "Train (s)": round(r["fit_seconds"], 2),
            "Predict (ms)": round(1000 * (time.perf_counter() - t0), 1),
        })
    out = pd.DataFrame(rows).sort_values("RMSE test").reset_index(drop=True)
    show("BƯỚC 11 — So sánh mô hình", out)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].barh(out["Model"], out["RMSE test"], color="#1f77b4")
    for i, v in enumerate(out["RMSE test"]):
        axes[0].text(v, i, f" {v:.2f}", va="center", fontsize=8)
    axes[0].set_xlabel("RMSE test (MPa)")
    axes[0].set_title("Độ chính xác")
    axes[1].barh(out["Model"], out["Train (s)"], color="#ff7f0e")
    for i, v in enumerate(out["Train (s)"]):
        axes[1].text(v, i, f" {v:.2f}s", va="center", fontsize=8)
    axes[1].set_xlabel("Thời gian train (s)")
    axes[1].set_title("Chi phí huấn luyện")
    save(fig, "model_comparison.png")
    return out


# --------------------------------------------------------------------------- #
# B12 — SVR không mở rộng được
# --------------------------------------------------------------------------- #
def step12_scalability(Xtr, ytr, best_params: dict) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    factors = [1, 2, 3, 5, 7, 10]
    rows = []
    for k in factors:
        X = np.tile(Xtr, (k, 1))
        y = np.tile(ytr, k)
        if k > 1:  # thêm nhiễu nhỏ để tránh trùng lặp hoàn toàn
            X = X + rng.normal(0, 1e-3 * (X.std(axis=0) + 1e-9), X.shape)
        t0 = time.perf_counter()
        make_svr(**best_params).fit(X, y)
        t_svr = time.perf_counter() - t0

        t0 = time.perf_counter()
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1).fit(X, y)
        t_rf = time.perf_counter() - t0
        rows.append({"Hệ số nhân": f"{k}×", "n mẫu": len(X),
                     "SVR (s)": round(t_svr, 2), "RandomForest (s)": round(t_rf, 2)})
        print(f"  {k:2d}× | n={len(X):6d} | SVR {t_svr:7.2f}s | RF {t_rf:6.2f}s")

    out = pd.DataFrame(rows)
    n = out["n mẫu"].to_numpy(float)
    t = out["SVR (s)"].to_numpy(float)
    expo = float(np.polyfit(np.log(n), np.log(np.maximum(t, 1e-6)), 1)[0])
    print(f"\n  Độ dốc log-log của SVR ≈ {expo:.2f} -> thời gian train ~ O(n^{expo:.1f})")
    print("  => Với 50.000+ dòng, SVR trở nên bất khả thi. Đây là hạn chế cốt lõi.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(n, t, "o-", label=f"SVR (~O(n^{expo:.1f}))", color="#d62728")
    axes[0].plot(n, out["RandomForest (s)"], "s-", label="Random Forest", color="#2ca02c")
    axes[0].set_xlabel("Số mẫu huấn luyện")
    axes[0].set_ylabel("Thời gian train (s)")
    axes[0].set_title("Thời gian train theo kích thước dữ liệu")
    axes[0].legend(fontsize=8)
    axes[1].loglog(n, t, "o-", color="#d62728", label="SVR")
    axes[1].loglog(n, out["RandomForest (s)"], "s-", color="#2ca02c", label="Random Forest")
    axes[1].set_xlabel("Số mẫu (log)")
    axes[1].set_ylabel("Thời gian (log)")
    axes[1].set_title("Thang log–log")
    axes[1].legend(fontsize=8)
    save(fig, "thoi_gian_train.png")
    show("BƯỚC 12 — Khả năng mở rộng", out)
    return out


# --------------------------------------------------------------------------- #
# Mở rộng — phân tích an toàn kết cấu + dự báo phân vị
# --------------------------------------------------------------------------- #
def step13_safety(model, Xtr, Xte, ytr, yte) -> pd.DataFrame:
    """Đếm mẫu bị dự báo CAO HƠN thực tế > 5 MPa (rủi ro kết cấu)."""
    q10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=400,
                                    learning_rate=0.05, max_depth=3, random_state=SEED)
    q10.fit(Xtr, ytr)

    preds = {"SVR (trung bình có điều kiện)": model.predict(Xte),
             "GBR phân vị 10% (cận dưới an toàn)": q10.predict(Xte)}

    rows = []
    for name, p in preds.items():
        over = p - yte
        rows.append({
            "Mô hình": name,
            "RMSE": round(rmse(yte, p), 3),
            "Dự báo CAO hơn thực tế": f"{100 * np.mean(over > 0):.1f}%",
            "Vượt > 5 MPa (nguy hiểm)": int(np.sum(over > 5)),
            "Vượt > 5 MPa (%)": f"{100 * np.mean(over > 5):.1f}%",
        })
    out = pd.DataFrame(rows)
    show("MỞ RỘNG — Phân tích an toàn kết cấu", out)
    print("  Dự báo phân vị thấp đánh đổi RMSE để gần như loại bỏ rủi ro đánh giá cao quá.")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    order = np.argsort(yte)
    ax.plot(yte[order], "k-", lw=1, label="Thực tế")
    ax.plot(model.predict(Xte)[order], ".", ms=3, alpha=0.6, label="SVR")
    ax.plot(q10.predict(Xte)[order], ".", ms=3, alpha=0.6, color="#2ca02c", label="Phân vị 10%")
    ax.set_xlabel("Mẫu test (sắp xếp theo cường độ thực tế)")
    ax.set_ylabel("MPa")
    ax.set_title("Dự báo trung bình vs cận dưới an toàn")
    ax.legend(fontsize=8)
    save(fig, "safety_quantile.png")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="TT-20 — SVR cường độ chịu nén bê tông")
    ap.add_argument("--data", default=None, help="Đường dẫn file dữ liệu (.xls/.csv)")
    ap.add_argument("--steps", nargs="*", default=None,
                    help="Các bước cần chạy, ví dụ: --steps 1 6 11 (mặc định: tất cả)")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    steps = set(args.steps) if args.steps else {str(i) for i in range(1, 14)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.data)
    X, y, cols = get_xy(df, use_domain=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=SEED)
    print(f"[data] Train {Xtr.shape} | Test {Xte.shape} | {len(cols)} đặc trưng "
          f"({len(DOMAIN_FEATURES)} từ kiến thức miền)")

    summary: dict = {"n_features": len(cols), "features": cols,
                     "n_train": int(len(Xtr)), "n_test": int(len(Xte))}

    if "1" in steps:
        step1_eda(df)
    if "3" in steps:
        summary["baseline"] = step3_baseline(Xtr, Xte, ytr, yte).to_dict("records")
    if "4" in steps or "5" in steps:
        summary["scaling"] = step45_scaling(Xtr, Xte, ytr, yte).to_dict("records")
    if "6" in steps:
        summary["kernels"] = step6_kernels(Xtr, Xte, ytr, yte).to_dict("records")

    best_params = dict(kernel="rbf", C=100, gamma="scale", epsilon=0.1)
    if "7" in steps or "8" in steps:
        gs = step78_grid(Xtr, Xte, ytr, yte)
        best_params = dict(kernel="rbf", **{k.split("__")[-1]: v for k, v in gs.best_params_.items()})
        summary["best_params"] = {k: str(v) for k, v in best_params.items()}
        summary["cv_rmse"] = round(-gs.best_score_, 3)

    model = make_svr(**best_params)
    model.fit(Xtr, ytr)

    if "9" in steps:
        summary["final"] = step9_support_vectors(model, Xtr, ytr, Xte, yte)
    if "10" in steps:
        summary["domain_ablation"] = step10_domain_ablation(df, best_params).to_dict("records")
    if "11" in steps:
        summary["model_comparison"] = step11_model_comparison(Xtr, Xte, ytr, yte, best_params).to_dict("records")
    if "12" in steps:
        summary["scalability"] = step12_scalability(Xtr, ytr, best_params).to_dict("records")
    if "13" in steps:
        summary["safety"] = step13_safety(model, Xtr, Xte, ytr, yte).to_dict("records")

    joblib.dump({"model": model, "features": cols}, MODELS / "svr_pipeline.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n[done] models/svr_pipeline.joblib | reports/metrics.json")
    if "final" in summary:
        ok = summary["final"]["r2_test"] > 0.88
        print(f"[tiêu chí] R² test = {summary['final']['r2_test']:.4f} "
              f"({'ĐẠT' if ok else 'CHƯA ĐẠT'} mốc 0.88)")


if __name__ == "__main__":
    main()
