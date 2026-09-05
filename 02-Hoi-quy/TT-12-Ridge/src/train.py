"""
TT-12 — Ridge Regression (L2): Phan bo ngan sach quang cao da kenh
====================================================================
Script end-to-end: chay `python src/train.py` tu thu muc goc du an.

Quy trinh (dung 10 buoc trong README):
 1. Nap du lieu that (advertising.csv) + sinh du lieu co da cong tuyen chu dich
 2. Ma tran tuong quan + VIF -> chung minh da cong tuyen
 3. Linear Regression baseline
 4. Thi nghiem on dinh he so bang bootstrap (Linear vs Ridge)
 5. RidgeCV do alpha toi uu (cross-validation, KHONG dung tap test)
 6. Ve coefficient path
 7. Ve duong RMSE train/test theo alpha
 8. So sanh RMSE test: Linear vs Ridge
 9. (Lasso/ElasticNet de lai cho TT-13/TT-14, khong lam o day)
10. De xuat phan bo ngan sach dua tren he so Ridge

Output: reports/*.csv, reports/*.png, models/ridge_pipeline.joblib
"""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # khong can man hinh, chi luu file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ----------------------------------------------------------------------
# Duong dan (chay tu thu muc goc du an, vd: python src/train.py)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)


# ========================================================================
# BUOC 1a — Nap du lieu that (Advertising, Kaggle) — chi de doi chieu
# ========================================================================
def load_real_advertising() -> pd.DataFrame:
    path = DATA_DIR / "advertising.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


# ========================================================================
# BUOC 1b — Sinh du lieu co da cong tuyen CHU DICH (theo muc 3 README)
# Ly do: bo du lieu that (advertising.csv, r(TV,Radio)=0.05) gan nhu
# KHONG da cong tuyen -> khong the minh hoa tac dung cua Ridge.
# README khuyen nghi tu sinh du lieu de thay ro hieu ung nay.
# ========================================================================
def make_synthetic_multicollinear(n: int = 500) -> pd.DataFrame:
    tv = rng.uniform(50, 500, n)
    fb = tv * 0.6 + rng.normal(0, 15, n)  # tuong quan cao voi TV (r ~ 0.95)
    gg = rng.uniform(20, 300, n)
    doanh_thu = 3.2 * tv + 1.8 * fb + 2.5 * gg + rng.normal(0, 50, n)
    return pd.DataFrame({"TV": tv, "Facebook": fb, "Google": gg, "DoanhThu": doanh_thu})


# ========================================================================
# BUOC 2 — Ma tran tuong quan + bang VIF
# ========================================================================
def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    X_const = X.copy()
    X_const.insert(0, "const", 1.0)
    vif_data = []
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X_const.values, i)
        vif_data.append({"Bien": col, "VIF": round(vif, 3)})
    return pd.DataFrame(vif_data)


# ========================================================================
# BUOC 4 — Thi nghiem on dinh he so: bootstrap 100 lan, 80% du lieu/lan
# So sanh do dao dong he so cua Linear vs Ridge
# ========================================================================
def bootstrap_coef_stability(X: pd.DataFrame, y: pd.Series, ridge_alpha: float,
                              n_boot: int = 100, frac: float = 0.8):
    n = len(X)
    sample_size = int(n * frac)
    cols = X.columns.tolist()

    linear_coefs = np.zeros((n_boot, len(cols)))
    ridge_coefs = np.zeros((n_boot, len(cols)))

    for b in range(n_boot):
        idx = rng.choice(n, size=sample_size, replace=False)
        Xb, yb = X.iloc[idx], y.iloc[idx]

        scaler = StandardScaler().fit(Xb)
        Xb_scaled = scaler.transform(Xb)

        lin = LinearRegression().fit(Xb_scaled, yb)
        linear_coefs[b, :] = lin.coef_

        rid = Ridge(alpha=ridge_alpha).fit(Xb_scaled, yb)
        ridge_coefs[b, :] = rid.coef_

    return (
        pd.DataFrame(linear_coefs, columns=cols),
        pd.DataFrame(ridge_coefs, columns=cols),
    )


def plot_bootstrap_stability(linear_df: pd.DataFrame, ridge_df: pd.DataFrame, out_path: Path):
    cols = linear_df.columns.tolist()
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5), sharey=False)
    if len(cols) == 1:
        axes = [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        data_lin, data_rid = linear_df[col].values, ridge_df[col].values
        ax.boxplot(
            [data_lin, data_rid],
            tick_labels=["Linear", "Ridge"],
            showmeans=True,
        )
        ax.set_title(f"He so: {col}")
        # Zoom vao vung du lieu thuc te (khong ep truc y ve 0) de thay ro do co hep
        lo = min(data_lin.min(), data_rid.min())
        hi = max(data_lin.max(), data_rid.max())
        pad = (hi - lo) * 0.15 if hi > lo else 1.0
        ax.set_ylim(lo - pad, hi + pad)
        if lo - pad <= 0 <= hi + pad:
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Gia tri he so (bootstrap)")

    fig.suptitle("Do on dinh he so qua 100 lan bootstrap: Linear vs Ridge", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # In do lech chuan de so sanh dinh luong
    print("\n--- Do lech chuan he so qua bootstrap (thap hon = on dinh hon) ---")
    summary = pd.DataFrame({
        "Bien": cols,
        "Std_Linear": [linear_df[c].std() for c in cols],
        "Std_Ridge": [ridge_df[c].std() for c in cols],
    })
    summary["Giam_dao_dong_%"] = (
        (summary["Std_Linear"] - summary["Std_Ridge"]) / summary["Std_Linear"] * 100
    ).round(1)
    print(summary.to_string(index=False))
    return summary


# ========================================================================
# BUOC 6 — Coefficient path: he so co dan theo alpha
# ========================================================================
def plot_coefficient_path(X_scaled: np.ndarray, y: pd.Series, cols: list, out_path: Path):
    alphas = np.logspace(-3, 4, 100)
    paths = np.array([Ridge(alpha=a).fit(X_scaled, y).coef_ for a in alphas])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, col in enumerate(cols):
        ax.plot(alphas, paths[:, i], label=col, linewidth=2)
    ax.set_xscale("log")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("alpha (lambda) — thang log")
    ax.set_ylabel("Gia tri he so")
    ax.set_title("Duong co he so Ridge (Coefficient Path)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ========================================================================
# BUOC 7 — RMSE train/test theo alpha
# ========================================================================
def plot_rmse_vs_alpha(X_train_s, y_train, X_test_s, y_test, out_path: Path):
    alphas = np.logspace(-3, 4, 100)
    rmse_train, rmse_test = [], []
    for a in alphas:
        model = Ridge(alpha=a).fit(X_train_s, y_train)
        rmse_train.append(np.sqrt(mean_squared_error(y_train, model.predict(X_train_s))))
        rmse_test.append(np.sqrt(mean_squared_error(y_test, model.predict(X_test_s))))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(alphas, rmse_train, label="RMSE train", linewidth=2)
    ax.plot(alphas, rmse_test, label="RMSE test", linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("alpha (lambda) — thang log")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE theo alpha (bias-variance tradeoff)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    best_idx = int(np.argmin(rmse_test))
    return alphas[best_idx], rmse_test[best_idx]


# ========================================================================
# MAIN PIPELINE
# ========================================================================
def main():
    print("=" * 70)
    print("TT-12 — RIDGE REGRESSION: PHAN BO NGAN SACH QUANG CAO DA KENH")
    print("=" * 70)

    # ---- 1a. Doi chieu voi du lieu that ----
    real_df = load_real_advertising()
    real_corr = real_df.drop(columns=["Sales"]).corr()
    print("\n[1] Bo du lieu that (advertising.csv) — ma tran tuong quan cac kenh:")
    print(real_corr.round(3).to_string())
    print(
        "-> Tuong quan gan 0 giua cac kenh trong bo nay: KHONG the hien da cong tuyen ro ret.\n"
        "   Vi vay bai nay sinh them du lieu mo phong co da cong tuyen chu dich (theo muc 3 README)\n"
        "   de minh hoa dung tac dung cua Ridge."
    )
    real_corr.round(3).to_csv(REPORTS_DIR / "real_data_correlation.csv")

    # ---- 1b. Du lieu mo phong co da cong tuyen ----
    df = make_synthetic_multicollinear(n=500)
    feature_cols = ["TV", "Facebook", "Google"]
    target_col = "DoanhThu"
    X = df[feature_cols]
    y = df[target_col]

    corr = X.corr()
    print("\n[2] Ma tran tuong quan (du lieu mo phong):")
    print(corr.round(3).to_string())
    corr.round(3).to_csv(REPORTS_DIR / "synthetic_correlation.csv")

    vif_table = compute_vif(X)
    print("\n[2] Bang VIF (Variance Inflation Factor):")
    print(vif_table.to_string(index=False))
    vif_table.to_csv(REPORTS_DIR / "vif_table.csv", index=False)
    if (vif_table["VIF"] > 10).any():
        print("-> Co bien VIF > 10 => XAC NHAN da cong tuyen nghiem trong.")
    else:
        print("-> VIF chua vuot 10, nhung van co tuong quan dang ke giua TV va Facebook.")

    # ---- Chia train/test TRUOC khi chon alpha (tranh ro ri du lieu) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ---- 3. Linear Regression baseline (tren du lieu da chuan hoa) ----
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    linear_model = LinearRegression().fit(X_train_s, y_train)
    linear_rmse_test = np.sqrt(mean_squared_error(y_test, linear_model.predict(X_test_s)))
    print("\n[3] Linear Regression baseline — he so (da chuan hoa):")
    for col, coef in zip(feature_cols, linear_model.coef_):
        print(f"    {col}: {coef:.3f}")
    print(f"    RMSE test: {linear_rmse_test:.3f}")

    # ---- 4. Bootstrap: on dinh he so Linear vs Ridge ----
    print("\n[4] Chay bootstrap 100 lan (80% du lieu moi lan)...")
    # dung alpha "vua phai" cho buoc minh hoa on dinh (khong phai alpha toi uu cuoi cung)
    demo_alpha = 10.0
    linear_boot, ridge_boot = bootstrap_coef_stability(X, y, ridge_alpha=demo_alpha)
    stability_summary = plot_bootstrap_stability(
        linear_boot, ridge_boot, REPORTS_DIR / "bootstrap_he_so.png"
    )
    stability_summary.to_csv(REPORTS_DIR / "bootstrap_stability_summary.csv", index=False)

    # ---- 5. RidgeCV do alpha toi uu bang cross-validation (chi tren train) ----
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)),
    ])
    pipe.fit(X_train, y_train)
    best_alpha_cv = pipe["ridge"].alpha_
    print(f"\n[5] RidgeCV (5-fold CV, chi tren train) -> alpha toi uu: {best_alpha_cv:.4f}")

    # ---- 6. Coefficient path ----
    plot_coefficient_path(X_train_s, y_train, feature_cols, REPORTS_DIR / "coefficient_path.png")
    print("[6] Da luu coefficient_path.png")

    # ---- 7. RMSE train/test theo alpha ----
    best_alpha_curve, best_rmse_curve = plot_rmse_vs_alpha(
        X_train_s, y_train, X_test_s, y_test, REPORTS_DIR / "rmse_theo_alpha.png"
    )
    print(f"[7] Da luu rmse_theo_alpha.png (alpha co RMSE-test thap nhat tren duong quet: {best_alpha_curve:.4f})")

    # ---- 8. So sanh RMSE test: Linear vs Ridge (dung alpha tu RidgeCV) ----
    final_ridge = Ridge(alpha=best_alpha_cv).fit(X_train_s, y_train)
    ridge_rmse_test = np.sqrt(mean_squared_error(y_test, final_ridge.predict(X_test_s)))
    print("\n[8] So sanh RMSE tren tap test:")
    print(f"    Linear Regression : {linear_rmse_test:.3f}")
    print(f"    Ridge (alpha={best_alpha_cv:.3f})  : {ridge_rmse_test:.3f}")

    # ---- 10. De xuat phan bo ngan sach dua tren he so Ridge ----
    ridge_coefs = dict(zip(feature_cols, final_ridge.coef_))
    total_effect = sum(abs(v) for v in ridge_coefs.values())
    allocation = {k: round(abs(v) / total_effect * 100, 1) for k, v in ridge_coefs.items()}

    print("\n[10] He so Ridge (da chuan hoa, sau khi chon alpha bang CV):")
    for col in feature_cols:
        print(f"    {col}: {ridge_coefs[col]:.3f}  -> ty le phan bo de xuat: {allocation[col]}%")

    budget_total_vnd = 2_000_000_000  # gia dinh 2 ty/thang nhu vi du trong README
    print(f"\n    De xuat phan bo ngan sach {budget_total_vnd:,.0f} VND/thang:")
    proposal_rows = []
    for col in feature_cols:
        amount = budget_total_vnd * allocation[col] / 100
        print(f"      {col:10s}: {allocation[col]:5.1f}%  ~ {amount:,.0f} VND")
        proposal_rows.append({"Kenh": col, "He_so_Ridge": round(ridge_coefs[col], 3),
                               "Ty_le_de_xuat_%": allocation[col], "So_tien_VND": round(amount)})
    pd.DataFrame(proposal_rows).to_csv(REPORTS_DIR / "de_xuat_phan_bo_ngan_sach.csv", index=False)

    # ---- Luu model pipeline (scaler + ridge, alpha tu RidgeCV, fit lai tren toan bo train) ----
    final_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha_cv)),
    ])
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, MODELS_DIR / "ridge_pipeline.joblib")
    print(f"\nDa luu model: {MODELS_DIR / 'ridge_pipeline.joblib'}")

    # ---- Luu tom tat ket qua dang JSON de tra cuu nhanh ----
    summary = {
        "best_alpha_ridgecv": float(best_alpha_cv),
        "linear_rmse_test": float(linear_rmse_test),
        "ridge_rmse_test": float(ridge_rmse_test),
        "vif_max": float(vif_table["VIF"].max()),
        "ridge_coefficients": {k: float(v) for k, v in ridge_coefs.items()},
        "budget_allocation_percent": allocation,
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("HOAN TAT. Xem ket qua trong thu muc reports/ va models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
