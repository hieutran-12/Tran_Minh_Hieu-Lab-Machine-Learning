"""
TT-11 - Linear Regression - Dinh gia nha o (California Housing)
================================================================
Script huan luyen day du: EDA co ban, kiem tra gia dinh hoi quy tuyen tinh,
huan luyen Linear Regression, kiem tra da cong tuyen (VIF), thu log-transform,
feature engineering, va so sanh voi Ridge / Random Forest.

Chay: python src/train.py
Ket qua: models/lr_pipeline.joblib, cac bieu do trong reports/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# ------------------------------------------------------------------
# Cau hinh duong dan
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mp = mape(y_true, y_pred)
    print(f"[{name}] RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  MAPE={mp:.2f}%")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mp}


def main():
    # ----------------------------------------------------------------
    # 1. NAP DU LIEU DUNG YEU CAU DE BAI (California Housing, KHONG dung Boston)
    # ----------------------------------------------------------------
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    print("Kich thuoc du lieu:", df.shape)
    print(df.describe())

    # ----------------------------------------------------------------
    # 2. PHAT HIEN NHAN BI CAT NGON O 5.0
    # ----------------------------------------------------------------
    n_capped = (df["MedHouseVal"] >= 5.0).sum()
    print(f"\nSo dong bi cat ngon nhan tai 5.0: {n_capped} "
          f"({n_capped / len(df) * 100:.2f}% du lieu)")

    # ----------------------------------------------------------------
    # 3. EDA
    # ----------------------------------------------------------------
    # 3.1 Scatter MedInc vs gia
    plt.figure(figsize=(7, 5))
    plt.scatter(df["MedInc"], df["MedHouseVal"], s=5, alpha=0.3)
    plt.xlabel("MedInc (thu nhap trung vi)")
    plt.ylabel("MedHouseVal (gia, don vi 100k USD)")
    plt.title("MedInc vs Gia nha")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "scatter_medinc.png"), dpi=120)
    plt.close()

    # 3.2 Heatmap tuong quan
    plt.figure(figsize=(9, 7))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Ma tran tuong quan")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "heatmap_correlation.png"), dpi=120)
    plt.close()

    # 3.3 Ban do gia theo toa do
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(df["Longitude"], df["Latitude"], c=df["MedHouseVal"],
                      cmap="viridis", s=8, alpha=0.5)
    plt.colorbar(sc, label="Gia (100k USD)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ban do gia nha theo toa do")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "ban_do_gia.png"), dpi=120)
    plt.close()

    # ----------------------------------------------------------------
    # Xu ly outlier cuc doan (AveRooms, AveOccup) - clip theo phan vi 99
    # ----------------------------------------------------------------
    df_clean = df.copy()
    for col in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
        upper = df_clean[col].quantile(0.99)
        df_clean[col] = df_clean[col].clip(upper=upper)

    # ----------------------------------------------------------------
    # Feature engineering: rooms_per_household, khoang cach toi SF / LA
    # ----------------------------------------------------------------
    df_clean["rooms_per_household"] = df_clean["AveRooms"] / df_clean["AveOccup"].replace(0, np.nan)
    df_clean["rooms_per_household"] = df_clean["rooms_per_household"].fillna(df_clean["rooms_per_household"].median())

    sf_lat, sf_lon = 37.7749, -122.4194
    la_lat, la_lon = 34.0522, -118.2437
    df_clean["dist_to_SF"] = np.sqrt((df_clean["Latitude"] - sf_lat) ** 2 + (df_clean["Longitude"] - sf_lon) ** 2)
    df_clean["dist_to_LA"] = np.sqrt((df_clean["Latitude"] - la_lat) ** 2 + (df_clean["Longitude"] - la_lon) ** 2)
    df_clean["dist_to_nearest_city"] = df_clean[["dist_to_SF", "dist_to_LA"]].min(axis=1)

    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
                     "AveOccup", "Latitude", "Longitude",
                     "rooms_per_household", "dist_to_nearest_city"]
    X = df_clean[feature_cols]
    y = df_clean["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------------
    # 4. BASELINE
    # ----------------------------------------------------------------
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    results = [print_metrics("Baseline (mean)", y_test, y_pred_dummy)]

    # ----------------------------------------------------------------
    # 5. LINEAR REGRESSION CO BAN
    # ----------------------------------------------------------------
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    y_pred_lr = pipe.predict(X_test)
    results.append(print_metrics("Linear Regression", y_test, y_pred_lr))

    # ----------------------------------------------------------------
    # 6. RESIDUAL PLOT
    # ----------------------------------------------------------------
    residuals = y_test - y_pred_lr
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred_lr, residuals, s=5, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Gia du doan")
    plt.ylabel("Phan du (residual)")
    plt.title("Residual Plot - Linear Regression")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "residual_plot.png"), dpi=120)
    plt.close()

    # ----------------------------------------------------------------
    # 7. Q-Q PLOT
    # ----------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot - Phan du")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "qq_plot.png"), dpi=120)
    plt.close()

    # ----------------------------------------------------------------
    # 8. THU DU DOAN LOG(GIA)
    # ----------------------------------------------------------------
    y_train_log = np.log1p(y_train.clip(lower=0))
    y_test_log = np.log1p(y_test.clip(lower=0))

    pipe_log = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    pipe_log.fit(X_train, y_train_log)
    y_pred_log = pipe_log.predict(X_test)
    y_pred_log_back = np.expm1(y_pred_log)
    results.append(print_metrics("Linear Regression (log target)", y_test, y_pred_log_back))

    residuals_log = y_test_log - y_pred_log
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred_log, residuals_log, s=5, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("log(gia) du doan")
    plt.ylabel("Phan du")
    plt.title("Residual Plot - Log Target")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "residual_plot_log.png"), dpi=120)
    plt.close()

    # ----------------------------------------------------------------
    # 9. KIEM TRA DA CONG TUYEN (VIF)
    # ----------------------------------------------------------------
    X_vif = X_train.copy()
    X_vif = (X_vif - X_vif.mean()) / X_vif.std()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print("\nBang VIF:\n", vif_data)
    vif_data.to_csv(os.path.join(REPORTS_DIR, "vif_table.csv"), index=False)

    # ----------------------------------------------------------------
    # 11. BANG HE SO DA CHUAN HOA
    # ----------------------------------------------------------------
    he_so = pd.Series(pipe["lr"].coef_, index=X.columns).sort_values(key=abs, ascending=False)
    print("\nHe so da chuan hoa (sap xep theo do lon):\n", he_so)
    he_so.to_csv(os.path.join(REPORTS_DIR, "he_so.csv"))

    plt.figure(figsize=(8, 6))
    he_so.plot(kind="barh", color=["#d62728" if v < 0 else "#2ca02c" for v in he_so])
    plt.xlabel("He so (sau chuan hoa)")
    plt.title("Muc do anh huong cua tung dac trung")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "he_so.png"), dpi=120)
    plt.close()

    # ----------------------------------------------------------------
    # 12. SO SANH VOI RIDGE VA RANDOM FOREST
    # ----------------------------------------------------------------
    ridge = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results.append(print_metrics("Ridge", y_test, y_pred_ridge))

    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append(print_metrics("Random Forest", y_test, y_pred_rf))

    results_df = pd.DataFrame(results)
    print("\nBang so sanh cac model:\n", results_df)
    results_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)

    # ----------------------------------------------------------------
    # Luu model
    # ----------------------------------------------------------------
    joblib.dump(pipe, os.path.join(MODELS_DIR, "lr_pipeline.joblib"))
    print(f"\nDa luu model tai: {os.path.join(MODELS_DIR, 'lr_pipeline.joblib')}")
    print(f"Cac bieu do da luu tai: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
