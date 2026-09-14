"""
train.py
--------
End-to-end training pipeline for the TT-18 Automated Valuation Model (AVM):

  1. Load + report missing values (thieu that vs khong co tien ich)
  2. Clean missing values + ordinal/one-hot encode
  3. Baselines: Dummy / Linear / Ridge (on log1p(SalePrice))
  4. Gradient Boosting Regressor with early stopping
  5. Train/validation loss curve vs. number of trees
  6. Feature-engineering ablation (before/after TotalSF, TuoiNha, DaSuaChua)
  7. Quantile regression (p10/p50/p90) -> price interval
  8. Median APE (the AVM industry metric) + APE distribution
  9. Human-in-the-loop rule: interval width > 25% of the point estimate ->
     route to a human appraiser instead of auto-approving
  10. GradientBoostingRegressor vs HistGradientBoostingRegressor timing
  11. Persist the final pipeline to models/gbr_pipeline.joblib

Usage:
    python src/train.py --data data/train.csv --out_dir .
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import features as feat


def median_ape(y_true, y_pred) -> float:
    """Median Absolute Percentage Error, in %. The standard AVM accuracy metric."""
    ape = np.abs((np.asarray(y_true) - np.asarray(y_pred)) / np.asarray(y_true)) * 100
    return float(np.median(ape))


def rmse_log(y_true_log, y_pred_log) -> float:
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def run(data_path: str, out_dir: str, random_state: int = 42):
    out_dir = Path(out_dir)
    reports_dir = out_dir / "reports"
    models_dir = out_dir / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Load + missing value report ----------
    df_raw = feat.load_data(data_path)
    report = feat.missing_value_report(df_raw)
    print("== Bao cao gia tri thieu ==")
    print(report.to_string(index=False))

    if len(report):
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = ["#d9534f" if "khong_co_tien_ich" in c else "#5bc0de" for c in report["phan_loai"]]
        ax.barh(report["cot"], report["so_dong_thieu"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("So dong thieu")
        ax.set_title("Phan tich gia tri thieu: do (khong co tien ich) vs xanh (thieu that)")
        fig.tight_layout()
        fig.savefig(reports_dir / "missing_analysis.png", dpi=130)
        plt.close(fig)

    n_none_cols = (report["phan_loai"].str.contains("khong_co_tien_ich")).sum() if len(report) else 0
    n_true_missing_cols = len(report) - n_none_cols
    print(f"\n-> {n_none_cols} cot 'khong co tien ich' (NaN co y nghia), "
          f"{n_true_missing_cols} cot 'thieu that' (se dien median/mode).\n")

    # ---------- 2. Clean + engineer ----------
    df_clean = feat.handle_missing(df_raw)
    df_fe = feat.engineer_features(df_clean)

    y = df_fe[feat.TARGET].astype(float)
    y_log = np.log1p(y)

    X_train_df, X_val_df, y_train, y_val, y_train_log, y_val_log = train_test_split(
        df_fe.drop(columns=[feat.TARGET]), y, y_log,
        test_size=0.2, random_state=random_state,
    )

    preprocessor, cols = feat.build_preprocessor(df_fe)
    n_ordinal = len(cols[1])
    print(f"-> {n_ordinal} cot duoc ma hoa THU TU (OrdinalEncoder) dung thu tu chat luong.")

    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)

    # ---------- 3. Baselines ----------
    print("\n== Baseline (RMSE tren thang log) ==")
    baseline_results = {}
    for name, model in [
        ("DummyRegressor", DummyRegressor(strategy="mean")),
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=10.0)),
    ]:
        model.fit(X_train, y_train_log)
        pred = model.predict(X_val)
        score = rmse_log(y_val_log, pred)
        baseline_results[name] = score
        print(f"  {name:20s} RMSE(log) = {score:.4f}")

    # ---------- 4. Gradient Boosting + early stopping ----------
    gbr = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        subsample=0.8, max_features="sqrt",
        validation_fraction=0.1, n_iter_no_change=50, tol=1e-4,
        random_state=random_state,
    )
    gbr.fit(X_train, y_train_log)
    n_trees_used = gbr.n_estimators_
    pred_log = gbr.predict(X_val)
    gbr_rmse_log = rmse_log(y_val_log, pred_log)
    print(f"\n== Gradient Boosting == RMSE(log) = {gbr_rmse_log:.4f} "
          f"(dung sau {n_trees_used} cay nho early stopping)")

    # ---------- 5. Train/validation loss curve ----------
    # Re-fit an identical model but WITHOUT early stopping so we can plot the
    # full train-loss curve, and score validation loss at every stage for comparison.
    gbr_curve = GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.03, max_depth=3,
        subsample=0.8, max_features="sqrt", random_state=random_state,
    )
    gbr_curve.fit(X_train, y_train_log)
    train_loss = gbr_curve.train_score_
    val_loss = np.array([
        mean_squared_error(y_val_log, pred_stage)
        for pred_stage in gbr_curve.staged_predict(X_val)
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_loss, label="Train loss (deviance)")
    ax.plot(val_loss, label="Validation loss (MSE, thang log)")
    ax.axvline(n_trees_used, color="gray", linestyle="--", label=f"Early stop @ {n_trees_used} cay")
    ax.set_xlabel("So cay (boosting stage)")
    ax.set_ylabel("Loss")
    ax.set_title("Train vs Validation loss theo so cay -> diem overfit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "loss_theo_so_cay.png", dpi=130)
    plt.close(fig)

    # ---------- 6. Feature engineering ablation ----------
    fe_cols_added = [c for c in ["TotalSF", "TuoiNha", "DaSuaChua"] if c in df_fe.columns]
    df_no_fe = df_clean.drop(columns=[c for c in fe_cols_added if c in df_clean.columns], errors="ignore")
    pre_no_fe, _ = feat.build_preprocessor(df_no_fe)
    Xtr_no_fe_df = X_train_df.drop(columns=fe_cols_added, errors="ignore")
    Xval_no_fe_df = X_val_df.drop(columns=fe_cols_added, errors="ignore")
    Xtr_no_fe = pre_no_fe.fit_transform(Xtr_no_fe_df)
    Xval_no_fe = pre_no_fe.transform(Xval_no_fe_df)
    gbr_no_fe = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.8, max_features="sqrt", random_state=random_state,
    )
    gbr_no_fe.fit(Xtr_no_fe, y_train_log)
    rmse_no_fe = rmse_log(y_val_log, gbr_no_fe.predict(Xval_no_fe))

    gbr_with_fe_quick = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.8, max_features="sqrt", random_state=random_state,
    )
    gbr_with_fe_quick.fit(X_train, y_train_log)
    rmse_with_fe = rmse_log(y_val_log, gbr_with_fe_quick.predict(X_val))
    fe_improvement_pct = 100 * (rmse_no_fe - rmse_with_fe) / rmse_no_fe
    print(f"\n== Feature engineering ablation ==")
    print(f"  RMSE(log) truoc khi them {fe_cols_added}: {rmse_no_fe:.4f}")
    print(f"  RMSE(log) sau khi them:                    {rmse_with_fe:.4f}")
    print(f"  Cai thien: {fe_improvement_pct:.1f}%")

    # ---------- 7. Quantile regression (10/50/90) ----------
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        qm = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=500, learning_rate=0.05,
            max_depth=3, subsample=0.8, random_state=random_state,
        )
        qm.fit(X_train, y_train_log)
        quantile_models[q] = qm

    pred_p10 = np.expm1(quantile_models[0.1].predict(X_val))
    pred_p50 = np.expm1(quantile_models[0.5].predict(X_val))
    pred_p90 = np.expm1(quantile_models[0.9].predict(X_val))
    pred_p10, pred_p90 = np.minimum(pred_p10, pred_p90), np.maximum(pred_p10, pred_p90)  # guard against crossing

    coverage = float(np.mean((y_val.values >= pred_p10) & (y_val.values <= pred_p90)) * 100)
    print(f"\n== Khoang gia 10-90% == Ty le phu thuc te (coverage): {coverage:.1f}% (muc tieu ~80%)")

    fig, ax = plt.subplots(figsize=(9, 6))
    order = np.argsort(y_val.values)
    idx = np.array(order)[:80]  # first 80 for readability
    x_axis = np.arange(len(idx))
    ax.fill_between(x_axis, pred_p10[idx], pred_p90[idx], color="#5bc0de", alpha=0.3, label="Khoang du bao 10-90%")
    ax.plot(x_axis, pred_p50[idx], color="#0275d8", label="Du doan p50")
    ax.scatter(x_axis, y_val.values[idx], color="#d9534f", s=12, label="Gia that", zorder=5)
    ax.set_xlabel("Can nha (sap xep theo gia that, 80 mau dau)")
    ax.set_ylabel("SalePrice")
    ax.set_title(f"Khoang gia du bao 10-90% (coverage thuc te: {coverage:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "khoang_gia.png", dpi=130)
    plt.close(fig)

    # ---------- 8. Median APE + distribution ----------
    m_ape = median_ape(y_val.values, pred_p50)
    ape_all = np.abs((y_val.values - pred_p50) / y_val.values) * 100
    print(f"\n== Median APE == {m_ape:.2f}% (yeu cau nghiep vu: < 10%, tieu chi hoan thanh: < 12%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ape_all, bins=30, color="#5cb85c", edgecolor="white")
    ax.axvline(m_ape, color="#d9534f", linestyle="--", label=f"Median APE = {m_ape:.1f}%")
    ax.set_xlabel("Absolute Percentage Error (%)")
    ax.set_ylabel("So can nha")
    ax.set_title("Phan phoi APE tren tap validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "ape_distribution.png", dpi=130)
    plt.close(fig)

    # ---------- 9. Human-in-the-loop ----------
    interval_width_pct = (pred_p90 - pred_p10) / pred_p50 * 100
    needs_human = interval_width_pct > 25
    pct_auto = float(np.mean(~needs_human) * 100)
    pct_human = 100 - pct_auto
    ape_auto = median_ape(y_val.values[~needs_human], pred_p50[~needs_human]) if (~needs_human).any() else float("nan")
    ape_human = median_ape(y_val.values[needs_human], pred_p50[needs_human]) if needs_human.any() else float("nan")
    hitl_table = pd.DataFrame([
        {"nhom": "Tu dong duyet (khoang <=25%)", "ty_le_%": round(pct_auto, 1), "so_ho_so": int((~needs_human).sum()), "median_ape_%": round(ape_auto, 2)},
        {"nhom": "Chuyen tham dinh vien (khoang >25%)", "ty_le_%": round(pct_human, 1), "so_ho_so": int(needs_human.sum()), "median_ape_%": round(ape_human, 2)},
    ])
    print("\n== Human-in-the-loop ==")
    print(hitl_table.to_string(index=False))
    hitl_table.to_csv(reports_dir / "human_in_the_loop.csv", index=False)

    # ---------- 10. GBR vs HistGBR timing ----------
    t0 = time.time()
    gbr_timing = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=random_state)
    gbr_timing.fit(X_train, y_train_log)
    t_gbr = time.time() - t0

    t0 = time.time()
    hgbr = HistGradientBoostingRegressor(max_iter=300, max_depth=3, learning_rate=0.05, random_state=random_state)
    hgbr.fit(X_train, y_train_log)
    t_hgbr = time.time() - t0
    hgbr_rmse = rmse_log(y_val_log, hgbr.predict(X_val))
    print(f"\n== Thoi gian train (300 cay/iter) ==")
    print(f"  GradientBoostingRegressor:     {t_gbr:.2f}s | RMSE(log)={rmse_log(y_val_log, gbr_timing.predict(X_val)):.4f}")
    print(f"  HistGradientBoostingRegressor: {t_hgbr:.2f}s | RMSE(log)={hgbr_rmse:.4f}")

    # ---------- 11. Persist final pipeline ----------
    full_pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", gbr),
    ])
    joblib.dump(full_pipeline, models_dir / "gbr_pipeline.joblib")
    joblib.dump(quantile_models, models_dir / "gbr_quantile_models.joblib")
    print(f"\nDa luu model vao {models_dir / 'gbr_pipeline.joblib'}")

    summary = {
        "median_ape_%": round(m_ape, 2),
        "coverage_p10_p90_%": round(coverage, 1),
        "pct_tu_dong_%": round(pct_auto, 1),
        "gbr_rmse_log": round(gbr_rmse_log, 4),
        "n_trees_early_stop": int(n_trees_used),
        "fe_improvement_%": round(fe_improvement_pct, 1),
        "time_gbr_s": round(t_gbr, 2),
        "time_hgbr_s": round(t_hgbr, 2),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()
    result = run(args.data, args.out_dir)
    print("\n=== TOM TAT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
