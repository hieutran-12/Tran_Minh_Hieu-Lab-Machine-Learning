"""
TT-22 — MLP Regressor: Du doan muc tieu hao nhien lieu (Auto MPG)
==================================================================
End-to-end pipeline: tai du lieu -> lam sach -> EDA -> baseline ->
MLPRegressor (khao sat scaling / kien truc / alpha / activation) ->
bang so sanh cuoi cung -> quy doi L/100km.

Chay:
    python src/train.py
    python src/train.py --data-path duong/dan/auto-mpg.data   # neu da tai san

Dau ra:
    models/mlp_reg.joblib
    reports/scale_comparison.png, kien_truc_overfit.png,
            loss_curve.png, alpha_sweep.png
    reports/summary.json   (toan bo so lieu do duoc, dung de dien vao README)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
COLUMN_NAMES = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin",
]
ORIGIN_MAP = {1: "usa", 2: "europe", 3: "japan"}


# --------------------------------------------------------------------------- #
# 1. Du lieu
# --------------------------------------------------------------------------- #
def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Tai bo Auto MPG (UCI). Neu khong co --data-path, tai truc tiep tu UCI.

    File goc auto-mpg.data la dang whitespace-separated; ten xe nam sau
    ky tu tab va co the chua khoang trang nen ta cat bo bang comment='\\t'
    (cung khong can car_name theo yeu cau bai).
    """
    source = data_path if data_path else DATA_URL
    # File goc co 9 truong (8 so lieu + ten xe, ten xe co the chua nhieu tu
    # sau dau tab) -> chi lay 8 token dau moi dong, bo han ten xe.
    if str(source).startswith("http"):
        import urllib.request
        with urllib.request.urlopen(source) as resp:
            lines = resp.read().decode("utf-8").splitlines()
    else:
        with open(source, encoding="utf-8") as f:
            lines = f.read().splitlines()

    rows = [line.split()[:8] for line in lines if line.strip()]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    for col in COLUMN_NAMES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # (1) horsepower: 6 gia tri '?' -> NaN -> dien median
    n_missing = int(df["horsepower"].isna().sum())
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    print(f"[clean] So gia tri '?' o horsepower da dien median: {n_missing}")

    # (2) origin la bien phan loai (1=My, 2=Chau Au, 3=Nhat) -> one-hot
    df["origin"] = df["origin"].astype(int).map(ORIGIN_MAP)
    df = pd.get_dummies(df, columns=["origin"], prefix="origin")

    return df


def eda_plots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(df["weight"], df["mpg"], alpha=0.5, s=18)
    axes[0].set_xlabel("weight (lbs)")
    axes[0].set_ylabel("mpg")
    axes[0].set_title("weight vs mpg (quan he nghich, hoi cong)")

    grp = df.groupby("model_year")["mpg"].mean()
    axes[1].plot(grp.index, grp.values, marker="o")
    axes[1].set_xlabel("model_year")
    axes[1].set_ylabel("mpg trung binh")
    axes[1].set_title("mpg trung binh theo model_year")

    fig.tight_layout()
    fig.savefig(REPORTS / "eda_overview.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 2. Tien ich danh gia
# --------------------------------------------------------------------------- #
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(model, X_train, y_train, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    return {
        "rmse_train": rmse(y_train, pred_train),
        "rmse_test": rmse(y_test, pred_test),
        "r2_test": float(r2_score(y_test, pred_test)),
        "train_time_s": train_time,
    }


# --------------------------------------------------------------------------- #
# 3. Cac buoc chinh cua bai
# --------------------------------------------------------------------------- #
def run_baselines(X_train, y_train, X_test, y_test) -> dict:
    results = {}
    results["Dummy (mean)"] = evaluate(
        DummyRegressor(strategy="mean"), X_train, y_train, X_test, y_test
    )
    results["Linear Regression"] = evaluate(
        LinearRegression(), X_train, y_train, X_test, y_test
    )
    results["Random Forest"] = evaluate(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    )
    return results


def make_mlp(hidden_layer_sizes=(64, 32), alpha=1e-2, activation="relu"):
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=1e-3,
        max_iter=2000,
        early_stopping=True,
        n_iter_no_change=30,
        validation_fraction=0.15,
        random_state=RANDOM_STATE,
    )


def scale_comparison(X_train, y_train, X_test, y_test) -> dict:
    """Buoc 5-6: MLP khong scale / scale X / scale ca X va y."""
    results = {}

    # (a) Khong scale gi ca -> thuong hoi tu kem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results["Khong scale"] = evaluate(
            make_mlp(), X_train, y_train, X_test, y_test
        )

    # (b) Chi scale X
    pipe_x = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp())])
    results["Chi scale X"] = evaluate(pipe_x, X_train, y_train, X_test, y_test)

    # (c) Scale ca X va y (TransformedTargetRegressor)
    net = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp())])
    model_xy = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
    results["Scale ca X va y"] = evaluate(model_xy, X_train, y_train, X_test, y_test)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    names = list(results.keys())
    rmses = [results[n]["rmse_test"] for n in names]
    bars = ax.bar(names, rmses, color=["#c0392b", "#e67e22", "#27ae60"])
    ax.set_ylabel("RMSE (test)")
    ax.set_title("So sanh scaling cho MLPRegressor")
    ax.bar_label(bars, fmt="%.2f")
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(REPORTS / "scale_comparison.png", dpi=130)
    plt.close(fig)

    return results, model_xy


def architecture_comparison(X_train, y_train, X_test, y_test) -> dict:
    """Buoc 7: 4 kien truc, RMSE / so tham so / dau hieu overfit."""
    architectures = {"(16)": (16,), "(64)": (64,), "(64,32)": (64, 32),
                      "(256,128,64)": (256, 128, 64)}
    n_features = X_train.shape[1]
    results = {}

    for name, hidden in architectures.items():
        net = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp(hidden_layer_sizes=hidden))])
        model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            metrics = evaluate(model, X_train, y_train, X_test, y_test)

        # dem so tham so: (n_in+1)*h1 + (h1+1)*h2 + ... + (h_last+1)*1
        sizes = [n_features, *hidden, 1]
        n_params = sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1))
        metrics["n_params"] = n_params
        metrics["overfit_gap"] = metrics["rmse_test"] - metrics["rmse_train"]
        results[name] = metrics

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(architectures))
    width = 0.35
    train_rmse = [results[n]["rmse_train"] for n in architectures]
    test_rmse = [results[n]["rmse_test"] for n in architectures]
    ax.bar(x - width / 2, train_rmse, width, label="RMSE train")
    ax.bar(x + width / 2, test_rmse, width, label="RMSE test")
    ax.set_xticks(x)
    ax.set_xticklabels(architectures.keys())
    ax.set_ylabel("RMSE")
    ax.set_title("Kien truc mang vs overfit (398 mau)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "kien_truc_overfit.png", dpi=130)
    plt.close(fig)

    return results


def plot_loss_curve(fitted_ttr: TransformedTargetRegressor) -> None:
    """Buoc 8: ve loss_curve_ va validation_scores_ cua model da fit."""
    mlp: MLPRegressor = fitted_ttr.regressor_.named_steps["mlp"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(mlp.loss_curve_)
    axes[0].set_title("loss_curve_ (training loss)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")

    if hasattr(mlp, "validation_scores_") and mlp.validation_scores_ is not None:
        axes[1].plot(mlp.validation_scores_)
        axes[1].set_title("validation_scores_ (early stopping)")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("R^2 (validation)")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "khong co validation_scores_", ha="center")

    fig.tight_layout()
    fig.savefig(REPORTS / "loss_curve.png", dpi=130)
    plt.close(fig)


def alpha_sweep(X_train, y_train, X_test, y_test) -> dict:
    """Buoc 9: do alpha in {1e-4, 1e-3, 1e-2, 1e-1}."""
    alphas = [1e-4, 1e-3, 1e-2, 1e-1]
    results = {}
    for a in alphas:
        net = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp(alpha=a))])
        model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            results[a] = evaluate(model, X_train, y_train, X_test, y_test)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(alphas, [results[a]["rmse_train"] for a in alphas], "o-", label="RMSE train")
    ax.plot(alphas, [results[a]["rmse_test"] for a in alphas], "o-", label="RMSE test")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (regularization)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE theo alpha")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "alpha_sweep.png", dpi=130)
    plt.close(fig)

    return {str(a): v for a, v in results.items()}


def activation_comparison(X_train, y_train, X_test, y_test) -> dict:
    """Buoc 10: so sanh relu / tanh / logistic."""
    results = {}
    for act in ["relu", "tanh", "logistic"]:
        net = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp(activation=act))])
        model = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            results[act] = evaluate(model, X_train, y_train, X_test, y_test)
    return results


def final_comparison(X_train, y_train, X_test, y_test, best_mlp_config) -> dict:
    """Buoc 11: bang so sanh cuoi - Linear / RF / SVR / MLP."""
    results = {}
    results["Linear Regression"] = evaluate(
        Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())]),
        X_train, y_train, X_test, y_test,
    )
    results["Random Forest"] = evaluate(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        X_train, y_train, X_test, y_test,
    )
    svr_net = Pipeline([("scale", StandardScaler()), ("svr", SVR(C=10, epsilon=0.3))])
    results["SVR"] = evaluate(
        TransformedTargetRegressor(regressor=svr_net, transformer=StandardScaler()),
        X_train, y_train, X_test, y_test,
    )
    hidden, alpha = best_mlp_config
    net = Pipeline([("scale", StandardScaler()), ("mlp", make_mlp(hidden_layer_sizes=hidden, alpha=alpha))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        results["MLPRegressor"] = evaluate(
            TransformedTargetRegressor(regressor=net, transformer=StandardScaler()),
            X_train, y_train, X_test, y_test,
        )
    return results


def mpg_to_l100km(mpg: float) -> float:
    return 235.215 / mpg


def estimate_annual_fuel_cost(l_per_100km: float, km_per_year: float = 15000,
                               price_per_liter_vnd: float = 21000) -> float:
    liters_per_year = l_per_100km * km_per_year / 100
    return liters_per_year * price_per_liter_vnd


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main(data_path: str | None = None) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("== 1. Tai va lam sach du lieu ==")
    df = clean_data(load_data(data_path))
    print(df.shape)

    eda_plots(df)

    X = df.drop(columns=["mpg"])
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("== 2. Baseline (Dummy / Linear / RF) ==")
    baselines = run_baselines(X_train, y_train, X_test, y_test)
    for k, v in baselines.items():
        print(f"  {k:20s} RMSE_test={v['rmse_test']:.3f}  R2_test={v['r2_test']:.3f}")

    print("== 3. So sanh scaling cho MLP ==")
    scale_results, best_scaled_model = scale_comparison(X_train, y_train, X_test, y_test)
    for k, v in scale_results.items():
        print(f"  {k:20s} RMSE_test={v['rmse_test']:.3f}")

    print("== 4. So sanh kien truc (overfit voi 398 mau) ==")
    arch_results = architecture_comparison(X_train, y_train, X_test, y_test)
    for k, v in arch_results.items():
        print(f"  {k:16s} params={v['n_params']:6d}  RMSE_test={v['rmse_test']:.3f}  gap={v['overfit_gap']:.3f}")

    best_arch_name = min(arch_results, key=lambda k: arch_results[k]["rmse_test"])
    best_arch = {"(16)": (16,), "(64)": (64,), "(64,32)": (64, 32),
                 "(256,128,64)": (256, 128, 64)}[best_arch_name]
    print(f"  -> Kien truc tot nhat tren test: {best_arch_name}")

    print("== 5. Loss curve cua model tot nhat ==")
    plot_loss_curve(best_scaled_model)

    print("== 6. Do alpha ==")
    alpha_results = alpha_sweep(X_train, y_train, X_test, y_test)
    for a, v in alpha_results.items():
        print(f"  alpha={a:8s} RMSE_test={v['rmse_test']:.3f}")
    best_alpha = float(min(alpha_results, key=lambda k: alpha_results[k]["rmse_test"]))

    print("== 7. So sanh activation ==")
    act_results = activation_comparison(X_train, y_train, X_test, y_test)
    for k, v in act_results.items():
        print(f"  {k:10s} RMSE_test={v['rmse_test']:.3f}")

    print("== 8. Bang so sanh cuoi: Linear vs RF vs SVR vs MLP ==")
    final_results = final_comparison(
        X_train, y_train, X_test, y_test,
        best_mlp_config=(best_arch, best_alpha),
    )
    for k, v in final_results.items():
        print(f"  {k:18s} RMSE_test={v['rmse_test']:.3f}  R2_test={v['r2_test']:.3f}  time={v['train_time_s']:.3f}s")

    print("== 9. Huan luyen model cuoi cung va luu ==")
    final_net = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", make_mlp(hidden_layer_sizes=best_arch, alpha=best_alpha)),
    ])
    final_model = TransformedTargetRegressor(regressor=final_net, transformer=StandardScaler())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(X_train, y_train)

    import joblib
    joblib.dump(final_model, MODELS / "mlp_reg.joblib")

    print("== 10. Quy doi vi du sang L/100km ==")
    sample = X_test.iloc[[0]]
    pred_mpg = float(final_model.predict(sample)[0])
    l100 = mpg_to_l100km(pred_mpg)
    annual_cost = estimate_annual_fuel_cost(l100)
    print(f"  Du doan: {pred_mpg:.1f} mpg  ~=  {l100:.1f} L/100km")
    print(f"  Uoc tinh chi phi xang/nam (15.000 km, 21.000 VND/L): {annual_cost:,.0f} VND")

    summary = {
        "n_rows": int(df.shape[0]),
        "baselines": baselines,
        "scale_comparison": scale_results,
        "architecture_comparison": arch_results,
        "best_architecture": best_arch_name,
        "alpha_sweep": alpha_results,
        "best_alpha": best_alpha,
        "activation_comparison": act_results,
        "final_comparison": final_results,
        "sample_conversion": {
            "pred_mpg": pred_mpg, "l_per_100km": l100, "annual_cost_vnd": annual_cost,
        },
    }
    with open(REPORTS / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_r2 = final_results["MLPRegressor"]["r2_test"]
    rf_r2 = final_results["Random Forest"]["r2_test"]
    print("\n== KET LUAN ==")
    if rf_r2 >= best_r2:
        print(f"  Random Forest (R2={rf_r2:.3f}) bang hoac tot hon MLP (R2={best_r2:.3f}) tren bo du lieu"
              f" chi 398 dong. Voi du lieu bang kich thuoc nho, cay quyet dinh van la lua chon thuc te hon;"
              f" MLP chi dang can nhac khi du lieu lon hon hoac quan he thuc su phi tuyen/tron.")
    else:
        print(f"  MLP (R2={best_r2:.3f}) nhinh hon Random Forest (R2={rf_r2:.3f}) tren bo nay,"
              f" nhung chenh lech can duoc doi chieu voi do bien thien giua cac lan chay (random_state khac nhau).")

    print(f"\nDa luu: {MODELS/'mlp_reg.joblib'}")
    print(f"Da luu bao cao: {REPORTS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLPRegressor cho Auto MPG (TT-22)")
    parser.add_argument("--data-path", type=str, default=None,
                         help="Duong dan file auto-mpg.data da tai san (tuy chon)")
    args = parser.parse_args()
    main(args.data_path)
