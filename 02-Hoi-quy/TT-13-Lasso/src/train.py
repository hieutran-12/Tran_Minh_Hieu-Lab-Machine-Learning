# -*- coding: utf-8 -*-
"""
TT-13 — LASSO REGRESSION (L1)
Chọn ra 10 chỉ số xét nghiệm quan trọng nhất trong 200 chỉ số (10 thật + 190 nhiễu)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, RidgeCV
from sklearn.linear_model import lasso_path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

RNG_SEED = 42
REPORTS_DIR = "../reports"
MODELS_DIR = "../models"

# ---------------------------------------------------------------------------
# 1. NẠP DỮ LIỆU + MỞ RỘNG THÀNH 200 CỘT (10 THẬT + 190 NHIỄU)
# ---------------------------------------------------------------------------
X, y = load_diabetes(return_X_y=True, as_frame=True)
BIEN_THAT = list(X.columns)  # 10 tên cột gốc — "đáp án đúng"

rng = np.random.default_rng(RNG_SEED)
nhieu = pd.DataFrame(
    rng.normal(size=(len(X), 190)),
    columns=[f"chi_so_nhieu_{i:03d}" for i in range(190)],
)
X_full = pd.concat([X, nhieu], axis=1)  # 200 cột

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RNG_SEED
)

print(f"Dữ liệu: {X_full.shape[0]} dòng x {X_full.shape[1]} cột "
      f"({len(BIEN_THAT)} biến thật + 190 biến nhiễu)")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. BASELINE: LINEAR REGRESSION TRÊN 200 CỘT → QUAN SÁT OVERFIT
# ---------------------------------------------------------------------------
scaler_base = StandardScaler()
Xtr_s = scaler_base.fit_transform(X_train)
Xte_s = scaler_base.transform(X_test)

lin = LinearRegression()
lin.fit(Xtr_s, y_train)
rmse_lin_train = mean_squared_error(y_train, lin.predict(Xtr_s)) ** 0.5
rmse_lin_test = mean_squared_error(y_test, lin.predict(Xte_s)) ** 0.5

print("\n[BASELINE] Linear Regression trên 200 cột")
print(f"  RMSE train = {rmse_lin_train:.2f}")
print(f"  RMSE test  = {rmse_lin_test:.2f}  (overfit rõ rệt so với train)")

# ---------------------------------------------------------------------------
# 3. LASSOCV DÒ ALPHA
# ---------------------------------------------------------------------------
pipe_lasso = Pipeline([
    ("scale", StandardScaler()),
    ("lasso", LassoCV(alphas=np.logspace(-4, 1, 100), cv=5,
                       max_iter=50000, random_state=RNG_SEED)),
])
pipe_lasso.fit(X_train, y_train)

lasso_model = pipe_lasso["lasso"]
alpha_toi_uu = lasso_model.alpha_
he_so = lasso_model.coef_
so_bien_giu = int((he_so != 0).sum())

rmse_lasso_train = mean_squared_error(
    y_train, pipe_lasso.predict(X_train)) ** 0.5
rmse_lasso_test = mean_squared_error(
    y_test, pipe_lasso.predict(X_test)) ** 0.5

print(f"\n[LASSO] alpha tối ưu (CV) = {alpha_toi_uu:.5f}")
print(f"  Lasso giữ lại {so_bien_giu}/200 biến")
print(f"  RMSE train = {rmse_lasso_train:.2f}")
print(f"  RMSE test  = {rmse_lasso_test:.2f}")

# ---------------------------------------------------------------------------
# 4. CHẤM ĐIỂM CHỌN BIẾN
# ---------------------------------------------------------------------------
ten_cot = X_full.columns
bien_duoc_chon = set(ten_cot[he_so != 0])

that_giu = sorted(bien_duoc_chon & set(BIEN_THAT))
that_bo = sorted(set(BIEN_THAT) - bien_duoc_chon)
nhieu_giu_nham = sorted(bien_duoc_chon - set(BIEN_THAT))

print(f"\n[CHẤM ĐIỂM CHỌN BIẾN]")
print(f"  Biến THẬT được giữ ({len(that_giu)}/10): {that_giu}")
print(f"  Biến THẬT bị bỏ    ({len(that_bo)}/10): {that_bo}")
print(f"  Biến NHIỄU bị giữ nhầm ({len(nhieu_giu_nham)}/190): {nhieu_giu_nham}")

bang_diem = pd.DataFrame({
    "Chỉ số": ["Tổng số biến giữ lại", "Biến thật giữ đúng (recall)",
               "Biến thật bị bỏ sót", "Biến nhiễu giữ nhầm (false positive)"],
    "Giá trị": [f"{so_bien_giu}/200", f"{len(that_giu)}/10",
                f"{len(that_bo)}/10", f"{len(nhieu_giu_nham)}/190"],
})
bang_diem.to_csv(f"{REPORTS_DIR}/bang_cham_diem_chon_bien.csv", index=False)

# ---------------------------------------------------------------------------
# 5. COEFFICIENT PATH CỦA LASSO
# ---------------------------------------------------------------------------
scaler_path = StandardScaler()
X_scaled_full = scaler_path.fit_transform(X_full)

alphas_path, coefs_path, _ = lasso_path(
    X_scaled_full, y, alphas=np.logspace(1, -4, 100), max_iter=50000
)

plt.figure(figsize=(9, 6))
for i, col in enumerate(ten_cot):
    mau = "crimson" if col in BIEN_THAT else "lightgray"
    do_day = 2.2 if col in BIEN_THAT else 0.6
    zorder = 3 if col in BIEN_THAT else 1
    plt.plot(alphas_path, coefs_path[i], color=mau, linewidth=do_day, zorder=zorder)
plt.xscale("log")
plt.gca().invert_xaxis()
plt.axvline(alpha_toi_uu, color="steelblue", linestyle="--",
            label=f"alpha tối ưu (CV) = {alpha_toi_uu:.4f}")
plt.xlabel("alpha (thang log, giảm dần)")
plt.ylabel("Hệ số hồi quy (đã chuẩn hoá)")
plt.title("Lasso Coefficient Path\n(đỏ = 10 biến thật, xám = 190 biến nhiễu)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/lasso_path.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 6. RMSE TRAIN/TEST THEO ALPHA
# ---------------------------------------------------------------------------
alphas_grid = np.logspace(-4, 1, 60)
rmse_train_list, rmse_test_list, n_bien_list = [], [], []

scaler_grid = StandardScaler().fit(X_train)
Xtr_g = scaler_grid.transform(X_train)
Xte_g = scaler_grid.transform(X_test)

for a in alphas_grid:
    m = Lasso(alpha=a, max_iter=50000, random_state=RNG_SEED)
    m.fit(Xtr_g, y_train)
    rmse_train_list.append(mean_squared_error(y_train, m.predict(Xtr_g)) ** 0.5)
    rmse_test_list.append(mean_squared_error(y_test, m.predict(Xte_g)) ** 0.5)
    n_bien_list.append(int((m.coef_ != 0).sum()))

plt.figure(figsize=(9, 6))
plt.plot(alphas_grid, rmse_train_list, label="RMSE train", marker="o", markersize=3)
plt.plot(alphas_grid, rmse_test_list, label="RMSE test", marker="o", markersize=3)
plt.axvline(alpha_toi_uu, color="gray", linestyle="--",
            label=f"alpha tối ưu = {alpha_toi_uu:.4f}")
plt.xscale("log")
plt.xlabel("alpha (thang log)")
plt.ylabel("RMSE")
plt.title("RMSE train/test theo alpha")
plt.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/rmse_vs_alpha.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 7. SO SÁNH RIDGE VS LASSO
# ---------------------------------------------------------------------------
pipe_ridge = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)),
])
pipe_ridge.fit(X_train, y_train)
ridge_model = pipe_ridge["ridge"]
so_bien_giu_ridge = int((ridge_model.coef_ != 0).sum())
rmse_ridge_test = mean_squared_error(y_test, pipe_ridge.predict(X_test)) ** 0.5
rmse_ridge_train = mean_squared_error(y_train, pipe_ridge.predict(X_train)) ** 0.5

bang_ridge_lasso = pd.DataFrame({
    "Mô hình": ["Linear (baseline, 200 biến)", "Ridge (L2)", "Lasso (L1)"],
    "Số biến giữ lại": [200, so_bien_giu_ridge, so_bien_giu],
    "RMSE train": [round(rmse_lin_train, 2), round(rmse_ridge_train, 2),
                    round(rmse_lasso_train, 2)],
    "RMSE test": [round(rmse_lin_test, 2), round(rmse_ridge_test, 2),
                   round(rmse_lasso_test, 2)],
})
bang_ridge_lasso.to_csv(f"{REPORTS_DIR}/so_sanh_ridge_lasso.csv", index=False)
print("\n[SO SÁNH RIDGE VS LASSO]")
print(bang_ridge_lasso.to_string(index=False))

plt.figure(figsize=(7, 5))
x_pos = np.arange(3)
plt.bar(x_pos, bang_ridge_lasso["Số biến giữ lại"],
        color=["gray", "seagreen", "crimson"])
plt.xticks(x_pos, bang_ridge_lasso["Mô hình"], rotation=15)
plt.ylabel("Số biến giữ lại (trên 200)")
plt.title("Ridge vs Lasso: số biến giữ lại")
for i, v in enumerate(bang_ridge_lasso["Số biến giữ lại"]):
    plt.text(i, v + 3, str(v), ha="center")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/ridge_vs_lasso.png", dpi=130)
plt.close()

plt.figure(figsize=(6, 5))
labels = bang_ridge_lasso["Mô hình"]
width = 0.35
xpos = np.arange(len(labels))
plt.bar(xpos - width/2, bang_ridge_lasso["RMSE train"], width, label="RMSE train")
plt.bar(xpos + width/2, bang_ridge_lasso["RMSE test"], width, label="RMSE test")
plt.xticks(xpos, labels, rotation=15)
plt.ylabel("RMSE")
plt.title("RMSE: Linear vs Ridge vs Lasso")
plt.legend()
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/rmse_ridge_vs_lasso.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 8. THÍ NGHIỆM BIẾN TƯƠNG QUAN — nhân đôi 1 cột thật + nhiễu nhỏ
# ---------------------------------------------------------------------------
X_corr = X_full.copy()
bien_goc = "bmi"  # biến thật có tín hiệu mạnh trong diabetes
X_corr[f"{bien_goc}_ban_sao"] = X_corr[bien_goc] + rng.normal(
    scale=0.016, size=len(X_corr))  # tương quan ~0.95 với biến gốc
corr_thuc_te = X_corr[bien_goc].corr(X_corr[f"{bien_goc}_ban_sao"])
print(f"\n[THÍ NGHIỆM TƯƠNG QUAN] Tương quan thực tế giữa '{bien_goc}' và bản sao: "
      f"{corr_thuc_te:.3f}")

ket_qua_seed = {}
n = len(X_corr)
for seed in [1, 2, 3, 42, 99]:
    boot_rng = np.random.default_rng(seed)
    idx = boot_rng.choice(n, size=n, replace=True)  # bootstrap resample
    X_boot, y_boot = X_corr.iloc[idx], y.iloc[idx]
    sc = StandardScaler().fit(X_boot)
    m = LassoCV(alphas=np.logspace(-4, 1, 60), cv=5,
                max_iter=50000, random_state=seed)
    m.fit(sc.transform(X_boot), y_boot)
    coefs = dict(zip(X_corr.columns, m.coef_))
    ket_qua_seed[seed] = {
        bien_goc: round(coefs[bien_goc], 4),
        f"{bien_goc}_ban_sao": round(coefs[f"{bien_goc}_ban_sao"], 4),
    }

bang_tuong_quan = pd.DataFrame(ket_qua_seed).T
bang_tuong_quan.index.name = "random_state (Lasso)"
bang_tuong_quan.to_csv(f"{REPORTS_DIR}/thi_nghiem_bien_tuong_quan.csv")
print(f"\n[THÍ NGHIỆM TƯƠNG QUAN] '{bien_goc}' vs bản sao gần giống, qua các seed:")
print(bang_tuong_quan.to_string())

# ---------------------------------------------------------------------------
# 9. TRAIN LẠI LINEAR REGRESSION CHỈ TRÊN BIẾN LASSO CHỌN (DEBIASED LASSO)
# ---------------------------------------------------------------------------
cot_duoc_chon = list(bien_duoc_chon)
X_train_sel = X_train[cot_duoc_chon]
X_test_sel = X_test[cot_duoc_chon]

sc_sel = StandardScaler().fit(X_train_sel)
lin_debiased = LinearRegression()
lin_debiased.fit(sc_sel.transform(X_train_sel), y_train)

rmse_debiased_train = mean_squared_error(
    y_train, lin_debiased.predict(sc_sel.transform(X_train_sel))) ** 0.5
rmse_debiased_test = mean_squared_error(
    y_test, lin_debiased.predict(sc_sel.transform(X_test_sel))) ** 0.5

print(f"\n[DEBIASED LASSO] Linear Regression chỉ trên {so_bien_giu} biến Lasso chọn")
print(f"  RMSE train = {rmse_debiased_train:.2f}")
print(f"  RMSE test  = {rmse_debiased_test:.2f}")
print(f"  (so với Lasso đầy đủ: RMSE test = {rmse_lasso_test:.2f})")

# ---------------------------------------------------------------------------
# TỔNG HỢP KẾT QUẢ RA FILE JSON (để README dùng lại)
# ---------------------------------------------------------------------------
tong_hop = {
    "alpha_toi_uu": round(float(alpha_toi_uu), 5),
    "so_bien_giu_lasso": so_bien_giu,
    "bien_that_giu": that_giu,
    "bien_that_bo": that_bo,
    "so_luong_nhieu_giu_nham": len(nhieu_giu_nham),
    "nhieu_giu_nham": nhieu_giu_nham,
    "rmse": {
        "linear_baseline_train": round(rmse_lin_train, 2),
        "linear_baseline_test": round(rmse_lin_test, 2),
        "ridge_train": round(rmse_ridge_train, 2),
        "ridge_test": round(rmse_ridge_test, 2),
        "lasso_train": round(rmse_lasso_train, 2),
        "lasso_test": round(rmse_lasso_test, 2),
        "debiased_lasso_train": round(rmse_debiased_train, 2),
        "debiased_lasso_test": round(rmse_debiased_test, 2),
    },
    "so_bien_giu_ridge": so_bien_giu_ridge,
    "thi_nghiem_tuong_quan": ket_qua_seed,
}
with open(f"{REPORTS_DIR}/tong_hop_ket_qua.json", "w", encoding="utf-8") as f:
    json.dump(tong_hop, f, ensure_ascii=False, indent=2)

# Lưu model cuối cùng
import joblib
joblib.dump(pipe_lasso, f"{MODELS_DIR}/lasso_pipeline.joblib")

print("\n=== HOÀN TẤT. Kết quả đã lưu vào reports/ và models/ ===")
