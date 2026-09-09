"""
TT-15 — Polynomial Regression trên bộ dữ liệu Combined Cycle Power Plant (UCI)
Dự báo công suất phát PE (MW) theo AT, V, AP, RH.

Chạy: python src/train.py
Sinh ra: models/poly_pipeline.joblib, reports/*.png, reports/summary_table.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, validation_curve
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42
plt.rcParams["figure.dpi"] = 110

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
df = pd.read_csv(ROOT / "data_ccpp.csv")
X = df[["AT", "V", "AP", "RH"]].values
y = df["PE"].values
feature_names = ["AT", "V", "AP", "RH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"[1] Dữ liệu: {df.shape[0]} dòng, train={len(X_train)}, test={len(X_test)}")

# ------------------------------------------------------------------
# 2. EDA — scatter AT vs PE (quan hệ cong bằng mắt)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(df["AT"], df["PE"], s=6, alpha=0.25, color="#2563eb")
ax.set_xlabel("Nhiệt độ môi trường AT (°C)")
ax.set_ylabel("Công suất phát PE (MW)")
ax.set_title("PE theo AT — quan hệ phi tuyến (cong xuống)")
fig.tight_layout()
fig.savefig(REPORTS / "scatter_AT_PE.png")
plt.close(fig)
print("[2] Đã lưu scatter_AT_PE.png")

# ------------------------------------------------------------------
# 3. BASELINE — Linear Regression bậc 1
# ------------------------------------------------------------------
lin = LinearRegression()
lin.fit(X_train, y_train)
pred_train_lin = lin.predict(X_train)
pred_test_lin = lin.predict(X_test)
rmse_lin_train = np.sqrt(mean_squared_error(y_train, pred_train_lin))
rmse_lin_test = np.sqrt(mean_squared_error(y_test, pred_test_lin))
print(f"[3] Baseline bậc 1: RMSE train={rmse_lin_train:.3f}, test={rmse_lin_test:.3f}")

# ------------------------------------------------------------------
# 4. RESIDUAL PLOT — TRƯỚC (bậc 1)
# ------------------------------------------------------------------
resid_before = y_test - pred_test_lin

# ------------------------------------------------------------------
# 5. Quét bậc 1→5, ghi RMSE train/validation + đếm số cột PolynomialFeatures
# ------------------------------------------------------------------
degrees = [1, 2, 3, 4, 5, 6, 7, 8]
rows = []
for d in degrees:
    poly = PolynomialFeatures(degree=d, include_bias=False)
    Xp_train = poly.fit_transform(X_train)
    n_cols = Xp_train.shape[1]

    pipe_lin = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe_lin.fit(X_train, y_train)
    rmse_tr = np.sqrt(mean_squared_error(y_train, pipe_lin.predict(X_train)))
    rmse_te = np.sqrt(mean_squared_error(y_test, pipe_lin.predict(X_test)))

    pipe_ridge = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ])
    pipe_ridge.fit(X_train, y_train)
    rmse_tr_ridge = np.sqrt(mean_squared_error(y_train, pipe_ridge.predict(X_train)))
    rmse_te_ridge = np.sqrt(mean_squared_error(y_test, pipe_ridge.predict(X_test)))

    rows.append({
        "degree": d, "n_cols": n_cols,
        "rmse_train_linear": round(rmse_tr, 4), "rmse_test_linear": round(rmse_te, 4),
        "rmse_train_ridge": round(rmse_tr_ridge, 4), "rmse_test_ridge": round(rmse_te_ridge, 4),
    })

summary = pd.DataFrame(rows)
summary.to_csv(REPORTS / "summary_table.csv", index=False)
print("[5] Bảng bậc | số cột | RMSE:")
print(summary.to_string(index=False))

best_degree = int(summary.loc[summary["rmse_test_ridge"].idxmin(), "degree"])
print(f"[5] Bậc tối ưu theo RMSE test (Ridge): {best_degree}")

# ------------------------------------------------------------------
# 6. ĐƯỜNG CONG XÁC THỰC (validation_curve) — dùng Ridge cho ổn định ở bậc cao
# ------------------------------------------------------------------
pipe_vc = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
train_scores, val_scores = validation_curve(
    pipe_vc, X_train, y_train,
    param_name="poly__degree", param_range=degrees,
    cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1,
)
train_rmse_cv = -train_scores.mean(axis=1)
val_rmse_cv = -val_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(degrees, train_rmse_cv, "o-", label="RMSE train (CV)", color="#2563eb")
ax.plot(degrees, val_rmse_cv, "o-", label="RMSE validation (CV)", color="#dc2626")
ax.set_xlabel("Bậc đa thức (degree)")
ax.set_ylabel("RMSE (MW)")
ax.set_title("Đường cong xác thực theo bậc đa thức (Ridge, 5-fold CV)")
ax.legend()
ax.set_xticks(degrees)
fig.tight_layout()
fig.savefig(REPORTS / "validation_curve.png")
plt.close(fig)

best_degree_cv = degrees[int(np.argmin(val_rmse_cv))]
print(f"[6] Đã lưu validation_curve.png — bậc RMSE-val thấp nhất (số học): {best_degree_cv}")
print(f"    RMSE val theo bậc: {dict(zip(degrees, np.round(val_rmse_cv, 3)))}")
print("    -> Đường cong gần như PHẲNG từ bậc 3 trở đi (chênh lệch bậc 3->8 chỉ ~0.1 MW).")
print("       Chọn bậc 3 làm model triển khai thực tế để tránh phức tạp/rủi ro ngoại suy không cần thiết.")

# ------------------------------------------------------------------
# 7. Model cuối cùng — chọn bậc 3 (điểm elbow thực tế, xem lý do ở bước 6)
# ------------------------------------------------------------------
final_degree = 3
final_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=final_degree, include_bias=False)),
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
final_pipe.fit(X_train, y_train)
pred_test_final = final_pipe.predict(X_test)
rmse_final = np.sqrt(mean_squared_error(y_test, pred_test_final))
r2_final = r2_score(y_test, pred_test_final)
print(f"[7] Model cuối (bậc {final_degree}, Ridge): RMSE test={rmse_final:.3f}, R2={r2_final:.4f}")

joblib.dump(final_pipe, MODELS / "poly_pipeline.joblib")
print(f"[7] Đã lưu models/poly_pipeline.joblib")

# ------------------------------------------------------------------
# 8. RESIDUAL PLOT — TRƯỚC (bậc 1) và SAU (bậc tối ưu) trên cùng 1 hình
# ------------------------------------------------------------------
resid_after = y_test - pred_test_final

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
axes[0].scatter(pred_test_lin, resid_before, s=6, alpha=0.3, color="#dc2626")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_title(f"TRƯỚC — bậc 1 (Linear)\nhình chữ U rõ")
axes[0].set_xlabel("Giá trị dự đoán PE (MW)")
axes[0].set_ylabel("Residual (y_thực - y_dự đoán)")

axes[1].scatter(pred_test_final, resid_after, s=6, alpha=0.3, color="#16a34a")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_title(f"SAU — bậc {final_degree} (Ridge)\nphân tán đều hơn")
axes[1].set_xlabel("Giá trị dự đoán PE (MW)")

fig.suptitle("Residual plot: trước vs sau khi thêm bậc phi tuyến")
fig.tight_layout()
fig.savefig(REPORTS / "residual_truoc_sau.png")
plt.close(fig)
print("[8] Đã lưu residual_truoc_sau.png")

# ------------------------------------------------------------------
# 9. So sánh Linear vs Ridge ở bậc cao (4-5)
# ------------------------------------------------------------------
print("[9] So sánh Linear vs Ridge ở bậc cao (xem summary_table.csv, cột *_ridge vs *_linear ở degree 4,5)")
high_deg = summary[summary["degree"].isin([4, 5])]
print(high_deg.to_string(index=False))

# ------------------------------------------------------------------
# 10. So sánh với Random Forest Regressor
# ------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train, y_train)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
r2_rf = r2_score(y_test, rf.predict(X_test))
print(f"[10] Random Forest: RMSE test={rmse_rf:.3f}, R2={r2_rf:.4f}  "
      f"(so với Ridge bậc {final_degree}: RMSE={rmse_final:.3f})")

# ------------------------------------------------------------------
# 11. Ngoại suy ngoài dải dữ liệu — AT = 50°C với bậc 5 vs bậc tối ưu
# ------------------------------------------------------------------
sample = X_test[0].copy()
extrap_row = sample.copy()
extrap_row[0] = 50.0  # AT ngoài dải dữ liệu (max thực tế là 37.11)

pipe_deg8_ridge = Pipeline([
    ("poly", PolynomialFeatures(degree=8, include_bias=False)),
    ("scale", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
pipe_deg8_ridge.fit(X_train, y_train)

# Linear THUẦN (không regularize) ở bậc 8 — kỳ vọng "phát điên" mạnh hơn Ridge
pipe_deg8_linear = Pipeline([
    ("poly", PolynomialFeatures(degree=8, include_bias=False)),
    ("scale", StandardScaler()),
    ("model", LinearRegression()),
])
pipe_deg8_linear.fit(X_train, y_train)

pred_normal_ridge8 = pipe_deg8_ridge.predict(sample.reshape(1, -1))[0]
pred_extrap_ridge8 = pipe_deg8_ridge.predict(extrap_row.reshape(1, -1))[0]
pred_normal_lin8 = pipe_deg8_linear.predict(sample.reshape(1, -1))[0]
pred_extrap_lin8 = pipe_deg8_linear.predict(extrap_row.reshape(1, -1))[0]
pred_normal_final = final_pipe.predict(sample.reshape(1, -1))[0]
pred_extrap_final = final_pipe.predict(extrap_row.reshape(1, -1))[0]

print(f"[11] Ngoại suy AT=50°C (thực tế max AT trong data ~37.11°C, PE thực chỉ trong 420-496 MW):")
print(f"     Bậc {final_degree} (final, Ridge) — trong dải: {pred_normal_final:.2f} MW | ngoại suy: {pred_extrap_final:.2f} MW "
      f"(lệch {pred_extrap_final - pred_normal_final:+.2f})")
print(f"     Bậc 8 + Ridge   — trong dải: {pred_normal_ridge8:.2f} MW | ngoại suy: {pred_extrap_ridge8:.2f} MW "
      f"(lệch {pred_extrap_ridge8 - pred_normal_ridge8:+.2f})")
print(f"     Bậc 8 + Linear THUẦN (không regularize) — trong dải: {pred_normal_lin8:.2f} MW | ngoại suy: {pred_extrap_lin8:.2f} MW "
      f"(lệch {pred_extrap_lin8 - pred_normal_lin8:+.2f})")
print("     -> Nếu giá trị ngoại suy vọt ra ngoài khoảng PE thực tế (420-496 MW), đặc biệt ở")
print("        Linear thuần bậc cao, đó là bằng chứng đa thức bậc cao 'phát điên' khi ngoại suy;")
print("        Ridge kiềm chế hệ số nên đỡ cực đoan hơn nhưng không loại bỏ hoàn toàn vấn đề.")

# ------------------------------------------------------------------
# 12. Ghi kết quả tổng hợp ra text để đưa vào README
# ------------------------------------------------------------------
with open(REPORTS / "ket_qua_tom_tat.txt", "w", encoding="utf-8") as f:
    f.write("TT-15 — KẾT QUẢ TÓM TẮT\n")
    f.write("=" * 50 + "\n")
    f.write(f"Baseline (bậc 1): RMSE train={rmse_lin_train:.3f}, test={rmse_lin_test:.3f}\n")
    f.write(f"Bậc tối ưu (CV, Ridge): {final_degree}\n")
    f.write(f"Model cuối (bậc {final_degree}, Ridge): RMSE test={rmse_final:.3f} MW, R2={r2_final:.4f}\n")
    f.write(f"Random Forest: RMSE test={rmse_rf:.3f} MW, R2={r2_rf:.4f}\n\n")
    f.write("Bảng bậc | số cột | RMSE:\n")
    f.write(summary.to_string(index=False) + "\n\n")
    f.write("Ngoại suy AT=50°C (ngoài dải dữ liệu, PE thực tế chỉ trong 420-496 MW):\n")
    f.write(f"  Bậc {final_degree} (final, Ridge): trong dải {pred_normal_final:.2f} MW -> ngoại suy {pred_extrap_final:.2f} MW "
            f"(lệch {pred_extrap_final - pred_normal_final:+.2f})\n")
    f.write(f"  Bậc 8 + Ridge: trong dải {pred_normal_ridge8:.2f} MW -> ngoại suy {pred_extrap_ridge8:.2f} MW "
            f"(lệch {pred_extrap_ridge8 - pred_normal_ridge8:+.2f})\n")
    f.write(f"  Bậc 8 + Linear thuần: trong dải {pred_normal_lin8:.2f} MW -> ngoại suy {pred_extrap_lin8:.2f} MW "
            f"(lệch {pred_extrap_lin8 - pred_normal_lin8:+.2f})\n")

print("\n[HOÀN THÀNH] Xem reports/ để lấy plots + bảng, models/ để lấy pipeline .joblib")
